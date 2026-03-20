import Eris from 'eris';
import { EventEmitter } from 'events';
import dotenv from 'dotenv';
import { logger } from './logger';
import { AudioSink } from './audiosink';
import { TranscriptionWorker } from './transcription';
import { startWebUI, stopWebUI } from './webui';
import {
  initDB, shutdownDB, createSession, listSessions,
  getIgnoredUsers, addIgnoredUser, removeIgnoredUser, IgnoredUserRow,
  getCharacterNamesDB, setCharacterNameDB,
  getNicknamePresets, addNicknamePreset, updateNicknamePreset, deleteNicknamePreset, NicknamePreset,
} from './db';

dotenv.config();

const TOKEN = process.env.TOKEN;
const GUILD_ID = process.env.GUILD_ID;
const WHISPER_MODEL = process.env.WHISPER_MODEL ?? 'base';
const WEB_PORT = parseInt(process.env.WEB_PORT ?? '3001', 10);

if (!TOKEN) throw new Error('TOKEN env var required');
if (!GUILD_ID) throw new Error('GUILD_ID env var required');

// State
const ignoredUsers = new Set<string>();
const transcriptionWorker = new TranscriptionWorker(WHISPER_MODEL);
const characterNames = new Map<string, string>();
// Persists save-recordings preference across sessions so it survives leave/rejoin
let saveRecordingsPreference = false;

// Per-guild voice state
interface GuildVoiceState {
  connection: Eris.VoiceConnection;
  receiver: Eris.VoiceDataStream;
  sink: AudioSink;
  channelID: string;
}
const voiceStates = new Map<string, GuildVoiceState>();

// Load ignored users from DB into memory set
async function loadIgnoredUsersFromDB(): Promise<void> {
  const rows = await getIgnoredUsers();
  ignoredUsers.clear();
  for (const row of rows) {
    ignoredUsers.add(row.discordUserId);
  }
  logger.info(`Loaded ${ignoredUsers.size} ignored users from DB`);
}

async function loadCharacterNamesFromDB(): Promise<void> {
  const map = await getCharacterNamesDB();
  characterNames.clear();
  for (const [k, v] of map) characterNames.set(k, v);
  logger.info(`Loaded ${characterNames.size} character names from DB`);
}

// Bot setup
const bot = new Eris.Client(TOKEN, {
  gateway: {
    intents: ['guilds', 'guildVoiceStates', 'guildMembers'],
  },
});

// Slash command definitions
const COMMANDS: Eris.ApplicationCommandStructure[] = [
  {
    name: 'transcribe',
    description: 'Start recording and transcribing the voice channel',
    type: 1, // CHAT_INPUT
  },
  {
    name: 'leave',
    description: 'Leave the voice channel and stop recording',
    type: 1,
  },
  {
    name: 'ignore',
    description: "Ignore a user's audio",
    type: 1,
    options: [
      {
        name: 'user',
        description: 'The user to ignore',
        type: 6, // USER
        required: true,
      },
    ],
  },
  {
    name: 'unignore',
    description: "Stop ignoring a user's audio",
    type: 1,
    options: [
      {
        name: 'user',
        description: 'The user to stop ignoring',
        type: 6, // USER
        required: true,
      },
    ],
  },
  {
    name: 'list_ignored',
    description: 'List all ignored users',
    type: 1,
  },
  {
    name: 'sync',
    description: 'Re-sync slash commands',
    type: 1,
  },
  {
    name: 'save_recordings',
    description: 'Toggle saving raw audio recordings to disk',
    type: 1,
  },
];

async function syncCommands(): Promise<number> {
  const synced = await bot.bulkEditGuildCommands(GUILD_ID!, COMMANDS);
  logger.info(`Synced ${synced.length} slash commands`);
  return synced.length;
}

// ---- Shared join/leave logic (used by both slash commands and web UI) ----

async function joinVoiceChannel(channelID: string): Promise<string> {
  const guild = bot.guilds.get(GUILD_ID!);
  if (!guild) throw new Error('Guild not found');

  if (voiceStates.has(GUILD_ID!)) throw new Error('Already recording');

  const channel = guild.channels.get(channelID);
  if (!channel || !(channel instanceof Eris.VoiceChannel || channel instanceof Eris.StageChannel)) {
    throw new Error('Voice channel not found');
  }

  // Create a DB session for this recording
  const sessionId = await createSession(channel.name, channelID);
  transcriptionWorker.setSessionId(sessionId);

  const connection = await channel.join({ opusOnly: true });
  const receiver = connection.receive('opus');
  const sink = new AudioSink(transcriptionWorker, ignoredUsers);
  sink.setSessionId(sessionId);

  // Apply saved save-recordings preference
  if (saveRecordingsPreference) {
    sink.setSaveRecordings(true);
  }

  // Populate user names and character names for current voice members
  for (const [, voiceMember] of channel.voiceMembers) {
    sink.setUserName(voiceMember.id, voiceMember.username);
    const charName = characterNames.get(voiceMember.id);
    if (charName) sink.setCharacterName(voiceMember.id, charName);
  }

  receiver.on('data', (data: Buffer, userID: string, timestamp: number) => {
    const user = bot.users.get(userID);
    if (user) sink.setUserName(userID, user.username);
    // Apply character name on first packet too
    const charName = characterNames.get(userID);
    if (charName) sink.setCharacterName(userID, charName);
    sink.onData(data, userID, timestamp);
  });

  voiceStates.set(GUILD_ID!, { connection, receiver, sink, channelID });

  connection.on('disconnect', (err?: Error) => {
    if (err) logger.error(`Voice disconnected in ${GUILD_ID}: ${err.message}`);
    else logger.info(`Voice disconnected in ${GUILD_ID}`);
    cleanupGuild(GUILD_ID!);
    botController.emit('status');
  });

  connection.on('error', (err: Error) => {
    logger.error(`Voice connection error in ${GUILD_ID}: ${err.message}`);
  });

  logger.info(`Started recording in ${channel.name} (${GUILD_ID}), session ${sessionId}`);
  botController.emit('status');
  botController.emit('members');
  botController.emit('session_started', { id: sessionId, channelName: channel.name });
  return channel.name;
}

async function leaveVoiceChannel(): Promise<string | null> {
  const state = voiceStates.get(GUILD_ID!);
  if (!state) throw new Error('Not in a voice channel');

  state.receiver.removeAllListeners('data');
  state.sink.stop();
  bot.leaveVoiceChannel(state.connection.channelID!);
  voiceStates.delete(GUILD_ID!);

  botController.emit('status');
  botController.emit('members');

  const outFile = await transcriptionWorker.finalize();
  logger.info(`Stopped recording in ${GUILD_ID}, transcription: ${outFile ?? 'none'}`);
  return outFile;
}

function cleanupGuild(guildID: string): void {
  const state = voiceStates.get(guildID);
  if (!state) return;

  state.receiver.removeAllListeners('data');
  state.sink.stop();
  voiceStates.delete(guildID);
}

// ---- Bot controller (exposes bot operations to the web UI) ----

export interface VoiceChannelInfo {
  id: string;
  name: string;
  members: string[]; // usernames of people in the channel
}

export interface VoiceMemberInfo {
  id: string;
  username: string;
  characterName: string | null;
}

export interface BotController extends EventEmitter {
  getVoiceChannels(): VoiceChannelInfo[];
  getVoiceMembers(): VoiceMemberInfo[];
  joinChannel(channelId: string): Promise<string>;
  leaveChannel(): Promise<string | null>;
  setCharacterName(userId: string, name: string | null): void;
  isRecording(): boolean;
  getStatus(): { recording: boolean; channelName: string | null };
  getSaveRecordings(): boolean;
  setSaveRecordings(enabled: boolean): void;
  togglePause(): boolean;
  getSessions(): Promise<any[]>;
  getIgnoredUsers(): Promise<IgnoredUserRow[]>;
  ignoreUser(userId: string, username: string): Promise<boolean>;
  unignoreUser(userId: string): Promise<boolean>;
  getNicknamePresets(userId?: string): Promise<NicknamePreset[]>;
}

class BotControllerImpl extends EventEmitter implements BotController {
  getVoiceChannels(): VoiceChannelInfo[] {
    const guild = bot.guilds.get(GUILD_ID!);
    if (!guild) return [];

    const channels: VoiceChannelInfo[] = [];
    for (const [, ch] of guild.channels) {
      if (ch instanceof Eris.VoiceChannel || ch instanceof Eris.StageChannel) {
        const members: string[] = [];
        for (const [, m] of ch.voiceMembers) {
          if (m.id !== bot.user.id) members.push(m.username);
        }
        channels.push({
          id: ch.id,
          name: ch.name,
          members,
        });
      }
    }
    // Sort by position in Discord
    channels.sort((a, b) => {
      const chA = guild.channels.get(a.id);
      const chB = guild.channels.get(b.id);
      return (chA && 'position' in chA ? chA.position : 0) - (chB && 'position' in chB ? chB.position : 0);
    });
    return channels;
  }

  getVoiceMembers(): VoiceMemberInfo[] {
    const state = voiceStates.get(GUILD_ID!);
    if (!state) return [];

    const guild = bot.guilds.get(GUILD_ID!);
    if (!guild) return [];

    const channel = guild.channels.get(state.channelID);
    if (!channel || !(channel instanceof Eris.VoiceChannel || channel instanceof Eris.StageChannel)) return [];

    const members: VoiceMemberInfo[] = [];
    for (const [, member] of channel.voiceMembers) {
      if (member.id === bot.user.id) continue; // Don't show the bot itself
      members.push({
        id: member.id,
        username: member.username,
        characterName: characterNames.get(member.id) ?? null,
      });
    }
    return members;
  }

  async joinChannel(channelId: string): Promise<string> {
    return joinVoiceChannel(channelId);
  }

  async leaveChannel(): Promise<string | null> {
    return leaveVoiceChannel();
  }

  setCharacterName(userId: string, name: string | null): void {
    const state = voiceStates.get(GUILD_ID!);

    // Flush the user's audio buffer BEFORE changing the name so any
    // speech in progress gets transcribed under the old name
    if (state) {
      state.sink.flushUserBuffer(userId);
    }

    if (name) {
      characterNames.set(userId, name);
    } else {
      characterNames.delete(userId);
    }
    setCharacterNameDB(userId, name).catch(e => logger.error(`Failed to save character name: ${e}`));

    if (state) {
      if (name) {
        state.sink.setCharacterName(userId, name);
      } else {
        state.sink.removeCharacterName(userId);
      }
    }

    this.emit('members');
  }

  isRecording(): boolean {
    return voiceStates.has(GUILD_ID!);
  }

  getStatus(): { recording: boolean; channelName: string | null } {
    const state = voiceStates.get(GUILD_ID!);
    if (!state) return { recording: false, channelName: null };
    const guild = bot.guilds.get(GUILD_ID!);
    const channel = guild?.channels.get(state.channelID);
    return { recording: true, channelName: channel?.name ?? null };
  }

  getSaveRecordings(): boolean {
    const state = voiceStates.get(GUILD_ID!);
    if (state) return state.sink.getSaveRecordings();
    return saveRecordingsPreference;
  }

  setSaveRecordings(enabled: boolean): void {
    saveRecordingsPreference = enabled;
    const state = voiceStates.get(GUILD_ID!);
    if (state) state.sink.setSaveRecordings(enabled);
    this.emit('save_recordings', enabled);
  }

  togglePause(): boolean {
    const state = voiceStates.get(GUILD_ID!);
    if (!state) return false;
    const newState = !state.sink.isPaused();
    state.sink.setPaused(newState);
    return newState;
  }

  async getSessions(): Promise<any[]> {
    return listSessions();
  }

  async getIgnoredUsers(): Promise<IgnoredUserRow[]> {
    return getIgnoredUsers();
  }

  async ignoreUser(userId: string, username: string): Promise<boolean> {
    const added = await addIgnoredUser(userId, username);
    if (added) {
      ignoredUsers.add(userId);
      for (const state of voiceStates.values()) {
        state.sink.updateIgnoredUsers(ignoredUsers);
      }
      this.emit('ignored_users');
      logger.info(`Ignoring user ${username} (${userId})`);
    }
    return added;
  }

  async getNicknamePresets(userId?: string): Promise<NicknamePreset[]> {
    return getNicknamePresets(userId);
  }

  async unignoreUser(userId: string): Promise<boolean> {
    const removed = await removeIgnoredUser(userId);
    if (removed) {
      ignoredUsers.delete(userId);
      for (const state of voiceStates.values()) {
        state.sink.updateIgnoredUsers(ignoredUsers);
      }
      this.emit('ignored_users');
      logger.info(`Unignored user ${userId}`);
    }
    return removed;
  }
}

const botController = new BotControllerImpl();

// ---- Command handlers (use shared join/leave logic) ----

async function handleTranscribe(interaction: Eris.CommandInteraction): Promise<void> {
  const guildID = interaction.guildID;
  if (!guildID) {
    await interaction.createMessage({ content: 'This command must be used in a server', flags: 64 });
    return;
  }

  const guild = bot.guilds.get(guildID);
  const member = interaction.member;
  if (!guild || !member) {
    await interaction.createMessage({ content: 'Could not find guild or member', flags: 64 });
    return;
  }

  const memberObj = guild.members.get(member.id);
  const voiceState = memberObj?.voiceState;

  if (!voiceState?.channelID) {
    await interaction.createMessage({ content: 'You are not in a voice channel', flags: 64 });
    return;
  }

  if (voiceStates.has(guildID)) {
    await interaction.createMessage({ content: 'Already recording in this server', flags: 64 });
    return;
  }

  await interaction.defer(64);

  try {
    const channelName = await joinVoiceChannel(voiceState.channelID);
    await interaction.editOriginalMessage({ content: `Recording in ${channelName}...` });
  } catch (e) {
    logger.error(`Failed to connect to voice: ${e}`);
    await interaction.editOriginalMessage({ content: `Failed to connect: ${e}` });
  }
}

async function handleLeave(interaction: Eris.CommandInteraction): Promise<void> {
  const guildID = interaction.guildID;
  if (!guildID || !voiceStates.has(guildID)) {
    await interaction.createMessage({ content: "I'm not in a voice channel", flags: 64 });
    return;
  }

  await interaction.defer(64);

  try {
    await interaction.editOriginalMessage({ content: 'Left voice channel. Finalizing transcription...' });
    const outFile = await leaveVoiceChannel();
    if (outFile) {
      await interaction.editOriginalMessage({ content: `Recording stopped. Transcription saved to \`${outFile}\`` });
    } else {
      await interaction.editOriginalMessage({ content: 'Recording stopped. No transcription to save.' });
    }
  } catch (e) {
    logger.error(`Error leaving voice: ${e}`);
    await interaction.editOriginalMessage({ content: `Error: ${e}` });
  }
}

async function handleIgnore(interaction: Eris.CommandInteraction): Promise<void> {
  const userOpt = interaction.data.options?.[0];
  if (!userOpt || !('value' in userOpt)) {
    await interaction.createMessage({ content: 'No user specified', flags: 64 });
    return;
  }

  const targetId = String(userOpt.value);
  const targetUser = interaction.data.resolved?.users?.get(targetId);
  const displayName = targetUser?.username ?? targetId;

  if (ignoredUsers.has(targetId)) {
    await interaction.createMessage({ content: `Already ignoring ${displayName}`, flags: 64 });
    return;
  }

  await botController.ignoreUser(targetId, displayName);
  await interaction.createMessage({ content: `Now ignoring ${displayName}`, flags: 64 });
}

async function handleUnignore(interaction: Eris.CommandInteraction): Promise<void> {
  const userOpt = interaction.data.options?.[0];
  if (!userOpt || !('value' in userOpt)) {
    await interaction.createMessage({ content: 'No user specified', flags: 64 });
    return;
  }

  const targetId = String(userOpt.value);
  const targetUser = interaction.data.resolved?.users?.get(targetId);
  const displayName = targetUser?.username ?? targetId;

  if (!ignoredUsers.has(targetId)) {
    await interaction.createMessage({ content: `Wasn't ignoring ${displayName}`, flags: 64 });
    return;
  }

  await botController.unignoreUser(targetId);
  await interaction.createMessage({ content: `No longer ignoring ${displayName}`, flags: 64 });
}

async function handleListIgnored(interaction: Eris.CommandInteraction): Promise<void> {
  if (ignoredUsers.size === 0) {
    await interaction.createMessage({ content: 'No users are being ignored', flags: 64 });
    return;
  }

  const lines: string[] = [];
  for (const userId of ignoredUsers) {
    const user = bot.users.get(userId);
    const name = user ? user.username : 'Unknown User';
    lines.push(`- ${name} (${userId})`);
  }

  await interaction.createMessage({
    content: `Currently ignored users:\n${lines.join('\n')}`,
    flags: 64,
  });
}

async function handleSync(interaction: Eris.CommandInteraction): Promise<void> {
  const count = await syncCommands();
  await interaction.createMessage({ content: `Synced ${count} commands`, flags: 64 });
}

async function handleSaveRecordings(interaction: Eris.CommandInteraction): Promise<void> {
  const guildID = interaction.guildID;
  if (!guildID || !voiceStates.has(guildID)) {
    await interaction.createMessage({ content: 'Not currently recording in this server', flags: 64 });
    return;
  }

  const state = voiceStates.get(guildID)!;
  const current = state.sink.getSaveRecordings();
  state.sink.setSaveRecordings(!current);

  const status = !current ? 'enabled' : 'disabled';
  await interaction.createMessage({
    content: `Recording save ${status}. Files will be saved to \`recordings/raw/\``,
    flags: 64,
  });
}

// Event handlers
let firstReady = true;
bot.on('ready', async () => {
  logger.info(`${bot.user.username} connected to Discord`);
  await syncCommands();
  if (firstReady) {
    firstReady = false;
    await initDB();
    const { migrateHotwordsToDB } = await import('./hotwords');
    await migrateHotwordsToDB();
    await loadIgnoredUsersFromDB();
    await loadCharacterNamesFromDB();
    await transcriptionWorker.init();
    startWebUI(transcriptionWorker, botController, WEB_PORT);
  }
});

// Track voice state changes to update member list in web UI
bot.on('voiceStateUpdate', (_member, _oldState) => {
  if (voiceStates.has(GUILD_ID!)) {
    botController.emit('members');
  }
});

bot.on('interactionCreate', async (interaction) => {
  if (!(interaction instanceof Eris.CommandInteraction)) return;

  try {
    switch (interaction.data.name) {
      case 'transcribe': await handleTranscribe(interaction); break;
      case 'leave': await handleLeave(interaction); break;
      case 'ignore': await handleIgnore(interaction); break;
      case 'unignore': await handleUnignore(interaction); break;
      case 'list_ignored': await handleListIgnored(interaction); break;
      case 'sync': await handleSync(interaction); break;
      case 'save_recordings': await handleSaveRecordings(interaction); break;
      default:
        await interaction.createMessage({ content: 'Unknown command', flags: 64 });
    }
  } catch (e) {
    logger.error(`Error handling command ${interaction.data.name}: ${e}`);
    try {
      await interaction.createMessage({ content: `Error: ${e}`, flags: 64 });
    } catch {
      // Interaction may have already been responded to
    }
  }
});

bot.on('error', (err) => {
  logger.error(`Bot error: ${err.message}`);
});

// Shutdown
function shutdown(): void {
  logger.info('Shutting down...');

  for (const [guildID] of voiceStates) {
    cleanupGuild(guildID);
  }
  transcriptionWorker.stop();
  stopWebUI();
  shutdownDB().catch(() => {});
  bot.disconnect({ reconnect: false });

  // Give time for cleanup
  setTimeout(() => process.exit(0), 2000);
}

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

// Start
bot.connect();
logger.info('Bot starting...');
