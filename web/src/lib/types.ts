export interface Campaign {
  id: number;
  name: string;
  description: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface Session {
  id: number;
  startedAt: string;
  endedAt: string | null;
  channelName: string | null;
  campaignId: number;
}

export interface TranscriptionLine {
  id: number;
  timestamp: string;
  discordUsername: string;
  nickname: string | null;
  text: string;
  audioFilenames: string[];
  rms?: number | null;
  confidence?: number | null;
  durationMs?: number | null;
  // Diarization (shared-mic): auto label ("S1") and assigned display name.
  speakerId?: number | null;
  speakerLabel?: string | null;
  speakerName?: string | null;
}

// Partial is a provisional live hypothesis from the streaming engine. It is
// ephemeral (never persisted) and is superseded by a committed TranscriptionLine
// when the utterance finalizes.
export interface Partial {
  userId: string;
  discordUsername: string;
  nickname: string | null;
  text: string;
  // Diarization label (display name once assigned); "" until identified.
  speakerLabel?: string;
}

export interface VoiceChannel {
  id: string;
  name: string;
  members: string[];
}

export interface VoiceMember {
  id: string;
  username: string;
  characterName: string | null;
}

export interface BotStatus {
  recording: boolean;
  /** Recording, but transcription is paused. */
  paused?: boolean;
  /** In a voice channel for the DJ only, not recording. */
  musicOnly?: boolean;
  channelName: string | null;
  channelId?: string | null;
  campaignId: number | null;
}

export interface IgnoredUser {
  discordUserId: string;
  discordUsername: string;
  ignoredAt: string;
}

export interface NicknamePreset {
  id: number;
  discordUserId: string;
  label: string;
  position: number;
  categoryId: number | null;
  campaignId: number;
}

export interface NicknameCategory {
  id: number;
  discordUserId: string;
  name: string;
  position: number;
  campaignId: number;
}

export interface SessionPreview {
  first: TranscriptionLine[];
  last: TranscriptionLine[];
}

export interface SearchResult extends TranscriptionLine {
  sessionId: number;
}

// ---- DJ ----

export interface DjTrack {
  id: number;
  path: string;
  title: string;
  artist: string;
  durationMs: number;
  missing: boolean;
}

export interface DjScene {
  id: number;
  name: string;
  /** Music-folder-relative folder paths; each includes its sub-folders. */
  folders: string[];
  color: string;
  volume: number;
  shuffle: boolean;
  loop: boolean;
  fadeSeconds: number | null;
  position: number;
}

export interface SfxTab {
  id: number;
  title: string;
  position: number;
}

export interface SfxPad {
  id: number;
  tabId: number;
  label: string;
  originalName: string;
  durationMs: number;
  color: string;
  /** Image key, served at /api/dj/sfx/images/{key}; null = no image. */
  image: string | null;
  /** 0..1 */
  gain: number;
  position: number;
}

export interface DjSfxPlaying {
  padId: number;
  /** Changes on every retrigger of the pad. */
  instance: number;
  label: string;
  positionMs: number;
  durationMs: number;
}

export interface DjTrackInfo {
  id: number;
  title: string;
  artist: string;
  durationMs: number;
  /** Music-dir-relative path (absent for non-library items). */
  path?: string;
}

export interface DjDeck {
  track: DjTrackInfo | null;
  playing: boolean;
  /** The track played to its end (Play starts it again). */
  ended: boolean;
  positionMs: number;
  gain: number;
  loading: boolean;
  /** Loaded by Auto DJ or a scene rather than by hand. */
  auto: boolean;
  error?: string;
}

/** An automated crossfader move; clients animate it locally. */
export interface DjTransition {
  kind: 'scene' | 'auto' | 'next' | 'track';
  toDeck: 0 | 1;
  from: number;
  to: number;
  startedAt: number;
  durationMs: number;
}

export interface DjQueueItem extends DjTrackInfo {
  qid: number;
}

export interface DjHistoryItem extends DjTrackInfo {
  playedAt: number;
}

/** Full DJ snapshot from the server (dj_state). See docs/dj.md. */
export interface DjState {
  rev: number;
  /** Server clock (unix ms) when positions were sampled. */
  at: number;
  connected: boolean;
  decks: [DjDeck, DjDeck];
  /** Deck the audience mainly hears, -1 = none. */
  onAir: number;
  crossfader: number;
  transition: DjTransition | null;
  autoDJ: boolean;
  /** Master volume, 0..1. */
  volume: number;
  fadeSeconds: number;
  sceneId: number | null;
  folder: string | null;
  /** What filled the queue (scene or folder name). */
  sourceName: string;
  queue: DjQueueItem[];
  queueTotal: number;
  loop: boolean;
  shuffle: boolean;
  history: DjHistoryItem[];
  error?: string;
  /** Sound effects playing now (soundboard). */
  sfx: DjSfxPlaying[];
  /** Soundboard bus gain, 0..1. */
  sfxGain: number;

  // Single-transport view for older UI.
  playing: boolean;
  paused: boolean;
  programName: string;
  track: DjTrackInfo | null;
  positionMs: number;
  upNext: DjTrackInfo[];

  /** Client clock when this state arrived; used to extrapolate positions. */
  receivedAt?: number;
}

export interface DjMeters {
  decks: [number, number];
  master: number;
}

export interface PresenceClient {
  device: string;
  page: string;
}

export interface DjSettings {
  fadeSeconds: number;
}

export interface DjWaveform {
  binsPerSecond: number;
  /** Two bytes per bin: peak, then RMS, each 0..255 of the track's loudest peak. */
  data: Uint8Array;
}

export interface ConnectUrl {
  label: string;
  url: string;
}
