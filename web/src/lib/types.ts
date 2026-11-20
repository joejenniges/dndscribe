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
  channelName: string | null;
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

export interface DjPlaylist {
  id: number;
  name: string;
  trackIds: number[];
}

export interface DjScene {
  id: number;
  name: string;
  playlistId: number | null;
  color: string;
  volume: number;
  shuffle: boolean;
  loop: boolean;
  fadeSeconds: number | null;
  position: number;
}

export interface DjTrackInfo {
  id: number;
  title: string;
  artist: string;
  durationMs: number;
}

export interface DjState {
  connected: boolean;
  playing: boolean;
  paused: boolean;
  sceneId: number | null;
  playlistId: number | null;
  programName: string;
  track: DjTrackInfo | null;
  positionMs: number;
  at: number;
  upNext: DjTrackInfo[];
  loop: boolean;
  shuffle: boolean;
  volume: number;
  fadeSeconds: number;
  error?: string;
  /** Client clock when this state arrived; used to interpolate position. */
  receivedAt?: number;
}
