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
}

// Partial is a provisional live hypothesis from the streaming engine. It is
// ephemeral (never persisted) and is superseded by a committed TranscriptionLine
// when the utterance finalizes.
export interface Partial {
  userId: string;
  discordUsername: string;
  nickname: string | null;
  text: string;
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
