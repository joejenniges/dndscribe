import type {
  Campaign, Session, TranscriptionLine, VoiceChannel, VoiceMember,
  BotStatus, IgnoredUser, NicknamePreset, NicknameCategory,
  SessionPreview, SearchResult,
} from './types';

const BASE = '';

async function request<T>(url: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(BASE + url, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(body.error || res.statusText);
  }
  return res.json();
}

// ---- Campaigns ----

export const campaigns = {
  list: () => request<Campaign[]>('/api/campaigns'),
  get: (id: number) => request<Campaign>(`/api/campaigns/${id}`),
  create: (name: string, description?: string) =>
    request<Campaign>('/api/campaigns', { method: 'POST', body: JSON.stringify({ name, description }) }),
  update: (id: number, name: string, description?: string) =>
    request<Campaign>(`/api/campaigns/${id}`, { method: 'PUT', body: JSON.stringify({ name, description }) }),
  delete: (id: number) =>
    request<{ success: boolean }>(`/api/campaigns/${id}`, { method: 'DELETE' }),
};

// ---- Campaign-scoped endpoints ----

function scoped(campaignId: number) {
  const prefix = `/api/campaigns/${campaignId}`;

  return {
    // Sessions
    sessions: {
      list: () => request<Session[]>(`${prefix}/sessions`),
      delete: (id: number) => request<{ success: boolean }>(`${prefix}/sessions/${id}`, { method: 'DELETE' }),
      update: (id: number, name: string) =>
        request<{ success: boolean }>(`${prefix}/sessions/${id}`, {
          method: 'PUT', body: JSON.stringify({ name }),
        }),
      preview: (id: number) => request<SessionPreview>(`${prefix}/sessions/${id}/preview`),
      merge: (sessionIds: number[], name: string) =>
        request<{ success: boolean; newSessionId: number }>(`${prefix}/sessions/merge`, {
          method: 'POST', body: JSON.stringify({ sessionIds, name }),
        }),
    },

    // Transcription lines
    lines: {
      list: (sessionId?: number) => {
        const params = sessionId ? `?session=${sessionId}` : '';
        return request<TranscriptionLine[]>(`${prefix}/lines${params}`);
      },
      update: (id: number, text: string) =>
        request<{ success: boolean; autoAdded: string[] }>(`${prefix}/lines/${id}`, {
          method: 'POST', body: JSON.stringify({ text }),
        }),
      delete: (id: number) =>
        request<{ success: boolean }>(`${prefix}/lines/${id}`, { method: 'DELETE' }),
    },

    // Search
    search: (q: string) => request<SearchResult[]>(`${prefix}/search?q=${encodeURIComponent(q)}`),

    // Hotwords
    hotwords: {
      list: () => request<string[]>(`${prefix}/hotwords`),
      add: (word: string) =>
        request<{ added: boolean }>(`${prefix}/hotwords`, { method: 'POST', body: JSON.stringify({ word }) }),
      remove: (word: string) =>
        request<{ removed: boolean }>(`${prefix}/hotwords`, { method: 'DELETE', body: JSON.stringify({ word }) }),
      update: (oldWord: string, newWord: string) =>
        request<{ updated: boolean }>(`${prefix}/hotwords`, { method: 'PUT', body: JSON.stringify({ oldWord, newWord }) }),
    },

    // Members
    members: () => request<VoiceMember[]>(`${prefix}/members`),

    // Character names
    setCharacterName: (userId: string, name: string | null) =>
      request<{ success: boolean }>(`${prefix}/character-name`, {
        method: 'POST', body: JSON.stringify({ userId, name }),
      }),

    // Bulk nickname
    bulkNickname: (lineIds: number[], nicknames: Record<string, string | null>) =>
      request<{ success: boolean }>(`${prefix}/bulk-nickname`, {
        method: 'POST', body: JSON.stringify({ lineIds, nicknames }),
      }),

    // Presets
    presets: {
      list: (userId?: string) => {
        const params = userId ? `?userId=${userId}` : '';
        return request<NicknamePreset[]>(`${prefix}/presets${params}`);
      },
      add: (userId: string, label: string, position: number) =>
        request<{ id: number }>(`${prefix}/presets`, {
          method: 'POST', body: JSON.stringify({ discordUserId: userId, label, position }),
        }),
      update: (id: number, label: string, position?: number) =>
        request<{ success: boolean }>(`${prefix}/presets/${id}`, {
          method: 'PUT', body: JSON.stringify({ label, position }),
        }),
      delete: (id: number) =>
        request<{ success: boolean }>(`${prefix}/presets/${id}`, { method: 'DELETE' }),
      move: (presetId: number, categoryId: number | null) =>
        request<{ success: boolean }>(`${prefix}/presets/move`, {
          method: 'POST', body: JSON.stringify({ presetId, categoryId }),
        }),
    },

    // Categories
    categories: {
      list: (userId?: string) => {
        const params = userId ? `?userId=${userId}` : '';
        return request<NicknameCategory[]>(`${prefix}/categories${params}`);
      },
      add: (userId: string, name: string, position: number) =>
        request<{ id: number }>(`${prefix}/categories`, {
          method: 'POST', body: JSON.stringify({ discordUserId: userId, name, position }),
        }),
      update: (id: number, name: string, position?: number) =>
        request<{ success: boolean }>(`${prefix}/categories/${id}`, {
          method: 'PUT', body: JSON.stringify({ name, position }),
        }),
      delete: (id: number) =>
        request<{ success: boolean }>(`${prefix}/categories/${id}`, { method: 'DELETE' }),
    },

    // Ignored users
    ignoredUsers: {
      list: () => request<IgnoredUser[]>(`${prefix}/ignored-users`),
      add: (userId: string, username: string) =>
        request<{ added: boolean }>(`${prefix}/ignored-users`, {
          method: 'POST', body: JSON.stringify({ userId, username }),
        }),
      remove: (userId: string) =>
        request<{ removed: boolean }>(`${prefix}/ignored-users`, {
          method: 'DELETE', body: JSON.stringify({ userId }),
        }),
    },
  };
}

export { scoped };

// ---- Global endpoints ----

export const status = {
  get: () => request<BotStatus>('/api/status'),
};

export const channels = {
  list: () => request<VoiceChannel[]>('/api/channels'),
};

export const voice = {
  join: (channelId: string, campaignId: number) =>
    request<{ success: boolean; channelName: string }>('/api/join', {
      method: 'POST', body: JSON.stringify({ channelId, campaignId }),
    }),
  leave: () =>
    request<{ success: boolean; outFile: string | null }>('/api/leave', { method: 'POST' }),
  pause: () =>
    request<{ paused: boolean }>('/api/pause', { method: 'POST' }),
};

export const saveRecordings = {
  get: () => request<{ enabled: boolean }>('/api/save-recordings'),
  set: (enabled: boolean) =>
    request<{ enabled: boolean }>('/api/save-recordings', {
      method: 'POST', body: JSON.stringify({ enabled }),
    }),
};
