import type { ConnectUrl,
  Campaign, Session, TranscriptionLine, VoiceChannel, VoiceMember,
  BotStatus, IgnoredUser, NicknamePreset, NicknameCategory,
  SessionPreview, SearchResult,
  DjTrack, DjScene, DjState, DjSettings, DjWaveform, SfxTab, SfxPad, DjTrackCues, DjAnalysis, DjGridEdit,
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

/** Join a voice channel for DJ playback only, without recording. */
export const music = {
  join: (channelId: string) =>
    request<{ channelName: string }>('/api/music/join', { method: 'POST', body: JSON.stringify({ channelId }) }),
  leave: () => request<{ success: boolean }>('/api/music/leave', { method: 'POST' }),
};

export const saveRecordings = {
  get: () => request<{ enabled: boolean }>('/api/save-recordings'),
  set: (enabled: boolean) =>
    request<{ enabled: boolean }>('/api/save-recordings', {
      method: 'POST', body: JSON.stringify({ enabled }),
    }),
};

// ---- DJ ----

export class UploadError extends Error {
  constructor(message: string, public status: number) {
    super(message);
  }
}

interface PutOpts {
  onProgress?: (sent: number) => void;
  signal?: AbortSignal;
}

/**
 * Upload one audio file to `path` (music-folder-relative, including the file
 * name). Rejects with UploadError (status 409 = already exists) or an
 * AbortError.
 */
function uploadTrack(path: string, file: File, opts: PutOpts & { overwrite?: boolean } = {}): Promise<DjTrack> {
  const q = `path=${encodeURIComponent(path)}${opts.overwrite ? '&overwrite=1' : ''}`;
  return putFile<DjTrack>(`/api/dj/files?${q}`, file, opts);
}

/** PUT a file as the raw request body. XHR rather than fetch: fetch has no
 * upload progress events. */
function putFile<T>(url: string, file: Blob, opts: PutOpts = {}): Promise<T> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('PUT', BASE + url);
    xhr.setRequestHeader('Content-Type', 'application/octet-stream');
    xhr.upload.onprogress = (e) => opts.onProgress?.(e.loaded);
    xhr.onload = () => {
      let body: { error?: string } = {};
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        /* non-JSON error page */
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body as T);
      else reject(new UploadError(body.error || xhr.statusText || `HTTP ${xhr.status}`, xhr.status));
    };
    xhr.onerror = () => reject(new UploadError('network error', 0));
    xhr.onabort = () => reject(new DOMException('Upload cancelled', 'AbortError'));
    if (opts.signal) {
      if (opts.signal.aborted) return reject(new DOMException('Upload cancelled', 'AbortError'));
      opts.signal.addEventListener('abort', () => xhr.abort(), { once: true });
    }
    xhr.send(file);
  });
}

const post = <T>(url: string, body?: unknown) =>
  request<T>(url, { method: 'POST', body: body === undefined ? undefined : JSON.stringify(body) });

type SceneInput = Omit<DjScene, 'id' | 'position' | 'image'>;

// Addresses another device can open this app on (shown as QR codes).
export const connect = {
  urls: () => request<{ urls: ConnectUrl[] }>('/api/connect'),
};

export const dj = {
  state: () => request<DjState>('/api/dj/state'),
  play: (target: { sceneId?: number; folder?: string; trackId?: number; fadeSeconds?: number }) =>
    post<DjState>('/api/dj/play', target),
  stop: (fadeSeconds?: number) => post<DjState>('/api/dj/stop', fadeSeconds === undefined ? {} : { fadeSeconds }),
  pause: () => post<DjState>('/api/dj/pause'),
  resume: () => post<DjState>('/api/dj/resume'),
  next: () => post<DjState>('/api/dj/next'),
  seek: (positionMs: number) => post<DjState>('/api/dj/seek', { positionMs: Math.round(positionMs) }),
  volume: (volume: number) => post<DjState>('/api/dj/volume', { volume }),
  fade: (seconds: number) => post<DjState>('/api/dj/fade', { seconds }),
  tone: (seconds = 2) => post<DjState>('/api/dj/tone', { seconds }),
  settings: () => request<DjSettings>('/api/dj/settings'),
  saveSettings: (s: Partial<DjSettings>) =>
    request<DjSettings>('/api/dj/settings', { method: 'PUT', body: JSON.stringify(s) }),
  /** Raw waveform: [peak, rms] bytes per bin. The browser revalidates by
   *  ETag, so a track's waveform is downloaded once until the file changes. */
  waveform: async (trackId: number, signal?: AbortSignal): Promise<DjWaveform> => {
    const res = await fetch(`${BASE}/api/dj/tracks/${trackId}/waveform`, { signal });
    if (!res.ok) throw new Error(`waveform: HTTP ${res.status}`);
    return {
      binsPerSecond: Number(res.headers.get('X-Bins-Per-Second')) || 100,
      data: new Uint8Array(await res.arrayBuffer()),
    };
  },

  /** Hotcue and goto definitions (live jumps and arming are WS commands). */
  cues: {
    get: (trackId: number) => request<DjTrackCues>(`/api/dj/tracks/${trackId}/cues`),
    setHotcue: (trackId: number, slot: number, h: { positionMs: number; label?: string; color?: string }) =>
      request<DjTrackCues>(`/api/dj/tracks/${trackId}/hotcues/${slot}`, { method: 'PUT', body: JSON.stringify(h) }),
    deleteHotcue: (trackId: number, slot: number) =>
      request<DjTrackCues>(`/api/dj/tracks/${trackId}/hotcues/${slot}`, { method: 'DELETE' }),
    createGoto: (trackId: number, g: { positionMs: number; targetSlot: number; enabled?: boolean; label?: string }) =>
      post<DjTrackCues>(`/api/dj/tracks/${trackId}/gotos`, g),
    updateGoto: (trackId: number, id: number, g: { positionMs: number; targetSlot: number; enabled: boolean; label?: string }) =>
      request<DjTrackCues>(`/api/dj/tracks/${trackId}/gotos/${id}`, { method: 'PUT', body: JSON.stringify(g) }),
    deleteGoto: (trackId: number, id: number) =>
      request<DjTrackCues>(`/api/dj/tracks/${trackId}/gotos/${id}`, { method: 'DELETE' }),
  },
  /** Beat analysis: GET queues it if missing; POST re-analyses; PATCH edits. */
  analysis: {
    get: (trackId: number) => request<DjAnalysis>(`/api/dj/tracks/${trackId}/analysis`),
    reanalyze: (trackId: number) => post<{ status: string }>(`/api/dj/tracks/${trackId}/analysis`),
    edit: (trackId: number, e: DjGridEdit) =>
      request<DjAnalysis>(`/api/dj/tracks/${trackId}/analysis`, { method: 'PATCH', body: JSON.stringify(e) }),
  },

  tracks: () => request<DjTrack[]>('/api/dj/tracks'),
  rescan: () => post<{ found: number; probed: number; missing: number }>('/api/dj/tracks/rescan'),

  deleteTrack: (id: number) => request<{ success: boolean }>(`/api/dj/tracks/${id}`, { method: 'DELETE' }),

  folders: {
    list: () => request<string[]>('/api/dj/folders'),
    create: (path: string) => post<{ path: string }>('/api/dj/folders', { path }),
    rename: (from: string, to: string) => post<{ path: string }>('/api/dj/folders/rename', { from, to }),
    /** Save the order of sibling folders. */
    order: (paths: string[]) =>
      request<{ success: boolean }>('/api/dj/folders/order', { method: 'PUT', body: JSON.stringify({ paths }) }),
    delete: (path: string) =>
      request<{ success: boolean }>(`/api/dj/folders?path=${encodeURIComponent(path)}`, { method: 'DELETE' }),
  },

  upload: uploadTrack,
  /** Which of these upload targets already exist (asked before a batch). */
  existing: (paths: string[]) => post<{ existing: string[] }>('/api/dj/files/exists', { paths }),

  sfx: {
    list: () => request<{ tabs: SfxTab[]; pads: SfxPad[] }>('/api/dj/sfx'),
    imageUrl: (key: string) => `${BASE}/api/dj/sfx/images/${key}`,
    trigger: (id: number, requestId: string) => post<DjState>(`/api/dj/sfx/pads/${id}/trigger`, { requestId }),
    stop: (id: number) => post<DjState>(`/api/dj/sfx/pads/${id}/stop`),
    stopAll: () => post<DjState>('/api/dj/sfx/stop'),
    gain: (gain: number) => post<DjState>('/api/dj/sfx/gain', { gain }),
    /** Add a pad, or with opts.replace overwrite that pad's sound. */
    upload: (tabId: number, file: File, opts: PutOpts & { replace?: number } = {}) =>
      putFile<SfxPad>(
        `/api/dj/sfx/upload?tab=${tabId}&name=${encodeURIComponent(file.name)}${opts.replace ? `&replace=${opts.replace}` : ''}`,
        file, opts),
    tabs: {
      create: (title: string) => post<SfxTab>('/api/dj/sfx/tabs', { title }),
      rename: (id: number, title: string) =>
        request<{ success: boolean }>(`/api/dj/sfx/tabs/${id}`, { method: 'PUT', body: JSON.stringify({ title }) }),
      delete: (id: number) => request<{ success: boolean }>(`/api/dj/sfx/tabs/${id}`, { method: 'DELETE' }),
      order: (ids: number[]) => post<{ success: boolean }>('/api/dj/sfx/tabs/order', { ids }),
      /** Set a tab's pads in order; pads from other tabs move into it. */
      arrange: (id: number, padIds: number[]) => post<{ success: boolean }>(`/api/dj/sfx/tabs/${id}/arrange`, { ids: padIds }),
    },
    pads: {
      update: (id: number, patch: Partial<Pick<SfxPad, 'label' | 'color' | 'gain' | 'tabId'>>) =>
        request<SfxPad>(`/api/dj/sfx/pads/${id}`, { method: 'PUT', body: JSON.stringify(patch) }),
      delete: (id: number) => request<{ success: boolean }>(`/api/dj/sfx/pads/${id}`, { method: 'DELETE' }),
      setImage: (id: number, file: File) => putFile<{ image: string }>(`/api/dj/sfx/pads/${id}/image`, file),
      clearImage: (id: number) => request<{ success: boolean }>(`/api/dj/sfx/pads/${id}/image`, { method: 'DELETE' }),
    },
  },

  scenes: {
    list: () => request<DjScene[]>('/api/dj/scenes'),
    create: (s: SceneInput) => post<DjScene>('/api/dj/scenes', s),
    update: (id: number, s: SceneInput) =>
      request<DjScene>(`/api/dj/scenes/${id}`, { method: 'PUT', body: JSON.stringify(s) }),
    delete: (id: number) => request<{ success: boolean }>(`/api/dj/scenes/${id}`, { method: 'DELETE' }),
    order: (ids: number[]) => post<{ success: boolean }>('/api/dj/scenes/order', { ids }),
    setImage: (id: number, file: File) => putFile<{ image: string }>(`/api/dj/scenes/${id}/image`, file),
    clearImage: (id: number) => request<{ success: boolean }>(`/api/dj/scenes/${id}/image`, { method: 'DELETE' }),
  },
};
