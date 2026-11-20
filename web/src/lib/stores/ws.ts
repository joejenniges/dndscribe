import { writable, type Writable } from 'svelte/store';
import { subscribe as wsSubscribe } from '$lib/ws';
import type { BotStatus, VoiceMember } from '$lib/types';
import { djState, djLibraryVersion } from '$lib/stores/dj';

export const botStatus: Writable<BotStatus> = writable({ recording: false, channelName: null, campaignId: null });
export const voiceMembers: Writable<VoiceMember[]> = writable([]);
export const saveRecordingsEnabled: Writable<boolean> = writable(false);
// Optimistically true so nothing flashes "offline" during the first connect.
export const wsConnected: Writable<boolean> = writable(true);

// Initialize WS message handling -- call once from root layout
export function initWsStores(): () => void {
  return wsSubscribe((msg) => {
    switch (msg.type) {
      case 'init':
        botStatus.set(msg.status);
        voiceMembers.set(msg.members || []);
        saveRecordingsEnabled.set(msg.saveRecordings ?? false);
        if (msg.dj) djState.set({ ...msg.dj, receivedAt: Date.now() });
        wsConnected.set(true);
        break;
      case 'status':
        botStatus.set(msg.data);
        break;
      case 'members':
        voiceMembers.set(msg.data || []);
        break;
      case 'ws_closed':
        wsConnected.set(false);
        break;
      case 'dj_state':
        djState.set({ ...msg.data, receivedAt: Date.now() });
        break;
      case 'dj_library':
        djLibraryVersion.update((n) => n + 1);
        break;
      case 'save_recordings':
        saveRecordingsEnabled.set(msg.data);
        break;
    }
  });
}
