import { writable, type Writable } from 'svelte/store';
import { subscribe as wsSubscribe } from '$lib/ws';
import type { BotStatus, VoiceMember } from '$lib/types';

export const botStatus: Writable<BotStatus> = writable({ recording: false, channelName: null, campaignId: null });
export const voiceMembers: Writable<VoiceMember[]> = writable([]);
export const saveRecordingsEnabled: Writable<boolean> = writable(false);
export const wsConnected: Writable<boolean> = writable(false);

// Initialize WS message handling -- call once from root layout
export function initWsStores(): () => void {
  return wsSubscribe((msg) => {
    switch (msg.type) {
      case 'init':
        botStatus.set(msg.status);
        voiceMembers.set(msg.members || []);
        saveRecordingsEnabled.set(msg.saveRecordings ?? false);
        wsConnected.set(true);
        break;
      case 'status':
        botStatus.set(msg.data);
        break;
      case 'members':
        voiceMembers.set(msg.data || []);
        break;
      case 'save_recordings':
        saveRecordingsEnabled.set(msg.data);
        break;
    }
  });
}
