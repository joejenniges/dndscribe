import { writable, type Writable } from 'svelte/store';
import type { Campaign, Session, TranscriptionLine, Partial, NicknamePreset, NicknameCategory, IgnoredUser } from '$lib/types';
import { subscribe as wsSubscribe } from '$lib/ws';

// Current campaign context -- set when navigating to /campaigns/[id]
export const currentCampaign: Writable<Campaign | null> = writable(null);

// Campaign-scoped data
export const sessions: Writable<Session[]> = writable([]);
export const lines: Writable<TranscriptionLine[]> = writable([]);
// Live partial hypotheses from the streaming engine, keyed by discordUsername.
// Ephemeral: cleared when the matching final line arrives or the session ends.
export const partials: Writable<Map<string, Partial>> = writable(new Map());
export const hotwords: Writable<string[]> = writable([]);
export const presets: Writable<NicknamePreset[]> = writable([]);
export const categories: Writable<NicknameCategory[]> = writable([]);
export const ignoredUsers: Writable<IgnoredUser[]> = writable([]);

// Initialize campaign-scoped WS message handling
export function initCampaignWsHandlers(): () => void {
  return wsSubscribe((msg) => {
    switch (msg.type) {
      case 'line':
        // Insert sorted by timestamp so lines from concurrent speakers
        // appear in chronological order, not arrival order.
        lines.update(l => {
          const newLine = msg.data;
          const newTime = new Date(newLine.timestamp).getTime();
          // Find insertion point -- most new lines go near the end
          let i = l.length;
          while (i > 0 && new Date(l[i - 1].timestamp).getTime() > newTime) {
            i--;
          }
          const updated = [...l];
          updated.splice(i, 0, newLine);
          return updated;
        });
        // A committed line supersedes that speaker's live partial.
        partials.update(m => {
          if (!m.has(msg.data.discordUsername)) return m;
          const n = new Map(m);
          n.delete(msg.data.discordUsername);
          return n;
        });
        break;
      case 'partial': {
        // Provisional streaming hypothesis: upsert by speaker, drop if empty.
        const p = msg.data as Partial;
        partials.update(m => {
          const n = new Map(m);
          if (p.text && p.text.trim()) {
            n.set(p.discordUsername, p);
          } else {
            n.delete(p.discordUsername);
          }
          return n;
        });
        break;
      }
      case 'edit':
        lines.update(l => l.map(line =>
          line.id === msg.data.id ? { ...line, text: msg.data.text } : line
        ));
        break;
      case 'delete':
        lines.update(l => l.filter(line => line.id !== msg.data.id));
        break;
      case 'reload_lines':
        lines.set(msg.data || []);
        partials.set(new Map());
        break;
      case 'hotwords':
        hotwords.set(msg.data || []);
        break;
      case 'sessions':
        sessions.set(msg.data || []);
        break;
      case 'session_started':
        sessions.update(s => [msg.data, ...s]);
        break;
      case 'presets_updated':
        presets.set(msg.presets || []);
        categories.set(msg.categories || []);
        break;
      case 'ignored_users_changed':
        // Legacy signal -- ignored
        break;
      case 'ignored_users':
        ignoredUsers.set(msg.data || []);
        break;
      case 'finalized':
        // Session ended -- lines stay visible, session list will update.
        // Clear any lingering partials.
        partials.set(new Map());
        break;
    }
  });
}
