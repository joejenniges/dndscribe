import { writable, type Writable } from 'svelte/store';
import type { DjMeters, DjState, PresenceClient } from '$lib/types';

// Server-owned DJ state, pushed over the WebSocket (dj_state). Pages render
// from this rather than from HTTP responses so the table monitor and a tablet
// always show the same thing.
export const djState: Writable<DjState | null> = writable(null);

// Bumped on every dj_library event (tracks, playlists or scenes changed on
// any client). Pages that list those refetch when it changes.
export const djLibraryVersion: Writable<number> = writable(0);

// Peak levels, ~10 updates a second while anything is audible (dj_meters).
// Kept apart from djState so a meter tick never re-renders the whole page.
export const djMeters: Writable<DjMeters> = writable({ decks: [0, 0], master: 0 });

// Devices connected to the server (presence). Informational only.
export const presence: Writable<PresenceClient[]> = writable([]);
