// Client-side views of the server's DJ state: positions and the crossfader
// are extrapolated between snapshots instead of being streamed.
import type { DjState } from '$lib/types';

/** Where deck i's track is now (ms), extrapolated from the last snapshot. */
export function deckPosition(st: DjState | null, i: number, now = Date.now()): number {
  const d = st?.decks?.[i];
  if (!st || !d?.track) return 0;
  let p = d.positionMs;
  if (d.playing && st.connected) p += now - (st.receivedAt ?? now);
  const dur = d.track.durationMs;
  return dur > 0 ? Math.max(0, Math.min(p, dur)) : Math.max(0, p);
}

/** The crossfader position now, following an automated move if one is
 *  running (the server sends its start and length, not every step). */
export function crossfaderNow(st: DjState | null, now = Date.now()): number {
  if (!st) return 0;
  const t = st.transition;
  if (!t || t.durationMs <= 0) return st.crossfader;
  // Elapsed = time on the server when sampled, plus local time since.
  const elapsed = st.at - t.startedAt + (now - (st.receivedAt ?? now));
  const p = Math.max(0, Math.min(1, elapsed / t.durationMs));
  return t.from + (t.to - t.from) * p;
}

/** Seconds left in an automated crossfader move. */
export function transitionLeft(st: DjState | null, now = Date.now()): number {
  const t = st?.transition;
  if (!st || !t) return 0;
  const elapsed = st.at - t.startedAt + (now - (st.receivedAt ?? now));
  return Math.max(0, (t.durationMs - elapsed) / 1000);
}

export const DECK_NAMES = ['A', 'B'] as const;

/** Each deck's level through the crossfader (the server's equal-power law). */
export function xfWeights(x: number): [number, number] {
  return [Math.cos((x * Math.PI) / 2), Math.sin((x * Math.PI) / 2)];
}

/** Whether deck i is actually being heard: playing, in voice, and not
 *  faded out by the crossfader. The server's `onAir` ignores the fader (a
 *  deck playing on the far side still counts), so the "On air" badge uses
 *  this instead. */
export function deckHeard(st: DjState | null, i: number, now = Date.now()): boolean {
  const d = st?.decks?.[i];
  if (!st?.connected || !d?.track || !d.playing || (d.gain ?? 1) <= 0.01) return false;
  return xfWeights(crossfaderNow(st, now))[i] > 0.05;
}

/** The deck the audience mainly hears, or -1. */
export function heardDeck(st: DjState | null, now = Date.now()): number {
  const h = [deckHeard(st, 0, now), deckHeard(st, 1, now)];
  if (h[0] && h[1]) {
    const w = xfWeights(crossfaderNow(st, now));
    return w[0] * (st!.decks[0].gain ?? 1) >= w[1] * (st!.decks[1].gain ?? 1) ? 0 : 1;
  }
  return h[0] ? 0 : h[1] ? 1 : -1;
}

/** Device kind for presence and the default layout. Touch-first screens
 *  (coarse pointer) are tablets or phones; everything else is a desktop. */
export function deviceKind(): 'desktop' | 'tablet' | 'phone' {
  if (typeof window === 'undefined') return 'desktop';
  const coarse = window.matchMedia?.('(pointer: coarse)').matches;
  if (!coarse) return 'desktop';
  return Math.min(window.screen.width, window.screen.height) < 600 ? 'phone' : 'tablet';
}

/** localStorage that never throws (private windows, blocked storage). */
export const local = {
  get(key: string): string | null {
    try {
      return localStorage.getItem(key);
    } catch {
      return null;
    }
  },
  set(key: string, v: string) {
    try {
      localStorage.setItem(key, v);
    } catch {
      /* per-device convenience only */
    }
  },
};
