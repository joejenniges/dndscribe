// Hotcues, goto triggers and beat grids on the client (v4).
//
// Definitions are edited over HTTP; jumping and arming are WS commands. The
// server then sends every device the new deck state (decks carry their
// loaded track's cues), so nothing here keeps its own copy of cue data for
// loaded tracks. Beat grids are too big for the state snapshot and are
// fetched per track and cached here, dropped when the server says that
// track's analysis changed.
import { writable, get } from 'svelte/store';
import { dj } from '$lib/api';
import { djCmd } from '$lib/dj/control';
import { djAnalysisChange } from '$lib/stores/dj';
import { addToast } from '$lib/stores/toast';
import type { DjAnalysis, DjGoto, DjHotcue } from '$lib/types';

/** Slot colours 1..8, the same everywhere (buttons, markers, menus). */
export const HOTCUE_COLORS = ['#a855f7', '#22d3ee', '#f59e0b', '#22c55e', '#ec4899', '#8b5cf6', '#f97316', '#3b82f6'];
export const GOTO_COLOR = '#facc15';
export const hotcueColor = (h: Pick<DjHotcue, 'slot' | 'color'>) => h.color || HOTCUE_COLORS[(h.slot - 1) % 8];

/** m:ss.mmm (h:mm:ss.mmm past an hour): cue positions are shown exactly. */
export function fmtCue(ms: number): string {
  ms = Math.max(0, Math.round(ms));
  const h = Math.floor(ms / 3_600_000);
  const m = Math.floor((ms % 3_600_000) / 60_000);
  const s = Math.floor((ms % 60_000) / 1000);
  const r = String(ms % 1000).padStart(3, '0');
  return h ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${r}` : `${m}:${String(s).padStart(2, '0')}.${r}`;
}

/** Parse h:mm:ss.mmm, m:ss.mmm or seconds. null if it isn't a time. */
export function parseCue(text: string): number | null {
  const t = text.trim();
  if (!/^\d+(:\d{1,2}){0,2}(\.\d{1,3})?$/.test(t)) return null;
  const [main, frac = ''] = t.split('.');
  const parts = main.split(':').map(Number);
  let secs = 0;
  for (const p of parts) secs = secs * 60 + p;
  return secs * 1000 + Number(frac.padEnd(3, '0'));
}

// ---- Snapping (per device) ----

function readNum(key: string, dflt: number): number {
  try {
    const v = localStorage.getItem(key);
    return v === null ? dflt : Number(v) || dflt;
  } catch {
    return dflt;
  }
}
function save(key: string, v: string) {
  try {
    localStorage.setItem(key, v);
  } catch {
    /* not remembered */
  }
}

/** Snap new and dragged cues to the nearest beat (Shift while dragging
 *  overrides). Local to this device. */
export const snapOn = writable<boolean>(readNum('dj.snap', 1) === 1);
snapOn.subscribe((v) => save('dj.snap', v ? '1' : '0'));
/** Only snap when a beat is within this many ms; otherwise the exact time
 *  is kept, so a cue is never moved far from where it was put. */
export const SNAP_TOLERANCES = [40, 80, 120, 200];
export const snapTolMs = writable<number>(readNum('dj.snapTol', 120));
snapTolMs.subscribe((v) => save('dj.snapTol', String(v)));

/** The nearest beat to ms within tolerance, else ms. */
export function snapMs(ms: number, a: DjAnalysis | null, override = false): number {
  if (override || !get(snapOn) || !a?.usable || !a.beats.length) return Math.round(ms);
  const b = a.beats;
  let lo = 0;
  let hi = b.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (b[mid] < ms) lo = mid + 1;
    else hi = mid;
  }
  let best = b[lo];
  if (lo > 0 && Math.abs(b[lo - 1] - ms) < Math.abs(best - ms)) best = b[lo - 1];
  return Math.abs(best - ms) <= get(snapTolMs) ? best : Math.round(ms);
}

// ---- Beat grids ----

const grids = new Map<number, Promise<DjAnalysis>>();
/** Bumped when any cached grid is replaced, so views redraw. */
export const gridVersion = writable(0);

export function loadAnalysis(trackId: number): Promise<DjAnalysis> {
  let p = grids.get(trackId);
  if (!p) {
    p = dj.analysis.get(trackId);
    p.catch(() => grids.delete(trackId));
    grids.set(trackId, p);
    while (grids.size > 16) grids.delete(grids.keys().next().value!);
  }
  return p;
}

djAnalysisChange.subscribe(({ trackId }) => {
  if (trackId && grids.delete(trackId)) gridVersion.update((n) => n + 1);
});

export function analysisLabel(a: DjAnalysis | null): string {
  if (!a) return '';
  switch (a.status) {
    case 'none':
    case 'pending':
      return 'Beat analysis queued';
    case 'running':
      return 'Analysing beats…';
    case 'failed':
      return 'Beat analysis failed';
    case 'unavailable':
      return a.error === 'beat grid turned off' ? 'Beat grid off' : 'No beat grid';
  }
  if (!a.usable) return 'Beat analysis uncertain';
  return `${a.bpm?.toFixed(1)} BPM${a.source === 'manual' ? ' (manual)' : ''}`;
}

// ---- Actions (shared by deck buttons, waveforms and menus) ----

async function attempt<T>(what: string, f: () => Promise<T>): Promise<T | undefined> {
  try {
    return await f();
  } catch (e) {
    addToast(`${what}: ${(e as Error).message}`, 'error');
    return undefined;
  }
}

export function jumpHotcue(deck: number, slot: number) {
  return attempt(`Hotcue ${slot}`, () => djCmd('deck.hotcue', { deck, slot }));
}

export function setHotcue(trackId: number, slot: number, positionMs: number, label = '', color = '') {
  return attempt(`Hotcue ${slot}`, () => dj.cues.setHotcue(trackId, slot, { positionMs: Math.round(positionMs), label, color }));
}

export function deleteHotcue(trackId: number, slot: number, gotos: DjGoto[]) {
  const dependent = gotos.filter((g) => g.targetSlot === slot && g.enabled).length;
  return attempt(`Delete hotcue ${slot}`, async () => {
    await dj.cues.deleteHotcue(trackId, slot);
    if (dependent) addToast(`${dependent} goto${dependent === 1 ? '' : 's'} aimed at hotcue ${slot} turned off (target missing)`, 'info');
  });
}

export function addGoto(trackId: number, positionMs: number, targetSlot: number) {
  return attempt('Add goto', () => dj.cues.createGoto(trackId, { positionMs: Math.round(positionMs), targetSlot, enabled: true }));
}

export function saveGoto(trackId: number, g: DjGoto) {
  return attempt('Save goto', () =>
    dj.cues.updateGoto(trackId, g.id, { positionMs: Math.round(g.positionMs), targetSlot: g.targetSlot, enabled: g.enabled, label: g.label }),
  );
}

export function deleteGoto(trackId: number, id: number) {
  return attempt('Delete goto', () => dj.cues.deleteGoto(trackId, id));
}

export function toggleGoto(trackId: number, g: DjGoto) {
  return attempt(g.enabled ? 'Disarm goto' : 'Arm goto', () => djCmd('goto.toggle', { trackId, gotoId: g.id, on: !g.enabled }));
}

/** First free hotcue slot, or 0. */
export function freeSlot(hotcues: DjHotcue[]): number {
  for (let s = 1; s <= 8; s++) if (!hotcues.some((h) => h.slot === s)) return s;
  return 0;
}

/** Goto display numbers: G1, G2... in position order (as stored). */
export const gotoName = (gotos: DjGoto[], g: DjGoto) => `G${gotos.findIndex((x) => x.id === g.id) + 1}`;

// ---- The cue menu (one per page, opened from any waveform or button) ----

export interface CueMenuRequest {
  deck: 0 | 1;
  x: number;
  y: number;
  /** Track time the menu was opened at (unsnapped). */
  atMs: number;
  /** Opened on an existing marker or button. */
  hotcue?: number;
  gotoId?: number;
}
export const cueMenu = writable<CueMenuRequest | null>(null);

/** Editor dialog request. */
export type CueEditRequest =
  | { deck: 0 | 1; kind: 'hotcue'; slot: number; atMs?: number }
  | { deck: 0 | 1; kind: 'goto'; gotoId?: number; atMs?: number }
  | { deck: 0 | 1; kind: 'grid'; atMs: number };
export const cueEdit = writable<CueEditRequest | null>(null);

/** Long press (touch) as a context menu: calls f after 500ms of holding
 *  still. Returns handlers to spread onto pointer events; call cancel()
 *  from the element's own move/up handlers too. */
export function longPress(f: (e: PointerEvent) => void, ms = 500) {
  let timer: ReturnType<typeof setTimeout> | null = null;
  let start: { x: number; y: number } | null = null;
  const cancel = () => {
    if (timer) clearTimeout(timer);
    timer = null;
    start = null;
  };
  return {
    down(e: PointerEvent) {
      if (e.pointerType === 'mouse') return;
      cancel();
      start = { x: e.clientX, y: e.clientY };
      timer = setTimeout(() => {
        timer = null;
        f(e);
      }, ms);
    },
    move(e: PointerEvent) {
      if (start && Math.hypot(e.clientX - start.x, e.clientY - start.y) > 10) cancel();
    },
    cancel,
    /** True if the press turned into a menu (suppress the click). */
    fired: () => timer === null && start !== null,
  };
}

/** Save a dragged marker's new position. */
export function moveMarker(
  trackId: number, ref: { kind: 'hotcue'; slot: number } | { kind: 'goto'; id: number }, ms: number,
  hotcues: DjHotcue[], gotos: DjGoto[],
) {
  if (ref.kind === 'hotcue') {
    const h = hotcues.find((x) => x.slot === ref.slot);
    return setHotcue(trackId, ref.slot, ms, h?.label ?? '', h?.color ?? '');
  }
  const g = gotos.find((x) => x.id === ref.id);
  return g ? saveGoto(trackId, { ...g, positionMs: ms }) : Promise.resolve(undefined);
}
