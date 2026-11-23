// Drawing and hit-testing cue markers and beat ticks on the waveform
// canvases (the deck overviews and the scrolling lanes share these).
import type { DjGoto, DjHotcue } from '$lib/types';
import { GOTO_COLOR, hotcueColor } from './cues';

export type MarkerRef = { kind: 'hotcue'; slot: number } | { kind: 'goto'; id: number };

/** A marker being dragged: drawn at ms instead of its saved position. */
export interface MarkerDrag {
  ref: MarkerRef;
  ms: number;
}

const sameRef = (a: MarkerRef | undefined, b: MarkerRef) =>
  !!a && a.kind === b.kind && (a.kind === 'hotcue' ? a.slot === (b as { slot: number }).slot : a.id === (b as { id: number }).id);

/** Beat ticks between t0 and t1 (ms). Thinned when they'd be closer than
 *  minPx; every 4th tick is drawn taller once there's room (no downbeat
 *  detection, so "every 4th" is from the first detected beat, not a bar). */
export function drawBeats(
  g: CanvasRenderingContext2D, beats: number[], t0: number, t1: number, xOf: (ms: number) => number,
  H: number, color: string, dpr: number, minPx = 5,
) {
  if (!beats.length) return;
  let lo = 0;
  let hi = beats.length;
  while (lo < hi) {
    const m = (lo + hi) >> 1;
    if (beats[m] < t0) lo = m + 1;
    else hi = m;
  }
  // Spacing check on a typical interval.
  const i2 = Math.min(beats.length - 1, lo + 1);
  const px = Math.abs(xOf(beats[i2]) - xOf(beats[Math.max(0, i2 - 1)]));
  let step = 1;
  while (px * step < minPx * dpr && step < 64) step *= 2;
  g.fillStyle = color;
  for (let i = lo - (lo % step); i < beats.length && beats[i] <= t1; i += step) {
    if (i < 0) continue;
    const x = Math.round(xOf(beats[i]));
    const tall = step === 1 && px >= 10 * dpr && i % 4 === 0;
    const h = tall ? H * 0.22 : H * 0.12;
    g.fillRect(x, 0, Math.max(1, dpr), h);
    g.fillRect(x, H - h, Math.max(1, dpr), h);
  }
}

/** Hotcue and goto markers: a line through the lane and a numbered flag
 *  on top. Gotos are dashed, gold when armed, grey when off, red when they
 *  can't fire. */
export function drawMarkers(
  g: CanvasRenderingContext2D, hotcues: DjHotcue[], gotos: DjGoto[], xOf: (ms: number) => number,
  W: number, H: number, dpr: number, opts: { flags?: boolean; drag?: MarkerDrag | null; gotoNames?: Map<number, string> } = {},
) {
  const flags = opts.flags ?? true;
  const fh = Math.round(14 * dpr);
  g.font = `700 ${Math.round(10 * dpr)}px system-ui, sans-serif`;
  g.textBaseline = 'middle';
  g.textAlign = 'center';
  for (const go of gotos) {
    const ref: MarkerRef = { kind: 'goto', id: go.id };
    const ms = sameRef(opts.drag?.ref, ref) ? opts.drag!.ms : go.positionMs;
    const x = Math.round(xOf(ms));
    if (x < -40 * dpr || x > W + 40 * dpr) continue;
    const c = go.problem ? '#f43f62' : go.enabled ? GOTO_COLOR : '#8a8f98';
    g.strokeStyle = c;
    g.lineWidth = Math.max(1, 1.5 * dpr);
    g.setLineDash([3 * dpr, 3 * dpr]);
    g.beginPath();
    g.moveTo(x + 0.5, flags ? fh : 0);
    g.lineTo(x + 0.5, H);
    g.stroke();
    g.setLineDash([]);
    if (flags) {
      const name = `${opts.gotoNames?.get(go.id) ?? 'G'}→${go.targetSlot}`;
      const w = g.measureText(name).width + 8 * dpr;
      g.fillStyle = go.enabled && !go.problem ? c : 'rgba(20,22,26,0.9)';
      g.strokeStyle = c;
      g.lineWidth = Math.max(1, dpr);
      g.fillRect(x - w / 2, 0, w, fh);
      g.strokeRect(x - w / 2 + 0.5, 0.5, w - 1, fh - 1);
      g.fillStyle = go.enabled && !go.problem ? '#111' : c;
      g.fillText(name, x, fh / 2 + 0.5);
    }
  }
  for (const h of hotcues) {
    const ref: MarkerRef = { kind: 'hotcue', slot: h.slot };
    const ms = sameRef(opts.drag?.ref, ref) ? opts.drag!.ms : h.positionMs;
    const x = Math.round(xOf(ms));
    if (x < -20 * dpr || x > W + 20 * dpr) continue;
    const c = hotcueColor(h);
    g.fillStyle = c;
    g.fillRect(x, flags ? fh : 0, Math.max(1, 2 * dpr), H);
    if (flags) {
      const w = fh;
      g.fillRect(x - w / 2 + dpr, 0, w, fh);
      g.fillStyle = '#111';
      g.fillText(String(h.slot), x + dpr, fh / 2 + 0.5);
    }
  }
}

/** The marker under css-pixel x (within 8px), gotos first only if closer. */
export function markerAt(
  x: number, hotcues: DjHotcue[], gotos: DjGoto[], xOfCss: (ms: number) => number, slop = 8,
): MarkerRef | null {
  let best: MarkerRef | null = null;
  let bd = slop + 1;
  for (const h of hotcues) {
    const d = Math.abs(xOfCss(h.positionMs) - x);
    if (d < bd) {
      bd = d;
      best = { kind: 'hotcue', slot: h.slot };
    }
  }
  for (const go of gotos) {
    const d = Math.abs(xOfCss(go.positionMs) - x);
    if (d < bd) {
      bd = d;
      best = { kind: 'goto', id: go.id };
    }
  }
  return best;
}

export const refMs = (r: MarkerRef, hotcues: DjHotcue[], gotos: DjGoto[]) =>
  r.kind === 'hotcue' ? (hotcues.find((h) => h.slot === r.slot)?.positionMs ?? 0) : (gotos.find((g) => g.id === r.id)?.positionMs ?? 0);
