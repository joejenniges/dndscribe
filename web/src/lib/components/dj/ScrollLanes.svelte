<script lang="ts">
  // The performance waveform (v4): one lane per deck, scrolling under a
  // fixed playhead in the middle. Each lane follows its own deck's position
  // and draws its own beat grid; the two are never shown as aligned or
  // synced, since nothing here beatmatches.
  //
  // Dragging a lane scrubs that deck only: the playhead stays in the middle
  // and the waveform moves under it, like pushing a record. The decks are
  // independent, so the other lane never moves. The lane follows the
  // pointer locally and one seek is sent on release (each seek restarts the
  // server's decoder, so audio doesn't scrub live while dragging).
  //
  // Local to this device: zoom (seconds across the lane). Markers can be
  // dragged (snapped to beats unless Shift is held; saved once on release),
  // and right click or a long press opens the cue menu at that time.
  //
  // Positions are extrapolated locally from the server's snapshots (with a
  // short easing when a snapshot disagrees); beat grids and waveforms are
  // fetched once per track over HTTP and cached.
  import { onMount } from 'svelte';
  import { djCmd } from '$lib/dj/control';
  import { addToast } from '$lib/stores/toast';
  import { deckPosition, DECK_NAMES } from '$lib/dj/live';
  import { waveCache } from './wavecache';
  import { cueMenu, snapMs, loadAnalysis, gridVersion, analysisLabel, fmtCue, moveMarker, longPress, gotoName } from '$lib/dj/cues';
  import { drawBeats, drawMarkers, markerAt, refMs, type MarkerRef } from '$lib/dj/wavedraw';
  import type { DjAnalysis, DjState, DjWaveform } from '$lib/types';
  import { fmtMs } from './format';

  let { st, live = true, laneHeight = 52 }: { st: DjState | null; live?: boolean; laneHeight?: number } = $props();

  const ZOOMS = [2, 4, 8, 16, 32, 64];
  function readZoom() {
    try {
      const z = Number(localStorage.getItem('dj.laneZoom'));
      return ZOOMS.includes(z) ? z : 8;
    } catch {
      return 8;
    }
  }
  let zoom = $state(readZoom());
  $effect(() => {
    try {
      localStorage.setItem('dj.laneZoom', String(zoom));
    } catch {
      /* not remembered */
    }
  });
  /** Per lane: where a scrub drag has the deck (ms), until its seek lands. */
  let scrub = $state<[number | null, number | null]>([null, null]);

  type Lane = {
    wave: DjWaveform | null;
    waveFor: number;
    analysis: DjAnalysis | null;
    corr: number;
    lastShown: number;
    lastKey: string;
  };
  const lanes: [Lane, Lane] = [
    { wave: null, waveFor: 0, analysis: null, corr: 0, lastShown: 0, lastKey: '' },
    { wave: null, waveFor: 0, analysis: null, corr: 0, lastShown: 0, lastKey: '' },
  ];
  let analysisText = $state<[string, string]>(['', '']);
  let dirty = 0;

  const trackIds = $derived([st?.decks?.[0]?.track?.id ?? 0, st?.decks?.[1]?.track?.id ?? 0] as const);

  // Waveforms and grids per lane.
  for (const i of [0, 1] as const) {
    $effect(() => {
      const id = trackIds[i];
      const L = lanes[i];
      if (!id) {
        L.wave = null;
        L.waveFor = 0;
        dirty++;
        return;
      }
      if (id === L.waveFor && L.wave) return;
      let alive = true;
      waveCache.load(id).then((w) => {
        if (!alive) return;
        L.wave = w;
        L.waveFor = id;
        dirty++;
      }, () => {});
      return () => (alive = false);
    });
    $effect(() => {
      void $gridVersion;
      const id = trackIds[i];
      lanes[i].analysis = null;
      analysisText[i] = '';
      if (!id) return;
      let alive = true;
      loadAnalysis(id).then((a) => {
        if (!alive) return;
        lanes[i].analysis = a;
        analysisText[i] = analysisLabel(a);
        dirty++;
      }, () => {});
      return () => (alive = false);
    });
    // Ease out small disagreements between extrapolation and a snapshot.
    $effect(() => {
      const L = lanes[i];
      const key = `${st?.receivedAt}|${trackIds[i]}`;
      if (key === L.lastKey) return;
      const same = L.lastKey.endsWith(`|${trackIds[i]}`);
      L.lastKey = key;
      const diff = L.lastShown - deckPosition(st, i);
      L.corr = same && Math.abs(diff) < 1000 ? diff : 0;
    });
  }

  // ---- Interaction ----
  let canvases: [HTMLCanvasElement | undefined, HTMLCanvasElement | undefined] = $state([undefined, undefined]);
  let rootEl: HTMLElement | undefined = $state();
  type Drag =
    | { kind: 'scrub'; lane: 0 | 1; x0: number; pos0: number; moved: boolean }
    | { kind: 'marker'; lane: 0 | 1; ref: MarkerRef; x0: number; ms0: number; ms: number; moved: boolean };
  let drag = $state<Drag | null>(null);
  const shown = [0, 0];

  const pxPerMs = (c: HTMLCanvasElement) => c.clientWidth / (zoom * 1000);
  /** Track time under css x on lane i. */
  function timeAt(i: 0 | 1, x: number): number {
    const c = canvases[i]!;
    return shown[i] + (x - c.clientWidth / 2) / pxPerMs(c);
  }
  function xOfCss(i: 0 | 1, ms: number): number {
    const c = canvases[i]!;
    return c.clientWidth / 2 + (ms - shown[i]) * pxPerMs(c);
  }

  function hasTrack(i: 0 | 1) {
    return !!st?.decks?.[i]?.track;
  }

  let lpLane: 0 | 1 = 0;
  const lp = longPress((e) => openMenu(lpLane, e));

  function openMenu(i: 0 | 1, e: { clientX: number; clientY: number }) {
    const d = st?.decks?.[i];
    const c = canvases[i];
    if (!d?.track || !c) return;
    drag = null;
    const x = e.clientX - c.getBoundingClientRect().left;
    const m = markerAt(x, d.hotcues, d.gotos, (ms) => xOfCss(i, ms));
    const dur = d.track.durationMs;
    const at = Math.max(0, dur ? Math.min(dur, timeAt(i, x)) : timeAt(i, x));
    cueMenu.set({ deck: i, x: e.clientX, y: e.clientY, atMs: m ? refMs(m, d.hotcues, d.gotos) : at,
      hotcue: m?.kind === 'hotcue' ? m.slot : undefined, gotoId: m?.kind === 'goto' ? m.id : undefined });
  }

  function onDown(i: 0 | 1, e: PointerEvent) {
    if (e.button !== 0 || !hasTrack(i)) return;
    const c = e.currentTarget as HTMLElement;
    c.setPointerCapture(e.pointerId);
    lpLane = i;
    lp.down(e);
    const d = st!.decks[i];
    const x = e.clientX - c.getBoundingClientRect().left;
    const m = live ? markerAt(x, d.hotcues, d.gotos, (ms) => xOfCss(i, ms), e.pointerType === 'mouse' ? 6 : 12) : null;
    if (m) {
      const ms = refMs(m, d.hotcues, d.gotos);
      drag = { kind: 'marker', lane: i, ref: m, x0: e.clientX, ms0: ms, ms, moved: false };
    } else if (live && !d.loading && d.track!.durationMs > 0) {
      drag = { kind: 'scrub', lane: i, x0: e.clientX, pos0: shown[i], moved: false };
    }
  }
  function onMove(e: PointerEvent) {
    lp.move(e);
    const dr = drag;
    if (!dr) return;
    const dx = e.clientX - dr.x0;
    if (Math.abs(dx) > 3 && !dr.moved) {
      dr.moved = true;
      lp.cancel();
    }
    if (!dr.moved) return;
    const c = canvases[dr.lane]!;
    if (dr.kind === 'scrub') {
      // Dragging left pulls later audio under the playhead.
      const dur = st?.decks?.[dr.lane]?.track?.durationMs ?? 0;
      scrub[dr.lane] = Math.max(0, Math.min(dur, dr.pos0 - dx / pxPerMs(c)));
    } else {
      const dur = st?.decks?.[dr.lane]?.track?.durationMs ?? 0;
      let ms = dr.ms0 + dx / pxPerMs(c);
      ms = Math.max(0, dur ? Math.min(dur, ms) : ms);
      dr.ms = snapMs(ms, lanes[dr.lane].analysis, e.shiftKey);
    }
    dirty++;
  }
  function onUp() {
    lp.cancel();
    const dr = drag;
    drag = null;
    if (dr?.kind === 'marker' && dr.moved) {
      const d = st?.decks?.[dr.lane];
      if (d?.track) moveMarker(d.track.id, dr.ref, dr.ms, d.hotcues, d.gotos);
    } else if (dr?.kind === 'scrub' && dr.moved && scrub[dr.lane] !== null) {
      seek(dr.lane, scrub[dr.lane]!);
    }
    dirty++;
  }

  async function seek(i: 0 | 1, ms: number) {
    try {
      await djCmd('deck.seek', { deck: i, positionMs: Math.round(ms) });
    } catch (e) {
      addToast(`Couldn't seek deck ${DECK_NAMES[i]}: ${(e as Error).message}`, 'error');
    } finally {
      // Hand back to the server's position; lastShown is the scrubbed spot,
      // so any small difference eases out instead of jumping.
      if (drag?.lane !== i) scrub[i] = null;
      dirty++;
    }
  }
  function onCancel() {
    lp.cancel();
    if (drag?.kind === 'scrub') scrub[drag.lane] = null;
    drag = null;
    dirty++;
  }
  function onContext(i: 0 | 1, e: MouseEvent) {
    e.preventDefault();
    openMenu(i, e);
  }
  function onWheel(e: WheelEvent) {
    if (!e.ctrlKey && !e.metaKey) return;
    e.preventDefault();
    zoomBy(e.deltaY > 0 ? 1 : -1);
  }
  function zoomBy(dir: number) {
    const i = ZOOMS.indexOf(zoom);
    zoom = ZOOMS[Math.max(0, Math.min(ZOOMS.length - 1, i + dir))];
    dirty++;
  }

  // ---- Drawing ----
  type Colors = { a: string; aDim: string; b: string; bDim: string; line: string; head: string; beat: string; bg: string };
  let colors: Colors | null = null;
  function readColors(el: Element): Colors {
    const cs = getComputedStyle(el);
    const v = (n: string, d: string) => cs.getPropertyValue(n).trim() || d;
    return {
      a: v('--deck-a', '#f43f62'), aDim: v('--deck-a-dim', '#bd1c47'),
      b: v('--deck-b', '#3b9cff'), bDim: v('--deck-b-dim', '#1f63b8'),
      line: v('--border-color', '#32353b'), head: v('--text', '#e8eaf0'),
      beat: 'rgba(255,255,255,0.28)', bg: v('--surface-sunken', '#101215'),
    };
  }

  function fit(c: HTMLCanvasElement) {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(c.clientWidth * dpr));
    const h = Math.max(1, Math.round(c.clientHeight * dpr));
    if (c.width !== w || c.height !== h) {
      c.width = w;
      c.height = h;
    }
  }

  function envelope(g: CanvasRenderingContext2D, w: DjWaveform, idx: 0 | 1, t0: number, pxPerSec: number, W: number, mid: number, scale: number) {
    const bps = w.binsPerSecond;
    const bins = w.data.length / 2;
    const b0 = Math.max(0, Math.floor(t0 * bps) - 1);
    const b1 = Math.min(bins - 1, Math.ceil((t0 + W / pxPerSec) * bps) + 1);
    if (b1 < b0) return false;
    const xOf = (b: number) => ((b + 0.5) / bps - t0) * pxPerSec;
    // At wide zooms several bins share a pixel: step so the path stays light.
    const step = Math.max(1, Math.floor(bps / pxPerSec / 2));
    g.beginPath();
    g.moveTo(xOf(b0), mid);
    for (let b = b0; b <= b1; b += step) g.lineTo(xOf(b), mid - (w.data[b * 2 + idx] / 255) * mid * scale);
    g.lineTo(xOf(b1), mid);
    for (let b = b1; b >= b0; b -= step) g.lineTo(xOf(b), mid + (w.data[b * 2 + idx] / 255) * mid * scale);
    g.closePath();
    return true;
  }

  function drawLane(i: 0 | 1, pos: number) {
    const c = canvases[i];
    if (!c || !colors) return;
    fit(c);
    const g = c.getContext('2d')!;
    const { width: W, height: H } = c;
    const dpr = window.devicePixelRatio || 1;
    const mid = H / 2;
    g.clearRect(0, 0, W, H);
    const d = st?.decks?.[i];
    const L = lanes[i];
    const pxPerSec = W / zoom;
    const t0 = pos / 1000 - zoom / 2; // seconds at x=0; the playhead is the middle
    const xOf = (ms: number) => (ms / 1000 - t0) * pxPerSec;
    const head = Math.round(xOf(pos));
    g.fillStyle = colors.line;
    g.fillRect(0, Math.floor(mid), W, 1);
    if (d?.track && L.wave) {
      const dur = d.track.durationMs;
      // Outside the track: shade.
      const xs = xOf(0);
      const xe = dur ? xOf(dur) : W;
      g.fillStyle = 'rgba(0,0,0,0.35)';
      if (xs > 0) g.fillRect(0, 0, xs, H);
      if (xe < W) g.fillRect(xe, 0, W - xe, H);
      const [full, dim] = i === 0 ? [colors.a, colors.aDim] : [colors.b, colors.bDim];
      for (const [x0, x1, alpha] of [[0, head, 0.55], [head, W, 1]] as const) {
        if (x1 <= x0) continue;
        g.save();
        g.beginPath();
        g.rect(x0, 0, x1 - x0, H);
        g.clip();
        g.globalAlpha = alpha;
        if (envelope(g, L.wave, 0, t0, pxPerSec, W, mid, 0.92)) {
          g.fillStyle = dim;
          g.fill();
        }
        if (envelope(g, L.wave, 1, t0, pxPerSec, W, mid, 0.92)) {
          g.fillStyle = full;
          g.fill();
        }
        g.restore();
      }
      const a = L.analysis;
      if (a?.usable) drawBeats(g, a.beats, t0 * 1000, t0 * 1000 + zoom * 1000, xOf, H, colors.beat, dpr);
      const dr = drag?.kind === 'marker' && drag.lane === i && drag.moved ? { ref: drag.ref, ms: drag.ms } : null;
      const names = new Map(d.gotos.map((x) => [x.id, gotoName(d.gotos, x)]));
      drawMarkers(g, d.hotcues, d.gotos, xOf, W, H, dpr, { flags: true, drag: dr, gotoNames: names });
    }
    // Playhead: fixed in the middle unless panned.
    g.fillStyle = colors.head;
    g.fillRect(head - Math.round(dpr), 0, Math.max(2, Math.round(2 * dpr)), H);
  }

  let posText = $state(['', '']);
  onMount(() => {
    let raf = 0;
    let prev = Date.now();
    let lastKey = '';
    const frame = () => {
      raf = requestAnimationFrame(frame);
      if (!colors && rootEl) colors = readColors(rootEl);
      const now = Date.now();
      const dt = now - prev;
      prev = now;
      const running = [0, 1].some((i) => st?.decks?.[i]?.playing && st?.connected);
      for (const i of [0, 1] as const) {
        const L = lanes[i];
        L.corr *= Math.exp(-dt / 250);
        if (Math.abs(L.corr) < 1) L.corr = 0;
        const dur = st?.decks?.[i]?.track?.durationMs ?? 0;
        let p = Math.max(0, deckPosition(st, i, now) + L.corr);
        if (dur > 0) p = Math.min(p, dur);
        const sc = scrub[i];
        if (sc !== null) p = sc;
        L.lastShown = p;
        shown[i] = p;
      }
      const key = `${Math.round(shown[0])}|${Math.round(shown[1])}|${dirty}|${zoom}|${scrub[0]}|${scrub[1]}|${canvases[0]?.clientWidth}|${JSON.stringify(st?.decks?.map((d) => [d.hotcues, d.gotos]))}`;
      if (!running && key === lastKey) return;
      lastKey = key;
      drawLane(0, shown[0]);
      drawLane(1, shown[1]);
      for (const i of [0, 1] as const) {
        const t = st?.decks?.[i]?.track ? fmtMs(shown[i]) : '';
        if (t !== posText[i]) posText[i] = t;
      }
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  });
</script>

<div class="lanes card" bind:this={rootEl} style:--lane-h="{laneHeight}px">
  {#each [0, 1] as const as i (i)}
    {@const d = st?.decks?.[i]}
    <div class="lane deck-{i}">
      <div class="meta">
        <span class="tag" aria-hidden="true">{DECK_NAMES[i]}</span>
        {#if d?.track}
          <span class="info" title={analysisText[i]}>{analysisText[i]}</span>
          <span class="time">{posText[i]}<span class="subtle"> / {fmtMs(d.track.durationMs)}</span></span>
        {:else}
          <span class="info subtle">Deck {DECK_NAMES[i]} empty</span>
        {/if}
      </div>
      <!-- The cue actions here are also in the deck's buttons and the cue
           editor (keyboard), so the canvas itself is pointer-only. -->
      <canvas bind:this={canvases[i]} class:grab={!!d?.track} aria-hidden="true"
        onpointerdown={(e) => onDown(i, e)} onpointermove={onMove} onpointerup={onUp} onpointercancel={onCancel}
        oncontextmenu={(e) => onContext(i, e)} onwheel={onWheel}></canvas>
      {#if drag?.kind === 'marker' && drag.lane === i && drag.moved}
        <span class="tip" style:left="{xOfCss(i, drag.ms)}px">{fmtCue(drag.ms)}</span>
      {/if}
    </div>
  {/each}
  <div class="tools" role="group" aria-label="Waveform zoom">
    <button class="btn btn-sm icon" onclick={() => zoomBy(-1)} disabled={zoom === ZOOMS[0]} aria-label="Zoom in" title="Zoom in (Ctrl + wheel)">+</button>
    <span class="zoom tiny">{zoom}s</span>
    <button class="btn btn-sm icon" onclick={() => zoomBy(1)} disabled={zoom === ZOOMS[ZOOMS.length - 1]} aria-label="Zoom out" title="Zoom out (Ctrl + wheel)">−</button>
  </div>
</div>

<style>
  .lanes { position: relative; padding: 6px 10px; display: flex; flex-direction: column; gap: 4px; min-width: 0; }
  .lane { position: relative; display: grid; grid-template-columns: 1fr; min-width: 0; }
  .meta {
    position: absolute; left: 6px; top: 4px; right: 150px; z-index: 2; pointer-events: none;
    display: flex; align-items: center; gap: 8px; min-width: 0; font-size: 12px;
  }
  .tag {
    flex-shrink: 0; width: 22px; height: 22px; border-radius: 5px; display: grid; place-items: center;
    background: var(--deck-a); color: #111; font-weight: 800; font-size: 12px;
  }
  .deck-1 .tag { background: var(--deck-b); }
  .info { color: var(--text-2); font-weight: 600; text-shadow: 0 0 4px #000; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-top: 14px; }
  .time { margin-top: 14px; font-variant-numeric: tabular-nums; font-weight: 700; text-shadow: 0 0 4px #000; white-space: nowrap; }
  canvas {
    display: block; width: 100%; height: var(--lane-h); border-radius: var(--radius-sm);
    background: var(--surface-sunken); touch-action: none; user-select: none;
  }
  canvas.grab { cursor: grab; }
  canvas.grab:active { cursor: grabbing; }
  .tip {
    position: absolute; top: -2px; transform: translate(-50%, -100%); pointer-events: none; white-space: nowrap; z-index: 5;
    background: var(--surface-raised); border: 1px solid var(--border-strong); border-radius: 4px; padding: 1px 6px;
    font-size: 11px; font-variant-numeric: tabular-nums;
  }
  .tools { position: absolute; right: 14px; top: 8px; display: flex; align-items: center; gap: 4px; z-index: 3; }
  .tools .icon { width: 28px; padding: 0; justify-content: center; }
  .zoom { min-width: 28px; text-align: center; color: var(--text-2); font-variant-numeric: tabular-nums; }
</style>
