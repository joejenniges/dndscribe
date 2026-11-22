<script lang="ts">
  // Waveform of one deck's track: the whole song (played part in the deck's
  // colour) and, unless `overviewOnly`, a zoomed view that scrolls past a
  // fixed playhead as the song plays.
  //
  // Waveforms are computed and cached by the server (dj_waveforms); this
  // component only draws them. A small in-memory cache shared by both decks
  // means a track moving between decks or coming back doesn't refetch, and
  // the next queued track's waveform is fetched ahead.
  //
  // Seeking: click or drag along the overview to jump; drag the lower view
  // sideways to scrub. The position follows the pointer locally and one seek
  // is sent on release, since each seek restarts the server's decoder.
  import { onMount } from 'svelte';
  import { dj } from '$lib/api';
  import { djCmd } from '$lib/dj/control';
  import { deckPosition, DECK_NAMES } from '$lib/dj/live';
  import { addToast } from '$lib/stores/toast';
  import type { DjState, DjWaveform } from '$lib/types';
  import { fmtMs } from './format';
  import { waveCache } from './wavecache';

  let {
    st,
    deck,
    overviewOnly = false,
    tall = false,
    live = true,
    bare = false,
    lane = false,
  }: {
    st: DjState | null;
    deck: 0 | 1;
    overviewOnly?: boolean;
    tall?: boolean;
    live?: boolean;
    /** No deck badge or time (the containing deck card shows them). */
    bare?: boolean;
    /** One tall whole-track lane (Studio): the overview only, bigger. */
    lane?: boolean;
  } = $props();

  /** Seconds of audio across the scrolling view. */
  const WINDOW_SEC = 8;

  const d = $derived(st?.decks?.[deck] ?? null);
  const trackId = $derived(d?.track?.id ?? 0);
  const nextId = $derived(st?.queue?.[0]?.id ?? 0);
  const running = $derived(!!d?.playing && !!st?.connected);
  const durationMs = $derived(d?.track?.durationMs ?? 0);

  let wave = $state<DjWaveform | null>(null);
  let waveFor = 0;
  let status = $state<'idle' | 'loading' | 'ready' | 'error'>('idle');

  $effect(() => {
    const id = trackId;
    if (!id) {
      wave = null;
      waveFor = 0;
      status = 'idle';
      overviewDirty = true;
      return;
    }
    if (id === waveFor && wave) return;
    let alive = true;
    status = 'loading';
    waveCache.load(id).then(
      (w) => {
        if (!alive) return;
        wave = w;
        waveFor = id;
        status = 'ready';
        overviewDirty = true;
      },
      () => {
        if (alive) status = 'error';
      },
    );
    return () => {
      alive = false;
    };
  });

  // Fetch the next queued track's waveform ahead (it lands on a deck soon).
  $effect(() => {
    if (nextId && nextId !== trackId) waveCache.load(nextId).catch(() => {});
  });

  // ---- Position ----
  // Extrapolated from the last snapshot. When a new snapshot disagrees
  // slightly with the extrapolation, the difference is eased out over a
  // fraction of a second instead of jumping, so the scroll stays smooth.
  let corr = 0;
  let lastShown = 0;
  let lastKey = '';
  $effect(() => {
    const key = `${st?.receivedAt}|${trackId}`;
    if (key === lastKey) return;
    const sameTrack = lastKey.endsWith(`|${trackId}`);
    lastKey = key;
    const diff = lastShown - deckPosition(st, deck);
    corr = sameTrack && Math.abs(diff) < 1000 ? diff : 0;
  });

  // ---- Seeking ----
  const seekable = $derived(live && !!d?.track && trackId > 0 && durationMs > 0 && !d.loading);
  let scrubPos: number | null = null;
  let drag: { kind: 'overview' | 'scroll'; startX: number; startPos: number; moved: boolean } | null = null;
  let seeking = $state(false);

  const clampPos = (ms: number) => Math.max(0, Math.min(durationMs, ms));
  const currentPos = () => scrubPos ?? clampPos(deckPosition(st, deck) + corr);

  function overviewPos(e: PointerEvent): number {
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
    return clampPos(((e.clientX - r.left) / r.width) * durationMs);
  }

  function onDown(kind: 'overview' | 'scroll', e: PointerEvent) {
    if (!seekable || e.button !== 0) return;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    drag = { kind, startX: e.clientX, startPos: currentPos(), moved: false };
    if (kind === 'overview') scrubPos = overviewPos(e);
  }

  function onMove(e: PointerEvent) {
    if (!drag) return;
    const dx = e.clientX - drag.startX;
    if (Math.abs(dx) > 3) drag.moved = true;
    if (drag.kind === 'overview') {
      scrubPos = overviewPos(e);
    } else if (drag.moved) {
      // Dragging left pulls later audio under the playhead.
      const pxPerSec = (e.currentTarget as HTMLElement).clientWidth / WINDOW_SEC;
      scrubPos = clampPos(drag.startPos - (dx / pxPerSec) * 1000);
    }
  }

  function onUp() {
    if (!drag) return;
    const dr = drag;
    drag = null;
    // A tap on the lower view (no drag) does nothing; on the overview it seeks.
    if (scrubPos === null || (dr.kind === 'scroll' && !dr.moved)) {
      scrubPos = null;
      return;
    }
    commit(scrubPos);
  }

  function onCancel() {
    drag = null;
    scrubPos = null;
  }

  async function commit(ms: number) {
    scrubPos = ms;
    seeking = true;
    try {
      await djCmd('deck.seek', { deck, positionMs: Math.round(ms) });
      corr = 0;
    } catch (err) {
      addToast(`Couldn't seek: ${(err as Error).message}`, 'error');
    } finally {
      if (!drag) scrubPos = null;
      seeking = false;
    }
  }

  function onKey(e: KeyboardEvent) {
    if (!seekable) return;
    const step = e.shiftKey ? 30_000 : 5_000;
    const from = currentPos();
    if (e.key === 'ArrowLeft') commit(clampPos(from - step));
    else if (e.key === 'ArrowRight') commit(clampPos(from + step));
    else if (e.key === 'Home') commit(0);
    else return;
    e.preventDefault();
  }

  // ---- Drawing ----
  let rootEl: HTMLElement | undefined = $state();
  let overviewEl: HTMLCanvasElement | undefined = $state();
  let scrollEl: HTMLCanvasElement | undefined = $state();
  let overviewDirty = true;
  let played: HTMLCanvasElement | null = null;
  let unplayed: HTMLCanvasElement | null = null;
  let posText = $state('0:00');
  let posSec = $state(0);

  type Colors = { deck: string; deckDim: string; muted: string; mutedDim: string; line: string; head: string };
  let colors: Colors | null = null;
  function readColors(el: Element): Colors {
    const cs = getComputedStyle(el);
    const v = (n: string, dflt: string) => cs.getPropertyValue(n).trim() || dflt;
    return {
      deck: v('--deck-color', '#f43f62'),
      deckDim: v('--deck-dim', '#bd1c47'),
      muted: v('--text-3', '#7c818b'),
      mutedDim: v('--border-strong', '#3d4148'),
      line: v('--border-color', '#32353b'),
      head: v('--text', '#e8eaf0'),
    };
  }

  function fit(c: HTMLCanvasElement): boolean {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(c.clientWidth * dpr));
    const h = Math.max(1, Math.round(c.clientHeight * dpr));
    if (c.width !== w || c.height !== h) {
      c.width = w;
      c.height = h;
      return true;
    }
    return false;
  }

  function renderOverview(w: DjWaveform, width: number, height: number, peakColor: string, rmsColor: string) {
    const c = document.createElement('canvas');
    c.width = width;
    c.height = height;
    const g = c.getContext('2d')!;
    const bins = w.data.length / 2;
    const mid = height / 2;
    for (let x = 0; x < width; x++) {
      const b0 = Math.floor((x / width) * bins);
      const b1 = Math.max(b0 + 1, Math.floor(((x + 1) / width) * bins));
      let pk = 0;
      let rm = 0;
      for (let b = b0; b < b1 && b < bins; b++) {
        pk = Math.max(pk, w.data[b * 2]);
        rm = Math.max(rm, w.data[b * 2 + 1]);
      }
      const ph = Math.max(1, (pk / 255) * mid);
      const rh = (rm / 255) * mid;
      g.fillStyle = peakColor;
      g.fillRect(x, mid - ph, 1, ph * 2);
      g.fillStyle = rmsColor;
      g.fillRect(x, mid - rh, 1, rh * 2);
    }
    return c;
  }

  function drawOverview(posMs: number) {
    const c = overviewEl;
    if (!c || !colors) return;
    const resized = fit(c);
    const g = c.getContext('2d')!;
    const { width: W, height: H } = c;
    g.clearRect(0, 0, W, H);
    if (!wave || !durationMs) {
      g.fillStyle = colors.line;
      g.fillRect(0, Math.floor(H / 2), W, 1);
      return;
    }
    if (overviewDirty || resized || !played || !unplayed) {
      played = renderOverview(wave, W, H, colors.deckDim, colors.deck);
      unplayed = renderOverview(wave, W, H, colors.mutedDim, colors.muted);
      overviewDirty = false;
    }
    const px = Math.round(Math.min(1, posMs / durationMs) * W);
    g.drawImage(unplayed, 0, 0);
    if (px > 0) g.drawImage(played, 0, 0, px, H, 0, 0, px, H);
    g.fillStyle = colors.head;
    g.fillRect(Math.min(px, W - 2), 0, 2, H);
  }

  function envelope(g: CanvasRenderingContext2D, w: DjWaveform, idx: 0 | 1, startSec: number, pxPerSec: number, W: number, mid: number) {
    const bps = w.binsPerSecond;
    const bins = w.data.length / 2;
    const b0 = Math.max(0, Math.floor(startSec * bps) - 1);
    const b1 = Math.min(bins - 1, Math.ceil((startSec + W / pxPerSec) * bps) + 1);
    if (b1 < b0) return false;
    const xOf = (b: number) => ((b + 0.5) / bps - startSec) * pxPerSec;
    g.beginPath();
    g.moveTo(xOf(b0), mid);
    for (let b = b0; b <= b1; b++) g.lineTo(xOf(b), mid - (w.data[b * 2 + idx] / 255) * mid);
    g.lineTo(xOf(b1), mid);
    for (let b = b1; b >= b0; b--) g.lineTo(xOf(b), mid + (w.data[b * 2 + idx] / 255) * mid);
    g.closePath();
    return true;
  }

  function drawScroll(posMs: number) {
    const c = scrollEl;
    if (!c || !colors) return;
    fit(c);
    const g = c.getContext('2d')!;
    const { width: W, height: H } = c;
    const mid = H / 2;
    const head = Math.round(W / 2);
    g.clearRect(0, 0, W, H);
    g.fillStyle = colors.line;
    g.fillRect(0, Math.floor(mid), W, 1);
    if (wave) {
      const pxPerSec = W / WINDOW_SEC;
      const startSec = posMs / 1000 - WINDOW_SEC / 2;
      for (const [x0, x1, pk, rm] of [
        [0, head, colors.deckDim, colors.deck],
        [head, W, colors.mutedDim, colors.muted],
      ] as const) {
        g.save();
        g.beginPath();
        g.rect(x0, 0, x1 - x0, H);
        g.clip();
        if (envelope(g, wave, 0, startSec, pxPerSec, W, mid)) {
          g.fillStyle = pk;
          g.fill();
        }
        if (envelope(g, wave, 1, startSec, pxPerSec, W, mid)) {
          g.fillStyle = rm;
          g.fill();
        }
        g.restore();
      }
      g.fillStyle = colors.line;
      const dpr = window.devicePixelRatio || 1;
      for (let s = Math.ceil(startSec); s <= startSec + WINDOW_SEC; s++) {
        if (s < 0) continue;
        const x = Math.round((s - startSec) * pxPerSec);
        g.fillRect(x, H - 6 * dpr, 1, 6 * dpr);
      }
    }
    g.fillStyle = colors.head;
    g.fillRect(head - 1, 0, 2, H);
  }

  onMount(() => {
    let raf = 0;
    let prev = Date.now();
    let lastDrawKey = '';
    const frame = () => {
      raf = requestAnimationFrame(frame);
      if (!colors && rootEl) colors = readColors(rootEl);
      const now = Date.now();
      const dt = now - prev;
      prev = now;
      corr *= Math.exp(-dt / 250);
      if (Math.abs(corr) < 1) corr = 0;
      let pos = Math.max(0, deckPosition(st, deck, now) + corr);
      if (durationMs > 0) pos = Math.min(pos, durationMs);
      if (scrubPos !== null) pos = scrubPos;
      else lastShown = pos;
      // While nothing moves, don't redraw every frame (the tablet's battery).
      const sizeKey = `${overviewEl?.clientWidth}x${overviewEl?.clientHeight}|${scrollEl?.clientWidth}`;
      const key = `${Math.round(pos)}|${status}|${trackId}|${sizeKey}|${overviewDirty}`;
      if (!running && key === lastDrawKey) return;
      lastDrawKey = key;
      drawOverview(pos);
      if (!overviewOnly) drawScroll(pos);
      const t = fmtMs(pos);
      if (t !== posText) posText = t;
      const sec = Math.round(pos / 1000);
      if (sec !== posSec) posSec = sec;
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  });
</script>

<div class="dw deck-{deck}" class:tall class:lane bind:this={rootEl}>
  <div class="row">
    {#if !bare}<span class="tag" aria-hidden="true">{DECK_NAMES[deck]}</span>{/if}
    <div class="waves">
      {#if !overviewOnly}
        <div class="wrap scroll-wrap" class:seekable class:dragging={seeking}
          onpointerdown={(e) => onDown('scroll', e)} onpointermove={onMove} onpointerup={onUp} onpointercancel={onCancel}
          role="presentation" title={seekable ? 'Drag sideways to scrub' : undefined}>
          <canvas bind:this={scrollEl} class="scroll" aria-hidden="true"></canvas>
        </div>
      {/if}
      <div class="wrap overview-wrap" class:seekable role="slider" tabindex={seekable ? 0 : -1}
        aria-label="Deck {DECK_NAMES[deck]} position. Click or drag to jump; arrow keys skip 5 seconds, with Shift 30."
        aria-valuemin={0} aria-valuemax={Math.round(durationMs / 1000)} aria-valuenow={posSec}
        aria-valuetext={posText} aria-disabled={!seekable}
        onpointerdown={(e) => onDown('overview', e)} onpointermove={onMove} onpointerup={onUp} onpointercancel={onCancel}
        onkeydown={onKey}>
        <canvas bind:this={overviewEl} class="overview" aria-hidden="true"></canvas>
        {#if !d?.track}
          <span class="overlay tiny subtle">Deck {DECK_NAMES[deck]} empty</span>
        {:else if d.loading || status === 'loading'}
          <span class="overlay tiny subtle">Loading…</span>
        {:else if status === 'error'}
          <span class="overlay tiny subtle">No waveform for this track</span>
        {/if}
      </div>
    </div>
    {#if !bare}<span class="time tiny">{#if d?.track}{posText}<span class="subtle">&nbsp;/ {fmtMs(durationMs)}</span>{/if}</span>{/if}
  </div>
</div>

<style>
  .dw { --deck-color: var(--deck-a); --deck-dim: var(--deck-a-dim); min-width: 0; }
  .dw.deck-1 { --deck-color: var(--deck-b); --deck-dim: var(--deck-b-dim); }
  .row { display: flex; align-items: center; gap: 10px; min-width: 0; }
  .tag {
    flex-shrink: 0; width: 26px; height: 26px; border-radius: 6px; display: grid; place-items: center;
    background: var(--deck-color); color: #111; font-weight: 800; font-size: 13px;
  }
  .waves { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 4px; }
  .time { flex-shrink: 0; min-width: 78px; text-align: right; font-variant-numeric: tabular-nums; }
  .wrap { position: relative; }
  canvas { display: block; width: 100%; }
  .overview { height: 30px; }
  .tall .overview { height: 46px; }
  .lane .overview { height: 40px; }
  .lane .overview-wrap { background: var(--surface-sunken); border-radius: var(--radius-sm); }
  .lane .tag { width: 26px; height: 26px; font-size: 13px; }
  .lane .time { font-size: 13px; min-width: 96px; }
  .scroll { height: 64px; background: var(--surface-sunken); border-radius: var(--radius-sm); }
  .seekable { touch-action: none; user-select: none; }
  .overview-wrap.seekable { cursor: pointer; }
  .overview-wrap:focus-visible { outline: none; box-shadow: var(--focus-ring); border-radius: var(--radius-sm); }
  .scroll-wrap.seekable { cursor: grab; }
  .scroll-wrap.seekable:active { cursor: grabbing; }
  .overlay { position: absolute; inset: 0; display: grid; place-items: center; pointer-events: none; }
</style>
