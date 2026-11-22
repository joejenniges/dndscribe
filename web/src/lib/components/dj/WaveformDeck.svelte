<script lang="ts">
  // Waveform of the playing track: the whole song on top (played part in the
  // accent colour), and below it a zoomed view that scrolls past a fixed
  // playhead as the song plays.
  //
  // Waveforms are computed and cached by the server (dj_waveforms); this
  // component only draws them. A small in-memory cache here means switching
  // DJ sub-views or a track change back and forth doesn't refetch, and the
  // next track's waveform is fetched ahead so a track change shows instantly.
  //
  // Seeking: click or drag along the overview to jump there; drag the lower
  // view sideways to scrub (like pulling tape). Either way the position
  // follows the pointer locally and one seek is sent on release, since each
  // seek restarts the server's decoder.
  import { onMount } from 'svelte';
  import { dj } from '$lib/api';
  import { djState } from '$lib/stores/dj';
  import { addToast } from '$lib/stores/toast';
  import type { DjState, DjWaveform } from '$lib/types';
  import { fmtMs } from './format';

  let { st }: { st: DjState | null } = $props();

  /** Seconds of audio across the scrolling view. */
  const WINDOW_SEC = 8;

  const cache = new Map<number, Promise<DjWaveform>>();
  const CACHE_MAX = 12;
  function load(id: number): Promise<DjWaveform> {
    let p = cache.get(id);
    if (!p) {
      p = dj.waveform(id);
      p.catch(() => cache.delete(id)); // let a failed one be retried
      cache.set(id, p);
      while (cache.size > CACHE_MAX) cache.delete(cache.keys().next().value!);
    }
    return p;
  }

  const trackId = $derived(st?.playing ? (st.track?.id ?? 0) : 0);
  const nextId = $derived(st?.upNext?.[0]?.id ?? 0);
  const running = $derived(!!st?.playing && !st.paused && st.connected);
  const durationMs = $derived(st?.track?.durationMs ?? 0);

  let wave = $state<DjWaveform | null>(null);
  let waveFor = 0;
  let status = $state<'idle' | 'loading' | 'ready' | 'error'>('idle');

  $effect(() => {
    const id = trackId;
    if (!id) {
      wave = null;
      waveFor = 0;
      status = 'idle';
      return;
    }
    if (id === waveFor && wave) return;
    let live = true;
    status = 'loading';
    load(id).then(
      (w) => {
        if (!live) return;
        wave = w;
        waveFor = id;
        status = 'ready';
        overviewDirty = true;
      },
      () => {
        if (live) status = 'error';
      },
    );
    return () => {
      live = false;
    };
  });

  // Fetch the next track's waveform ahead of time. Errors are ignored here;
  // the real request will report them.
  $effect(() => {
    if (nextId && nextId !== trackId) load(nextId).catch(() => {});
  });

  // ---- Position ----
  // The server sends position on changes and every few seconds while
  // playing; in between it's extrapolated from the local clock. When a new
  // update disagrees slightly with the extrapolation, the difference is eased
  // out over a fraction of a second instead of jumping, so the scroll stays
  // smooth.
  let corr = 0;
  let lastShown = 0;
  let lastKey = '';
  function predicted(now: number): number {
    if (!st) return 0;
    const p = st.positionMs + (running ? now - (st.receivedAt ?? now) : 0);
    return durationMs > 0 ? Math.min(p, durationMs) : p;
  }
  $effect(() => {
    // Re-anchor whenever a new state arrives.
    const key = `${st?.receivedAt}`;
    if (key === lastKey) return;
    lastKey = key;
    const now = Date.now();
    const diff = lastShown - predicted(now);
    corr = Math.abs(diff) < 1000 ? diff : 0;
  });

  // ---- Seeking ----
  const seekable = $derived(!!st?.playing && trackId > 0 && durationMs > 0);
  /** Position shown while dragging, and until the seek's state comes back. */
  let scrubPos: number | null = null;
  let drag: { kind: 'overview' | 'scroll'; startX: number; startPos: number; moved: boolean } | null = null;
  let seeking = $state(false);

  const clampPos = (ms: number) => Math.max(0, Math.min(durationMs, ms));
  /** Where the song is now, independent of when the last frame was drawn. */
  const currentPos = () => scrubPos ?? clampPos(predicted(Date.now()) + corr);

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
    const d = drag;
    drag = null;
    // A tap on the lower view (no drag) does nothing; on the overview it seeks.
    if (scrubPos === null || (d.kind === 'scroll' && !d.moved)) {
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
      // Take the state from the response: it's guaranteed to be after the
      // seek, unlike a periodic update that might cross it in flight.
      const s = await dj.seek(ms);
      djState.set({ ...s, receivedAt: Date.now() });
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
  let overviewEl: HTMLCanvasElement | undefined = $state();
  let scrollEl: HTMLCanvasElement | undefined = $state();
  let overviewDirty = true;
  let played: HTMLCanvasElement | null = null; // overview pre-rendered in the played colour
  let unplayed: HTMLCanvasElement | null = null;
  let posText = $state('0:00');
  let posSec = $state(0); // for the slider's aria-valuenow

  type Colors = { accent: string; accentDim: string; muted: string; mutedDim: string; line: string; head: string };
  let colors: Colors | null = null;
  function readColors(el: Element): Colors {
    const cs = getComputedStyle(el);
    const v = (n: string, d: string) => cs.getPropertyValue(n).trim() || d;
    return {
      accent: v('--accent', '#f43f62'),
      accentDim: v('--accent-strong', '#bd1c47'),
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

  /** Renders the whole-track waveform in one colour into an offscreen canvas. */
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
      played = renderOverview(wave, W, H, colors.accentDim, colors.accent);
      unplayed = renderOverview(wave, W, H, colors.mutedDim, colors.muted);
      overviewDirty = false;
    }
    const px = Math.round(Math.min(1, posMs / durationMs) * W);
    g.drawImage(unplayed, 0, 0);
    if (px > 0) g.drawImage(played, 0, 0, px, H, 0, 0, px, H);
    g.fillStyle = colors.head;
    g.fillRect(Math.min(px, W - 2), 0, 2, H);
  }

  /** One half (top) of the envelope as a closed, mirrored polygon. */
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
      // Left of the playhead has been heard: accent, like the overview and
      // the progress bar; what's coming is grey.
      for (const [x0, x1, pk, rm] of [
        [0, head, colors.accentDim, colors.accent],
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
      // Second ticks along the bottom.
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
      if (!colors && overviewEl) colors = readColors(overviewEl);
      const now = Date.now();
      const dt = now - prev;
      prev = now;
      corr *= Math.exp(-dt / 250);
      if (Math.abs(corr) < 1) corr = 0;
      let pos = Math.max(0, predicted(now) + corr);
      if (durationMs > 0) pos = Math.min(pos, durationMs);
      if (scrubPos !== null) pos = scrubPos;
      else lastShown = pos;
      // While nothing moves, don't redraw every frame (the tablet's battery).
      const sizeKey = `${overviewEl?.clientWidth}x${scrollEl?.clientWidth}`;
      const key = `${Math.round(pos)}|${status}|${trackId}|${sizeKey}|${overviewDirty}`;
      if (!running && key === lastDrawKey) return;
      lastDrawKey = key;
      drawOverview(pos);
      drawScroll(pos);
      const t = fmtMs(pos);
      if (t !== posText) posText = t;
      const sec = Math.round(pos / 1000);
      if (sec !== posSec) posSec = sec;
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  });
</script>

<section class="deck card" aria-label="Waveform of the playing track">
  <div class="deck-head">
    <div class="title">
      {#if st?.playing && st.track}
        <strong>{st.track.title}</strong>
        <span class="tiny subtle">{st.programName}{st.paused ? ' · Paused' : ''}</span>
      {:else}
        <strong class="subtle">Nothing playing</strong>
      {/if}
    </div>
    <span class="tiny subtle time">
      {#if st?.playing && durationMs}{posText} / {fmtMs(durationMs)}{/if}
    </span>
  </div>
  <div class="wrap overview-wrap" class:seekable role="slider" tabindex={seekable ? 0 : -1}
    aria-label="Song position. Click or drag to jump; arrow keys skip 5 seconds, with Shift 30."
    aria-valuemin={0} aria-valuemax={Math.round(durationMs / 1000)} aria-valuenow={posSec}
    aria-valuetext={posText} aria-disabled={!seekable}
    onpointerdown={(e) => onDown('overview', e)} onpointermove={onMove} onpointerup={onUp} onpointercancel={onCancel}
    onkeydown={onKey}>
    <canvas bind:this={overviewEl} class="overview" aria-hidden="true"></canvas>
  </div>
  <div class="wrap scroll-wrap" class:seekable class:dragging={seeking}
    onpointerdown={(e) => onDown('scroll', e)} onpointermove={onMove} onpointerup={onUp} onpointercancel={onCancel}
    role="presentation" title={seekable ? 'Drag sideways to scrub' : undefined}>
    <canvas bind:this={scrollEl} class="scroll" aria-hidden="true"></canvas>
    {#if st?.playing && trackId === 0}
      <span class="overlay tiny subtle">No waveform for the test tone</span>
    {:else if status === 'loading'}
      <span class="overlay tiny subtle">Reading waveform…</span>
    {:else if status === 'error'}
      <span class="overlay tiny subtle">Couldn't read this track's waveform</span>
    {/if}
  </div>
</section>

<style>
  .deck { padding: 14px 16px 12px; display: flex; flex-direction: column; gap: 10px; }
  .deck-head { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; min-width: 0; }
  .title { display: flex; align-items: baseline; gap: 10px; min-width: 0; }
  .title strong, .title span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .time { font-variant-numeric: tabular-nums; white-space: nowrap; }
  .wrap { position: relative; }
  canvas { display: block; width: 100%; }
  .overview { height: 40px; }
  .scroll { height: 96px; background: var(--surface-sunken); border-radius: var(--radius-sm); }
  .seekable { touch-action: none; user-select: none; }
  .overview-wrap.seekable { cursor: pointer; }
  .overview-wrap:focus-visible { outline: none; box-shadow: var(--focus-ring); border-radius: var(--radius-sm); }
  .scroll-wrap.seekable { cursor: grab; }
  .scroll-wrap.seekable:active { cursor: grabbing; }
  .overlay { position: absolute; inset: 0; display: grid; place-items: center; pointer-events: none; }

  @media (max-width: 760px) {
    .scroll { height: 72px; }
  }
</style>
