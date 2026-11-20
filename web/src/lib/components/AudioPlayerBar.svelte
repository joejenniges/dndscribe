<script lang="ts" module>
  export interface PlayRequest {
    lineId: number;
    filenames: string[];
    speaker: string;
    color: string;
  }
</script>

<script lang="ts">
  import { onDestroy } from 'svelte';

  // Bottom-docked player for a transcript line's audio. A line can own several
  // clip files; they play back-to-back with a short gap, as one timeline.
  //
  // WHY the playback internals are imperative (plain vars + setInterval DOM
  // writes) rather than reactive state: this was lifted as-is from the old
  // live/session pages where it was battle-tested. Time display updates 5x/s
  // and multi-segment gap scheduling is timing-sensitive; routing that through
  // reactivity bought nothing.
  let {
    request,
    autoplay = $bindable(false),
    onended,
    onclose,
  }: {
    request: PlayRequest;
    autoplay?: boolean;
    // Fired when the last segment finishes naturally (not on stop/close).
    onended?: () => void;
    onclose?: () => void;
  } = $props();

  const GAP_MS = 400;

  let canvasEl: HTMLCanvasElement | undefined = $state();
  let waveWrap: HTMLDivElement | undefined = $state();
  let progressEl: HTMLDivElement | undefined = $state();
  let timeEl: HTMLSpanElement | undefined = $state();
  let playing = $state(false);

  let segments: { audio: HTMLAudioElement; duration: number; startOffset: number }[] = [];
  let totalDuration = 0;
  let currentIdx = 0;
  let isPlaying = false;
  let tick: ReturnType<typeof setInterval> | null = null;
  let reachedEnd = false;
  let loadToken = 0;

  function fmt(sec: number): string {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return m + ':' + String(s).padStart(2, '0');
  }

  function updateDisplay() {
    if (!segments.length || !timeEl || !progressEl) return;
    const seg = segments[currentIdx];
    if (!seg?.audio) return;
    const t = seg.startOffset + seg.audio.currentTime;
    timeEl.textContent = fmt(t) + ' / ' + fmt(totalDuration);
    progressEl.style.width = (totalDuration > 0 ? (t / totalDuration) * 100 : 0) + '%';
  }

  function setPlaying(v: boolean) {
    isPlaying = v;
    playing = v;
  }

  function teardown() {
    reachedEnd = false;
    setPlaying(false);
    if (tick) { clearInterval(tick); tick = null; }
    for (const seg of segments) { seg.audio.pause(); seg.audio.onended = null; seg.audio.ontimeupdate = null; seg.audio.src = ''; }
    segments = [];
    totalDuration = 0;
    currentIdx = 0;
  }

  async function load(fnames: string[]) {
    teardown();
    const token = ++loadToken;
    if (timeEl) timeEl.textContent = 'Loading…';
    if (progressEl) progressEl.style.width = '0%';

    const loaded: typeof segments = [];
    for (const fname of fnames) {
      const audio = new Audio('/recordings/raw/' + encodeURIComponent(fname));
      await new Promise<void>((resolve) => {
        audio.addEventListener('loadedmetadata', () => resolve(), { once: true });
        audio.addEventListener('error', () => resolve(), { once: true });
        audio.load();
      });
      if (audio.duration && isFinite(audio.duration)) {
        loaded.push({ audio, duration: audio.duration, startOffset: 0 });
      }
    }
    // A newer request arrived while we were loading; drop this one.
    if (token !== loadToken) { for (const s of loaded) s.audio.src = ''; return; }

    if (loaded.length === 0) {
      if (timeEl) timeEl.textContent = 'No audio';
      return;
    }

    let off = 0;
    for (let i = 0; i < loaded.length; i++) {
      loaded[i].startOffset = off;
      off += loaded[i].duration;
      if (i < loaded.length - 1) off += GAP_MS / 1000;
    }
    segments = loaded;
    totalDuration = off;
    if (timeEl) timeEl.textContent = '0:00 / ' + fmt(totalDuration);

    startPlayback();
    renderWaveform(fnames, token);
  }

  function startPlayback() {
    if (segments.length === 0) return;
    setPlaying(true);
    currentIdx = 0;
    playSegment(0);
    if (!tick) tick = setInterval(updateDisplay, 200);
  }

  function playSegment(idx: number, seekTime?: number) {
    if (idx >= segments.length || !isPlaying) { stopPlayback(); return; }
    currentIdx = idx;
    const seg = segments[idx];
    if (!seg.audio || !seg.duration) { playSegment(idx + 1); return; }

    seg.audio.onended = null;
    seg.audio.ontimeupdate = null;
    seg.audio.currentTime = seekTime ?? 0;
    seg.audio.play().catch(() => {});

    let advanced = false;
    function advance() {
      if (advanced) return;
      advanced = true;
      seg.audio.pause();
      seg.audio.ontimeupdate = null;
      seg.audio.onended = null;
      if (!isPlaying) return;
      if (idx + 1 < segments.length) {
        setTimeout(() => playSegment(idx + 1), GAP_MS);
      } else {
        reachedEnd = true;
        stopPlayback();
      }
    }
    seg.audio.onended = advance;
    // Some browsers fire 'ended' late on short clips; this catches the tail.
    seg.audio.ontimeupdate = () => {
      if (seg.audio.currentTime >= seg.audio.duration - 0.05) advance();
    };
  }

  function stopPlayback() {
    setPlaying(false);
    for (const seg of segments) seg.audio.pause();
    updateDisplay();
    if (reachedEnd) {
      reachedEnd = false;
      onended?.();
    }
  }

  export function togglePlayPause() {
    if (isPlaying) {
      stopPlayback();
    } else if (segments.length > 0) {
      setPlaying(true);
      playSegment(currentIdx);
      if (!tick) tick = setInterval(updateDisplay, 200);
    }
  }

  function seekTo(target: number) {
    for (let i = 0; i < segments.length; i++) {
      const seg = segments[i];
      if (target <= seg.startOffset + seg.duration || i === segments.length - 1) {
        for (const s of segments) { s.audio.pause(); s.audio.ontimeupdate = null; s.audio.onended = null; }
        const pos = Math.max(0, target - seg.startOffset);
        currentIdx = i;
        if (isPlaying) playSegment(i, pos);
        else seg.audio.currentTime = pos;
        updateDisplay();
        return;
      }
    }
  }

  function pctFromEvent(clientX: number): number {
    if (!waveWrap) return 0;
    const rect = waveWrap.getBoundingClientRect();
    return Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
  }

  function handleWaveClick(e: MouseEvent) {
    if (!totalDuration) return;
    seekTo(pctFromEvent(e.clientX) * totalDuration);
  }

  function handleWaveKey(e: KeyboardEvent) {
    if (!totalDuration || !segments.length) return;
    const seg = segments[currentIdx];
    const now = seg.startOffset + seg.audio.currentTime;
    if (e.key === 'ArrowRight') { e.preventDefault(); seekTo(Math.min(totalDuration, now + 2)); }
    if (e.key === 'ArrowLeft') { e.preventDefault(); seekTo(Math.max(0, now - 2)); }
  }

  async function renderWaveform(fnames: string[], token: number) {
    if (!canvasEl || !waveWrap) return;
    let ctxA: AudioContext;
    try { ctxA = new AudioContext(); } catch { return; }

    const parts: Float32Array[] = [];
    for (let i = 0; i < fnames.length; i++) {
      try {
        const resp = await fetch('/recordings/raw/' + encodeURIComponent(fnames[i]));
        const buf = await resp.arrayBuffer();
        if (buf.byteLength === 0) continue;
        const decoded = await ctxA.decodeAudioData(buf);
        parts.push(decoded.getChannelData(0));
        if (i < fnames.length - 1) parts.push(new Float32Array(Math.floor(decoded.sampleRate * GAP_MS / 1000)));
      } catch { /* skip undecodable clip */ }
    }
    ctxA.close();
    if (token !== loadToken || parts.length === 0 || !canvasEl || !waveWrap) return;

    let total = 0;
    for (const p of parts) total += p.length;
    const combined = new Float32Array(total);
    let off = 0;
    for (const p of parts) { combined.set(p, off); off += p.length; }

    const dpr = window.devicePixelRatio || 1;
    const w = waveWrap.clientWidth;
    const h = waveWrap.clientHeight;
    canvasEl.width = w * dpr;
    canvasEl.height = h * dpr;
    const ctx = canvasEl.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const bars = Math.floor(w / 3);
    const per = Math.max(1, Math.floor(combined.length / bars));
    const mid = h / 2;
    ctx.fillStyle = '#4a4f59';
    for (let b = 0; b < bars; b++) {
      let max = 0;
      for (let s = b * per; s < (b + 1) * per && s < combined.length; s++) {
        const a = Math.abs(combined[s]);
        if (a > max) max = a;
      }
      const bh = Math.max(1, max * mid * 0.9);
      ctx.fillRect(b * 3, mid - bh, 2, bh * 2);
    }
  }

  // (Re)load whenever a new request object arrives. Identity, not lineId, so
  // pressing play on the same line again restarts it.
  let loadedReq: PlayRequest | null = null;
  $effect(() => {
    const r = request;
    if (!timeEl || !waveWrap) return; // refs bind after first render
    if (r === loadedReq) return;
    loadedReq = r;
    load([...r.filenames]);
  });

  onDestroy(() => {
    loadToken++;
    teardown();
  });
</script>

<div class="player" role="region" aria-label="Audio player">
  <button class="icon-btn play" onclick={togglePlayPause} aria-label={playing ? 'Pause' : 'Play'} title="Play/Pause (Space)">
    {#if playing}&#10074;&#10074;{:else}&#9654;{/if}
  </button>
  <span class="speaker" style:color={request.color}>{request.speaker}</span>
  <span bind:this={timeEl} class="time">0:00 / 0:00</span>
  <div
    bind:this={waveWrap}
    class="wave"
    onclick={handleWaveClick}
    onkeydown={handleWaveKey}
    role="slider"
    tabindex="0"
    aria-label="Seek"
    aria-valuemin={0}
    aria-valuemax={100}
    aria-valuenow={0}
  >
    <canvas bind:this={canvasEl}></canvas>
    <div bind:this={progressEl} class="progress" style:background-color={request.color}></div>
  </div>
  <button
    class="btn btn-sm auto"
    class:active={autoplay}
    aria-pressed={autoplay}
    onclick={() => { autoplay = !autoplay; }}
    title="When a line finishes, play the next line that has audio"
  >Auto-next</button>
  <button class="icon-btn" onclick={() => onclose?.()} aria-label="Close player">&times;</button>
</div>

<style>
  .player {
    position: fixed; left: 0; right: 0; bottom: 0; z-index: 45;
    height: 64px; display: flex; align-items: center; gap: 12px; padding: 0 16px;
    background: var(--surface); border-top: 1px solid var(--border-strong);
    box-shadow: 0 -8px 24px rgba(0, 0, 0, 0.35);
  }
  .play { background: var(--surface-raised); color: var(--text); font-size: 15px; }
  .speaker { font-weight: 700; font-size: 14px; white-space: nowrap; max-width: 160px; overflow: hidden; text-overflow: ellipsis; }
  .time { color: var(--text-2); font-size: 13px; font-family: var(--font-mono); white-space: nowrap; min-width: 92px; }
  .wave { position: relative; flex: 1; height: 40px; cursor: pointer; overflow: hidden; border-radius: var(--radius-sm); }
  canvas { width: 100%; height: 100%; display: block; }
  .progress { position: absolute; inset: 0 auto 0 0; width: 0%; opacity: 0.22; pointer-events: none; }
  @media (max-width: 650px) {
    .player { gap: 8px; padding: 0 8px; }
    .speaker { display: none; }
    .time { min-width: 0; font-size: 12px; }
  }
</style>
