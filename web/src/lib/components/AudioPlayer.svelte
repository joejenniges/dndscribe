<script lang="ts">
  import { onMount, onDestroy } from 'svelte';

  let {
    filenames,
    speaker,
    speakerColor,
    open = true,
    onclose,
  }: {
    filenames: string[];
    speaker: string;
    speakerColor: string;
    open?: boolean;
    onclose?: () => void;
  } = $props();

  let canvasEl: HTMLCanvasElement | undefined = $state(undefined);
  let waveformWrap: HTMLDivElement | undefined = $state(undefined);
  let progressEl: HTMLDivElement | undefined = $state(undefined);
  let scrubEl: HTMLInputElement | undefined = $state(undefined);
  let timeEl: HTMLSpanElement | undefined = $state(undefined);
  let playBtnEl: HTMLButtonElement | undefined = $state(undefined);

  // All mutable state as plain variables -- updated via setInterval to DOM directly
  // This avoids Svelte 5 async reactivity issues entirely
  let segments: { audio: HTMLAudioElement; duration: number; startOffset: number }[] = [];
  let totalDuration = 0;
  let currentIdx = 0;
  let isPlaying = false;
  let gapMs = 400;
  let tickInterval: ReturnType<typeof setInterval> | null = null;

  function fmt(sec: number): string {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return m + ':' + String(s).padStart(2, '0');
  }

  function updateDisplay() {
    if (!segments.length || !timeEl || !scrubEl || !progressEl) return;
    const seg = segments[currentIdx];
    if (!seg?.audio) return;
    const globalTime = seg.startOffset + seg.audio.currentTime;
    const pct = totalDuration > 0 ? (globalTime / totalDuration) * 100 : 0;
    timeEl.textContent = fmt(globalTime) + ' / ' + fmt(totalDuration);
    scrubEl.value = String(pct);
    progressEl.style.width = pct + '%';
  }

  // Auto-load on mount (component created when parent {#if} renders)
  onMount(() => {
    if (filenames.length > 0) load([...filenames]);
  });

  export async function load(fnames: string[]) {
    closePlayer();
    segments = [];
    totalDuration = 0;
    currentIdx = 0;

    // Load all audio segments
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

    if (loaded.length === 0) {
      if (timeEl) timeEl.textContent = 'No audio';
      return;
    }

    // Compute offsets
    let off = 0;
    for (let i = 0; i < loaded.length; i++) {
      loaded[i].startOffset = off;
      off += loaded[i].duration;
      if (i < loaded.length - 1) off += gapMs / 1000;
    }
    segments = loaded;
    totalDuration = off;

    if (timeEl) timeEl.textContent = '0:00 / ' + fmt(totalDuration);

    // Render waveform
    await renderWaveform(fnames);

    // Auto-play
    startPlayback();
  }

  function startPlayback() {
    if (segments.length === 0) return;
    isPlaying = true;
    if (playBtnEl) playBtnEl.innerHTML = '&#9646;&#9646;';
    currentIdx = 0;
    playSegment(0);
    if (!tickInterval) tickInterval = setInterval(updateDisplay, 200);
  }

  function playSegment(idx: number, seekTime?: number) {
    if (idx >= segments.length || !isPlaying) {
      stopPlayback();
      return;
    }
    currentIdx = idx;
    const seg = segments[idx];
    if (!seg.audio || !seg.duration) {
      playSegment(idx + 1);
      return;
    }

    seg.audio.onended = null;
    seg.audio.ontimeupdate = null;
    seg.audio.currentTime = seekTime ?? 0;
    seg.audio.play().catch(() => {});

    let advanced = false;
    function advanceToNext() {
      if (advanced) return;
      advanced = true;
      seg.audio.pause();
      seg.audio.ontimeupdate = null;
      seg.audio.onended = null;
      if (!isPlaying) return;
      if (idx + 1 < segments.length) {
        setTimeout(() => playSegment(idx + 1), gapMs);
      } else {
        stopPlayback();
      }
    }

    seg.audio.onended = advanceToNext;
    seg.audio.ontimeupdate = function () {
      if (seg.audio.currentTime >= seg.audio.duration - 0.05) advanceToNext();
    };
  }

  function stopPlayback() {
    isPlaying = false;
    if (playBtnEl) playBtnEl.innerHTML = '&#9654;';
    for (const seg of segments) seg.audio.pause();
  }

  function togglePlayPause() {
    if (isPlaying) {
      stopPlayback();
    } else if (segments.length > 0) {
      isPlaying = true;
      if (playBtnEl) playBtnEl.innerHTML = '&#9646;&#9646;';
      playSegment(currentIdx);
      if (!tickInterval) tickInterval = setInterval(updateDisplay, 200);
    }
  }

  function handleScrub() {
    if (!totalDuration || !scrubEl) return;
    const targetTime = (parseFloat(scrubEl.value) / 100) * totalDuration;
    seekTo(targetTime);
  }

  function seekTo(targetTime: number) {
    for (let i = 0; i < segments.length; i++) {
      const seg = segments[i];
      const segEnd = seg.startOffset + seg.duration;
      if (targetTime <= segEnd || i === segments.length - 1) {
        for (const s of segments) { s.audio.pause(); s.audio.ontimeupdate = null; s.audio.onended = null; }
        const seekPos = Math.max(0, targetTime - seg.startOffset);
        currentIdx = i;
        if (isPlaying) {
          playSegment(i, seekPos);
        } else {
          seg.audio.currentTime = seekPos;
        }
        updateDisplay();
        return;
      }
    }
  }

  function handleWaveformClick(e: MouseEvent) {
    if (!waveformWrap || !totalDuration) return;
    const rect = waveformWrap.getBoundingClientRect();
    const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    seekTo(pct * totalDuration);
  }

  async function renderWaveform(fnames: string[]) {
    if (!canvasEl || !waveformWrap) return;
    let audioCtx: AudioContext;
    try { audioCtx = new AudioContext(); } catch { return; }

    const allSamples: Float32Array[] = [];
    for (const fname of fnames) {
      try {
        const resp = await fetch('/recordings/raw/' + encodeURIComponent(fname));
        const buf = await resp.arrayBuffer();
        if (buf.byteLength === 0) continue;
        const decoded = await audioCtx.decodeAudioData(buf);
        allSamples.push(decoded.getChannelData(0));
        if (fnames.indexOf(fname) < fnames.length - 1) {
          allSamples.push(new Float32Array(Math.floor(decoded.sampleRate * 0.4)));
        }
      } catch { /* skip */ }
    }
    audioCtx.close();
    if (allSamples.length === 0) return;

    let totalLen = 0;
    for (const s of allSamples) totalLen += s.length;
    const combined = new Float32Array(totalLen);
    let offset = 0;
    for (const s of allSamples) { combined.set(s, offset); offset += s.length; }

    const dpr = window.devicePixelRatio || 1;
    const w = waveformWrap.clientWidth;
    const h = waveformWrap.clientHeight;
    canvasEl.width = w * dpr;
    canvasEl.height = h * dpr;
    const ctx = canvasEl.getContext('2d')!;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const barsCount = Math.floor(w / 3);
    const samplesPerBar = Math.floor(combined.length / barsCount);
    const mid = h / 2;

    ctx.fillStyle = '#555';
    for (let b = 0; b < barsCount; b++) {
      const start = b * samplesPerBar;
      const end = start + samplesPerBar;
      let max = 0;
      for (let s = start; s < end && s < combined.length; s++) {
        const abs = Math.abs(combined[s]);
        if (abs > max) max = abs;
      }
      const barH = Math.max(1, max * mid * 0.9);
      ctx.fillRect(b * 3, mid - barH, 2, barH * 2);
    }
  }

  function closePlayer() {
    stopPlayback();
    if (tickInterval) { clearInterval(tickInterval); tickInterval = null; }
    for (const seg of segments) { seg.audio.pause(); seg.audio.src = ''; }
    segments = [];
    totalDuration = 0;
    if (onclose) onclose();
  }

  onDestroy(() => {
    if (tickInterval) clearInterval(tickInterval);
    for (const seg of segments) { seg.audio.pause(); seg.audio.src = ''; }
  });

  export function playAudio() { if (!isPlaying) togglePlayPause(); }
  export function pauseAudio() { if (isPlaying) stopPlayback(); }
  export function toggleAudio() { togglePlayPause(); }
</script>

<div class="audio-player">
  <div class="controls">
    <span class="speaker-name" style:color={speakerColor}>{speaker}</span>
    <button bind:this={playBtnEl} class="play-pause-btn" onclick={togglePlayPause} title="Play/Pause">&#9654;</button>
    <span bind:this={timeEl} class="time-display">0:00 / 0:00</span>
    <input bind:this={scrubEl} class="scrub-slider" type="range" min="0" max="100" step="0.1" value="0" oninput={handleScrub} />
    <button class="close-btn" onclick={closePlayer} title="Close">&times;</button>
  </div>
  <div bind:this={waveformWrap} class="waveform-container" onclick={handleWaveformClick} role="slider" tabindex="0" aria-label="Waveform">
    <canvas bind:this={canvasEl} class="waveform-canvas"></canvas>
    <div bind:this={progressEl} class="waveform-progress" style:background-color={speakerColor}></div>
  </div>
</div>

<style>
  .audio-player {
    position: fixed; bottom: 0; left: 0; right: 0;
    background: var(--bg-secondary); border-top: 1px solid var(--border);
    z-index: 100; display: flex; flex-direction: column;
  }
  .controls {
    display: flex; align-items: center; gap: 10px;
    padding: 6px 16px; height: 36px; flex-shrink: 0;
  }
  .speaker-name { font-weight: 600; font-size: 13px; white-space: nowrap; }
  .play-pause-btn {
    background: none; border: none; color: var(--text-primary);
    cursor: pointer; font-size: 14px; padding: 2px 6px; line-height: 1;
  }
  .play-pause-btn:hover { color: var(--accent); }
  .time-display {
    color: var(--text-secondary); font-size: 12px; font-family: monospace;
    white-space: nowrap; min-width: 80px;
  }
  .scrub-slider {
    flex: 1; height: 4px; accent-color: var(--accent);
    cursor: pointer; min-width: 60px; max-width: 200px;
  }
  .close-btn {
    background: none; border: none; color: var(--text-secondary);
    cursor: pointer; font-size: 18px; padding: 0 4px; line-height: 1;
  }
  .close-btn:hover { color: var(--text-primary); }
  .waveform-container {
    position: relative; height: 40px; cursor: pointer; overflow: hidden;
  }
  .waveform-canvas { width: 100%; height: 100%; display: block; }
  .waveform-progress {
    position: absolute; top: 0; left: 0; bottom: 0;
    opacity: 0.2; pointer-events: none; width: 0%;
  }
</style>
