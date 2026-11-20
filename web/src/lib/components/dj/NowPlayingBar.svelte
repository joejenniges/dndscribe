<script lang="ts">
  import { onDestroy } from 'svelte';
  import { dj } from '$lib/api';
  import { addToast } from '$lib/stores/toast';
  import type { DjState } from '$lib/types';
  import { fmtMs, FADE_OPTIONS } from './format';

  let { st }: { st: DjState | null } = $props();

  // Position is interpolated locally between server updates (sent on every
  // change and every 5s while playing). receivedAt is the client clock, so
  // server/client clock skew doesn't matter.
  let now = $state(Date.now());
  const timer = setInterval(() => (now = Date.now()), 250);
  onDestroy(() => clearInterval(timer));

  const running = $derived(!!st?.playing && !st.paused && st.connected);
  const duration = $derived(st?.track?.durationMs ?? 0);
  const position = $derived.by(() => {
    if (!st) return 0;
    const p = st.positionMs + (running ? now - (st.receivedAt ?? now) : 0);
    return duration > 0 ? Math.min(p, duration) : p;
  });
  const pct = $derived(duration > 0 ? (position / duration) * 100 : 0);

  // Volume: local value while dragging so server echoes don't fight the
  // thumb; sent at most every 120ms.
  let volume = $state(50);
  let dragging = false;
  let sendTimer: ReturnType<typeof setTimeout> | null = null;
  $effect(() => {
    if (st && !dragging) volume = Math.round(st.volume * 100);
  });
  function onVolumeInput() {
    dragging = true;
    if (sendTimer) return;
    sendTimer = setTimeout(async () => {
      sendTimer = null;
      try {
        await dj.volume(volume / 100);
      } catch (e) {
        addToast(`Volume: ${(e as Error).message}`, 'error');
      }
    }, 120);
  }
  async function onVolumeChange() {
    dragging = false;
    try {
      await dj.volume(volume / 100);
    } catch {
      /* already reported by input */
    }
  }

  async function act(fn: () => Promise<unknown>) {
    try {
      await fn();
    } catch (e) {
      addToast((e as Error).message, 'error');
    }
  }

  async function setFade(e: Event) {
    const v = Number((e.target as HTMLSelectElement).value);
    await act(() => dj.fade(v));
  }
</script>

<div class="now-playing" aria-label="Now playing">
  <div class="np-info">
    {#if st?.playing && st.track}
      <span class="np-title">{st.track.title}</span>
      <span class="subtle tiny np-sub">
        {st.programName}{st.track.artist ? ` · ${st.track.artist}` : ''}
        {#if st.paused} · Paused{/if}
        {#if !st.connected} · Waiting for voice connection{/if}
      </span>
      <div class="progress" aria-hidden="true">
        <span class="tiny subtle">{fmtMs(position)}</span>
        <div class="bar"><div class="fill" style:width="{pct}%"></div></div>
        <span class="tiny subtle">{duration ? fmtMs(duration) : ''}</span>
      </div>
    {:else}
      <span class="np-title">Nothing playing</span>
      <span class="subtle tiny">{st?.connected ? 'Tap a scene to start' : 'Discord output: bot is not in a voice channel'}</span>
    {/if}
  </div>
  <div class="np-controls">
    {#if st?.paused}
      <button class="icon-btn bordered big" onclick={() => act(dj.resume)} aria-label="Resume" title="Resume">&#9654;</button>
    {:else}
      <button class="icon-btn bordered big" onclick={() => act(dj.pause)} disabled={!st?.playing} aria-label="Pause" title="Pause">&#10074;&#10074;</button>
    {/if}
    <button class="icon-btn bordered big" onclick={() => act(dj.next)} disabled={!st?.playing} aria-label="Next track" title="Next track">&#9197;</button>
    <button class="icon-btn bordered big" onclick={() => act(() => dj.stop())} disabled={!st?.playing} aria-label="Stop (fade out)" title="Stop (fade out)">&#9632;</button>
    <label class="vol">
      <span class="subtle tiny">Volume {volume}%</span>
      <input type="range" min="0" max="100" bind:value={volume} oninput={onVolumeInput} onchange={onVolumeChange} aria-label="Master volume" />
    </label>
    <label class="vol">
      <span class="subtle tiny">Fade</span>
      <select class="select" value={st?.fadeSeconds ?? 6} onchange={setFade} aria-label="Crossfade length">
        {#each FADE_OPTIONS as f}<option value={f}>{f} s</option>{/each}
      </select>
    </label>
  </div>
</div>

<style>
  .now-playing {
    position: fixed; left: 0; right: 0; bottom: 0; z-index: 45;
    min-height: 84px; padding: 12px 24px; display: flex; align-items: center; justify-content: space-between; gap: 20px;
    background: var(--surface); border-top: 1px solid var(--border-strong); box-shadow: 0 -8px 24px rgba(0,0,0,0.35);
  }
  .np-info { display: flex; flex-direction: column; min-width: 0; flex: 1; max-width: 560px; gap: 2px; }
  .np-title { font-weight: 700; font-size: 16px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .np-sub { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .progress { display: flex; align-items: center; gap: 10px; margin-top: 4px; }
  .progress span { min-width: 38px; font-variant-numeric: tabular-nums; }
  .progress span:last-child { text-align: right; }
  .bar { flex: 1; height: 4px; background: var(--surface-raised); border-radius: 2px; overflow: hidden; }
  .fill { height: 100%; background: var(--accent); transition: width 0.25s linear; }
  .np-controls { display: flex; align-items: center; gap: 12px; }
  .big { width: var(--touch); height: var(--touch); font-size: 16px; }
  .bordered { border: 1px solid var(--border-strong); border-radius: var(--radius); }
  .vol { display: flex; flex-direction: column; gap: 2px; }
  .vol input { width: 150px; accent-color: var(--accent); height: 28px; }
  .vol .select { min-height: 36px; width: 84px; }

  @media (max-width: 760px) {
    .now-playing { flex-direction: column; align-items: stretch; padding: 12px 16px; gap: 10px; }
    .np-info { max-width: none; }
    .np-controls { flex-wrap: wrap; }
    .vol input { width: 120px; }
  }
</style>
