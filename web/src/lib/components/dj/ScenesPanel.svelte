<script lang="ts">
  // Scene tiles. Tapping one makes its folders the queue and crossfades to
  // its first track on the free deck (server-side; every device sees the
  // change). Tapping the scene that's already playing does nothing, so a
  // double tap doesn't restart the music. `strip` lays the tiles out in one
  // scrolling row for Performance mode.
  import { dj } from '$lib/api';
  import { djCmd } from '$lib/dj/control';
  import { addToast } from '$lib/stores/toast';
  import type { DjScene, DjState, DjTrack } from '$lib/types';
  import SceneEditor from './SceneEditor.svelte';
  import { tracksIn } from './library';

  let {
    st,
    live,
    scenes = $bindable(),
    tracks,
    folders,
    loaded,
    strip = false,
    onchanged,
  }: {
    st: DjState | null;
    live: boolean;
    scenes: DjScene[];
    tracks: DjTrack[];
    folders: string[];
    loaded: boolean;
    strip?: boolean;
    onchanged?: () => void;
  } = $props();

  let editorOpen = $state(false);
  let editing = $state<DjScene | null>(null);
  let pending = $state<number | null>(null);
  let arranging = $state(false);

  const inVoice = $derived(!!st?.connected);
  const musicLoaded = $derived(!!st?.decks?.some((d) => d.track));

  function sceneState(s: DjScene): 'playing' | 'starting' | 'paused' | null {
    if (!st || st.sceneId !== s.id) return null;
    if (st.transition?.kind === 'scene') return 'starting';
    if (st.onAir >= 0) return 'playing';
    return musicLoaded ? 'paused' : null;
  }

  function trackCount(s: DjScene) {
    const n = tracksIn(s.folders, tracks).length;
    return `${n} track${n === 1 ? '' : 's'}`;
  }

  async function tap(s: DjScene) {
    if (arranging) return;
    if (s.folders.length === 0) return edit(s);
    pending = s.id;
    try {
      await djCmd('scene.play', { sceneId: s.id });
    } catch (e) {
      addToast((e as Error).message, 'error');
    } finally {
      pending = null;
    }
  }

  function edit(s: DjScene | null) {
    editing = s;
    editorOpen = true;
  }

  // Arrow buttons rather than drag and drop: HTML5 drag events don't fire on
  // touch screens, and the tablet is a main place this is used.
  async function move(i: number, d: number) {
    const j = i + d;
    if (j < 0 || j >= scenes.length) return;
    const next = [...scenes];
    [next[i], next[j]] = [next[j], next[i]];
    scenes = next;
    try {
      await dj.scenes.order(next.map((s) => s.id));
    } catch (e) {
      addToast(`Couldn't save order: ${(e as Error).message}`, 'error');
      onchanged?.();
    }
  }
</script>

<section class="scenes-panel" class:strip aria-label="Scenes">
  <header>
    <h2 class="section-title">Scenes</h2>
    <div class="tools">
      {#if scenes.length > 1}
        <button class="btn btn-sm" class:btn-primary={arranging} onclick={() => (arranging = !arranging)} aria-pressed={arranging}>
          {arranging ? 'Done' : 'Arrange'}
        </button>
      {/if}
      <button class="btn btn-sm" onclick={() => edit(null)} disabled={arranging}>+ New</button>
    </div>
  </header>

  {#if loaded && scenes.length === 0}
    <button class="scene empty" onclick={() => edit(null)}>
      <span class="scene-name">Create your first scene</span>
      <span class="scene-meta">One or more music folders plus how they play: tavern, travel, combat.</span>
    </button>
  {:else}
    <div class="grid">
      {#each scenes as s, i (s.id)}
        {@const state = sceneState(s)}
        <div class="scene-wrap" class:arranging>
          <button
            class="scene"
            class:active={state === 'playing' || state === 'starting'}
            class:paused={state === 'paused'}
            style:--scene-color={s.color || 'var(--accent)'}
            onclick={() => tap(s)}
            disabled={pending === s.id || (!arranging && (!live || (!inVoice && s.folders.length > 0)))}
            tabindex={arranging ? -1 : 0}
            aria-pressed={state === 'playing'}
            title={!inVoice ? 'The bot is not in a voice channel' : undefined}
          >
            <span class="scene-name" title={s.name}>{s.name}</span>
            <span class="scene-foot">
              <span class="scene-meta">
                {#if state === 'starting'}Starting…
                {:else if state === 'playing'}Playing
                {:else if state === 'paused'}Paused
                {:else if s.folders.length}{trackCount(s)}
                {:else}No music folders{/if}
              </span>
              <span class="scene-flags">
                {#if s.shuffle}
                  <svg viewBox="0 0 24 24" role="img" aria-label="Shuffle"><title>Shuffle</title><path d="M16 3h5v5M4 20 21 3M21 16v5h-5M15 15l6 6M4 4l5 5" /></svg>
                {/if}
                {#if s.loop}
                  <svg viewBox="0 0 24 24" role="img" aria-label="Repeat"><title>Repeat</title><path d="M17 1l4 4-4 4M3 11V9a4 4 0 0 1 4-4h14M7 23l-4-4 4-4M21 13v2a4 4 0 0 1-4 4H3" /></svg>
                {/if}
              </span>
            </span>
          </button>
          {#if arranging}
            <div class="arrange">
              <button class="icon-btn" onclick={() => move(i, -1)} disabled={i === 0} aria-label="Move {s.name} earlier">&#9664;</button>
              <button class="icon-btn" onclick={() => move(i, 1)} disabled={i === scenes.length - 1} aria-label="Move {s.name} later">&#9654;</button>
            </div>
          {:else}
            <button class="scene-edit icon-btn" onclick={() => edit(s)} aria-label="Edit {s.name}">&#8943;</button>
          {/if}
        </div>
      {/each}
    </div>
  {/if}
</section>

<SceneEditor bind:open={editorOpen} scene={editing} {folders} {tracks} onsaved={() => onchanged?.()} />

<style>
  .scenes-panel { display: flex; flex-direction: column; gap: 10px; min-height: 0; }
  header { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
  .tools { display: flex; gap: 6px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; }
  .strip .grid { display: flex; overflow-x: auto; gap: 10px; padding-bottom: 4px; scroll-snap-type: x proximity; }
  .strip .scene-wrap { flex: 0 0 200px; scroll-snap-align: start; }
  .scene-wrap { position: relative; }
  .scene {
    --scene-color: var(--accent);
    width: 100%; min-height: 92px; padding: 14px 14px 12px; border-radius: var(--radius-lg); text-align: left;
    display: flex; flex-direction: column; justify-content: space-between; gap: 10px;
    background: var(--surface); border: 1px solid var(--border-strong); color: var(--text);
    border-top: 5px solid var(--scene-color);
    transition: background 0.15s, border-color 0.15s, transform 0.08s;
  }
  .strip .scene { min-height: 68px; padding: 8px 12px; gap: 6px; }
  .strip { gap: 6px; }
  .strip header :global(.btn) { min-height: 32px; }
  .strip header :global(.section-title) { font-size: 14px; }
  .scene:hover:not(:disabled) { background: var(--surface-hover); }
  .scene:active:not(:disabled) { transform: scale(0.985); }
  .scene.active {
    background: color-mix(in srgb, var(--scene-color) 16%, var(--surface));
    border-color: var(--scene-color);
    box-shadow: 0 0 0 1px var(--scene-color), 0 6px 24px color-mix(in srgb, var(--scene-color) 25%, transparent);
  }
  .scene.paused { border-color: var(--scene-color); }
  .scene:disabled { opacity: 0.6; cursor: not-allowed; }
  .scene.empty { border: 1px dashed var(--border-strong); color: var(--text-2); min-height: 96px; }
  /* Name pinned to the top, count and flags to the bottom, so every tile
     lines up the same whatever its contents. */
  .scene-name { font-size: 16px; font-weight: 700; line-height: 1.3; padding-right: 34px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .scene-foot { display: flex; align-items: center; justify-content: space-between; gap: 8px; min-height: 18px; }
  .scene-meta { font-size: 13px; color: var(--text-2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .scene.active .scene-meta { color: var(--scene-color); font-weight: 600; }
  .scene-flags { display: flex; gap: 6px; color: var(--text-2); flex-shrink: 0; }
  .scene.active .scene-flags { color: var(--scene-color); }
  .scene-flags svg { width: 16px; height: 16px; fill: none; stroke: currentColor; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
  .scene-wrap.arranging .scene { border-style: dashed; cursor: default; }
  .scene-wrap.arranging .scene-name { padding-right: 96px; }
  .arrange { position: absolute; top: 8px; right: 6px; display: flex; gap: 4px; }
  .arrange .icon-btn { width: 40px; height: 40px; background: var(--surface-raised); border: 1px solid var(--border-strong); border-radius: var(--radius); }
  .scene-edit { position: absolute; top: 8px; right: 6px; width: 40px; height: 40px; font-size: 20px; color: var(--text-2); }
</style>
