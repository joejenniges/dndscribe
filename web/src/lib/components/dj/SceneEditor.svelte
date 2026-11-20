<script lang="ts">
  import Modal from '$lib/components/ui/Modal.svelte';
  import Toggle from '$lib/components/ui/Toggle.svelte';
  import { dj } from '$lib/api';
  import { addToast } from '$lib/stores/toast';
  import type { DjPlaylist, DjScene } from '$lib/types';
  import { SCENE_COLORS, FADE_OPTIONS } from './format';

  let {
    open = $bindable(false),
    scene = null,
    playlists,
    onsaved,
  }: {
    open?: boolean;
    scene?: DjScene | null;
    playlists: DjPlaylist[];
    onsaved?: () => void;
  } = $props();

  let name = $state('');
  let playlistId = $state<number | null>(null);
  let color = $state(SCENE_COLORS[0]);
  let volume = $state(100);
  let shuffle = $state(false);
  let loop = $state(true);
  let fade = $state<string>('default'); // 'default' or seconds as string
  let saving = $state(false);
  let confirmDelete = $state(false);

  // Seed the form each time the dialog opens.
  let seededFor: DjScene | null | undefined = undefined;
  $effect(() => {
    if (!open) {
      seededFor = undefined;
      return;
    }
    if (seededFor === scene) return;
    seededFor = scene;
    name = scene?.name ?? '';
    playlistId = scene?.playlistId ?? playlists[0]?.id ?? null;
    color = scene?.color || SCENE_COLORS[0];
    volume = Math.round((scene?.volume ?? 1) * 100);
    shuffle = scene?.shuffle ?? false;
    loop = scene?.loop ?? true;
    fade = scene?.fadeSeconds == null ? 'default' : String(scene.fadeSeconds);
    confirmDelete = false;
  });

  async function save() {
    if (!name.trim()) return;
    saving = true;
    const body = {
      name: name.trim(),
      playlistId,
      color,
      volume: volume / 100,
      shuffle,
      loop,
      fadeSeconds: fade === 'default' ? null : Number(fade),
    };
    try {
      if (scene) await dj.scenes.update(scene.id, body);
      else await dj.scenes.create(body);
      open = false;
      onsaved?.();
    } catch (e) {
      addToast(`Couldn't save scene: ${(e as Error).message}`, 'error');
    } finally {
      saving = false;
    }
  }

  async function remove() {
    if (!scene) return;
    try {
      await dj.scenes.delete(scene.id);
      open = false;
      onsaved?.();
    } catch (e) {
      addToast(`Couldn't delete scene: ${(e as Error).message}`, 'error');
    }
  }
</script>

<Modal bind:open title={scene ? 'Edit scene' : 'New scene'} width="520px">
  <form class="form" onsubmit={(e) => { e.preventDefault(); save(); }}>
    <label class="field"><span>Name</span><input class="input" bind:value={name} placeholder="Boss fight" required /></label>

    <label class="field">
      <span>Playlist</span>
      {#if playlists.length === 0}
        <span class="subtle small">Create a playlist first, then point this scene at it.</span>
      {:else}
        <select class="select" bind:value={playlistId}>
          <option value={null}>None</option>
          {#each playlists as p (p.id)}<option value={p.id}>{p.name} ({p.trackIds.length})</option>{/each}
        </select>
      {/if}
    </label>

    <div class="field">
      <span>Colour</span>
      <div class="swatches" role="radiogroup" aria-label="Scene colour">
        {#each SCENE_COLORS as c}
          <button type="button" class="swatch" class:on={color === c} style:background={c}
            role="radio" aria-checked={color === c} aria-label={c} onclick={() => (color = c)}></button>
        {/each}
      </div>
    </div>

    <label class="field">
      <span>Scene volume: {volume}%</span>
      <input type="range" min="0" max="100" bind:value={volume} />
    </label>

    <label class="field">
      <span>Crossfade into this scene</span>
      <select class="select" bind:value={fade}>
        <option value="default">Use the global fade</option>
        {#each FADE_OPTIONS as f}<option value={String(f)}>{f} s</option>{/each}
      </select>
    </label>

    <Toggle bind:checked={shuffle} label="Shuffle" />
    <Toggle bind:checked={loop} label="Loop playlist" hint="Off: the scene stops after its last track" />
  </form>

  {#snippet footer()}
    {#if scene}
      {#if confirmDelete}
        <button class="btn-danger" onclick={remove}>Delete scene</button>
      {:else}
        <button class="btn-ghost" onclick={() => (confirmDelete = true)}>Delete</button>
      {/if}
    {/if}
    <span class="spacer"></span>
    <button class="btn" onclick={() => (open = false)}>Cancel</button>
    <button class="btn-primary" onclick={save} disabled={saving || !name.trim()}>{saving ? 'Saving…' : 'Save'}</button>
  {/snippet}
</Modal>

<style>
  .form { display: flex; flex-direction: column; gap: 14px; }
  .swatches { display: flex; gap: 10px; flex-wrap: wrap; }
  .swatch { width: 36px; height: 36px; border-radius: 50%; border: 2px solid transparent; padding: 0; }
  .swatch.on { border-color: var(--text); box-shadow: 0 0 0 2px var(--canvas) inset; }
  input[type='range'] { accent-color: var(--accent); height: 28px; }
  .spacer { flex: 1; }
</style>
