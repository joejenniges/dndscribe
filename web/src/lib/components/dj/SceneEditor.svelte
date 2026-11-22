<script lang="ts">
  import Modal from '$lib/components/ui/Modal.svelte';
  import Toggle from '$lib/components/ui/Toggle.svelte';
  import { dj } from '$lib/api';
  import { addToast } from '$lib/stores/toast';
  import type { DjScene, DjTrack } from '$lib/types';
  import { SCENE_COLORS, FADE_OPTIONS, fmtMs } from './format';
  import { folderName, folderDepth, folderStats, tracksIn, parentOf } from './library';

  let {
    open = $bindable(false),
    scene = null,
    folders,
    tracks,
    onsaved,
  }: {
    open?: boolean;
    scene?: DjScene | null;
    folders: string[];
    tracks: DjTrack[];
    onsaved?: () => void;
  } = $props();

  let name = $state('');
  let picked = $state<string[]>([]);
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
    picked = [...(scene?.folders ?? [])];
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
      folders: picked,
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

  const stats = $derived(folderStats(folders, tracks));
  const pickedTracks = $derived(tracksIn(picked, tracks));
  // A folder already includes its sub-folders, so a ticked ancestor makes
  // ticking a descendant redundant: show it as included, not as a choice.
  const coveredByAncestor = (f: string) => {
    for (let p = parentOf(f); p !== ''; p = parentOf(p)) if (picked.includes(p)) return true;
    return false;
  };
  // Folders the scene points at that no longer exist on disk.
  const gone = $derived(picked.filter((f) => !folders.includes(f)));

  function toggle(f: string) {
    picked = picked.includes(f) ? picked.filter((x) => x !== f) : [...picked, f];
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

    <div class="field">
      <span>Music folders
        {#if picked.length}<span class="subtle"> · {pickedTracks.length} track{pickedTracks.length === 1 ? '' : 's'}, {fmtMs(pickedTracks.reduce((s, t) => s + t.durationMs, 0))}</span>{/if}
      </span>
      {#if folders.length === 0}
        <span class="subtle small">No folders yet. Upload some music into a folder first; each folder is a playlist.</span>
      {:else}
        <ul class="folders" aria-label="Folders this scene plays">
          {#each folders as f (f)}
            {@const covered = coveredByAncestor(f)}
            <li style:--depth={folderDepth(f) - 1}>
              <label class:covered>
                <input type="checkbox" checked={covered || picked.includes(f)} disabled={covered} onchange={() => toggle(f)} />
                <span class="fname">{folderName(f)}</span>
                <span class="tiny subtle">{stats.get(f)?.count ?? 0}</span>
              </label>
            </li>
          {/each}
        </ul>
        <span class="tiny subtle">A folder includes its sub-folders.</span>
      {/if}
      {#if gone.length}
        <span class="tiny warn">No longer on disk: {gone.join(', ')}.
          <button type="button" class="linkish" onclick={() => (picked = picked.filter((f) => folders.includes(f)))}>Remove</button></span>
      {/if}
    </div>

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
  .folders { list-style: none; margin: 0; padding: 4px 0; max-height: 260px; overflow-y: auto; border: 1px solid var(--border-color); border-radius: var(--radius); }
  .folders li { padding-left: calc(10px + var(--depth) * 20px); }
  .folders label { display: flex; align-items: center; gap: 10px; min-height: 40px; padding-right: 12px; cursor: pointer; }
  .folders label.covered { color: var(--text-3); cursor: default; }
  .folders input { width: 18px; height: 18px; accent-color: var(--accent); }
  .fname { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .warn { color: var(--warning); }
  .linkish { background: none; border: 0; padding: 0; color: inherit; text-decoration: underline; font: inherit; }
</style>
