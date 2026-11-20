<script lang="ts">
  import Modal from '$lib/components/ui/Modal.svelte';
  import { dj } from '$lib/api';
  import { addToast } from '$lib/stores/toast';
  import type { DjPlaylist, DjTrack } from '$lib/types';
  import { fmtMs, trackTitle } from './format';

  let {
    open = $bindable(false),
    playlist = null,
    tracks,
    onsaved,
  }: {
    open?: boolean;
    playlist?: DjPlaylist | null;
    tracks: DjTrack[];
    onsaved?: () => void;
  } = $props();

  let name = $state('');
  let ids = $state<number[]>([]);
  let query = $state('');
  let saving = $state(false);
  let confirmDelete = $state(false);

  let seededFor: DjPlaylist | null | undefined = undefined;
  $effect(() => {
    if (!open) {
      seededFor = undefined;
      return;
    }
    if (seededFor === playlist) return;
    seededFor = playlist;
    name = playlist?.name ?? '';
    ids = [...(playlist?.trackIds ?? [])];
    query = '';
    confirmDelete = false;
  });

  const byId = $derived(new Map(tracks.map((t) => [t.id, t])));
  const totalMs = $derived(ids.reduce((sum, id) => sum + (byId.get(id)?.durationMs ?? 0), 0));
  const results = $derived.by(() => {
    const q = query.trim().toLowerCase();
    const list = tracks.filter((t) => !t.missing);
    if (!q) return list;
    return list.filter((t) => `${t.title} ${t.artist} ${t.path}`.toLowerCase().includes(q));
  });

  function move(i: number, d: number) {
    const j = i + d;
    if (j < 0 || j >= ids.length) return;
    const next = [...ids];
    [next[i], next[j]] = [next[j], next[i]];
    ids = next;
  }
  function removeAt(i: number) {
    ids = ids.filter((_, k) => k !== i);
  }
  function add(id: number) {
    ids = [...ids, id];
  }
  function addAll() {
    const have = new Set(ids);
    ids = [...ids, ...results.map((t) => t.id).filter((id) => !have.has(id))];
  }

  async function save() {
    if (!name.trim()) return;
    saving = true;
    try {
      if (playlist) await dj.playlists.update(playlist.id, { name: name.trim(), trackIds: ids });
      else await dj.playlists.create(name.trim(), ids);
      open = false;
      onsaved?.();
    } catch (e) {
      addToast(`Couldn't save playlist: ${(e as Error).message}`, 'error');
    } finally {
      saving = false;
    }
  }

  async function remove() {
    if (!playlist) return;
    try {
      await dj.playlists.delete(playlist.id);
      open = false;
      onsaved?.();
    } catch (e) {
      addToast(`Couldn't delete playlist: ${(e as Error).message}`, 'error');
    }
  }
</script>

<Modal bind:open title={playlist ? 'Edit playlist' : 'New playlist'} width="860px">
  <label class="field"><span>Name</span><input class="input" bind:value={name} placeholder="Combat" /></label>

  <div class="cols">
    <section class="col">
      <div class="col-head">
        <h3>In this playlist <span class="subtle">({ids.length}{totalMs ? ` · ${fmtMs(totalMs)}` : ''})</span></h3>
      </div>
      {#if ids.length === 0}
        <p class="subtle small empty">Add tracks from the list.</p>
      {:else}
        <ol class="list">
          {#each ids as id, i (i + ':' + id)}
            {@const t = byId.get(id)}
            <li class:missing={!t || t.missing}>
              <span class="num tiny subtle">{i + 1}</span>
              <span class="t">
                <span class="title">{t ? trackTitle(t) : `Track #${id}`}</span>
                <span class="tiny subtle">{t?.missing || !t ? 'File missing' : t.artist || t.path}</span>
              </span>
              <span class="row-actions">
                <button class="icon-btn" onclick={() => move(i, -1)} disabled={i === 0} aria-label="Move up">&#9650;</button>
                <button class="icon-btn" onclick={() => move(i, 1)} disabled={i === ids.length - 1} aria-label="Move down">&#9660;</button>
                <button class="icon-btn" onclick={() => removeAt(i)} aria-label="Remove">&times;</button>
              </span>
            </li>
          {/each}
        </ol>
      {/if}
    </section>

    <section class="col">
      <div class="col-head">
        <input class="input" type="search" bind:value={query} placeholder="Search tracks" aria-label="Search tracks" />
        <button class="btn btn-sm" onclick={addAll} disabled={results.length === 0}>Add all</button>
      </div>
      {#if tracks.length === 0}
        <p class="subtle small empty">No tracks yet. Put audio files in the server's music folder and press Rescan.</p>
      {:else}
        <ul class="list">
          {#each results as t (t.id)}
            <li>
              <span class="t">
                <span class="title">{trackTitle(t)}</span>
                <span class="tiny subtle">{t.artist || t.path}{t.durationMs ? ` · ${fmtMs(t.durationMs)}` : ''}</span>
              </span>
              <span class="row-actions">
                {#if ids.includes(t.id)}<span class="tiny subtle">added</span>{/if}
                <button class="icon-btn" onclick={() => add(t.id)} aria-label="Add {trackTitle(t)}">+</button>
              </span>
            </li>
          {/each}
        </ul>
      {/if}
    </section>
  </div>

  {#snippet footer()}
    {#if playlist}
      {#if confirmDelete}
        <button class="btn-danger" onclick={remove}>Delete playlist</button>
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
  .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; min-height: 0; }
  .col { display: flex; flex-direction: column; gap: 8px; min-width: 0; }
  .col-head { display: flex; gap: 8px; align-items: center; min-height: var(--control-h); }
  .col-head .input { flex: 1; }
  h3 { font-size: 14px; font-weight: 600; }
  .list {
    list-style: none; display: flex; flex-direction: column; overflow-y: auto; max-height: 46vh;
    border: 1px solid var(--border-color); border-radius: var(--radius); background: var(--surface-sunken);
  }
  li { display: flex; align-items: center; gap: 10px; padding: 6px 8px 6px 12px; border-bottom: 1px solid var(--border-color); min-height: 52px; }
  li:last-child { border-bottom: 0; }
  li.missing .title { color: var(--text-3); text-decoration: line-through; }
  .num { width: 18px; text-align: right; font-variant-numeric: tabular-nums; }
  .t { display: flex; flex-direction: column; min-width: 0; flex: 1; }
  .title, .t .tiny { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .title { font-size: 14px; }
  .row-actions { display: flex; align-items: center; gap: 2px; }
  .row-actions .icon-btn { width: 40px; height: 40px; }
  .empty { padding: 12px 0; }
  .spacer { flex: 1; }
  @media (max-width: 760px) { .cols { grid-template-columns: 1fr; } .list { max-height: 32vh; } }
</style>
