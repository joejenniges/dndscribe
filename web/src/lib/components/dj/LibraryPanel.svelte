<script lang="ts">
  // Music library: the folder tree (each folder is a playlist and includes
  // its sub-folders) above the selected folder's tracks. Tracks load onto a
  // deck, go in the queue, or play now (crossfade). Folder management,
  // uploads and deletes are hidden in the compact (Performance) version.
  import { dj } from '$lib/api';
  import { djCmd } from '$lib/dj/control';
  import { addToast } from '$lib/stores/toast';
  import { local } from '$lib/dj/live';
  import type { DjState, DjTrack } from '$lib/types';
  import { fmtMs, trackTitle } from './format';
  import { folderName, folderDepth, folderStats, inFolder, parentOf } from './library';

  let {
    st,
    live,
    tracks,
    folders,
    selected = $bindable(''),
    compact = false,
    onupload,
    onchanged,
  }: {
    st: DjState | null;
    live: boolean;
    tracks: DjTrack[];
    folders: string[];
    selected?: string;
    compact?: boolean;
    onupload?: () => void;
    onchanged?: () => void;
  } = $props();

  const stats = $derived(folderStats(folders, tracks));
  const libraryMs = $derived(tracks.reduce((s, t) => s + t.durationMs, 0));
  const inVoice = $derived(!!st?.connected);
  const deckTrackIds = $derived(new Set((st?.decks ?? []).map((d) => d.track?.id).filter(Boolean)));

  // Collapsed folders are a per-device view preference.
  let collapsed = $state<Set<string>>(new Set(JSON.parse(local.get('dj.lib.collapsed') ?? '[]')));
  function toggle(f: string) {
    const next = new Set(collapsed);
    if (next.has(f)) next.delete(f);
    else next.add(f);
    collapsed = next;
    local.set('dj.lib.collapsed', JSON.stringify([...next]));
  }
  const hasChildren = (f: string) => folders.some((o) => o !== f && parentOf(o) === f);
  const hidden = (f: string) => {
    for (let p = parentOf(f); p !== ''; p = parentOf(p)) if (collapsed.has(p)) return true;
    return false;
  };

  let query = $state('');
  const visibleTracks = $derived.by(() => {
    const q = query.trim().toLowerCase();
    // Searching looks through the whole library, not just the open folder.
    const base = q ? tracks : selected ? tracks.filter((t) => inFolder(t.path, selected)) : tracks;
    if (!q) return base;
    return base.filter((t) => `${t.title} ${t.artist} ${t.path}`.toLowerCase().includes(q));
  });
  const shownPath = (t: DjTrack) => {
    const dir = parentOf(t.path);
    if (query.trim()) return dir;
    return selected ? (dir === selected ? '' : dir.slice(selected.length + 1)) : dir;
  };

  let pending = $state<string | null>(null);
  async function run(key: string, cmd: string, args: Record<string, unknown>) {
    pending = key;
    try {
      await djCmd(cmd, args);
    } catch (e) {
      addToast((e as Error).message, 'error');
    } finally {
      pending = null;
    }
  }

  // ---- Management (Studio only) ----
  let newFolderOpen = $state(false);
  let newFolderName = $state('');
  let scanning = $state(false);
  let armed = $state<string | null>(null);
  let armTimer: ReturnType<typeof setTimeout> | undefined;
  function arm(key: string) {
    armed = key;
    clearTimeout(armTimer);
    armTimer = setTimeout(() => (armed = null), 4000);
  }

  async function createFolder() {
    const name = newFolderName.trim();
    if (!name) return;
    const path = [selected, name].filter(Boolean).join('/');
    try {
      await dj.folders.create(path);
      newFolderName = '';
      newFolderOpen = false;
      onchanged?.();
      selected = path;
    } catch (e) {
      addToast(`Couldn't create folder: ${(e as Error).message}`, 'error');
    }
  }

  async function deleteFolder(f: string) {
    if (armed !== `f:${f}`) return arm(`f:${f}`);
    armed = null;
    try {
      await dj.folders.delete(f);
      if (selected === f || inFolder(selected, f)) selected = '';
      addToast(`Deleted ${folderName(f)}`, 'success');
    } catch (e) {
      addToast(`Couldn't delete folder: ${(e as Error).message}`, 'error');
    }
  }

  async function deleteTrack(t: DjTrack) {
    if (armed !== `t:${t.id}`) return arm(`t:${t.id}`);
    armed = null;
    try {
      await dj.deleteTrack(t.id);
    } catch (e) {
      addToast(`Couldn't delete track: ${(e as Error).message}`, 'error');
    }
  }

  async function rescan() {
    scanning = true;
    try {
      const r = await dj.rescan();
      addToast(`Found ${r.found} file${r.found === 1 ? '' : 's'}${r.missing ? `, ${r.missing} missing` : ''}`, 'success');
    } catch (e) {
      addToast(`Rescan failed: ${(e as Error).message}`, 'error');
    } finally {
      scanning = false;
    }
  }
</script>

<section class="lib card" class:compact class:stale={!live} aria-label="Music library">
  <header>
    <h2 class="section-title">Library</h2>
    {#if !compact}
      <div class="tools">
        <button class="btn btn-sm" onclick={() => (newFolderOpen = !newFolderOpen)} aria-expanded={newFolderOpen}>+ Folder</button>
        <button class="btn-primary btn-sm" onclick={() => onupload?.()}>Upload</button>
      </div>
    {/if}
  </header>
  <div class="search">
    <input class="input" type="search" bind:value={query} placeholder="Search library" aria-label="Search the music library" />
    {#if !compact}
      <button class="btn btn-sm" onclick={rescan} disabled={scanning}
        title="Pick up files copied into the music folder outside this page">{scanning ? '…' : 'Rescan'}</button>
    {/if}
  </div>
  {#if newFolderOpen && !compact}
    <form class="new-folder" onsubmit={(e) => { e.preventDefault(); createFolder(); }}>
      <input class="input" bind:value={newFolderName} placeholder={selected ? `New folder in ${folderName(selected)}` : 'New folder'} aria-label="New folder name" />
      <button class="btn btn-sm" type="submit" disabled={!newFolderName.trim()}>Create</button>
    </form>
  {/if}

  <ul class="folders" aria-label="Folders">
    <li class:selected={selected === ''} class:current={st?.folder === ''}>
      <button class="row-main" onclick={() => (selected = '')}>
        <span class="row-title">All music</span>
        <span class="tiny subtle">{tracks.length}{libraryMs ? ` · ${fmtMs(libraryMs)}` : ''}</span>
      </button>
      <button class="icon-btn play" onclick={() => run('f:', 'folder.play', { folder: '' })}
        disabled={!live || !inVoice || tracks.length === 0 || pending !== null} aria-label="Play all music"
        title="Play all music (replaces the queue)">&#9654;</button>
    </li>
    {#each folders as f (f)}
      {#if !hidden(f)}
        {@const fs = stats.get(f)}
        <li class:selected={selected === f} class:current={st?.folder === f} style:--depth={folderDepth(f)}>
          {#if hasChildren(f)}
            <button class="twist" onclick={() => toggle(f)} aria-expanded={!collapsed.has(f)}
              aria-label="{collapsed.has(f) ? 'Expand' : 'Collapse'} {folderName(f)}">{collapsed.has(f) ? '▸' : '▾'}</button>
          {:else}
            <span class="twist" aria-hidden="true"></span>
          {/if}
          <button class="row-main" onclick={() => (selected = f)}>
            <span class="row-title">{folderName(f)}</span>
            <span class="tiny subtle">{fs?.count ?? 0}{fs?.ms ? ` · ${fmtMs(fs.ms)}` : ''}</span>
          </button>
          {#if selected === f && !compact}
            <button class="btn btn-sm" class:btn-danger={armed === `f:${f}`} onclick={() => deleteFolder(f)}
              aria-label="Delete folder {folderName(f)}">{armed === `f:${f}` ? 'Delete all?' : 'Delete'}</button>
          {/if}
          <button class="icon-btn play" onclick={() => run(`f:${f}`, 'folder.play', { folder: f })}
            disabled={!live || !inVoice || !fs?.count || pending !== null} aria-label="Play {folderName(f)}"
            title="Play this folder (replaces the queue)">&#9654;</button>
        </li>
      {/if}
    {/each}
  </ul>

  <div class="tracks-head">
    <span class="tiny subtle">
      {query.trim() ? `Search results (${visibleTracks.length})` : `${selected ? folderName(selected) : 'All music'} (${visibleTracks.length})`}
    </span>
  </div>
  {#if tracks.length === 0}
    <p class="empty subtle small">No tracks yet.{compact ? '' : ' Press Upload, or drop audio files or a folder anywhere on this page.'}</p>
  {:else if visibleTracks.length === 0}
    <p class="empty subtle small">{query ? 'No matches.' : 'This folder is empty.'}</p>
  {:else}
    <ul class="tracks" aria-label="Tracks">
      {#each visibleTracks as t (t.id)}
        <li class:missing={t.missing} class:current={deckTrackIds.has(t.id)}>
          <span class="row-main static">
            <span class="row-title">{trackTitle(t)}</span>
            <span class="tiny subtle">{[t.artist, shownPath(t), t.durationMs ? fmtMs(t.durationMs) : ''].filter(Boolean).join(' · ')}</span>
          </span>
          <span class="acts">
            <button class="chip-btn a" onclick={() => run(`a:${t.id}`, 'deck.load', { deck: 0, trackId: t.id })}
              disabled={!live || t.missing || pending !== null} aria-label="Load {trackTitle(t)} on deck A">A</button>
            <button class="chip-btn b" onclick={() => run(`b:${t.id}`, 'deck.load', { deck: 1, trackId: t.id })}
              disabled={!live || t.missing || pending !== null} aria-label="Load {trackTitle(t)} on deck B">B</button>
            {#if !compact}
              <button class="icon-btn" onclick={() => run(`q:${t.id}`, 'queue.add', { trackId: t.id })}
                disabled={!live || t.missing || pending !== null} aria-label="Add {trackTitle(t)} to the queue" title="Add to queue">+</button>
            {/if}
            <button class="icon-btn play" onclick={() => run(`p:${t.id}`, 'track.play', { trackId: t.id })}
              disabled={!live || !inVoice || t.missing || pending !== null} aria-label="Play {trackTitle(t)} now"
              title="Crossfade to this track now">&#9654;</button>
            {#if !compact}
              <button class="icon-btn del" class:armed={armed === `t:${t.id}`} onclick={() => deleteTrack(t)}
                aria-label={armed === `t:${t.id}` ? `Confirm delete ${trackTitle(t)}` : `Delete ${trackTitle(t)}`}
                title={armed === `t:${t.id}` ? 'Tap again to delete the file' : 'Delete'}>{armed === `t:${t.id}` ? 'Delete?' : '✕'}</button>
            {/if}
          </span>
        </li>
      {/each}
    </ul>
  {/if}
</section>

<style>
  .lib { display: flex; flex-direction: column; min-height: 0; height: 100%; overflow: hidden; }
  .stale { opacity: 0.6; }
  header { display: flex; align-items: center; justify-content: space-between; gap: 8px; padding: 12px 12px 8px 14px; }
  .tools { display: flex; gap: 6px; }
  .search { display: flex; gap: 6px; padding: 0 12px 8px; }
  .search .input { flex: 1; min-width: 0; }
  .new-folder { display: flex; gap: 6px; padding: 0 12px 8px; }
  .new-folder .input { flex: 1; min-width: 0; }
  ul { list-style: none; margin: 0; padding: 0; }
  .folders { flex: 0 1 auto; max-height: 42%; overflow-y: auto; border-top: 1px solid var(--border-color); border-bottom: 1px solid var(--border-color); padding: 4px 6px; }
  .folders li { display: flex; align-items: center; gap: 4px; padding-left: calc(var(--depth, 0) * 14px); border-radius: var(--radius-sm); min-height: 44px; }
  .folders li.selected { background: var(--surface-raised); }
  .folders li.current .row-title { color: var(--accent-text, var(--accent)); }
  .twist { width: 24px; height: 36px; flex-shrink: 0; background: none; border: 0; color: var(--text-3); font-size: 12px; padding: 0; }
  .tracks-head { padding: 8px 14px 4px; }
  .tracks { flex: 1; min-height: 0; overflow-y: auto; padding: 0 6px 6px; }
  .tracks li { display: flex; align-items: center; gap: 6px; padding: 2px 6px; min-height: 52px; border-bottom: 1px solid var(--border-color); }
  .tracks li:last-child { border-bottom: 0; }
  .tracks li.current .row-title { color: var(--accent-text, var(--accent)); }
  .tracks li.missing .row-title { color: var(--text-3); text-decoration: line-through; }
  .row-main { flex: 1; min-width: 0; display: flex; flex-direction: column; align-items: flex-start; text-align: left; padding: 6px 4px; background: none; border: 0; color: inherit; }
  button.row-main:hover .row-title { text-decoration: underline; }
  .row-title { font-size: 14px; font-weight: 600; max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .row-main .tiny { max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .acts { display: flex; align-items: center; gap: 2px; flex-shrink: 0; }
  .acts .icon-btn, .play { width: 32px; height: 36px; flex-shrink: 0; }
  .chip-btn {
    width: 30px; height: 32px; border-radius: 7px; font-weight: 800; font-size: 12px;
    background: var(--surface-raised); border: 1px solid var(--border-strong);
  }
  .chip-btn.a { border-color: color-mix(in srgb, var(--deck-a) 60%, var(--border-strong)); color: var(--deck-a); }
  .chip-btn.b { border-color: color-mix(in srgb, var(--deck-b) 60%, var(--border-strong)); color: var(--deck-b); }
  .chip-btn:disabled { opacity: 0.4; }
  .del { color: var(--text-3); font-size: 13px; }
  .del.armed { width: auto; padding: 0 8px; color: var(--danger); font-weight: 600; }
  .empty { padding: 12px 14px; margin: 0; }
  .compact .chip-btn, .compact .acts .icon-btn, .compact .play { width: 44px; height: 44px; }
  .compact .tracks li { min-height: 56px; }
</style>
