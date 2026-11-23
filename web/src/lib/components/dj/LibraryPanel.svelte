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

  // Playlist (folder) list: foldable from the "All music" row, and its height is set by dragging the
  // handle under it; the track list takes the rest. Both scroll on overflow.
  // null height = automatic (content up to 42% of the panel).
  const MIN_FOLDERS = 72;
  const MIN_TRACKS = 140;
  let foldersFolded = $state(local.get('dj.lib.foldersFolded') === '1');
  let foldersH = $state<number | null>(Number(local.get('dj.lib.foldersH')) || null);
  let libH = $state(0);
  let libEl = $state<HTMLElement | null>(null);
  let foldersEl = $state<HTMLElement | null>(null);
  // Space above the list plus the track-list header, measured from the
  // panel's own box (offsetTop would be relative to some outer ancestor).
  let chrome = $state(150);
  $effect(() => {
    libH;
    if (!libEl || !foldersEl) return;
    const head = libEl.querySelector('.tracks-head') as HTMLElement | null;
    chrome = foldersEl.getBoundingClientRect().top - libEl.getBoundingClientRect().top + (head?.offsetHeight ?? 0) + 10;
  });
  const maxFolders = $derived(Math.max(MIN_FOLDERS, libH - chrome - MIN_TRACKS));
  function saveFolders() {
    local.set('dj.lib.foldersFolded', foldersFolded ? '1' : '0');
    local.set('dj.lib.foldersH', foldersH ? String(foldersH) : '');
  }
  function toggleFolders() {
    foldersFolded = !foldersFolded;
    saveFolders();
  }
  function setFoldersH(h: number) {
    foldersH = Math.max(MIN_FOLDERS, Math.min(maxFolders, Math.round(h)));
  }
  let fdrag: { startY: number; start: number } | null = null;
  function fDown(e: PointerEvent) {
    if (e.button !== 0) return;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    fdrag = { startY: e.clientY, start: foldersEl?.offsetHeight ?? 200 };
  }
  function fMove(e: PointerEvent) {
    if (fdrag) setFoldersH(fdrag.start + (e.clientY - fdrag.startY));
  }
  function fUp() {
    if (!fdrag) return;
    fdrag = null;
    saveFolders();
  }
  function fKey(e: KeyboardEvent) {
    const step = e.shiftKey ? 80 : 20;
    let d = 0;
    if (e.key === 'ArrowUp') d = -step;
    else if (e.key === 'ArrowDown') d = step;
    else if (e.key === 'Home') foldersH = null; // back to automatic
    else return;
    e.preventDefault();
    e.stopPropagation();
    if (d) setFoldersH((foldersEl?.offsetHeight ?? 200) + d);
    saveFolders();
  }

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

  // Rename edits the folder's own name in place (its parent stays the same).
  let renaming = $state<string | null>(null);
  let renameName = $state('');
  function startRename(f: string) {
    renaming = f;
    renameName = folderName(f);
  }
  function focusSelect(el: HTMLInputElement) {
    el.focus();
    el.select();
  }
  async function submitRename() {
    const from = renaming;
    const name = renameName.trim();
    renaming = null;
    if (!from || !name || name === folderName(from)) return;
    const to = [parentOf(from), name].filter(Boolean).join('/');
    try {
      await dj.folders.rename(from, to);
      remapLocal(from, to);
      onchanged?.();
    } catch (e) {
      addToast(`Couldn't rename: ${(e as Error).message}`, 'error');
    }
  }
  /** After a rename or move, keep the selection and collapsed set pointing
   *  at the same folders. */
  function remapLocal(from: string, to: string) {
    const move = (p: string) => (p === from ? to : p && inFolder(p, from) ? to + p.slice(from.length) : p);
    if (selected) selected = move(selected);
    if ([...collapsed].some((c) => c === from || inFolder(c, from))) {
      collapsed = new Set([...collapsed].map(move));
      local.set('dj.lib.collapsed', JSON.stringify([...collapsed]));
    }
  }

  // ---- Arrange: drag a folder before/after another, or into one ----
  // Dropping on the top or bottom quarter of a row places the folder next to
  // it; the middle puts it inside; "All music" moves it to the top level.
  // Moving to another parent is a rename on the server; the order among the
  // new siblings is saved separately (PUT /folders/order).
  // Keyboard equivalent on a focused row (HTML5 drag doesn't work on touch):
  // Alt+Up/Down reorders, Alt+Right moves into the folder above, Alt+Left
  // moves out to the parent's level.
  type Zone = 'before' | 'after' | 'into';
  let dragF = $state<string | null>(null);
  let dropAt = $state<{ f: string; zone: Zone } | null>(null);
  const siblings = (parent: string) => folders.filter((o) => parentOf(o) === parent);
  const join = (parent: string, name: string) => (parent ? `${parent}/${name}` : name);
  // A folder can't go into itself or its own sub-folders.
  const canDrop = (src: string, target: string) => target !== src && !(target && inFolder(target, src));

  function onFolderDragStart(e: DragEvent, f: string) {
    if (compact || renaming) return;
    dragF = f;
    e.dataTransfer?.setData('text/plain', f);
    if (e.dataTransfer) e.dataTransfer.effectAllowed = 'move';
  }
  function zoneOf(e: DragEvent, target: string): Zone {
    if (target === '') return 'into';
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const y = (e.clientY - r.top) / r.height;
    return y < 0.25 ? 'before' : y > 0.75 ? 'after' : 'into';
  }
  function onFolderDragOver(e: DragEvent, target: string) {
    if (dragF === null || !canDrop(dragF, target)) return;
    e.preventDefault();
    if (e.dataTransfer) e.dataTransfer.dropEffect = 'move';
    dropAt = { f: target, zone: zoneOf(e, target) };
  }
  function onFolderDrop(e: DragEvent, target: string) {
    if (dragF === null || !canDrop(dragF, target)) return endFolderDrag();
    e.preventDefault();
    const src = dragF;
    const zone = zoneOf(e, target);
    endFolderDrag();
    if (zone === 'into') void moveFolder(src, target, null, false);
    else void moveFolder(src, parentOf(target), target, zone === 'after');
  }
  function endFolderDrag() {
    dragF = null;
    dropAt = null;
  }

  /** Move src under newParent, next to `anchor` (before or after it), or
   *  at the end when anchor is null. */
  async function moveFolder(src: string, newParent: string, anchor: string | null, after: boolean) {
    const name = folderName(src);
    const dst = join(newParent, name);
    const moving = dst !== src;
    if (moving && folders.some((o) => o !== src && o.toLowerCase() === dst.toLowerCase())) {
      addToast(`${newParent ? folderName(newParent) : 'The top level'} already has a folder called ${name}`, 'error');
      return;
    }
    const order = siblings(newParent).filter((o) => o !== src);
    let at = anchor === null ? order.length : order.indexOf(anchor) + (after ? 1 : 0);
    if (at < 0) at = order.length;
    order.splice(at, 0, dst);
    try {
      if (moving) {
        await dj.folders.rename(src, dst);
        remapLocal(src, dst);
        if (newParent && collapsed.has(newParent)) toggle(newParent); // show where it went
      }
      await dj.folders.order(order);
    } catch (e) {
      addToast(`Couldn't move ${name}: ${(e as Error).message}`, 'error');
    }
    onchanged?.();
  }

  function folderKey(e: KeyboardEvent, f: string) {
    if (!e.altKey || compact) return;
    const parent = parentOf(f);
    const sibs = siblings(parent);
    const i = sibs.indexOf(f);
    if (e.key === 'ArrowUp' && i > 0) void moveFolder(f, parent, sibs[i - 1], false);
    else if (e.key === 'ArrowDown' && i < sibs.length - 1) void moveFolder(f, parent, sibs[i + 1], true);
    else if (e.key === 'ArrowRight' && i > 0) void moveFolder(f, sibs[i - 1], null, false);
    else if (e.key === 'ArrowLeft' && parent !== '') void moveFolder(f, parentOf(parent), parent, true);
    else return;
    e.preventDefault();
    e.stopPropagation();
  }
  function renameKey(e: KeyboardEvent) {
    if (e.key === 'Escape') {
      e.stopPropagation();
      renaming = null;
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

  // Row menu: the rarely used actions (play now, queue, delete), so each row
  // shows only the deck buttons. One menu open at a time.
  let menuFor = $state<number | null>(null);
  function toggleMenu(id: number) {
    menuFor = menuFor === id ? null : id;
    if (armed?.startsWith('t:')) armed = null;
  }
  function closeMenus(e: PointerEvent) {
    if (menuFor !== null && !(e.target as Element).closest?.('.row-menu, .more')) menuFor = null;
  }
  function menuKey(e: KeyboardEvent) {
    if (e.key === 'Escape' && menuFor !== null) {
      e.stopPropagation();
      menuFor = null;
    }
  }
  async function menuRun(key: string, cmd: string, args: Record<string, unknown>) {
    menuFor = null;
    await run(key, cmd, args);
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

<section class="lib card" class:compact class:stale={!live} aria-label="Music library" bind:clientHeight={libH} bind:this={libEl}>
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

  <ul class="folders" aria-label="Folders" bind:this={foldersEl}
    class:sized={foldersH !== null && !foldersFolded}
    style:height={foldersH !== null && !foldersFolded ? `${Math.min(foldersH, maxFolders)}px` : null}>
    <li class:selected={selected === ''} class:current={st?.folder === ''}
      class:drop-into={dropAt?.f === '' && dropAt.zone === 'into'}
      ondragover={(e) => onFolderDragOver(e, '')} ondragleave={() => (dropAt = null)} ondrop={(e) => onFolderDrop(e, '')}>
      {#if folders.length}
        <button class="twist" onclick={toggleFolders} aria-expanded={!foldersFolded}
          aria-label="{foldersFolded ? 'Expand' : 'Collapse'} all music">{foldersFolded ? '▸' : '▾'}</button>
      {/if}
      <button class="row-main" onclick={() => (selected = '')}>
        <span class="row-title">All music</span>
        <span class="tiny subtle">{tracks.length}{libraryMs ? ` · ${fmtMs(libraryMs)}` : ''}</span>
      </button>
      <button class="icon-btn play" onclick={() => run('f:', 'folder.play', { folder: '' })}
        disabled={!live || !inVoice || tracks.length === 0 || pending !== null} aria-label="Play all music"
        title="Play all music (replaces the queue)">&#9654;</button>
    </li>
    {#each foldersFolded ? [] : folders as f (f)}
      {#if !hidden(f)}
        {@const fs = stats.get(f)}
        <li class:selected={selected === f} class:current={st?.folder === f} style:--depth={folderDepth(f)}
          draggable={!compact && renaming !== f} class:dragging={dragF === f}
          class:drop-before={dropAt?.f === f && dropAt.zone === 'before'}
          class:drop-after={dropAt?.f === f && dropAt.zone === 'after'}
          class:drop-into={dropAt?.f === f && dropAt.zone === 'into'}
          ondragstart={(e) => onFolderDragStart(e, f)} ondragend={endFolderDrag}
          ondragover={(e) => onFolderDragOver(e, f)} ondragleave={() => (dropAt = null)} ondrop={(e) => onFolderDrop(e, f)}>
          {#if hasChildren(f)}
            <button class="twist" onclick={() => toggle(f)} aria-expanded={!collapsed.has(f)}
              aria-label="{collapsed.has(f) ? 'Expand' : 'Collapse'} {folderName(f)}">{collapsed.has(f) ? '▸' : '▾'}</button>
          {:else}
            <span class="twist" aria-hidden="true"></span>
          {/if}
          {#if renaming === f}
            <form class="rename" onsubmit={(e) => { e.preventDefault(); submitRename(); }}>
              <input class="input" bind:value={renameName} use:focusSelect onkeydown={renameKey} onblur={submitRename}
                aria-label="New name for {folderName(f)}" />
            </form>
          {:else}
            <button class="row-main" onclick={() => (selected = f)} ondblclick={() => !compact && startRename(f)}
              onkeydown={(e) => folderKey(e, f)}
              title={compact ? undefined : 'Drag to rearrange, or onto another folder to put it inside (Alt+arrow keys also work). Double-click to rename.'}>
              <span class="row-title">{folderName(f)}</span>
              <span class="tiny subtle">{fs?.count ?? 0}{fs?.ms ? ` · ${fmtMs(fs.ms)}` : ''}</span>
            </button>
          {/if}
          {#if selected === f && !compact && renaming !== f}
            <button class="btn btn-sm" onclick={() => startRename(f)} aria-label="Rename folder {folderName(f)}">Rename</button>
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
  {#if !foldersFolded && folders.length}
  <!-- A focusable separator is the ARIA window-splitter pattern (arrow keys
       resize); the lint rule doesn't know it. -->
  <!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions -->
  <div class="fgutter" role="separator" aria-orientation="horizontal" aria-label="Resize playlists (Home: automatic)"
    aria-valuemin={MIN_FOLDERS} aria-valuemax={maxFolders} aria-valuenow={foldersH ?? undefined} tabindex="0"
    onpointerdown={fDown} onpointermove={fMove} onpointerup={fUp} onpointercancel={fUp} onkeydown={fKey}
    ondblclick={() => { foldersH = null; saveFolders(); }}></div>
  {/if}

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
        <li class:missing={t.missing} class:current={deckTrackIds.has(t.id)} class:menu-open={menuFor === t.id}>
          <span class="row-main static">
            <span class="row-title" title={trackTitle(t)}>{trackTitle(t)}</span>
            <span class="tiny subtle">{[t.artist, shownPath(t)].filter(Boolean).join(' · ') || ' '}</span>
          </span>
          <span class="dur">{t.durationMs ? fmtMs(t.durationMs) : ''}</span>
          <span class="acts">
            <button class="chip-btn a" onclick={() => run(`a:${t.id}`, 'deck.load', { deck: 0, trackId: t.id })}
              disabled={!live || t.missing || pending !== null} aria-label="Load {trackTitle(t)} on deck A" title="Load on deck A">A</button>
            <button class="chip-btn b" onclick={() => run(`b:${t.id}`, 'deck.load', { deck: 1, trackId: t.id })}
              disabled={!live || t.missing || pending !== null} aria-label="Load {trackTitle(t)} on deck B" title="Load on deck B">B</button>
            <button class="icon-btn more" onclick={() => toggleMenu(t.id)} aria-haspopup="menu" aria-expanded={menuFor === t.id}
              aria-label="More for {trackTitle(t)}" title="More">&#8943;</button>
          </span>
          {#if menuFor === t.id}
            <div class="row-menu" role="menu" aria-label="{trackTitle(t)} actions">
              <button role="menuitem" onclick={() => menuRun(`p:${t.id}`, 'track.play', { trackId: t.id })}
                disabled={!live || !inVoice || t.missing}>Play now (crossfade)</button>
              <button role="menuitem" onclick={() => menuRun(`n:${t.id}`, 'queue.add', { trackId: t.id, next: true })}
                disabled={!live || t.missing}>Play next</button>
              <button role="menuitem" onclick={() => menuRun(`q:${t.id}`, 'queue.add', { trackId: t.id })}
                disabled={!live || t.missing}>Add to queue</button>
              {#if !compact}
                <hr />
                <button role="menuitem" class="danger" onclick={() => deleteTrack(t)}>
                  {armed === `t:${t.id}` ? 'Tap again to delete the file' : 'Delete file…'}
                </button>
              {/if}
            </div>
          {/if}
        </li>
      {/each}
    </ul>
  {/if}
</section>

<svelte:window onpointerdown={closeMenus} onkeydown={menuKey} />

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
  /* A dragged height holds (the track list keeps MIN_TRACKS via maxFolders). */
  .folders.sized { flex: 0 0 auto; max-height: none; }
  .fgutter { flex-shrink: 0; height: 10px; margin: -5px 0; position: relative; z-index: 1; cursor: row-resize; touch-action: none; }
  .fgutter::after { content: ''; position: absolute; left: 50%; top: 50%; width: 40px; height: 4px; transform: translate(-50%, -50%); border-radius: 2px; background: var(--border-strong); }
  .fgutter:hover::after, .fgutter:focus-visible::after { background: var(--accent); }
  .fgutter:focus-visible { outline: none; }
  .folders li { display: flex; align-items: center; gap: 4px; padding-left: calc(var(--depth, 0) * 14px); border-radius: var(--radius-sm); min-height: 44px; }
  .folders li.selected { background: var(--surface-raised); }
  .folders li.current .row-title { color: var(--accent-text, var(--accent)); }
  .folders li[draggable='true'] { cursor: grab; }
  .folders li.dragging { opacity: 0.45; }
  .folders li.drop-before { box-shadow: inset 0 2px 0 var(--accent); }
  .folders li.drop-after { box-shadow: inset 0 -2px 0 var(--accent); }
  .folders li.drop-into { outline: 2px solid var(--accent); outline-offset: -2px; background: color-mix(in srgb, var(--accent) 12%, transparent); }
  .rename { flex: 1; min-width: 0; padding: 4px 4px 4px 0; }
  .rename .input { width: 100%; min-height: 36px; padding: 4px 8px; }
  .twist { width: 24px; height: 36px; flex-shrink: 0; background: none; border: 0; color: var(--text-2); font-size: 16px; line-height: 1; padding: 0; }
  .twist:hover { color: var(--text); }
  .tracks-head { padding: 8px 14px 4px; }
  .tracks { flex: 1; min-height: 0; overflow-y: auto; padding: 0 6px 6px; }
  .tracks li { position: relative; display: flex; align-items: center; gap: 8px; padding: 2px 4px 2px 8px; min-height: 52px; border-bottom: 1px solid var(--border-color); border-radius: var(--radius-sm); }
  .tracks li:hover, .tracks li.menu-open { background: var(--surface-raised); }
  .dur { flex-shrink: 0; font-size: 13px; color: var(--text-2); font-variant-numeric: tabular-nums; }
  .more { font-size: 18px; color: var(--text-2); }
  .row-menu {
    position: absolute; right: 6px; top: calc(100% - 4px); z-index: 30; min-width: 200px;
    display: flex; flex-direction: column; padding: 6px; border-radius: var(--radius);
    background: var(--surface-raised); border: 1px solid var(--border-strong); box-shadow: var(--shadow);
  }
  .row-menu button { text-align: left; background: none; border: 0; color: var(--text); padding: 10px 12px; border-radius: var(--radius-sm); font-size: 14px; min-height: 40px; }
  .row-menu button:hover:not(:disabled) { background: var(--surface-hover); }
  .row-menu button:disabled { color: var(--text-3); }
  .row-menu .danger { color: var(--danger); }
  .row-menu hr { border: 0; border-top: 1px solid var(--border-color); margin: 4px 0; }
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
    width: 34px; height: 34px; border-radius: 7px; font-weight: 800; font-size: 12px;
    background: var(--surface-raised); border: 1px solid var(--border-strong);
  }
  .chip-btn.a { border-color: color-mix(in srgb, var(--deck-a) 60%, var(--border-strong)); color: var(--deck-a); }
  .chip-btn.b { border-color: color-mix(in srgb, var(--deck-b) 60%, var(--border-strong)); color: var(--deck-b); }
  .chip-btn:disabled { opacity: 0.4; }
  .empty { padding: 12px 14px; margin: 0; }
  .compact .chip-btn, .compact .acts .icon-btn, .compact .play { width: 44px; height: 44px; }
  .compact .tracks li { min-height: 56px; }
</style>
