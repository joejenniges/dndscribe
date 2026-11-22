<script lang="ts">
  // DJ Studio, scenes first. The server owns playback (internal/dj); this
  // page is a remote control. Live state arrives over the WebSocket
  // (djState); the library (scenes/folders/tracks) is refetched whenever any
  // client changes it (djLibraryVersion).
  //
  // Folders in the server's music dir are the playlists; scenes pick one or
  // more of them. Music gets there through the upload dialog (pickers, or a
  // drop anywhere on this page).
  //
  // Layout intent (tablet at the table): big scene tiles you can hit without
  // looking, a persistent now-playing bar, the library and tracks below.
  import { onMount } from 'svelte';
  import { dj, channels as channelsApi, music } from '$lib/api';
  import { djState, djLibraryVersion } from '$lib/stores/dj';
  import { botStatus } from '$lib/stores/ws';
  import { addToast } from '$lib/stores/toast';
  import type { DjScene, DjTrack, VoiceChannel } from '$lib/types';
  import NowPlayingBar from '$lib/components/dj/NowPlayingBar.svelte';
  import WaveformDeck from '$lib/components/dj/WaveformDeck.svelte';
  import SceneEditor from '$lib/components/dj/SceneEditor.svelte';
  import UploadDialog from '$lib/components/dj/UploadDialog.svelte';
  import SoundboardPanel from '$lib/components/dj/SoundboardPanel.svelte';
  import { page } from '$app/stores';
  import { goto } from '$app/navigation';
  import { fmtMs, trackTitle } from '$lib/components/dj/format';
  import {
    folderName, folderDepth, folderStats, tracksIn, inFolder, fromDrop,
  } from '$lib/components/dj/library';

  // Sub-views, kept in the URL (?view=) so a reload or a bookmark on the
  // tablet lands on the same one. Scenes first: it's what gets used mid-game.
  const VIEWS = [
    { id: 'scenes', label: 'Scenes' },
    { id: 'music', label: 'Music & Playlists' },
    { id: 'soundboard', label: 'Soundboard' },
  ] as const;
  type View = (typeof VIEWS)[number]['id'];
  const view = $derived.by((): View => {
    const v = $page.url.searchParams.get('view');
    return VIEWS.some((x) => x.id === v) ? (v as View) : 'scenes';
  });
  function setView(v: View) {
    goto(v === 'scenes' ? '?' : `?view=${v}`, { replaceState: true, noScroll: true, keepFocus: true });
  }
  let soundboard: ReturnType<typeof SoundboardPanel> | undefined = $state();

  let tracks = $state<DjTrack[]>([]);
  let folders = $state<string[]>([]);
  let scenes = $state<DjScene[]>([]);
  let loaded = $state(false);

  let sceneEditorOpen = $state(false);
  let editingScene = $state<DjScene | null>(null);

  let pending = $state<string | null>(null); // key of the tile/button awaiting the server
  let trackQuery = $state('');
  let scanning = $state(false);

  async function load() {
    try {
      [tracks, folders, scenes] = await Promise.all([dj.tracks(), dj.folders.list(), dj.scenes.list()]);
      if (selectedFolder && !folders.includes(selectedFolder)) selectedFolder = '';
    } catch (e) {
      addToast(`Couldn't load DJ library: ${(e as Error).message}`, 'error');
    } finally {
      loaded = true;
    }
  }

  // Runs on mount and again whenever any client changes the library.
  $effect(() => {
    $djLibraryVersion;
    load();
  });

  onMount(async () => {
    if (!$djState) {
      try {
        djState.set({ ...(await dj.state()), receivedAt: Date.now() });
      } catch {
        /* WS init will fill it */
      }
    }
  });

  const st = $derived($djState);
  const connected = $derived(!!st?.connected);
  const stats = $derived(folderStats(folders, tracks));
  const libraryMs = $derived(tracks.reduce((s, t) => s + t.durationMs, 0));

  // '' = all music. Selecting a folder filters the track list and is the
  // default destination for uploads.
  let selectedFolder = $state('');

  const visibleTracks = $derived.by(() => {
    const q = trackQuery.trim().toLowerCase();
    const inSel = selectedFolder ? tracks.filter((t) => inFolder(t.path, selectedFolder)) : tracks;
    if (!q) return inSel;
    return inSel.filter((t) => `${t.title} ${t.artist} ${t.path}`.toLowerCase().includes(q));
  });

  /** Path shown under a track title, relative to the selected folder. */
  const shownPath = (t: DjTrack) => (selectedFolder ? t.path.slice(selectedFolder.length + 1) : t.path);

  function sceneTrackCount(s: DjScene) {
    const n = tracksIn(s.folders, tracks).length;
    return `${n} track${n === 1 ? '' : 's'}`;
  }

  async function run(key: string, fn: () => Promise<unknown>) {
    pending = key;
    try {
      await fn();
    } catch (e) {
      addToast((e as Error).message, 'error');
    } finally {
      pending = null;
    }
  }

  function tapScene(s: DjScene) {
    if (s.folders.length === 0) {
      editScene(s);
      return;
    }
    run(`scene:${s.id}`, () => dj.play({ sceneId: s.id }));
  }

  function editScene(s: DjScene | null) {
    editingScene = s;
    sceneEditorOpen = true;
  }

  // ---- Folders and deletes ----
  let newFolderOpen = $state(false);
  let newFolderName = $state('');
  // Two-tap delete: the first tap arms the button, the second deletes. No
  // modal, so it works one-handed on the tablet; arming expires after 4s.
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
    const path = [selectedFolder, name].filter(Boolean).join('/');
    try {
      await dj.folders.create(path);
      newFolderName = '';
      newFolderOpen = false;
      await load();
      selectedFolder = path;
    } catch (e) {
      addToast(`Couldn't create folder: ${(e as Error).message}`, 'error');
    }
  }

  async function deleteFolder(f: string) {
    if (armed !== `f:${f}`) return arm(`f:${f}`);
    armed = null;
    try {
      await dj.folders.delete(f);
      if (selectedFolder === f || inFolder(selectedFolder, f)) selectedFolder = '';
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

  // ---- Upload ----
  let uploadOpen = $state(false);
  let uploading = $state(false);
  let uploadProgress = $state(0);
  let uploadDialog: ReturnType<typeof UploadDialog> | undefined = $state();

  // Page-wide drop target. dragenter/leave fire for every child element, so
  // count them to know when the drag has really left the window.
  let dragDepth = $state(0);
  const hasFiles = (e: DragEvent) => !!e.dataTransfer?.types.includes('Files');
  function onDragEnter(e: DragEvent) {
    if (!hasFiles(e)) return;
    e.preventDefault();
    dragDepth++;
  }
  function onDragLeave(e: DragEvent) {
    if (!hasFiles(e)) return;
    dragDepth = Math.max(0, dragDepth - 1);
  }
  function onDragOver(e: DragEvent) {
    // Without this the browser opens a dropped file in place of the page.
    if (hasFiles(e)) e.preventDefault();
  }
  async function onDrop(e: DragEvent) {
    if (!hasFiles(e) || !e.dataTransfer) return;
    e.preventDefault();
    dragDepth = 0;
    const files = await fromDrop(e.dataTransfer);
    // On the soundboard a drop adds effects to the open tab; anywhere else
    // it's music, and the dialog asks which folder.
    if (view === 'soundboard') {
      soundboard?.addFiles(files);
      return;
    }
    uploadDialog?.addFiles(files);
    uploadOpen = true;
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

  // ---- Voice: join a channel for music without recording ----
  let channelList = $state<VoiceChannel[]>([]);
  let joinChannelId = $state('');
  let joining = $state(false);

  async function loadChannels() {
    try {
      channelList = await channelsApi.list();
      // Default to the busiest channel: that's where the table is.
      if (!joinChannelId && channelList.length) {
        joinChannelId = [...channelList].sort((a, b) => b.members.length - a.members.length)[0].id;
      }
    } catch (e) {
      addToast(`Couldn't load voice channels: ${(e as Error).message}`, 'error');
    }
  }

  $effect(() => {
    if (!$botStatus.recording && !$botStatus.musicOnly) loadChannels();
  });

  async function joinForMusic() {
    if (!joinChannelId) return;
    joining = true;
    try {
      await music.join(joinChannelId);
    } catch (e) {
      addToast(`Couldn't join: ${(e as Error).message}`, 'error');
    } finally {
      joining = false;
    }
  }

  async function leaveMusic() {
    try {
      await dj.stop(0);
      await music.leave();
    } catch (e) {
      addToast(`Couldn't leave: ${(e as Error).message}`, 'error');
    }
  }

  // ---- Arrange mode: reorder scene tiles ----
  // Arrow buttons rather than drag and drop: HTML5 drag events don't fire on
  // touch screens, and the tablet is the main place this page is used.
  let arranging = $state(false);

  async function moveScene(i: number, d: number) {
    const j = i + d;
    if (j < 0 || j >= scenes.length) return;
    const next = [...scenes];
    [next[i], next[j]] = [next[j], next[i]];
    scenes = next;
    try {
      await dj.scenes.order(next.map((s) => s.id));
    } catch (e) {
      addToast(`Couldn't save order: ${(e as Error).message}`, 'error');
      load();
    }
  }
</script>

<svelte:head><title>DJ Studio - dndscribe</title></svelte:head>

<div class="subnav">
  <nav aria-label="DJ Studio">
    {#each VIEWS as v (v.id)}
      <a href="?view={v.id}" class:active={view === v.id} aria-current={view === v.id ? 'page' : undefined}
        onclick={(e) => { e.preventDefault(); setView(v.id); }}>{v.label}</a>
    {/each}
  </nav>
</div>

<main class="dj">
  <div class="heading">
    <div>
      <div class="eyebrow">Discord music</div>
      <h1 class="page-title">DJ Studio</h1>
    </div>
    <div class="conn">
      {#if $botStatus.recording}
        <span class="badge success">{connected ? 'Playing into' : 'Connecting to'} {$botStatus.channelName ?? 'voice'} (recording)</span>
      {:else if $botStatus.musicOnly}
        <span class="badge success">{connected ? 'Playing into' : 'Connecting to'} {$botStatus.channelName ?? 'voice'}</span>
        <button class="btn btn-sm" onclick={leaveMusic}>Leave</button>
      {:else}
        <select class="select" bind:value={joinChannelId} aria-label="Voice channel for music">
          {#each channelList as c (c.id)}
            <option value={c.id}>{c.name}{c.members.length ? ` (${c.members.length})` : ''}</option>
          {/each}
        </select>
        <button class="btn-primary btn-sm" onclick={joinForMusic} disabled={!joinChannelId || joining}>
          {joining ? 'Joining…' : 'Join for music'}
        </button>
      {/if}
      <button class="btn btn-sm btn-ghost" onclick={() => run('tone', () => dj.tone(2))} disabled={!connected || pending === 'tone'}
        title="Play a 2 second 440 Hz tone to check the bot can be heard">Test tone</button>
    </div>
  </div>

  {#if st?.error}
    <div class="notice card warn" role="status">{st.error}</div>
  {/if}

  <WaveformDeck {st} />

  {#if view === 'scenes'}
  <section>
    <div class="section-head">
      <h2 class="section-title">Scenes</h2>
      <div class="pane-tools">
        {#if scenes.length > 1}
          <button class="btn btn-sm" class:btn-primary={arranging} onclick={() => (arranging = !arranging)} aria-pressed={arranging}>
            {arranging ? 'Done' : 'Arrange'}
          </button>
        {/if}
        <button class="btn btn-sm" onclick={() => editScene(null)} disabled={arranging}>+ New scene</button>
      </div>
    </div>
    {#if loaded && scenes.length === 0}
      <div class="scenes">
        <button class="scene empty" onclick={() => editScene(null)}>
          <span class="scene-name">Create your first scene</span>
          <span class="scene-meta">A scene is one or more music folders plus how they play: tavern, travel, combat, boss.</span>
        </button>
      </div>
    {:else}
      <div class="scenes">
        {#each scenes as s, i (s.id)}
          {@const active = st?.playing && st.sceneId === s.id}
          <div class="scene-wrap" class:arranging>
            <button
              class="scene"
              class:active
              class:paused={active && st?.paused}
              style:--scene-color={s.color || 'var(--accent)'}
              onclick={() => !arranging && tapScene(s)}
              disabled={pending === `scene:${s.id}`}
              tabindex={arranging ? -1 : 0}
              aria-pressed={active}
            >
              <span class="scene-name" title={s.name}>{s.name}</span>
              <span class="scene-foot">
                <span class="scene-meta">
                  {#if s.folders.length}
                    {sceneTrackCount(s)}{#if active && st?.paused} · Paused{/if}
                  {:else}
                    No music folders
                  {/if}
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
                <button class="icon-btn" onclick={() => moveScene(i, -1)} disabled={i === 0} aria-label="Move {s.name} earlier">&#9664;</button>
                <button class="icon-btn" onclick={() => moveScene(i, 1)} disabled={i === scenes.length - 1} aria-label="Move {s.name} later">&#9654;</button>
              </div>
            {:else}
              <button class="scene-edit icon-btn" onclick={() => editScene(s)} aria-label="Edit {s.name}">&#8943;</button>
            {/if}
          </div>
        {/each}
      </div>
    {/if}
  </section>

    <div class="card pane">
      <div class="pane-head">
        <h2 class="section-title">Up next</h2>
        {#if st?.playing}<span class="tiny subtle">{st.programName}{st.loop ? ' · loops' : ''}</span>{/if}
      </div>
      {#if !st?.playing}
        <p class="subtle small">The queue for the playing scene shows here.</p>
      {:else if st.upNext.length === 0}
        <p class="subtle small">Nothing after this track.</p>
      {:else}
        <ol class="rows queue">
          {#each st.upNext as t, i (i)}
            <li>
              <span class="num tiny subtle">{i + 1}</span>
              <span class="row-main static">
                <span class="row-title">{t.title}</span>
                <span class="tiny subtle">{t.artist}{t.durationMs ? `${t.artist ? ' · ' : ''}${fmtMs(t.durationMs)}` : ''}</span>
              </span>
            </li>
          {/each}
        </ol>
      {/if}
    </div>
  {:else if view === 'music'}
  <div class="music-grid">
    <div class="card pane">
      <div class="pane-head">
        <h2 class="section-title">Library</h2>
        <div class="pane-tools">
          <button class="btn btn-sm" onclick={() => (newFolderOpen = !newFolderOpen)} aria-expanded={newFolderOpen}>+ Folder</button>
          <button class="btn-primary btn-sm" onclick={() => (uploadOpen = true)}>Upload</button>
        </div>
      </div>
      {#if newFolderOpen}
        <form class="new-folder" onsubmit={(e) => { e.preventDefault(); createFolder(); }}>
          <input class="input" bind:value={newFolderName} placeholder={selectedFolder ? `New folder in ${folderName(selectedFolder)}` : 'New folder'} aria-label="New folder name" />
          <button class="btn btn-sm" type="submit" disabled={!newFolderName.trim()}>Create</button>
        </form>
      {/if}
      <ul class="rows folders">
        <li class:selected={selectedFolder === ''} class:current={st?.playing && st.folder === ''}>
          <button class="row-main" onclick={() => (selectedFolder = '')}>
            <span class="row-title">All music</span>
            <span class="tiny subtle">{tracks.length} track{tracks.length === 1 ? '' : 's'}{libraryMs ? ` · ${fmtMs(libraryMs)}` : ''}</span>
          </button>
          <button class="icon-btn play" onclick={() => run('f:', () => dj.play({ folder: '' }))}
            disabled={!connected || tracks.length === 0 || pending === 'f:'} aria-label="Play all music">&#9654;</button>
        </li>
        {#each folders as f (f)}
          {@const fs = stats.get(f)}
          <li class:selected={selectedFolder === f} class:current={st?.playing && st.folder === f} style:--depth={folderDepth(f)}>
            <button class="row-main" onclick={() => (selectedFolder = f)}>
              <span class="row-title">{folderName(f)}</span>
              <span class="tiny subtle">{fs?.count ?? 0} track{fs?.count === 1 ? '' : 's'}{fs?.ms ? ` · ${fmtMs(fs.ms)}` : ''}</span>
            </button>
            {#if selectedFolder === f}
              <button class="btn btn-sm" class:btn-danger={armed === `f:${f}`} onclick={() => deleteFolder(f)}
                aria-label="Delete folder {folderName(f)}">{armed === `f:${f}` ? 'Delete all?' : 'Delete'}</button>
            {/if}
            <button class="icon-btn play" onclick={() => run(`f:${f}`, () => dj.play({ folder: f }))}
              disabled={!connected || !fs?.count || pending === `f:${f}`} aria-label="Play {folderName(f)}">&#9654;</button>
          </li>
        {/each}
      </ul>
      {#if folders.length === 0}
        <p class="subtle small">Each folder is a playlist, and sub-folders are part of their parent. Upload music or drop a folder anywhere on this page.</p>
      {/if}
    </div>

  <section class="card pane">
    <div class="pane-head">
      <h2 class="section-title">
        {selectedFolder ? folderName(selectedFolder) : 'All tracks'} <span class="subtle">({visibleTracks.length})</span>
      </h2>
      <div class="pane-tools">
        <input class="input" type="search" bind:value={trackQuery} placeholder="Search" aria-label="Search tracks" />
        <button class="btn btn-sm" onclick={rescan} disabled={scanning}
          title="Pick up files copied into the music folder outside this page">{scanning ? 'Scanning…' : 'Rescan'}</button>
      </div>
    </div>
    {#if tracks.length === 0}
      <p class="subtle small">No tracks yet. Press Upload, or drop audio files or a folder anywhere on this page.</p>
    {:else if visibleTracks.length === 0}
      <p class="subtle small">{trackQuery ? 'No matches.' : 'This folder is empty. Drop music here to upload into it.'}</p>
    {:else}
      <ul class="rows tracks">
        {#each visibleTracks as t (t.id)}
          <li class:missing={t.missing} class:current={st?.track?.id === t.id}>
            <span class="row-main static">
              <span class="row-title">{trackTitle(t)}</span>
              <span class="tiny subtle">{[t.artist, shownPath(t)].filter(Boolean).join(' · ')}</span>
            </span>
            <span class="tiny subtle dur">{t.durationMs ? fmtMs(t.durationMs) : ''}</span>
            <button class="icon-btn del" class:armed={armed === `t:${t.id}`} onclick={() => deleteTrack(t)}
              aria-label={armed === `t:${t.id}` ? `Confirm delete ${trackTitle(t)}` : `Delete ${trackTitle(t)}`}
              title={armed === `t:${t.id}` ? 'Tap again to delete the file' : 'Delete'}>{armed === `t:${t.id}` ? 'Delete?' : '✕'}</button>
            <button class="icon-btn play" onclick={() => run(`t:${t.id}`, () => dj.play({ trackId: t.id }))}
              disabled={!connected || t.missing || pending === `t:${t.id}`} aria-label="Play {trackTitle(t)}">&#9654;</button>
          </li>
        {/each}
      </ul>
    {/if}
  </section>
  </div>
  {:else}
  <SoundboardPanel bind:this={soundboard} {st} channelName={$botStatus.channelName} />
  {/if}
</main>

<NowPlayingBar {st} />

<svelte:window ondragenter={onDragEnter} ondragleave={onDragLeave} ondragover={onDragOver} ondrop={onDrop} />

{#if dragDepth > 0 && !uploadOpen}
  <div class="drop-overlay" aria-hidden="true">
    <div>
      {#if view === 'soundboard'}
        <strong>Drop to add sound effects</strong>
        <span>They go into the open tab</span>
      {:else}
        <strong>Drop to upload music</strong>
        <span>You'll choose the folder next</span>
      {/if}
    </div>
  </div>
{/if}

{#if uploading && !uploadOpen}
  <button class="upload-chip" onclick={() => (uploadOpen = true)}>Uploading {Math.round(uploadProgress * 100)}%</button>
{/if}

<SceneEditor bind:open={sceneEditorOpen} scene={editingScene} {folders} {tracks} onsaved={load} />
<UploadDialog bind:this={uploadDialog} bind:open={uploadOpen} bind:uploading bind:progress={uploadProgress}
  {folders} defaultFolder={selectedFolder} />

<style>
  .dj { max-width: 1240px; width: 100%; margin: 0 auto; padding: 36px 32px 160px; display: flex; flex-direction: column; gap: 28px; }
  .heading { display: flex; justify-content: space-between; align-items: flex-end; gap: 20px; flex-wrap: wrap; }
  .heading .eyebrow { margin-bottom: 10px; }
  .conn { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .notice { padding: 14px 18px; font-size: 14px; color: var(--text-2); }
  .notice.warn { border-color: var(--warning); color: var(--warning); background: var(--warning-soft); }

  .section-head, .pane-head { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 14px; flex-wrap: wrap; }
  .pane-tools { display: flex; gap: 8px; align-items: center; }
  .pane-tools .input { width: 220px; }

  /* Tablet-first: tiles are large touch targets. */
  .scenes { display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 14px; }
  .scene-wrap { position: relative; }
  .scene {
    --scene-color: var(--accent);
    width: 100%; min-height: 128px; padding: 18px 18px 16px; border-radius: var(--radius-lg); text-align: left;
    display: flex; flex-direction: column; justify-content: space-between; gap: 12px;
    background: var(--surface); border: 1px solid var(--border-strong); color: var(--text);
    border-top: 5px solid var(--scene-color);
    transition: background 0.15s, border-color 0.15s, transform 0.08s;
  }
  .scene:hover { background: var(--surface-hover); }
  .scene:active { transform: scale(0.985); }
  .scene.active {
    background: color-mix(in srgb, var(--scene-color) 16%, var(--surface));
    border-color: var(--scene-color);
    box-shadow: 0 0 0 1px var(--scene-color), 0 6px 24px color-mix(in srgb, var(--scene-color) 25%, transparent);
  }
  .scene.active.paused { box-shadow: 0 0 0 1px var(--scene-color); }
  .scene:disabled { opacity: 0.7; }
  .scene.empty { border: 1px dashed var(--border-strong); color: var(--text-2); grid-column: span 2; }
  /* Name pinned to the top, count and flags to the bottom, so every tile
     lines up the same whatever its contents. */
  .scene-name { font-size: 18px; font-weight: 700; line-height: 1.3; padding-right: 36px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .scene-foot { display: flex; align-items: center; justify-content: space-between; gap: 8px; min-height: 20px; }
  .scene-meta { font-size: 13px; color: var(--text-2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .scene-flags { display: flex; gap: 8px; color: var(--text-2); flex-shrink: 0; }
  .scene.active .scene-flags { color: var(--scene-color); }
  .scene-flags svg { width: 18px; height: 18px; fill: none; stroke: currentColor; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
  .scene-wrap.arranging .scene { border-style: dashed; cursor: default; }
  .scene-wrap.arranging .scene-name { padding-right: 100px; }
  .arrange { position: absolute; top: 10px; right: 8px; display: flex; gap: 6px; }
  .arrange .icon-btn { width: var(--touch); height: var(--touch); background: var(--surface-raised); border: 1px solid var(--border-strong); border-radius: var(--radius); }
  .conn .select { min-height: var(--control-h-sm); max-width: 220px; }
  .scene-edit { position: absolute; top: 10px; right: 8px; width: var(--touch); height: var(--touch); font-size: 20px; color: var(--text-2); }

  .music-grid { display: grid; grid-template-columns: 360px minmax(0, 1fr); gap: 18px; align-items: start; }

  /* Same look as the campaign sub-navigation. */
  .subnav {
    position: sticky; top: var(--header-h); z-index: 30;
    height: var(--subnav-h); display: flex; align-items: stretch;
    padding: 0 24px; background: #17191c; border-bottom: 1px solid var(--border-color);
  }
  .subnav nav { display: flex; align-items: stretch; gap: 4px; overflow-x: auto; scrollbar-width: none; }
  .subnav nav::-webkit-scrollbar { display: none; }
  .subnav a {
    display: flex; align-items: center; padding: 0 14px; white-space: nowrap;
    color: var(--text-2); font-weight: 600; font-size: 14px;
    border-bottom: 2px solid transparent; margin-bottom: -1px;
  }
  .subnav a:hover { color: var(--text); text-decoration: none; }
  .subnav a.active { color: var(--text); border-bottom-color: var(--accent); }
  .pane { padding: 20px 22px; display: flex; flex-direction: column; min-height: 160px; }

  .rows { list-style: none; display: flex; flex-direction: column; }
  .rows li { display: flex; align-items: center; gap: 10px; border-bottom: 1px solid var(--border-color); min-height: 56px; }
  .rows li:last-child { border-bottom: 0; }
  .rows li.current .row-title { color: var(--accent-text); }
  .rows li.missing .row-title { color: var(--text-3); text-decoration: line-through; }
  .row-main { flex: 1; min-width: 0; display: flex; flex-direction: column; align-items: flex-start; text-align: left; padding: 8px 0; background: none; border: 0; color: inherit; }
  button.row-main:hover .row-title { text-decoration: underline; }
  .row-title { font-size: 15px; font-weight: 600; max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .row-main .tiny { max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .play { width: var(--touch); height: var(--touch); }
  .num { width: 18px; text-align: right; }
  .dur { font-variant-numeric: tabular-nums; }
  .tracks { max-height: 640px; overflow-y: auto; }
  .del { width: var(--touch); height: var(--touch); color: var(--text-3); font-size: 14px; flex-shrink: 0; }
  .del.armed { width: auto; padding: 0 10px; color: var(--danger); font-weight: 600; }

  .new-folder { display: flex; gap: 8px; margin-bottom: 10px; }
  .new-folder .input { flex: 1; }
  .folders { max-height: 600px; overflow-y: auto; }
  .folders li { padding-left: calc(var(--depth, 0) * 18px); }
  .folders li.selected { background: var(--surface-raised); border-radius: var(--radius); }
  .folders li .row-main { padding-left: 10px; }

  .drop-overlay {
    position: fixed; inset: 0; z-index: 50; pointer-events: none;
    display: grid; place-items: center;
    background: color-mix(in srgb, var(--canvas) 70%, transparent);
    border: 3px dashed var(--accent);
  }
  .drop-overlay div { display: flex; flex-direction: column; align-items: center; gap: 6px; padding: 28px 36px; border-radius: var(--radius-lg); background: var(--surface); border: 1px solid var(--border-strong); }
  .drop-overlay strong { font-size: 20px; }
  .drop-overlay span { color: var(--text-2); font-size: 14px; }
  .upload-chip {
    position: fixed; right: 20px; bottom: 120px; z-index: 40;
    padding: 10px 16px; border-radius: 999px; background: var(--surface); border: 1px solid var(--accent); color: var(--text);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25); font-weight: 600;
  }

  @media (max-width: 760px) {
    .dj { padding: 24px 16px 210px; }
    .music-grid { grid-template-columns: 1fr; }
    .subnav { padding: 0 8px; }
    .subnav a { padding: 0 10px; }
    .scenes { grid-template-columns: repeat(2, 1fr); }
    .scene.empty { grid-column: span 2; }
    .pane-tools { width: 100%; }
    .pane-tools .input { flex: 1; width: auto; }
  }
</style>
