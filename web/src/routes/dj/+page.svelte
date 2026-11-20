<script lang="ts">
  // DJ Studio, scenes first. The server owns playback (internal/dj); this
  // page is a remote control. Live state arrives over the WebSocket
  // (djState); the library (scenes/playlists/tracks) is refetched whenever any
  // client changes it (djLibraryVersion).
  //
  // Layout intent (tablet at the table): big scene tiles you can hit without
  // looking, a persistent now-playing bar, playlists and tracks below.
  import { onMount } from 'svelte';
  import { dj, channels as channelsApi, music } from '$lib/api';
  import { djState, djLibraryVersion } from '$lib/stores/dj';
  import { botStatus } from '$lib/stores/ws';
  import { addToast } from '$lib/stores/toast';
  import type { DjPlaylist, DjScene, DjTrack, VoiceChannel } from '$lib/types';
  import NowPlayingBar from '$lib/components/dj/NowPlayingBar.svelte';
  import SceneEditor from '$lib/components/dj/SceneEditor.svelte';
  import PlaylistEditor from '$lib/components/dj/PlaylistEditor.svelte';
  import { fmtMs, trackTitle } from '$lib/components/dj/format';

  let tracks = $state<DjTrack[]>([]);
  let playlists = $state<DjPlaylist[]>([]);
  let scenes = $state<DjScene[]>([]);
  let loaded = $state(false);

  let sceneEditorOpen = $state(false);
  let editingScene = $state<DjScene | null>(null);
  let playlistEditorOpen = $state(false);
  let editingPlaylist = $state<DjPlaylist | null>(null);

  let pending = $state<string | null>(null); // key of the tile/button awaiting the server
  let trackQuery = $state('');
  let scanning = $state(false);

  async function load() {
    try {
      [tracks, playlists, scenes] = await Promise.all([dj.tracks(), dj.playlists.list(), dj.scenes.list()]);
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
  const playlistById = $derived(new Map(playlists.map((p) => [p.id, p])));
  const trackById = $derived(new Map(tracks.map((t) => [t.id, t])));

  function playlistMs(p: DjPlaylist) {
    return p.trackIds.reduce((s, id) => s + (trackById.get(id)?.durationMs ?? 0), 0);
  }

  const visibleTracks = $derived.by(() => {
    const q = trackQuery.trim().toLowerCase();
    if (!q) return tracks;
    return tracks.filter((t) => `${t.title} ${t.artist} ${t.path}`.toLowerCase().includes(q));
  });

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
    if (!s.playlistId) {
      editScene(s);
      return;
    }
    run(`scene:${s.id}`, () => dj.play({ sceneId: s.id }));
  }

  function editScene(s: DjScene | null) {
    editingScene = s;
    sceneEditorOpen = true;
  }
  function editPlaylist(p: DjPlaylist | null) {
    editingPlaylist = p;
    playlistEditorOpen = true;
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
      <button class="btn btn-sm btn-ghost" onclick={() => run('tone', () => dj.tone(10))} disabled={!connected || pending === 'tone'}
        title="Play a 10 second 440 Hz tone to check the bot can be heard">Test tone</button>
    </div>
  </div>

  {#if st?.error}
    <div class="notice card warn" role="status">{st.error}</div>
  {/if}

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
          <span class="scene-meta">A scene is a playlist plus how it plays: tavern, travel, combat, boss.</span>
        </button>
      </div>
    {:else}
      <div class="scenes">
        {#each scenes as s, i (s.id)}
          {@const pl = s.playlistId ? playlistById.get(s.playlistId) : undefined}
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
              <span class="scene-name">{s.name}</span>
              <span class="scene-meta">
                {#if active}
                  {st?.paused ? 'Paused' : 'Playing'}{st?.track ? ` · ${st.track.title}` : ''}
                {:else if pl}
                  {pl.name} · {pl.trackIds.length} track{pl.trackIds.length === 1 ? '' : 's'}{s.shuffle ? ' · shuffle' : ''}
                {:else}
                  No playlist. Tap to set one.
                {/if}
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

  <section class="two-col">
    <div class="card pane">
      <div class="pane-head">
        <h2 class="section-title">Playlists</h2>
        <button class="btn btn-sm" onclick={() => editPlaylist(null)}>+ New playlist</button>
      </div>
      {#if playlists.length === 0}
        <p class="subtle small">Playlists group tracks from the server's music folder. A scene points at a playlist.</p>
      {:else}
        <ul class="rows">
          {#each playlists as p (p.id)}
            <li class:current={st?.playing && st.playlistId === p.id}>
              <button class="row-main" onclick={() => editPlaylist(p)}>
                <span class="row-title">{p.name}</span>
                <span class="tiny subtle">{p.trackIds.length} track{p.trackIds.length === 1 ? '' : 's'}{playlistMs(p) ? ` · ${fmtMs(playlistMs(p))}` : ''}</span>
              </button>
              <button class="icon-btn play" onclick={() => run(`pl:${p.id}`, () => dj.play({ playlistId: p.id }))}
                disabled={!connected || p.trackIds.length === 0 || pending === `pl:${p.id}`} aria-label="Play {p.name}">&#9654;</button>
            </li>
          {/each}
        </ul>
      {/if}
    </div>

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
  </section>

  <section class="card pane">
    <div class="pane-head">
      <h2 class="section-title">Tracks <span class="subtle">({tracks.length})</span></h2>
      <div class="pane-tools">
        <input class="input" type="search" bind:value={trackQuery} placeholder="Search" aria-label="Search tracks" />
        <button class="btn btn-sm" onclick={rescan} disabled={scanning}>{scanning ? 'Scanning…' : 'Rescan'}</button>
      </div>
    </div>
    {#if tracks.length === 0}
      <p class="subtle small">No tracks yet. Copy audio files (mp3, ogg, flac, wav, m4a) into the <code>music</code> folder next to the bot, then press Rescan.</p>
    {:else}
      <ul class="rows tracks">
        {#each visibleTracks as t (t.id)}
          <li class:missing={t.missing} class:current={st?.track?.id === t.id}>
            <span class="row-main static">
              <span class="row-title">{trackTitle(t)}</span>
              <span class="tiny subtle">{t.missing ? 'File missing' : [t.artist, t.path].filter(Boolean).join(' · ')}</span>
            </span>
            <span class="tiny subtle dur">{t.durationMs ? fmtMs(t.durationMs) : ''}</span>
            <button class="icon-btn play" onclick={() => run(`t:${t.id}`, () => dj.play({ trackId: t.id }))}
              disabled={!connected || t.missing || pending === `t:${t.id}`} aria-label="Play {trackTitle(t)}">&#9654;</button>
          </li>
        {/each}
      </ul>
    {/if}
  </section>
</main>

<NowPlayingBar {st} />

<SceneEditor bind:open={sceneEditorOpen} scene={editingScene} {playlists} onsaved={load} />
<PlaylistEditor bind:open={playlistEditorOpen} playlist={editingPlaylist} {tracks} onsaved={load} />

<style>
  .dj { max-width: 1240px; width: 100%; margin: 0 auto; padding: 36px 32px 160px; display: flex; flex-direction: column; gap: 28px; }
  .heading { display: flex; justify-content: space-between; align-items: flex-end; gap: 20px; flex-wrap: wrap; }
  .heading .eyebrow { margin-bottom: 10px; }
  .conn { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .notice { padding: 14px 18px; font-size: 14px; color: var(--text-2); }
  .notice.warn { border-color: var(--warning); color: var(--warning); background: var(--warning-soft); }
  code { font-family: var(--font-mono); font-size: 13px; background: var(--surface-raised); padding: 1px 6px; border-radius: 4px; }

  .section-head, .pane-head { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 14px; flex-wrap: wrap; }
  .pane-tools { display: flex; gap: 8px; align-items: center; }
  .pane-tools .input { width: 220px; }

  /* Tablet-first: tiles are large touch targets. */
  .scenes { display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 14px; }
  .scene-wrap { position: relative; }
  .scene {
    --scene-color: var(--accent);
    width: 100%; min-height: 128px; padding: 18px 18px 16px; border-radius: var(--radius-lg); text-align: left;
    display: flex; flex-direction: column; justify-content: flex-end; gap: 4px;
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
  .scene-name { font-size: 18px; font-weight: 700; padding-right: 36px; }
  .scene-meta { font-size: 13px; color: var(--text-2); overflow: hidden; text-overflow: ellipsis; display: -webkit-box; -webkit-line-clamp: 2; line-clamp: 2; -webkit-box-orient: vertical; }
  .scene-wrap.arranging .scene { border-style: dashed; cursor: default; }
  .scene-wrap.arranging .scene-name { padding-right: 100px; }
  .arrange { position: absolute; top: 10px; right: 8px; display: flex; gap: 6px; }
  .arrange .icon-btn { width: var(--touch); height: var(--touch); background: var(--surface-raised); border: 1px solid var(--border-strong); border-radius: var(--radius); }
  .conn .select { min-height: var(--control-h-sm); max-width: 220px; }
  .scene-edit { position: absolute; top: 10px; right: 8px; width: var(--touch); height: var(--touch); font-size: 20px; color: var(--text-2); }

  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
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
  .tracks { max-height: 520px; overflow-y: auto; }

  @media (max-width: 760px) {
    .dj { padding: 24px 16px 210px; }
    .two-col { grid-template-columns: 1fr; }
    .scenes { grid-template-columns: repeat(2, 1fr); }
    .scene.empty { grid-column: span 2; }
    .pane-tools { width: 100%; }
    .pane-tools .input { flex: 1; width: auto; }
  }
</style>
