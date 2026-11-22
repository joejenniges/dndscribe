<script lang="ts">
  // DJ Studio: one page for everything live. The server owns playback
  // (internal/dj, docs/dj.md); this page sends commands over the WebSocket
  // and renders the server's state, so the desktop and a tablet can both
  // drive it at the same time and always agree.
  //
  // Two presentation modes, chosen per device and never shared:
  //   Studio      - library | decks, mixer, queue | scenes + soundboard,
  //                 with resizable side panes.
  //   Performance - big scene and effect triggers, both decks, crossfader
  //                 and master; the library is a drawer. The tablet default.
  // Switching modes only changes the layout: no audio, queue or socket
  // change.
  import { onMount } from 'svelte';
  import { dj } from '$lib/api';
  import { djState, djLibraryVersion, presence } from '$lib/stores/dj';
  import { wsConnected } from '$lib/stores/ws';
  import { addToast } from '$lib/stores/toast';
  import { deviceKind, local } from '$lib/dj/live';
  import type { DjScene, DjTrack } from '$lib/types';
  import DeckWave from '$lib/components/dj/DeckWave.svelte';
  import DeckPanel from '$lib/components/dj/DeckPanel.svelte';
  import Mixer from '$lib/components/dj/Mixer.svelte';
  import QueuePanel from '$lib/components/dj/QueuePanel.svelte';
  import LibraryPanel from '$lib/components/dj/LibraryPanel.svelte';
  import ScenesPanel from '$lib/components/dj/ScenesPanel.svelte';
  import SoundboardPanel from '$lib/components/dj/SoundboardPanel.svelte';
  import VoiceControl from '$lib/components/dj/VoiceControl.svelte';
  import DjStatusBar from '$lib/components/dj/DjStatusBar.svelte';
  import UploadDialog from '$lib/components/dj/UploadDialog.svelte';
  import { fromDrop } from '$lib/components/dj/library';
  import Modal from '$lib/components/ui/Modal.svelte';
  import Toggle from '$lib/components/ui/Toggle.svelte';
  import { installHotkeys, selectedDeck, hotkeysOn, HOTKEYS } from '$lib/dj/hotkeys';

  // ---- Shared data ----
  let tracks = $state<DjTrack[]>([]);
  let folders = $state<string[]>([]);
  let scenes = $state<DjScene[]>([]);
  let loaded = $state(false);
  let selectedFolder = $state('');

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

  // Runs on mount and again whenever any device changes the library.
  $effect(() => {
    $djLibraryVersion;
    load();
  });

  onMount(async () => {
    if (!$djState?.decks) {
      try {
        const s = await dj.state();
        // The WebSocket may have delivered a newer snapshot meanwhile;
        // only fill in if this one is newer.
        djState.update((cur) => (cur?.decks && cur.rev >= s.rev ? cur : { ...s, receivedAt: Date.now() }));
      } catch {
        /* the WebSocket init fills it */
      }
    }
  });

  const st = $derived($djState?.decks ? $djState : null);
  const live = $derived($wsConnected);

  // ---- Mode (per device) ----
  type Mode = 'studio' | 'performance';
  let mode = $state<Mode>(
    (local.get('dj.mode') as Mode | null) ?? (deviceKind() === 'desktop' ? 'studio' : 'performance'),
  );
  function setMode(m: Mode) {
    mode = m;
    local.set('dj.mode', m);
  }

  // ---- Presence ----
  const devices = $derived.by(() => {
    const kinds = $presence.map((c) => c.device);
    if (kinds.length <= 1) return '';
    const counts = new Map<string, number>();
    for (const k of kinds) counts.set(k, (counts.get(k) ?? 0) + 1);
    return [...counts].map(([k, n]) => (n > 1 ? `${n} ${k}s` : k)).join(' + ');
  });

  // ---- Studio panes: widths and collapse, per device ----
  const PANES_KEY = 'dj.panes';
  const MIN_LEFT = 260, MAX_LEFT = 560, MIN_RIGHT = 300, MAX_RIGHT = 680, MIN_CENTER = 540;
  const saved = (() => {
    try {
      return JSON.parse(local.get(PANES_KEY) ?? '{}');
    } catch {
      return {};
    }
  })();
  let leftW = $state<number>(saved.leftW ?? 420);
  let rightW = $state<number>(saved.rightW ?? 400);
  let leftOpen = $state<boolean>(saved.leftOpen ?? true);
  let rightOpen = $state<boolean>(saved.rightOpen ?? true);
  let workW = $state(1400);

  function savePanes() {
    local.set(PANES_KEY, JSON.stringify({ leftW, rightW, leftOpen, rightOpen, consoleH, queueCollapsed }));
  }

  // Below ~1340px there isn't room for three panes and a usable deck
  // console: the library becomes a drawer over the page (closed until asked
  // for). Below ~860px everything stacks.
  const narrow = $derived(workW < 1340);
  let drawerOpen = $state(false);
  const stacked = $derived(workW < 860);

  // Widths actually used: clamped so the centre never drops under its
  // minimum, whatever was saved on a bigger screen.
  const eff = $derived.by(() => {
    const pad = 24; // .studio padding
    let l = leftOpen && !narrow ? Math.max(MIN_LEFT, Math.min(MAX_LEFT, leftW)) : 0;
    let r = rightOpen ? Math.max(MIN_RIGHT, Math.min(MAX_RIGHT, rightW)) : 0;
    const gutters = (narrow ? 0 : leftOpen ? 14 : 30) + (rightOpen ? 14 : 30);
    const over = l + r + gutters + pad + MIN_CENTER - workW;
    if (over > 0) {
      const takeR = Math.min(over, Math.max(0, r - MIN_RIGHT));
      r -= takeR;
      l -= Math.min(over - takeR, Math.max(0, l - MIN_LEFT));
    }
    return { l, r };
  });

  let drag: { side: 'l' | 'r'; startX: number; start: number } | null = null;
  function gutDown(side: 'l' | 'r', e: PointerEvent) {
    if (e.button !== 0) return;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    drag = { side, startX: e.clientX, start: side === 'l' ? eff.l : eff.r };
  }
  function gutMove(e: PointerEvent) {
    if (!drag) return;
    const dx = e.clientX - drag.startX;
    if (drag.side === 'l') leftW = Math.max(MIN_LEFT, Math.min(MAX_LEFT, drag.start + dx));
    else rightW = Math.max(MIN_RIGHT, Math.min(MAX_RIGHT, drag.start - dx));
  }
  function gutUp() {
    if (!drag) return;
    drag = null;
    savePanes();
  }
  function gutKey(side: 'l' | 'r', e: KeyboardEvent) {
    const step = e.shiftKey ? 64 : 16;
    let d = 0;
    if (e.key === 'ArrowLeft') d = -step;
    else if (e.key === 'ArrowRight') d = step;
    else if (e.key === 'Home') d = -9999;
    else if (e.key === 'End') d = 9999;
    else return;
    e.preventDefault();
    if (side === 'l') leftW = Math.max(MIN_LEFT, Math.min(MAX_LEFT, eff.l + d));
    else rightW = Math.max(MIN_RIGHT, Math.min(MAX_RIGHT, eff.r - d));
    savePanes();
  }
  // ---- Centre column: console height and queue collapse, per device ----
  // The decks and mixer sit at their natural height (or taller if dragged
  // taller with the splitter); the queue takes the rest. Collapsed, the queue
  // is a single bar and the console takes the room. The queue starts
  // collapsed when it's empty (unless this device chose otherwise), since an
  // empty list is a lot of wasted height.
  const MIN_QUEUE = 140;
  let consoleH = $state<number | null>(saved.consoleH ?? null);
  let consoleEl: HTMLDivElement | undefined = $state();
  let queueCollapsed = $state<boolean>(saved.queueCollapsed ?? false);
  let queueSeeded = saved.queueCollapsed !== undefined;
  $effect(() => {
    if (queueSeeded || !st) return;
    queueSeeded = true;
    queueCollapsed = (st.queueTotal ?? 0) === 0;
  });
  let centerH = $state(800);
  let wavesH = $state(140);
  // Tallest the console may get while leaving the queue its minimum.
  const maxConsole = $derived(Math.max(200, centerH - wavesH - MIN_QUEUE - 36));
  function setConsole(h: number) {
    // CSS min-height: min-content keeps it from going under its content.
    consoleH = Math.max(160, Math.min(maxConsole, Math.round(h)));
  }
  let vdrag: { startY: number; start: number } | null = null;
  function vDown(e: PointerEvent) {
    if (e.button !== 0) return;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    vdrag = { startY: e.clientY, start: consoleEl?.offsetHeight ?? 300 };
  }
  function vMove(e: PointerEvent) {
    if (!vdrag) return;
    setConsole(vdrag.start + (e.clientY - vdrag.startY));
  }
  function vUp() {
    if (!vdrag) return;
    vdrag = null;
    savePanes();
  }
  function vKey(e: KeyboardEvent) {
    const step = e.shiftKey ? 80 : 20;
    let d = 0;
    if (e.key === 'ArrowUp') d = -step;
    else if (e.key === 'ArrowDown') d = step;
    else if (e.key === 'Home') {
      consoleH = null; // back to natural height
      e.preventDefault();
      savePanes();
      return;
    } else return;
    e.preventDefault();
    e.stopPropagation();
    setConsole((consoleEl?.offsetHeight ?? 300) + d);
    savePanes();
  }
  $effect(() => {
    queueCollapsed;
    savePanes();
  });

  function toggleLeft() {
    leftOpen = !leftOpen;
    savePanes();
  }
  function toggleRight() {
    rightOpen = !rightOpen;
    savePanes();
  }

  // Performance mode library drawer.
  let perfLib = $state(local.get('dj.perf.lib') === '1');
  function togglePerfLib() {
    perfLib = !perfLib;
    local.set('dj.perf.lib', perfLib ? '1' : '0');
  }

  // ---- Upload: dialog, plus drops anywhere on the page ----
  let uploadOpen = $state(false);
  let uploading = $state(false);
  let uploadProgress = $state(0);
  let uploadDialog: ReturnType<typeof UploadDialog> | undefined = $state();
  let soundboard: ReturnType<typeof SoundboardPanel> | undefined = $state();

  // dragenter/leave fire for every child element, so count them to know
  // when the drag has really left the window.
  let dragDepth = $state(0);
  let overSoundboard = $state(false);
  const hasFiles = (e: DragEvent) => !!e.dataTransfer?.types.includes('Files');
  const onBoard = (e: DragEvent) => !!(e.target as Element | null)?.closest?.('.sb');
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
    if (!hasFiles(e)) return;
    e.preventDefault();
    overSoundboard = onBoard(e);
  }
  async function onDrop(e: DragEvent) {
    if (!hasFiles(e) || !e.dataTransfer) return;
    e.preventDefault();
    dragDepth = 0;
    const board = onBoard(e);
    const files = await fromDrop(e.dataTransfer);
    // Dropped on the soundboard: effects for the open tab. Anywhere else:
    // music, and the dialog asks which folder.
    if (board && soundboard) {
      soundboard.addFiles(files);
      return;
    }
    uploadDialog?.addFiles(files);
    uploadOpen = true;
  }

  // ---- Keyboard shortcuts (lib/dj/hotkeys.ts) ----
  let keysOpen = $state(false);
  const hasKeyboard = deviceKind() === 'desktop';
  onMount(() =>
    installHotkeys({
      state: () => st,
      live: () => live,
      scenes: () => scenes,
      pad: (i) => soundboard?.triggerAt(i) ?? false,
      help: () => (keysOpen = true),
    }),
  );
  // Start with the on-air deck selected, so Space first pauses what's
  // playing rather than starting the other deck.
  let deckSeeded = false;
  $effect(() => {
    if (deckSeeded || !st) return;
    deckSeeded = true;
    if (st.onAir === 0 || st.onAir === 1) selectedDeck.set(st.onAir);
  });
</script>

<svelte:head><title>DJ Studio - dndscribe</title></svelte:head>

<div class="dj" class:perf={mode === 'performance'}>
  <header class="dj-head">
    <div class="title">
      <div class="eyebrow">Discord music</div>
      <h1>DJ Studio</h1>
    </div>
    <div class="modes" role="radiogroup" aria-label="Layout on this device">
      <button role="radio" aria-checked={mode === 'studio'} class:on={mode === 'studio'} onclick={() => setMode('studio')}>Studio</button>
      <button role="radio" aria-checked={mode === 'performance'} class:on={mode === 'performance'} onclick={() => setMode('performance')}>Performance</button>
    </div>
    <span class="sync" class:down={!live} role="status">
      <span class="dot" aria-hidden="true"></span>
      {#if live}Live{devices ? ` · ${devices}` : ''}{:else}Reconnecting…{/if}
    </span>
    {#if st?.error}
      <span class="err tiny" title={st.error}>{st.error}</span>
    {/if}
    <div class="head-right">
      <VoiceControl {st} {live} />
      {#if hasKeyboard}
        <button class="btn btn-sm keys-btn" onclick={() => (keysOpen = true)} title="Keyboard shortcuts (?)"
          aria-label="Keyboard shortcuts">
          <svg viewBox="0 0 24 24" aria-hidden="true"><rect x="2" y="6" width="20" height="12" rx="2" /><path d="M6 10h.01M10 10h.01M14 10h.01M18 10h.01M7 14h10" /></svg>
        </button>
      {/if}
      {#if mode === 'performance'}
        <button class="btn btn-sm" class:active={perfLib} onclick={togglePerfLib} aria-expanded={perfLib}>Library</button>
      {:else if narrow && !stacked}
        <button class="btn btn-sm" class:active={drawerOpen} onclick={() => (drawerOpen = !drawerOpen)} aria-expanded={drawerOpen}>Library</button>
      {/if}
    </div>
  </header>

  {#if !live}
    <div class="offline" role="alert">
      Lost the connection to the server. Showing the last known state; controls are off until it reconnects. Music on the server keeps playing.
    </div>
  {/if}

  <div class="work" bind:clientWidth={workW}>
    {#if mode === 'studio'}
      <div class="studio" class:narrow class:stacked
        style:grid-template-columns={stacked ? '1fr' : `${narrow ? 0 : eff.l}px ${narrow ? 0 : leftOpen ? 14 : 30}px minmax(0, 1fr) ${rightOpen ? 14 : 30}px ${eff.r}px`}>
        {#if leftOpen && (!narrow || stacked)}
          <aside class="pane left" aria-label="Library pane">
            <LibraryPanel {st} {live} {tracks} {folders} bind:selected={selectedFolder}
              onupload={() => (uploadOpen = true)} onchanged={load} />
          </aside>
        {:else if !stacked}
          <div></div>
        {/if}
        {#if !stacked}
          {#if leftOpen && !narrow}
            <!-- A focusable separator is the ARIA window-splitter pattern
                 (arrow keys resize); the lint rule doesn't know it. -->
            <div class="gcol">
              <!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions -->
              <div class="gutter" role="separator" aria-orientation="vertical" aria-label="Resize the library"
                aria-valuemin={MIN_LEFT} aria-valuemax={MAX_LEFT} aria-valuenow={eff.l} tabindex="0"
                onpointerdown={(e) => gutDown('l', e)} onpointermove={gutMove} onpointerup={gutUp} onpointercancel={gutUp}
                onkeydown={(e) => gutKey('l', e)}></div>
              <button class="toggle-pane" onclick={toggleLeft} aria-label="Hide the library" title="Hide the library">&#8249;</button>
            </div>
          {:else}
            <div class="rail">
              {#if !narrow}<button class="toggle-pane" onclick={toggleLeft} aria-label="Show the library" title="Show the library">&#8250;</button>{/if}
            </div>
          {/if}
        {/if}

        <section class="center" aria-label="Decks and mixer" bind:clientHeight={centerH}>
          <div class="waves card" bind:clientHeight={wavesH}>
            <DeckWave {st} deck={0} overviewOnly lane {live} />
            <DeckWave {st} deck={1} overviewOnly lane {live} />
          </div>
          <div class="console" bind:this={consoleEl} class:fill={queueCollapsed}
            style:height={!queueCollapsed && !stacked && consoleH ? `${Math.min(consoleH, maxConsole)}px` : null}>
            <DeckPanel {st} deck={0} {live} />
            <Mixer {st} {live} />
            <DeckPanel {st} deck={1} {live} />
          </div>
          {#if !queueCollapsed && !stacked}
            <!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions -->
            <div class="vgutter" role="separator" aria-orientation="horizontal" aria-label="Resize decks and queue (Home: natural height)"
              aria-valuemin={160} aria-valuemax={maxConsole} aria-valuenow={consoleH ?? undefined} tabindex="0"
              onpointerdown={vDown} onpointermove={vMove} onpointerup={vUp} onpointercancel={vUp} onkeydown={vKey}></div>
          {/if}
          <div class="qwrap" class:collapsed={queueCollapsed}>
            <QueuePanel {st} {live} bind:collapsed={queueCollapsed} />
          </div>
        </section>

        {#if !stacked}
          {#if rightOpen}
            <div class="gcol">
              <!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions -->
              <div class="gutter" role="separator" aria-orientation="vertical" aria-label="Resize scenes and soundboard"
                aria-valuemin={MIN_RIGHT} aria-valuemax={MAX_RIGHT} aria-valuenow={eff.r} tabindex="0"
                onpointerdown={(e) => gutDown('r', e)} onpointermove={gutMove} onpointerup={gutUp} onpointercancel={gutUp}
                onkeydown={(e) => gutKey('r', e)}></div>
              <button class="toggle-pane" onclick={toggleRight} aria-label="Hide scenes and soundboard" title="Hide scenes and soundboard">&#8250;</button>
            </div>
          {:else}
            <div class="rail">
              <button class="toggle-pane" onclick={toggleRight} aria-label="Show scenes and soundboard" title="Show scenes and soundboard">&#8249;</button>
            </div>
          {/if}
        {/if}
        {#if rightOpen || stacked}
          <aside class="pane right" aria-label="Scenes and soundboard">
            <div class="card scenes-box">
              <ScenesPanel {st} {live} bind:scenes {tracks} {folders} {loaded} onchanged={load} />
            </div>
            <div class="card sb-box">
              <SoundboardPanel bind:this={soundboard} {st} {live} />
            </div>
          </aside>
        {/if}
        {#if stacked && !leftOpen}
          <button class="btn show-lib" onclick={toggleLeft}>Show the library</button>
        {/if}
      </div>

      {#if narrow && !stacked && drawerOpen}
        <div class="drawer-scrim" role="presentation" onclick={() => (drawerOpen = false)}></div>
        <aside class="drawer" aria-label="Library">
          <LibraryPanel {st} {live} {tracks} {folders} bind:selected={selectedFolder}
            onupload={() => (uploadOpen = true)} onchanged={load} />
        </aside>
      {/if}
    {:else}
      <div class="perf-grid" class:with-lib={perfLib}>
        <div class="perf-main">
          <div class="pdecks">
            {#each [0, 1] as const as i (i)}
              <DeckPanel {st} deck={i} {live} compact />
            {/each}
          </div>
          <Mixer {st} {live} strip />
          <div class="card scenes-box">
            <ScenesPanel {st} {live} bind:scenes {tracks} {folders} {loaded} strip onchanged={load} />
          </div>
          <div class="card sb-box">
            <SoundboardPanel bind:this={soundboard} {st} {live} big />
          </div>
        </div>
        {#if perfLib}
          <aside class="perf-lib" aria-label="Library">
            <LibraryPanel {st} {live} {tracks} {folders} bind:selected={selectedFolder} compact />
          </aside>
        {/if}
      </div>
    {/if}
  </div>

  <DjStatusBar {st} {live} />
</div>

<svelte:window ondragenter={onDragEnter} ondragleave={onDragLeave} ondragover={onDragOver} ondrop={onDrop} />

{#if dragDepth > 0 && !uploadOpen}
  <div class="drop-overlay" aria-hidden="true">
    <div>
      {#if overSoundboard}
        <strong>Drop to add sound effects</strong>
        <span>They go into the open soundboard tab</span>
      {:else}
        <strong>Drop to upload music</strong>
        <span>You'll choose the folder next. Drop on the soundboard for effects.</span>
      {/if}
    </div>
  </div>
{/if}

{#if uploading && !uploadOpen}
  <button class="upload-chip" onclick={() => (uploadOpen = true)}>Uploading {Math.round(uploadProgress * 100)}%</button>
{/if}

<UploadDialog bind:this={uploadDialog} bind:open={uploadOpen} bind:uploading bind:progress={uploadProgress}
  {folders} defaultFolder={selectedFolder} />

<Modal bind:open={keysOpen} title="Keyboard shortcuts" width="520px">
  <Toggle bind:checked={$hotkeysOn} label="Shortcuts on this device"
    hint="Off gives Tab back to moving between controls" />
  <table class="keys" class:off={!$hotkeysOn}>
    <tbody>
      {#each HOTKEYS as k (k.keys)}
        <tr><td><kbd>{k.keys}</kbd></td><td>{k.what}</td></tr>
      {/each}
    </tbody>
  </table>
  <p class="tiny subtle">Ignored while typing in a field or with a dialog open. The selected deck has a ring around it; clicking a deck also selects it.</p>
  {#snippet footer()}
    <span style="flex: 1"></span>
    <button class="btn-primary" onclick={() => (keysOpen = false)}>Done</button>
  {/snippet}
</Modal>

<style>
  .keys-btn { min-width: 40px; padding: 0 8px; display: inline-grid; place-items: center; }
  .keys-btn svg { width: 20px; height: 20px; fill: none; stroke: currentColor; stroke-width: 1.8; stroke-linecap: round; }
  .keys { width: 100%; border-collapse: collapse; margin: 12px 0 8px; font-size: 14px; }
  .keys.off { opacity: 0.5; }
  .keys td { padding: 6px 4px; border-bottom: 1px solid var(--border-color); vertical-align: top; }
  .keys td:first-child { width: 132px; white-space: nowrap; }
  kbd { font: 600 12px/1 ui-monospace, monospace; padding: 4px 7px; border-radius: 5px; background: var(--surface-raised); border: 1px solid var(--border-strong); border-bottom-width: 2px; }

  /* The page fills the viewport under the app header; panes scroll inside. */
  .dj { height: calc(100dvh - var(--header-h)); display: flex; flex-direction: column; min-height: 0; }

  .dj-head { display: flex; align-items: center; gap: 14px 18px; padding: 10px 16px; flex-wrap: wrap; border-bottom: 1px solid var(--border-color); }
  .title .eyebrow { margin-bottom: 2px; }
  .perf .title .eyebrow { display: none; }
  .perf .dj-head { padding: 8px 16px; }
  .title h1 { font-size: 22px; font-weight: 800; margin: 0; letter-spacing: -0.3px; }
  .modes { display: flex; padding: 3px; gap: 3px; background: var(--surface-sunken); border: 1px solid var(--border-color); border-radius: var(--radius); }
  .modes button { min-height: 38px; padding: 0 14px; border: 0; border-radius: var(--radius-sm); background: none; color: var(--text-2); font-weight: 600; font-size: 13px; }
  .modes button.on { background: var(--accent-strong); color: #fff; }
  .sync { display: inline-flex; align-items: center; gap: 8px; font-size: 13px; font-weight: 600; color: var(--success); }
  .sync .dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; }
  .sync.down { color: var(--warning); }
  .err { color: var(--warning); max-width: 360px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .head-right { margin-left: auto; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .offline { padding: 8px 16px; background: var(--warning-soft); color: var(--warning); font-size: 13px; }

  .work { flex: 1; min-height: 0; position: relative; }

  /* ---- Studio ---- */
  .studio { display: grid; height: 100%; padding: 12px; min-height: 0; }
  .pane { position: relative; min-width: 0; min-height: 0; display: flex; flex-direction: column; gap: 12px; }
  .pane.right .scenes-box { flex: 0 1 auto; max-height: 44%; overflow-y: auto; padding: 12px 14px; }
  .pane.right .sb-box { flex: 1; min-height: 240px; padding: 12px 14px; overflow: hidden; }
  /* Scrolls only when the decks, mixer and the queue's minimum don't fit
     (short windows, or the mixer stacked under the decks). */
  .center { min-width: 0; min-height: 0; display: flex; flex-direction: column; gap: 12px; container-type: inline-size; overflow-y: auto; overscroll-behavior: contain; }
  .waves { padding: 10px 12px; display: flex; flex-direction: column; gap: 8px; flex-shrink: 0; }
  /* Decks and mixer take whatever height the queue leaves; their contents
     stay centred rather than stretching. */
  .console {
    display: grid; grid-template-columns: minmax(0, 1fr) minmax(240px, 0.8fr) minmax(0, 1fr); gap: 12px;
    flex: 0 0 auto; min-height: min-content; align-items: stretch;
  }
  .console.fill { flex: 1 1 auto; }
  .console > :global(.deck) { justify-content: center; }
  .vgutter {
    flex-shrink: 0; height: 10px; margin: -8px 0; cursor: row-resize; touch-action: none; position: relative; border-radius: 4px;
  }
  .vgutter::after { content: ''; position: absolute; left: 50%; top: 50%; width: 48px; height: 4px; transform: translate(-50%, -50%); border-radius: 2px; background: var(--border-strong); }
  .vgutter:hover::after, .vgutter:focus-visible::after { background: var(--accent); }
  .vgutter:focus-visible { outline: none; }
  .qwrap { flex: 1 1 0; display: flex; min-height: 140px; }
  .qwrap.collapsed { flex: 0 0 auto; min-height: 0; }
  .qwrap :global(.qp) { flex: 1; }
  @container (max-width: 820px) {
    .console { grid-template-columns: 1fr 1fr; }
    .console > :global(.mixer) { grid-column: 1 / -1; order: 3; }
  }
  .gcol, .rail { position: relative; }
  .gutter { position: absolute; inset: 0; cursor: col-resize; touch-action: none; }
  .gutter::after { content: ''; position: absolute; top: 0; bottom: 0; left: calc(50% - 1px); width: 2px; border-radius: 1px; background: transparent; transition: background 0.15s; }
  .gutter:hover::after, .gutter:focus-visible::after { background: var(--accent); }
  .gutter:focus-visible { outline: none; }
  .toggle-pane {
    position: absolute; top: 8px; left: 50%; transform: translateX(-50%); z-index: 2; width: 24px; height: 40px; padding: 0;
    border: 1px solid var(--border-strong); border-radius: var(--radius-sm); background: var(--surface-raised); color: var(--text-2); font-size: 18px;
  }
  .toggle-pane:hover { color: var(--text); }
  /* Stacked: the column grows with its content and the work area scrolls.
     (min-height: 0 on the panes, needed for inner scrolling side by side,
     would otherwise let the rows collapse on top of each other.) */
  .work:has(.stacked) { overflow-y: auto; }
  .studio.stacked { height: auto; min-height: 100%; grid-auto-rows: auto; gap: 12px; }
  .stacked .pane, .stacked .center { min-height: auto; }
  .stacked .pane.left { height: 70vh; order: 3; }
  .stacked .center { order: 1; }
  .stacked .pane.right { order: 2; }
  .stacked .pane.right .scenes-box { max-height: none; }
  .stacked .pane.right .sb-box { min-height: 420px; }
  .stacked .qwrap:not(.collapsed) :global(.qp) { min-height: 300px; }
  .show-lib { order: 4; }
  .drawer-scrim { position: absolute; inset: 0; background: rgba(0, 0, 0, 0.45); z-index: 20; }
  .drawer { position: absolute; top: 12px; bottom: 12px; left: 12px; width: min(380px, 90vw); z-index: 21; display: flex; box-shadow: var(--shadow); }
  .drawer :global(.lib) { flex: 1; }

  /* ---- Performance ---- */
  .perf-grid { height: 100%; display: grid; grid-template-columns: minmax(0, 1fr); gap: 12px; padding: 12px; min-height: 0; }
  .perf-grid.with-lib { grid-template-columns: minmax(0, 1fr) minmax(320px, 380px); }
  .perf-main { min-height: 0; overflow-y: auto; display: flex; flex-direction: column; gap: 10px; }
  .pdecks { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .perf-main .scenes-box { padding: 10px 12px; flex-shrink: 0; }
  .perf-main .sb-box { padding: 10px 12px; flex: 1 0 auto; min-height: 260px; }
  .perf-lib { min-height: 0; display: flex; }
  .perf-lib :global(.lib) { flex: 1; }
  @media (max-width: 1000px) {
    .perf-grid.with-lib { grid-template-columns: minmax(0, 1fr); }
    .perf-lib { position: absolute; top: 12px; right: 12px; bottom: 12px; width: min(380px, 92vw); z-index: 21; box-shadow: var(--shadow); }
  }
  @media (max-width: 700px) {
    .pdecks { grid-template-columns: 1fr; }
    .dj-head { padding: 8px 12px; }
    .title .eyebrow { display: none; }
  }

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
    position: fixed; right: 20px; bottom: 84px; z-index: 40;
    padding: 10px 16px; border-radius: 999px; background: var(--surface); border: 1px solid var(--accent); color: var(--text);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25); font-weight: 600;
  }
</style>
