<script lang="ts">
  // Soundboard: pads organised in tabs. Tapping a pad plays it into the
  // Discord channel through the server mixer, on top of the music; tapping a
  // playing pad restarts it from the beginning (the server does that, and
  // never stacks copies). Different pads overlap.
  //
  // Edit layout mode: drag pads to reorder them or onto a tab to move them
  // there. Every pad also gets arrow buttons, because HTML5 drag and drop
  // doesn't work on touch screens or from the keyboard.
  import { dj } from '$lib/api';
  import { djCmd } from '$lib/dj/control';
  import { addToast } from '$lib/stores/toast';
  import { djLibraryVersion } from '$lib/stores/dj';
  import type { DjState, SfxPad, SfxTab } from '$lib/types';
  import PadEditor from './PadEditor.svelte';
  import Fader from './Fader.svelte';
  import { SCENE_COLORS, fmtMs } from './format';
  import { isAudioName, AUDIO_EXTS, type PickedFile } from './library';

  let {
    st,
    live,
    big = false,
  }: {
    st: DjState | null;
    /** WebSocket up: commands can be sent. */
    live: boolean;
    /** Performance mode: larger pads. */
    big?: boolean;
  } = $props();

  let tabs = $state<SfxTab[]>([]);
  let pads = $state<SfxPad[]>([]);
  let loaded = $state(false);
  let activeTabId = $state<number | null>(null);
  let editMode = $state(false);

  const TAB_KEY = 'dj.sfx.tab';

  async function load() {
    try {
      const r = await dj.sfx.list();
      tabs = r.tabs;
      pads = r.pads;
      if (activeTabId === null || !tabs.some((t) => t.id === activeTabId)) {
        let saved: number | null = null;
        try {
          saved = Number(localStorage.getItem(TAB_KEY)) || null;
        } catch {
          /* storage unavailable */
        }
        activeTabId = tabs.find((t) => t.id === saved)?.id ?? tabs[0]?.id ?? null;
      }
    } catch (e) {
      addToast(`Couldn't load the soundboard: ${(e as Error).message}`, 'error');
    } finally {
      loaded = true;
    }
  }

  $effect(() => {
    $djLibraryVersion;
    load();
  });

  function selectTab(id: number) {
    activeTabId = id;
    try {
      localStorage.setItem(TAB_KEY, String(id));
    } catch {
      /* storage unavailable */
    }
  }

  const connected = $derived(!!st?.connected);
  const activeTab = $derived(tabs.find((t) => t.id === activeTabId) ?? null);
  const tabPads = $derived(
    pads.filter((p) => p.tabId === activeTabId).sort((a, b) => a.position - b.position || a.id - b.id),
  );
  const padCount = (tabId: number) => pads.filter((p) => p.tabId === tabId).length;
  const playing = $derived(new Map((st?.sfx ?? []).map((s) => [s.padId, s])));

  // Progress bars: interpolate between server pushes while anything plays.
  let now = $state(Date.now());
  $effect(() => {
    if (!st?.sfx?.length) return;
    const t = setInterval(() => (now = Date.now()), 100);
    return () => clearInterval(t);
  });
  function progress(padId: number): number {
    const s = playing.get(padId);
    if (!s || !s.durationMs) return 0;
    const elapsed = s.positionMs + (st?.receivedAt ? now - st.receivedAt : 0);
    return Math.min(1, Math.max(0, elapsed / s.durationMs));
  }
  function elapsedMs(padId: number): number {
    const s = playing.get(padId);
    return s ? Math.min(s.durationMs, s.positionMs + (st?.receivedAt ? now - st.receivedAt : 0)) : 0;
  }

  const padColor = (p: SfxPad) => p.color || SCENE_COLORS[p.id % SCENE_COLORS.length];

  // ---- Playback ----
  // Each tap is a new command with a new id, so the server restarts the pad;
  // a network retry of the same tap (same id) is ignored.
  async function tap(p: SfxPad) {
    if (editMode) {
      openEditor(p);
      return;
    }
    try {
      await djCmd('sfx.trigger', { padId: p.id });
    } catch (e) {
      addToast((e as Error).message, 'error');
    }
  }

  async function stopAll() {
    try {
      await djCmd('sfx.stop');
    } catch (e) {
      addToast((e as Error).message, 'error');
    }
  }

  // ---- Pad editor ----
  let editorOpen = $state(false);
  let editingPad = $state<SfxPad | null>(null);
  function openEditor(p: SfxPad) {
    editingPad = p;
    editorOpen = true;
  }

  // ---- Tabs ----
  let addingTab = $state(false);
  let newTabTitle = $state('');
  let tabTitleEdit = $state('');
  let confirmTabDelete = $state(false);

  $effect(() => {
    tabTitleEdit = activeTab?.title ?? '';
    confirmTabDelete = false;
  });

  async function addTab() {
    const title = newTabTitle.trim();
    if (!title) return;
    try {
      const t = await dj.sfx.tabs.create(title);
      newTabTitle = '';
      addingTab = false;
      await load();
      selectTab(t.id);
    } catch (e) {
      addToast(`Couldn't add tab: ${(e as Error).message}`, 'error');
    }
  }

  async function renameTab() {
    const title = tabTitleEdit.trim();
    if (!activeTab || !title || title === activeTab.title) return;
    try {
      await dj.sfx.tabs.rename(activeTab.id, title);
    } catch (e) {
      addToast(`Couldn't rename: ${(e as Error).message}`, 'error');
    }
  }

  async function moveTab(d: number) {
    const i = tabs.findIndex((t) => t.id === activeTabId);
    const j = i + d;
    if (i < 0 || j < 0 || j >= tabs.length) return;
    const next = [...tabs];
    [next[i], next[j]] = [next[j], next[i]];
    tabs = next;
    try {
      await dj.sfx.tabs.order(next.map((t) => t.id));
    } catch (e) {
      addToast(`Couldn't save order: ${(e as Error).message}`, 'error');
      load();
    }
  }

  async function deleteTab() {
    if (!activeTab) return;
    if (!confirmTabDelete && padCount(activeTab.id) > 0) {
      confirmTabDelete = true;
      return;
    }
    try {
      await dj.sfx.tabs.delete(activeTab.id);
      activeTabId = null;
    } catch (e) {
      addToast(`Couldn't delete tab: ${(e as Error).message}`, 'error');
    }
  }

  // ---- Pad layout ----

  async function arrange(tabId: number, ids: number[]) {
    // Optimistic: reflect the new layout now, then persist.
    const pos = new Map(ids.map((id, i) => [id, i]));
    pads = pads.map((p) => (pos.has(p.id) ? { ...p, tabId, position: pos.get(p.id)! } : p));
    try {
      await dj.sfx.tabs.arrange(tabId, ids);
    } catch (e) {
      addToast(`Couldn't save layout: ${(e as Error).message}`, 'error');
      load();
    }
  }

  function movePad(i: number, d: number) {
    const j = i + d;
    if (!activeTabId || j < 0 || j >= tabPads.length) return;
    const ids = tabPads.map((p) => p.id);
    [ids[i], ids[j]] = [ids[j], ids[i]];
    arrange(activeTabId, ids);
  }

  let dragId = $state<number | null>(null);
  let dropBefore = $state<number | null>(null); // pad id, or -1 for the end
  let dropTab = $state<number | null>(null);

  function onPadDragStart(e: DragEvent, p: SfxPad) {
    dragId = p.id;
    e.dataTransfer?.setData('text/x-dndscribe-pad', String(p.id));
    if (e.dataTransfer) e.dataTransfer.effectAllowed = 'move';
  }
  function onPadDragOver(e: DragEvent, target: number) {
    if (dragId === null) return;
    e.preventDefault();
    dropBefore = target;
  }
  function onPadDrop(e: DragEvent, target: number) {
    if (dragId === null || !activeTabId) return;
    e.preventDefault();
    const ids = tabPads.map((p) => p.id).filter((id) => id !== dragId);
    const at = target === -1 ? ids.length : ids.indexOf(target);
    ids.splice(at < 0 ? ids.length : at, 0, dragId);
    arrange(activeTabId, ids);
    endDrag();
  }
  function onTabDrop(e: DragEvent, tabId: number) {
    if (dragId === null) return;
    e.preventDefault();
    if (tabId !== activeTabId) {
      const ids = pads.filter((p) => p.tabId === tabId).sort((a, b) => a.position - b.position).map((p) => p.id);
      arrange(tabId, [...ids, dragId]);
    }
    endDrag();
  }
  function endDrag() {
    dragId = dropBefore = dropTab = null;
  }

  // ---- Upload ----
  let fileInput: HTMLInputElement | undefined = $state();
  let uploadTotal = $state(0);
  let uploadDone = $state(0);

  /** Upload effects into the current tab (also called by the page on a drop). */
  export async function addFiles(files: PickedFile[]) {
    const audio = files.filter((f) => isAudioName(f.file.name));
    const skipped = files.length - audio.length;
    if (audio.length === 0) {
      addToast('No audio files to add', 'error');
      return;
    }
    if (uploadTotal > uploadDone) {
      addToast('Wait for the current upload to finish', 'error');
      return;
    }
    let tabId = activeTabId;
    try {
      if (tabId === null) {
        const t = await dj.sfx.tabs.create('Effects');
        tabId = t.id;
        selectTab(t.id);
      }
    } catch (e) {
      addToast(`Couldn't create a tab: ${(e as Error).message}`, 'error');
      return;
    }
    uploadTotal = audio.length;
    uploadDone = 0;
    const failures: string[] = [];
    let next = 0;
    const worker = async () => {
      while (next < audio.length) {
        const f = audio[next++];
        try {
          await dj.sfx.upload(tabId!, f.file);
        } catch (e) {
          failures.push((e as Error).message);
        }
        uploadDone++;
      }
    };
    await Promise.all([worker(), worker()]);
    const ok = audio.length - failures.length;
    const extra = skipped ? `, ${skipped} not audio` : '';
    if (failures.length) addToast(`Added ${ok}, ${failures.length} failed: ${failures[0]}${extra}`, 'error');
    else addToast(`Added ${ok} sound${ok === 1 ? '' : 's'}${extra}`, 'success');
    uploadTotal = uploadDone = 0;
    load();
  }

  function onPick(e: Event) {
    const input = e.currentTarget as HTMLInputElement;
    const files = Array.from(input.files ?? []).map((file) => ({ file, rel: file.name }));
    input.value = '';
    addFiles(files);
  }
</script>

<section class="sb" class:big aria-label="Soundboard">
  <header>
    <h2 class="section-title">Soundboard</h2>
    <div class="gain">
      <span class="tiny subtle">Volume</span>
      <Fader value={st?.sfxGain ?? 1} cmd="mix.sfxGain" args={(v) => ({ value: v })} label="Effects volume" disabled={!live} />
    </div>
    <div class="tools">
      <button class="btn btn-sm" onclick={stopAll} disabled={!live || !st?.sfx?.length}
        title="Stop every effect">&#9632; Stop{st?.sfx?.length ? ` (${st.sfx.length})` : ''}</button>
      <button class="btn btn-sm" class:btn-primary={editMode} onclick={() => (editMode = !editMode)} aria-pressed={editMode}>
        {editMode ? 'Done' : 'Edit'}
      </button>
      <button class="btn btn-sm" onclick={() => fileInput?.click()} disabled={uploadTotal > 0} title="Upload sound effects into this tab">
        {uploadTotal > 0 ? `${uploadDone}/${uploadTotal}` : '+ Add'}
      </button>
      <input bind:this={fileInput} type="file" multiple accept={AUDIO_EXTS.map((e) => '.' + e).join(',')} onchange={onPick} hidden />
    </div>
  </header>

  <div class="tabstrip" role="tablist" aria-label="Soundboard tabs">
    {#each tabs as t (t.id)}
      <button
        role="tab"
        class="tab"
        class:active={t.id === activeTabId}
        class:droptarget={dropTab === t.id}
        aria-selected={t.id === activeTabId}
        onclick={() => selectTab(t.id)}
        ondragover={(e) => { if (dragId !== null) { e.preventDefault(); dropTab = t.id; } }}
        ondragleave={() => (dropTab = null)}
        ondrop={(e) => onTabDrop(e, t.id)}
      >
        {t.title} <span class="count">{padCount(t.id)}</span>
      </button>
    {/each}
    {#if editMode}
      {#if addingTab}
        <form class="addtab" onsubmit={(e) => { e.preventDefault(); addTab(); }}>
          <input class="input" bind:value={newTabTitle} placeholder="Tab name" aria-label="New tab name" />
          <button class="btn btn-sm" type="submit" disabled={!newTabTitle.trim()}>Add</button>
          <button class="btn-ghost btn-sm" type="button" onclick={() => (addingTab = false)}>Cancel</button>
        </form>
      {:else}
        <button class="tab add" onclick={() => (addingTab = true)}>+ Tab</button>
      {/if}
    {/if}
  </div>

  {#if editMode && activeTab}
    <div class="tab-edit">
      <input class="input" bind:value={tabTitleEdit} onchange={renameTab} aria-label="Tab name" />
      <button class="icon-btn" onclick={() => moveTab(-1)} disabled={tabs[0]?.id === activeTabId} aria-label="Move tab left">&#9664;</button>
      <button class="icon-btn" onclick={() => moveTab(1)} disabled={tabs[tabs.length - 1]?.id === activeTabId} aria-label="Move tab right">&#9654;</button>
      <button class="btn btn-sm" class:btn-danger={confirmTabDelete} onclick={deleteTab}>
        {confirmTabDelete ? `Delete tab and its ${padCount(activeTab.id)} sounds?` : 'Delete tab'}
      </button>
    </div>
  {/if}

  <div class="board">
    {#if loaded && tabs.length === 0}
      <div class="empty">
        <p>No sound effects yet.</p>
        <p class="subtle small">Press + Add, or drop audio files on the soundboard.</p>
      </div>
    {:else}
      <div class="grid" class:editing={editMode}>
        {#each tabPads as p, i (p.id)}
          {@const play = playing.get(p.id)}
          <div
            class="pad-wrap"
            class:dragging={dragId === p.id}
            class:dropbefore={dropBefore === p.id && dragId !== p.id}
            draggable={editMode}
            ondragstart={(e) => onPadDragStart(e, p)}
            ondragover={(e) => onPadDragOver(e, p.id)}
            ondrop={(e) => onPadDrop(e, p.id)}
            ondragend={endDrag}
            role="listitem"
          >
            <button
              class="pad"
              class:playing={!!play}
              class:hasimg={!!p.image}
              style:--pad-color={padColor(p)}
              style:background-image={p.image ? `url(${dj.sfx.imageUrl(p.image)})` : undefined}
              onclick={() => tap(p)}
              oncontextmenu={(e) => { e.preventDefault(); openEditor(p); }}
              disabled={!editMode && (!connected || !live)}
              aria-label={editMode ? `Edit ${p.label}` : `${play ? 'Restart' : 'Play'} ${p.label}`}
              title={editMode ? 'Edit' : !connected ? 'The bot is not in a voice channel' : 'Right-click to customise'}
            >
              <span class="pad-label">{p.label}</span>
              <span class="pad-meta">
                {#if play}{fmtMs(elapsedMs(p.id))} / {fmtMs(p.durationMs)}{:else}{fmtMs(p.durationMs)}{/if}
              </span>
              {#if play}
                {#key play.instance}<span class="bar" style:width="{progress(p.id) * 100}%"></span>{/key}
              {/if}
            </button>
            {#if editMode}
              <div class="pad-move">
                <button class="icon-btn" onclick={() => movePad(i, -1)} disabled={i === 0} aria-label="Move {p.label} earlier">&#9664;</button>
                <button class="icon-btn" onclick={() => movePad(i, 1)} disabled={i === tabPads.length - 1} aria-label="Move {p.label} later">&#9654;</button>
              </div>
            {/if}
          </div>
        {/each}
        {#if editMode}
          <button
            class="pad add-pad"
            class:dropbefore={dropBefore === -1}
            onclick={() => fileInput?.click()}
            ondragover={(e) => onPadDragOver(e, -1)}
            ondrop={(e) => onPadDrop(e, -1)}
          >
            <span class="plus">+</span>
            <span>Add sound</span>
          </button>
        {/if}
      </div>
      {#if editMode}
        <p class="tiny subtle hint">Editing: taps open a pad's settings and don't play. Drag pads to reorder, or onto a tab to move them.</p>
      {/if}
    {/if}
  </div>
</section>

<PadEditor bind:open={editorOpen} pad={editingPad} {tabs} onchanged={load} />

<style>
  .sb { display: flex; flex-direction: column; gap: 10px; min-height: 0; height: 100%; }
  header { display: flex; justify-content: space-between; align-items: center; gap: 8px; flex-wrap: wrap; }
  .tools { display: flex; gap: 6px; flex-wrap: wrap; }
  .gain { display: flex; align-items: center; gap: 8px; flex: 1 1 160px; min-width: 140px; max-width: 320px; }
  .gain .tiny { white-space: nowrap; }
  .gain :global(.fader) { flex: 1; }
  .tabstrip { display: flex; gap: 6px; overflow-x: auto; align-items: center; padding-bottom: 2px; scrollbar-width: thin; flex-shrink: 0; }
  .tab {
    min-height: 40px; padding: 0 14px; border-radius: var(--radius); font-weight: 600; font-size: 13px; white-space: nowrap;
    background: var(--surface-raised); border: 1px solid var(--border-strong); color: var(--text-2);
  }
  .tab.active { color: var(--text); border-color: var(--accent); background: color-mix(in srgb, var(--accent) 14%, var(--surface)); }
  .tab.droptarget { outline: 2px dashed var(--accent); outline-offset: 2px; }
  .tab .count { color: var(--text-3); font-weight: 500; margin-left: 4px; }
  .tab.add { background: none; border-style: dashed; color: var(--accent-text, var(--accent)); }
  .addtab { display: flex; gap: 6px; align-items: center; }
  .addtab .input { width: 140px; }
  .tab-edit { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
  .tab-edit .input { width: 160px; }
  .tab-edit .icon-btn { width: 40px; height: 40px; border: 1px solid var(--border-strong); border-radius: var(--radius); }
  .board { flex: 1; min-height: 0; overflow-y: auto; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(112px, 1fr)); gap: 8px; }
  .big .grid { grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; }
  .pad-wrap { position: relative; }
  .pad-wrap.dragging { opacity: 0.4; }
  .pad-wrap.dropbefore::before, .add-pad.dropbefore::before {
    content: ''; position: absolute; left: -5px; top: 6px; bottom: 6px; width: 3px; border-radius: 2px; background: var(--accent);
  }
  .pad {
    --pad-color: var(--accent);
    position: relative; overflow: hidden; width: 100%; min-height: 84px; padding: 10px; text-align: left;
    display: flex; flex-direction: column; justify-content: flex-end; gap: 2px;
    border-radius: var(--radius-lg); border: 1px solid color-mix(in srgb, var(--pad-color) 45%, var(--border-strong));
    background-color: color-mix(in srgb, var(--pad-color) 22%, var(--surface));
    background-size: cover; background-position: center;
    color: var(--text); transition: transform 0.06s, box-shadow 0.15s;
  }
  .big .pad { min-height: 92px; padding: 12px; }
  /* Darken the bottom of image pads so the label stays readable. */
  .pad.hasimg::after { content: ''; position: absolute; inset: 0; background: linear-gradient(to top, rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0.05) 60%); }
  .pad > * { position: relative; z-index: 1; }
  .pad:active:not(:disabled) { transform: scale(0.97); }
  .pad:disabled { opacity: 0.5; cursor: not-allowed; }
  .pad.playing { border-color: var(--pad-color); box-shadow: 0 0 0 2px var(--pad-color), 0 4px 18px color-mix(in srgb, var(--pad-color) 30%, transparent); animation: pulse 1.2s ease-in-out infinite; }
  @keyframes pulse { 50% { box-shadow: 0 0 0 2px var(--pad-color), 0 4px 26px color-mix(in srgb, var(--pad-color) 50%, transparent); } }
  @media (prefers-reduced-motion: reduce) { .pad.playing { animation: none; } }
  .grid.editing .pad { border-style: dashed; }
  .pad-label { font-size: 14px; font-weight: 700; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .big .pad-label { font-size: 16px; }
  .pad-meta { font-size: 11px; color: var(--text-2); font-variant-numeric: tabular-nums; }
  .pad.hasimg .pad-meta { color: rgba(255, 255, 255, 0.8); }
  .bar { position: absolute !important; left: 0; bottom: 0; height: 4px; background: var(--pad-color); transition: width 0.1s linear; }
  .pad-move { position: absolute; top: 6px; right: 6px; display: flex; gap: 4px; z-index: 2; }
  .pad-move .icon-btn { width: 36px; height: 36px; background: var(--surface-raised); border: 1px solid var(--border-strong); border-radius: var(--radius); }
  .add-pad {
    background: none; border: 1px dashed var(--border-strong); color: var(--text-2);
    align-items: center; justify-content: center; text-align: center;
  }
  .add-pad .plus { font-size: 22px; color: var(--accent); }
  .hint { margin: 10px 0 0; }
  .empty { padding: 24px 10px; text-align: center; }
  .empty p { margin: 4px 0; }
</style>
