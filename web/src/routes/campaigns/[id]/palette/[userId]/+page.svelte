<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { scoped } from '$lib/api';
  import { connect, subscribe as wsSubscribe } from '$lib/ws';
  import { voiceMembers } from '$lib/stores/ws';
  import { addToast } from '$lib/stores/toast';
  import type { NicknamePreset, NicknameCategory } from '$lib/types';

  let campaignId = $derived(Number($page.params.id));
  let userId = $derived($page.params.userId);
  let username = $derived($page.url.searchParams.get('username') ?? userId);
  let api = $derived(scoped(campaignId));

  let presetsData: NicknamePreset[] = $state([]);
  let categoriesData: NicknameCategory[] = $state([]);
  let currentNickname: string | null = $state(null);
  let searchMode = $state(false);
  let searchQuery = $state('');

  // Inline editing state
  let editingPresetId = $state<number | null>(null);
  let editingPresetValue = $state('');
  let addingPreset = $state(false);
  let addingPresetValue = $state('');
  let addingPresetSaving = false; // guard against double-fire from Enter+blur
  let editingCategoryId = $state<number | null>(null);
  let editingCategoryValue = $state('');
  let addingCategory = $state(false);
  let addingCategoryValue = $state('');
  let addingCategorySaving = false;
  let editingPresetSaving = false;
  let shakeInput = $state(false);
  let addPresetInputEl: HTMLInputElement | undefined = $state();
  let addCategoryInputEl: HTMLInputElement | undefined = $state();

  // Focus inputs when they appear
  $effect(() => { if (addingPreset && addPresetInputEl) addPresetInputEl.focus(); });
  $effect(() => { if (addingCategory && addCategoryInputEl) addCategoryInputEl.focus(); });

  // Context menu state
  let contextMenuOpen = $state(false);
  let contextMenuX = $state(0);
  let contextMenuY = $state(0);
  let contextMenuPreset = $state<NicknamePreset | null>(null);
  let contextMenuCategory = $state<NicknameCategory | null>(null);

  // Keyboard nav state
  let selectedCatIdx = $state<number | null>(null);
  let focusedItemIdx = $state(-1);
  let keyModeText = $state('');

  onMount(() => {
    connect();
    loadData();

    const unsub = wsSubscribe((msg) => {
      if (msg.type === 'members' || msg.type === 'init') {
        const members = msg.type === 'init' ? msg.members : msg.data;
        const member = (members || []).find((m: any) => m.id === userId);
        if (member) currentNickname = member.characterName || null;
      }
      if (msg.type === 'presets_updated') {
        presetsData = (msg.presets || []).filter((p: NicknamePreset) => p.discordUserId === userId);
        categoriesData = (msg.categories || []).filter((c: NicknameCategory) => c.discordUserId === userId);
      }
    });

    const member = $voiceMembers.find(m => m.id === userId);
    if (member) currentNickname = member.characterName;

    return () => unsub();
  });

  async function loadData() {
    try {
      const [pr, cats] = await Promise.all([
        api.presets.list(userId),
        api.categories.list(userId),
      ]);
      presetsData = pr;
      categoriesData = cats;
    } catch (e) {
      addToast(`Failed to load: ${e}`, 'error');
    }
  }

  // Group presets by category
  const groups = $derived((() => {
    const query = searchMode ? searchQuery.toLowerCase() : '';
    const filtered = query
      ? presetsData.filter(p => p.label.toLowerCase().includes(query))
      : presetsData;

    const result: { catId: number | null; catName: string; presets: NicknamePreset[] }[] = [];
    result.push({ catId: null, catName: 'Default', presets: filtered.filter(p => !p.categoryId) });
    for (const cat of categoriesData) {
      result.push({ catId: cat.id, catName: cat.name, presets: filtered.filter(p => p.categoryId === cat.id) });
    }
    return result;
  })());

  async function setNickname(name: string | null) {
    // Optimistic update: the backend's SetCharacterName does a sequential
    // Recorder.FlushUserBuffer → DB write → recorder update → members
    // broadcast, which adds up to ~500ms round-trip. Hotkey-driven nickname
    // selection felt visibly laggy because the UI waited on the whole chain
    // before showing any change. Update local state first, let the network
    // catch up in the background, revert on error. The WS 'members'
    // broadcast that eventually fires reasserts the same value, so no churn.
    const prev = currentNickname;
    currentNickname = name;
    selectedCatIdx = null;
    focusedItemIdx = -1;
    keyModeText = '';
    if (searchMode) { searchMode = false; searchQuery = ''; }
    try {
      await api.setCharacterName(userId, name);
    } catch (e) {
      currentNickname = prev;
      addToast(`Failed: ${e}`, 'error');
    }
  }

  // ---- Preset CRUD ----

  function startAddPreset() {
    addingPreset = true;
    addingPresetValue = '';
  }

  async function finishAddPreset() {
    if (addingPresetSaving) return;
    const label = addingPresetValue.trim();
    if (!label) { addingPreset = false; return; }
    if (presetsData.some(p => p.label.toLowerCase() === label.toLowerCase())) {
      shakeInput = true;
      setTimeout(() => { shakeInput = false; }, 300);
      return; // Keep input open so user can fix it
    }
    addingPresetSaving = true;
    addingPreset = false;
    try {
      await api.presets.add(userId, label, presetsData.length);
    } catch (e) {
      addToast(`Failed: ${e}`, 'error');
    } finally {
      addingPresetSaving = false;
    }
  }

  function startEditPreset(preset: NicknamePreset) {
    editingPresetId = preset.id;
    editingPresetValue = preset.label;
    closeContextMenu();
  }

  async function finishEditPreset() {
    if (editingPresetSaving || editingPresetId == null) return;
    const label = editingPresetValue.trim();
    const id = editingPresetId;
    if (!label) { editingPresetId = null; return; }
    const existing = presetsData.find(p => p.id === id);
    if (!existing || existing.label === label) { editingPresetId = null; return; }
    if (presetsData.some(p => p.id !== id && p.label.toLowerCase() === label.toLowerCase())) {
      shakeInput = true;
      setTimeout(() => { shakeInput = false; }, 300);
      return;
    }
    editingPresetSaving = true;
    editingPresetId = null;
    try {
      await api.presets.update(id, label);
    } catch (e) {
      addToast(`Failed: ${e}`, 'error');
    } finally {
      editingPresetSaving = false;
    }
  }

  async function deletePreset(preset: NicknamePreset) {
    closeContextMenu();
    // Optimistic: remove locally now, let the WS 'presets_updated'
    // broadcast reassert the same state later. Reverts on API error.
    const prev = presetsData;
    presetsData = presetsData.filter(p => p.id !== preset.id);
    try {
      await api.presets.delete(preset.id);
    } catch (e) {
      presetsData = prev;
      addToast(`Failed: ${e}`, 'error');
    }
  }

  async function movePresetToCategory(preset: NicknamePreset, categoryId: number | null) {
    closeContextMenu();
    // Optimistic: update categoryId locally now; WS broadcast will reassert.
    const prev = presetsData;
    presetsData = presetsData.map(p =>
      p.id === preset.id ? { ...p, categoryId } : p
    );
    try {
      await api.presets.move(preset.id, categoryId);
    } catch (e) {
      presetsData = prev;
      addToast(`Failed: ${e}`, 'error');
    }
  }

  // ---- Category CRUD ----

  function startAddCategory() {
    addingCategory = true;
    addingCategoryValue = '';
  }

  async function finishAddCategory() {
    if (addingCategorySaving) return;
    const name = addingCategoryValue.trim();
    if (!name) { addingCategory = false; return; }
    addingCategorySaving = true;
    addingCategory = false;
    try {
      await api.categories.add(userId, name, categoriesData.length);
    } catch (e) {
      addToast(`Failed: ${e}`, 'error');
    } finally {
      addingCategorySaving = false;
    }
  }

  function startEditCategory(cat: NicknameCategory) {
    editingCategoryId = cat.id;
    editingCategoryValue = cat.name;
    closeContextMenu();
  }

  async function finishEditCategory() {
    if (editingCategoryId == null) return;
    const name = editingCategoryValue.trim();
    const id = editingCategoryId;
    editingCategoryId = null;
    if (!name) return;
    try {
      await api.categories.update(id, name);
      // WS broadcast will update
    } catch (e) {
      addToast(`Failed: ${e}`, 'error');
    }
  }

  async function deleteCategory(cat: NicknameCategory) {
    closeContextMenu();
    try {
      await api.categories.delete(cat.id);
      // WS broadcast will update
    } catch (e) {
      addToast(`Failed: ${e}`, 'error');
    }
  }

  // ---- Context menu ----

  function showPresetContextMenu(e: MouseEvent, preset: NicknamePreset) {
    e.preventDefault();
    contextMenuPreset = preset;
    contextMenuCategory = null;
    contextMenuX = e.clientX;
    contextMenuY = e.clientY;
    contextMenuOpen = true;
  }

  function showCategoryContextMenu(e: MouseEvent, cat: NicknameCategory) {
    e.preventDefault();
    contextMenuCategory = cat;
    contextMenuPreset = null;
    contextMenuX = e.clientX;
    contextMenuY = e.clientY;
    contextMenuOpen = true;
  }

  function closeContextMenu() {
    contextMenuOpen = false;
    contextMenuPreset = null;
    contextMenuCategory = null;
  }

  // ---- Keyboard shortcuts ----

  function handleKeydown(e: KeyboardEvent) {
    if (e.target instanceof HTMLInputElement) return;

    if (e.key === '/') { e.preventDefault(); searchMode = true; return; }
    if (e.key === 'Escape') {
      if (selectedCatIdx !== null) { selectedCatIdx = null; focusedItemIdx = -1; keyModeText = ''; return; }
      if (searchMode) { searchMode = false; searchQuery = ''; return; }
      return;
    }
    if (e.key === 'c' || e.key === 'C') { setNickname(null); return; }
    if (e.key === 'd' || e.key === 'D') { setDM(); return; }
    if (e.key === 'n' || e.key === 'N') { e.preventDefault(); startAddPreset(); return; }

    // Arrow navigation within selected category
    if (selectedCatIdx !== null) {
      const catPresets = groups[selectedCatIdx]?.presets ?? [];
      if (e.key === 'ArrowDown') { e.preventDefault(); focusedItemIdx = Math.min(focusedItemIdx + 1, catPresets.length - 1); return; }
      if (e.key === 'ArrowUp') { e.preventDefault(); focusedItemIdx = Math.max(focusedItemIdx - 1, 0); return; }
      if (e.key === 'Enter' && focusedItemIdx >= 0 && focusedItemIdx < catPresets.length) {
        e.preventDefault(); setNickname(catPresets[focusedItemIdx].label); return;
      }
      if (e.key === 'e' || e.key === 'E') {
        if (focusedItemIdx >= 0 && focusedItemIdx < catPresets.length) {
          e.preventDefault(); startEditPreset(catPresets[focusedItemIdx]); return;
        }
      }
      if (e.key === 'Delete' || e.key === 'Backspace') {
        if (focusedItemIdx >= 0 && focusedItemIdx < catPresets.length) {
          e.preventDefault(); deletePreset(catPresets[focusedItemIdx]); return;
        }
      }
    }

    // Number keys: select category or item within category
    if (e.key >= '1' && e.key <= '9') {
      const num = parseInt(e.key, 10) - 1;
      if (selectedCatIdx === null) {
        if (num < groups.length) {
          selectedCatIdx = num;
          focusedItemIdx = 0;
          keyModeText = `Category: ${groups[num].catName} -- press 1-9 for item, Esc to cancel`;
        }
      } else {
        const catPresets = groups[selectedCatIdx]?.presets ?? [];
        if (num < catPresets.length) {
          setNickname(catPresets[num].label);
        }
      }
      return;
    }
  }

  async function setDM() {
    // setNickname is already optimistic; call it first so the UI flips
    // immediately. The DM-preset creation (if missing) can fire in
    // parallel — it's not required for the nickname to be set.
    setNickname('DM');
    const hasDM = presetsData.some(p => p.label === 'DM');
    if (!hasDM) {
      try {
        await api.presets.add(userId, 'DM', presetsData.length);
      } catch (e) { /* ignore */ }
    }
  }

  function handlePresetKeydown(e: KeyboardEvent, action: () => void) {
    if (e.key === 'Enter') { e.preventDefault(); action(); }
    if (e.key === 'Escape') { editingPresetId = null; addingPreset = false; }
  }

  function handleCategoryKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter') { e.preventDefault(); finishEditCategory(); }
    if (e.key === 'Escape') { editingCategoryId = null; addingCategory = false; }
  }
</script>

<svelte:window onkeydown={handleKeydown} onclick={closeContextMenu} />

<svelte:head>
  <title>Name Palette - {username}</title>
</svelte:head>

<div class="palette-page">
  <div class="pal-header">
    <h1>Name Palette</h1>
    <span class="user-info">{username}</span>
  </div>

  <div class="current-name">
    Current: {#if currentNickname}<strong>{currentNickname}</strong>{:else}<em>none</em>{/if}
  </div>

  <div class="shortcuts">
    <kbd>/</kbd> search
    <kbd>C</kbd> clear
    <kbd>D</kbd> DM
    <kbd>1-9</kbd> category, then item
    <kbd>Esc</kbd> cancel
    <kbd>N</kbd> new name
    <kbd>E</kbd> edit
    <kbd>Del</kbd> delete
  </div>

  {#if keyModeText}
    <div class="key-mode">{keyModeText}</div>
  {/if}

  {#if searchMode}
    <div class="search-wrap">
      <input
        type="text"
        bind:value={searchQuery}
        placeholder="Search names..."
        autofocus
        onkeydown={(e) => { if (e.key === 'Escape') { searchMode = false; searchQuery = ''; } }}
      />
    </div>
  {/if}

  <div class="content">
    {#each groups as group, gi}
      <div class="category-section">
        <div
          class="category-header"
          class:focused={selectedCatIdx === gi}
          oncontextmenu={(e) => { if (group.catId != null) showCategoryContextMenu(e, categoriesData.find(c => c.id === group.catId)!); }}
        >
          <span class="cat-key">{gi + 1}</span>
          {#if editingCategoryId === group.catId && group.catId != null}
            <input
              class="inline-input"
              type="text"
              bind:value={editingCategoryValue}
              onkeydown={handleCategoryKeydown}
              onblur={finishEditCategory}
              autofocus
            />
          {:else}
            <span class="cat-name">{group.catName}</span>
          {/if}
        </div>
        <div class="preset-grid">
          {#if group.catId === null && !searchMode}
            <button
              class="preset-btn clear-btn"
              class:active={!currentNickname}
              onclick={() => setNickname(null)}
            >Clear</button>
          {/if}
          {#each group.presets as preset, pi}
            {#if editingPresetId === preset.id}
              <input
                class="inline-input preset-input"
                type="text"
                bind:value={editingPresetValue}
                onkeydown={(e) => handlePresetKeydown(e, finishEditPreset)}
                onblur={finishEditPreset}
                autofocus
              />
            {:else}
              <button
                class="preset-btn"
                class:active={currentNickname === preset.label}
                class:focused={selectedCatIdx === gi && focusedItemIdx === pi}
                onclick={() => setNickname(preset.label)}
                oncontextmenu={(e) => showPresetContextMenu(e, preset)}
              >
                {preset.label}
                {#if selectedCatIdx === gi && pi < 9}
                  <span class="key-hint">{pi + 1}</span>
                {/if}
              </button>
            {/if}
          {/each}
        </div>
      </div>
    {/each}
  </div>

  {#if addingPreset}
    <div class="add-inline">
      <input
        bind:this={addPresetInputEl}
        class="inline-input"
        class:shake={shakeInput}
        type="text"
        placeholder="New name..."
        bind:value={addingPresetValue}
        onkeydown={(e) => handlePresetKeydown(e, finishAddPreset)}
        onblur={finishAddPreset}
      />
    </div>
  {/if}

  {#if addingCategory}
    <div class="add-inline">
      <input
        bind:this={addCategoryInputEl}
        class="inline-input"
        type="text"
        placeholder="Category name..."
        bind:value={addingCategoryValue}
        onkeydown={(e) => { if (e.key === 'Enter') { e.preventDefault(); finishAddCategory(); } if (e.key === 'Escape') { addingCategory = false; } }}
        onblur={finishAddCategory}
      />
    </div>
  {/if}

  <div class="action-bar">
    <button class="action-btn" onclick={startAddCategory}>+ Category</button>
    <button class="action-btn" onclick={startAddPreset}>+ Name</button>
    <button class="action-btn" onclick={() => { searchMode = !searchMode; searchQuery = ''; }}>
      {searchMode ? 'Close Search' : 'Search'}
    </button>
  </div>

  <!-- Context menu -->
  {#if contextMenuOpen}
    <div class="context-menu" style="left:{contextMenuX}px;top:{contextMenuY}px">
      {#if contextMenuPreset}
        <button class="context-item" onclick={() => startEditPreset(contextMenuPreset!)}>Edit</button>
        <div class="context-sub">
          <button class="context-item">Move to &raquo;</button>
          <div class="submenu">
            <button class="context-item" class:current={!contextMenuPreset!.categoryId} onclick={() => movePresetToCategory(contextMenuPreset!, null)}>Default</button>
            {#each categoriesData as cat}
              <button class="context-item" class:current={contextMenuPreset!.categoryId === cat.id} onclick={() => movePresetToCategory(contextMenuPreset!, cat.id)}>{cat.name}</button>
            {/each}
          </div>
        </div>
        <button class="context-item danger" onclick={() => deletePreset(contextMenuPreset!)}>Delete</button>
      {/if}
      {#if contextMenuCategory}
        <button class="context-item" onclick={() => startEditCategory(contextMenuCategory!)}>Rename</button>
        <button class="context-item danger" onclick={() => deleteCategory(contextMenuCategory!)}>Delete (moves items to Default)</button>
      {/if}
    </div>
  {/if}
</div>

<style>
  .palette-page { padding: 12px; min-height: 100vh; }
  .pal-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
  .pal-header h1 { font-size: 18px; color: var(--accent); }
  .user-info { font-size: 15px; color: var(--text-secondary); }

  .current-name { font-size: 16px; color: var(--text-secondary); margin-bottom: 8px; }
  .current-name strong { color: var(--success); }
  .current-name em { color: var(--text-muted); }

  .shortcuts { font-size: 10px; color: #444; margin-bottom: 8px; }
  .shortcuts kbd {
    background: var(--bg-elevated); border: 1px solid var(--border-light); border-radius: 2px;
    padding: 0 3px; font-family: inherit; color: var(--text-secondary);
  }
  @media (pointer: coarse) { .shortcuts { display: none; } }

  .key-mode { font-size: 13px; color: var(--accent); margin-bottom: 8px; font-weight: 600; }

  .search-wrap { margin-bottom: 8px; }
  .search-wrap input {
    width: 100%; background: var(--bg-secondary); border: 2px solid var(--accent);
    color: var(--text-primary); padding: 10px 14px; border-radius: 8px; font-size: 16px; outline: none;
  }

  .category-section { margin-bottom: 12px; }
  .category-header {
    display: flex; align-items: center; gap: 8px;
    font-size: 13px; font-weight: 600; color: var(--text-secondary);
    padding: 4px 0; margin-bottom: 6px; border-bottom: 1px dashed var(--border-light);
    cursor: default; user-select: none;
  }
  .category-header.focused { border-bottom-color: var(--accent); }
  .category-header.focused .cat-name { color: var(--accent); }
  .cat-key {
    font-size: 10px; background: var(--bg-elevated); border: 1px solid var(--border-light);
    border-radius: 2px; padding: 0 4px; color: var(--text-secondary);
  }
  .cat-name { flex: 1; }

  .preset-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 8px; }
  .preset-btn {
    padding: 14px 16px; border-radius: 8px; cursor: pointer; font-size: 16px;
    font-weight: 600; border: 2px solid var(--border-light); background: var(--bg-elevated);
    color: var(--text-primary); text-align: center; min-height: 48px;
    position: relative; user-select: none;
    -webkit-tap-highlight-color: rgba(233, 69, 96, 0.3);
  }
  .preset-btn:hover { background: var(--border-light); }
  .preset-btn:active { transform: scale(0.96); }
  .preset-btn.active { border-color: var(--success); background: var(--success-bg); color: var(--success); }
  .preset-btn.focused { outline: 2px solid var(--accent); outline-offset: 1px; }
  .preset-btn.clear-btn { background: var(--bg-secondary); color: var(--text-secondary); border-style: dashed; }
  .preset-btn.clear-btn:hover { color: var(--text-primary); }
  .preset-btn.clear-btn.active { border-color: var(--success); color: var(--success); }
  .key-hint {
    font-size: 10px; color: var(--text-muted); margin-left: 4px; vertical-align: super;
  }
  @media (pointer: coarse) { .key-hint { display: none; } }

  .inline-input {
    background: var(--bg-primary); border: 2px solid var(--accent); color: var(--text-primary);
    padding: 12px 14px; border-radius: 8px; font-size: 16px; font-weight: 600;
    width: 100%; outline: none;
  }
  .inline-input.shake { animation: shake 0.3s ease; border-color: #ff4444; }
  @keyframes shake {
    0%, 100% { transform: translateX(0); }
    20%, 60% { transform: translateX(-4px); }
    40%, 80% { transform: translateX(4px); }
  }
  .preset-input { padding: 14px 16px; min-height: 48px; }
  .add-inline { margin-bottom: 8px; }

  .action-bar { display: flex; gap: 6px; margin-top: 12px; flex-wrap: wrap; }
  .action-btn {
    padding: 8px 14px; border-radius: 6px; cursor: pointer; font-size: 13px;
    border: 1px solid var(--border-light); background: var(--bg-elevated); color: var(--text-primary);
  }
  .action-btn:hover { background: var(--border-light); }

  /* Context menu */
  .context-menu {
    position: fixed; background: var(--bg-secondary); border: 1px solid var(--border-light);
    border-radius: 4px; z-index: 100; box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    min-width: 140px;
  }
  .context-item {
    display: block; width: 100%; text-align: left;
    padding: 6px 12px; cursor: pointer; font-size: 13px;
    background: none; border: none; color: var(--text-primary);
  }
  .context-item:hover { background: var(--bg-elevated); }
  .context-item.danger { color: var(--accent); }
  .context-item.current { color: var(--success); }
  .context-sub { position: relative; }
  .context-sub > .context-item::after { content: ' \25B6'; font-size: 10px; float: right; }
  .submenu {
    display: none; position: absolute; left: 100%; top: 0;
    background: var(--bg-secondary); border: 1px solid var(--border-light);
    border-radius: 4px; min-width: 120px; box-shadow: 0 4px 12px rgba(0,0,0,0.5);
  }
  .context-sub:hover .submenu { display: block; }
</style>
