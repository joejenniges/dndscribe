<script lang="ts">
  import { tick, type Snippet } from 'svelte';
  import { scoped } from '$lib/api';
  import { addToast } from '$lib/stores/toast';
  import TranscriptionLine from '$lib/components/TranscriptionLine.svelte';
  import SearchBar from '$lib/components/SearchBar.svelte';
  import AudioPlayerBar, { type PlayRequest } from '$lib/components/AudioPlayerBar.svelte';
  import Toggle from '$lib/components/ui/Toggle.svelte';
  import type { TranscriptionLine as TLine, Partial } from '$lib/types';

  // Shared transcript workspace for the live page and the session page.
  // The page owns where lines come from (WS store vs one-off fetch) and what
  // goes in the header; everything about reading/searching/playing/editing
  // lives here so both pages behave identically.
  let {
    campaignId,
    lines,
    partials = [],
    live = false,
    loading = false,
    emptyMessage = 'No transcription lines yet',
    onEditLine,
    onDeleteLine,
    onNicknamesSaved,
    header,
    panelTop,
    panelBottom,
  }: {
    campaignId: number;
    lines: TLine[];
    partials?: Partial[];
    // Live mode: pin to the newest line and show partials + "Jump to live".
    live?: boolean;
    loading?: boolean;
    emptyMessage?: string;
    onEditLine: (id: number, text: string) => Promise<void> | void;
    onDeleteLine: (id: number) => Promise<void> | void;
    // Called after a bulk nickname save so the page can refetch lines.
    onNicknamesSaved?: () => Promise<void> | void;
    header?: Snippet;
    panelTop?: Snippet;
    panelBottom?: Snippet;
  } = $props();

  const api = $derived(scoped(campaignId));

  // ── View preferences (persisted per browser) ──
  const PREFS_KEY = 'dndscribe.transcript.prefs';
  interface Prefs { compact: boolean; grouped: boolean; showConfidence: boolean; panelOpen: boolean; }
  function loadPrefs(): Prefs {
    const d: Prefs = { compact: false, grouped: true, showConfidence: true, panelOpen: true };
    try { return { ...d, ...JSON.parse(localStorage.getItem(PREFS_KEY) ?? '{}') }; } catch { return d; }
  }
  const initial = loadPrefs();
  let compact = $state(initial.compact);
  let grouped = $state(initial.grouped);
  let showConfidence = $state(initial.showConfidence);
  // On narrow screens the panel is an overlay drawer; never auto-open it there.
  let panelOpen = $state(initial.panelOpen && (typeof window === 'undefined' || window.innerWidth > 1100));
  $effect(() => {
    const p: Prefs = { compact, grouped, showConfidence, panelOpen };
    try { localStorage.setItem(PREFS_KEY, JSON.stringify(p)); } catch { /* private mode */ }
  });

  // Not persisted: a stale confidence filter silently hiding lines is confusing.
  let minConfidence = $state(100);

  // ── Search ──
  let searchQuery = $state('');
  let filterActive = $state(false);
  let searchHighlightId = $state<number | null>(null);
  let searchResultIndex = $state(-1);

  // ── Scrolling ──
  let scrollEl: HTMLDivElement | undefined = $state();
  let pinned = $state(true);

  // ── Player ──
  let playRequest = $state<PlayRequest | null>(null);
  let autoplay = $state(false);
  let player: ReturnType<typeof AudioPlayerBar> | undefined = $state();

  // ── Nickname editor ──
  let nicknameEditorOpen = $state(false);
  let selectedLineIds = $state<number[]>([]);
  let selectedUsernames = $state<string[]>([]);
  let nicknameInputs = $state<Record<string, string>>({});

  const SPEAKER_COLORS = [
    '#f06e94', '#5bd69a', '#7cb6fa', '#f2b45c', '#c08cf0',
    '#4fd1dd', '#ff8a7a', '#9ad46f', '#ffcf5c', '#9a8cf5',
    '#ff7eb6', '#5cc8ff', '#b5e06c', '#ffd27a', '#c4a484',
  ];

  // Map discord usernames to colors (first-seen order). Stable within a session.
  const speakerColorMap = $derived((() => {
    const map = new Map<string, string>();
    for (const line of lines) {
      if (!map.has(line.discordUsername)) map.set(line.discordUsername, SPEAKER_COLORS[map.size % SPEAKER_COLORS.length]);
    }
    return map;
  })());

  // Parse search query: optional "@speaker" prefix, then /regex/flags or plain text.
  const parsedSearch = $derived((() => {
    const q = searchQuery.trim();
    if (!q) return { speaker: null as string | null, regex: null as RegExp | null };
    let speaker: string | null = null;
    let textPart = q;
    if (q.startsWith('@')) {
      const sp = q.indexOf(' ');
      speaker = (sp >= 0 ? q.slice(1, sp) : q.slice(1)).toLowerCase();
      textPart = sp >= 0 ? q.slice(sp + 1).trim() : '';
    }
    let regex: RegExp | null = null;
    if (textPart) {
      const m = textPart.match(/^\/(.+?)\/([gimsuy]*)$/);
      if (m) {
        try { regex = new RegExp(m[1], m[2].includes('i') ? m[2] : m[2] + 'i'); } catch { /* fall through */ }
      }
      if (!regex) regex = new RegExp(textPart.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi');
    }
    return { speaker, regex };
  })());

  const speakersList = $derived((() => {
    const map = new Map<string, Set<string>>();
    for (const line of lines) {
      if (!map.has(line.discordUsername)) map.set(line.discordUsername, new Set());
      if (line.nickname) map.get(line.discordUsername)!.add(line.nickname);
    }
    return Array.from(map.entries()).map(([username, nicks]) => ({ username, nicknames: Array.from(nicks) }));
  })());

  function lineMatchesSearch(l: TLine): boolean {
    const { speaker, regex } = parsedSearch;
    if (speaker) {
      const ok = l.discordUsername.toLowerCase().includes(speaker) || (l.nickname?.toLowerCase().includes(speaker) ?? false);
      if (!ok) return false;
    }
    if (regex) {
      regex.lastIndex = 0;
      if (regex.test(l.text)) return true;
      // Plain (non-@) queries also match speaker names.
      if (!speaker) {
        regex.lastIndex = 0;
        if (regex.test(l.discordUsername)) return true;
        regex.lastIndex = 0;
        return l.nickname ? regex.test(l.nickname) : false;
      }
      return false;
    }
    return speaker !== null;
  }

  const searchMatchIds = $derived(searchQuery.trim() ? lines.filter(lineMatchesSearch).map(l => l.id) : []);
  const searchMatchIdSet = $derived(new Set(searchMatchIds));

  $effect(() => {
    searchQuery;
    searchResultIndex = -1;
    searchHighlightId = null;
  });

  const filteredLines = $derived(lines.filter(l => {
    if (minConfidence < 100 && l.confidence != null && l.confidence * 100 > minConfidence) return false;
    if (filterActive && searchQuery.trim() && !searchMatchIdSet.has(l.id)) return false;
    return true;
  }));

  // Users with >1 diarization speaker (shared mic). Only then is the badge
  // worth showing; solo users would get a noisy constant "S1".
  const multiSpeakerUsers = $derived((() => {
    const perUser = new Map<string, Set<number>>();
    for (const l of lines) {
      if (l.speakerId == null) continue;
      let set = perUser.get(l.discordUsername);
      if (!set) { set = new Set(); perUser.set(l.discordUsername, set); }
      set.add(l.speakerId);
    }
    return new Set([...perUser.entries()].filter(([, s]) => s.size > 1).map(([u]) => u));
  })());

  function speakerBadgeFor(line: { discordUsername: string; speakerName?: string | null; speakerLabel?: string | null }): string {
    if (!multiSpeakerUsers.has(line.discordUsername)) return '';
    return line.speakerName || line.speakerLabel || '';
  }

  // Group consecutive lines from the same speaker under one header.
  const displayLines = $derived((() => {
    const out: (TLine & { isContinuation: boolean; nextIsContinuation: boolean })[] = [];
    for (const line of filteredLines) {
      const prev = out[out.length - 1];
      const isContinuation = grouped && prev != null
        && prev.discordUsername === line.discordUsername
        && prev.nickname === line.nickname
        // Different diarization speakers never merge; that's the point of diarization.
        && prev.speakerId === line.speakerId;
      out.push({ ...line, isContinuation, nextIsContinuation: false });
      if (isContinuation && prev) prev.nextIsContinuation = true;
    }
    return out;
  })());

  // ── Live tail ──
  // Follow new lines only while the reader is at the bottom; if they've
  // scrolled up to read, leave them alone and offer "Jump to live".
  $effect(() => {
    const n = displayLines.length + partials.length;
    if (!live || n === 0 || !pinned || !scrollEl) return;
    tick().then(() => { if (scrollEl) scrollEl.scrollTop = scrollEl.scrollHeight; });
  });

  // The session page starts at the top; the live page starts at the bottom.
  let initialScrollDone = false;
  $effect(() => {
    if (initialScrollDone || !scrollEl || displayLines.length === 0) return;
    initialScrollDone = true;
    if (live) tick().then(() => { if (scrollEl) scrollEl.scrollTop = scrollEl.scrollHeight; });
  });

  function handleScroll() {
    if (!scrollEl) return;
    const { scrollTop, scrollHeight, clientHeight } = scrollEl;
    pinned = scrollHeight - scrollTop - clientHeight < 60;
  }

  function jumpToLive() {
    if (!scrollEl) return;
    scrollEl.scrollTo({ top: scrollEl.scrollHeight, behavior: 'smooth' });
    pinned = true;
  }

  // ── Playback ──
  function play(line: TLine) {
    playRequest = {
      lineId: line.id,
      filenames: line.audioFilenames,
      speaker: line.nickname || line.discordUsername,
      color: speakerColorMap.get(line.discordUsername) ?? '#f43f62',
    };
  }

  function playNextLine() {
    if (!autoplay || !playRequest) return;
    const idx = displayLines.findIndex(l => l.id === playRequest!.lineId);
    if (idx === -1) return;
    for (let i = idx + 1; i < displayLines.length; i++) {
      const next = displayLines[i];
      if (next.audioFilenames?.length) {
        play(next);
        document.querySelector(`.line[data-id="${next.id}"]`)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
        return;
      }
    }
  }

  // ── Nickname editor ──
  function handleSelectLines(lineIds: number[], usernames: string[]) {
    selectedLineIds = lineIds;
    selectedUsernames = usernames;
    nicknameInputs = {};
    for (const u of usernames) {
      nicknameInputs[u] = lines.find(l => l.discordUsername === u)?.nickname ?? '';
    }
    nicknameEditorOpen = true;
    panelOpen = true;
  }

  function closeNicknameEditor() {
    nicknameEditorOpen = false;
    selectedLineIds = [];
    selectedUsernames = [];
    nicknameInputs = {};
  }

  async function saveNicknames() {
    const nicknames: Record<string, string | null> = {};
    for (const u of selectedUsernames) nicknames[u] = nicknameInputs[u]?.trim() || null;
    try {
      await api.bulkNickname(selectedLineIds, nicknames);
      addToast('Nicknames updated', 'success');
      closeNicknameEditor();
      await onNicknamesSaved?.();
    } catch (e) {
      addToast(`Failed: ${e}`, 'error');
    }
  }

  // ── Search navigation ──
  let searchWrapEl: HTMLDivElement | undefined = $state();

  function navigateSearchResult(direction: 1 | -1) {
    if (searchMatchIds.length === 0) return;
    searchResultIndex = (searchResultIndex + direction + searchMatchIds.length) % searchMatchIds.length;
    const id = searchMatchIds[searchResultIndex];
    searchHighlightId = id;
    document.querySelector(`.line[data-id="${id}"]`)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  function handleKeydown(e: KeyboardEvent) {
    const target = e.target as HTMLElement;
    const isEditing = target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT' || target.isContentEditable;

    if (e.key === 'Escape' && nicknameEditorOpen) { closeNicknameEditor(); e.preventDefault(); return; }

    // "/" or Ctrl+F focuses transcript search
    if ((e.key === '/' && !isEditing) || (e.key === 'f' && (e.ctrlKey || e.metaKey))) {
      e.preventDefault();
      searchWrapEl?.querySelector('input')?.focus();
      return;
    }
    // Ctrl+G / Ctrl+Shift+G: next/previous match
    if (e.key === 'g' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      navigateSearchResult(e.shiftKey ? -1 : 1);
      return;
    }
    if (isEditing || !scrollEl) return;

    if (e.key === 'PageDown') { e.preventDefault(); scrollEl.scrollBy({ top: scrollEl.clientHeight * 0.8, behavior: 'smooth' }); return; }
    if (e.key === 'PageUp') { e.preventDefault(); scrollEl.scrollBy({ top: -scrollEl.clientHeight * 0.8, behavior: 'smooth' }); return; }
    if (e.key === 'Home') { e.preventDefault(); scrollEl.scrollTop = 0; return; }
    if (e.key === 'End') { e.preventDefault(); scrollEl.scrollTop = scrollEl.scrollHeight; return; }
    if (e.key === ' ' && playRequest && target.tagName !== 'BUTTON') { e.preventDefault(); player?.togglePlayPause(); return; }
  }

  const lineCount = $derived(lines.length);
  const hiddenCount = $derived(lines.length - filteredLines.length);
</script>

<svelte:window onkeydown={handleKeydown} />

<div class="workspace" class:panel-open={panelOpen}>
  <section class="main">
    <div class="main-head">
      {#if header}{@render header()}{/if}
      <div class="toolbar">
        <div class="search-wrap" bind:this={searchWrapEl}>
          <SearchBar
            bind:searchQuery
            bind:filterActive
            matchCount={searchMatchIds.length}
            currentMatchIndex={searchResultIndex}
            speakerColors={speakerColorMap}
            speakers={speakersList}
            onNavigate={navigateSearchResult}
          />
        </div>
        <div class="toolbar-right">
          <span class="count subtle tiny">
            {lineCount} line{lineCount === 1 ? '' : 's'}{#if hiddenCount > 0} · {hiddenCount} hidden{/if}
          </span>
          <button
            class="btn btn-sm"
            class:active={panelOpen}
            aria-pressed={panelOpen}
            aria-controls="transcript-panel"
            onclick={() => { panelOpen = !panelOpen; }}
          >Panel</button>
        </div>
      </div>
    </div>

    <div class="scroll" bind:this={scrollEl} onscroll={handleScroll} class:with-player={playRequest != null}>
      <div class="transcript">
        {#if loading}
          <div class="placeholder subtle">Loading transcript…</div>
        {:else if displayLines.length > 0}
          {#each displayLines as line (line.id)}
            <TranscriptionLine
              {line}
              {compact}
              {showConfidence}
              isContinuation={line.isContinuation}
              nextIsContinuation={line.nextIsContinuation}
              onEdit={onEditLine}
              onDelete={onDeleteLine}
              onSpeakerClick={() => {}}
              onSelectLines={handleSelectLines}
              onPlayAudio={() => play(line)}
              isPlaying={playRequest?.lineId === line.id}
              speakerColor={speakerColorMap.get(line.discordUsername) ?? ''}
              speakerBadge={speakerBadgeFor(line)}
              searchHighlight={searchMatchIdSet.has(line.id) ? { regex: parsedSearch.regex, isCurrentMatch: line.id === searchHighlightId } : undefined}
            />
          {/each}
        {:else if partials.length === 0}
          <div class="placeholder subtle">{searchQuery && filterActive ? 'No matching lines' : emptyMessage}</div>
        {/if}

        {#if live}
          {#each partials as p (p.userId)}
            <div class="partial" class:compact>
              <span class="partial-speaker" style:color={speakerColorMap.get(p.discordUsername) ?? 'var(--text-2)'}>
                {p.nickname || p.discordUsername}{#if p.speakerLabel && multiSpeakerUsers.has(p.discordUsername)}<span class="partial-badge">{p.speakerLabel}</span>{/if}
              </span>
              <span class="partial-text">{p.text}</span>
            </div>
          {/each}
        {/if}
      </div>
    </div>

    {#if live && !pinned}
      <button class="jump" class:with-player={playRequest != null} onclick={jumpToLive}>&darr; Jump to live</button>
    {/if}
  </section>

  {#if panelOpen}
    <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
    <div class="scrim" onclick={() => { panelOpen = false; }}></div>
  {/if}
  <aside id="transcript-panel" class="panel" class:open={panelOpen} aria-label="Transcript tools">
    <div class="panel-head">
      <span class="slim">Transcript tools</span>
      <button class="icon-btn" aria-label="Close panel" onclick={() => { panelOpen = false; }}>&times;</button>
    </div>

    {#if nicknameEditorOpen}
      <div class="panel-group nickname-editor">
        <h3>Set nicknames <span class="count">({selectedLineIds.length} line{selectedLineIds.length === 1 ? '' : 's'})</span></h3>
        {#each selectedUsernames as username}
          <label class="field">
            <span style:color={speakerColorMap.get(username) ?? 'var(--accent)'}>{username}</span>
            <input
              class="input"
              type="text"
              placeholder="Nickname / character"
              bind:value={nicknameInputs[username]}
              onkeydown={(e) => { if (e.key === 'Enter') saveNicknames(); if (e.key === 'Escape') closeNicknameEditor(); }}
            />
          </label>
        {/each}
        <div class="row-actions">
          <button class="btn-primary btn-sm" onclick={saveNicknames}>Save</button>
          <button class="btn btn-sm" onclick={closeNicknameEditor}>Cancel</button>
        </div>
      </div>
    {/if}

    {#if panelTop}{@render panelTop()}{/if}

    <div class="panel-group">
      <h3>Display</h3>
      <Toggle label="Compact lines" hint="One row per line" bind:checked={compact} />
      <Toggle label="Group by speaker" hint="Merge consecutive lines from the same voice" bind:checked={grouped} />
      <Toggle label="Show confidence" bind:checked={showConfidence} />
      <label class="conf-filter">
        <span class="conf-row"><span>Only show lines at or below</span><span class="conf-val">{minConfidence}%</span></span>
        <input type="range" min="0" max="100" step="5" bind:value={minConfidence} aria-label="Maximum confidence shown" />
        <span class="subtle tiny">{minConfidence === 100 ? 'Showing all lines' : 'Use this to review low-confidence lines'}</span>
      </label>
    </div>

    {#if panelBottom}{@render panelBottom()}{/if}

    <div class="panel-group shortcuts">
      <h3>Shortcuts</h3>
      <dl>
        <dt>/</dt><dd>Search</dd>
        <dt>Ctrl G</dt><dd>Next match</dd>
        <dt>Space</dt><dd>Play / pause</dd>
        <dt>Enter</dt><dd>Save line edit</dd>
        <dt>Drag name</dt><dd>Select lines to set nicknames</dd>
      </dl>
    </div>
  </aside>
</div>

{#if playRequest}
  <AudioPlayerBar
    bind:this={player}
    request={playRequest}
    bind:autoplay
    onended={playNextLine}
    onclose={() => { playRequest = null; }}
  />
{/if}

<style>
  .workspace {
    --panel-w: 320px;
    position: relative; display: grid; grid-template-columns: minmax(0, 1fr);
    height: calc(100vh - var(--header-h) - var(--subnav-h));
    min-height: 0;
  }
  .workspace.panel-open { grid-template-columns: minmax(0, 1fr) var(--panel-w); }

  .main { position: relative; display: flex; flex-direction: column; min-width: 0; min-height: 0; }
  .main-head { padding: 20px 28px 12px; border-bottom: 1px solid var(--border-color); display: flex; flex-direction: column; gap: 14px; }
  .toolbar { display: flex; align-items: center; gap: 12px; }
  .search-wrap { flex: 1; max-width: 560px; }
  .toolbar-right { margin-left: auto; display: flex; align-items: center; gap: 12px; }

  /* Bring the shared SearchBar up to the new control sizing. */
  .search-wrap :global(.search-bar input) {
    min-height: var(--control-h); border-radius: var(--radius); font-size: 14px;
    background: var(--surface-sunken); border-color: var(--border-color); padding-left: 12px;
  }
  .search-wrap :global(.search-bar .filter-toggle) {
    min-height: var(--control-h); min-width: var(--control-h); border-radius: var(--radius);
    background: var(--surface-raised); border-color: var(--border-strong);
  }
  .search-wrap :global(.search-bar .filter-toggle.active) { background: var(--accent-strong); border-color: var(--accent-strong); }
  .search-wrap :global(.search-bar .clear-btn) { font-size: 18px; padding: 0 8px; }

  .scroll { flex: 1; overflow-y: auto; min-height: 0; padding: 8px 28px 24px; }
  .scroll.with-player { padding-bottom: 88px; }
  .transcript { max-width: 1100px; }
  .placeholder { padding: 48px 0; text-align: center; }

  .partial {
    display: grid; grid-template-columns: 40px minmax(0, 1fr); gap: 14px;
    padding: 12px 12px 12px 17px; font-style: italic;
    animation: partial-pulse 1.6s ease-in-out infinite;
  }
  .partial::before { content: ''; }
  .partial-speaker { grid-column: 2; font-weight: 700; font-size: 14px; font-style: normal; }
  .partial-text { grid-column: 2; color: var(--text-2); font-size: 15px; }
  .partial.compact { grid-template-columns: 32px auto minmax(0, 1fr); padding: 4px 10px; gap: 10px; }
  .partial.compact .partial-speaker { grid-column: 2; }
  .partial.compact .partial-text { grid-column: 3; font-size: 14px; }
  .partial-badge {
    margin-left: 6px; padding: 1px 6px; border-radius: 8px; font-size: 10px; font-weight: 700;
    background: color-mix(in srgb, currentColor 18%, transparent);
  }
  @keyframes partial-pulse { 0%, 100% { opacity: 0.75; } 50% { opacity: 0.45; } }

  .jump {
    position: absolute; left: 50%; bottom: 20px; transform: translateX(-50%); z-index: 20;
    min-height: var(--control-h); padding: 0 18px; border-radius: 99px;
    background: var(--accent-strong); color: #fff; border: 0; font-weight: 700; font-size: 14px;
    box-shadow: var(--shadow);
  }
  .jump.with-player { bottom: 84px; }

  .panel {
    display: none; min-height: 0; overflow-y: auto;
    padding: 18px 20px 40px; background: #17191c; border-left: 1px solid var(--border-color);
  }
  .panel.open { display: block; }
  .panel-head { display: flex; align-items: center; justify-content: space-between; margin: -6px 0 10px; }
  .slim { font-size: 11px; letter-spacing: 1.1px; text-transform: uppercase; color: var(--text-3); font-weight: 800; }
  .scrim { display: none; }

  .nickname-editor { display: flex; flex-direction: column; gap: 10px; }
  .row-actions { display: flex; gap: 8px; }

  .conf-filter { display: flex; flex-direction: column; gap: 2px; margin-top: 8px; }
  .conf-row { display: flex; justify-content: space-between; font-size: 14px; }
  .conf-val { color: var(--text-2); font-variant-numeric: tabular-nums; }
  .conf-filter input { width: 100%; }

  .shortcuts dl { display: grid; grid-template-columns: auto 1fr; gap: 6px 12px; font-size: 13px; }
  .shortcuts dt { font-family: var(--font-mono); font-size: 12px; color: var(--text-2); background: var(--surface-raised); border-radius: 4px; padding: 1px 6px; justify-self: start; }
  .shortcuts dd { color: var(--text-3); }

  /* Below ~1100px the panel overlays as a drawer instead of squeezing the transcript. */
  @media (max-width: 1100px) {
    .workspace.panel-open { grid-template-columns: minmax(0, 1fr); }
    .panel.open {
      position: absolute; top: 0; right: 0; bottom: 0; z-index: 35;
      width: min(var(--panel-w), 92vw); box-shadow: var(--shadow);
    }
    .scrim { display: block; position: absolute; inset: 0; z-index: 34; background: rgba(0, 0, 0, 0.45); }
  }
  @media (max-width: 650px) {
    .main-head { padding: 14px 12px 10px; }
    .scroll { padding: 4px 4px 24px; }
    .toolbar { flex-wrap: wrap; }
    .search-wrap { max-width: none; flex-basis: 100%; }
    .search-wrap :global(.search-bar input) { min-width: 0; }
  }
</style>
