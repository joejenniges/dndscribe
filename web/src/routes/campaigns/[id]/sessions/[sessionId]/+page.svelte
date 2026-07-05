<script lang="ts">
  import { onMount, onDestroy, tick } from 'svelte';
  import { page } from '$app/stores';
  import { scoped } from '$lib/api';
  import { currentCampaign } from '$lib/stores/campaign';
  import { addToast } from '$lib/stores/toast';
  import TranscriptionLine from '$lib/components/TranscriptionLine.svelte';
  import SearchBar from '$lib/components/SearchBar.svelte';
  import ConfidenceFilter from '$lib/components/ConfidenceFilter.svelte';
  import type { TranscriptionLine as TLine } from '$lib/types';

  let campaignId = $derived(Number($page.params.id));
  let sessionId = $derived(Number($page.params.sessionId));
  let api = $derived(scoped(campaignId));

  let allLines: TLine[] = $state([]);
  let searchQuery = $state('');
  let filterActive = $state(false);
  let searchHighlightId = $state<number | null>(null);
  let minConfidence = $state(100);
  let showConfidence = $state(true);
  let condensed = $state(false);

  // ── Nickname editor state ──
  let nicknameEditorOpen = $state(false);
  let selectedLineIds = $state<number[]>([]);
  let selectedUsernames = $state<string[]>([]);
  let nicknameInputs = $state<Record<string, string>>({});

  // ── Inline audio player state ──
  let apAutoplay = $state(false);
  let audioOpen = $state(false);
  let audioFilenames = $state<string[]>([]);
  let audioSpeaker = $state('');
  let audioSpeakerColor = $state('');

  // DOM refs for audio player
  let apCanvasEl: HTMLCanvasElement | undefined = $state(undefined);
  let apWaveformWrap: HTMLDivElement | undefined = $state(undefined);
  let apProgressEl: HTMLDivElement | undefined = $state(undefined);
  let apTimeEl: HTMLSpanElement | undefined = $state(undefined);
  let apPlayBtnEl: HTMLButtonElement | undefined = $state(undefined);

  // Track which line is currently playing
  let playingLineId = $state<number | null>(null);

  // Audio player mutable state (plain variables, not reactive -- updated via setInterval to DOM)
  let apSegments: { audio: HTMLAudioElement; duration: number; startOffset: number }[] = [];
  let apTotalDuration = 0;
  let apCurrentIdx = 0;
  let apIsPlaying = false;
  let apGapMs = 400;
  let apTickInterval: ReturnType<typeof setInterval> | null = null;
  let apReachedEnd = false;

  function apFmt(sec: number): string {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return m + ':' + String(s).padStart(2, '0');
  }

  function apUpdateDisplay() {
    if (!apSegments.length || !apTimeEl || !apProgressEl) return;
    const seg = apSegments[apCurrentIdx];
    if (!seg?.audio) return;
    const globalTime = seg.startOffset + seg.audio.currentTime;
    const pct = apTotalDuration > 0 ? (globalTime / apTotalDuration) * 100 : 0;
    apTimeEl.textContent = apFmt(globalTime) + ' / ' + apFmt(apTotalDuration);
    apProgressEl.style.width = pct + '%';
  }

  async function apLoad(fnames: string[]) {
    apClosePlayer(false);
    apSegments = [];
    apTotalDuration = 0;
    apCurrentIdx = 0;

    const loaded: typeof apSegments = [];
    for (const fname of fnames) {
      const audio = new Audio('/recordings/raw/' + encodeURIComponent(fname));
      await new Promise<void>((resolve) => {
        audio.addEventListener('loadedmetadata', () => resolve(), { once: true });
        audio.addEventListener('error', () => resolve(), { once: true });
        audio.load();
      });
      if (audio.duration && isFinite(audio.duration)) {
        loaded.push({ audio, duration: audio.duration, startOffset: 0 });
      }
    }

    if (loaded.length === 0) {
      if (apTimeEl) apTimeEl.textContent = 'No audio';
      return;
    }

    let off = 0;
    for (let i = 0; i < loaded.length; i++) {
      loaded[i].startOffset = off;
      off += loaded[i].duration;
      if (i < loaded.length - 1) off += apGapMs / 1000;
    }
    apSegments = loaded;
    apTotalDuration = off;

    if (apTimeEl) apTimeEl.textContent = '0:00 / ' + apFmt(apTotalDuration);

    await apRenderWaveform(fnames);
    apStartPlayback();
  }

  function apStartPlayback() {
    if (apSegments.length === 0) return;
    apIsPlaying = true;
    if (apPlayBtnEl) apPlayBtnEl.innerHTML = '&#9646;&#9646;';
    apCurrentIdx = 0;
    apPlaySegment(0);
    if (!apTickInterval) apTickInterval = setInterval(apUpdateDisplay, 200);
  }

  function apPlaySegment(idx: number, seekTime?: number) {
    if (idx >= apSegments.length || !apIsPlaying) {
      apStopPlayback();
      return;
    }
    apCurrentIdx = idx;
    const seg = apSegments[idx];
    if (!seg.audio || !seg.duration) {
      apPlaySegment(idx + 1);
      return;
    }

    seg.audio.onended = null;
    seg.audio.ontimeupdate = null;
    seg.audio.currentTime = seekTime ?? 0;
    seg.audio.play().catch(() => {});

    let advanced = false;
    function advanceToNext() {
      if (advanced) return;
      advanced = true;
      seg.audio.pause();
      seg.audio.ontimeupdate = null;
      seg.audio.onended = null;
      if (!apIsPlaying) return;
      if (idx + 1 < apSegments.length) {
        setTimeout(() => apPlaySegment(idx + 1), apGapMs);
      } else {
        apReachedEnd = true;
        apStopPlayback();
      }
    }

    seg.audio.onended = advanceToNext;
    seg.audio.ontimeupdate = function () {
      if (seg.audio.currentTime >= seg.audio.duration - 0.05) advanceToNext();
    };
  }

  function apStopPlayback() {
    apIsPlaying = false;
    if (apPlayBtnEl) apPlayBtnEl.innerHTML = '&#9654;';
    for (const seg of apSegments) seg.audio.pause();
    if (apReachedEnd) {
      apReachedEnd = false;
      apPlayNextLine();
    }
  }

  function apPlayNextLine() {
    if (!apAutoplay || playingLineId == null) return;

    const currentIdx = displayLines.findIndex(l => l.id === playingLineId);
    if (currentIdx === -1) return;

    for (let i = currentIdx + 1; i < displayLines.length; i++) {
      const nextLine = displayLines[i];
      if (nextLine.audioFilenames && nextLine.audioFilenames.length > 0) {
        const speaker = nextLine.nickname || nextLine.discordUsername;
        const color = speakerColorMap.get(nextLine.discordUsername) ?? '#e94560';
        handlePlayAudio(nextLine.id, nextLine.audioFilenames, speaker, color);

        const el = document.querySelector(`.line[data-id="${nextLine.id}"]`);
        if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
        return;
      }
    }

    // No more lines with audio
    playingLineId = null;
  }

  function apTogglePlayPause() {
    if (apIsPlaying) {
      apStopPlayback();
    } else if (apSegments.length > 0) {
      apIsPlaying = true;
      if (apPlayBtnEl) apPlayBtnEl.innerHTML = '&#9646;&#9646;';
      apPlaySegment(apCurrentIdx);
      if (!apTickInterval) apTickInterval = setInterval(apUpdateDisplay, 200);
    }
  }

  function apSeekTo(targetTime: number) {
    for (let i = 0; i < apSegments.length; i++) {
      const seg = apSegments[i];
      const segEnd = seg.startOffset + seg.duration;
      if (targetTime <= segEnd || i === apSegments.length - 1) {
        for (const s of apSegments) { s.audio.pause(); s.audio.ontimeupdate = null; s.audio.onended = null; }
        const seekPos = Math.max(0, targetTime - seg.startOffset);
        apCurrentIdx = i;
        if (apIsPlaying) {
          apPlaySegment(i, seekPos);
        } else {
          seg.audio.currentTime = seekPos;
        }
        apUpdateDisplay();
        return;
      }
    }
  }

  function apHandleWaveformClick(e: MouseEvent) {
    if (!apWaveformWrap || !apTotalDuration) return;
    const rect = apWaveformWrap.getBoundingClientRect();
    const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    apSeekTo(pct * apTotalDuration);
  }

  async function apRenderWaveform(fnames: string[]) {
    if (!apCanvasEl || !apWaveformWrap) return;
    let audioCtx: AudioContext;
    try { audioCtx = new AudioContext(); } catch { return; }

    const allSamples: Float32Array[] = [];
    for (const fname of fnames) {
      try {
        const resp = await fetch('/recordings/raw/' + encodeURIComponent(fname));
        const buf = await resp.arrayBuffer();
        if (buf.byteLength === 0) continue;
        const decoded = await audioCtx.decodeAudioData(buf);
        allSamples.push(decoded.getChannelData(0));
        if (fnames.indexOf(fname) < fnames.length - 1) {
          allSamples.push(new Float32Array(Math.floor(decoded.sampleRate * 0.4)));
        }
      } catch { /* skip */ }
    }
    audioCtx.close();
    if (allSamples.length === 0) return;

    let totalLen = 0;
    for (const s of allSamples) totalLen += s.length;
    const combined = new Float32Array(totalLen);
    let offset = 0;
    for (const s of allSamples) { combined.set(s, offset); offset += s.length; }

    const dpr = window.devicePixelRatio || 1;
    const w = apWaveformWrap.clientWidth;
    const h = apWaveformWrap.clientHeight;
    apCanvasEl.width = w * dpr;
    apCanvasEl.height = h * dpr;
    const ctx = apCanvasEl.getContext('2d')!;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const barsCount = Math.floor(w / 3);
    const samplesPerBar = Math.floor(combined.length / barsCount);
    const mid = h / 2;

    ctx.fillStyle = '#555';
    for (let b = 0; b < barsCount; b++) {
      const start = b * samplesPerBar;
      const end = start + samplesPerBar;
      let max = 0;
      for (let s = start; s < end && s < combined.length; s++) {
        const abs = Math.abs(combined[s]);
        if (abs > max) max = abs;
      }
      const barH = Math.max(1, max * mid * 0.9);
      ctx.fillRect(b * 3, mid - barH, 2, barH * 2);
    }
  }

  function apClosePlayer(close = true) {
    apReachedEnd = false; // prevent next-line from firing during cleanup
    apStopPlayback();
    if (apTickInterval) { clearInterval(apTickInterval); apTickInterval = null; }
    for (const seg of apSegments) { seg.audio.pause(); seg.audio.src = ''; }
    apSegments = [];
    apTotalDuration = 0;
    if (close) {
      audioOpen = false;
      playingLineId = null;
    }
  }

  onDestroy(() => {
    if (apTickInterval) clearInterval(apTickInterval);
    for (const seg of apSegments) { seg.audio.pause(); seg.audio.src = ''; }
  });

  // When audioOpen becomes true and we have filenames, load audio
  $effect(() => {
    if (audioOpen && audioFilenames.length > 0) {
      const fnames = [...audioFilenames];
      const poll = setInterval(() => {
        if (apTimeEl && apWaveformWrap) {
          clearInterval(poll);
          apLoad(fnames);
        }
      }, 50);
      return () => clearInterval(poll);
    }
  });

  // Speaker colors
  const SPEAKER_COLORS = [
    '#e94560', '#4caf50', '#42a5f5', '#ff9800', '#ab47bc',
    '#26c6da', '#ef5350', '#66bb6a', '#ffa726', '#7e57c2',
    '#ec407a', '#29b6f6', '#9ccc65', '#ffca28', '#8d6e63',
    '#00bcd4', '#e91e63', '#cddc39', '#ff5722', '#3f51b5',
    '#009688', '#f06292', '#aed581', '#ffab40', '#ba68c8',
    '#4dd0e1', '#d4e157', '#ff8a65', '#81c784', '#ce93d8',
  ];

  const speakerColorMap = $derived((() => {
    const map = new Map<string, string>();
    for (const line of allLines) {
      if (!map.has(line.discordUsername)) {
        map.set(line.discordUsername, SPEAKER_COLORS[map.size % SPEAKER_COLORS.length]);
      }
    }
    return map;
  })());

  // Parse search query
  const parsedSearch = $derived((() => {
    const q = searchQuery.trim();
    if (!q) return { speaker: null, regex: null, textQuery: '' };

    let speaker: string | null = null;
    let textPart = q;

    if (q.startsWith('@')) {
      const spaceIdx = q.indexOf(' ');
      if (spaceIdx >= 0) {
        speaker = q.slice(1, spaceIdx).toLowerCase();
        textPart = q.slice(spaceIdx + 1).trim();
      } else {
        speaker = q.slice(1).toLowerCase();
        textPart = '';
      }
    }

    let regex: RegExp | null = null;
    if (textPart) {
      const regexMatch = textPart.match(/^\/(.+?)\/([gimsuy]*)$/);
      if (regexMatch) {
        try {
          regex = new RegExp(regexMatch[1], regexMatch[2].includes('i') ? regexMatch[2] : regexMatch[2] + 'i');
        } catch { /* invalid regex */ }
      }
      if (!regex) {
        const escaped = textPart.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        regex = new RegExp(escaped, 'gi');
      }
    }

    return { speaker, regex, textQuery: textPart };
  })());

  // Speakers list for autocomplete
  const speakersList = $derived((() => {
    const map = new Map<string, Set<string>>();
    for (const line of allLines) {
      if (!map.has(line.discordUsername)) map.set(line.discordUsername, new Set());
      if (line.nickname) map.get(line.discordUsername)!.add(line.nickname);
    }
    return Array.from(map.entries()).map(([username, nicks]) => ({
      username,
      nicknames: Array.from(nicks),
    }));
  })());

  function lineMatchesSearch(l: TLine): boolean {
    const { speaker, regex } = parsedSearch;
    if (speaker) {
      const matchesSpeaker = l.discordUsername.toLowerCase().includes(speaker) ||
        (l.nickname?.toLowerCase().includes(speaker) ?? false);
      if (!matchesSpeaker) return false;
    }
    if (regex) {
      regex.lastIndex = 0;
      const matchesText = regex.test(l.text);
      if (!matchesText && !speaker) {
        regex.lastIndex = 0;
        return regex.test(l.discordUsername) || (l.nickname ? (regex.lastIndex = 0, regex.test(l.nickname)) : false);
      }
      return matchesText;
    }
    return speaker !== null;
  }

  const searchMatchIds = $derived(
    searchQuery.trim() ? allLines.filter(l => lineMatchesSearch(l)).map(l => l.id) : []
  );

  let searchResultIndex = $state(-1);

  $effect(() => {
    searchQuery;
    searchResultIndex = -1;
    searchHighlightId = null;
  });

  const searchMatchIdSet = $derived(new Set(searchMatchIds));

  const filteredLines = $derived(
    allLines.filter(l => {
      if (minConfidence < 100 && l.confidence != null && l.confidence * 100 > minConfidence) return false;
      if (filterActive && searchQuery.trim() && !searchMatchIdSet.has(l.id)) return false;
      return true;
    })
  );

  const displayLines = $derived((() => {
    const result: (TLine & { isContinuation: boolean })[] = [];
    for (const line of filteredLines) {
      const prev = result[result.length - 1];
      const isContinuation = condensed && prev != null
        && prev.discordUsername === line.discordUsername
        && prev.nickname === line.nickname;
      result.push({ ...line, isContinuation });
    }
    return result;
  })());

  onMount(async () => {
    try {
      allLines = await api.lines.list(sessionId);
    } catch (e) {
      addToast(`Failed to load session: ${e}`, 'error');
    }
  });

  async function editLine(id: number, text: string) {
    try {
      const result = await api.lines.update(id, text);
      allLines = allLines.map(l => l.id === id ? { ...l, text } : l);
      if (result.autoAdded.length > 0) {
        addToast(`Auto-added hotwords: ${result.autoAdded.join(', ')}`, 'success');
      }
    } catch (e) {
      addToast(`Failed: ${e}`, 'error');
    }
  }

  async function deleteLine(id: number) {
    try {
      await api.lines.delete(id);
      allLines = allLines.filter(l => l.id !== id);
    } catch (e) {
      addToast(`Failed: ${e}`, 'error');
    }
  }

  function handleSelectLines(lineIds: number[], usernames: string[]) {
    selectedLineIds = lineIds;
    selectedUsernames = usernames;
    nicknameInputs = {};
    for (const u of usernames) {
      const found = allLines.find(l => l.discordUsername === u);
      nicknameInputs[u] = found?.nickname ?? '';
    }
    nicknameEditorOpen = true;
  }

  function closeNicknameEditor() {
    nicknameEditorOpen = false;
    selectedLineIds = [];
    selectedUsernames = [];
    nicknameInputs = {};
  }

  async function saveNicknames() {
    const nicknames: Record<string, string | null> = {};
    for (const u of selectedUsernames) {
      const val = nicknameInputs[u]?.trim();
      nicknames[u] = val || null;
    }
    try {
      await api.bulkNickname(selectedLineIds, nicknames);
      addToast('Nicknames updated', 'success');
      closeNicknameEditor();
      allLines = await api.lines.list(sessionId);
    } catch (e) {
      addToast(`Failed: ${e}`, 'error');
    }
  }

  function navigateSearchResult(direction: 1 | -1) {
    if (searchMatchIds.length === 0) return;
    searchResultIndex = (searchResultIndex + direction + searchMatchIds.length) % searchMatchIds.length;
    const id = searchMatchIds[searchResultIndex];
    searchHighlightId = id;
    const el = document.querySelector(`.line[data-id="${id}"]`);
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  function handlePlayAudio(lineId: number, fnames: string[], speaker: string, color: string) {
    if (audioOpen) apClosePlayer(false);
    playingLineId = lineId;
    audioFilenames = fnames;
    audioSpeaker = speaker;
    audioSpeakerColor = color;
    audioOpen = true;
  }

  function handleKeydown(e: KeyboardEvent) {
    const target = e.target as HTMLElement;
    const isEditing = target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable;

    if (e.key === 'Escape' && nicknameEditorOpen) {
      closeNicknameEditor();
      e.preventDefault();
      return;
    }

    if ((e.key === '/' && !isEditing) || (e.key === 'f' && (e.ctrlKey || e.metaKey))) {
      e.preventDefault();
      const input = document.querySelector('.header-controls .search-bar input') as HTMLInputElement | null;
      if (input) input.focus();
      return;
    }

    if (e.key === 'g' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      navigateSearchResult(e.shiftKey ? -1 : 1);
      return;
    }

    if (isEditing) return;

    const transcriptEl = document.querySelector('.transcript-lines') as HTMLElement | null;
    if (e.key === 'PageDown' && transcriptEl) {
      e.preventDefault();
      transcriptEl.scrollBy({ top: transcriptEl.clientHeight * 0.8, behavior: 'smooth' });
    }
    if (e.key === 'PageUp' && transcriptEl) {
      e.preventDefault();
      transcriptEl.scrollBy({ top: -transcriptEl.clientHeight * 0.8, behavior: 'smooth' });
    }
    if (e.key === 'Home' && transcriptEl) { e.preventDefault(); transcriptEl.scrollTop = 0; }
    if (e.key === 'End' && transcriptEl) { e.preventDefault(); transcriptEl.scrollTop = transcriptEl.scrollHeight; }
  }
</script>

<svelte:window onkeydown={handleKeydown} />

<svelte:head>
  <title>Session #{sessionId} - {$currentCampaign?.name ?? 'Campaign'} - dndscribe</title>
</svelte:head>

<div class="session-page">
  <header>
    <div class="header-left">
      <a href="/campaigns/{campaignId}" class="back-link">&larr;</a>
      <h1>Session #{sessionId}</h1>
      <span class="line-count">{allLines.length} lines</span>
      <a
        class="btn export-btn"
        href="/api/campaigns/{campaignId}/sessions/{sessionId}/export"
        download
        title="Download transcript as .txt"
      >Export</a>
    </div>
    <div class="header-controls">
      <ConfidenceFilter bind:minConfidence bind:condensed bind:showConfidence />
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
  </header>

  {#if nicknameEditorOpen}
    <div class="nickname-editor">
      <div class="nickname-editor-header">
        <h3>Edit Nicknames</h3>
        <span class="nickname-info">{selectedLineIds.length} line{selectedLineIds.length !== 1 ? 's' : ''} selected</span>
        <button class="nickname-close" onclick={closeNicknameEditor}>&times;</button>
      </div>
      <div class="nickname-fields">
        {#each selectedUsernames as username}
          <div class="nickname-row">
            <label class="nickname-label" style:color={speakerColorMap.get(username) ?? 'var(--accent)'}>{username}</label>
            <input
              class="nickname-input"
              type="text"
              placeholder="Nickname..."
              bind:value={nicknameInputs[username]}
              onkeydown={(e) => { if (e.key === 'Enter') saveNicknames(); if (e.key === 'Escape') closeNicknameEditor(); }}
            />
          </div>
        {/each}
      </div>
      <div class="nickname-actions">
        <button class="btn btn-primary" onclick={saveNicknames}>Save</button>
        <button class="btn" onclick={closeNicknameEditor}>Cancel</button>
      </div>
    </div>
  {/if}

  <div class="transcript-lines" style:padding-bottom={audioOpen ? '60px' : '8px'}>
    {#if displayLines.length > 0}
      {#each displayLines as line (line.id)}
        <TranscriptionLine
          {line}
          isContinuation={line.isContinuation}
          {showConfidence}
          onEdit={editLine}
          onDelete={deleteLine}
          onSpeakerClick={() => {}}
          onSelectLines={handleSelectLines}
          onPlayAudio={(fnames) => handlePlayAudio(line.id, fnames, line.nickname || line.discordUsername, speakerColorMap.get(line.discordUsername) ?? '#e94560')}
          isPlaying={playingLineId === line.id}
          speakerColor={speakerColorMap.get(line.discordUsername) ?? ''}
          searchHighlight={searchMatchIdSet.has(line.id) ? { regex: parsedSearch.regex, isCurrentMatch: line.id === searchHighlightId } : undefined}
        />
      {/each}
    {:else}
      <div class="empty-state">
        {searchQuery ? 'No matching lines' : 'No transcription lines in this session'}
      </div>
    {/if}
  </div>

  {#if audioOpen}
    <div class="audio-player">
      <button class="ap-auto-btn" class:active={apAutoplay} onclick={() => { apAutoplay = !apAutoplay; }} title="Auto-play on load">Auto</button>
      <button bind:this={apPlayBtnEl} class="ap-play-pause-btn" onclick={apTogglePlayPause} title="Play/Pause">&#9654;</button>
      <span class="ap-speaker-name" style:color={audioSpeakerColor}>{audioSpeaker}</span>
      <span bind:this={apTimeEl} class="ap-time-display">0:00 / 0:00</span>
      <div bind:this={apWaveformWrap} class="ap-waveform-container" onclick={apHandleWaveformClick} role="slider" tabindex="0" aria-label="Waveform">
        <canvas bind:this={apCanvasEl} class="ap-waveform-canvas"></canvas>
        <div bind:this={apProgressEl} class="ap-waveform-progress" style:background-color={audioSpeakerColor}></div>
      </div>
      <button class="ap-close-btn" onclick={() => apClosePlayer()} title="Close">&times;</button>
    </div>
  {/if}
</div>

<style>
  .session-page { height: 100vh; display: flex; flex-direction: column; }
  header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 12px 20px; background: var(--bg-secondary);
    border-bottom: 1px solid var(--border); gap: 12px; flex-wrap: wrap;
  }
  .header-left { display: flex; align-items: center; gap: 10px; }
  .header-controls { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .back-link { font-size: 18px; color: var(--text-secondary); text-decoration: none; }
  .back-link:hover { color: var(--text-primary); }
  h1 { font-size: 18px; font-weight: 600; color: var(--accent); }
  .line-count { font-size: 13px; color: var(--text-secondary); }
  .export-btn { text-decoration: none; }
  .transcript-lines { flex: 1; overflow-y: auto; padding: 8px 0; }
  .empty-state { padding: 40px; text-align: center; color: var(--text-muted); font-size: 14px; }

  /* Nickname editor */
  .nickname-editor {
    display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
    padding: 8px 16px; background: var(--bg-elevated);
    border-bottom: 1px solid var(--border);
  }
  .nickname-editor-header {
    display: flex; align-items: center; gap: 8px;
  }
  .nickname-editor-header h3 { font-size: 13px; font-weight: 600; margin: 0; white-space: nowrap; }
  .nickname-info { font-size: 11px; color: var(--text-muted); white-space: nowrap; }
  .nickname-close {
    background: none; border: none; color: var(--text-muted); cursor: pointer;
    font-size: 18px; padding: 0 4px; line-height: 1;
  }
  .nickname-close:hover { color: var(--text-primary); }
  .nickname-fields { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .nickname-row { display: flex; align-items: center; gap: 4px; }
  .nickname-label { font-size: 12px; font-weight: 600; white-space: nowrap; }
  .nickname-input {
    background: var(--bg-primary); border: 1px solid var(--border);
    color: var(--text-primary); padding: 3px 8px; border-radius: 3px; font-size: 12px;
    width: 120px;
  }
  .nickname-input:focus { outline: 1px solid var(--accent); border-color: var(--accent); }
  .nickname-actions { display: flex; gap: 4px; }
  .btn { padding: 3px 10px; border: 1px solid var(--border); background: var(--bg-secondary); color: var(--text-primary); border-radius: 3px; cursor: pointer; font-size: 12px; }
  .btn:hover { background: var(--bg-elevated); }
  .btn-primary { background: var(--accent); border-color: var(--accent); color: #fff; }
  .btn-primary:hover { opacity: 0.9; }

  /* Inline audio player -- single row */
  .audio-player {
    position: fixed; bottom: 0; left: 0; right: 0;
    background: var(--bg-secondary); border-top: 1px solid var(--border);
    z-index: 100; display: flex; align-items: center;
    gap: 8px; padding: 0 12px; height: 50px;
  }
  .ap-auto-btn {
    background: var(--bg-elevated); border: 1px solid var(--border-light);
    color: var(--text-muted); cursor: pointer; font-size: 11px;
    padding: 3px 8px; border-radius: 3px; white-space: nowrap;
  }
  .ap-auto-btn.active { color: var(--success); border-color: var(--success); }
  .ap-speaker-name { font-weight: 600; font-size: 13px; white-space: nowrap; }
  .ap-play-pause-btn {
    background: none; border: none; color: var(--text-primary);
    cursor: pointer; font-size: 14px; padding: 2px 6px; line-height: 1;
  }
  .ap-play-pause-btn:hover { color: var(--accent); }
  .ap-time-display {
    color: var(--text-secondary); font-size: 12px; font-family: monospace;
    white-space: nowrap; min-width: 80px;
  }
  .ap-close-btn {
    background: none; border: none; color: var(--text-secondary);
    cursor: pointer; font-size: 18px; padding: 0 4px; line-height: 1;
  }
  .ap-close-btn:hover { color: var(--text-primary); }
  .ap-waveform-container {
    position: relative; flex: 1; height: 36px; cursor: pointer; overflow: hidden;
  }
  .ap-waveform-canvas { width: 100%; height: 100%; display: block; }
  .ap-waveform-progress {
    position: absolute; top: 0; left: 0; bottom: 0;
    opacity: 0.2; pointer-events: none; width: 0%;
  }
</style>
