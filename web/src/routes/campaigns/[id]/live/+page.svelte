<script lang="ts">
  import { onMount, onDestroy, tick } from 'svelte';
  import { page } from '$app/stores';
  import { scoped, voice, saveRecordings as saveRecApi } from '$lib/api';
  import { currentCampaign, sessions, lines, partials, hotwords, ignoredUsers } from '$lib/stores/campaign';
  import { botStatus, voiceMembers, saveRecordingsEnabled } from '$lib/stores/ws';
  import { addToast } from '$lib/stores/toast';
  import TranscriptionLine from '$lib/components/TranscriptionLine.svelte';
  import ChannelPicker from '$lib/components/ChannelPicker.svelte';
  import SessionPicker from '$lib/components/SessionPicker.svelte';
  import MembersList from '$lib/components/MembersList.svelte';
  import HotwordManager from '$lib/components/HotwordManager.svelte';
  import ConfidenceFilter from '$lib/components/ConfidenceFilter.svelte';
  import SearchBar from '$lib/components/SearchBar.svelte';
  import MergeModal from '$lib/components/MergeModal.svelte';

  let campaignId = $derived(Number($page.params.id));
  let api = $derived(scoped(campaignId));

  let transcriptEl: HTMLDivElement | undefined = $state();
  let autoScroll = $state(true);

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
    apClosePlayer(false); // cleanup without setting audioOpen=false
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

  // close=true sets audioOpen=false (full close), close=false just cleans up internals (for re-load)
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
      // Poll until DOM refs are available (bind:this is async in {#if} blocks)
      const poll = setInterval(() => {
        if (apTimeEl && apWaveformWrap) {
          clearInterval(poll);
          apLoad(fnames);
        }
      }, 50);
      // Cleanup if effect re-runs
      return () => clearInterval(poll);
    }
  });
  let selectedSession = $state<number | undefined>(undefined);
  let searchQuery = $state('');
  let filterActive = $state(false);
  let searchHighlightId = $state<number | null>(null);
  let minConfidence = $state(100);
  let showConfidence = $state(true);
  let condensed = $state(false);
  let mergeOpen = $state(false);
  let showNameToggle = $state(false);

  // Name display mode: 0=nickname, 1=discord, 2=both
  let nameDisplayMode = $state(0);

  // Nickname editor state
  let nicknameEditorOpen = $state(false);
  let selectedLineIds = $state<number[]>([]);
  let selectedUsernames = $state<string[]>([]);
  let nicknameInputs = $state<Record<string, string>>({});

  // Speaker color palette
  const SPEAKER_COLORS = [
    '#e94560', '#4caf50', '#42a5f5', '#ff9800', '#ab47bc',
    '#26c6da', '#ef5350', '#66bb6a', '#ffa726', '#7e57c2',
    '#ec407a', '#29b6f6', '#9ccc65', '#ffca28', '#8d6e63',
    '#00bcd4', '#e91e63', '#cddc39', '#ff5722', '#3f51b5',
    '#009688', '#f06292', '#aed581', '#ffab40', '#ba68c8',
    '#4dd0e1', '#d4e157', '#ff8a65', '#81c784', '#ce93d8',
  ];

  // Map discord usernames to colors (first-seen order)
  const speakerColorMap = $derived((() => {
    const map = new Map<string, string>();
    for (const line of $lines) {
      if (!map.has(line.discordUsername)) {
        map.set(line.discordUsername, SPEAKER_COLORS[map.size % SPEAKER_COLORS.length]);
      }
    }
    return map;
  })());

  // Parse search query into speaker filter and text/regex pattern
  const parsedSearch = $derived((() => {
    const q = searchQuery.trim();
    if (!q) return { speaker: null, regex: null, textQuery: '' };

    let speaker: string | null = null;
    let textPart = q;

    // Parse @speaker prefix
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

    // Parse regex: /pattern/ or /pattern/flags
    let regex: RegExp | null = null;
    if (textPart) {
      const regexMatch = textPart.match(/^\/(.+?)\/([gimsuy]*)$/);
      if (regexMatch) {
        try {
          regex = new RegExp(regexMatch[1], regexMatch[2].includes('i') ? regexMatch[2] : regexMatch[2] + 'i');
        } catch {
          // Invalid regex, fall through to plain text
        }
      }
      if (!regex) {
        // Plain text search - escape for regex use, case insensitive
        const escaped = textPart.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        regex = new RegExp(escaped, 'gi');
      }
    }

    return { speaker, regex, textQuery: textPart };
  })());

  // Build unique speakers list for autocomplete
  const speakersList = $derived((() => {
    const map = new Map<string, Set<string>>();
    for (const line of $lines) {
      if (!map.has(line.discordUsername)) {
        map.set(line.discordUsername, new Set());
      }
      if (line.nickname) {
        map.get(line.discordUsername)!.add(line.nickname);
      }
    }
    return Array.from(map.entries()).map(([username, nicks]) => ({
      username,
      nicknames: Array.from(nicks),
    }));
  })());

  // Test if a line matches the current search
  function lineMatchesSearch(l: typeof $lines[number]): boolean {
    const { speaker, regex } = parsedSearch;
    if (speaker) {
      const matchesSpeaker = l.discordUsername.toLowerCase().includes(speaker) ||
        (l.nickname?.toLowerCase().includes(speaker) ?? false);
      if (!matchesSpeaker) return false;
    }
    if (regex) {
      regex.lastIndex = 0;
      const matchesText = regex.test(l.text);
      // For non-@ queries, also match against speaker name
      if (!matchesText && !speaker) {
        regex.lastIndex = 0;
        const matchesSpeakerText = regex.test(l.discordUsername) || (l.nickname ? (regex.lastIndex = 0, regex.test(l.nickname)) : false);
        return matchesSpeakerText;
      }
      return matchesText;
    }
    // If only @speaker with no text, match on speaker alone
    return speaker !== null;
  }

  const searchMatchIds = $derived(
    searchQuery.trim()
      ? $lines.filter(l => lineMatchesSearch(l)).map(l => l.id)
      : []
  );

  let searchResultIndex = $state(-1);

  // Reset search result index when query changes
  $effect(() => {
    // Access searchQuery to create dependency
    searchQuery;
    searchResultIndex = -1;
    searchHighlightId = null;
  });

  const searchMatchIdSet = $derived(new Set(searchMatchIds));

  const filteredLines = $derived(
    $lines.filter(l => {
      if (minConfidence < 100 && l.confidence != null && l.confidence * 100 > minConfidence) return false;
      if (filterActive && searchQuery.trim() && !searchMatchIdSet.has(l.id)) return false;
      return true;
    })
  );

  // Mark consecutive same-speaker lines as continuations (for condensed view)
  // Also mark if a line is followed by a continuation (to remove its bottom border)
  const displayLines = $derived((() => {
    const result: (typeof filteredLines[number] & { isContinuation: boolean; nextIsContinuation: boolean })[] = [];
    for (const line of filteredLines) {
      const prev = result[result.length - 1];
      const isContinuation = condensed && prev != null
        && prev.discordUsername === line.discordUsername
        && prev.nickname === line.nickname;
      result.push({ ...line, isContinuation, nextIsContinuation: false });
      // Mark previous line as followed by continuation
      if (isContinuation && prev) {
        prev.nextIsContinuation = true;
      }
    }
    return result;
  })());

  // Live partials (streaming engine), one per active speaker.
  const partialList = $derived([...$partials.values()]);

  // Auto-scroll when new lines or partials arrive
  $effect(() => {
    // Touch both lengths so partial updates also keep the view pinned to bottom.
    const _ = displayLines.length + partialList.length;
    if (_ > 0 && autoScroll && transcriptEl) {
      tick().then(() => {
        if (transcriptEl) transcriptEl.scrollTop = transcriptEl.scrollHeight;
      });
    }
  });

  function handleScroll() {
    if (!transcriptEl) return;
    const { scrollTop, scrollHeight, clientHeight } = transcriptEl;
    autoScroll = scrollHeight - scrollTop - clientHeight < 50;
  }

  async function loadSession(sessionId: number) {
    selectedSession = sessionId;
    try {
      const data = await api.lines.list(sessionId);
      lines.set(data);
    } catch (e) {
      addToast(`Failed to load session: ${e}`, 'error');
    }
  }

  async function deleteSession(sessionId: number) {
    try {
      await api.sessions.delete(sessionId);
      addToast('Session deleted', 'success');
    } catch (e) {
      addToast(`Failed: ${e}`, 'error');
    }
  }

  async function renameSession(sessionId: number, name: string) {
    try {
      await api.sessions.update(sessionId, name);
      sessions.update(list => list.map(s => s.id === sessionId ? { ...s, channelName: name } : s));
      addToast('Session renamed', 'success');
    } catch (e) {
      addToast(`Failed to rename: ${e}`, 'error');
    }
  }

  async function editLine(id: number, text: string) {
    try {
      const result = await api.lines.update(id, text);
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
    } catch (e) {
      addToast(`Failed: ${e}`, 'error');
    }
  }

  async function togglePause() {
    try {
      const result = await voice.pause();
      addToast(result.paused ? 'Paused' : 'Resumed', 'info');
    } catch (e) {
      addToast(`Failed: ${e}`, 'error');
    }
  }

  async function toggleSaveRecordings() {
    try {
      const result = await saveRecApi.set(!$saveRecordingsEnabled);
      saveRecordingsEnabled.set(result.enabled);
    } catch (e) {
      addToast(`Failed: ${e}`, 'error');
    }
  }

  function handleSpeakerClick(username: string) {
    showNameToggle = !showNameToggle;
  }

  async function unignoreUser(userId: string) {
    try {
      await api.ignoredUsers.remove(userId);
      ignoredUsers.update(u => u.filter(i => i.discordUserId !== userId));
      addToast('User unignored', 'success');
    } catch (e) {
      addToast(`Failed: ${e}`, 'error');
    }
  }

  function handlePlayAudio(lineId: number, fnames: string[], speaker: string, color: string) {
    // Close existing player first if open (cleans up audio elements)
    if (audioOpen) apClosePlayer(false);
    playingLineId = lineId;
    audioFilenames = fnames;
    audioSpeaker = speaker;
    audioSpeakerColor = color;
    audioOpen = true;
  }

  function handleSelectLines(lineIds: number[], usernames: string[]) {
    selectedLineIds = lineIds;
    selectedUsernames = usernames;
    nicknameInputs = {};
    for (const u of usernames) {
      const found = $lines.find(l => l.discordUsername === u);
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
      if (selectedSession != null) {
        const data = await api.lines.list(selectedSession);
        lines.set(data);
      }
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

  // Keyboard shortcuts
  function handleKeydown(e: KeyboardEvent) {
    const target = e.target as HTMLElement;
    const isEditing = target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable;

    // Escape always closes nickname editor
    if (e.key === 'Escape' && nicknameEditorOpen) {
      closeNicknameEditor();
      e.preventDefault();
      return;
    }

    // Ctrl+F or / focuses search
    if ((e.key === '/' && !isEditing) || (e.key === 'f' && (e.ctrlKey || e.metaKey))) {
      e.preventDefault();
      const input = document.querySelector('.header-controls .search-bar input') as HTMLInputElement | null;
      if (input) input.focus();
      return;
    }

    // Ctrl+G / Ctrl+Shift+G: navigate search results
    if (e.key === 'g' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      navigateSearchResult(e.shiftKey ? -1 : 1);
      return;
    }

    // Skip remaining shortcuts when editing
    if (isEditing) return;

    // PageDown / PageUp: scroll transcript by 80% of viewport
    if (e.key === 'PageDown' && transcriptEl) {
      e.preventDefault();
      transcriptEl.scrollBy({ top: transcriptEl.clientHeight * 0.8, behavior: 'smooth' });
      return;
    }
    if (e.key === 'PageUp' && transcriptEl) {
      e.preventDefault();
      transcriptEl.scrollBy({ top: -transcriptEl.clientHeight * 0.8, behavior: 'smooth' });
      return;
    }

    // Home / End: jump to top/bottom
    if (e.key === 'Home' && transcriptEl) {
      e.preventDefault();
      transcriptEl.scrollTop = 0;
      return;
    }
    if (e.key === 'End' && transcriptEl) {
      e.preventDefault();
      transcriptEl.scrollTop = transcriptEl.scrollHeight;
      return;
    }

    // 1/2/3: cycle name display mode
    if (e.key === '1') { nameDisplayMode = 0; addToast('Names: nickname', 'info'); return; }
    if (e.key === '2') { nameDisplayMode = 1; addToast('Names: discord', 'info'); return; }
    if (e.key === '3') { nameDisplayMode = 2; addToast('Names: both', 'info'); return; }

    // Space: toggle play/pause on audio player
    if (e.key === ' ' && audioOpen) {
      e.preventDefault();
      apTogglePlayPause();
      return;
    }
  }

  // Auto-select first session on load
  $effect(() => {
    if ($sessions.length > 0 && selectedSession == null) {
      loadSession($sessions[0].id);
    }
  });
</script>

<svelte:window onkeydown={handleKeydown} />

<svelte:head>
  <title>Live - {$currentCampaign?.name ?? 'Campaign'} - dndscribe</title>
</svelte:head>

<div class="live-page">
  <header>
    <div class="header-left">
      <a href="/campaigns/{campaignId}" class="back-link">&larr;</a>
      <h1>{$currentCampaign?.name ?? ''}</h1>
      <div class="recording-status">
        <span class="rec-dot" class:active={$botStatus.recording}></span>
        <span>{$botStatus.recording ? $botStatus.channelName : 'Not recording'}</span>
      </div>
    </div>
    <div class="header-controls">
      <ChannelPicker {campaignId} />
      {#if $botStatus.recording}
        <button class="btn" onclick={togglePause}>Pause</button>
      {/if}
      <label class="save-toggle" class:active={$saveRecordingsEnabled}>
        <input type="checkbox" checked={$saveRecordingsEnabled} onchange={toggleSaveRecordings} />
        Save audio
      </label>
      <SessionPicker bind:value={selectedSession} onchange={loadSession} ondelete={deleteSession} onmerge={() => { mergeOpen = true; }} onrename={renameSession} />
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

  <div class="container">
    <div class="transcript-panel">
      <div class="transcript-lines" style:padding-bottom={audioOpen ? '60px' : '8px'} bind:this={transcriptEl} onscroll={handleScroll}>
        {#if displayLines.length > 0}
          {#each displayLines as line (line.id)}
            <TranscriptionLine
              {line}
              isContinuation={line.isContinuation}
              nextIsContinuation={line.nextIsContinuation}
              {showConfidence}
              onEdit={editLine}
              onDelete={deleteLine}
              onSpeakerClick={handleSpeakerClick}
              onSelectLines={handleSelectLines}
              onPlayAudio={(fnames) => { try { handlePlayAudio(line.id, fnames, line.nickname || line.discordUsername, speakerColorMap.get(line.discordUsername) ?? '#e94560'); } catch(e) { console.error('PLAY ERROR:', e); } }}
              isPlaying={playingLineId === line.id}
              speakerColor={speakerColorMap.get(line.discordUsername) ?? ''}
              searchHighlight={searchMatchIdSet.has(line.id) ? { regex: parsedSearch.regex, isCurrentMatch: line.id === searchHighlightId } : undefined}
            />
          {/each}
        {:else if partialList.length === 0}
          <div class="empty-state">
            {searchQuery ? 'No matching lines' : 'No transcription lines yet'}
          </div>
        {/if}

        {#each partialList as p (p.userId)}
          <div class="partial-line">
            <span class="partial-speaker" style:color={speakerColorMap.get(p.discordUsername) ?? '#8a8a99'}>
              {p.nickname || p.discordUsername}
            </span>
            <span class="partial-text">{p.text}</span>
          </div>
        {/each}
      </div>
    </div>

    <div class="sidebar">
      {#if nicknameEditorOpen}
        <div class="nickname-editor">
          <div class="nickname-editor-header">
            <h3>Edit Nicknames</h3>
            <button class="nickname-close" onclick={closeNicknameEditor}>&times;</button>
          </div>
          <p class="nickname-info">{selectedLineIds.length} line{selectedLineIds.length !== 1 ? 's' : ''} selected</p>
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
          <div class="nickname-actions">
            <button class="btn btn-primary" onclick={saveNicknames}>Save</button>
            <button class="btn" onclick={closeNicknameEditor}>Cancel</button>
          </div>
        </div>
      {/if}
      <MembersList {campaignId} />
      {#if $ignoredUsers.length > 0}
        <div class="ignored-panel">
          <div class="panel-header">Ignored ({$ignoredUsers.length})</div>
          <div class="ignored-list">
            {#each $ignoredUsers as u}
              <div class="ignored-item">
                <span>{u.discordUsername}</span>
                <button class="unignore-btn" onclick={() => unignoreUser(u.discordUserId)} title="Unignore">&check;</button>
              </div>
            {/each}
          </div>
        </div>
      {/if}
      <HotwordManager {campaignId} />
    </div>
  </div>

  <MergeModal {campaignId} bind:open={mergeOpen} />
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
  .live-page { height: 100vh; display: flex; flex-direction: column; }
  header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 20px; background: var(--bg-secondary);
    border-bottom: 1px solid var(--border); gap: 12px; flex-wrap: wrap;
  }
  .header-left { display: flex; align-items: center; gap: 10px; }
  .header-controls { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
  .back-link { font-size: 18px; color: var(--text-secondary); text-decoration: none; }
  .back-link:hover { color: var(--text-primary); }
  h1 { font-size: 18px; font-weight: 600; color: var(--accent); white-space: nowrap; }

  .recording-status { display: flex; align-items: center; gap: 8px; font-size: 13px; color: var(--text-secondary); }
  .rec-dot { width: 10px; height: 10px; border-radius: 50%; background: var(--text-muted); }
  .rec-dot.active { background: var(--accent); animation: pulse 1.5s ease-in-out infinite; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

  .save-toggle { display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--text-secondary); white-space: nowrap; }
  .save-toggle input[type="checkbox"] { cursor: pointer; }
  .save-toggle.active { color: var(--success); }

  .container { display: flex; flex: 1; overflow: hidden; }

  .transcript-panel { flex: 1; display: flex; flex-direction: column; }
  .transcript-lines { flex: 1; overflow-y: auto; padding: 8px 0; }
  .empty-state { padding: 40px; text-align: center; color: var(--text-muted); font-size: 14px; }

  /* Live streaming partials -- provisional, dim/italic, not yet committed. */
  .partial-line {
    display: flex; gap: 8px; align-items: baseline;
    padding: 2px 16px; font-style: italic; opacity: 0.65;
    animation: partial-pulse 1.6s ease-in-out infinite;
  }
  .partial-speaker { font-weight: 600; font-size: 13px; white-space: nowrap; }
  .partial-text { color: var(--text-secondary); font-size: 14px; }
  @keyframes partial-pulse { 0%, 100% { opacity: 0.65; } 50% { opacity: 0.4; } }

  .sidebar {
    width: 280px; min-width: 280px; max-width: 280px;
    display: flex; flex-direction: column; background: var(--bg-secondary); overflow: hidden;
  }

  /* Ignored users panel */
  .ignored-panel { border-bottom: 1px solid var(--border); }
  .panel-header {
    padding: 8px 16px; font-size: 13px; font-weight: 600;
    color: var(--text-secondary); border-bottom: 1px solid var(--border);
  }
  .ignored-list { max-height: 120px; overflow-y: auto; padding: 4px 0; }
  .ignored-item {
    display: flex; align-items: center; justify-content: space-between;
    padding: 4px 12px; font-size: 12px; color: var(--text-secondary);
  }
  .ignored-item:hover { background: var(--bg-primary); }
  .unignore-btn {
    background: none; border: none; color: var(--text-muted);
    cursor: pointer; font-size: 14px; padding: 0 4px;
  }
  .unignore-btn:hover { color: var(--success); }

  /* Nickname editor */
  .nickname-editor {
    padding: 12px; border-bottom: 1px solid var(--border);
    background: var(--bg-elevated);
  }
  .nickname-editor-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 8px;
  }
  .nickname-editor-header h3 { font-size: 14px; font-weight: 600; margin: 0; }
  .nickname-close {
    background: none; border: none; color: var(--text-muted); cursor: pointer;
    font-size: 18px; padding: 0 4px; line-height: 1;
  }
  .nickname-close:hover { color: var(--text-primary); }
  .nickname-info { font-size: 11px; color: var(--text-muted); margin: 0 0 8px; }
  .nickname-row { margin-bottom: 6px; }
  .nickname-label { display: block; font-size: 12px; font-weight: 600; margin-bottom: 2px; }
  .nickname-input {
    width: 100%; background: var(--bg-primary); border: 1px solid var(--border);
    color: var(--text-primary); padding: 4px 8px; border-radius: 3px; font-size: 13px;
    box-sizing: border-box;
  }
  .nickname-input:focus { outline: 1px solid var(--accent); border-color: var(--accent); }
  .nickname-actions { display: flex; gap: 6px; margin-top: 8px; }
  .btn { padding: 4px 12px; border: 1px solid var(--border); background: var(--bg-secondary); color: var(--text-primary); border-radius: 3px; cursor: pointer; font-size: 12px; }
  .btn:hover { background: var(--bg-elevated); }
  .btn-primary { background: var(--accent); border-color: var(--accent); color: #fff; }
  .btn-primary:hover { opacity: 0.9; }

  @media (max-width: 900px) {
    .sidebar { display: none; }
  }

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
