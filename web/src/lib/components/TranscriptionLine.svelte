<script lang="ts">
  import type { TranscriptionLine } from '$lib/types';

  let {
    line,
    onEdit,
    onDelete,
    onSpeakerClick,
    onSelectLines,
    onPlayAudio,
    selected = false,
    isPlaying = false,
    speakerColor = '',
    searchHighlight,
    isContinuation = false,
    nextIsContinuation = false,
    showConfidence = true,
  }: {
    line: TranscriptionLine;
    onEdit: (id: number, text: string) => void;
    onDelete: (id: number) => void;
    onSpeakerClick: (username: string) => void;
    onSelectLines?: (lineIds: number[], usernames: string[]) => void;
    onPlayAudio?: (filenames: string[]) => void;
    selected?: boolean;
    isPlaying?: boolean;
    speakerColor?: string;
    searchHighlight?: { regex: RegExp | null; isCurrentMatch: boolean };
    isContinuation?: boolean;
    nextIsContinuation?: boolean;
    showConfidence?: boolean;
  } = $props();

  // Track whether the user is currently editing (focused on the text span)
  let isEditing = $state(false);
  let showStats = $state(false);

  // Escape HTML entities to prevent XSS when using {@html}
  function escapeHtml(text: string): string {
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // Build highlighted HTML from line text and search regex
  const highlightedHtml = $derived((() => {
    if (!searchHighlight?.regex || isEditing) return '';
    const escaped = escapeHtml(line.text);
    // We need to run the regex on the original text to find match positions,
    // then map those positions to the escaped string.
    // Simpler: run regex on escaped text with an escaped-aware pattern won't work.
    // Instead: find matches in original, build result from escaped segments.
    const re = new RegExp(searchHighlight.regex.source, searchHighlight.regex.flags.includes('g') ? searchHighlight.regex.flags : searchHighlight.regex.flags + 'g');
    const parts: string[] = [];
    let lastIndex = 0;
    let match: RegExpExecArray | null;
    while ((match = re.exec(line.text)) !== null) {
      if (match.index > lastIndex) {
        parts.push(escapeHtml(line.text.slice(lastIndex, match.index)));
      }
      parts.push(`<mark>${escapeHtml(match[0])}</mark>`);
      lastIndex = match.index + match[0].length;
      if (match[0].length === 0) {
        re.lastIndex++; // prevent infinite loop on zero-length match
        if (re.lastIndex > line.text.length) break;
      }
    }
    if (lastIndex < line.text.length) {
      parts.push(escapeHtml(line.text.slice(lastIndex)));
    }
    return parts.join('');
  })());

  const isSearchMatch = $derived(!!searchHighlight?.regex);
  const isCurrentMatch = $derived(!!searchHighlight?.isCurrentMatch);

  let confirmDelete = $state(false);
  let playingIdx = $state(-1);
  let audioEl: HTMLAudioElement | null = $state(null);
  let textEl: HTMLSpanElement | undefined = $state();
  let lineEl: HTMLDivElement | undefined = $state();
  let wordHighlightEl: HTMLDivElement | undefined = $state();

  // Per-line undo stack (max 10)
  let undoStack = $state<string[]>([]);
  let originalText = $state(line.text);

  // Keep originalText in sync when line.text changes from outside
  $effect(() => {
    originalText = line.text;
    undoStack = [];
  });

  const time = $derived(new Date(line.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }));
  // WHY || not ??: backend persists `nickname = ""` (empty string) for lines
  // transcribed while the user had no character name set. Nullish coalescing
  // only falls back on null/undefined, so "" slipped through and rendered as
  // an empty speaker button.
  const speaker = $derived(line.nickname || line.discordUsername);
  const hasAudio = $derived(line.audioFilenames.length > 0);

  function handleFocus() {
    isEditing = true;
    // When entering edit mode, ensure the element has plain text content
    if (textEl) textEl.textContent = line.text;
  }

  function handleBlur() {
    isEditing = false;
    if (!textEl) return;
    const current = textEl.textContent?.trim() ?? '';
    if (current && current !== originalText) {
      pushUndo(originalText);
      onEdit(line.id, current);
    } else {
      // Revert to original if empty or unchanged
      textEl.textContent = originalText;
    }
  }

  function handleTextKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      textEl?.blur();
    }
    if (e.key === 'Escape') {
      if (textEl) textEl.textContent = originalText;
      textEl?.blur();
    }
    if (e.key === 'z' && (e.ctrlKey || e.metaKey) && !e.shiftKey) {
      e.preventDefault();
      if (undoStack.length > 0) {
        const prev = undoStack.pop()!;
        if (textEl) textEl.textContent = prev;
        onEdit(line.id, prev);
      }
    }
  }

  function pushUndo(text: string) {
    undoStack.push(text);
    if (undoStack.length > 10) undoStack.shift();
  }

  function handleDelete() {
    if (!confirmDelete) {
      confirmDelete = true;
      setTimeout(() => { confirmDelete = false; }, 3000);
      return;
    }
    onDelete(line.id);
  }

  function togglePlay() {
    if (!hasAudio || !onPlayAudio) return;
    onPlayAudio(line.audioFilenames);
  }

  // Word-snap selection logic: expand selection to word boundaries during drag within text
  $effect(() => {
    if (!textEl) return;
    const el = textEl;
    let dragging = false;

    function isInsideTextEl(node: Node | null): boolean {
      while (node) {
        if (node === el) return true;
        node = node.parentNode;
      }
      return false;
    }

    function expandToWordBoundaries(sel: Selection) {
      if (!sel.rangeCount) return null;
      const range = sel.getRangeAt(0);
      if (!isInsideTextEl(range.commonAncestorContainer)) return null;

      const text = el.textContent ?? '';
      // Get offsets relative to the text element
      const preRange = document.createRange();
      preRange.selectNodeContents(el);
      preRange.setEnd(range.startContainer, range.startOffset);
      let start = preRange.toString().length;

      const preRange2 = document.createRange();
      preRange2.selectNodeContents(el);
      preRange2.setEnd(range.endContainer, range.endOffset);
      let end = preRange2.toString().length;

      // Expand start to word boundary (go left to whitespace)
      while (start > 0 && !/\s/.test(text[start - 1])) start--;
      // Expand end to word boundary (go right to whitespace)
      while (end < text.length && !/\s/.test(text[end])) end++;

      return { start, end, text: text.slice(start, end) };
    }

    function showHighlight(wordRange: { start: number; end: number } | null) {
      if (!wordHighlightEl || !wordRange) {
        if (wordHighlightEl) wordHighlightEl.style.display = 'none';
        return;
      }
      // Use Range to get bounding rects for the word range
      const textNode = el.firstChild;
      if (!textNode || textNode.nodeType !== Node.TEXT_NODE) {
        wordHighlightEl.style.display = 'none';
        return;
      }
      const r = document.createRange();
      r.setStart(textNode, Math.min(wordRange.start, textNode.textContent!.length));
      r.setEnd(textNode, Math.min(wordRange.end, textNode.textContent!.length));
      const rects = r.getClientRects();
      if (rects.length === 0) {
        wordHighlightEl.style.display = 'none';
        return;
      }
      const first = rects[0];
      const last = rects[rects.length - 1];
      wordHighlightEl.style.display = 'block';
      wordHighlightEl.style.position = 'fixed';
      wordHighlightEl.style.left = `${first.left}px`;
      wordHighlightEl.style.top = `${first.top}px`;
      wordHighlightEl.style.width = `${last.right - first.left}px`;
      wordHighlightEl.style.height = `${last.bottom - first.top}px`;
    }

    function onMouseDown(e: MouseEvent) {
      if (isInsideTextEl(e.target as Node)) {
        dragging = true;
        // Hide native blue selection while our red overlay is showing
        el.classList.add('dragging-words');
      }
    }

    function onSelectionChange() {
      if (!dragging) return;
      const sel = document.getSelection();
      if (!sel || sel.isCollapsed) {
        showHighlight(null);
        return;
      }
      const expanded = expandToWordBoundaries(sel);
      showHighlight(expanded);
    }

    function onMouseUp() {
      if (!dragging) return;
      dragging = false;
      el.classList.remove('dragging-words');
      const sel = document.getSelection();
      if (!sel || sel.isCollapsed) {
        showHighlight(null);
        return;
      }
      const expanded = expandToWordBoundaries(sel);
      showHighlight(null);
      if (expanded && expanded.start !== expanded.end) {
        // Apply expanded selection
        const textNode = el.firstChild;
        if (textNode && textNode.nodeType === Node.TEXT_NODE) {
          const r = document.createRange();
          r.setStart(textNode, Math.min(expanded.start, textNode.textContent!.length));
          r.setEnd(textNode, Math.min(expanded.end, textNode.textContent!.length));
          sel.removeAllRanges();
          sel.addRange(r);
        }
      }
    }

    document.addEventListener('mousedown', onMouseDown);
    document.addEventListener('selectionchange', onSelectionChange);
    document.addEventListener('mouseup', onMouseUp);

    return () => {
      document.removeEventListener('mousedown', onMouseDown);
      document.removeEventListener('selectionchange', onSelectionChange);
      document.removeEventListener('mouseup', onMouseUp);
    };
  });

  // Speaker drag-select: mousedown on speaker starts selection mode
  let speakerDragOccurred = false;
  function handleSpeakerMousedown(e: MouseEvent) {
    e.preventDefault();
    if (!lineEl) return;
    speakerDragOccurred = false;

    const startId = line.id;
    const startUsername = line.discordUsername;
    let currentIds = [startId];
    let currentUsernames = [startUsername];

    function getLineElsInRange(endEl: HTMLElement | null): { ids: number[]; usernames: string[] } {
      const ids: number[] = [];
      const usernames: string[] = [];
      if (!endEl) return { ids: [startId], usernames: [startUsername] };

      // Find the closest .line ancestor
      const endLine = endEl.closest?.('.line') as HTMLElement | null;
      if (!endLine) return { ids: [startId], usernames: [startUsername] };

      const endId = Number(endLine.dataset.id);
      if (!endId) return { ids: [startId], usernames: [startUsername] };

      // Collect all .line elements between start and end
      const allLines = Array.from(document.querySelectorAll('.line[data-id]')) as HTMLElement[];
      const startIdx = allLines.findIndex(el => Number(el.dataset.id) === startId);
      const endIdx = allLines.findIndex(el => Number(el.dataset.id) === endId);
      if (startIdx < 0 || endIdx < 0) return { ids: [startId], usernames: [startUsername] };

      const lo = Math.min(startIdx, endIdx);
      const hi = Math.max(startIdx, endIdx);
      for (let i = lo; i <= hi; i++) {
        const el = allLines[i];
        ids.push(Number(el.dataset.id));
        const u = el.dataset.discordUsername;
        if (u && !usernames.includes(u)) usernames.push(u);
      }
      return { ids, usernames };
    }

    function onMouseMove(e: MouseEvent) {
      speakerDragOccurred = true;
      const target = document.elementFromPoint(e.clientX, e.clientY) as HTMLElement | null;
      const result = getLineElsInRange(target);
      currentIds = result.ids;
      currentUsernames = result.usernames;

      // Visual feedback: add .speaker-selected to lines in range
      document.querySelectorAll('.line.speaker-selected').forEach(el => el.classList.remove('speaker-selected'));
      for (const id of currentIds) {
        const el = document.querySelector(`.line[data-id="${id}"]`);
        if (el) el.classList.add('speaker-selected');
      }
    }

    function onMouseUp() {
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
      document.querySelectorAll('.line.speaker-selected').forEach(el => el.classList.remove('speaker-selected'));
      speakerDragOccurred = true; // Always treat as drag to suppress onclick

      if (onSelectLines && currentIds.length > 0) {
        onSelectLines(currentIds, currentUsernames);
      }
    }

    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  }
</script>

<div
  class="line"
  class:has-audio={hasAudio}
  class:selected
  class:playing={isPlaying}
  class:search-match={isSearchMatch}
  class:search-highlight={isCurrentMatch}
  class:condensed-continuation={isContinuation}
  class:condensed-group-member={nextIsContinuation}
  bind:this={lineEl}
  data-id={line.id}
  data-discord-username={line.discordUsername}
  data-nickname={line.nickname ?? ''}
>
  <span class="line-actions">
    {#if hasAudio}
      <button class="play-btn" class:playing={playingIdx >= 0} onclick={togglePlay}>
        {playingIdx >= 0 ? '\u25A0' : '\u25B6'}
      </button>
    {/if}
    {#if showConfidence && line.confidence != null}
      <span class="line-confidence" class:high={line.confidence >= 0.8} class:med={line.confidence >= 0.5 && line.confidence < 0.8} class:low={line.confidence < 0.5}>
        {Math.round(line.confidence * 100)}%
      </span>
    {/if}
    <button class="delete-btn" class:confirm={confirmDelete} onclick={handleDelete}>
      {confirmDelete ? '\u2716' : '\u00D7'}
    </button>
    <button class="stats-btn" class:active={showStats} onclick={() => { showStats = !showStats; }} title="Audio stats">
      i
    </button>
  </span>

  {#if showStats}
    <div class="stats-flyout">
      <div class="stats-row"><span class="stats-label">ID</span><span>{line.id}</span></div>
      <div class="stats-row"><span class="stats-label">User</span><span>{line.discordUsername}</span></div>
      {#if line.nickname}<div class="stats-row"><span class="stats-label">Nickname</span><span>{line.nickname}</span></div>{/if}
      <div class="stats-row"><span class="stats-label">Confidence</span><span class:low={line.confidence != null && line.confidence < 0.5} class:med={line.confidence != null && line.confidence >= 0.5 && line.confidence < 0.8} class:high={line.confidence != null && line.confidence >= 0.8}>{line.confidence != null ? (line.confidence * 100).toFixed(1) + '%' : 'N/A'}</span></div>
      <div class="stats-row"><span class="stats-label">RMS</span><span>{line.rms != null ? Math.round(line.rms) : 'N/A'}</span></div>
      <div class="stats-row"><span class="stats-label">Duration</span><span>{line.durationMs != null ? (line.durationMs / 1000).toFixed(2) + 's' : 'N/A'}</span></div>
      <div class="stats-row"><span class="stats-label">Audio files</span><span>{line.audioFilenames.length}</span></div>
      {#each line.audioFilenames as f}
        <div class="stats-row stats-file"><span class="stats-label">File</span><span>{f}</span></div>
      {/each}
      <div class="stats-row"><span class="stats-label">Words</span><span>{line.text.split(/\s+/).length}</span></div>
      <div class="stats-row"><span class="stats-label">Chars</span><span>{line.text.length}</span></div>
      {#if line.durationMs && line.text}
        <div class="stats-row"><span class="stats-label">Words/sec</span><span>{(line.text.split(/\s+/).length / (line.durationMs / 1000)).toFixed(1)}</span></div>
      {/if}
      {#if line.rms != null && line.durationMs != null}
        <div class="stats-row"><span class="stats-label">RMS/duration</span><span>{(line.rms / (line.durationMs / 1000)).toFixed(0)}</span></div>
      {/if}
    </div>
  {/if}

  {#if isContinuation}
    <span class="line-time" style="visibility:hidden"></span>
    <span class="line-speaker" style="visibility:hidden">{speaker}</span>
  {:else}
    <span class="line-time">{time}</span>
    <button
      class="line-speaker"
      style:color={speakerColor || 'var(--accent)'}
      onmousedown={handleSpeakerMousedown}
      onclick={(e) => { e.preventDefault(); if (!speakerDragOccurred) onSpeakerClick(line.discordUsername); speakerDragOccurred = false; }}
    >
      {speaker}
    </button>
  {/if}

  {#if highlightedHtml && !isEditing}
    <span
      class="line-text"
      bind:this={textEl}
      contenteditable="true"
      spellcheck="false"
      role="textbox"
      tabindex={0}
      onfocus={handleFocus}
      onblur={handleBlur}
      onkeydown={handleTextKeydown}
    >{@html highlightedHtml}</span>
  {:else}
    <span
      class="line-text"
      bind:this={textEl}
      contenteditable="true"
      spellcheck="false"
      role="textbox"
      tabindex={0}
      onfocus={handleFocus}
      onblur={handleBlur}
      onkeydown={handleTextKeydown}
    >{line.text}</span>
  {/if}
  <div class="word-highlight" bind:this={wordHighlightEl}></div>
</div>

<style>
  .line {
    padding: 6px 16px; font-size: 14px; line-height: 1.5;
    display: flex; gap: 8px; align-items: baseline;
    border-bottom: 1px solid #1e1e1e;
    border-left: 3px solid transparent;
    position: relative;
  }
  .line:hover { background: var(--bg-secondary); }
  .line.has-audio { border-left-color: var(--success); }
  .line.playing { border-left-color: var(--accent); background: rgba(233, 69, 96, 0.05); }
  .line.selected { background: var(--bg-elevated); }
  .line.search-match { background: #1a2a1a; }
  .line.search-highlight { background: #1a331a; }
  :global(mark) { background: #4caf50; color: #000; font-weight: 600; border-radius: 2px; padding: 0 1px; }
  :global(.line.speaker-selected) { background: rgba(66, 165, 245, 0.12); }
  .line-time { color: var(--text-muted); font-size: 12px; white-space: nowrap; min-width: 60px; }
  .line-speaker {
    color: var(--accent); font-weight: 600; white-space: nowrap;
    background: none; border: none; cursor: pointer; padding: 0; font-size: 14px;
    user-select: none;
  }
  .line-speaker:hover { text-decoration: underline; }
  .line-text {
    flex: 1; cursor: text; padding: 0 4px; border-radius: 3px;
    outline: none; min-width: 0;
  }
  .line-text:hover { background: #1e1e1e; }
  .line-text:focus { background: var(--bg-elevated); outline: 1px solid var(--accent); }
  .word-highlight {
    display: none; position: fixed;
    background: rgba(233, 69, 96, 0.3); border: 1px solid #e94560;
    border-radius: 2px; pointer-events: none; z-index: 10;
  }
  .line-actions { display: flex; gap: 4px; align-items: center; white-space: nowrap; }
  .play-btn, .delete-btn {
    background: none; border: none; cursor: pointer; font-size: 14px;
    padding: 0 4px; opacity: 0.5;
  }
  .play-btn { color: var(--success); }
  .play-btn:hover, .play-btn.playing { opacity: 1; }
  .play-btn.playing { color: var(--accent); }
  .delete-btn { color: var(--text-secondary); }
  .delete-btn:hover { opacity: 1; }
  .delete-btn.confirm { color: var(--accent); opacity: 1; }
  .line-confidence { font-size: 10px; padding: 1px 4px; border-radius: 3px; white-space: nowrap; }
  .line-confidence.high { color: #4caf50; }
  .line-confidence.med { color: #ff9800; }
  .line-confidence.low { color: #e94560; font-weight: 600; }
  .condensed-continuation { border-bottom: none; padding-top: 0; }
  .condensed-group-member { border-bottom: none; }
  .stats-btn {
    background: none; border: none; cursor: pointer; font-size: 10px;
    padding: 0 3px; opacity: 0.3; color: var(--text-secondary);
    font-style: italic; font-weight: 700; line-height: 1;
  }
  .stats-btn:hover, .stats-btn.active { opacity: 1; color: var(--accent); }
  .stats-flyout {
    position: absolute; right: 0; top: 100%; z-index: 50;
    background: var(--bg-secondary); border: 1px solid var(--border-light);
    border-radius: 6px; padding: 8px 12px; min-width: 250px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5); font-size: 12px;
  }
  .stats-row {
    display: flex; justify-content: space-between; gap: 12px;
    padding: 2px 0; border-bottom: 1px solid #1e1e1e;
  }
  .stats-row:last-child { border-bottom: none; }
  .stats-label { color: var(--text-muted); white-space: nowrap; }
  .stats-file span { font-size: 10px; word-break: break-all; }
  .stats-flyout .high { color: #4caf50; }
  .stats-flyout .med { color: #ff9800; }
  .stats-flyout .low { color: #e94560; font-weight: 600; }
  /* Hide native blue selection while our red word-snap overlay is active */
  :global(.dragging-words)::selection { background: transparent; color: inherit; }
  :global(.dragging-words) *::selection { background: transparent; color: inherit; }
</style>
