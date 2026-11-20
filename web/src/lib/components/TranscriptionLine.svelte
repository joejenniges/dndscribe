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
    speakerBadge = '',
    compact = false,
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
    // Diarization badge text (display name or "S1"); empty hides the badge.
    // The page decides visibility (only users with >1 detected speaker).
    speakerBadge?: string;
    // Compact density: single-row lines (the pre-redesign layout).
    compact?: boolean;
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
  class:compact
  class:has-audio={hasAudio}
  class:selected
  class:playing={isPlaying}
  class:search-match={isSearchMatch}
  class:search-highlight={isCurrentMatch}
  class:continuation={isContinuation}
  class:group-member={nextIsContinuation}
  bind:this={lineEl}
  data-id={line.id}
  data-discord-username={line.discordUsername}
  data-nickname={line.nickname ?? ''}
>
  <div class="gutter">
    {#if hasAudio}
      <button
        class="play-btn"
        class:playing={isPlaying}
        onclick={togglePlay}
        aria-label={isPlaying ? `Replay audio from ${speaker}` : `Play audio from ${speaker}`}
        title="Play line audio"
      >{isPlaying ? '■' : '▶'}</button>
    {/if}
  </div>

  <div class="content">
    {#if !isContinuation || compact}
      <div class="head" class:invisible={isContinuation}>
        <button
          class="line-speaker"
          style:color={speakerColor || 'var(--accent)'}
          onmousedown={handleSpeakerMousedown}
          onclick={(e) => { e.preventDefault(); if (!speakerDragOccurred) onSpeakerClick(line.discordUsername); speakerDragOccurred = false; }}
          title="Drag across lines to set nicknames"
          tabindex={isContinuation ? -1 : 0}
        >
          {speaker}{#if speakerBadge}<span class="speaker-badge">{speakerBadge}</span>{/if}
        </button>
        <span class="line-time">{time}</span>
        {#if showConfidence && line.confidence != null}
          <span
            class="line-confidence"
            class:high={line.confidence >= 0.8}
            class:med={line.confidence >= 0.5 && line.confidence < 0.8}
            class:low={line.confidence < 0.5}
            title="Transcription confidence"
          >{Math.round(line.confidence * 100)}%</span>
        {/if}
      </div>
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
  </div>

  <div class="line-actions">
    {#if isContinuation && !compact && showConfidence && line.confidence != null}
      <span class="line-confidence subtle-conf" title="Transcription confidence">{Math.round(line.confidence * 100)}%</span>
    {/if}
    <button class="act-btn" class:active={showStats} onclick={() => { showStats = !showStats; }} aria-label="Line details" title="Line details">i</button>
    <button
      class="act-btn danger"
      class:confirm={confirmDelete}
      onclick={handleDelete}
      aria-label={confirmDelete ? 'Confirm delete line' : 'Delete line'}
      title={confirmDelete ? 'Click again to delete' : 'Delete line'}
    >{confirmDelete ? 'Delete?' : '×'}</button>
  </div>

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
  <div class="word-highlight" bind:this={wordHighlightEl}></div>
</div>

<style>
  /* ── Comfortable (default) ── */
  .line {
    position: relative;
    display: grid; grid-template-columns: 40px minmax(0, 1fr) auto; gap: 14px;
    padding: 14px 12px 14px 14px;
    border-bottom: 1px solid #23262b;
    border-left: 3px solid transparent;
  }
  .line:hover { background: var(--surface-sunken); }
  .line.continuation { padding-top: 2px; }
  .line.group-member { border-bottom-color: transparent; padding-bottom: 6px; }
  .line.playing { border-left-color: var(--accent); background: rgba(244, 63, 98, 0.06); }
  .line.selected { background: var(--surface-raised); }
  .line.search-match { background: rgba(116, 215, 169, 0.06); }
  .line.search-highlight { background: rgba(116, 215, 169, 0.14); }
  :global(.line mark) { background: var(--success); color: #0b1a13; font-weight: 600; border-radius: 3px; padding: 0 2px; }
  :global(.line.speaker-selected) { background: rgba(124, 182, 250, 0.12); }

  .gutter { display: flex; justify-content: center; padding-top: 2px; }
  .play-btn {
    width: 36px; height: 36px; border-radius: var(--radius); border: 1px solid transparent;
    background: var(--surface-raised); color: var(--text-2); font-size: 12px; line-height: 1;
    display: grid; place-items: center;
  }
  .play-btn:hover { color: var(--text); background: #30343b; }
  .play-btn.playing { background: var(--accent-soft); color: var(--accent-text); border-color: #5a2b42; }
  .continuation .play-btn { width: 32px; height: 28px; background: transparent; }
  .continuation .play-btn:hover { background: var(--surface-raised); }

  .content { min-width: 0; display: flex; flex-direction: column; gap: 4px; }
  .head { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; min-height: 22px; }
  .head.invisible { visibility: hidden; }
  .line-speaker {
    background: none; border: 0; padding: 0; font-weight: 700; font-size: 14px;
    cursor: pointer; user-select: none; border-radius: 4px;
  }
  .line-speaker:hover { text-decoration: underline; }
  .speaker-badge {
    margin-left: 6px; padding: 1px 6px; border-radius: 8px;
    font-size: 10px; font-weight: 700; vertical-align: 1px;
    background: color-mix(in srgb, currentColor 18%, transparent); color: inherit;
  }
  .line-time { color: var(--text-3); font-size: 12px; white-space: nowrap; font-variant-numeric: tabular-nums; }
  .line-confidence {
    font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 99px;
    background: var(--surface-raised); color: var(--text-2); white-space: nowrap;
  }
  .line-confidence.high { color: var(--success); }
  .line-confidence.med { color: var(--warning); }
  .line-confidence.low { color: var(--danger); background: rgba(255, 107, 129, 0.12); }
  .subtle-conf { background: transparent; color: var(--text-3); }

  .line-text {
    font-size: 15px; line-height: 1.6; color: #e2e5e9;
    cursor: text; padding: 1px 6px; margin-left: -6px; border-radius: 5px; outline: none;
    white-space: pre-wrap; overflow-wrap: anywhere;
  }
  .line-text:hover { background: rgba(255, 255, 255, 0.03); }
  .line-text:focus { background: var(--surface-raised); box-shadow: 0 0 0 1px var(--accent); }

  .line-actions { display: flex; align-items: flex-start; gap: 4px; opacity: 0; transition: opacity 0.1s; }
  .line:hover .line-actions, .line:focus-within .line-actions, .line-actions:has(.active), .line-actions:has(.confirm) { opacity: 1; }
  /* Touch devices have no hover: keep actions visible. */
  @media (hover: none) { .line-actions { opacity: 1; } }
  .act-btn {
    min-width: 32px; height: 32px; padding: 0 8px; border-radius: var(--radius-sm);
    border: 1px solid transparent; background: transparent; color: var(--text-3);
    font-size: 15px; font-weight: 700; line-height: 1;
  }
  .act-btn:hover { background: var(--surface-raised); color: var(--text); }
  .act-btn.active { color: var(--accent-text); background: var(--accent-soft); }
  .act-btn.danger:hover { color: var(--danger); }
  .act-btn.confirm { color: #fff; background: var(--accent-strong); font-size: 12px; }

  /* ── Compact: one row per line ── */
  .line.compact {
    grid-template-columns: 32px minmax(0, 1fr) auto; gap: 8px;
    padding: 4px 10px; align-items: start;
  }
  .line.compact.continuation { padding-top: 0; }
  .compact .gutter { padding-top: 0; }
  .compact .play-btn { width: 30px; height: 28px; font-size: 10px; background: transparent; }
  .compact .play-btn:hover { background: var(--surface-raised); }
  .compact .content { flex-direction: row; align-items: baseline; gap: 10px; }
  .compact .head { flex-wrap: nowrap; flex-shrink: 0; gap: 8px; min-height: 28px; }
  .compact .line-time { order: -1; min-width: 78px; }
  /* Fixed speaker column so text lines up regardless of name length. */
  .compact .line-speaker { width: 120px; text-align: left; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .compact .line-confidence { order: -2; padding: 1px 6px; font-size: 10px; min-width: 38px; text-align: center; }
  .compact .line-text { font-size: 14px; line-height: 1.5; padding-top: 3px; }
  .compact .act-btn { height: 28px; min-width: 28px; }

  .word-highlight {
    display: none; position: fixed;
    background: rgba(244, 63, 98, 0.3); border: 1px solid var(--accent);
    border-radius: 2px; pointer-events: none; z-index: 10;
  }

  .stats-flyout {
    position: absolute; right: 12px; top: calc(100% - 8px); z-index: 50;
    background: var(--surface-raised); border: 1px solid var(--border-strong);
    border-radius: var(--radius); padding: 10px 14px; min-width: 260px;
    box-shadow: var(--shadow); font-size: 13px;
  }
  .stats-row { display: flex; justify-content: space-between; gap: 12px; padding: 3px 0; border-bottom: 1px solid var(--border-color); }
  .stats-row:last-child { border-bottom: none; }
  .stats-label { color: var(--text-3); white-space: nowrap; }
  .stats-file span { font-size: 11px; word-break: break-all; }
  .stats-flyout .high { color: var(--success); }
  .stats-flyout .med { color: var(--warning); }
  .stats-flyout .low { color: var(--danger); font-weight: 600; }

  /* Hide native selection while the word-snap overlay is active */
  :global(.dragging-words)::selection { background: transparent; color: inherit; }
  :global(.dragging-words) *::selection { background: transparent; color: inherit; }

  @media (max-width: 650px) {
    .line { grid-template-columns: 36px minmax(0, 1fr); gap: 10px; padding: 12px 8px; }
    .line-actions { position: absolute; top: 8px; right: 6px; }
    .line.compact .content { flex-direction: column; gap: 2px; }
  }
</style>
