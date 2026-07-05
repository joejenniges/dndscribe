<script lang="ts">
  let {
    searchQuery = $bindable(''),
    matchCount = 0,
    currentMatchIndex = -1,
    filterActive = $bindable(false),
    speakerColors = new Map<string, string>(),
    speakers = [] as { username: string; nicknames: string[] }[],
    onNavigate = (_direction: 1 | -1) => {},
  }: {
    searchQuery?: string;
    matchCount?: number;
    currentMatchIndex?: number;
    filterActive?: boolean;
    speakerColors?: Map<string, string>;
    speakers?: { username: string; nicknames: string[] }[];
    onNavigate?: (direction: 1 | -1) => void;
  } = $props();

  let inputEl: HTMLInputElement | undefined = $state();
  let autocompleteOpen = $state(false);
  let autocompleteIndex = $state(0);
  let regexError = $state('');

  // Check if current query has a regex error
  $effect(() => {
    regexError = '';
    const q = searchQuery.trim();
    // Strip @speaker prefix if present
    let textPart = q;
    if (textPart.startsWith('@')) {
      const spaceIdx = textPart.indexOf(' ');
      textPart = spaceIdx >= 0 ? textPart.slice(spaceIdx + 1).trim() : '';
    }
    if (textPart) {
      const regexMatch = textPart.match(/^\/(.+?)\/([gimsuy]*)$/);
      if (regexMatch) {
        try {
          new RegExp(regexMatch[1], regexMatch[2]);
        } catch {
          regexError = 'Invalid regex';
        }
      }
    }
  });

  // Autocomplete: show when input starts with @ and no space yet
  const autocompleteQuery = $derived((() => {
    if (!searchQuery.startsWith('@')) return null;
    const spaceIdx = searchQuery.indexOf(' ');
    if (spaceIdx >= 0) return null; // already selected a speaker
    return searchQuery.slice(1).toLowerCase();
  })());

  const filteredSpeakers = $derived((() => {
    if (autocompleteQuery === null) return [];
    if (autocompleteQuery === '') return speakers;
    return speakers.filter(s =>
      s.username.toLowerCase().includes(autocompleteQuery) ||
      s.nicknames.some(n => n.toLowerCase().includes(autocompleteQuery))
    );
  })());

  $effect(() => {
    autocompleteOpen = filteredSpeakers.length > 0;
    if (filteredSpeakers.length > 0) {
      autocompleteIndex = 0;
    }
  });

  function selectSpeaker(username: string) {
    searchQuery = `@${username} `;
    autocompleteOpen = false;
    inputEl?.focus();
  }

  function handleKeydown(e: KeyboardEvent) {
    // Autocomplete navigation
    if (autocompleteOpen && filteredSpeakers.length > 0) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        autocompleteIndex = (autocompleteIndex + 1) % filteredSpeakers.length;
        return;
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        autocompleteIndex = (autocompleteIndex - 1 + filteredSpeakers.length) % filteredSpeakers.length;
        return;
      }
      if (e.key === 'Enter' || e.key === 'Tab') {
        e.preventDefault();
        selectSpeaker(filteredSpeakers[autocompleteIndex].username);
        return;
      }
    }

    // Search navigation
    if (e.key === 'Enter' || e.key === 'ArrowDown') {
      e.preventDefault();
      onNavigate(1);
      return;
    }
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      onNavigate(-1);
      return;
    }
    if (e.key === 'Escape') {
      e.preventDefault();
      searchQuery = '';
      autocompleteOpen = false;
      inputEl?.blur();
      return;
    }
  }
</script>

<div class="search-bar">
  <div class="search-input-wrap">
    <input
      type="text"
      bind:this={inputEl}
      bind:value={searchQuery}
      placeholder="Search... (/ for regex, @ for speaker)"
      onkeydown={handleKeydown}
    />
    {#if regexError}
      <span class="regex-error">{regexError}</span>
    {:else if searchQuery && matchCount >= 0}
      <span class="match-count">
        {#if currentMatchIndex >= 0}
          {currentMatchIndex + 1}/{matchCount}
        {:else}
          {matchCount} matches
        {/if}
      </span>
    {/if}
    {#if searchQuery}
      <button class="clear-btn" onclick={() => { searchQuery = ''; }}>&times;</button>
    {/if}
  </div>
  <button
    class="filter-toggle"
    class:active={filterActive}
    onclick={() => { filterActive = !filterActive; }}
    title={filterActive ? 'Show all lines' : 'Show only matches'}
  >
    &#x1D50D;
  </button>

  {#if autocompleteOpen}
    <div class="autocomplete">
      {#each filteredSpeakers as speaker, i}
        <button
          class="autocomplete-item"
          class:selected={i === autocompleteIndex}
          onmousedown={(e) => { e.preventDefault(); selectSpeaker(speaker.username); }}
        >
          <span class="speaker-dot" style:background={speakerColors.get(speaker.username) ?? 'var(--accent)'}></span>
          <span class="speaker-name">{speaker.username}</span>
          {#if speaker.nicknames.length > 0}
            <span class="speaker-nicknames">({speaker.nicknames.join(', ')})</span>
          {/if}
        </button>
      {/each}
    </div>
  {/if}
</div>

<style>
  .search-bar { position: relative; display: flex; align-items: center; gap: 4px; }
  .search-input-wrap { position: relative; display: flex; align-items: center; flex: 1; }
  input {
    width: 100%; background: var(--bg-primary); border: 1px solid var(--border);
    color: var(--text-primary); padding: 6px 80px 6px 10px; border-radius: 4px; font-size: 13px;
    min-width: 200px;
  }
  input:focus { outline: 1px solid var(--accent); border-color: var(--accent); }
  .clear-btn {
    position: absolute; right: 4px; background: none; border: none;
    color: var(--text-muted); cursor: pointer; font-size: 16px; padding: 0 4px;
  }
  .clear-btn:hover { color: var(--text-primary); }
  .match-count {
    position: absolute; right: 24px; font-size: 11px; color: var(--text-muted);
    white-space: nowrap; pointer-events: none;
  }
  .regex-error {
    position: absolute; right: 24px; font-size: 11px; color: #ef5350;
    white-space: nowrap; pointer-events: none;
  }
  .filter-toggle {
    background: var(--bg-primary); border: 1px solid var(--border); color: var(--text-muted);
    cursor: pointer; font-size: 14px; padding: 5px 8px; border-radius: 4px; line-height: 1;
  }
  .filter-toggle:hover { color: var(--text-primary); border-color: var(--text-muted); }
  .filter-toggle.active { background: var(--accent); color: #fff; border-color: var(--accent); }

  .autocomplete {
    position: absolute; top: 100%; left: 0; right: 0; z-index: 100;
    background: var(--bg-elevated); border: 1px solid var(--border); border-radius: 4px;
    margin-top: 2px; max-height: 200px; overflow-y: auto;
  }
  .autocomplete-item {
    display: flex; align-items: center; gap: 8px; width: 100%;
    padding: 6px 10px; background: none; border: none; color: var(--text-primary);
    cursor: pointer; font-size: 13px; text-align: left;
  }
  .autocomplete-item:hover, .autocomplete-item.selected { background: var(--bg-secondary); }
  .speaker-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  .speaker-name { font-weight: 600; }
  .speaker-nicknames { color: var(--text-muted); font-size: 12px; }
</style>
