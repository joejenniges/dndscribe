<script lang="ts">
  import { scoped } from '$lib/api';
  import { hotwords } from '$lib/stores/campaign';
  import { addToast } from '$lib/stores/toast';

  let { campaignId }: { campaignId: number } = $props();

  let newWord = $state('');
  let editingWord = $state<string | null>(null);
  let editValue = $state('');

  const api = $derived(scoped(campaignId));

  async function addWord() {
    const word = newWord.trim();
    if (!word) return;
    try {
      await api.hotwords.add(word);
      newWord = '';
    } catch (e) {
      addToast(`Failed to add hotword: ${e}`, 'error');
    }
  }

  async function removeWord(word: string) {
    try {
      await api.hotwords.remove(word);
    } catch (e) {
      addToast(`Failed to remove: ${e}`, 'error');
    }
  }

  function startEdit(word: string) {
    editingWord = word;
    editValue = word;
  }

  async function saveEdit() {
    if (!editingWord) return;
    const newW = editValue.trim();
    if (newW && newW !== editingWord) {
      try {
        await api.hotwords.update(editingWord, newW);
      } catch (e) {
        addToast(`Failed to update: ${e}`, 'error');
      }
    }
    editingWord = null;
  }

  function handleAddKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter') addWord();
  }

  function handleEditKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter') saveEdit();
    if (e.key === 'Escape') { editingWord = null; }
  }
</script>

<div class="hotwords-panel">
  <div class="panel-header">Hotwords ({$hotwords.length})</div>
  <div class="hotwords-add">
    <input
      type="text"
      bind:value={newWord}
      placeholder="Add hotword..."
      onkeydown={handleAddKeydown}
    />
    <button class="btn-primary" onclick={addWord}>+</button>
  </div>
  <div class="hotwords-list">
    {#each $hotwords as word}
      <div class="hotword-item">
        {#if editingWord === word}
          <input
            class="hotword-edit-input"
            bind:value={editValue}
            onkeydown={handleEditKeydown}
            onblur={saveEdit}
          />
        {:else}
          <span ondblclick={() => startEdit(word)} role="button" tabindex="0">{word}</span>
        {/if}
        <button class="hotword-remove" onclick={() => removeWord(word)}>&times;</button>
      </div>
    {/each}
  </div>
</div>

<style>
  .hotwords-panel { flex: 1; display: flex; flex-direction: column; min-height: 0; }
  .panel-header {
    padding: 8px 16px; font-size: 13px; font-weight: 600;
    color: var(--text-secondary); border-bottom: 1px solid var(--border);
  }
  .hotwords-add {
    display: flex; padding: 8px; gap: 6px; border-bottom: 1px solid var(--border);
  }
  .hotwords-add input {
    flex: 1; background: var(--bg-primary); border: 1px solid var(--border);
    color: var(--text-primary); padding: 6px 10px; border-radius: 4px; font-size: 13px;
  }
  .hotwords-add input:focus { outline: 1px solid var(--accent); border-color: var(--accent); }
  .hotwords-list { flex: 1; overflow-y: auto; padding: 4px 0; }
  .hotword-item {
    display: flex; align-items: center; justify-content: space-between;
    padding: 4px 12px; font-size: 13px;
  }
  .hotword-item:hover { background: var(--bg-primary); }
  .hotword-item span { cursor: pointer; }
  .hotword-item span:hover { text-decoration: underline; }
  .hotword-edit-input {
    background: var(--bg-primary); border: 1px solid var(--accent);
    color: var(--text-primary); padding: 2px 6px; border-radius: 3px;
    font-size: 13px; width: 120px;
  }
  .hotword-remove {
    background: none; border: none; color: var(--text-muted);
    cursor: pointer; font-size: 16px; padding: 0 4px;
  }
  .hotword-remove:hover { color: var(--accent); }
</style>
