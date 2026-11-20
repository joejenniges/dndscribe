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

<div class="hotwords">
  <div class="add-row">
    <input
      class="input"
      type="text"
      bind:value={newWord}
      placeholder="Add a name, place or term…"
      aria-label="New hotword"
      onkeydown={handleAddKeydown}
    />
    <button class="btn-primary" onclick={addWord} disabled={!newWord.trim()}>Add</button>
  </div>
  {#if $hotwords.length === 0}
    <p class="subtle small">No hotwords yet.</p>
  {:else}
    <ul class="word-list">
      {#each $hotwords as word (word)}
        <li class="word">
          {#if editingWord === word}
            <!-- svelte-ignore a11y_autofocus -->
            <input class="input edit" bind:value={editValue} autofocus onkeydown={handleEditKeydown} onblur={saveEdit} aria-label="Edit hotword" />
          {:else}
            <button class="word-text" onclick={() => startEdit(word)} title="Click to edit">{word}</button>
          {/if}
          <button class="remove" onclick={() => removeWord(word)} aria-label="Remove {word}" title="Remove">&times;</button>
        </li>
      {/each}
    </ul>
  {/if}
</div>

<style>
  .hotwords { display: flex; flex-direction: column; gap: 14px; }
  .add-row { display: flex; gap: 8px; max-width: 520px; }
  .word-list { list-style: none; display: flex; flex-wrap: wrap; gap: 8px; }
  .word {
    display: inline-flex; align-items: center; min-height: 38px;
    border: 1px solid var(--border-strong); border-radius: var(--radius-sm); background: var(--surface-sunken);
  }
  .word-text { background: none; border: 0; padding: 0 4px 0 12px; min-height: 36px; font-size: 14px; }
  .word-text:hover { color: var(--accent-text); }
  .edit { min-height: 36px; width: 160px; border: 0; }
  .remove { background: none; border: 0; width: 34px; min-height: 36px; color: var(--text-3); font-size: 18px; }
  .remove:hover { color: var(--danger); }
</style>
