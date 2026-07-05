<script lang="ts">
  import { onMount } from 'svelte';
  import { campaigns as campaignsApi } from '$lib/api';
  import { addToast } from '$lib/stores/toast';
  import type { Campaign } from '$lib/types';

  let campaignList: Campaign[] = $state([]);
  let newName = $state('');
  let newDesc = $state('');
  let creating = $state(false);
  let editingId = $state<number | null>(null);
  let editName = $state('');
  let editDesc = $state('');

  onMount(async () => {
    try {
      campaignList = await campaignsApi.list();
    } catch (e) {
      addToast(`Failed to load campaigns: ${e}`, 'error');
    }
  });

  async function createCampaign() {
    const name = newName.trim();
    if (!name) return;
    creating = true;
    try {
      const c = await campaignsApi.create(name, newDesc.trim() || undefined);
      campaignList = [...campaignList, c];
      newName = '';
      newDesc = '';
      addToast(`Campaign "${c.name}" created`, 'success');
    } catch (e) {
      addToast(`Failed: ${e}`, 'error');
    } finally {
      creating = false;
    }
  }

  function startEdit(c: Campaign) {
    editingId = c.id;
    editName = c.name;
    editDesc = c.description ?? '';
  }

  async function saveEdit() {
    if (editingId == null) return;
    const name = editName.trim();
    if (!name) return;
    try {
      const updated = await campaignsApi.update(editingId, name, editDesc.trim() || undefined);
      if (updated) {
        campaignList = campaignList.map(c => c.id === editingId ? updated : c);
      }
      editingId = null;
    } catch (e) {
      addToast(`Failed: ${e}`, 'error');
    }
  }

  async function deleteCampaign(id: number) {
    if (id === 1) { addToast('Cannot delete default campaign', 'error'); return; }
    try {
      await campaignsApi.delete(id);
      campaignList = campaignList.filter(c => c.id !== id);
      addToast('Campaign deleted', 'success');
    } catch (e) {
      addToast(`Failed: ${e}`, 'error');
    }
  }

  function handleCreateKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter') createCampaign();
  }
</script>

<svelte:head>
  <title>dndscribe - Campaigns</title>
</svelte:head>

<div class="page">
  <header>
    <h1>dndscribe</h1>
  </header>

  <main>
    <h2>Campaigns</h2>

    <div class="campaign-list">
      {#each campaignList as c}
        <a href="/campaigns/{c.id}" class="campaign-card">
          {#if editingId === c.id}
            <div class="campaign-edit" onclick={(e) => e.preventDefault()}>
              <input bind:value={editName} class="edit-input" />
              <input bind:value={editDesc} class="edit-input" placeholder="Description" />
              <div class="edit-actions">
                <button class="btn-primary" onclick={saveEdit}>Save</button>
                <button class="btn" onclick={() => { editingId = null; }}>Cancel</button>
              </div>
            </div>
          {:else}
            <div class="campaign-info">
              <span class="campaign-name">{c.name}</span>
              {#if c.description}
                <span class="campaign-desc">{c.description}</span>
              {/if}
            </div>
            <div class="campaign-actions">
              <button class="btn" onclick={(e) => { e.preventDefault(); startEdit(c); }}>Edit</button>
              {#if c.id !== 1}
                <button class="btn-danger" onclick={(e) => { e.preventDefault(); deleteCampaign(c.id); }}>Delete</button>
              {/if}
            </div>
          {/if}
        </a>
      {/each}
    </div>

    <div class="create-form">
      <h3>New Campaign</h3>
      <input
        type="text"
        bind:value={newName}
        placeholder="Campaign name"
        onkeydown={handleCreateKeydown}
        class="create-input"
      />
      <input
        type="text"
        bind:value={newDesc}
        placeholder="Description (optional)"
        class="create-input"
      />
      <button class="btn-primary" onclick={createCampaign} disabled={creating || !newName.trim()}>
        {creating ? 'Creating...' : 'Create'}
      </button>
    </div>
  </main>
</div>

<style>
  .page { min-height: 100vh; display: flex; flex-direction: column; }
  header {
    padding: 16px 24px; background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
  }
  h1 { font-size: 20px; font-weight: 600; color: var(--accent); }
  main { padding: 24px; max-width: 800px; margin: 0 auto; width: 100%; }
  h2 { font-size: 18px; margin-bottom: 16px; color: var(--text-primary); }
  h3 { font-size: 14px; margin-bottom: 8px; color: var(--text-secondary); }

  .campaign-list { display: flex; flex-direction: column; gap: 8px; margin-bottom: 24px; }
  .campaign-card {
    display: flex; justify-content: space-between; align-items: center;
    padding: 16px; background: var(--bg-secondary); border: 1px solid var(--border);
    border-radius: 6px; text-decoration: none; color: inherit;
    transition: border-color 0.1s;
  }
  .campaign-card:hover { border-color: var(--accent); text-decoration: none; }
  .campaign-info { display: flex; flex-direction: column; gap: 4px; }
  .campaign-name { font-weight: 600; font-size: 16px; }
  .campaign-desc { font-size: 13px; color: var(--text-secondary); }
  .campaign-actions { display: flex; gap: 6px; }

  .campaign-edit { display: flex; flex-direction: column; gap: 6px; width: 100%; }
  .edit-input {
    background: var(--bg-primary); border: 1px solid var(--border);
    color: var(--text-primary); padding: 6px 10px; border-radius: 4px; font-size: 13px;
  }
  .edit-input:focus { outline: 1px solid var(--accent); border-color: var(--accent); }
  .edit-actions { display: flex; gap: 6px; }

  .create-form {
    padding: 16px; background: var(--bg-secondary); border: 1px dashed var(--border);
    border-radius: 6px; display: flex; flex-direction: column; gap: 8px;
  }
  .create-input {
    background: var(--bg-primary); border: 1px solid var(--border);
    color: var(--text-primary); padding: 8px 12px; border-radius: 4px; font-size: 14px;
  }
  .create-input:focus { outline: 1px solid var(--accent); border-color: var(--accent); }
</style>
