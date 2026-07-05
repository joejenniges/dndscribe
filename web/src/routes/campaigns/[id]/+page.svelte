<script lang="ts">
  import { page } from '$app/stores';
  import { currentCampaign, sessions, ignoredUsers } from '$lib/stores/campaign';
  import { botStatus } from '$lib/stores/ws';
  import { scoped } from '$lib/api';
  import { addToast } from '$lib/stores/toast';
  import ChannelPicker from '$lib/components/ChannelPicker.svelte';
  import type { Session } from '$lib/types';

  let campaignId = $derived(Number($page.params.id));
  let editingSessionId = $state<number | null>(null);
  let editingName = $state('');

  function formatSession(s: Session): string {
    const d = new Date(s.startedAt);
    const date = d.toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' });
    const time = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    return `${date} ${time}`;
  }

  function startEditing(s: Session) {
    editingSessionId = s.id;
    editingName = s.channelName ?? '';
  }

  async function saveSessionName(sessionId: number) {
    const name = editingName.trim();
    if (!name) { editingSessionId = null; return; }
    try {
      await scoped(campaignId).sessions.update(sessionId, name);
      sessions.update(list => list.map(s => s.id === sessionId ? { ...s, channelName: name } : s));
    } catch (e) {
      addToast(`Failed to rename session: ${e}`, 'error');
    }
    editingSessionId = null;
  }

  function handleEditKeydown(e: KeyboardEvent, sessionId: number) {
    if (e.key === 'Enter') {
      e.preventDefault();
      saveSessionName(sessionId);
    } else if (e.key === 'Escape') {
      editingSessionId = null;
    }
  }

  let confirmDeleteSessionId = $state<number | null>(null);

  async function deleteSession(sessionId: number) {
    if (confirmDeleteSessionId !== sessionId) {
      confirmDeleteSessionId = sessionId;
      setTimeout(() => { confirmDeleteSessionId = null; }, 3000);
      return;
    }
    try {
      await scoped(campaignId).sessions.delete(sessionId);
      sessions.update(list => list.filter(s => s.id !== sessionId));
      addToast('Session deleted', 'success');
    } catch (e) {
      addToast(`Failed to delete: ${e}`, 'error');
    }
    confirmDeleteSessionId = null;
  }

  async function unignore(userId: string) {
    try {
      await scoped(campaignId).ignoredUsers.remove(userId);
      ignoredUsers.update(u => u.filter(i => i.discordUserId !== userId));
    } catch (e) {
      addToast(`Failed: ${e}`, 'error');
    }
  }
</script>

<svelte:head>
  <title>{$currentCampaign?.name ?? 'Campaign'} - dndscribe</title>
</svelte:head>

<div class="page">
  <header>
    <div class="header-left">
      <a href="/" class="back-link">&larr;</a>
      <h1>{$currentCampaign?.name ?? 'Loading...'}</h1>
      {#if $currentCampaign?.description}
        <span class="campaign-desc">{$currentCampaign.description}</span>
      {/if}
    </div>
    <div class="header-right">
      <ChannelPicker {campaignId} />
      <a href="/campaigns/{campaignId}/live" class="btn-primary">Live View</a>
    </div>
  </header>

  <main>
    <section>
      <h2>Sessions ({$sessions.length})</h2>
      {#if $sessions.length > 0}
        <div class="session-list">
          {#each $sessions as s}
            <div class="session-card">
              <a href="/campaigns/{campaignId}/sessions/{s.id}" class="session-link">
                <span class="session-id">#{s.id}</span>
                <span class="session-date">{formatSession(s)}</span>
              </a>
              <div class="session-meta">
                {#if editingSessionId === s.id}
                  <input
                    class="session-name-input"
                    type="text"
                    bind:value={editingName}
                    onkeydown={(e) => handleEditKeydown(e, s.id)}
                    onblur={() => saveSessionName(s.id)}
                    autofocus
                  />
                {:else}
                  <button class="session-name-btn" onclick={() => startEditing(s)} title="Click to rename">
                    {s.channelName || 'Unnamed'}
                  </button>
                {/if}
                <span class="session-status">
                  {s.endedAt ? 'Ended' : 'Active'}
                </span>
                <button
                  class="session-delete-btn"
                  class:confirm={confirmDeleteSessionId === s.id}
                  onclick={() => deleteSession(s.id)}
                  title={confirmDeleteSessionId === s.id ? 'Click again to confirm' : 'Delete session'}
                >
                  {confirmDeleteSessionId === s.id ? '?' : '\u00D7'}
                </button>
              </div>
            </div>
          {/each}
        </div>
      {:else}
        <p class="empty">No sessions yet. Join a voice channel to start recording.</p>
      {/if}
    </section>

    <section>
      <h2>Ignored Users ({$ignoredUsers.length})</h2>
      {#if $ignoredUsers.length > 0}
        <div class="ignored-list">
          {#each $ignoredUsers as u}
            <div class="ignored-item">
              <span>{u.discordUsername}</span>
              <button class="btn" onclick={() => unignore(u.discordUserId)}>Unignore</button>
            </div>
          {/each}
        </div>
      {:else}
        <p class="empty">No ignored users.</p>
      {/if}
    </section>
  </main>
</div>

<style>
  .page { min-height: 100vh; display: flex; flex-direction: column; }
  header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 12px 20px; background: var(--bg-secondary);
    border-bottom: 1px solid var(--border); gap: 12px; flex-wrap: wrap;
  }
  .header-left { display: flex; align-items: center; gap: 12px; }
  .header-right { display: flex; align-items: center; gap: 8px; }
  .back-link { font-size: 18px; color: var(--text-secondary); text-decoration: none; }
  .back-link:hover { color: var(--text-primary); }
  h1 { font-size: 18px; font-weight: 600; color: var(--accent); }
  .campaign-desc { font-size: 13px; color: var(--text-secondary); }
  main { padding: 24px; max-width: 900px; margin: 0 auto; width: 100%; }
  section { margin-bottom: 24px; }
  h2 { font-size: 16px; margin-bottom: 12px; color: var(--text-primary); }
  .empty { color: var(--text-muted); font-size: 13px; }

  .session-list { display: flex; flex-direction: column; gap: 6px; }
  .session-card {
    display: flex; justify-content: space-between; align-items: center;
    padding: 12px 16px; background: var(--bg-secondary); border: 1px solid var(--border);
    border-radius: 4px;
  }
  .session-card:hover { border-color: var(--accent); }
  .session-link {
    display: flex; align-items: center; gap: 10px;
    text-decoration: none; color: inherit;
  }
  .session-link:hover { text-decoration: none; }
  .session-id { color: var(--text-muted); font-size: 12px; }
  .session-date { font-size: 13px; }
  .session-meta { display: flex; align-items: center; gap: 10px; }
  .session-name-btn {
    background: none; border: 1px solid transparent; color: var(--text-secondary);
    font-size: 12px; cursor: pointer; padding: 2px 6px; border-radius: 3px;
  }
  .session-name-btn:hover { border-color: var(--border-light); color: var(--text-primary); }
  .session-name-input {
    background: var(--bg-elevated); color: var(--text-primary);
    border: 1px solid var(--accent); font-size: 12px; padding: 2px 6px;
    border-radius: 3px; outline: none; width: 140px;
  }
  .session-status { font-size: 12px; color: var(--text-secondary); }
  .session-delete-btn {
    background: none; border: none; color: var(--text-muted);
    cursor: pointer; font-size: 16px; padding: 2px 6px; opacity: 0.5;
  }
  .session-delete-btn:hover { opacity: 1; color: var(--accent); }
  .session-delete-btn.confirm { color: var(--accent); opacity: 1; font-weight: 600; }

  .ignored-list { display: flex; flex-direction: column; gap: 4px; }
  .ignored-item {
    display: flex; justify-content: space-between; align-items: center;
    padding: 6px 12px; font-size: 13px; background: var(--bg-secondary);
    border-radius: 4px;
  }
</style>
