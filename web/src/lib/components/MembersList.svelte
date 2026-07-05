<script lang="ts">
  import { voiceMembers } from '$lib/stores/ws';
  import { scoped } from '$lib/api';
  import { addToast } from '$lib/stores/toast';

  let { campaignId }: { campaignId: number } = $props();

  const api = $derived(scoped(campaignId));

  let charInputs: Record<string, string> = $state({});

  // Sync inputs from members -- always update to reflect server state
  // unless the user is actively editing (input is focused).
  $effect(() => {
    for (const m of $voiceMembers) {
      const el = document.querySelector(`input[data-user-id="${m.id}"]`) as HTMLInputElement | null;
      const isFocused = el && document.activeElement === el;
      if (!isFocused) {
        charInputs[m.id] = m.characterName ?? '';
      }
    }
  });

  async function setName(userId: string) {
    const name = charInputs[userId]?.trim() || null;
    try {
      await api.setCharacterName(userId, name);
    } catch (e) {
      addToast(`Failed: ${e}`, 'error');
    }
  }

  async function ignoreUser(userId: string, username: string) {
    try {
      await api.ignoredUsers.add(userId, username);
      addToast(`Ignoring ${username}`, 'success');
    } catch (e) {
      addToast(`Failed: ${e}`, 'error');
    }
  }

  function openPalette(userId: string, username: string) {
    const url = `/campaigns/${campaignId}/palette/${userId}?username=${encodeURIComponent(username)}`;
    window.open(url, `palette-${userId}`, 'width=400,height=600,resizable=yes');
  }

  function handleKeydown(e: KeyboardEvent, userId: string) {
    if (e.key === 'Enter') setName(userId);
  }
</script>

<div class="members-panel">
  <div class="panel-header">Voice Members ({$voiceMembers.length})</div>
  {#if $voiceMembers.length > 0}
    <div class="members-list">
      {#each $voiceMembers as member}
        <div class="member-item">
          <span class="member-username">{member.username}</span>
          <div class="member-row">
            <input
              class="member-char-input"
              data-user-id={member.id}
              placeholder="Character name"
              bind:value={charInputs[member.id]}
              onblur={() => setName(member.id)}
              onkeydown={(e) => handleKeydown(e, member.id)}
            />
            <button class="palette-btn" title="Name palette" onclick={() => openPalette(member.id, member.username)}>
              &#9776;
            </button>
            <button class="ignore-btn" title="Ignore user" onclick={() => ignoreUser(member.id, member.username)}>
              &#128683;
            </button>
          </div>
        </div>
      {/each}
    </div>
  {:else}
    <div class="members-empty">Not in a voice channel</div>
  {/if}
</div>

<style>
  .members-panel { border-bottom: 1px solid var(--border); }
  .panel-header {
    padding: 8px 16px; font-size: 13px; font-weight: 600;
    color: var(--text-secondary); border-bottom: 1px solid var(--border);
  }
  .members-list { max-height: 250px; overflow-y: auto; }
  .member-item {
    padding: 6px 12px; font-size: 13px;
    display: flex; flex-direction: column; gap: 2px;
    border-bottom: 1px solid #1e1e1e;
  }
  .member-username { color: #aaa; font-size: 12px; }
  .member-row { display: flex; align-items: center; gap: 6px; }
  .member-char-input {
    flex: 1; background: var(--bg-primary); border: 1px solid var(--border);
    color: var(--text-primary); padding: 3px 8px; border-radius: 3px; font-size: 13px;
  }
  .member-char-input:focus { outline: 1px solid var(--accent); border-color: var(--accent); }
  .member-char-input::placeholder { color: var(--text-muted); }
  .ignore-btn, .palette-btn {
    background: none; border: none; color: var(--text-muted);
    cursor: pointer; font-size: 14px; padding: 2px 4px;
  }
  .ignore-btn:hover { color: var(--accent); }
  .palette-btn:hover { color: var(--text-primary); }
  .members-empty { padding: 12px; text-align: center; color: var(--text-muted); font-size: 12px; }
</style>
