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

<div class="panel-group members">
  <h3>Voice members <span class="count">({$voiceMembers.length})</span></h3>
  {#if $voiceMembers.length > 0}
    <ul class="members-list">
      {#each $voiceMembers as member (member.id)}
        <li class="member-item">
          <span class="member-username">{member.username}</span>
          <div class="member-row">
            <input
              class="input member-char-input"
              data-user-id={member.id}
              placeholder="Character name"
              aria-label="Character name for {member.username}"
              bind:value={charInputs[member.id]}
              onblur={() => setName(member.id)}
              onkeydown={(e) => handleKeydown(e, member.id)}
            />
            <button class="icon-btn bordered" title="Open name palette" aria-label="Open name palette for {member.username}" onclick={() => openPalette(member.id, member.username)}>&#9776;</button>
            <button class="icon-btn bordered danger" title="Ignore user" aria-label="Ignore {member.username}" onclick={() => ignoreUser(member.id, member.username)}>&#8856;</button>
          </div>
        </li>
      {/each}
    </ul>
  {:else}
    <p class="subtle small">Not in a voice channel.</p>
  {/if}
</div>

<style>
  .members-list { list-style: none; display: flex; flex-direction: column; gap: 12px; }
  .member-item { display: flex; flex-direction: column; gap: 4px; }
  .member-username { color: var(--text-2); font-size: 12px; font-weight: 600; }
  .member-row { display: flex; align-items: center; gap: 6px; }
  .member-char-input { flex: 1; min-width: 0; }
</style>
