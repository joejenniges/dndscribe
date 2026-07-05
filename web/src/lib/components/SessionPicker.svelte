<script lang="ts">
  import { sessions } from '$lib/stores/campaign';
  import type { Session } from '$lib/types';

  let {
    value = $bindable<number | undefined>(undefined),
    onchange,
    ondelete,
    onmerge,
    onrename,
  }: {
    value?: number;
    onchange?: (sessionId: number) => void;
    ondelete?: (sessionId: number) => void;
    onmerge?: () => void;
    onrename?: (sessionId: number, currentName: string) => void;
  } = $props();

  let confirmDeleteId = $state<number | null>(null);
  let renamingId = $state<number | null>(null);
  let renameValue = $state('');

  function formatSession(s: Session): string {
    const d = new Date(s.startedAt);
    const date = d.toLocaleDateString([], { month: 'short', day: 'numeric' });
    const time = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const active = !s.endedAt ? ' (active)' : '';
    const name = s.channelName ? ` - ${s.channelName}` : '';
    return `#${s.id} ${date} ${time}${name}${active}`;
  }

  function handleChange(e: Event) {
    const val = parseInt((e.target as HTMLSelectElement).value, 10);
    if (!isNaN(val)) {
      value = val;
      onchange?.(val);
    }
  }

  function handleDelete() {
    if (value == null) return;
    if (confirmDeleteId !== value) {
      confirmDeleteId = value;
      setTimeout(() => { confirmDeleteId = null; }, 3000);
      return;
    }
    ondelete?.(value);
    confirmDeleteId = null;
  }

  function startRename() {
    if (value == null) return;
    const session = $sessions.find(s => s.id === value);
    if (!session) return;
    renameValue = session.channelName ?? '';
    renamingId = value;
  }

  function submitRename() {
    if (renamingId == null) return;
    const name = renameValue.trim();
    if (name) {
      onrename?.(renamingId, name);
    }
    renamingId = null;
  }

  function handleRenameKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter') {
      e.preventDefault();
      submitRename();
    } else if (e.key === 'Escape') {
      renamingId = null;
    }
  }
</script>

<div class="session-picker">
  {#if renamingId != null}
    <input
      class="rename-input"
      type="text"
      bind:value={renameValue}
      onkeydown={handleRenameKeydown}
      onblur={submitRename}
      autofocus
    />
  {:else}
    <select onchange={handleChange} value={value}>
      {#each $sessions as s}
        <option value={s.id}>{formatSession(s)}</option>
      {/each}
    </select>
  {/if}
  {#if onrename}
    <button class="session-action-btn" onclick={startRename} title="Rename session">
      &#9998;
    </button>
  {/if}
  <button
    class="session-action-btn"
    class:confirm={confirmDeleteId === value}
    onclick={handleDelete}
    title={confirmDeleteId === value ? 'Click again to confirm delete' : 'Delete session'}
  >
    &#128465;
  </button>
  {#if onmerge}
    <button class="session-action-btn" onclick={onmerge} title="Merge sessions">
      &#8644;
    </button>
  {/if}
</div>

<style>
  .session-picker { display: flex; align-items: center; gap: 4px; }
  select {
    background: var(--bg-elevated); color: var(--text-primary);
    border: 1px solid var(--border-light); padding: 5px 8px;
    border-radius: 4px; font-size: 12px; cursor: pointer; max-width: 200px;
  }
  select:focus { outline: 1px solid var(--accent); }
  .rename-input {
    background: var(--bg-elevated); color: var(--text-primary);
    border: 1px solid var(--accent); padding: 5px 8px;
    border-radius: 4px; font-size: 12px; outline: none; max-width: 200px;
  }
  .session-action-btn {
    background: none; border: none; color: var(--text-secondary);
    cursor: pointer; font-size: 14px; padding: 2px 4px; opacity: 0.7;
  }
  .session-action-btn:hover { opacity: 1; }
  .session-action-btn.confirm { color: var(--accent); opacity: 1; }
</style>
