<script lang="ts">
  import { sessions } from '$lib/stores/campaign';
  import { scoped } from '$lib/api';
  import { addToast } from '$lib/stores/toast';
  import type { Session, TranscriptionLine } from '$lib/types';

  let {
    campaignId,
    open = $bindable(false),
  }: {
    campaignId: number;
    open?: boolean;
  } = $props();

  let selectedIds: number[] = $state([]);
  let mergeName = $state('');
  let previews = $state<Record<number, { first: TranscriptionLine[]; last: TranscriptionLine[] }>>({});
  let merging = $state(false);

  const api = $derived(scoped(campaignId));

  function reset() {
    selectedIds = [];
    mergeName = '';
    previews = {};
    merging = false;
  }

  function addSlot() {
    selectedIds = [...selectedIds, 0];
  }

  function removeSlot(idx: number) {
    selectedIds = selectedIds.filter((_, i) => i !== idx);
  }

  async function onSessionSelect(idx: number, value: number) {
    selectedIds[idx] = value;
    if (value && !previews[value]) {
      try {
        previews[value] = await api.sessions.preview(value);
      } catch { /* ignore */ }
    }
  }

  async function doMerge() {
    const ids = selectedIds.filter(id => id > 0);
    const name = mergeName.trim();
    if (ids.length < 2 || !name) return;

    merging = true;
    try {
      await api.sessions.merge(ids, name);
      addToast('Sessions merged', 'success');
      open = false;
      reset();
    } catch (e) {
      addToast(`Merge failed: ${e}`, 'error');
    } finally {
      merging = false;
    }
  }

  function formatSession(s: Session): string {
    const d = new Date(s.startedAt);
    return `#${s.id} ${d.toLocaleDateString([], { month: 'short', day: 'numeric' })} ${d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}${s.channelName ? ' - ' + s.channelName : ''}`;
  }

  function formatTime(ts: string): string {
    return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  }

  $effect(() => {
    if (open && selectedIds.length === 0) {
      addSlot();
      addSlot();
    }
  });

  const canMerge = $derived(selectedIds.filter(id => id > 0).length >= 2 && mergeName.trim().length > 0);
</script>

{#if open}
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div class="modal-overlay" onclick={() => { open = false; reset(); }}>
    <div class="modal" onclick={(e) => e.stopPropagation()}>
      <div class="modal-header">
        <span>Merge Sessions</span>
        <button class="modal-close" onclick={() => { open = false; reset(); }}>&times;</button>
      </div>
      <div class="modal-body">
        <div class="modal-left">
          {#each selectedIds as sid, idx}
            <div class="merge-session-item">
              <span class="drag-handle">\u2630</span>
              <select
                value={sid}
                onchange={(e) => onSessionSelect(idx, parseInt((e.target as HTMLSelectElement).value, 10))}
              >
                <option value={0}>Select session...</option>
                {#each $sessions as s}
                  <option value={s.id}>{formatSession(s)}</option>
                {/each}
              </select>
              <button class="remove-merge-btn" onclick={() => removeSlot(idx)}>&times;</button>
            </div>
          {/each}
          <button class="btn" onclick={addSlot}>+ Add session</button>
        </div>
        <div class="modal-right">
          {#each selectedIds.filter(id => id > 0) as sid}
            {#if previews[sid]}
              <div class="preview-section">
                <div class="preview-separator">Session #{sid}</div>
                {#each previews[sid].first as line}
                  <div class="preview-line">
                    <span class="pv-time">{formatTime(line.timestamp)}</span>
                    <span class="pv-speaker">{line.nickname || line.discordUsername}:</span>
                    {line.text}
                  </div>
                {/each}
                <div class="preview-dots">...</div>
                {#each previews[sid].last as line}
                  <div class="preview-line">
                    <span class="pv-time">{formatTime(line.timestamp)}</span>
                    <span class="pv-speaker">{line.nickname || line.discordUsername}:</span>
                    {line.text}
                  </div>
                {/each}
              </div>
            {/if}
          {/each}
        </div>
      </div>
      <div class="modal-footer">
        <input type="text" bind:value={mergeName} placeholder="Merged session name..." />
        <button class="btn-primary" onclick={doMerge} disabled={!canMerge || merging}>
          {merging ? 'Merging...' : 'Merge'}
        </button>
      </div>
    </div>
  </div>
{/if}

<style>
  .modal-overlay {
    position: fixed; inset: 0; background: rgba(0,0,0,0.7); z-index: 200;
    display: flex; align-items: center; justify-content: center;
  }
  .modal {
    background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 8px;
    width: 800px; max-width: 95vw; max-height: 85vh; display: flex; flex-direction: column;
  }
  .modal-header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 12px 16px; border-bottom: 1px solid var(--border); font-weight: 600;
  }
  .modal-close { background: none; border: none; color: var(--text-secondary); cursor: pointer; font-size: 18px; }
  .modal-close:hover { color: var(--text-primary); }
  .modal-body { display: flex; flex: 1; overflow: hidden; min-height: 300px; }
  .modal-left {
    width: 280px; border-right: 1px solid var(--border);
    display: flex; flex-direction: column; padding: 12px; gap: 8px;
  }
  .modal-right { flex: 1; overflow-y: auto; padding: 12px; }
  .modal-footer {
    display: flex; justify-content: space-between; align-items: center;
    padding: 12px 16px; border-top: 1px solid var(--border); gap: 8px;
  }
  .modal-footer input {
    flex: 1; background: var(--bg-primary); border: 1px solid var(--border);
    color: var(--text-primary); padding: 6px 10px; border-radius: 4px; font-size: 13px;
  }
  .modal-footer input:focus { outline: 1px solid var(--accent); border-color: var(--accent); }
  .merge-session-item { display: flex; align-items: center; gap: 6px; }
  .drag-handle { cursor: grab; color: var(--text-muted); font-size: 16px; user-select: none; }
  .merge-session-item select {
    flex: 1; background: var(--bg-elevated); color: var(--text-primary);
    border: 1px solid var(--border-light); padding: 4px 6px; border-radius: 4px; font-size: 12px;
  }
  .remove-merge-btn { background: none; border: none; color: var(--text-muted); cursor: pointer; font-size: 14px; }
  .remove-merge-btn:hover { color: var(--accent); }
  .preview-separator {
    text-align: center; color: var(--accent); font-size: 11px;
    padding: 6px 0; border-top: 1px dashed var(--border-light); margin-top: 4px;
  }
  .preview-line { font-size: 12px; padding: 2px 0; color: #aaa; }
  .preview-line .pv-time { color: var(--text-muted); margin-right: 6px; }
  .preview-line .pv-speaker { color: var(--accent); font-weight: 600; }
  .preview-dots { color: var(--text-muted); text-align: center; padding: 4px 0; font-size: 12px; }
</style>
