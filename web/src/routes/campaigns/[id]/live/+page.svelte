<script lang="ts">
  import { page } from '$app/stores';
  import { goto } from '$app/navigation';
  import { scoped, voice, saveRecordings as saveRecApi } from '$lib/api';
  import { currentCampaign, sessions, lines, partials, hotwords, ignoredUsers } from '$lib/stores/campaign';
  import { botStatus, saveRecordingsEnabled } from '$lib/stores/ws';
  import { addToast } from '$lib/stores/toast';
  import TranscriptView from '$lib/components/TranscriptView.svelte';
  import ChannelPicker from '$lib/components/ChannelPicker.svelte';
  import MembersList from '$lib/components/MembersList.svelte';
  import MergeModal from '$lib/components/MergeModal.svelte';
  import Modal from '$lib/components/ui/Modal.svelte';
  import Menu from '$lib/components/ui/Menu.svelte';
  import Toggle from '$lib/components/ui/Toggle.svelte';
  import type { Session } from '$lib/types';

  let campaignId = $derived(Number($page.params.id));
  let api = $derived(scoped(campaignId));

  let selectedSession = $state<number | undefined>(undefined);
  let loadingLines = $state(false);
  let mergeOpen = $state(false);
  // From the server, so it is right even if paused from another tab or a slash command.
  const paused = $derived($botStatus.paused);

  let renameOpen = $state(false);
  let renameValue = $state('');
  let deleteOpen = $state(false);

  const recordingHere = $derived($botStatus.recording && $botStatus.campaignId === campaignId);
  const partialList = $derived([...$partials.values()]);
  const current = $derived($sessions.find(s => s.id === selectedSession));

  function sessionLabel(s: Session): string {
    const d = new Date(s.startedAt);
    const date = d.toLocaleDateString([], { month: 'short', day: 'numeric' });
    const time = d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
    return `${s.channelName || `Session #${s.id}`} — ${date}, ${time}${!s.endedAt ? ' (active)' : ''}`;
  }

  async function loadSession(sessionId: number) {
    selectedSession = sessionId;
    loadingLines = true;
    try {
      lines.set(await api.lines.list(sessionId));
    } catch (e) {
      addToast(`Failed to load session: ${e}`, 'error');
    } finally {
      loadingLines = false;
    }
  }

  // Auto-select the newest session on load (and when a new one starts).
  $effect(() => {
    if ($sessions.length > 0 && selectedSession == null) loadSession($sessions[0].id);
  });

  async function editLine(id: number, text: string) {
    try {
      const result = await api.lines.update(id, text);
      if (result.autoAdded.length > 0) addToast(`Auto-added hotwords: ${result.autoAdded.join(', ')}`, 'success');
    } catch (e) {
      addToast(`Failed: ${e}`, 'error');
    }
  }

  async function deleteLine(id: number) {
    try { await api.lines.delete(id); } catch (e) { addToast(`Failed: ${e}`, 'error'); }
  }

  async function togglePause() {
    try {
      const result = await voice.pause();
      addToast(result.paused ? 'Transcription paused' : 'Transcription resumed', 'info');
    } catch (e) {
      addToast(`Failed: ${e}`, 'error');
    }
  }

  async function setSaveRecordings(value: boolean) {
    try {
      const result = await saveRecApi.set(value);
      saveRecordingsEnabled.set(result.enabled);
    } catch (e) {
      saveRecordingsEnabled.set(!value);
      addToast(`Failed: ${e}`, 'error');
    }
  }

  function openRename() {
    if (!current) return;
    renameValue = current.channelName ?? '';
    renameOpen = true;
  }

  async function saveRename() {
    const name = renameValue.trim();
    if (!current || !name) return;
    const id = current.id;
    try {
      await api.sessions.update(id, name);
      sessions.update(list => list.map(s => s.id === id ? { ...s, channelName: name } : s));
      renameOpen = false;
    } catch (e) {
      addToast(`Failed to rename: ${e}`, 'error');
    }
  }

  async function confirmDelete() {
    if (!current) return;
    try {
      await api.sessions.delete(current.id);
      addToast('Session deleted', 'success');
      selectedSession = undefined;
    } catch (e) {
      addToast(`Failed: ${e}`, 'error');
    }
    deleteOpen = false;
  }

  async function unignoreUser(userId: string) {
    try {
      await api.ignoredUsers.remove(userId);
      ignoredUsers.update(u => u.filter(i => i.discordUserId !== userId));
      addToast('User unignored', 'success');
    } catch (e) {
      addToast(`Failed: ${e}`, 'error');
    }
  }
</script>

<svelte:head>
  <title>Live - {$currentCampaign?.name ?? 'Campaign'} - dndscribe</title>
</svelte:head>

<TranscriptView
  {campaignId}
  lines={$lines}
  partials={partialList}
  live
  loading={loadingLines && $lines.length === 0}
  emptyMessage={recordingHere ? 'Listening… lines will appear as people speak.' : 'No transcription lines in this session.'}
  onEditLine={editLine}
  onDeleteLine={deleteLine}
  onNicknamesSaved={async () => { if (selectedSession != null) lines.set(await api.lines.list(selectedSession)); }}
>
  {#snippet header()}
    <div class="live-head">
      <div class="title-block">
        <div class="status-line">
          {#if recordingHere}
            <span class="badge live"><span class="rec-dot on"></span>{paused ? 'Paused' : 'Recording'} · {$botStatus.channelName}</span>
          {:else if $botStatus.recording}
            <span class="badge warning">Recording in another campaign</span>
          {:else}
            <span class="badge">Not recording</span>
          {/if}
        </div>
        <select
          class="select session-select"
          value={selectedSession}
          onchange={(e) => loadSession(Number((e.target as HTMLSelectElement).value))}
          aria-label="Session"
        >
          {#each $sessions as s (s.id)}<option value={s.id}>{sessionLabel(s)}</option>{/each}
        </select>
      </div>
      <div class="head-actions">
        {#if recordingHere}
          <button class="btn" onclick={togglePause}>{paused ? 'Resume' : 'Pause'}</button>
        {/if}
        <ChannelPicker {campaignId} />
        <Menu items={[
          { label: 'Rename session', disabled: !current, onselect: openRename },
          { label: 'Open as session page', disabled: !current, onselect: () => { if (current) goto(`/campaigns/${campaignId}/sessions/${current.id}`); } },
          { label: 'Merge sessions…', onselect: () => { mergeOpen = true; } },
          { label: 'Delete session', danger: true, disabled: !current, onselect: () => { deleteOpen = true; } },
        ]} label="Session actions" />
      </div>
    </div>
  {/snippet}

  {#snippet panelTop()}
    <div class="panel-group">
      <h3>Recording</h3>
      <Toggle
        label="Save audio"
        hint="Keep clips so lines can be replayed"
        checked={$saveRecordingsEnabled}
        onchange={setSaveRecordings}
      />
    </div>
    <MembersList {campaignId} />
  {/snippet}

  {#snippet panelBottom()}
    <div class="panel-group">
      <h3>Ignored users <span class="count">({$ignoredUsers.length})</span></h3>
      {#if $ignoredUsers.length > 0}
        <div class="chips">
          {#each $ignoredUsers as u}
            <button class="chip" onclick={() => unignoreUser(u.discordUserId)} title="Stop ignoring {u.discordUsername}">
              {u.discordUsername} <span aria-hidden="true">&times;</span>
            </button>
          {/each}
        </div>
      {:else}
        <p class="subtle small">Nobody is ignored.</p>
      {/if}
    </div>
    <div class="panel-group">
      <h3>Hotwords <span class="count">({$hotwords.length})</span></h3>
      <div class="chips">
        {#each $hotwords.slice(0, 8) as w}<span class="chip">{w}</span>{/each}
      </div>
      <a class="manage-link small" href="/campaigns/{campaignId}/settings#hotwords">Manage hotwords &rarr;</a>
    </div>
  {/snippet}
</TranscriptView>

<MergeModal {campaignId} bind:open={mergeOpen} />

<Modal bind:open={renameOpen} title="Rename session">
  <label class="field">
    <span>Name</span>
    <!-- svelte-ignore a11y_autofocus -->
    <input class="input" bind:value={renameValue} autofocus onkeydown={(e) => { if (e.key === 'Enter') saveRename(); }} />
  </label>
  {#snippet footer()}
    <button class="btn" onclick={() => { renameOpen = false; }}>Cancel</button>
    <button class="btn-primary" onclick={saveRename} disabled={!renameValue.trim()}>Save</button>
  {/snippet}
</Modal>

<Modal bind:open={deleteOpen} title="Delete session?">
  <p>{current ? sessionLabel(current) : ''} and all its transcript lines will be deleted. This cannot be undone.</p>
  {#snippet footer()}
    <button class="btn" onclick={() => { deleteOpen = false; }}>Cancel</button>
    <button class="btn-danger" onclick={confirmDelete}>Delete session</button>
  {/snippet}
</Modal>

<style>
  .live-head { display: flex; align-items: flex-end; justify-content: space-between; gap: 16px; flex-wrap: wrap; }
  .title-block { display: flex; flex-direction: column; gap: 8px; min-width: 0; flex: 1; }
  .status-line { display: flex; gap: 8px; }
  .session-select { max-width: 460px; font-weight: 600; }
  .head-actions { display: flex; align-items: center; gap: 10px; }
  .chips { display: flex; flex-wrap: wrap; gap: 6px; }
  .manage-link { display: inline-block; margin-top: 10px; font-weight: 600; }
  @media (max-width: 650px) {
    .head-actions { width: 100%; }
    .session-select { max-width: none; }
  }
</style>
