<script lang="ts">
  import { page } from '$app/stores';
  import { goto } from '$app/navigation';
  import { currentCampaign, sessions } from '$lib/stores/campaign';
  import { botStatus } from '$lib/stores/ws';
  import { scoped } from '$lib/api';
  import { addToast } from '$lib/stores/toast';
  import ChannelPicker from '$lib/components/ChannelPicker.svelte';
  import MergeModal from '$lib/components/MergeModal.svelte';
  import Modal from '$lib/components/ui/Modal.svelte';
  import Menu from '$lib/components/ui/Menu.svelte';
  import type { Session } from '$lib/types';

  let campaignId = $derived(Number($page.params.id));

  let query = $state('');
  let sort = $state<'newest' | 'oldest' | 'name'>('newest');
  let mergeOpen = $state(false);

  let renameTarget = $state<Session | null>(null);
  let renameOpen = $state(false);
  let renameValue = $state('');

  let deleteTarget = $state<Session | null>(null);
  let deleteOpen = $state(false);

  function sessionTitle(s: Session): string {
    return s.channelName || `Session #${s.id}`;
  }

  function fmtDate(iso: string): string {
    return new Date(iso).toLocaleDateString([], { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' });
  }
  function fmtTime(iso: string): string {
    return new Date(iso).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
  }
  function fmtDuration(s: Session): string {
    if (!s.endedAt) return '';
    const mins = Math.round((new Date(s.endedAt).getTime() - new Date(s.startedAt).getTime()) / 60000);
    if (mins < 60) return `${mins}m`;
    return `${Math.floor(mins / 60)}h ${String(mins % 60).padStart(2, '0')}m`;
  }

  const visible = $derived((() => {
    const q = query.trim().toLowerCase();
    let list = q
      ? $sessions.filter(s =>
          sessionTitle(s).toLowerCase().includes(q) ||
          fmtDate(s.startedAt).toLowerCase().includes(q) ||
          String(s.id).includes(q))
      : [...$sessions];
    if (sort === 'newest') list.sort((a, b) => b.startedAt.localeCompare(a.startedAt));
    else if (sort === 'oldest') list.sort((a, b) => a.startedAt.localeCompare(b.startedAt));
    else list.sort((a, b) => sessionTitle(a).localeCompare(sessionTitle(b), undefined, { numeric: true }));
    return list;
  })());

  function openRename(s: Session) {
    renameTarget = s;
    renameValue = s.channelName ?? '';
    renameOpen = true;
  }

  async function saveRename() {
    if (!renameTarget) return;
    const name = renameValue.trim();
    if (!name) return;
    const id = renameTarget.id;
    try {
      await scoped(campaignId).sessions.update(id, name);
      sessions.update(list => list.map(s => s.id === id ? { ...s, channelName: name } : s));
      renameOpen = false;
    } catch (e) {
      addToast(`Failed to rename session: ${e}`, 'error');
    }
  }

  function openDelete(s: Session) {
    deleteTarget = s;
    deleteOpen = true;
  }

  async function confirmDelete() {
    if (!deleteTarget) return;
    const id = deleteTarget.id;
    try {
      await scoped(campaignId).sessions.delete(id);
      sessions.update(list => list.filter(s => s.id !== id));
      addToast('Session deleted', 'success');
    } catch (e) {
      addToast(`Failed to delete: ${e}`, 'error');
    }
    deleteOpen = false;
  }
</script>

<svelte:head>
  <title>Sessions - {$currentCampaign?.name ?? 'Campaign'} - dndscribe</title>
</svelte:head>

<main class="wrap">
  <div class="heading">
    <div>
      <div class="eyebrow">{$currentCampaign?.name ?? ''}</div>
      <h1 class="page-title">Sessions</h1>
      {#if $currentCampaign?.description}<p class="muted">{$currentCampaign.description}</p>{/if}
    </div>
    <div class="actions">
      <Menu items={[{ label: 'Merge sessions…', onselect: () => { mergeOpen = true; } }]} label="Session tools" />
      <a href="/campaigns/{campaignId}/live" class="btn">Open live view</a>
      <ChannelPicker {campaignId} />
    </div>
  </div>

  <div class="toolbar">
    <input class="input search" type="search" placeholder="Search sessions by name, date or #" bind:value={query} aria-label="Search sessions" />
    <select class="select sort" bind:value={sort} aria-label="Sort sessions">
      <option value="newest">Newest first</option>
      <option value="oldest">Oldest first</option>
      <option value="name">By name</option>
    </select>
  </div>

  {#if $sessions.length === 0}
    <div class="empty-state">
      <h3>No sessions yet</h3>
      <p>Join a voice channel to start recording. Each recording becomes a session here.</p>
    </div>
  {:else if visible.length === 0}
    <div class="empty-state"><h3>No matches</h3><p>No session matches “{query}”.</p></div>
  {:else}
    <ul class="list">
      {#each visible as s (s.id)}
        {@const live = !s.endedAt}
        {@const recordingNow = live && $botStatus.recording && $botStatus.campaignId === campaignId}
        <li class="row">
          <a class="row-link" href={live ? `/campaigns/${campaignId}/live` : `/campaigns/${campaignId}/sessions/${s.id}`}>
            <span class="num">#{s.id}</span>
            <span class="main">
              <span class="title">{sessionTitle(s)}</span>
              <span class="sub">{[fmtDate(s.startedAt), fmtTime(s.startedAt), fmtDuration(s)].filter(Boolean).join(' · ')}</span>
            </span>
            {#if recordingNow}
              <span class="badge live"><span class="rec-dot on"></span>Recording</span>
            {:else if live}
              <span class="badge warning" title="Session has no end time; the bot may have stopped unexpectedly">Open</span>
            {:else}
              <span class="badge">Ended</span>
            {/if}
          </a>
          <Menu items={[
            { label: 'Open transcript', onselect: () => goto(`/campaigns/${campaignId}/sessions/${s.id}`) },
            { label: 'Rename', onselect: () => openRename(s) },
            { label: 'Delete session', danger: true, onselect: () => openDelete(s) },
          ]} />
        </li>
      {/each}
    </ul>
  {/if}
</main>

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
  <p>
    <strong>{deleteTarget ? sessionTitle(deleteTarget) : ''}</strong>
    {#if deleteTarget}({fmtDate(deleteTarget.startedAt)}){/if}
    and its transcript lines will be deleted. This cannot be undone.
  </p>
  {#snippet footer()}
    <button class="btn" onclick={() => { deleteOpen = false; }}>Cancel</button>
    <button class="btn-danger" onclick={confirmDelete}>Delete session</button>
  {/snippet}
</Modal>

<style>
  .wrap { max-width: 1240px; width: 100%; margin: 0 auto; padding: 36px 32px 64px; }
  .heading { display: flex; justify-content: space-between; align-items: flex-end; gap: 20px; margin-bottom: 24px; flex-wrap: wrap; }
  .heading .eyebrow { margin-bottom: 10px; }
  .heading p { margin-top: 6px; }
  .actions { display: flex; align-items: center; gap: 10px; }

  .toolbar { display: flex; gap: 10px; margin-bottom: 16px; }
  .search { max-width: 420px; }
  .sort { width: 170px; }

  .list { list-style: none; border: 1px solid var(--border-color); border-radius: var(--radius-lg); background: var(--surface); }
  .row { display: flex; align-items: center; gap: 8px; padding-right: 12px; border-bottom: 1px solid var(--border-color); }
  .row:last-child { border-bottom: 0; }
  .row:hover { background: var(--surface-hover); }
  .row:first-child { border-radius: var(--radius-lg) var(--radius-lg) 0 0; }
  .row:last-child { border-radius: 0 0 var(--radius-lg) var(--radius-lg); }
  .row-link {
    flex: 1; min-width: 0; display: flex; align-items: center; gap: 16px;
    min-height: 68px; padding: 12px 8px 12px 20px; color: var(--text);
  }
  .row-link:hover { text-decoration: none; }
  .num {
    flex-shrink: 0; width: 46px; height: 40px; border-radius: var(--radius);
    display: grid; place-items: center; background: var(--accent-soft); color: var(--accent-text);
    font-size: 13px; font-weight: 700;
  }
  .main { flex: 1; min-width: 0; display: flex; flex-direction: column; }
  .title { font-weight: 650; font-size: 15px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .sub { font-size: 13px; color: var(--text-3); }

  @media (max-width: 650px) {
    .wrap { padding: 24px 16px 48px; }
    .actions { width: 100%; flex-wrap: wrap; }
    .toolbar { flex-direction: column; }
    .search, .sort { max-width: none; width: 100%; }
    .row-link { gap: 12px; padding-left: 12px; }
    .num { display: none; }
  }
</style>
