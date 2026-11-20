<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import { campaigns as campaignsApi, scoped } from '$lib/api';
  import { botStatus } from '$lib/stores/ws';
  import { addToast } from '$lib/stores/toast';
  import Modal from '$lib/components/ui/Modal.svelte';
  import Menu from '$lib/components/ui/Menu.svelte';
  import type { Campaign, Session } from '$lib/types';

  interface CampaignStats { count: number; last: Session | null; }

  let campaignList: Campaign[] = $state([]);
  let stats: Record<number, CampaignStats> = $state({});
  let loaded = $state(false);

  // Create / edit dialog
  let dialogOpen = $state(false);
  let dialogMode = $state<'create' | 'edit'>('create');
  let editingId = $state<number | null>(null);
  let formName = $state('');
  let formDesc = $state('');
  let saving = $state(false);

  // Delete confirmation dialog
  let deleteTarget = $state<Campaign | null>(null);
  let deleteOpen = $state(false);

  onMount(async () => {
    try {
      campaignList = await campaignsApi.list();
      loaded = true;
      loadStats(campaignList);
    } catch (e) {
      addToast(`Failed to load campaigns: ${e}`, 'error');
    }
  });

  // Session count / last played are derived from each campaign's session list.
  // Fine at a handful of campaigns; would want a summary endpoint at dozens.
  async function loadStats(list: Campaign[]) {
    await Promise.all(list.map(async (c) => {
      try {
        const s = await scoped(c.id).sessions.list();
        const sorted = [...s].sort((a, b) => b.startedAt.localeCompare(a.startedAt));
        stats[c.id] = { count: s.length, last: sorted[0] ?? null };
      } catch {
        // Stats are decorative; the card still works without them.
      }
    }));
  }

  function openCreate() {
    dialogMode = 'create';
    editingId = null;
    formName = '';
    formDesc = '';
    dialogOpen = true;
  }

  function openEdit(c: Campaign) {
    dialogMode = 'edit';
    editingId = c.id;
    formName = c.name;
    formDesc = c.description ?? '';
    dialogOpen = true;
  }

  async function save() {
    const name = formName.trim();
    if (!name) return;
    saving = true;
    try {
      if (dialogMode === 'create') {
        const c = await campaignsApi.create(name, formDesc.trim() || undefined);
        campaignList = [...campaignList, c];
        stats[c.id] = { count: 0, last: null };
        addToast(`Campaign "${c.name}" created`, 'success');
      } else if (editingId != null) {
        const updated = await campaignsApi.update(editingId, name, formDesc.trim() || undefined);
        if (updated) campaignList = campaignList.map(c => c.id === editingId ? updated : c);
      }
      dialogOpen = false;
    } catch (e) {
      addToast(`Failed: ${e}`, 'error');
    } finally {
      saving = false;
    }
  }

  function askDelete(c: Campaign) {
    deleteTarget = c;
    deleteOpen = true;
  }

  async function confirmDelete() {
    if (!deleteTarget) return;
    const id = deleteTarget.id;
    try {
      await campaignsApi.delete(id);
      campaignList = campaignList.filter(c => c.id !== id);
      addToast('Campaign deleted', 'success');
    } catch (e) {
      addToast(`Failed: ${e}`, 'error');
    }
    deleteOpen = false;
    deleteTarget = null;
  }

  function fmtDate(iso: string): string {
    return new Date(iso).toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' });
  }

  // Decorative header tint, stable per campaign id.
  const tints = [
    'radial-gradient(circle at 78% 15%, #854062 0%, transparent 45%), linear-gradient(115deg, #342334, #171c29)',
    'radial-gradient(circle at 26% 40%, #407d83 0%, transparent 46%), linear-gradient(110deg, #203940, #1b2833)',
    'radial-gradient(circle at 60% 70%, #6b5a2e 0%, transparent 45%), linear-gradient(120deg, #2e2a1e, #191b20)',
    'radial-gradient(circle at 30% 20%, #3d4f85 0%, transparent 45%), linear-gradient(120deg, #1f2540, #17191f)',
  ];
</script>

<svelte:head>
  <title>Campaigns - dndscribe</title>
</svelte:head>

<main class="wrap">
  <div class="heading">
    <div>
      <div class="eyebrow">Your table</div>
      <h1 class="page-title">Campaigns</h1>
      <p class="muted">Pick a campaign to browse sessions, open the live transcript, or manage its settings.</p>
    </div>
    <button class="btn-primary" onclick={openCreate}>+ New campaign</button>
  </div>

  {#if loaded && campaignList.length === 0}
    <div class="empty-state">
      <h3>No campaigns yet</h3>
      <p>Create a campaign, then join a voice channel from it to start recording.</p>
      <button class="btn-primary" style="margin-top:16px" onclick={openCreate}>Create a campaign</button>
    </div>
  {:else}
    <div class="cards">
      {#each campaignList as c, i (c.id)}
        {@const st = stats[c.id]}
        {@const isLive = $botStatus.recording && $botStatus.campaignId === c.id}
        <article class="card campaign">
          <a href="/campaigns/{c.id}" class="art" style:background={tints[i % tints.length]} aria-label="Open {c.name}">
            <span class="initial">{c.name.slice(0, 1).toUpperCase()}</span>
            {#if isLive}<span class="badge live live-badge"><span class="rec-dot on"></span>Recording</span>{/if}
          </a>
          <div class="body">
            <div class="title-row">
              <h2><a href="/campaigns/{c.id}">{c.name}</a></h2>
              <Menu items={[
                { label: 'Edit details', onselect: () => openEdit(c) },
                { label: 'Settings', onselect: () => goto(`/campaigns/${c.id}/settings`) },
                // Campaign 1 is the default the backend falls back to; it refuses deletion.
                { label: 'Delete campaign', danger: true, disabled: c.id === 1, onselect: () => askDelete(c) },
              ]} />
            </div>
            {#if c.description}<p class="desc muted">{c.description}</p>{/if}
            <div class="meta">
              <span>{st ? `${st.count} session${st.count === 1 ? '' : 's'}` : '…'}</span>
              {#if st?.last}<span>Last played {fmtDate(st.last.startedAt)}</span>{/if}
            </div>
            <div class="foot">
              <a href="/campaigns/{c.id}" class="btn btn-sm">Sessions</a>
              <a href="/campaigns/{c.id}/live" class="btn btn-sm">Live view</a>
            </div>
          </div>
        </article>
      {/each}
      <button class="card new-card" onclick={openCreate}>
        <span class="plus">+</span>
        <span>New campaign</span>
      </button>
    </div>
  {/if}
</main>

<Modal bind:open={dialogOpen} title={dialogMode === 'create' ? 'New campaign' : 'Edit campaign'}>
  <label class="field">
    <span>Name</span>
    <!-- svelte-ignore a11y_autofocus -->
    <input class="input" bind:value={formName} autofocus onkeydown={(e) => { if (e.key === 'Enter') save(); }} />
  </label>
  <label class="field">
    <span>Description <span class="subtle">(optional)</span></span>
    <textarea class="input" bind:value={formDesc} rows="3"></textarea>
  </label>
  {#snippet footer()}
    <button class="btn" onclick={() => { dialogOpen = false; }}>Cancel</button>
    <button class="btn-primary" onclick={save} disabled={saving || !formName.trim()}>
      {saving ? 'Saving…' : dialogMode === 'create' ? 'Create campaign' : 'Save'}
    </button>
  {/snippet}
</Modal>

<Modal bind:open={deleteOpen} title="Delete campaign?">
  <p>
    <strong>{deleteTarget?.name}</strong> and everything in it will be removed: sessions, transcripts,
    hotwords and ignored users. This cannot be undone.
  </p>
  {#snippet footer()}
    <button class="btn" onclick={() => { deleteOpen = false; }}>Cancel</button>
    <button class="btn-danger" onclick={confirmDelete}>Delete campaign</button>
  {/snippet}
</Modal>

<style>
  .wrap { max-width: 1240px; width: 100%; margin: 0 auto; padding: 40px 32px 64px; }
  .heading { display: flex; justify-content: space-between; align-items: flex-end; gap: 20px; margin-bottom: 28px; }
  .heading .eyebrow { margin-bottom: 10px; }
  .heading p { margin-top: 6px; }

  .cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 18px; }
  .campaign { overflow: visible; display: flex; flex-direction: column; }
  .art {
    position: relative; height: 112px; border-radius: var(--radius-lg) var(--radius-lg) 0 0;
    overflow: hidden; display: block;
  }
  .initial {
    position: absolute; right: 22px; bottom: 6px; font-size: 64px; font-weight: 800;
    color: rgba(255, 255, 255, 0.12); line-height: 1;
  }
  .live-badge { position: absolute; left: 14px; top: 14px; }
  .body { padding: 18px 20px 20px; display: flex; flex-direction: column; gap: 8px; flex: 1; }
  .title-row { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
  h2 { font-size: 18px; letter-spacing: -0.2px; }
  h2 a { color: var(--text); }
  .desc { font-size: 14px; }
  .meta { display: flex; flex-wrap: wrap; gap: 4px 16px; font-size: 13px; color: var(--text-3); }
  .foot { display: flex; gap: 8px; margin-top: auto; padding-top: 14px; border-top: 1px solid var(--border-color); }

  .new-card {
    min-height: 260px; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 8px;
    background: transparent; border-style: dashed; color: var(--text-2); font-weight: 600;
  }
  .new-card:hover { border-color: var(--accent); color: var(--text); }
  .plus { font-size: 30px; line-height: 1; }

  @media (max-width: 650px) {
    .wrap { padding: 24px 16px 48px; }
    .heading { flex-direction: column; align-items: stretch; }
    .cards { grid-template-columns: 1fr; }
    .new-card { min-height: 120px; }
  }
</style>
