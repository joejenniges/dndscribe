<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { page } from '$app/stores';
  import { goto } from '$app/navigation';
  import { campaigns as campaignsApi, scoped } from '$lib/api';
  import { setCampaign } from '$lib/ws';
  import { currentCampaign, sessions, hotwords, presets, categories, ignoredUsers, initCampaignWsHandlers } from '$lib/stores/campaign';
  import { botStatus } from '$lib/stores/ws';
  import { addToast } from '$lib/stores/toast';
  import type { Campaign } from '$lib/types';

  let { children } = $props();

  let unsub: (() => void) | undefined;
  let campaignId = $derived(Number($page.params.id));
  let allCampaigns: Campaign[] = $state([]);

  const bare = $derived(($page.route.id ?? '').includes('/palette/'));
  const base = $derived(`/campaigns/${campaignId}`);
  const path = $derived($page.url.pathname);

  const tabs = $derived([
    { href: base, label: 'Sessions', active: path === base || path.startsWith(`${base}/sessions`) },
    { href: `${base}/live`, label: 'Live', active: path.startsWith(`${base}/live`) },
    { href: `${base}/characters`, label: 'Characters', active: path.startsWith(`${base}/characters`) },
    { href: `${base}/notes`, label: 'Notes', active: path.startsWith(`${base}/notes`) },
    { href: `${base}/settings`, label: 'Settings', active: path.startsWith(`${base}/settings`) },
  ]);

  const recordingHere = $derived($botStatus.recording && $botStatus.campaignId === campaignId);

  async function loadCampaignData(cid: number) {
    try {
      const [campaign, sess, hw, pr, cats, ignored] = await Promise.all([
        campaignsApi.get(cid),
        scoped(cid).sessions.list(),
        scoped(cid).hotwords.list(),
        scoped(cid).presets.list(),
        scoped(cid).categories.list(),
        scoped(cid).ignoredUsers.list(),
      ]);
      currentCampaign.set(campaign);
      sessions.set(sess);
      hotwords.set(hw);
      presets.set(pr);
      categories.set(cats);
      ignoredUsers.set(ignored);
    } catch (e) {
      addToast(`Failed to load campaign: ${e}`, 'error');
    }
  }

  function switchCampaign(e: Event) {
    const id = Number((e.target as HTMLSelectElement).value);
    if (!id || id === campaignId) return;
    // Keep the same tab when switching, except deep session links which are
    // campaign-specific.
    const sub = path.slice(base.length);
    const keep = sub.startsWith('/sessions') ? '' : sub;
    goto(`/campaigns/${id}${keep}`);
  }

  onMount(async () => {
    unsub = initCampaignWsHandlers();
    try { allCampaigns = await campaignsApi.list(); } catch { /* switcher just shows current */ }
  });

  // React to campaign ID changes
  $effect(() => {
    if (campaignId) {
      setCampaign(campaignId);
      loadCampaignData(campaignId);
    }
  });

  onDestroy(() => {
    unsub?.();
    currentCampaign.set(null);
  });
</script>

{#if bare}
  {@render children()}
{:else}
  <div class="subnav">
    <div class="switcher">
      {#if allCampaigns.length > 1}
        <select class="campaign-select" value={campaignId} onchange={switchCampaign} aria-label="Switch campaign">
          {#each allCampaigns as c}<option value={c.id}>{c.name}</option>{/each}
        </select>
      {:else}
        <span class="campaign-name">{$currentCampaign?.name ?? ''}</span>
      {/if}
    </div>
    <nav aria-label="Campaign">
      {#each tabs as t}
        <a href={t.href} class:active={t.active} aria-current={t.active ? 'page' : undefined}>
          {#if t.label === 'Live'}<span class="rec-dot" class:on={recordingHere}></span>{/if}
          {t.label}
        </a>
      {/each}
    </nav>
  </div>
  <div class="campaign-body">
    {@render children()}
  </div>
{/if}

<style>
  .subnav {
    position: sticky; top: var(--header-h); z-index: 30;
    height: var(--subnav-h); display: flex; align-items: stretch; gap: 20px;
    padding: 0 24px; background: #17191c; border-bottom: 1px solid var(--border-color);
  }
  .switcher { display: flex; align-items: center; flex-shrink: 0; }
  .campaign-select {
    appearance: none; background: transparent; border: 0; color: var(--text);
    font-weight: 700; font-size: 15px; padding: 6px 26px 6px 8px; border-radius: var(--radius-sm);
    min-height: 36px; cursor: pointer;
    background-image: linear-gradient(45deg, transparent 50%, var(--text-2) 50%), linear-gradient(135deg, var(--text-2) 50%, transparent 50%);
    background-position: calc(100% - 13px) 50%, calc(100% - 8px) 50%;
    background-size: 5px 5px; background-repeat: no-repeat;
  }
  .campaign-select:hover { background-color: var(--surface-raised); }
  .campaign-select option { background: var(--surface); }
  .campaign-name { font-weight: 700; padding: 0 8px; }

  nav { display: flex; align-items: stretch; gap: 4px; overflow-x: auto; scrollbar-width: none; }
  nav::-webkit-scrollbar { display: none; }
  nav a {
    display: flex; align-items: center; gap: 8px; padding: 0 14px; white-space: nowrap;
    color: var(--text-2); font-weight: 600; font-size: 14px;
    border-bottom: 2px solid transparent; margin-bottom: -1px;
  }
  nav a:hover { color: var(--text); text-decoration: none; }
  nav a.active { color: var(--text); border-bottom-color: var(--accent); }
  nav .rec-dot:not(.on) { display: none; }

  .campaign-body { flex: 1; display: flex; flex-direction: column; min-height: 0; }

  @media (max-width: 760px) {
    .subnav { padding: 0 8px; gap: 4px; }
    .campaign-select { max-width: 130px; }
    nav a { padding: 0 10px; }
  }
</style>
