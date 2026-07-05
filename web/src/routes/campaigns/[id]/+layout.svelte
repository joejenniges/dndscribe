<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { page } from '$app/stores';
  import { campaigns as campaignsApi, scoped } from '$lib/api';
  import { setCampaign } from '$lib/ws';
  import { currentCampaign, sessions, hotwords, presets, categories, ignoredUsers, initCampaignWsHandlers } from '$lib/stores/campaign';
  import { addToast } from '$lib/stores/toast';

  let { children } = $props();

  let unsub: (() => void) | undefined;
  let campaignId = $derived(Number($page.params.id));

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

  onMount(() => {
    unsub = initCampaignWsHandlers();
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

{@render children()}
