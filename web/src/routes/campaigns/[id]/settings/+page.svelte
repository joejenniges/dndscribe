<script lang="ts">
  import { page } from '$app/stores';
  import { campaigns as campaignsApi, scoped } from '$lib/api';
  import { currentCampaign, hotwords, ignoredUsers } from '$lib/stores/campaign';
  import { addToast } from '$lib/stores/toast';
  import HotwordManager from '$lib/components/HotwordManager.svelte';

  let campaignId = $derived(Number($page.params.id));

  let name = $state('');
  let description = $state('');
  let saving = $state(false);
  let loadedFor = 0;

  // Seed the form once per campaign; don't clobber typing on store updates.
  $effect(() => {
    const c = $currentCampaign;
    if (c && c.id !== loadedFor) {
      loadedFor = c.id;
      name = c.name;
      description = c.description ?? '';
    }
  });

  const dirty = $derived(
    !!$currentCampaign && (name.trim() !== $currentCampaign.name || (description.trim() || null) !== ($currentCampaign.description || null))
  );

  async function saveDetails() {
    if (!name.trim()) return;
    saving = true;
    try {
      const updated = await campaignsApi.update(campaignId, name.trim(), description.trim() || undefined);
      if (updated) currentCampaign.set(updated);
      addToast('Campaign updated', 'success');
    } catch (e) {
      addToast(`Failed: ${e}`, 'error');
    } finally {
      saving = false;
    }
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
  <title>Settings - {$currentCampaign?.name ?? 'Campaign'} - dndscribe</title>
</svelte:head>

<main class="wrap">
  <div class="eyebrow">{$currentCampaign?.name ?? ''}</div>
  <h1 class="page-title">Campaign settings</h1>

  <section class="card section">
    <div class="section-head">
      <h2 class="section-title">Details</h2>
    </div>
    <div class="form">
      <label class="field"><span>Name</span><input class="input" bind:value={name} /></label>
      <label class="field"><span>Description</span><textarea class="input" rows="3" bind:value={description}></textarea></label>
      <div><button class="btn-primary" onclick={saveDetails} disabled={!dirty || saving || !name.trim()}>{saving ? 'Saving…' : 'Save changes'}</button></div>
    </div>
  </section>

  <section class="card section" id="hotwords">
    <div class="section-head">
      <h2 class="section-title">Hotwords <span class="subtle">({$hotwords.length})</span></h2>
      <p class="muted small">Names and terms the transcriber should favour: characters, places, factions. Editing a transcript line can add new ones automatically.</p>
    </div>
    <HotwordManager {campaignId} />
  </section>

  <section class="card section" id="ignored">
    <div class="section-head">
      <h2 class="section-title">Ignored users <span class="subtle">({$ignoredUsers.length})</span></h2>
      <p class="muted small">Ignored users are not transcribed. Ignore someone from the voice members list on the Live page.</p>
    </div>
    {#if $ignoredUsers.length === 0}
      <p class="subtle small">Nobody is ignored.</p>
    {:else}
      <ul class="ignored">
        {#each $ignoredUsers as u (u.discordUserId)}
          <li>
            <span class="who">{u.discordUsername}</span>
            <span class="subtle tiny">since {new Date(u.ignoredAt).toLocaleDateString()}</span>
            <button class="btn btn-sm" onclick={() => unignore(u.discordUserId)}>Stop ignoring</button>
          </li>
        {/each}
      </ul>
    {/if}
  </section>
</main>

<style>
  .wrap { max-width: 900px; width: 100%; margin: 0 auto; padding: 36px 32px 64px; }
  .eyebrow { margin-bottom: 10px; }
  .page-title { margin-bottom: 24px; }
  .section { padding: 22px 24px; margin-bottom: 18px; }
  .section-head { margin-bottom: 16px; display: flex; flex-direction: column; gap: 4px; }
  .form { display: flex; flex-direction: column; gap: 14px; max-width: 560px; }
  .ignored { list-style: none; display: flex; flex-direction: column; }
  .ignored li { display: flex; align-items: center; gap: 12px; padding: 10px 0; border-bottom: 1px solid var(--border-color); }
  .ignored li:last-child { border-bottom: 0; }
  .who { font-weight: 600; }
  .ignored .btn { margin-left: auto; }
  @media (max-width: 650px) { .wrap { padding: 24px 16px 48px; } .section { padding: 18px 16px; } }
</style>
