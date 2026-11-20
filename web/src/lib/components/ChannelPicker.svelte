<script lang="ts">
  import { goto } from '$app/navigation';
  import { channels as channelsApi, voice } from '$lib/api';
  import { botStatus } from '$lib/stores/ws';
  import { addToast } from '$lib/stores/toast';
  import type { VoiceChannel } from '$lib/types';

  let { campaignId }: { campaignId: number } = $props();

  let channelList: VoiceChannel[] = $state([]);
  let open = $state(false);
  let loading = $state(false);

  async function loadChannels() {
    try {
      channelList = await channelsApi.list();
    } catch (e) {
      addToast(`Failed to load channels: ${e}`, 'error');
    }
  }

  function toggle() {
    open = !open;
    if (open) loadChannels();
  }

  async function joinChannel(channelId: string) {
    loading = true;
    try {
      await voice.join(channelId, campaignId);
      open = false;
      // Navigate to live view after successful join
      goto(`/campaigns/${campaignId}/live`);
    } catch (e) {
      addToast(`Failed to join: ${e}`, 'error');
    } finally {
      loading = false;
    }
  }

  async function leave() {
    try {
      await voice.leave();
    } catch (e) {
      addToast(`Failed to leave: ${e}`, 'error');
    }
  }
</script>

<div class="channel-picker">
  {#if $botStatus.recording}
    <button class="btn-danger" onclick={leave} title="Stop recording and leave the voice channel">Leave {$botStatus.channelName}</button>
  {:else}
    <button class="btn-primary" onclick={toggle} disabled={loading} aria-haspopup="listbox" aria-expanded={open}>
      {loading ? 'Joining…' : 'Join channel'}
    </button>
  {/if}

  {#if open}
    <!-- svelte-ignore a11y_click_events_have_key_events -->
    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <div class="dropdown-overlay" onclick={() => open = false}></div>
    <div class="dropdown">
      {#each channelList as ch}
        <button class="dropdown-item" onclick={() => joinChannel(ch.id)}>
          <div class="dropdown-item-header">
            <span>{ch.name}</span>
            <span class="member-count">{ch.members.length}</span>
          </div>
          {#if ch.members.length > 0}
            <div class="dropdown-item-sub">{ch.members.join(', ')}</div>
          {/if}
        </button>
      {/each}
      {#if channelList.length === 0}
        <div class="dropdown-empty">No voice channels</div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .channel-picker { position: relative; }
  .dropdown-overlay { position: fixed; inset: 0; z-index: 49; }
  .dropdown {
    position: absolute; top: calc(100% + 6px); right: 0;
    background: var(--surface-raised); border: 1px solid var(--border-strong);
    border-radius: var(--radius); min-width: 280px; z-index: 50; padding: 6px;
    box-shadow: var(--shadow);
  }
  .dropdown-item {
    display: block; width: 100%; text-align: left; min-height: 48px;
    padding: 10px 12px; background: none; border: none; border-radius: var(--radius-sm);
    color: var(--text); font-size: 14px;
  }
  .dropdown-item:hover { background: var(--surface-hover); }
  .dropdown-item-header { display: flex; justify-content: space-between; gap: 12px; font-weight: 600; }
  .member-count { color: var(--text-3); font-size: 12px; }
  .dropdown-item-sub { color: var(--text-2); font-size: 12px; margin-top: 3px; }
  .dropdown-empty { padding: 14px; text-align: center; color: var(--text-3); font-size: 13px; }
</style>
