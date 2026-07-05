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
    <button class="btn-primary" onclick={leave}>Leave {$botStatus.channelName}</button>
  {:else}
    <button class="btn" onclick={toggle} disabled={loading}>
      {loading ? 'Joining...' : 'Join Channel'}
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
  .dropdown-overlay {
    position: fixed; inset: 0; z-index: 49;
  }
  .dropdown {
    position: absolute; top: 100%; left: 0; margin-top: 4px;
    background: var(--bg-secondary); border: 1px solid var(--border);
    border-radius: 4px; min-width: 220px; z-index: 50;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
  }
  .dropdown-item {
    display: block; width: 100%; text-align: left;
    padding: 8px 14px; background: none; border: none;
    color: var(--text-primary); font-size: 13px;
    border-bottom: 1px solid var(--border);
  }
  .dropdown-item:last-child { border-bottom: none; }
  .dropdown-item:hover { background: var(--bg-elevated); }
  .dropdown-item-header { display: flex; justify-content: space-between; }
  .member-count { color: var(--text-muted); font-size: 11px; }
  .dropdown-item-sub { color: var(--text-secondary); font-size: 11px; margin-top: 3px; }
  .dropdown-empty { padding: 12px; text-align: center; color: var(--text-muted); font-size: 12px; }
</style>
