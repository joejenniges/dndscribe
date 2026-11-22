<script lang="ts">
  // Which voice channel the bot plays into, with Join / Leave. Music and
  // effects only play while the bot is in a channel. A recording session
  // already has the bot in voice; otherwise it can join just for music.
  import { channels as channelsApi, music } from '$lib/api';
  import { djCmd } from '$lib/dj/control';
  import { botStatus } from '$lib/stores/ws';
  import { addToast } from '$lib/stores/toast';
  import type { DjState, VoiceChannel } from '$lib/types';

  let { st, live }: { st: DjState | null; live: boolean } = $props();

  const inVoice = $derived(!!st?.connected);
  let channelList = $state<VoiceChannel[]>([]);
  let joinChannelId = $state('');
  let busy = $state(false);

  async function loadChannels() {
    try {
      channelList = await channelsApi.list();
      // Default to the busiest channel: that's where the table is.
      if (!joinChannelId && channelList.length) {
        joinChannelId = [...channelList].sort((a, b) => b.members.length - a.members.length)[0].id;
      }
    } catch (e) {
      addToast(`Couldn't load voice channels: ${(e as Error).message}`, 'error');
    }
  }

  $effect(() => {
    if (!$botStatus.recording && !$botStatus.musicOnly) loadChannels();
  });

  async function join() {
    if (!joinChannelId) return;
    busy = true;
    try {
      await music.join(joinChannelId);
    } catch (e) {
      addToast(`Couldn't join: ${(e as Error).message}`, 'error');
    } finally {
      busy = false;
    }
  }

  async function leave() {
    busy = true;
    try {
      await djCmd('mix.stopAll').catch(() => {});
      await music.leave();
    } catch (e) {
      addToast(`Couldn't leave: ${(e as Error).message}`, 'error');
    } finally {
      busy = false;
    }
  }

  async function tone() {
    try {
      await djCmd('tone.play', { seconds: 2 });
    } catch (e) {
      addToast((e as Error).message, 'error');
    }
  }
</script>

<div class="voice">
  {#if $botStatus.recording || $botStatus.musicOnly}
    <span class="state" class:ok={inVoice}>
      <span class="dot" aria-hidden="true"></span>
      {inVoice ? 'In' : 'Connecting to'} {$botStatus.channelName ?? 'voice'}{$botStatus.recording ? ' (recording)' : ''}
    </span>
    {#if $botStatus.musicOnly}
      <button class="btn btn-sm" onclick={leave} disabled={busy}>Leave</button>
    {/if}
    <button class="btn btn-sm btn-ghost" onclick={tone} disabled={!inVoice || !live}
      title="Play a 2 second 440 Hz tone to check the bot can be heard">Test tone</button>
  {:else}
    <span class="state off"><span class="dot" aria-hidden="true"></span>Not in voice</span>
    <select class="select" bind:value={joinChannelId} aria-label="Voice channel for music">
      {#each channelList as c (c.id)}
        <option value={c.id}>{c.name}{c.members.length ? ` (${c.members.length})` : ''}</option>
      {/each}
    </select>
    <button class="btn-primary btn-sm" onclick={join} disabled={!joinChannelId || busy}>
      {busy ? 'Joining…' : 'Join'}
    </button>
  {/if}
</div>

<style>
  .voice { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .state { display: inline-flex; align-items: center; gap: 8px; font-size: 13px; font-weight: 600; color: var(--text-2); }
  .state.ok { color: var(--success); }
  .state.off { color: var(--warning); }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; }
  .select { min-height: var(--control-h-sm); max-width: 200px; }
</style>
