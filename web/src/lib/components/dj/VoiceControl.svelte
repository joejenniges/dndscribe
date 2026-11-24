<script lang="ts">
  // Which voice channel the bot plays into, with Join / Leave. Music and
  // effects only play while the bot is in a channel. A recording session
  // already has the bot in voice; otherwise it can join just for music.
  //
  // The test channel plays in the browser instead of Discord: "Listen here"
  // is per device, remembered, and switched on automatically on the device
  // that joins it.
  import { channels as channelsApi, music } from '$lib/api';
  import { djCmd } from '$lib/dj/control';
  import { botStatus } from '$lib/stores/ws';
  import { addToast } from '$lib/stores/toast';
  import { listenWanted, listenState, startListening, stopListening } from '$lib/dj/listen';
  import type { DjState, VoiceChannel } from '$lib/types';

  // The one place the page shows connection state: server link, voice
  // channel and (test channel) browser audio. Other devices on the page are
  // in the tooltip.
  let { st, live, devices = '' }: { st: DjState | null; live: boolean; devices?: string } = $props();

  const inVoice = $derived(!!st?.connected);
  const TEST_ID = 'test-channel';
  const inTest = $derived(!!$botStatus.musicOnly && $botStatus.channelId === TEST_ID);

  // Follow the bot: listen while it's in the test channel (if this device
  // wants to), stop the socket when it leaves. Doesn't change the remembered
  // choice. Stops only on the in -> out transition: Join starts listening
  // before the bot reports the channel, and stopping then would undo it.
  let wasInTest = false;
  $effect(() => {
    // 'elsewhere': another tab of this browser took over; don't fight it.
    if (inTest && $listenWanted && $listenState !== 'elsewhere') void startListening(false);
    else if (!inTest && wasInTest) stopListening(false);
    wasInTest = inTest;
  });

  function toggleListen() {
    if ($listenWanted && $listenState !== 'blocked' && $listenState !== 'elsewhere') stopListening();
    else void startListening();
  }
  // 'off' while wanted (between leaving and the effect restarting it) reads
  // as "Listen here", not "Connecting…": that used to look stuck.
  const listenLabel = $derived(
    !$listenWanted ? 'Listen here'
      : $listenState === 'on' ? 'Listening'
      : $listenState === 'blocked' ? 'Tap to hear'
      : $listenState === 'error' ? "Can't play audio"
      : $listenState === 'unreachable' ? 'Audio unavailable, retrying'
      : $listenState === 'elsewhere' ? 'Playing in another tab'
      : $listenState === 'connecting' ? 'Starting audio…'
      : 'Listen here',
  );
  const tip = $derived(devices ? `Also open on: ${devices}` : undefined);
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
    // Joining the test channel from this device means "let me hear it
    // here". Started inside the click so the browser allows the audio.
    if (joinChannelId === TEST_ID) void startListening();
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
  {#if !live}
    <span class="state off" role="status" title={tip}><span class="dot" aria-hidden="true"></span>Server offline, reconnecting…</span>
  {:else if $botStatus.recording || $botStatus.musicOnly}
    <span class="state" class:ok={inVoice} role="status" title={tip}>
      <span class="dot" aria-hidden="true"></span>
      {#if inTest}
        Test channel{inVoice ? '' : ' (starting)'}
      {:else}
        {inVoice ? 'In' : 'Connecting to'} {$botStatus.channelName ?? 'voice'}{$botStatus.recording ? ' (recording)' : ''}
      {/if}
    </span>
    {#if $botStatus.musicOnly}
      <button class="btn btn-sm" onclick={leave} disabled={busy}>Leave</button>
    {/if}
    {#if inTest}
      <button class="btn btn-sm listen" class:on={$listenWanted && $listenState === 'on'} aria-pressed={$listenWanted}
        onclick={toggleListen}
        title="Play the test channel through this browser. Each device chooses for itself.">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3A4.5 4.5 0 0 0 14 8v8a4.5 4.5 0 0 0 2.5-4zM14 3.2v2.1a7 7 0 0 1 0 13.4v2.1a9 9 0 0 0 0-17.6z" /></svg>
        {listenLabel}
      </button>
    {/if}
    <button class="btn btn-sm btn-ghost" onclick={tone} disabled={!inVoice || !live}
      title="Play a 2 second 440 Hz tone to check the bot can be heard">Test tone</button>
  {:else}
    <span class="state off" role="status" title={tip}><span class="dot" aria-hidden="true"></span>Not in voice</span>
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
  .listen { display: inline-flex; align-items: center; gap: 6px; }
  .listen svg { width: 16px; height: 16px; fill: currentColor; }
  .listen.on { border-color: var(--success); color: var(--success); }
</style>
