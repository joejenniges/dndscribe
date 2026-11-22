<script lang="ts">
  // Persistent bottom bar: Discord voice status, what the audience hears,
  // master volume and Stop all. No previous/next/pause here: those live on
  // the decks and the mixer, so there's one place for each control.
  import { djCmd } from '$lib/dj/control';
  import { deckPosition, DECK_NAMES } from '$lib/dj/live';
  import { clock } from '$lib/dj/clock';
  import { botStatus } from '$lib/stores/ws';
  import { addToast } from '$lib/stores/toast';
  import type { DjState } from '$lib/types';
  import Fader from './Fader.svelte';
  import { fmtMs } from './format';

  let { st, live }: { st: DjState | null; live: boolean } = $props();

  const inVoice = $derived(!!st?.connected);
  const air = $derived(st && st.onAir >= 0 ? st.onAir : -1);
  const airDeck = $derived(air >= 0 ? st!.decks[air] : null);
  const pos = $derived(air >= 0 ? deckPosition(st, air, $clock) : 0);
  const anything = $derived(!!st && (st.decks.some((d) => d.track) || st.sfx.length > 0 || st.queueTotal > 0));

  async function stopAll() {
    try {
      await djCmd('mix.stopAll');
    } catch (e) {
      addToast((e as Error).message, 'error');
    }
  }
</script>

<footer class="bar" class:stale={!live}>
  <div class="discord" class:ok={inVoice}>
    <span class="dot" aria-hidden="true"></span>
    <span class="txt">
      <strong>{inVoice ? 'Discord connected' : 'Not in voice'}</strong>
      <span class="tiny subtle">{inVoice ? ($botStatus.channelName ?? 'Voice') : 'Join a channel to play'}</span>
    </span>
  </div>

  <div class="air">
    {#if airDeck?.track}
      <span class="tag deck-{air}" aria-hidden="true">{DECK_NAMES[air]}</span>
      <span class="txt">
        <strong title={airDeck.track.title}>{airDeck.track.title}</strong>
        <span class="tiny subtle">{st?.sourceName || airDeck.track.artist || 'On air'}</span>
      </span>
      <span class="time tiny">{fmtMs(pos)}{airDeck.track.durationMs ? ` / ${fmtMs(airDeck.track.durationMs)}` : ''}</span>
    {:else}
      <span class="txt"><strong class="subtle">Nothing on air</strong></span>
    {/if}
  </div>

  <div class="master">
    <span class="tiny subtle">Master</span>
    <Fader value={st?.volume ?? 0.5} cmd="mix.master" args={(v) => ({ value: v })} label="Master volume" disabled={!live} />
  </div>

  <button class="btn stop" onclick={stopAll} disabled={!live || !anything}
    title="Fade out both decks and every effect, and clear the queue">&#9632; Stop all</button>
</footer>

<style>
  .bar {
    display: flex; align-items: center; gap: 18px; padding: 8px 16px; min-height: 64px;
    background: var(--surface-sunken); border-top: 1px solid var(--border-color);
  }
  .stale { opacity: 0.7; }
  .txt { display: flex; flex-direction: column; min-width: 0; }
  .txt strong { font-size: 14px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .discord { display: flex; align-items: center; gap: 10px; flex-shrink: 0; color: var(--warning); }
  .discord.ok { color: var(--success); }
  .discord strong { color: inherit; }
  .dot { width: 10px; height: 10px; border-radius: 50%; background: currentColor; flex-shrink: 0; }
  .air { flex: 1; min-width: 0; display: flex; align-items: center; gap: 10px; }
  .tag { width: 26px; height: 26px; border-radius: 6px; display: grid; place-items: center; font-weight: 800; font-size: 12px; color: #111; flex-shrink: 0; background: var(--deck-a); }
  .tag.deck-1 { background: var(--deck-b); }
  .time { font-variant-numeric: tabular-nums; color: var(--text-2); flex-shrink: 0; }
  .master { display: flex; align-items: center; gap: 8px; width: 240px; flex-shrink: 0; }
  .master :global(.fader) { flex: 1; }
  .stop { min-height: 44px; flex-shrink: 0; }
  @media (max-width: 900px) {
    .bar { flex-wrap: wrap; gap: 10px; }
    .master { width: auto; flex: 1 1 200px; }
    .discord .tiny { display: none; }
  }
</style>
