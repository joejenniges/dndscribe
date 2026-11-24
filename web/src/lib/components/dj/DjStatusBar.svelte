<script lang="ts">
  // Persistent bottom bar: what the audience hears and Stop all. Voice and
  // server status live in one place only, the header's VoiceControl. Master volume is in the mixer. No previous/next/pause here: those live on
  // the decks and the mixer, so there's one place for each control.
  import { djCmd } from '$lib/dj/control';
  import { deckPosition, heardDeck, DECK_NAMES } from '$lib/dj/live';
  import { clock } from '$lib/dj/clock';
  import { addToast } from '$lib/stores/toast';
  import type { DjState } from '$lib/types';
  import { fmtMs } from './format';

  let { st, live }: { st: DjState | null; live: boolean } = $props();

  const inVoice = $derived(!!st?.connected);
  const air = $derived(heardDeck(st, $clock));
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
  <div class="air">
    {#if airDeck?.track}
      <span class="tag deck-{air}" aria-hidden="true">{DECK_NAMES[air]}</span>
      <span class="txt">
        <strong title={airDeck.track.title}>{airDeck.track.title}</strong>
        <span class="tiny subtle">{st?.sourceName || airDeck.track.artist || 'On air'}</span>
      </span>
      <span class="time tiny">{fmtMs(pos)}{airDeck.track.durationMs ? ` / ${fmtMs(airDeck.track.durationMs)}` : ''}</span>
    {:else}
      {@const silent = inVoice ? (st?.decks.findIndex((d) => d.playing && d.track) ?? -1) : -1}
      <span class="txt">
        <strong class="subtle">Nothing on air</strong>
        {#if silent >= 0}<span class="tiny warn">Deck {DECK_NAMES[silent]} is playing but faded out by the crossfader</span>{/if}
      </span>
    {/if}
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
  .air { flex: 1; min-width: 0; display: flex; align-items: center; gap: 10px; }
  .tag { width: 26px; height: 26px; border-radius: 6px; display: grid; place-items: center; font-weight: 800; font-size: 12px; color: #111; flex-shrink: 0; background: var(--deck-a); }
  .tag.deck-1 { background: var(--deck-b); }
  .warn { color: var(--warning); }
  .time { font-variant-numeric: tabular-nums; color: var(--text-2); flex-shrink: 0; }
  .stop { min-height: 44px; flex-shrink: 0; }
  @media (max-width: 900px) {
    .bar { flex-wrap: wrap; gap: 10px; }
  }
</style>
