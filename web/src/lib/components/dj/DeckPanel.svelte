<script lang="ts">
  // One music deck: what's loaded, play/pause, back to start, eject, channel
  // gain and level. Everything here is a command to the server; the panel
  // only shows the server's state (plus a local position extrapolation).
  //
  // The round "art" is a progress ring around the track's initials. There's
  // no album art in the library, and it's deliberately not a turntable: the
  // decks don't scratch or beatmatch.
  import { djCmd } from '$lib/dj/control';
  import { deckPosition, DECK_NAMES } from '$lib/dj/live';
  import { clock } from '$lib/dj/clock';
  import { selectedDeck, hotkeysOn } from '$lib/dj/hotkeys';
  import { addToast } from '$lib/stores/toast';
  import type { DjState } from '$lib/types';
  import Fader from './Fader.svelte';
  import Meter from './Meter.svelte';
  import DeckWave from './DeckWave.svelte';
  import { fmtMs } from './format';
  import { folderName, parentOf } from './library';

  let {
    st,
    deck,
    live,
    compact = false,
  }: {
    st: DjState | null;
    deck: 0 | 1;
    /** WebSocket up: commands can be sent. */
    live: boolean;
    compact?: boolean;
  } = $props();

  const d = $derived(st?.decks?.[deck] ?? null);
  const t = $derived(d?.track ?? null);
  const inVoice = $derived(!!st?.connected);
  const onAir = $derived(st?.onAir === deck);
  const fadingIn = $derived(st?.transition?.toDeck === deck);
  const pos = $derived(deckPosition(st, deck, $clock));
  const progress = $derived(t?.durationMs ? Math.min(1, pos / t.durationMs) : 0);
  const initials = $derived(
    (t?.title ?? '')
      .split(/\s+/)
      .filter((w) => /^[\p{L}\p{N}]/u.test(w))
      .slice(0, 2)
      .map((w) => w[0].toUpperCase())
      .join('') || '♪',
  );
  const folder = $derived(t?.path ? folderName(parentOf(t.path)) : '');

  let busy = $state<string | null>(null);
  async function run(key: string, cmd: string, args: Record<string, unknown> = {}) {
    busy = key;
    try {
      await djCmd(cmd, { deck, ...args });
    } catch (e) {
      addToast((e as Error).message, 'error');
    } finally {
      busy = null;
    }
  }

  const status = $derived.by(() => {
    if (!d) return '';
    if (d.loading) return 'Loading…';
    if (d.error) return d.error;
    if (!t) return 'Empty';
    if (d.ended) return 'Ended';
    if (fadingIn) return 'Fading in';
    if (d.playing) return onAir ? 'On air' : 'Playing';
    return d.auto ? 'Up next' : 'Cued';
  });
</script>

<!-- Pointer-down anywhere on the card selects the deck for the keyboard
     shortcuts; it's a convenience, the deck's own buttons do the work. -->
<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
<section class="deck card deck-{deck}" class:compact class:on-air={onAir} class:stale={!live}
  class:selected={$hotkeysOn && $selectedDeck === deck} onpointerdown={() => selectedDeck.set(deck)}
  aria-label="Deck {DECK_NAMES[deck]}{$hotkeysOn && $selectedDeck === deck ? ', selected' : ''}">
  <header>
    <span class="tag" aria-hidden="true">{DECK_NAMES[deck]}</span>
    <div class="title">
      {#if t}
        <strong title={t.title}>{t.title}</strong>
        <span class="tiny subtle">{[t.artist, folder].filter(Boolean).join(' · ') || ' '}</span>
      {:else}
        <strong class="subtle">{d?.loading ? 'Loading…' : 'No track loaded'}</strong>
        <span class="tiny subtle">Load one from the library{compact ? '' : ' or the queue'}</span>
      {/if}
    </div>
    <span class="status tiny" class:err={!!d?.error} class:air={onAir}>{status}</span>
  </header>

  {#if compact}
    <DeckWave {st} {deck} overviewOnly tall bare {live} />
  {/if}

  <div class="body">
    {#if !compact}
      <div class="ring" style:--p={progress} aria-hidden="true">
        <span>{initials}</span>
      </div>
    {/if}

    <div class="mid">
      <div class="time">
        <span class="big">{t ? fmtMs(pos) : '–:––'}</span>
        <span class="subtle">{t?.durationMs ? `-${fmtMs(t.durationMs - pos)}` : ''}</span>
      </div>
      <div class="transport">
        {#if d?.playing}
          <button class="btn play on" onclick={() => run('pause', 'deck.pause')} disabled={!live || busy !== null}
            aria-label="Pause deck {DECK_NAMES[deck]}">
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 5h4v14H7zM13 5h4v14h-4z" /></svg>
          </button>
        {:else}
          <button class="btn play" onclick={() => run('play', 'deck.play')}
            disabled={!live || !inVoice || !t || d?.loading || busy !== null}
            aria-label="Play deck {DECK_NAMES[deck]}" title={!inVoice ? 'The bot is not in a voice channel' : undefined}>
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z" /></svg>
          </button>
        {/if}
        <button class="btn icon" onclick={() => run('restart', 'deck.seek', { positionMs: 0 })}
          disabled={!live || !t || busy !== null} aria-label="Back to the start of deck {DECK_NAMES[deck]}" title="Back to start">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 5h2v14H6zM20 5v14L9 12z" /></svg>
        </button>
        <button class="btn icon" onclick={() => run('eject', 'deck.eject')}
          disabled={!live || !t || busy !== null} aria-label="Unload deck {DECK_NAMES[deck]}" title="Unload">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 15h14l-7-9zM5 17h14v2H5z" /></svg>
        </button>
      </div>
    </div>

    {#if !compact}
      <div class="gain">
        <Fader vertical value={d?.gain ?? 1} cmd="deck.gain" args={(v) => ({ deck, value: v })}
          label="Deck {DECK_NAMES[deck]} gain" disabled={!live} color={`var(--deck-${deck ? 'b' : 'a'})`} />
        <Meter source={deck} label="Deck {DECK_NAMES[deck]} level" />
      </div>
    {/if}
  </div>
</section>

<style>
  .deck { --deck-color: var(--deck-a); padding: 12px 14px; display: flex; flex-direction: column; gap: 10px; min-width: 0; }
  .deck.deck-1 { --deck-color: var(--deck-b); }
  .deck.on-air { border-color: color-mix(in srgb, var(--deck-color) 60%, var(--border-color)); }
  /* Selected for the keyboard (Tab switches): a solid ring in the deck
     colour, distinct from the softer on-air border. */
  .deck.selected { box-shadow: 0 0 0 2px var(--deck-color); }
  .deck.selected .tag { box-shadow: 0 0 0 3px color-mix(in srgb, var(--deck-color) 35%, transparent); }
  .stale { opacity: 0.6; }
  header { display: flex; align-items: center; gap: 10px; min-width: 0; }
  .tag {
    flex-shrink: 0; width: 28px; height: 28px; border-radius: 7px; display: grid; place-items: center;
    background: var(--deck-color); color: #111; font-weight: 800;
  }
  .title { flex: 1; min-width: 0; display: flex; flex-direction: column; }
  .title strong { font-size: 15px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .title .tiny { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .status { flex-shrink: 0; color: var(--text-2); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
  .status.air { color: var(--deck-color); }
  .status.err { color: var(--danger); text-transform: none; letter-spacing: 0; max-width: 40%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

  .body { display: flex; align-items: center; gap: 14px; min-height: 0; }
  .ring {
    --p: 0;
    flex-shrink: 0; width: 84px; height: 84px; border-radius: 50%; display: grid; place-items: center;
    background: conic-gradient(var(--deck-color) calc(var(--p) * 360deg), var(--surface-sunken) 0);
    position: relative;
  }
  .ring::before { content: ''; position: absolute; inset: 7px; border-radius: 50%; background: var(--surface-raised); }
  .ring span { position: relative; font-weight: 800; font-size: 22px; color: var(--text-2); letter-spacing: 0.04em; }

  .mid { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 10px; }
  .time { display: flex; align-items: baseline; gap: 10px; font-variant-numeric: tabular-nums; }
  .time .big { font-size: 24px; font-weight: 700; }
  .time .subtle { font-size: 13px; }
  .transport { display: flex; gap: 6px; flex-wrap: wrap; }
  .transport .btn { min-height: 46px; min-width: 46px; padding: 0; display: grid; place-items: center; }
  .transport svg { width: 22px; height: 22px; fill: currentColor; }
  .play { flex: 1 1 46px; max-width: 96px; background: color-mix(in srgb, var(--deck-color) 16%, var(--surface-raised)); border-color: color-mix(in srgb, var(--deck-color) 50%, var(--border-strong)); }
  .play.on { background: var(--deck-color); color: #111; border-color: var(--deck-color); }

  .gain { display: flex; gap: 6px; align-items: stretch; height: 120px; flex-shrink: 0; }

  /* Performance: waveform inside the card, time and transport on one row,
     no ring or gain (gain is a setup control, kept in Studio). */
  .compact { gap: 8px; padding: 10px 12px; }
  .compact .mid { flex-direction: row; align-items: center; justify-content: space-between; }
  .compact .time .big { font-size: 20px; }
  .compact .transport { flex-wrap: nowrap; }
  .compact .transport .btn { min-height: 52px; min-width: 52px; }
  .compact .play { flex: 0 0 auto; width: 96px; max-width: none; }
</style>
