<script lang="ts">
  // One music deck: what's loaded, transport (play/pause, stop, back to
  // start, unload), channel gain and level. Everything here is a command to
  // the server; the panel only shows the server's state (plus a local
  // position extrapolation).
  //
  // "On air" means actually heard: playing, in voice and not faded out by
  // the crossfader. A deck playing on the far side of the fader says
  // "Faded out" instead, since play never moves the crossfader.
  //
  // The round "art" is a progress ring around the track's initials. There's
  // no album art in the library, and it's deliberately not a turntable: the
  // decks don't scratch or beatmatch.
  import { djCmd } from '$lib/dj/control';
  import { deckPosition, deckHeard, DECK_NAMES } from '$lib/dj/live';
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
  const heard = $derived(deckHeard(st, deck, $clock));
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
  const selected = $derived($hotkeysOn && $selectedDeck === deck);

  let busy = $state<string | null>(null);
  async function run(key: string, ...steps: [string, Record<string, unknown>?][]) {
    busy = key;
    try {
      for (const [cmd, args] of steps) await djCmd(cmd, { deck, ...(args ?? {}) });
    } catch (e) {
      addToast((e as Error).message, 'error');
    } finally {
      busy = null;
    }
  }
  function stop() {
    const steps: [string, Record<string, unknown>?][] = [];
    if (d?.playing) steps.push(['deck.pause']);
    steps.push(['deck.seek', { positionMs: 0 }]);
    run('stop', ...steps);
  }

  type Status = { text: string; kind: 'air' | 'warn' | 'err' | 'muted' | 'plain' };
  const status = $derived.by((): Status => {
    if (!d) return { text: '', kind: 'plain' };
    if (d.loading) return { text: 'Loading', kind: 'muted' };
    if (d.error) return { text: d.error, kind: 'err' };
    if (!t) return { text: 'Empty', kind: 'muted' };
    if (d.ended) return { text: 'Ended', kind: 'muted' };
    if (d.playing && heard) return { text: fadingIn ? 'Fading in' : 'On air', kind: 'air' };
    if (d.playing) return { text: inVoice ? 'Faded out' : 'Not in voice', kind: 'warn' };
    if (pos > 0) return { text: 'Paused', kind: 'plain' };
    return { text: d.auto ? 'Up next' : 'Cued', kind: 'plain' };
  });
  const canTransport = $derived(live && !!t && !d?.loading && busy === null);
</script>

<!-- Pointer-down anywhere on the card selects the deck for the keyboard
     shortcuts; it's a convenience, the deck's own buttons do the work. -->
<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
<section class="deck card deck-{deck}" class:compact class:on-air={heard} class:stale={!live} class:selected
  onpointerdown={() => selectedDeck.set(deck)}
  aria-label="Deck {DECK_NAMES[deck]}{selected ? ', selected' : ''}">
  <header>
    <span class="tag" aria-hidden="true">{DECK_NAMES[deck]}</span>
    <div class="title">
      {#if t}
        <strong title={t.title}>{t.title}</strong>
        <span class="sub" title={[t.artist, folder].filter(Boolean).join(' · ')}>{[t.artist, folder].filter(Boolean).join(' · ') || ' '}</span>
      {:else}
        <strong class="dim">{d?.loading ? 'Loading…' : 'No track loaded'}</strong>
        <span class="sub">Load one from the library{compact ? '' : ' or the queue'}</span>
      {/if}
    </div>
    <span class="status {status.kind}" title={status.text}>{status.text}</span>
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
        <span class="rest">{t?.durationMs ? `-${fmtMs(t.durationMs - pos)}` : ''}</span>
      </div>
      <div class="transport" role="group" aria-label="Deck {DECK_NAMES[deck]} transport">
        {#if d?.playing}
          <button class="btn play on" onclick={() => run('pause', ['deck.pause'])} disabled={!live || busy !== null}
            aria-label="Pause deck {DECK_NAMES[deck]}" title="Pause (Space when selected)">
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 5h4v14H7zM13 5h4v14h-4z" /></svg>
          </button>
        {:else}
          <button class="btn play" onclick={() => run('play', ['deck.play'])}
            disabled={!canTransport || !inVoice}
            aria-label="Play deck {DECK_NAMES[deck]}" title={!inVoice ? 'The bot is not in a voice channel' : 'Play (Space when selected)'}>
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z" /></svg>
          </button>
        {/if}
        <button class="btn icon" onclick={stop}
          disabled={!canTransport || (!d?.playing && pos === 0)} aria-label="Stop deck {DECK_NAMES[deck]}" title="Stop (pause and go back to the start)">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6h12v12H6z" /></svg>
        </button>
        <button class="btn icon" onclick={() => run('restart', ['deck.seek', { positionMs: 0 }])}
          disabled={!canTransport} aria-label="Back to the start of deck {DECK_NAMES[deck]}" title="Back to start (R when selected)">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 5h2v14H6zM20 5v14L9 12z" /></svg>
        </button>
        <button class="btn icon" onclick={() => run('eject', ['deck.eject'])}
          disabled={!live || !t || busy !== null} aria-label="Unload deck {DECK_NAMES[deck]}" title="Unload">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 15h14l-7-9zM5 17h14v2H5z" /></svg>
        </button>
      </div>
    </div>

    {#if !compact}
      <div class="gain">
        <Fader vertical value={d?.gain ?? 1} cmd="deck.gain" args={(v) => ({ deck, value: v })}
          label="Deck {DECK_NAMES[deck]} gain" disabled={!live} color={`var(--deck-${deck ? 'b' : 'a'})`} ticks={6} />
        <Meter source={deck} label="Deck {DECK_NAMES[deck]} level" />
      </div>
    {/if}
  </div>
</section>

<style>
  .deck {
    --deck-color: var(--deck-a);
    position: relative; overflow: hidden;
    padding: 12px 14px 12px 16px; display: flex; flex-direction: column; gap: 12px; min-width: 0;
    background: linear-gradient(180deg, color-mix(in srgb, var(--deck-color) 7%, var(--surface)), var(--surface) 55%);
    transition: border-color 0.2s, box-shadow 0.2s;
  }
  /* Deck colour stripe down the left edge, full strength when heard. */
  .deck::before { content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: var(--deck-color); opacity: 0.45; }
  .deck.deck-1 { --deck-color: var(--deck-b); }
  .deck.on-air { border-color: color-mix(in srgb, var(--deck-color) 65%, var(--border-color)); }
  .deck.on-air::before { opacity: 1; }
  .deck.selected { box-shadow: 0 0 0 2px var(--deck-color); }
  .stale { opacity: 0.6; }

  header { display: flex; align-items: center; gap: 10px; min-width: 0; }
  .tag {
    flex-shrink: 0; width: 32px; height: 32px; border-radius: 8px; display: grid; place-items: center;
    background: var(--deck-color); color: #111; font-weight: 800; font-size: 16px;
  }
  .selected .tag { box-shadow: 0 0 0 3px color-mix(in srgb, var(--deck-color) 35%, transparent); }
  .title { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 1px; }
  .title strong { font-size: 15px; line-height: 1.25; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .title .dim { color: var(--text-2); }
  .sub { font-size: 13px; color: var(--text-2); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

  .status {
    flex-shrink: 0; max-width: 45%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    font-size: 11px; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase;
    padding: 4px 8px; border-radius: 99px; color: var(--text-2); background: var(--surface-raised); border: 1px solid var(--border-color);
  }
  .status.muted { color: var(--text-3); }
  .status.air { color: #fff; background: var(--accent); border-color: var(--accent); box-shadow: 0 0 12px color-mix(in srgb, var(--accent) 55%, transparent); }
  .status.warn { color: var(--warning); border-color: color-mix(in srgb, var(--warning) 45%, transparent); }
  .status.err { color: var(--danger); text-transform: none; letter-spacing: 0; font-weight: 600; }

  .body { display: flex; align-items: center; gap: 16px; min-height: 0; }
  .ring {
    --p: 0;
    flex-shrink: 0; width: 92px; height: 92px; border-radius: 50%; display: grid; place-items: center;
    background: conic-gradient(var(--deck-color) calc(var(--p) * 360deg), var(--surface-sunken) 0);
    position: relative; box-shadow: 0 4px 14px rgba(0, 0, 0, 0.45);
  }
  .ring::before { content: ''; position: absolute; inset: 6px; border-radius: 50%; background: radial-gradient(circle at 50% 40%, var(--surface-raised), var(--surface-sunken)); }
  .ring span { position: relative; font-weight: 800; font-size: 22px; color: var(--text-2); letter-spacing: 0.04em; }
  .on-air .ring span { color: var(--deck-color); }

  .mid { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 10px; }
  .time { display: flex; align-items: baseline; gap: 10px; font-variant-numeric: tabular-nums; }
  .time .big { font-size: 26px; font-weight: 700; letter-spacing: -0.01em; }
  .time .rest { font-size: 14px; color: var(--text-2); }

  .transport { display: flex; gap: 6px; }
  .transport .btn { min-height: 46px; min-width: 46px; padding: 0; display: grid; place-items: center; flex: 0 0 auto; }
  .transport svg { width: 20px; height: 20px; fill: currentColor; }
  .transport .play {
    flex: 1 1 64px; max-width: 110px;
    background: color-mix(in srgb, var(--deck-color) 16%, var(--surface-raised));
    border-color: color-mix(in srgb, var(--deck-color) 50%, var(--border-strong));
  }
  .play svg { width: 24px; height: 24px; }
  .transport .play.on { background: var(--deck-color); color: #111; border-color: var(--deck-color); }

  .gain { display: flex; gap: 6px; align-items: stretch; height: 132px; flex-shrink: 0; }

  /* Performance: waveform inside the card, time and transport on one row,
     no ring or gain (gain is a setup control, kept in Studio). */
  .compact { gap: 8px; padding: 10px 12px 10px 14px; }
  .compact .mid { flex-direction: row; align-items: center; justify-content: space-between; }
  .compact .time .big { font-size: 20px; }
  .compact .transport .btn { min-height: 52px; min-width: 52px; }
  .compact .transport .play { flex: 0 0 96px; }

  /* Narrow deck card: smaller ring and buttons before anything overlaps.
     The card is the container, so this follows the card, not the window. */
  .deck { container: deck / inline-size; }
  @container deck (max-width: 440px) {
    .body { gap: 12px; }
    .ring { width: 60px; height: 60px; }
    .ring span { font-size: 16px; }
    .ring::before { inset: 5px; }
    .time .big { font-size: 22px; }
    .transport { gap: 5px; }
    .transport .btn { min-width: 38px; min-height: 42px; }
    .transport .play { flex: 1 1 52px; }
    .transport svg { width: 18px; height: 18px; }
  }
  @container deck (max-width: 320px) {
    .ring { display: none; }
  }
</style>
