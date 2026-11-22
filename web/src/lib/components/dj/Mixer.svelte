<script lang="ts">
  // Mixer: crossfader between the decks, master volume and level, Auto DJ
  // and Next. Compact on purpose (no title row; the glide status replaces
  // the fader readout) so the deck row stays ~200-240px tall.
  //
  // Crossfades are manual. The A and B buttons glide the fader to that side
  // over the global fade time (mix.fadeTo); it only moves the fader, never
  // starts or stops a deck. Scene changes and Auto DJ also move it. During
  // any such glide the server sends where the fader started, where it's
  // going and for how long, and the fader is animated here from that;
  // grabbing the fader cancels the glide on the server, so it never fights
  // the user.
  import { onMount } from 'svelte';
  import { djCmd } from '$lib/dj/control';
  import { crossfaderNow, transitionLeft, DECK_NAMES } from '$lib/dj/live';
  import { addToast } from '$lib/stores/toast';
  import type { DjState } from '$lib/types';
  import Fader from './Fader.svelte';
  import Meter from './Meter.svelte';

  let { st, live, strip = false }: { st: DjState | null; live: boolean; strip?: boolean } = $props();

  // Animated crossfader position (rAF only does work while a glide runs).
  let xf = $state(0);
  let left = $state(0);
  onMount(() => {
    let raf = 0;
    const frame = () => {
      raf = requestAnimationFrame(frame);
      const now = Date.now();
      const v = crossfaderNow(st, now);
      if (Math.abs(v - xf) > 0.0005) xf = v;
      const l = Math.ceil(transitionLeft(st, now));
      if (l !== left) left = l;
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  });

  const tr = $derived(st?.transition ?? null);
  const trLabel = $derived.by(() => {
    if (!tr) return '';
    const what = { scene: 'Scene change', auto: 'Auto DJ', next: 'Next', track: 'Crossfade', manual: 'Fading' }[tr.kind] ?? 'Crossfade';
    return `${what} to ${DECK_NAMES[tr.toDeck]}${left > 0 ? ` · ${left}s` : ''}`;
  });

  let busy = $state(false);
  async function run(cmd: string, args: Record<string, unknown> = {}) {
    busy = true;
    try {
      await djCmd(cmd, args);
    } catch (e) {
      addToast((e as Error).message, 'error');
    } finally {
      busy = false;
    }
  }
  const pct = (v: number) => `${Math.round(v * 100)}%`;
  const xfLabel = (v: number) =>
    v < 0.005 ? 'Full A' : v > 0.995 ? 'Full B' : Math.abs(v - 0.5) < 0.01 ? 'Centre' : `A ${Math.round((1 - v) * 100)} · ${Math.round(v * 100)} B`;
  const gliding = (i: number) => tr?.toDeck === i;
</script>

{#snippet side(i: 0 | 1)}
  <button class="side side-{i}" class:at={(i === 0 && xf < 0.005) || (i === 1 && xf > 0.995)} class:gliding={gliding(i)}
    onclick={() => run('mix.fadeTo', { deck: i })} disabled={!live}
    aria-label="Fade the crossfader to deck {DECK_NAMES[i]}" title="Fade to deck {DECK_NAMES[i]} (X fades to the selected deck)">
    {DECK_NAMES[i]}
  </button>
{/snippet}

<section class="mixer card" class:strip class:stale={!live} aria-label="Mixer">
  <div class="xf">
    <div class="xf-row">
      {@render side(0)}
      <Fader value={xf} cmd="mix.crossfader" args={(v) => ({ value: v })} label="Crossfader, A to B"
        disabled={!live} showValue={false} crossfader big={strip} format={xfLabel} ticks={11} />
      {@render side(1)}
    </div>
    <div class="xf-read" aria-hidden="true">
      {#if tr}<span class="glide">{trLabel}</span>{:else}{xfLabel(xf)}{/if}
    </div>
  </div>

  <div class="master">
    <span class="lbl" aria-hidden="true">Master</span>
    <div class="master-ctl">
      <Fader value={st?.volume ?? 0.5} cmd="mix.master" args={(v) => ({ value: v })} label="Master volume"
        disabled={!live} format={pct} color="var(--accent)" />
      <Meter source="master" vertical={false} label="Master level" />
    </div>
  </div>

  <div class="actions">
    <button class="btn toggle" class:on={st?.autoDJ} aria-pressed={!!st?.autoDJ} disabled={!live || busy}
      onclick={() => run('mix.autoDJ', { on: !st?.autoDJ })}
      title="Auto DJ loads the next queued track on the free deck and crossfades to it before the current one ends">
      <span class="dot" aria-hidden="true"></span>Auto DJ
    </button>
    <button class="btn" onclick={() => run('mix.next')} disabled={!live || busy || !st?.connected}
      title="Crossfade to the next queued track now (N)">
      Next
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 5v14l9-7zM16 5h2v14h-2z" /></svg>
    </button>
  </div>
</section>

<style>
  .mixer {
    padding: 10px 14px; display: flex; flex-direction: column; gap: 10px; min-width: 0;
    background: linear-gradient(180deg, var(--surface-raised), var(--surface));
  }
  .stale { opacity: 0.6; }
  .glide { font-size: 12px; font-weight: 700; color: var(--warning); white-space: nowrap; }

  .xf { display: flex; flex-direction: column; gap: 4px; }
  .xf-row { display: flex; align-items: center; gap: 8px; }
  .xf-row :global(.fader) { flex: 1; }
  .xf-read { text-align: center; font-size: 13px; font-weight: 600; color: var(--text-2); font-variant-numeric: tabular-nums; min-height: 18px; margin-top: -4px; }

  .side {
    --c: var(--deck-a);
    flex-shrink: 0; width: 36px; height: 40px; border-radius: 8px; padding: 0;
    font-weight: 800; font-size: 15px; color: var(--c);
    background: color-mix(in srgb, var(--c) 12%, var(--surface-raised));
    border: 1px solid color-mix(in srgb, var(--c) 45%, var(--border-strong));
    transition: background 0.15s, color 0.15s;
  }
  .side-1 { --c: var(--deck-b); }
  .side:hover:not(:disabled) { background: color-mix(in srgb, var(--c) 24%, var(--surface-raised)); }
  .side.at { background: var(--c); color: #111; border-color: var(--c); }
  .side.gliding { animation: glow 1s ease-in-out infinite; }
  @keyframes glow { 50% { box-shadow: 0 0 0 3px color-mix(in srgb, var(--c) 45%, transparent); } }
  @media (prefers-reduced-motion: reduce) { .side.gliding { animation: none; box-shadow: 0 0 0 2px var(--c); } }
  .side:disabled { opacity: 0.5; cursor: not-allowed; }

  .master { display: flex; align-items: center; gap: 10px; }
  .lbl { font-size: 12px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: var(--text-3); }
  .master-ctl { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 2px; }

  .actions { display: flex; gap: 8px; }
  .actions .btn { flex: 1 1 0; min-width: 0; padding: 0 8px; white-space: nowrap; min-height: 40px; justify-content: center; gap: 6px; display: inline-flex; align-items: center; }
  .actions svg { width: 16px; height: 16px; fill: currentColor; }
  .toggle .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--text-3); }
  .toggle.on { border-color: var(--success); color: var(--success); }
  .toggle.on .dot { background: var(--success); box-shadow: 0 0 6px var(--success); }

  /* Performance: one row. Crossfader | master | buttons. */
  .strip { flex-direction: row; align-items: center; gap: 18px; flex-wrap: wrap; padding: 8px 14px; }
  .strip .xf { flex: 3 1 360px; }
  .strip .side { width: 52px; height: 52px; font-size: 18px; }
  .strip .master { flex: 1 1 200px; }
  .strip .actions { flex: 0 1 auto; flex-wrap: wrap; }
  .strip .actions .btn { min-height: 52px; min-width: 110px; }
</style>
