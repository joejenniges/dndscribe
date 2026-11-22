<script lang="ts">
  // Mixer: the crossfader between the decks, master volume, Auto DJ and
  // Next. During an automated crossfade (scene change, Auto DJ, Next) the
  // server sends where the fader started, where it's going and for how
  // long; the fader is animated here from that. Grabbing it cancels the
  // automation on the server, so it never fights the user.
  import { onMount } from 'svelte';
  import { djCmd } from '$lib/dj/control';
  import { crossfaderNow, transitionLeft, DECK_NAMES } from '$lib/dj/live';
  import { addToast } from '$lib/stores/toast';
  import type { DjState } from '$lib/types';
  import Fader from './Fader.svelte';
  import Meter from './Meter.svelte';

  let { st, live, strip = false }: { st: DjState | null; live: boolean; strip?: boolean } = $props();

  // Animated crossfader position (rAF only while a transition runs).
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
    const what = { scene: 'Scene change', auto: 'Auto DJ', next: 'Next', track: 'Crossfade' }[tr.kind] ?? 'Crossfade';
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
  const xfLabel = (v: number) => (v < 0.02 ? 'A' : v > 0.98 ? 'B' : v > 0.48 && v < 0.52 ? 'Centre' : `${Math.round((1 - v) * 100)}/${Math.round(v * 100)}`);
</script>

<section class="mixer card" class:strip class:stale={!live} aria-label="Mixer">
  <div class="xf" class:auto={!!tr}>
    <div class="xf-head">
      <span class="tiny subtle">Crossfader</span>
      <span class="tiny" class:moving={!!tr}>{trLabel || xfLabel(xf)}</span>
    </div>
    <div class="xf-row">
      <span class="end a" aria-hidden="true">A</span>
      <Fader value={xf} cmd="mix.crossfader" args={(v) => ({ value: v })} label="Crossfader, A to B"
        disabled={!live} showValue={false} big color="var(--text-2)" format={xfLabel} />
      <span class="end b" aria-hidden="true">B</span>
    </div>
  </div>

  <div class="master">
    <span class="tiny subtle">Master</span>
    <div class="master-row">
      <Fader value={st?.volume ?? 0.5} cmd="mix.master" args={(v) => ({ value: v })} label="Master volume"
        disabled={!live} format={pct} />
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
      title="Crossfade to the next track now">
      Next
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 5v14l9-7zM16 5h2v14h-2z" /></svg>
    </button>
  </div>
</section>

<style>
  .mixer { padding: 12px 14px; display: flex; flex-direction: column; gap: 14px; min-width: 0; justify-content: center; }
  .stale { opacity: 0.6; }
  .xf-head { display: flex; justify-content: space-between; gap: 8px; }
  .xf-head .moving { color: var(--warning); font-weight: 600; }
  .xf-row { display: flex; align-items: center; gap: 8px; }
  .xf-row :global(.fader) { flex: 1; }
  .end { width: 24px; height: 24px; border-radius: 6px; display: grid; place-items: center; font-weight: 800; font-size: 12px; color: #111; flex-shrink: 0; }
  .end.a { background: var(--deck-a); }
  .end.b { background: var(--deck-b); }
  .master { display: flex; flex-direction: column; gap: 4px; }
  .master-row { display: flex; flex-direction: column; gap: 6px; }
  .actions { display: flex; gap: 8px; flex-wrap: wrap; }
  .actions .btn { flex: 1; min-height: 44px; justify-content: center; gap: 8px; display: inline-flex; align-items: center; }
  .actions svg { width: 16px; height: 16px; fill: currentColor; }
  .toggle .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--text-3); }
  .toggle.on { border-color: var(--success); color: var(--success); }
  .toggle.on .dot { background: var(--success); }

  /* One row for Performance mode: master | crossfader | buttons. */
  .strip { flex-direction: row; align-items: center; gap: 18px; flex-wrap: wrap; padding: 6px 14px; }
  /* The status bar has the master fader; don't spend tablet height on two. */
  .strip .master { display: none; }
  .strip .xf { flex: 2 1 320px; }
  .strip .master { flex: 1 1 200px; }
  .strip .actions { flex: 0 1 auto; }
  .strip .actions .btn { min-height: 52px; min-width: 110px; }
</style>
