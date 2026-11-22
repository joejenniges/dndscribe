<script lang="ts">
  // Peak level meter for a deck or the master, from the server's dj_meters
  // reports (~10/s). Bars fall back smoothly between reports instead of
  // jumping to zero.
  import { untrack } from 'svelte';
  import { djMeters } from '$lib/stores/dj';

  let {
    source,
    vertical = true,
    label,
  }: { source: 0 | 1 | 'master'; vertical?: boolean; label: string } = $props();

  const raw = $derived(source === 'master' ? $djMeters.master : $djMeters.decks[source]);
  // dB scale over the top 48dB, so quiet ambience still shows movement.
  const level = $derived(raw > 0 ? Math.max(0, 1 + (20 * Math.log10(raw)) / 48) : 0);
  let shown = $state(0);
  // Once per meter report: rise to the new level at once, fall by 40%.
  // WHY untrack: reading `shown` here would make this effect depend on what
  // it writes, so it re-runs itself. With the level at 0 (a paused deck) it
  // then decays ~1400 times in one go and trips Svelte's update-depth guard,
  // which stops every update on the page.
  $effect(() => {
    $djMeters; // a new object per report, so this runs even if the level repeats
    const l = level;
    shown = untrack(() => Math.max(l, shown * 0.6));
  });
  const hot = $derived(raw > 0.89); // within ~1dB of full scale: the limiter is working
</script>

<div class="meter" class:vertical class:hot role="meter" aria-label={label} aria-valuemin={0} aria-valuemax={1}
  aria-valuenow={Math.round(shown * 100) / 100}>
  <div class="fill" style={vertical ? `height:${shown * 100}%` : `width:${shown * 100}%`}></div>
</div>

<style>
  .meter { position: relative; background: var(--surface-sunken); border-radius: 3px; overflow: hidden; }
  .meter.vertical { width: 8px; height: 100%; min-height: 60px; }
  .meter:not(.vertical) { height: 6px; width: 100%; }
  .fill {
    position: absolute; left: 0; bottom: 0; border-radius: 3px;
    background: linear-gradient(to top, var(--success), var(--warning) 80%, var(--danger));
    transition: height 0.1s linear, width 0.1s linear;
  }
  .vertical .fill { width: 100%; }
  .meter:not(.vertical) .fill { height: 100%; top: 0; background: linear-gradient(to right, var(--success), var(--warning) 80%, var(--danger)); }
  .hot { box-shadow: 0 0 0 1px var(--danger); }
  @media (prefers-reduced-motion: reduce) { .fill { transition: none; } }
</style>
