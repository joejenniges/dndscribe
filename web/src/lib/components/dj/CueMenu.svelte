<script lang="ts">
  // Context menu for cues, opened by right click (or a long press on touch)
  // on either waveform or a cue button. One instance per page; what it shows
  // comes from the cueMenu store and the live deck state.
  import { tick } from 'svelte';
  import { dj } from '$lib/api';
  import { djState } from '$lib/stores/dj';
  import { addToast } from '$lib/stores/toast';
  import { DECK_NAMES } from '$lib/dj/live';
  import {
    cueMenu, cueEdit, snapOn, snapTolMs, SNAP_TOLERANCES, snapMs, loadAnalysis, gridVersion, analysisLabel,
    fmtCue, hotcueColor, HOTCUE_COLORS, setHotcue, deleteHotcue, addGoto, deleteGoto, toggleGoto, jumpHotcue, gotoName,
  } from '$lib/dj/cues';
  import type { DjAnalysis, DjGridEdit } from '$lib/types';

  const req = $derived($cueMenu);
  const deck = $derived(req ? ($djState?.decks?.[req.deck] ?? null) : null);
  const track = $derived(deck?.track ?? null);
  const hotcues = $derived(deck?.hotcues ?? []);
  const gotos = $derived(deck?.gotos ?? []);
  const onHot = $derived(req?.hotcue ? hotcues.find((h) => h.slot === req.hotcue) : undefined);
  const onGoto = $derived(req?.gotoId ? gotos.find((g) => g.id === req.gotoId) : undefined);

  let analysis = $state<DjAnalysis | null>(null);
  $effect(() => {
    void $gridVersion;
    const id = track?.id;
    analysis = null;
    if (!id || !req) return;
    let alive = true;
    loadAnalysis(id).then((a) => alive && (analysis = a), () => {});
    return () => (alive = false);
  });

  const at = $derived(req ? snapMs(req.atMs, analysis) : 0);
  const snapped = $derived(!!req && at !== Math.round(req.atMs));
  const used = $derived(new Set(hotcues.map((h) => h.slot)));

  let el: HTMLDivElement | undefined = $state();
  let left = $state(0);
  let top = $state(0);
  $effect(() => {
    if (!req) return;
    left = req.x;
    top = req.y;
    tick().then(() => {
      if (!el) return;
      const r = el.getBoundingClientRect();
      left = Math.max(8, Math.min(req.x, window.innerWidth - r.width - 8));
      top = Math.max(8, Math.min(req.y, window.innerHeight - r.height - 8));
      el.querySelector<HTMLButtonElement>('button:not(:disabled)')?.focus();
    });
  });

  const close = () => cueMenu.set(null);
  // Run first, then close: the actions read this menu's track and time,
  // which are gone once the menu closes.
  function run(f: () => unknown) {
    f();
    close();
  }

  function onWindowDown(e: PointerEvent) {
    if (req && el && !el.contains(e.target as Node)) close();
  }
  function onKey(e: KeyboardEvent) {
    if (!req) return;
    if (e.key === 'Escape') {
      e.preventDefault();
      close();
    } else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      const items = [...(el?.querySelectorAll<HTMLButtonElement>('button:not(:disabled)') ?? [])];
      const i = items.indexOf(document.activeElement as HTMLButtonElement);
      items[(i + (e.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length]?.focus();
    }
  }

  async function grid(e: DjGridEdit) {
    if (!track) return;
    try {
      await dj.analysis.edit(track.id, e);
    } catch (err) {
      addToast(`Beat grid: ${(err as Error).message}`, 'error');
    }
  }
  async function reanalyze() {
    if (!track) return;
    try {
      await dj.analysis.reanalyze(track.id);
      addToast('Re-analysing beats', 'info');
    } catch (err) {
      addToast(`Beat analysis: ${(err as Error).message}`, 'error');
    }
  }
  function cycleTol() {
    const i = SNAP_TOLERANCES.indexOf($snapTolMs);
    snapTolMs.set(SNAP_TOLERANCES[(i + 1) % SNAP_TOLERANCES.length]);
  }
</script>

<svelte:window onpointerdown={onWindowDown} onkeydown={onKey} onresize={close} />

{#if req && track}
  <div class="cm" bind:this={el} role="menu" tabindex="-1" style:left="{left}px" style:top="{top}px"
    aria-label="Cue menu, deck {DECK_NAMES[req.deck]}" oncontextmenu={(e) => e.preventDefault()}>
    <div class="head">
      <span class="tag deck-{req.deck}">{DECK_NAMES[req.deck]}</span>
      <span class="time">{fmtCue(at)}</span>
      {#if snapped}<span class="hint">snapped to beat</span>{/if}
    </div>

    {#if onHot}
      <div class="sec">Hotcue {onHot.slot}{onHot.label ? ` · ${onHot.label}` : ''} · {fmtCue(onHot.positionMs)}</div>
      <button role="menuitem" onclick={() => run(() => jumpHotcue(req.deck, onHot.slot))}>Jump to hotcue {onHot.slot}</button>
      <button role="menuitem" onclick={() => run(() => cueEdit.set({ deck: req.deck, kind: 'hotcue', slot: onHot.slot }))}>Edit / rename…</button>
      <button role="menuitem" onclick={() => run(() => setHotcue(track.id, onHot.slot, at, onHot.label, onHot.color))}>Move here ({fmtCue(at)})</button>
      <button role="menuitem" class="danger" onclick={() => run(() => deleteHotcue(track.id, onHot.slot, gotos))}>Delete hotcue {onHot.slot}</button>
    {:else if onGoto}
      <div class="sec">{gotoName(gotos, onGoto)} → HC{onGoto.targetSlot} · {fmtCue(onGoto.positionMs)}{onGoto.problem ? ` · ${onGoto.problem}` : ''}</div>
      <button role="menuitem" disabled={!onGoto.enabled && !!onGoto.problem} onclick={() => run(() => toggleGoto(track.id, onGoto))}>
        {onGoto.enabled ? 'Disarm (play through)' : 'Arm'}
      </button>
      <button role="menuitem" onclick={() => run(() => cueEdit.set({ deck: req.deck, kind: 'goto', gotoId: onGoto.id }))}>Edit / rename…</button>
      <button role="menuitem" class="danger" onclick={() => run(() => deleteGoto(track.id, onGoto.id))}>Delete {gotoName(gotos, onGoto)}</button>
    {:else}
      <div class="sec">Add hotcue</div>
      <div class="slots">
        {#each HOTCUE_COLORS as c, i (i)}
          {@const slot = i + 1}
          <button role="menuitem" class="slot" style:--c={c} disabled={used.has(slot)}
            title={used.has(slot) ? `Hotcue ${slot} is set` : `Add hotcue ${slot} at ${fmtCue(at)}`}
            onclick={() => run(() => setHotcue(track.id, slot, at))}>{slot}</button>
        {/each}
      </div>
      <div class="sec">Add goto here, jumping to</div>
      {#if hotcues.length}
        <div class="slots">
          {#each hotcues as h (h.slot)}
            <button role="menuitem" class="slot goto" style:--c={hotcueColor(h)}
              title="Goto at {fmtCue(at)} jumping to hotcue {h.slot} ({fmtCue(h.positionMs)})"
              disabled={Math.abs(h.positionMs - at) < 500}
              onclick={() => run(() => addGoto(track.id, at, h.slot))}>{h.slot}</button>
          {/each}
        </div>
      {:else}
        <p class="note">Set a hotcue first: a goto jumps to one.</p>
      {/if}
    {/if}

    <div class="sec">Snap</div>
    <button role="menuitemcheckbox" aria-checked={$snapOn} onclick={() => snapOn.set(!$snapOn)}>
      <span class="check">{$snapOn ? '✓' : ''}</span>Snap to beat
    </button>
    <button role="menuitem" onclick={cycleTol} title="Only snap when a beat is this close">
      <span class="check"></span>Snap within {$snapTolMs} ms
    </button>

    <div class="sec">Beat grid · {analysisLabel(analysis) || '…'}</div>
    {#if analysis?.status === 'ready' && analysis.beats.length}
      <div class="row">
        <button role="menuitem" onclick={() => run(() => grid({ op: 'half' }))} title="Keep every other beat">½ tempo</button>
        <button role="menuitem" onclick={() => run(() => grid({ op: 'double' }))} title="Add a beat between each pair">×2 tempo</button>
        <button role="menuitem" onclick={() => run(() => grid({ op: 'shift', shiftMs: -10 }))} title="Move the grid 10 ms earlier">−10 ms</button>
        <button role="menuitem" onclick={() => run(() => grid({ op: 'shift', shiftMs: 10 }))} title="Move the grid 10 ms later">+10 ms</button>
      </div>
    {/if}
    <div class="row">
      <button role="menuitem" onclick={() => run(() => cueEdit.set({ deck: req.deck, kind: 'grid', atMs: at }))}
        title="Set a constant grid by BPM, starting at this point">Set grid here…</button>
      <button role="menuitem" onclick={() => run(reanalyze)}>Re-analyse</button>
      {#if analysis?.status === 'ready'}
        <button role="menuitem" onclick={() => run(() => grid({ op: 'off' }))} title="No beat ticks or snapping for this track">Grid off</button>
      {/if}
    </div>
  </div>
{/if}

<style>
  .cm {
    position: fixed; z-index: 200; min-width: 236px; max-width: 300px; padding: 6px;
    background: var(--surface-raised); border: 1px solid var(--border-strong); border-radius: var(--radius);
    box-shadow: var(--shadow); display: flex; flex-direction: column; gap: 1px;
  }
  .head { display: flex; align-items: center; gap: 8px; padding: 4px 6px 6px; border-bottom: 1px solid var(--border-color); margin-bottom: 2px; }
  .tag { width: 20px; height: 20px; border-radius: 5px; display: grid; place-items: center; font-weight: 800; font-size: 12px; color: #111; background: var(--deck-a); }
  .tag.deck-1 { background: var(--deck-b); }
  .time { font-variant-numeric: tabular-nums; font-weight: 700; }
  .hint { font-size: 11px; color: var(--text-3); }
  .sec { font-size: 11px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; color: var(--text-3); padding: 8px 6px 3px; }
  button {
    display: flex; align-items: center; gap: 8px; text-align: left; padding: 7px 8px; min-height: 34px;
    border-radius: 6px; background: none; border: 0; color: var(--text); font-size: 13px; cursor: pointer;
  }
  button:hover:not(:disabled), button:focus-visible { background: var(--surface-hover); outline: none; }
  button:disabled { color: var(--text-3); cursor: default; }
  .danger { color: var(--danger, #f43f62); }
  .check { width: 14px; display: inline-block; color: var(--success); }
  .slots { display: grid; grid-template-columns: repeat(8, 1fr); gap: 4px; padding: 2px 4px; }
  .slot {
    justify-content: center; padding: 0; min-height: 30px; font-weight: 800;
    background: color-mix(in srgb, var(--c) 22%, var(--surface)); border: 1px solid var(--c); color: var(--text);
  }
  .slot:hover:not(:disabled) { background: color-mix(in srgb, var(--c) 40%, var(--surface)); }
  .slot:disabled { opacity: 0.3; }
  .slot.goto { border-style: dashed; }
  .row { display: flex; flex-wrap: wrap; gap: 2px; }
  .row button { flex: 1 1 auto; justify-content: center; }
  .note { margin: 0; padding: 4px 8px; font-size: 12px; color: var(--text-3); }
</style>
