<script lang="ts">
  // A deck's eight hotcue buttons and its goto toggles.
  //
  // Set slot: jump there (server seek; a paused deck stays paused).
  // Empty slot: set it at the deck's current position (snapped to a beat
  // if snapping is on). Never a seek.
  // Goto chip: arm / disarm. Right click or long press on any of them opens
  // the cue menu (edit, rename, delete).
  import { djState } from '$lib/stores/dj';
  import { deckPosition, DECK_NAMES } from '$lib/dj/live';
  import {
    HOTCUE_COLORS, GOTO_COLOR, hotcueColor, fmtCue, jumpHotcue, setHotcue, toggleGoto, gotoName,
    cueMenu, cueEdit, snapMs, loadAnalysis, gridVersion, longPress,
  } from '$lib/dj/cues';
  import type { DjAnalysis, DjState } from '$lib/types';

  let { st, deck, live, compact = false }: { st: DjState | null; deck: 0 | 1; live: boolean; compact?: boolean } = $props();

  const d = $derived(st?.decks?.[deck] ?? null);
  const track = $derived(d?.track ?? null);
  const hotcues = $derived(d?.hotcues ?? []);
  const gotos = $derived(d?.gotos ?? []);
  const lastJump = $derived(d?.lastJump);
  const usable = $derived(live && !!track && !d?.loading);

  let analysis = $state<DjAnalysis | null>(null);
  $effect(() => {
    void $gridVersion;
    const id = track?.id;
    analysis = null;
    if (!id) return;
    let alive = true;
    loadAnalysis(id).then((a) => alive && (analysis = a), () => {});
    return () => (alive = false);
  });

  // Flash the button of the cue just jumped to.
  let flash = $state<string>('');
  let flashAt = 0;
  $effect(() => {
    const j = lastJump;
    if (!j || j.at === flashAt) return;
    flashAt = j.at;
    flash = j.kind === 'goto' ? `g${j.gotoId}` : `h${j.slot}`;
    const t = setTimeout(() => (flash = ''), 600);
    return () => clearTimeout(t);
  });

  function press(slot: number, e: MouseEvent) {
    if (lp.fired()) return;
    if (!usable || !track) return;
    const h = hotcues.find((x) => x.slot === slot);
    if (h) jumpHotcue(deck, slot);
    else setHotcue(track.id, slot, snapMs(deckPosition($djState, deck), analysis, e.shiftKey));
  }

  function menuAt(e: { clientX: number; clientY: number }, opts: { hotcue?: number; gotoId?: number }) {
    if (!track) return;
    const at = opts.hotcue
      ? (hotcues.find((h) => h.slot === opts.hotcue)?.positionMs ?? deckPosition($djState, deck))
      : opts.gotoId
        ? (gotos.find((g) => g.id === opts.gotoId)?.positionMs ?? 0)
        : deckPosition($djState, deck);
    cueMenu.set({ deck, x: e.clientX, y: e.clientY, atMs: at, ...opts });
  }
  function onContext(e: MouseEvent, opts: { hotcue?: number; gotoId?: number }) {
    e.preventDefault();
    if (opts.hotcue && !hotcues.some((h) => h.slot === opts.hotcue)) opts = {};
    menuAt(e, opts);
  }

  let pending: { hotcue?: number; gotoId?: number } = {};
  const lp = longPress((e) => menuAt(e, pending));
  const down = (e: PointerEvent, opts: { hotcue?: number; gotoId?: number }) => {
    pending = opts.hotcue && !hotcues.some((h) => h.slot === opts.hotcue) ? {} : opts;
    lp.down(e);
  };
</script>

<div class="cues" class:compact role="group" aria-label="Deck {DECK_NAMES[deck]} hotcues and gotos">
  <div class="hot">
    <span class="lbl">Hotcues</span>
    <div class="slots">
      {#each HOTCUE_COLORS as c, i (i)}
        {@const slot = i + 1}
        {@const h = hotcues.find((x) => x.slot === slot)}
        <button class="hc" class:set={!!h} class:flash={flash === `h${slot}`} style:--c={h ? hotcueColor(h) : c}
          disabled={!usable}
          aria-label={h ? `Hotcue ${slot}${h.label ? `, ${h.label}` : ''}, ${fmtCue(h.positionMs)}. Jump` : `Hotcue ${slot}, empty. Set at the current position`}
          title={h ? `HC${slot}${h.label ? ` · ${h.label}` : ''} · ${fmtCue(h.positionMs)}\nClick: jump · Right click: edit` : `Empty: click to set here${analysis?.usable ? ' (snapped to the beat; Shift for exact)' : ''}`}
          onclick={(e) => press(slot, e)}
          oncontextmenu={(e) => onContext(e, { hotcue: slot })}
          onpointerdown={(e) => down(e, { hotcue: slot })} onpointermove={lp.move} onpointerup={lp.cancel} onpointercancel={lp.cancel}
        >{slot}</button>
      {/each}
    </div>
  </div>
  <div class="go">
    <span class="lbl">Goto</span>
    <div class="gotos">
      {#each gotos as g (g.id)}
        <button class="gt" class:on={g.enabled && !g.problem} class:bad={!!g.problem} class:flash={flash === `g${g.id}`}
          style:--c={GOTO_COLOR} disabled={!usable || (!g.enabled && !!g.problem)} aria-pressed={g.enabled}
          aria-label="{gotoName(gotos, g)} to hotcue {g.targetSlot} at {fmtCue(g.positionMs)}, {g.problem ?? (g.enabled ? 'armed' : 'off')}. Toggle"
          title="{gotoName(gotos, g)}{g.label ? ` · ${g.label}` : ''}: at {fmtCue(g.positionMs)} jump to HC{g.targetSlot}\n{g.problem ? `Can't fire: ${g.problem}` : g.enabled ? 'Armed: click to play through' : 'Off: click to arm'}"
          onclick={() => { if (!lp.fired() && track) toggleGoto(track.id, g); }}
          oncontextmenu={(e) => onContext(e, { gotoId: g.id })}
          onpointerdown={(e) => down(e, { gotoId: g.id })} onpointermove={lp.move} onpointerup={lp.cancel} onpointercancel={lp.cancel}
        >
          <span class="dot" aria-hidden="true"></span>{gotoName(gotos, g)}<span class="arrow">→{g.targetSlot}</span>
        </button>
      {/each}
      <button class="gt add" disabled={!usable || !hotcues.length}
        title={hotcues.length ? 'Add a goto at the current position' : 'Set a hotcue first: a goto jumps to one'}
        onclick={() => cueEdit.set({ deck, kind: 'goto', atMs: snapMs(deckPosition($djState, deck), analysis) })}>+</button>
    </div>
  </div>
</div>

<style>
  .cues { display: flex; gap: 10px 14px; flex-wrap: wrap; align-items: flex-end; min-width: 0; }
  /* Side by side when they fit (most decks), gotos wrap under otherwise. */
  .cues { flex-wrap: nowrap; }
  .hot { flex: 1 1 220px; min-width: 0; }
  .go { flex: 0 1 auto; min-width: 70px; max-width: 45%; }
  .lbl { display: block; font-size: 11px; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; color: var(--text-3); margin-bottom: 4px; }
  .slots { display: grid; grid-template-columns: repeat(8, minmax(24px, 1fr)); gap: 4px; }
  .hc {
    min-height: 30px; padding: 0; border-radius: 6px; font-weight: 800; font-size: 13px;
    background: var(--surface-sunken); color: var(--text-3);
    border: 1px dashed color-mix(in srgb, var(--c) 45%, var(--border-strong));
  }
  .hc.set { background: color-mix(in srgb, var(--c) 24%, var(--surface-raised)); color: var(--text); border: 1px solid var(--c); }
  .hc.set:hover:not(:disabled) { background: color-mix(in srgb, var(--c) 40%, var(--surface-raised)); }
  .hc:not(.set):hover:not(:disabled) { color: var(--text-2); }
  .hc:disabled { opacity: 0.45; }
  .flash { animation: flash 0.6s ease-out; }
  @keyframes flash { 0% { box-shadow: 0 0 0 3px var(--c); } 100% { box-shadow: 0 0 0 0 transparent; } }
  @media (prefers-reduced-motion: reduce) { .flash { animation: none; outline: 2px solid var(--c); } }
  .gotos { display: flex; gap: 4px; flex-wrap: wrap; }
  .gt {
    min-height: 30px; padding: 0 9px; border-radius: 6px; font-weight: 700; font-size: 13px;
    display: inline-flex; align-items: center; gap: 5px;
    background: var(--surface-sunken); color: var(--text-3); border: 1px dashed var(--border-strong);
  }
  .gt .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--text-3); }
  .gt .arrow { font-weight: 600; opacity: 0.8; }
  .gt.on { color: var(--c); border: 1px solid var(--c); background: color-mix(in srgb, var(--c) 14%, var(--surface-raised)); }
  .gt.on .dot { background: var(--c); box-shadow: 0 0 6px var(--c); }
  .gt.bad { border-color: var(--danger, #f43f62); color: var(--danger, #f43f62); }
  .gt.add { padding: 0 11px; }
  .compact .slots { grid-template-columns: repeat(8, minmax(32px, 1fr)); }
  .compact .hc, .compact .gt { min-height: 40px; }
  /* Narrow deck: two rows of four. */
  @container deck (max-width: 240px) {
    .slots { grid-template-columns: repeat(4, 1fr); }
  }
  @container deck (max-width: 305px) {
    .cues { flex-wrap: wrap; }
    .go { max-width: none; }
  }
</style>
