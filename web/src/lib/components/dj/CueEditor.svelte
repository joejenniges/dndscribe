<script lang="ts">
  // Edit dialog for a hotcue, a goto trigger, or a manual beat grid. Also the
  // keyboard way to place cues exactly: times are typed as m:ss.mmm.
  //
  // Rule shown to the user: deleting a hotcue turns off any goto aimed at
  // it ("target missing") rather than deleting or retargeting the goto.
  import Modal from '$lib/components/ui/Modal.svelte';
  import { dj } from '$lib/api';
  import { djState } from '$lib/stores/dj';
  import { addToast } from '$lib/stores/toast';
  import { DECK_NAMES, deckPosition } from '$lib/dj/live';
  import {
    cueEdit, fmtCue, parseCue, snapMs, loadAnalysis, hotcueColor, HOTCUE_COLORS,
    setHotcue, deleteHotcue, addGoto, saveGoto, deleteGoto, gotoName,
  } from '$lib/dj/cues';
  import type { DjAnalysis } from '$lib/types';

  const req = $derived($cueEdit);
  let open = $state(false);
  $effect(() => {
    open = !!req;
  });
  $effect(() => {
    if (!open && $cueEdit) cueEdit.set(null);
  });

  const deck = $derived(req ? ($djState?.decks?.[req.deck] ?? null) : null);
  const track = $derived(deck?.track ?? null);
  const hotcues = $derived(deck?.hotcues ?? []);
  const gotos = $derived(deck?.gotos ?? []);

  let analysis = $state<DjAnalysis | null>(null);
  let time = $state('');
  let label = $state('');
  let color = $state('');
  let target = $state(1);
  let enabled = $state(true);
  let bpm = $state('');
  let busy = $state(false);

  // Fill the form when a request arrives (only then: the deck state ticks on).
  let filledFor: object | null = null;
  $effect(() => {
    const r = req;
    if (!r || r === filledFor) return;
    filledFor = r;
    const d = $djState?.decks?.[r.deck];
    const now = r.atMs ?? deckPosition($djState, r.deck);
    analysis = null;
    if (d?.track) loadAnalysis(d.track.id).then((a) => (analysis = a), () => {});
    if (r.kind === 'hotcue') {
      const h = d?.hotcues.find((x) => x.slot === r.slot);
      time = fmtCue(h?.positionMs ?? now);
      label = h?.label ?? '';
      color = h?.color ?? '';
    } else if (r.kind === 'goto') {
      const g = r.gotoId ? d?.gotos.find((x) => x.id === r.gotoId) : undefined;
      time = fmtCue(g?.positionMs ?? now);
      label = g?.label ?? '';
      target = g?.targetSlot ?? d?.hotcues[0]?.slot ?? 1;
      enabled = g?.enabled ?? true;
    } else {
      time = fmtCue(now);
      bpm = '120'; // replaced by the detected BPM once the grid loads (below)
    }
  });
  $effect(() => {
    if (req?.kind === 'grid' && analysis?.bpm && bpm === '120') bpm = String(analysis.bpm);
  });

  const ms = $derived(parseCue(time));
  const dur = $derived(track?.durationMs ?? 0);
  const timeError = $derived(ms === null ? 'Use m:ss.mmm, e.g. 1:02.500' : dur && ms > dur ? 'Past the end of the track' : '');
  const existingGoto = $derived(req?.kind === 'goto' && req.gotoId ? gotos.find((g) => g.id === req.gotoId) : undefined);
  const targetHot = $derived(hotcues.find((h) => h.slot === target));
  const gotoError = $derived.by(() => {
    if (req?.kind !== 'goto' || ms === null) return '';
    if (!targetHot) return `Hotcue ${target} isn't set: the goto would have nothing to jump to`;
    if (Math.abs(targetHot.positionMs - ms) < 500) return 'A goto must be at least 0.5 s from its target';
    return '';
  });
  const title = $derived(
    !req ? '' : req.kind === 'hotcue' ? `Hotcue ${req.slot}` : req.kind === 'grid' ? 'Beat grid' : existingGoto ? `Goto ${gotoName(gotos, existingGoto)}` : 'New goto',
  );

  function snapNow() {
    if (ms !== null) time = fmtCue(snapMs(ms, analysis));
  }

  async function save() {
    if (!req || !track || ms === null || timeError || gotoError) return;
    busy = true;
    try {
      if (req.kind === 'hotcue') {
        await setHotcue(track.id, req.slot, ms, label.trim(), color);
      } else if (req.kind === 'goto') {
        if (existingGoto) await saveGoto(track.id, { ...existingGoto, positionMs: ms, targetSlot: target, enabled, label: label.trim() });
        else await addGoto(track.id, ms, target);
      } else {
        const b = Number(bpm);
        if (!(b >= 30 && b <= 300)) {
          addToast('BPM must be 30-300', 'error');
          return;
        }
        await dj.analysis.edit(track.id, { op: 'grid', bpm: b, firstBeatMs: ms });
      }
      open = false;
    } catch (e) {
      addToast((e as Error).message, 'error');
    } finally {
      busy = false;
    }
  }

  async function remove() {
    if (!req || !track) return;
    if (req.kind === 'hotcue') await deleteHotcue(track.id, req.slot, gotos);
    else if (req.kind === 'goto' && existingGoto) await deleteGoto(track.id, existingGoto.id);
    open = false;
  }
</script>

<Modal bind:open {title} width="420px">
  {#if req && track}
    <p class="tiny subtle ctx">Deck {DECK_NAMES[req.deck]} · {track.title}</p>
    <form class="form" onsubmit={(e) => { e.preventDefault(); save(); }}>
      <label class="field">
        <span>{req.kind === 'goto' ? 'Trigger time' : req.kind === 'grid' ? 'A beat falls at' : 'Time'}</span>
        <div class="trow">
          <input class="input mono" bind:value={time} inputmode="decimal" aria-invalid={!!timeError} />
          {#if req.kind !== 'grid'}
            <button type="button" class="btn btn-sm" onclick={snapNow} disabled={!analysis?.usable}
              title={analysis?.usable ? 'Move to the nearest beat' : 'No usable beat grid for this track'}>Snap</button>
          {/if}
        </div>
        {#if timeError}<small class="err">{timeError}</small>{/if}
      </label>

      {#if req.kind === 'grid'}
        <label class="field"><span>BPM</span>
          <input class="input mono" bind:value={bpm} inputmode="decimal" />
        </label>
        <p class="tiny subtle">Replaces the detected grid with a steady one at this tempo through the beat above. Re-analyse from the cue menu to go back to detection.</p>
      {:else}
        <label class="field"><span>Name</span>
          <input class="input" bind:value={label} maxlength="60" placeholder={req.kind === 'goto' ? 'e.g. Loop the chorus' : 'e.g. Drop'} />
        </label>
      {/if}

      {#if req.kind === 'hotcue'}
        <div class="field"><span>Colour</span>
          <div class="colors">
            {#each HOTCUE_COLORS as c, i (c)}
              <button type="button" class="sw" style:--c={c} aria-label="Colour {i + 1}" aria-pressed={(color || HOTCUE_COLORS[req.slot - 1]) === c}
                onclick={() => (color = c === HOTCUE_COLORS[req.slot - 1] ? '' : c)}></button>
            {/each}
          </div>
        </div>
      {:else if req.kind === 'goto'}
        <label class="field"><span>Jumps to</span>
          <select class="select" bind:value={target}>
            {#each [1, 2, 3, 4, 5, 6, 7, 8] as s (s)}
              {@const h = hotcues.find((x) => x.slot === s)}
              <option value={s} disabled={!h}>HC{s}{h ? ` · ${fmtCue(h.positionMs)}${h.label ? ` · ${h.label}` : ''}` : ' (not set)'}</option>
            {/each}
          </select>
        </label>
        {#if existingGoto}
          <label class="check"><input type="checkbox" bind:checked={enabled} /> Armed (jump when playback reaches it)</label>
        {/if}
        {#if gotoError}<small class="err">{gotoError}</small>{/if}
        <p class="tiny subtle">When armed and playback crosses the trigger, the server jumps this deck to the hotcue, every time round. Seeking past it doesn't trigger it. If the target hotcue is deleted, the goto is turned off and shows "target missing" until it's pointed at another hotcue.</p>
      {/if}
    </form>
  {/if}
  {#snippet footer()}
    {#if req && ((req.kind === 'hotcue' && hotcues.some((h) => h.slot === req.slot)) || existingGoto)}
      <button class="btn-ghost danger" onclick={remove} disabled={busy}>Delete</button>
    {/if}
    <span class="spacer"></span>
    <button class="btn" onclick={() => (open = false)}>Cancel</button>
    <button class="btn-primary" onclick={save} disabled={busy || !!timeError || !!gotoError || !track}>Save</button>
  {/snippet}
</Modal>

<style>
  .ctx { margin: 0 0 10px; }
  .form { display: flex; flex-direction: column; gap: 12px; }
  .field { display: flex; flex-direction: column; gap: 4px; font-size: 13px; font-weight: 600; }
  .trow { display: flex; gap: 6px; }
  .trow .input { flex: 1; }
  .mono { font-variant-numeric: tabular-nums; font-family: var(--font-mono); }
  .err { color: var(--danger, #f43f62); font-weight: 500; }
  .colors { display: flex; gap: 6px; flex-wrap: wrap; }
  .sw { width: 28px; height: 28px; border-radius: 6px; background: var(--c); border: 2px solid transparent; padding: 0; }
  .sw[aria-pressed='true'] { border-color: var(--text); }
  .check { display: flex; gap: 8px; align-items: center; font-size: 13px; }
  .danger { color: var(--danger, #f43f62); }
  .spacer { flex: 1; }
</style>
