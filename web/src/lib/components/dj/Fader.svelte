<script lang="ts">
  // A server-backed fader drawn by hand (not a native range input) so the
  // crossfader, channel gains, master and soundboard volume all look and
  // behave the same on desktop and touch.
  //
  // Authority: while the user drags (or steps with the keyboard) it shows
  // their value, sends at most ~30 updates a second and ignores incoming
  // snapshots for this control; on release it sends the final value and,
  // once the server has acknowledged it, shows the server's value again.
  // Another device moving the same control wins the next snapshot (last
  // write wins).
  //
  // Pointer: grabbing the thumb drags it relative to where it was grabbed,
  // so it never jumps. Pressing the track elsewhere jumps there for gain
  // faders, but not for the crossfader: a stray tap on the tablet must not
  // slam the mix across.
  import { throttled } from '$lib/dj/control';
  import { addToast } from '$lib/stores/toast';

  let {
    value,
    cmd,
    args,
    label,
    vertical = false,
    disabled = false,
    format = (v: number) => `${Math.round(v * 100)}%`,
    color = 'var(--accent)',
    showValue = true,
    big = false,
    crossfader = false,
    ticks = 11,
  }: {
    /** Server value, 0..1. */
    value: number;
    cmd: string;
    /** Builds the command args from a 0..1 value. */
    args: (v: number) => Record<string, unknown>;
    label: string;
    vertical?: boolean;
    disabled?: boolean;
    format?: (v: number) => string;
    /** Fill and thumb accent. Ignored by the crossfader (no fill). */
    color?: string;
    showValue?: boolean;
    big?: boolean;
    /** Balanced track (no fill), centre marker, rectangular thumb. */
    crossfader?: boolean;
    ticks?: number;
  } = $props();

  // svelte-ignore state_referenced_locally
  const sender = throttled(cmd, (v) => args(v));
  let local = $state<number | null>(null);
  const shown = $derived(local ?? value);

  let trackEl: HTMLDivElement | undefined = $state();
  let drag: { id: number; offset: number } | null = null;

  const clamp = (v: number) => Math.max(0, Math.min(1, v));

  function valueAt(e: PointerEvent): number {
    const r = trackEl!.getBoundingClientRect();
    return vertical ? clamp(1 - (e.clientY - r.top) / r.height) : clamp((e.clientX - r.left) / r.width);
  }

  function down(e: PointerEvent) {
    if (disabled || e.button !== 0 || !trackEl) return;
    const onThumb = (e.target as Element).closest('.thumb');
    const at = valueAt(e);
    if (!onThumb && crossfader) return;
    e.preventDefault();
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    (e.currentTarget as HTMLElement).focus();
    drag = { id: e.pointerId, offset: onThumb ? shown - at : 0 };
    set(onThumb ? shown : at);
  }
  function move(e: PointerEvent) {
    if (!drag || e.pointerId !== drag.id) return;
    set(clamp(valueAt(e) + drag.offset));
  }
  function up(e: PointerEvent) {
    if (!drag || e.pointerId !== drag.id) return;
    drag = null;
    if (local !== null) commit(local);
  }

  function set(v: number) {
    v = Math.round(v * 1000) / 1000;
    local = v;
    sender.input(v);
  }

  let commitTimer: ReturnType<typeof setTimeout> | undefined;
  async function commit(v: number) {
    clearTimeout(commitTimer);
    try {
      await sender.commit(v);
    } catch (err) {
      addToast((err as Error).message, 'error');
    } finally {
      // A newer drag may have started meanwhile; leave its value alone.
      if (!drag) local = null;
    }
  }

  function key(e: KeyboardEvent) {
    if (disabled) return;
    const inc = vertical ? ['ArrowUp', 'ArrowRight'] : ['ArrowRight', 'ArrowUp'];
    const dec = vertical ? ['ArrowDown', 'ArrowLeft'] : ['ArrowLeft', 'ArrowDown'];
    const step = e.shiftKey ? 0.1 : 0.01;
    let v: number | null = null;
    if (inc.includes(e.key)) v = shown + step;
    else if (dec.includes(e.key)) v = shown - step;
    else if (e.key === 'PageUp') v = shown + 0.1;
    else if (e.key === 'PageDown') v = shown - 0.1;
    else if (e.key === 'Home') v = 0;
    else if (e.key === 'End') v = 1;
    else if (crossfader && (e.key === 'c' || e.key === 'C')) v = 0.5;
    if (v === null) return;
    e.preventDefault();
    e.stopPropagation();
    set(clamp(v));
    // Keyboard steps commit after a short pause, like a release.
    clearTimeout(commitTimer);
    commitTimer = setTimeout(() => local !== null && commit(local), 250);
  }

  const tickList = $derived(Array.from({ length: ticks }, (_, i) => i / (ticks - 1)));
</script>

<div class="fader" class:vertical class:big class:xf={crossfader} class:off={disabled} style:--fader-color={color}
  style:--v={shown}>
  <div
    class="track-hit"
    bind:this={trackEl}
    role="slider"
    tabindex={disabled ? -1 : 0}
    aria-label={label}
    aria-orientation={vertical ? 'vertical' : 'horizontal'}
    aria-valuemin={0}
    aria-valuemax={100}
    aria-valuenow={Math.round(shown * 100)}
    aria-valuetext={format(shown)}
    aria-disabled={disabled}
    onpointerdown={down}
    onpointermove={move}
    onpointerup={up}
    onpointercancel={up}
    onkeydown={key}
  >
    <div class="ticks" aria-hidden="true">
      {#each tickList as t, i (i)}
        <span class="tick" class:major={i === 0 || i === ticks - 1 || (crossfader && Math.abs(t - 0.5) < 1e-6)}
          style:--t={t}></span>
      {/each}
    </div>
    <div class="rail" aria-hidden="true">
      {#if !crossfader}<div class="fill"></div>{/if}
      {#if crossfader}<div class="centre"></div>{/if}
    </div>
    <div class="thumb" aria-hidden="true"><span class="grip"></span></div>
  </div>
  {#if showValue}<span class="val">{format(shown)}</span>{/if}
</div>

<style>
  .fader {
    --rail: 6px;
    --thumb-w: 16px;
    --thumb-h: 26px;
    --hit: 34px;
    display: flex; align-items: center; gap: 10px; min-width: 0;
  }
  .big { --hit: 48px; --thumb-h: 38px; --thumb-w: 20px; }
  .xf { --thumb-w: 22px; --thumb-h: 40px; --hit: 48px; }
  .xf.big { --thumb-w: 26px; --thumb-h: 46px; --hit: 56px; }
  .vertical { flex-direction: column-reverse; gap: 6px; height: 100%; --thumb-w: 30px; --thumb-h: 14px; }

  .track-hit {
    position: relative; flex: 1; min-width: 0; height: var(--hit);
    cursor: pointer; touch-action: none; user-select: none; outline: none; border-radius: 6px;
  }
  .vertical .track-hit { width: var(--hit); height: auto; min-height: 90px; flex: 1; }
  .xf .track-hit { cursor: default; }
  .track-hit:focus-visible { box-shadow: var(--focus-ring); }
  .off .track-hit { cursor: not-allowed; opacity: 0.5; }

  /* Rail: the same colour along its whole length; gain faders add a fill
     from the bottom/left, the crossfader never does. */
  .rail {
    position: absolute; left: calc(var(--thumb-w) / 2); right: calc(var(--thumb-w) / 2);
    top: 50%; height: var(--rail); transform: translateY(-50%);
    border-radius: 99px; background: var(--surface-sunken);
    box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.6), 0 0 0 1px var(--border-color);
  }
  .vertical .rail { left: 50%; right: auto; top: calc(var(--thumb-h) / 2); bottom: calc(var(--thumb-h) / 2); height: auto; width: var(--rail); transform: translateX(-50%); }
  .fill { position: absolute; left: 0; top: 0; bottom: 0; width: calc(var(--v) * 100%); border-radius: inherit; background: var(--fader-color); opacity: 0.85; }
  .vertical .fill { top: auto; right: 0; width: auto; height: calc(var(--v) * 100%); }
  .centre { position: absolute; left: 50%; top: -9px; bottom: -9px; width: 2px; transform: translateX(-50%); background: var(--text-3); border-radius: 1px; }

  .ticks { position: absolute; inset: 0 calc(var(--thumb-w) / 2); pointer-events: none; }
  .vertical .ticks { inset: calc(var(--thumb-h) / 2) 0; }
  .tick { position: absolute; left: calc(var(--t) * 100%); top: 3px; width: 1px; height: 6px; background: var(--border-strong); transform: translateX(-50%); }
  .tick::after { content: ''; position: absolute; left: 0; top: calc(var(--hit) - 15px); width: 1px; height: 6px; background: inherit; }
  .tick.major { background: var(--text-3); height: 8px; }
  .vertical .tick { left: 3px; top: auto; bottom: calc(var(--t) * 100%); width: 6px; height: 1px; transform: translateY(50%); }
  .vertical .tick::after { left: calc(var(--hit) - 15px); top: 0; width: 6px; height: 1px; }
  .vertical .tick.major { width: 8px; height: 1px; }

  /* Thumb: a fader cap. Rectangular with a centre line, like the real thing. */
  .thumb {
    position: absolute; top: 50%; left: calc(var(--v) * (100% - var(--thumb-w)));
    width: var(--thumb-w); height: var(--thumb-h); transform: translateY(-50%);
    border-radius: 4px; cursor: grab;
    background: linear-gradient(180deg, #e9e9ee, #a9abb3 48%, #8d9099 52%, #c9cad0);
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.6), inset 0 0 0 1px rgba(255, 255, 255, 0.35);
    display: grid; place-items: center;
  }
  .vertical .thumb { left: 50%; top: auto; bottom: calc(var(--v) * (100% - var(--thumb-h))); transform: translateX(-50%); }
  .thumb:active { cursor: grabbing; }
  .grip { width: 70%; height: 2px; background: var(--fader-color); border-radius: 1px; }
  .xf .grip { width: 2px; height: 70%; background: #222; }
  .fader:not(.vertical):not(.xf) .grip { width: 2px; height: 60%; }
  .track-hit:focus-visible .thumb { box-shadow: 0 0 0 2px var(--canvas), 0 0 0 4px var(--accent); }

  .val { color: var(--text-2); font-size: 13px; font-weight: 600; font-variant-numeric: tabular-nums; min-width: 3.4em; text-align: right; }
  .vertical .val { text-align: center; min-width: 0; font-size: 12px; }
</style>
