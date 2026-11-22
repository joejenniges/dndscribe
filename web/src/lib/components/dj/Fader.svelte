<script lang="ts">
  // A server-backed fader (gain, master, crossfader). While the user drags
  // it shows their pointer position, sends at most ~30 updates a second and
  // ignores incoming snapshots for this control; on release it sends the
  // final value and, once the server has acknowledged it, shows the server's
  // value again. Another device moving the same control simply wins the
  // next snapshot (last write wins, no locking).
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
    color?: string;
    showValue?: boolean;
    big?: boolean;
  } = $props();

  // svelte-ignore state_referenced_locally
  const sender = throttled(cmd, (v) => args(v));
  let dragVal = $state<number | null>(null);
  const shown = $derived(dragVal ?? value);

  function onInput(e: Event) {
    const v = Number((e.currentTarget as HTMLInputElement).value) / 1000;
    dragVal = v;
    sender.input(v);
  }

  async function onChange(e: Event) {
    const v = Number((e.currentTarget as HTMLInputElement).value) / 1000;
    dragVal = v;
    try {
      await sender.commit(v);
    } catch (err) {
      addToast((err as Error).message, 'error');
    } finally {
      dragVal = null;
    }
  }
</script>

<label class="fader" class:vertical class:big style:--fader-color={color}>
  <span class="sr-only">{label}</span>
  <input
    type="range"
    min="0"
    max="1000"
    step="1"
    value={Math.round(shown * 1000)}
    oninput={onInput}
    onchange={onChange}
    {disabled}
    aria-label={label}
    aria-valuetext={format(shown)}
  />
  {#if showValue}<span class="val tiny">{format(shown)}</span>{/if}
</label>

<style>
  .fader { display: flex; align-items: center; gap: 8px; min-width: 0; }
  .fader.vertical { flex-direction: column-reverse; gap: 6px; height: 100%; }
  input {
    flex: 1; min-width: 0; width: 100%; height: 28px; margin: 0;
    accent-color: var(--fader-color); cursor: pointer; touch-action: none;
  }
  .big input { height: 44px; }
  /* Native vertical range: bottom = 0. */
  .vertical input { writing-mode: vertical-lr; direction: rtl; width: 28px; height: auto; min-height: 90px; flex: 1; }
  input:disabled { cursor: not-allowed; opacity: 0.5; }
  .val { color: var(--text-2); font-variant-numeric: tabular-nums; min-width: 3.2em; text-align: right; }
  .vertical .val { text-align: center; min-width: 0; }
  .sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
</style>
