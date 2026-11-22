<script lang="ts">
  // Colour choice for scenes and pads: a fixed palette that reads well on the
  // dark theme, plus "Custom" (the native colour input) for anything else.
  // A value that isn't in the palette shows as the selected custom swatch.
  import { PALETTE } from './format';

  let {
    value = $bindable(''),
    label,
    allowNone = false,
  }: { value?: string; label: string; allowNone?: boolean } = $props();

  const norm = (c: string) => c.trim().toLowerCase();
  const custom = $derived(value !== '' && !PALETTE.some((p) => norm(p.hex) === norm(value)));
</script>

<div class="swatches" role="radiogroup" aria-label={label}>
  {#if allowNone}
    <button type="button" class="swatch none" class:on={value === ''} role="radio" aria-checked={value === ''}
      aria-label="Default" title="Default" onclick={() => (value = '')}></button>
  {/if}
  {#each PALETTE as c (c.hex)}
    {@const on = norm(value) === norm(c.hex)}
    <button type="button" class="swatch" class:on style:background={c.hex}
      role="radio" aria-checked={on} aria-label={c.name} title={c.name} onclick={() => (value = c.hex)}></button>
  {/each}
  <label class="swatch custom" class:on={custom} style:background={custom ? value : undefined} title="Custom colour">
    <span aria-hidden="true">{custom ? '' : '+'}</span>
    <input type="color" value={custom ? value : '#888888'} oninput={(e) => (value = e.currentTarget.value)}
      aria-label="Custom colour" />
  </label>
</div>

<style>
  .swatches { display: grid; grid-template-columns: repeat(auto-fill, 34px); gap: 8px; }
  .swatch { width: 34px; height: 34px; border-radius: 50%; border: 2px solid transparent; padding: 0; position: relative; cursor: pointer; }
  .swatch.none { background: var(--surface-raised); border-color: var(--border-strong); }
  .swatch.none::after { content: ''; position: absolute; left: 50%; top: 4px; bottom: 4px; width: 2px; background: var(--text-3); transform: rotate(45deg); }
  .swatch.on { border-color: var(--text); box-shadow: 0 0 0 2px var(--canvas) inset; }
  .custom {
    display: grid; place-items: center; font-weight: 700; color: var(--text-2);
    background: conic-gradient(#f43f62, #f2b45c, #facc15, #74d7a9, #22d3ee, #7cb6fa, #b18cf5, #ec4899, #f43f62);
  }
  .custom span { width: 20px; height: 20px; border-radius: 50%; display: grid; place-items: center; background: var(--surface); font-size: 15px; line-height: 1; }
  .custom.on span { display: none; }
  .custom:focus-within { outline: 2px solid var(--accent); outline-offset: 2px; }
  /* The native input sits over the swatch so a click/tap opens its picker. */
  .custom input { position: absolute; inset: 0; opacity: 0; width: 100%; height: 100%; cursor: pointer; border: 0; padding: 0; }
</style>
