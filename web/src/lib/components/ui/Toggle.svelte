<script lang="ts">
  // Switch-style boolean control. Renders a real <button role="switch"> so it is
  // keyboard and screen-reader accessible; the whole row is the hit target.
  let {
    checked = $bindable(false),
    label,
    hint = '',
    disabled = false,
    onchange,
  }: {
    checked?: boolean;
    label: string;
    hint?: string;
    disabled?: boolean;
    onchange?: (value: boolean) => void;
  } = $props();

  function toggle() {
    if (disabled) return;
    checked = !checked;
    onchange?.(checked);
  }
</script>

<button
  type="button"
  class="toggle-row"
  role="switch"
  aria-checked={checked}
  {disabled}
  onclick={toggle}
>
  <span class="text">
    <span class="label">{label}</span>
    {#if hint}<span class="hint">{hint}</span>{/if}
  </span>
  <span class="track" class:on={checked}><span class="thumb"></span></span>
</button>

<style>
  .toggle-row {
    display: flex; align-items: center; justify-content: space-between; gap: 12px;
    width: 100%; min-height: var(--control-h); padding: 4px 0;
    background: none; border: 0; text-align: left; border-radius: var(--radius-sm);
  }
  .text { display: flex; flex-direction: column; }
  .label { font-size: 14px; color: var(--text); }
  .hint { font-size: 12px; color: var(--text-3); }
  .track {
    position: relative; flex-shrink: 0; width: 38px; height: 22px; border-radius: 99px;
    background: var(--border-strong); transition: background 0.15s;
  }
  .track.on { background: var(--accent-strong); }
  .thumb {
    position: absolute; top: 3px; left: 3px; width: 16px; height: 16px; border-radius: 50%;
    background: #fff; transition: transform 0.15s;
  }
  .track.on .thumb { transform: translateX(16px); }
</style>
