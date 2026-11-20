<script lang="ts" module>
  export interface MenuItem {
    label: string;
    onselect: () => void;
    danger?: boolean;
    disabled?: boolean;
  }
</script>

<script lang="ts">
  // Overflow ("...") menu. Items are plain data so callers don't need to
  // manage open state or click-outside handling.
  let {
    items,
    label = 'More actions',
    align = 'right',
  }: {
    items: MenuItem[];
    label?: string;
    align?: 'left' | 'right';
  } = $props();

  let open = $state(false);
  let rootEl: HTMLDivElement | undefined = $state();

  function select(item: MenuItem, e: MouseEvent) {
    e.preventDefault();
    e.stopPropagation();
    if (item.disabled) return;
    open = false;
    item.onselect();
  }

  function toggle(e: MouseEvent) {
    // Menus often sit inside clickable rows/links; don't trigger those.
    e.preventDefault();
    e.stopPropagation();
    open = !open;
  }

  function onWindowClick(e: MouseEvent) {
    if (open && rootEl && !rootEl.contains(e.target as Node)) open = false;
  }

  function onKeydown(e: KeyboardEvent) {
    if (e.key === 'Escape' && open) { open = false; }
  }
</script>

<svelte:window onclick={onWindowClick} onkeydown={onKeydown} />

<div class="menu" bind:this={rootEl}>
  <button class="icon-btn" aria-label={label} aria-haspopup="menu" aria-expanded={open} onclick={toggle}>
    &#8943;
  </button>
  {#if open}
    <div class="menu-list" class:left={align === 'left'} role="menu">
      {#each items as item}
        <button
          role="menuitem"
          class="menu-item"
          class:danger={item.danger}
          disabled={item.disabled}
          onclick={(e) => select(item, e)}
        >{item.label}</button>
      {/each}
    </div>
  {/if}
</div>

<style>
  .menu { position: relative; }
  .menu-list {
    position: absolute; top: calc(100% + 4px); right: 0; z-index: 60;
    min-width: 180px; padding: 6px; border-radius: var(--radius);
    background: var(--surface-raised); border: 1px solid var(--border-strong);
    box-shadow: var(--shadow); display: flex; flex-direction: column;
  }
  .menu-list.left { right: auto; left: 0; }
  .menu-item {
    min-height: var(--control-h); padding: 0 12px; text-align: left;
    background: none; border: 0; border-radius: var(--radius-sm); font-size: 14px; color: var(--text);
  }
  .menu-item:hover:not(:disabled) { background: var(--surface-hover); }
  .menu-item.danger { color: var(--danger); }
</style>
