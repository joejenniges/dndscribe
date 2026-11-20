<script lang="ts">
  import type { Snippet } from 'svelte';

  let {
    open = $bindable(false),
    title,
    width = '480px',
    onclose,
    children,
    footer,
  }: {
    open?: boolean;
    title: string;
    width?: string;
    onclose?: () => void;
    children: Snippet;
    footer?: Snippet;
  } = $props();

  let dialogEl: HTMLDialogElement | undefined = $state();

  // Native <dialog> gives focus trapping, Esc handling and top-layer stacking
  // without a dependency.
  $effect(() => {
    if (!dialogEl) return;
    if (open && !dialogEl.open) dialogEl.showModal();
    if (!open && dialogEl.open) dialogEl.close();
  });

  function handleClose() {
    open = false;
    onclose?.();
  }

  function handleBackdrop(e: MouseEvent) {
    if (e.target === dialogEl) handleClose();
  }
</script>

<dialog bind:this={dialogEl} style:width onclose={handleClose} onclick={handleBackdrop}>
  {#if open}
    <div class="modal-inner">
      <header>
        <h2>{title}</h2>
        <button class="icon-btn" aria-label="Close" onclick={handleClose}>&times;</button>
      </header>
      <div class="body">{@render children()}</div>
      {#if footer}<footer>{@render footer()}</footer>{/if}
    </div>
  {/if}
</dialog>

<style>
  dialog {
    max-width: calc(100vw - 32px); max-height: calc(100vh - 48px);
    margin: auto; padding: 0; border: 1px solid var(--border-strong);
    border-radius: var(--radius-lg); background: var(--surface); color: var(--text);
    box-shadow: var(--shadow);
  }
  dialog::backdrop { background: rgba(0, 0, 0, 0.65); }
  .modal-inner { display: flex; flex-direction: column; max-height: calc(100vh - 50px); }
  header {
    display: flex; align-items: center; justify-content: space-between; gap: 12px;
    padding: 14px 14px 14px 22px; border-bottom: 1px solid var(--border-color);
  }
  h2 { font-size: 17px; font-weight: 700; }
  .body { padding: 20px 22px; overflow-y: auto; display: flex; flex-direction: column; gap: 14px; }
  footer {
    display: flex; justify-content: flex-end; gap: 10px;
    padding: 14px 22px; border-top: 1px solid var(--border-color);
  }
</style>
