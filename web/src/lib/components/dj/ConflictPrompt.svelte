<script lang="ts" module>
  export type ConflictChoice = 'skip' | 'overwrite' | 'cancel';
</script>

<script lang="ts">
  // "N items will be overwritten, skip or overwrite?" Shared by the music
  // upload dialog and the soundboard. Call ask(names) and await the choice;
  // closing the dialog (Esc, backdrop, x) counts as cancel.
  import Modal from '$lib/components/ui/Modal.svelte';

  let open = $state(false);
  let names = $state<string[]>([]);
  let total = $state(0);
  let resolve: ((c: ConflictChoice) => void) | null = null;

  export function ask(list: string[], of: number): Promise<ConflictChoice> {
    resolve?.('cancel');
    names = list;
    total = of;
    open = true;
    return new Promise((r) => (resolve = r));
  }

  function pick(c: ConflictChoice) {
    const r = resolve;
    resolve = null;
    open = false;
    r?.(c);
  }

  const n = $derived(names.length);
  const fresh = $derived(total - n);
</script>

<Modal bind:open title="Already uploaded" width="480px" onclose={() => pick('cancel')}>
  <p class="lead"><strong>{n} item{n === 1 ? '' : 's'}</strong> will be overwritten, skip or overwrite?</p>
  <p class="tiny subtle">
    {fresh > 0 ? `${fresh} new item${fresh === 1 ? '' : 's'} will be uploaded either way.` : 'Nothing else is new.'}
  </p>
  <ul class="names">
    {#each names.slice(0, 100) as name (name)}<li class="tiny">{name}</li>{/each}
    {#if n > 100}<li class="tiny subtle">and {n - 100} more</li>{/if}
  </ul>
  {#snippet footer()}
    <button class="btn-ghost" onclick={() => pick('cancel')}>Cancel</button>
    <span class="spacer"></span>
    <button class="btn" onclick={() => pick('overwrite')}>Overwrite</button>
    <button class="btn-primary" onclick={() => pick('skip')}>Skip</button>
  {/snippet}
</Modal>

<style>
  .lead { margin: 0 0 4px; }
  .names { list-style: none; margin: 10px 0 0; padding: 0; max-height: 220px; overflow-y: auto; font-family: var(--font-mono); }
  .names li { padding: 2px 0; word-break: break-all; }
  .spacer { flex: 1; }
</style>
