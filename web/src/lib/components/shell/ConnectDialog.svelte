<script lang="ts">
  // QR codes for opening this app on another device (the GM's tablet).
  // The server lists its own Tailscale and LAN addresses, because the
  // desktop's URL is often localhost, which is useless to a tablet.
  import Modal from '$lib/components/ui/Modal.svelte';
  import { connect } from '$lib/api';
  import type { ConnectUrl } from '$lib/types';

  let { open = $bindable(false) }: { open?: boolean } = $props();

  let urls: ConnectUrl[] = $state([]);
  let selected = $state(0);
  let loading = $state(false);
  let error = $state('');

  $effect(() => {
    if (!open) return;
    loading = true;
    error = '';
    connect
      .urls()
      .then((r) => { urls = r.urls; selected = 0; })
      .catch((e) => { error = (e as Error).message; })
      .finally(() => { loading = false; });
  });

  const current = $derived(urls[selected]);
</script>

<Modal bind:open title="Connect a device" width="440px">
  {#if loading && !urls.length}
    <p class="subtle">Finding this computer's addresses…</p>
  {:else if error}
    <p class="subtle">Couldn't list addresses: {error}</p>
  {:else if !urls.length}
    <p class="subtle">No LAN or Tailscale address found on this computer.</p>
  {:else}
    <p class="small subtle">Scan with the tablet's camera. Tailscale works anywhere the tablet is signed in to your tailnet; LAN only works on the home network.</p>
    {#if urls.length > 1}
      <div class="choices" role="tablist" aria-label="Address">
        {#each urls as u, i}
          <button class="btn btn-sm" class:on={i === selected} role="tab" aria-selected={i === selected}
            onclick={() => (selected = i)}>{u.label}</button>
        {/each}
      </div>
    {/if}
    {#if current}
      <img class="qr" src={`/api/connect/qr?url=${encodeURIComponent(current.url)}`} alt={`QR code for ${current.url}`} />
      <a class="url" href={current.url}>{current.url}</a>
    {/if}
  {/if}
</Modal>

<style>
  .choices { display: flex; flex-wrap: wrap; gap: 6px; }
  .choices .on { border-color: var(--accent); color: var(--accent-text); }
  /* QR codes need a light quiet zone to scan reliably, whatever the theme. */
  .qr {
    display: block; width: min(100%, 300px); aspect-ratio: 1; margin: 0 auto;
    background: #fff; border-radius: var(--radius-sm); image-rendering: pixelated;
  }
  .url { text-align: center; font-family: var(--font-mono, monospace); font-size: 14px; word-break: break-all; }
</style>
