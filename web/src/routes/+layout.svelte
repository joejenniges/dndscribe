<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { page } from '$app/stores';
  import { connect, disconnect, sayHello } from '$lib/ws';
  import { deviceKind } from '$lib/dj/live';
  import { initWsStores } from '$lib/stores/ws';
  import Toast from '$lib/components/Toast.svelte';
  import AppHeader from '$lib/components/shell/AppHeader.svelte';
  import '../app.css';

  let { children } = $props();

  let unsub: (() => void) | undefined;

  // WHY: the nickname palette is opened with window.open() as a small popup
  // window; it must render without the app chrome.
  const bare = $derived(($page.route.id ?? '').includes('/palette/'));

  // Presence: tell the server what this device is and which section it's
  // on, so the DJ page can show "desktop + tablet connected".
  $effect(() => {
    sayHello(deviceKind(), $page.url.pathname.split('/')[1] || 'home');
  });

  onMount(() => {
    connect();
    unsub = initWsStores();
  });

  onDestroy(() => {
    unsub?.();
    disconnect();
  });
</script>

{#if bare}
  {@render children()}
{:else}
  <div class="app">
    <AppHeader />
    <div class="app-body">
      {@render children()}
    </div>
  </div>
{/if}
<Toast />

<style>
  .app { min-height: 100vh; display: flex; flex-direction: column; }
  .app-body { flex: 1; display: flex; flex-direction: column; min-height: 0; }
</style>
