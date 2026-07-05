<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { connect, disconnect } from '$lib/ws';
  import { initWsStores } from '$lib/stores/ws';
  import Toast from '$lib/components/Toast.svelte';
  import '../app.css';

  let { children } = $props();

  let unsub: (() => void) | undefined;

  onMount(() => {
    connect();
    unsub = initWsStores();
  });

  onDestroy(() => {
    unsub?.();
    disconnect();
  });
</script>

{@render children()}
<Toast />
