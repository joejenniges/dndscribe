<script lang="ts">
  import { toasts } from '$lib/stores/toast';
</script>

<div class="toast-container">
  {#each $toasts as toast (toast.id)}
    <div class="toast {toast.type}">{toast.message}</div>
  {/each}
</div>
