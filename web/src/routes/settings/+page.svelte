<script lang="ts">
  import Placeholder from '$lib/components/ui/Placeholder.svelte';
</script>

<svelte:head><title>Settings - dndscribe</title></svelte:head>

<Placeholder
  title="Settings"
  description="App-wide settings. Transcription engine, models and Discord credentials are configured in config.yaml on the server."
  ideas={['Show active engine and model', 'Default campaign', 'Music library root (DJ)']}
/>
