<script lang="ts">
  import { currentCampaign } from '$lib/stores/campaign';
  import Placeholder from '$lib/components/ui/Placeholder.svelte';
</script>

<svelte:head><title>Notes - {$currentCampaign?.name ?? 'Campaign'} - dndscribe</title></svelte:head>

<Placeholder
  eyebrow={$currentCampaign?.name ?? ''}
  title="Notes"
  description="Campaign notes and session recaps."
  ideas={['Per-session recap', 'Pin transcript lines as notes', 'Running campaign log']}
/>
