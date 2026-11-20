<script lang="ts">
  import { currentCampaign } from '$lib/stores/campaign';
  import Placeholder from '$lib/components/ui/Placeholder.svelte';
</script>

<svelte:head><title>Characters - {$currentCampaign?.name ?? 'Campaign'} - dndscribe</title></svelte:head>

<Placeholder
  eyebrow={$currentCampaign?.name ?? ''}
  title="Characters"
  description="Player characters and NPCs for this campaign."
  ideas={['Link Discord users to their characters', 'NPC roster pulled from transcript mentions', 'Seed hotwords from character names']}
/>
