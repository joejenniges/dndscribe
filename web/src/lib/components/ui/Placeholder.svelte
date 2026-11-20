<script lang="ts">
  // Stand-in for sections that exist in the navigation but have no backend yet
  // (Characters, Notes, Tools, Library). Deliberately says so plainly rather
  // than showing fake data.
  let {
    eyebrow = '',
    title,
    description,
    ideas = [],
  }: {
    eyebrow?: string;
    title: string;
    description: string;
    ideas?: string[];
  } = $props();
</script>

<div class="placeholder-page">
  {#if eyebrow}<div class="eyebrow">{eyebrow}</div>{/if}
  <h1 class="page-title">{title}</h1>
  <p class="muted lead">{description}</p>

  <div class="empty-state">
    <h3>Not built yet</h3>
    <p>This section is a placeholder.</p>
    {#if ideas.length}
      <ul>
        {#each ideas as idea}<li>{idea}</li>{/each}
      </ul>
    {/if}
  </div>
</div>

<style>
  .placeholder-page { max-width: 1240px; margin: 0 auto; padding: 40px 32px; }
  .eyebrow { margin-bottom: 10px; }
  .lead { margin: 8px 0 28px; max-width: 640px; }
  ul { list-style: none; margin-top: 16px; display: flex; flex-direction: column; gap: 6px; font-size: 14px; color: var(--text-3); }
  @media (max-width: 650px) { .placeholder-page { padding: 28px 16px; } }
</style>
