<script lang="ts">
  let {
    minConfidence = $bindable(100),
    condensed = $bindable(false),
    showConfidence = $bindable(true),
  }: {
    minConfidence?: number;
    condensed?: boolean;
    showConfidence?: boolean;
  } = $props();
</script>

<div class="confidence-filter">
  <label>
    <input type="checkbox" bind:checked={condensed} />
    Condensed
  </label>
  <label>
    <input type="checkbox" bind:checked={showConfidence} />
    Show Conf
  </label>
  <label>
    Min conf:
    <input
      type="range"
      min="0"
      max="100"
      bind:value={minConfidence}
      class="slider"
    />
    <span class="value">{minConfidence}%</span>
  </label>
</div>

<style>
  .confidence-filter {
    display: flex; align-items: center; gap: 12px;
    font-size: 12px; color: var(--text-secondary);
  }
  label { display: flex; align-items: center; gap: 4px; white-space: nowrap; }
  input[type="checkbox"] { cursor: pointer; }
  .slider { width: 60px; cursor: pointer; }
  .value { color: var(--text-muted); min-width: 28px; }
</style>
