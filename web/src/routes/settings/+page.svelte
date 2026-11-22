<script lang="ts">
  // App-wide settings. Only what can safely change at runtime lives here;
  // Discord credentials, the transcription engine and folders stay in
  // config.yaml on the server. Values saved here are stored in the database
  // and override config.yaml's defaults.
  import { onMount } from 'svelte';
  import { dj } from '$lib/api';
  import { addToast } from '$lib/stores/toast';

  let loaded = $state(false);
  let fade = $state(6);
  let saved = $state(6);
  let saving = $state(false);

  onMount(async () => {
    try {
      const s = await dj.settings();
      fade = saved = s.fadeSeconds;
      loaded = true;
    } catch (e) {
      addToast(`Couldn't load settings: ${(e as Error).message}`, 'error');
    }
  });

  // Saved when the slider is released (or a preset tapped), not on every
  // step of a drag.
  async function saveFade(v: number) {
    fade = v;
    if (v === saved) return;
    saving = true;
    try {
      const s = await dj.saveSettings({ fadeSeconds: v });
      fade = saved = s.fadeSeconds;
    } catch (e) {
      addToast(`Couldn't save: ${(e as Error).message}`, 'error');
      fade = saved;
    } finally {
      saving = false;
    }
  }

  const PRESETS = [0, 2, 4, 6, 8, 12, 20];
  const fmt = (s: number) => (s === 0 ? 'Cut' : `${s % 1 ? s.toFixed(1) : s} s`);
</script>

<svelte:head><title>Settings - dndscribe</title></svelte:head>

<main class="settings">
  <div>
    <div class="eyebrow">dndscribe</div>
    <h1 class="page-title">Settings</h1>
    <p class="subtle lead">
      App-wide settings. Discord credentials, the transcription engine and the music and soundboard folders are set in
      config.yaml on the server.
    </p>
  </div>

  <section class="card block" aria-labelledby="dj-h">
    <h2 id="dj-h" class="section-title">DJ Studio</h2>

    <div class="setting">
      <div class="label">
        <strong>Crossfade</strong>
        <span class="small subtle">
          How long one scene or playlist takes to fade into the next. A scene can set its own in the scene editor;
          this is used for every scene that doesn't. Stop and the first scene after silence always start quickly.
        </span>
      </div>
      <div class="control">
        <div class="value">
          <span class="big">{fmt(fade)}</span>
          {#if saving}<span class="tiny subtle">Saving…</span>{:else if loaded && fade === saved}<span class="tiny subtle">Saved</span>{/if}
        </div>
        <input type="range" min="0" max="30" step="0.5" bind:value={fade} disabled={!loaded}
          onchange={() => saveFade(fade)} aria-label="Crossfade length in seconds" />
        <div class="presets">
          {#each PRESETS as p}
            <button class="btn btn-sm" class:on={fade === p} onclick={() => saveFade(p)} disabled={!loaded}>{fmt(p)}</button>
          {/each}
        </div>
      </div>
    </div>
  </section>
</main>

<style>
  .settings { max-width: 900px; width: 100%; margin: 0 auto; padding: 36px 32px 80px; display: flex; flex-direction: column; gap: 24px; }
  .eyebrow { margin-bottom: 10px; }
  .lead { margin-top: 8px; max-width: 640px; }
  .block { padding: 20px 22px; display: flex; flex-direction: column; gap: 18px; }
  .setting { display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 24px; align-items: start; }
  .label { display: flex; flex-direction: column; gap: 6px; }
  .control { display: flex; flex-direction: column; gap: 10px; }
  .value { display: flex; align-items: baseline; justify-content: space-between; }
  .big { font-size: 22px; font-weight: 700; font-variant-numeric: tabular-nums; }
  input[type='range'] { width: 100%; accent-color: var(--accent); height: 28px; }
  .presets { display: flex; flex-wrap: wrap; gap: 6px; }
  .presets .on { border-color: var(--accent); color: var(--accent-text); }

  @media (max-width: 700px) {
    .settings { padding: 28px 16px 60px; }
    .setting { grid-template-columns: 1fr; }
  }
</style>
