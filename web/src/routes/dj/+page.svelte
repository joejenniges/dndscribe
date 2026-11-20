<script lang="ts">
  // DJ Studio, scenes-first. Layout only for now: the Go mixer and the
  // /api/dj endpoints don't exist yet, so every control is disabled and the
  // page says so. No sample data, on purpose.
  //
  // Layout intent (tablet at the table): big scene tiles you can hit without
  // looking, a persistent now-playing bar, and the playlist/queue as the
  // secondary view.
</script>

<svelte:head><title>DJ Studio - dndscribe</title></svelte:head>

<main class="dj">
  <div class="heading">
    <div>
      <div class="eyebrow">Discord music</div>
      <h1 class="page-title">DJ Studio</h1>
      <p class="muted">Tap a scene to crossfade the table's music in Discord. Audio is mixed on the server, so it keeps playing if this page is closed.</p>
    </div>
    <div class="conn">
      <span class="badge warning">Not built yet</span>
    </div>
  </div>

  <div class="notice card">
    <strong>Coming next.</strong>
    The bot will play through the same voice connection it records on, using discordgo's
    <code>OpusSend</code> channel. Scenes and playlists will live in the database; tracks come from a music folder on the server.
  </div>

  <section>
    <div class="section-head">
      <h2 class="section-title">Scenes</h2>
      <button class="btn btn-sm" disabled>+ New scene</button>
    </div>
    <div class="scenes">
      {#each Array(6) as _, i}
        <button class="scene" disabled aria-label="Empty scene slot {i + 1}">
          <span class="scene-name">Empty scene</span>
          <span class="scene-meta">Assign a playlist</span>
        </button>
      {/each}
    </div>
  </section>

  <section class="two-col">
    <div class="card pane">
      <h2 class="section-title">Playlists</h2>
      <p class="subtle small">Playlists group tracks from the server's music folder. A scene points at a playlist.</p>
    </div>
    <div class="card pane">
      <h2 class="section-title">Up next</h2>
      <p class="subtle small">The queue for the playing scene shows here.</p>
    </div>
  </section>
</main>

<div class="now-playing" aria-label="Now playing">
  <div class="np-info">
    <span class="np-title">Nothing playing</span>
    <span class="subtle tiny">Discord output: not connected</span>
  </div>
  <div class="np-controls">
    <button class="icon-btn bordered big" disabled aria-label="Play">&#9654;</button>
    <button class="icon-btn bordered big" disabled aria-label="Stop">&#9632;</button>
    <label class="vol">
      <span class="subtle tiny">Volume</span>
      <input type="range" min="0" max="100" value="70" disabled aria-label="Master volume" />
    </label>
    <label class="vol">
      <span class="subtle tiny">Fade</span>
      <select class="select" disabled aria-label="Crossfade length"><option>8 s</option></select>
    </label>
  </div>
</div>

<style>
  .dj { max-width: 1240px; width: 100%; margin: 0 auto; padding: 36px 32px 140px; display: flex; flex-direction: column; gap: 28px; }
  .heading { display: flex; justify-content: space-between; align-items: flex-end; gap: 20px; flex-wrap: wrap; }
  .heading .eyebrow { margin-bottom: 10px; }
  .heading p { margin-top: 6px; max-width: 640px; }
  .notice { padding: 16px 20px; font-size: 14px; color: var(--text-2); line-height: 1.6; }
  .notice strong { color: var(--text); }
  code { font-family: var(--font-mono); font-size: 13px; background: var(--surface-raised); padding: 1px 6px; border-radius: 4px; }

  .section-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; }
  /* Tablet-first: tiles are large touch targets. */
  .scenes { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 14px; }
  .scene {
    min-height: 120px; padding: 18px; border-radius: var(--radius-lg); text-align: left;
    display: flex; flex-direction: column; justify-content: flex-end; gap: 4px;
    background: var(--surface); border: 1px dashed var(--border-strong); color: var(--text-2);
  }
  .scene:disabled { opacity: 0.7; cursor: default; }
  .scene-name { font-size: 17px; font-weight: 700; color: var(--text-2); }
  .scene-meta { font-size: 13px; color: var(--text-3); }

  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
  .pane { padding: 20px 22px; display: flex; flex-direction: column; gap: 6px; min-height: 160px; }

  .now-playing {
    position: fixed; left: 0; right: 0; bottom: 0; z-index: 45;
    min-height: 84px; padding: 12px 24px; display: flex; align-items: center; justify-content: space-between; gap: 20px;
    background: var(--surface); border-top: 1px solid var(--border-strong); box-shadow: 0 -8px 24px rgba(0,0,0,0.35);
  }
  .np-info { display: flex; flex-direction: column; min-width: 0; }
  .np-title { font-weight: 700; font-size: 16px; }
  .np-controls { display: flex; align-items: center; gap: 12px; }
  .big { width: var(--touch); height: var(--touch); font-size: 18px; }
  .vol { display: flex; flex-direction: column; gap: 2px; }
  .vol input { width: 140px; }
  .vol .select { min-height: 36px; width: 90px; }

  @media (max-width: 760px) {
    .dj { padding: 24px 16px 180px; }
    .two-col { grid-template-columns: 1fr; }
    .scenes { grid-template-columns: repeat(2, 1fr); }
    .now-playing { flex-direction: column; align-items: stretch; padding: 12px 16px; gap: 10px; }
    .np-controls { flex-wrap: wrap; }
  }
</style>
