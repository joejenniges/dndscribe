<script lang="ts">
  import { page } from '$app/stores';
  import { botStatus, wsConnected } from '$lib/stores/ws';

  const nav = [
    { href: '/', label: 'Campaigns', match: (p: string) => p === '/' || p.startsWith('/campaigns') },
    { href: '/library', label: 'Library', match: (p: string) => p.startsWith('/library') },
    { href: '/dj', label: 'DJ Studio', match: (p: string) => p.startsWith('/dj') },
    { href: '/tools', label: 'Tools', match: (p: string) => p.startsWith('/tools') },
  ];

  const path = $derived($page.url.pathname);
  const liveHref = $derived($botStatus.campaignId ? `/campaigns/${$botStatus.campaignId}/live` : null);
</script>

<header class="app-header">
  <a href="/" class="brand" aria-label="dndscribe home">dnd<span>scribe</span></a>

  <nav class="global-nav" aria-label="Primary">
    {#each nav as item}
      <a href={item.href} class:active={item.match(path)} aria-current={item.match(path) ? 'page' : undefined}>
        {item.label}
      </a>
    {/each}
  </nav>

  <div class="status">
    {#if $botStatus.recording && liveHref}
      <a href={liveHref} class="rec-pill" title="Open live transcript">
        <span class="rec-dot on"></span>
        <span class="rec-label">Recording</span>
        <span class="rec-channel">{$botStatus.channelName}</span>
      </a>
    {/if}
    {#if !$wsConnected}
      <span class="badge warning" title="Live updates disconnected; reconnecting">Offline</span>
    {/if}
    <a href="/settings" class="icon-btn" class:active={path.startsWith('/settings')} aria-label="Settings" title="Settings">&#9881;</a>
  </div>
</header>

<style>
  .app-header {
    position: sticky; top: 0; z-index: 40;
    height: var(--header-h); display: flex; align-items: center; gap: 28px;
    padding: 0 24px; background: var(--surface-sunken); border-bottom: 1px solid var(--border-color);
  }
  .brand { font-weight: 800; font-size: 20px; letter-spacing: -0.7px; color: var(--accent); flex-shrink: 0; }
  .brand span { color: var(--text); }
  .brand:hover { text-decoration: none; }

  .global-nav { display: flex; align-items: stretch; height: 100%; gap: 4px; overflow-x: auto; scrollbar-width: none; flex: 1; }
  .global-nav::-webkit-scrollbar { display: none; }
  .global-nav a {
    display: flex; align-items: center; padding: 0 14px; white-space: nowrap;
    color: var(--text-2); font-weight: 600; font-size: 14px;
    border-bottom: 2px solid transparent; margin-bottom: -1px;
  }
  .global-nav a:hover { color: var(--text); text-decoration: none; }
  .global-nav a.active { color: var(--text); border-bottom-color: var(--accent); }

  .status { display: flex; align-items: center; gap: 10px; flex-shrink: 0; }
  .rec-pill {
    display: flex; align-items: center; gap: 8px; min-height: 34px; padding: 0 12px;
    border-radius: 99px; background: var(--accent-soft); color: var(--accent-text);
    font-size: 13px; font-weight: 700;
  }
  .rec-pill:hover { text-decoration: none; background: #4a2a3b; }
  .rec-channel { color: var(--text-2); font-weight: 600; max-width: 160px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .icon-btn.active { color: var(--text); background: var(--surface-raised); }

  @media (max-width: 760px) {
    .app-header { gap: 12px; padding: 0 12px; }
    .brand { font-size: 17px; }
    .global-nav a { padding: 0 10px; }
    .rec-label, .rec-channel { display: none; }
    .rec-pill { padding: 0 10px; }
  }
</style>
