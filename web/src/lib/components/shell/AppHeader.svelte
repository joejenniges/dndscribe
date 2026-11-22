<script lang="ts">
  import { page } from '$app/stores';
  import { botStatus, wsConnected } from '$lib/stores/ws';
  import ConnectDialog from './ConnectDialog.svelte';

  let connectOpen = $state(false);

  const path = $derived($page.url.pathname);
  const liveHref = $derived(
    $botStatus.recording && $botStatus.campaignId ? `/campaigns/${$botStatus.campaignId}/live` : null,
  );

  // Campaigns returns to the last campaign page visited (a session, notes,
  // the live view) rather than the campaign list, so hopping to DJ Studio
  // and back doesn't lose your place. Kept in sessionStorage so a reload
  // keeps it too; the list is one click on the brand away.
  const LAST_KEY = 'dndscribe.lastCampaignPath';
  let lastCampaignPath = $state(readLast());
  function readLast(): string | null {
    try { return sessionStorage.getItem(LAST_KEY); } catch { return null; }
  }
  $effect(() => {
    if (!path.startsWith('/campaigns/')) return;
    lastCampaignPath = path;
    try { sessionStorage.setItem(LAST_KEY, path); } catch { /* per-tab convenience only */ }
  });

  const onLive = $derived(liveHref !== null && path === liveHref);
  const nav = $derived([
    {
      href: path.startsWith('/campaigns/') ? '/' : (lastCampaignPath ?? '/'),
      label: 'Campaigns',
      active: !onLive && (path === '/' || path.startsWith('/campaigns')),
      live: false,
    },
    // Shown only while a session is being recorded.
    ...(liveHref ? [{ href: liveHref, label: 'Live', active: onLive, live: true }] : []),
    { href: '/library', label: 'Library', active: path.startsWith('/library'), live: false },
    { href: '/dj', label: 'DJ Studio', active: path.startsWith('/dj'), live: false },
    { href: '/tools', label: 'Tools', active: path.startsWith('/tools'), live: false },
  ]);
</script>

<header class="app-header">
  <a href="/" class="brand" aria-label="dndscribe home">dnd<span>scribe</span></a>

  <nav class="global-nav" aria-label="Primary">
    {#each nav as item}
      <a href={item.href} class:active={item.active} aria-current={item.active ? 'page' : undefined}
        title={item.live ? `Live transcript: recording in ${$botStatus.channelName ?? 'voice'}` : undefined}>
        {#if item.live}<span class="rec-dot on"></span>{/if}
        {item.label}
      </a>
    {/each}
  </nav>

  <div class="status">
    {#if !$wsConnected}
      <span class="badge warning" title="Live updates disconnected; reconnecting">Offline</span>
    {/if}
    <button class="icon-btn" onclick={() => (connectOpen = true)} aria-label="Connect a device" title="Connect a device (QR code)">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
        <rect x="3" y="3" width="7" height="7" rx="1" /><rect x="14" y="3" width="7" height="7" rx="1" /><rect x="3" y="14" width="7" height="7" rx="1" />
        <path d="M14 14h3v3h-3zM20 14v.01M14 20h.01M17 20h4v-3" />
      </svg>
    </button>
    <a href="/settings" class="icon-btn" class:active={path.startsWith('/settings')} aria-label="Settings" title="Settings">&#9881;</a>
  </div>
</header>

<ConnectDialog bind:open={connectOpen} />

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
    display: flex; align-items: center; gap: 8px; padding: 0 14px; white-space: nowrap;
    color: var(--text-2); font-weight: 600; font-size: 14px;
    border-bottom: 2px solid transparent; margin-bottom: -1px;
  }
  .global-nav a:hover { color: var(--text); text-decoration: none; }
  .global-nav a.active { color: var(--text); border-bottom-color: var(--accent); }

  .status { display: flex; align-items: center; gap: 10px; flex-shrink: 0; }
  .icon-btn.active { color: var(--text); background: var(--surface-raised); }

  @media (max-width: 760px) {
    .app-header { gap: 12px; padding: 0 12px; }
    .brand { font-size: 17px; }
    .global-nav a { padding: 0 10px; }
  }
</style>
