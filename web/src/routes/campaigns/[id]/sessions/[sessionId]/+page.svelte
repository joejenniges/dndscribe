<script lang="ts">
  import { page } from '$app/stores';
  import { scoped } from '$lib/api';
  import { currentCampaign, sessions } from '$lib/stores/campaign';
  import { addToast } from '$lib/stores/toast';
  import TranscriptView from '$lib/components/TranscriptView.svelte';
  import type { TranscriptionLine as TLine } from '$lib/types';

  let campaignId = $derived(Number($page.params.id));
  let sessionId = $derived(Number($page.params.sessionId));
  let api = $derived(scoped(campaignId));

  let allLines: TLine[] = $state([]);
  let loading = $state(true);

  const session = $derived($sessions.find(s => s.id === sessionId));
  const title = $derived(session?.channelName || `Session #${sessionId}`);

  const speakers = $derived((() => {
    const counts = new Map<string, number>();
    for (const l of allLines) {
      const name = l.nickname || l.discordUsername;
      counts.set(name, (counts.get(name) ?? 0) + 1);
    }
    return [...counts.entries()].sort((a, b) => b[1] - a[1]);
  })());

  async function load(id: number) {
    loading = true;
    try {
      allLines = await api.lines.list(id);
    } catch (e) {
      addToast(`Failed to load session: ${e}`, 'error');
    } finally {
      loading = false;
    }
  }

  $effect(() => { if (sessionId) load(sessionId); });

  async function editLine(id: number, text: string) {
    try {
      const result = await api.lines.update(id, text);
      allLines = allLines.map(l => l.id === id ? { ...l, text } : l);
      if (result.autoAdded.length > 0) addToast(`Auto-added hotwords: ${result.autoAdded.join(', ')}`, 'success');
    } catch (e) {
      addToast(`Failed: ${e}`, 'error');
    }
  }

  async function deleteLine(id: number) {
    try {
      await api.lines.delete(id);
      allLines = allLines.filter(l => l.id !== id);
    } catch (e) {
      addToast(`Failed: ${e}`, 'error');
    }
  }

  function fmt(iso: string, opts: Intl.DateTimeFormatOptions) {
    return new Date(iso).toLocaleString([], opts);
  }
  function duration(): string {
    if (!session?.endedAt) return '';
    const mins = Math.round((new Date(session.endedAt).getTime() - new Date(session.startedAt).getTime()) / 60000);
    return mins < 60 ? `${mins}m` : `${Math.floor(mins / 60)}h ${String(mins % 60).padStart(2, '0')}m`;
  }
</script>

<svelte:head>
  <title>{title} - {$currentCampaign?.name ?? 'Campaign'} - dndscribe</title>
</svelte:head>

<TranscriptView
  {campaignId}
  lines={allLines}
  {loading}
  emptyMessage="This session has no transcript lines."
  onEditLine={editLine}
  onDeleteLine={deleteLine}
  onNicknamesSaved={() => load(sessionId)}
>
  {#snippet header()}
    <div class="session-head">
      <div>
        <div class="eyebrow crumbs">
          <a href="/campaigns/{campaignId}">{$currentCampaign?.name ?? 'Campaign'} / Sessions</a>
        </div>
        <h1 class="title">
          {title}
          <span class="badge">{session?.endedAt ? 'Ended' : session ? 'Open' : ''}</span>
        </h1>
        {#if session}
          <p class="muted small">
            {fmt(session.startedAt, { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' })}
            · {fmt(session.startedAt, { hour: 'numeric', minute: '2-digit' })}
            {#if duration()} · {duration()}{/if}
            · #{sessionId}
          </p>
        {/if}
      </div>
      <a
        class="btn"
        href="/api/campaigns/{campaignId}/sessions/{sessionId}/export"
        download
        title="Download transcript as .txt"
      >&darr; Export</a>
    </div>
  {/snippet}

  {#snippet panelTop()}
    <div class="panel-group">
      <h3>Speakers <span class="count">({speakers.length})</span></h3>
      {#if speakers.length}
        <ul class="speakers">
          {#each speakers as [name, n]}
            <li><span>{name}</span><span class="subtle">{n}</span></li>
          {/each}
        </ul>
      {:else}
        <p class="subtle small">No lines.</p>
      {/if}
    </div>
  {/snippet}
</TranscriptView>

<style>
  .session-head { display: flex; justify-content: space-between; align-items: flex-end; gap: 16px; flex-wrap: wrap; }
  .crumbs a { color: inherit; }
  .crumbs { margin-bottom: 8px; }
  .title { font-size: 25px; letter-spacing: -0.5px; line-height: 1.2; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .title .badge:empty { display: none; }
  .speakers { list-style: none; display: flex; flex-direction: column; gap: 6px; font-size: 14px; }
  .speakers li { display: flex; justify-content: space-between; gap: 8px; }
</style>
