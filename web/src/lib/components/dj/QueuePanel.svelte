<script lang="ts">
  // The shared queue (what Auto DJ plays next) and the history of what
  // played. Shuffle and repeat belong to the queue: a scene sets them when
  // it fills the queue, and they can be changed here.
  import { djCmd } from '$lib/dj/control';
  import { addToast } from '$lib/stores/toast';
  import { local } from '$lib/dj/live';
  import type { DjState, DjTrackInfo } from '$lib/types';
  import { fmtMs } from './format';

  let {
    st,
    live,
    collapsed = $bindable(false),
  }: { st: DjState | null; live: boolean; collapsed?: boolean } = $props();

  let tab = $state<'queue' | 'history'>(local.get('dj.queueTab') === 'history' ? 'history' : 'queue');
  function setTab(t: 'queue' | 'history') {
    tab = t;
    local.set('dj.queueTab', t);
  }

  const queue = $derived(st?.queue ?? []);
  const history = $derived(st?.history ?? []);
  const inVoice = $derived(!!st?.connected);

  async function run(cmd: string, args: Record<string, unknown> = {}) {
    try {
      await djCmd(cmd, args);
    } catch (e) {
      addToast((e as Error).message, 'error');
    }
  }

  // Two-tap clear, like deletes elsewhere: no modal on the tablet.
  let armClear = $state(false);
  let armTimer: ReturnType<typeof setTimeout> | undefined;
  function clear() {
    if (!armClear) {
      armClear = true;
      clearTimeout(armTimer);
      armTimer = setTimeout(() => (armClear = false), 4000);
      return;
    }
    armClear = false;
    run('queue.clear');
  }

  const ago = (ms: number) => {
    const m = Math.round((Date.now() - ms) / 60000);
    return m < 1 ? 'just now' : m < 60 ? `${m} min ago` : `${Math.round(m / 60)} h ago`;
  };
  const sub = (t: DjTrackInfo) => [t.artist, t.durationMs ? fmtMs(t.durationMs) : ''].filter(Boolean).join(' · ');
</script>

<section class="qp card" class:stale={!live} class:collapsed aria-label="Queue and history">
  <header>
    <button class="icon-btn fold" onclick={() => (collapsed = !collapsed)} aria-expanded={!collapsed}
      aria-label={collapsed ? 'Show the queue' : 'Hide the queue'} title={collapsed ? 'Show the queue' : 'Hide the queue'}>
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 9l6 6 6-6" /></svg>
    </button>
    <div class="tabs" role="tablist">
      <button role="tab" class:active={tab === 'queue'} aria-selected={tab === 'queue'} onclick={() => setTab('queue')}>
        Queue <span class="count">{st?.queueTotal ?? 0}</span>
      </button>
      <button role="tab" class:active={tab === 'history'} aria-selected={tab === 'history'} onclick={() => setTab('history')}>
        History
      </button>
    </div>
    {#if tab === 'queue'}
      <span class="source tiny subtle" title={st?.sourceName}>{st?.sourceName ? `From ${st.sourceName}` : ''}</span>
      <div class="tools">
        <button class="icon-btn toggle" class:on={st?.shuffle} aria-pressed={!!st?.shuffle} disabled={!live}
          onclick={() => run('queue.shuffle', { on: !st?.shuffle })} title="Shuffle" aria-label="Shuffle">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M16 3h5v5M4 20 21 3M21 16v5h-5M15 15l6 6M4 4l5 5" /></svg>
        </button>
        <button class="icon-btn toggle" class:on={st?.loop} aria-pressed={!!st?.loop} disabled={!live}
          onclick={() => run('queue.loop', { on: !st?.loop })} title="Repeat: refill the queue when it runs out" aria-label="Repeat">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M17 1l4 4-4 4M3 11V9a4 4 0 0 1 4-4h14M7 23l-4-4 4-4M21 13v2a4 4 0 0 1-4 4H3" /></svg>
        </button>
        <button class="btn btn-sm" class:btn-danger={armClear} onclick={clear} disabled={!live || queue.length === 0}>
          {armClear ? 'Clear all?' : 'Clear'}
        </button>
      </div>
    {/if}
  </header>

  {#if !collapsed}
  <div class="list">
    {#if tab === 'queue'}
      {#if queue.length === 0}
        <p class="empty subtle small">
          {st?.loop && st?.sourceName ? 'The queue refills from the scene when Auto DJ needs the next track.' : 'Nothing queued. Start a scene, play a folder, or add tracks from the library.'}
        </p>
      {:else}
        <ol>
          {#each queue as q, i (q.qid)}
            <li>
              <span class="num tiny subtle">{i + 1}</span>
              <span class="main">
                <span class="t">{q.title}</span>
                <span class="tiny subtle">{sub(q)}</span>
              </span>
              <span class="acts">
                <button class="icon-btn" onclick={() => run('queue.move', { qid: q.qid, to: i - 1 })} disabled={!live || i === 0}
                  aria-label="Move {q.title} up">&#9650;</button>
                <button class="icon-btn" onclick={() => run('queue.move', { qid: q.qid, to: i + 1 })} disabled={!live || i === queue.length - 1}
                  aria-label="Move {q.title} down">&#9660;</button>
                <button class="chip-btn a" onclick={() => run('deck.load', { deck: 0, trackId: q.id })} disabled={!live}
                  aria-label="Load {q.title} on deck A">A</button>
                <button class="chip-btn b" onclick={() => run('deck.load', { deck: 1, trackId: q.id })} disabled={!live}
                  aria-label="Load {q.title} on deck B">B</button>
                <button class="icon-btn" onclick={() => run('queue.remove', { qid: q.qid })} disabled={!live}
                  aria-label="Remove {q.title} from the queue">&#10005;</button>
              </span>
            </li>
          {/each}
        </ol>
        {#if (st?.queueTotal ?? 0) > queue.length}
          <p class="tiny subtle more">and {(st?.queueTotal ?? 0) - queue.length} more</p>
        {/if}
      {/if}
    {:else if history.length === 0}
      <p class="empty subtle small">Tracks appear here after they've played.</p>
    {:else}
      <ol>
        {#each history as h (h.playedAt + ':' + h.id)}
          <li>
            <span class="main">
              <span class="t">{h.title}</span>
              <span class="tiny subtle">{[h.artist, ago(h.playedAt)].filter(Boolean).join(' · ')}</span>
            </span>
            <span class="acts">
              <button class="btn btn-sm" onclick={() => run('queue.add', { trackId: h.id, next: true })} disabled={!live}
                title="Play it again after the current track">Queue next</button>
              <button class="chip-btn a" onclick={() => run('deck.load', { deck: 0, trackId: h.id })} disabled={!live}
                aria-label="Load {h.title} on deck A">A</button>
              <button class="chip-btn b" onclick={() => run('deck.load', { deck: 1, trackId: h.id })} disabled={!live}
                aria-label="Load {h.title} on deck B">B</button>
            </span>
          </li>
        {/each}
      </ol>
    {/if}
  </div>
  {#if !inVoice && tab === 'queue' && queue.length > 0}
    <p class="tiny subtle foot">The bot isn't in a voice channel, so nothing will play until it joins.</p>
  {/if}
  {/if}
</section>

<style>
  .fold { width: 32px; height: 32px; flex-shrink: 0; }
  .fold svg { width: 18px; height: 18px; fill: none; stroke: currentColor; stroke-width: 2.2; stroke-linecap: round; stroke-linejoin: round; transition: transform 0.15s; }
  .collapsed .fold svg { transform: rotate(-90deg); }
  .collapsed header { border-bottom: 0; }
  .qp { display: flex; flex-direction: column; min-height: 0; overflow: hidden; }
  .stale { opacity: 0.6; }
  header { display: flex; align-items: center; gap: 10px; padding: 6px 10px 6px 6px; border-bottom: 1px solid var(--border-color); flex-wrap: wrap; }
  .tabs { display: flex; gap: 2px; }
  .tabs button {
    min-height: 40px; padding: 0 12px; background: none; border: 0; border-radius: var(--radius-sm);
    color: var(--text-2); font-weight: 600; font-size: 14px;
  }
  .tabs button.active { color: var(--text); background: var(--surface-raised); }
  .count { color: var(--text-3); font-weight: 500; margin-left: 4px; }
  .source { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .tools { display: flex; gap: 4px; align-items: center; margin-left: auto; }
  .toggle svg { width: 18px; height: 18px; fill: none; stroke: currentColor; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
  .toggle { width: 40px; height: 40px; color: var(--text-3); }
  .toggle.on { color: var(--accent-text, var(--accent)); background: var(--accent-soft); }
  .list { flex: 1; min-height: 0; overflow-y: auto; }
  ol { list-style: none; margin: 0; padding: 0; }
  li { display: flex; align-items: center; gap: 10px; padding: 4px 10px; min-height: 48px; border-bottom: 1px solid var(--border-color); }
  li:last-child { border-bottom: 0; }
  .num { width: 20px; text-align: right; flex-shrink: 0; font-variant-numeric: tabular-nums; }
  .main { flex: 1; min-width: 0; display: flex; flex-direction: column; }
  .t { font-weight: 600; font-size: 14px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .main .tiny { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .acts { display: flex; gap: 4px; align-items: center; flex-shrink: 0; }
  .acts .icon-btn { width: 36px; height: 36px; font-size: 12px; color: var(--text-2); }
  .chip-btn {
    width: 36px; height: 36px; border-radius: 7px; font-weight: 800; font-size: 13px; color: var(--text);
    background: var(--surface-raised); border: 1px solid var(--border-strong);
  }
  .chip-btn.a { border-color: color-mix(in srgb, var(--deck-a) 60%, var(--border-strong)); color: var(--deck-a); }
  .chip-btn.b { border-color: color-mix(in srgb, var(--deck-b) 60%, var(--border-strong)); color: var(--deck-b); }
  .chip-btn:disabled { opacity: 0.4; }
  .empty { padding: 18px 14px; margin: 0; }
  .more { padding: 8px 14px; margin: 0; }
  .foot { padding: 8px 12px; margin: 0; border-top: 1px solid var(--border-color); color: var(--warning); }
</style>
