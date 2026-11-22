<script lang="ts">
  // Upload tracks into the server's music folder. Files come from the
  // pickers here, a drop on this dialog, or a drop anywhere on the DJ page
  // (the page calls addFiles). Before uploading it asks where they go: an
  // existing folder or a new one.
  //
  // Closing the dialog doesn't stop an upload in progress; the page shows a
  // progress chip (bound `uploading`) that reopens it.
  import { untrack } from 'svelte';
  import Modal from '$lib/components/ui/Modal.svelte';
  import Toggle from '$lib/components/ui/Toggle.svelte';
  import { dj, UploadError } from '$lib/api';
  import { addToast } from '$lib/stores/toast';
  import {
    type PickedFile, fromDrop, fromFileList, isAudioName, folderName, folderDepth, fmtBytes, AUDIO_EXTS,
  } from './library';

  let {
    open = $bindable(false),
    uploading = $bindable(false),
    progress = $bindable(0),
    folders,
    defaultFolder = '',
  }: {
    open?: boolean;
    uploading?: boolean;
    /** 0..1 over the whole batch, for the page's chip. */
    progress?: number;
    folders: string[];
    /** Folder selected on the page; the default destination for loose files. */
    defaultFolder?: string;
  } = $props();

  type Status = 'queued' | 'uploading' | 'done' | 'exists' | 'failed' | 'cancelled';
  interface Item extends PickedFile {
    target: string;
    status: Status;
    sent: number;
    error: string;
  }

  let picked = $state<PickedFile[]>([]);
  let ignored = $state<string[]>([]); // non-audio names, reported not uploaded
  let phase = $state<'pick' | 'uploading' | 'done'>('pick');
  let items = $state<Item[]>([]);
  let ctrl: AbortController | null = null;

  let mode = $state<'existing' | 'new'>('new');
  let existingFolder = $state('');
  let newName = $state('');
  let newParent = $state('');
  let keepRoot = $state(true);
  let overwrite = $state(false);
  let dragOver = $state(false);

  let fileInput: HTMLInputElement | undefined = $state();
  let folderInput: HTMLInputElement | undefined = $state();

  // When everything picked sits under one top-level folder (a dropped or
  // picked folder), that folder is the natural playlist: offer it as the new
  // folder's name, and when adding to an existing folder, ask whether to keep
  // it as a sub-folder.
  const commonRoot = $derived.by(() => {
    if (picked.length === 0) return '';
    const first = picked[0].rel.split('/')[0];
    return picked.every((p) => p.rel.includes('/') && p.rel.split('/')[0] === first) ? first : '';
  });

  const totalBytes = $derived(picked.reduce((s, p) => s + p.file.size, 0));

  const nameError = $derived.by(() => {
    if (mode !== 'new') return '';
    const n = newName.trim();
    if (!n) return 'Give the new folder a name';
    if (/[\\/<>:"|?*]/.test(n)) return 'Names can\'t contain / \\ < > : " | ? *';
    if (/[. ]$/.test(n)) return "Names can't end with a dot or space";
    return '';
  });
  const destPath = $derived(
    mode === 'new' ? [newParent, newName.trim()].filter(Boolean).join('/') : existingFolder,
  );
  const destExists = $derived(mode === 'new' && folders.includes(destPath));

  function targetFor(rel: string): string {
    const strip = commonRoot && (mode === 'new' || !keepRoot);
    const inside = strip ? rel.slice(commonRoot.length + 1) : rel;
    return [destPath, inside].filter(Boolean).join('/');
  }

  /** Add files (from the page-level drop). Ignored while uploading. */
  export function addFiles(files: PickedFile[]) {
    if (phase === 'uploading') {
      addToast('Wait for the current upload to finish', 'error');
      return;
    }
    if (phase === 'done') reset();
    const have = new Set(picked.map((p) => p.rel));
    const audio: PickedFile[] = [];
    for (const f of files) {
      if (!isAudioName(f.file.name)) {
        // Cover art, playlists and OS clutter are expected in music folders;
        // list them rather than fail.
        if (!f.file.name.startsWith('.') && f.file.name !== 'desktop.ini') ignored = [...ignored, f.rel];
        continue;
      }
      if (!have.has(f.rel)) {
        have.add(f.rel);
        audio.push(f);
      }
    }
    picked = [...picked, ...audio];
    // Sensible destination defaults for what was just picked.
    if (commonRoot) {
      mode = 'new';
      newName = commonRoot;
      newParent = '';
    } else if (defaultFolder) {
      mode = 'existing';
      existingFolder = defaultFolder;
    } else if (folders.length) {
      mode = 'new';
    }
  }

  function reset() {
    picked = [];
    ignored = [];
    items = [];
    phase = 'pick';
    newName = '';
    keepRoot = true;
    progress = 0;
  }

  // Fresh form when the dialog is reopened after a finished batch. Only on
  // the closed -> open transition: reacting to phase too would wipe the
  // results the moment a batch finishes.
  let wasOpen = false;
  $effect(() => {
    if (open && !wasOpen && untrack(() => phase) === 'done') reset();
    wasOpen = open;
  });

  async function onDrop(e: DragEvent) {
    e.preventDefault();
    e.stopPropagation(); // the page has its own drop handler
    dragOver = false;
    if (e.dataTransfer) addFiles(await fromDrop(e.dataTransfer));
  }

  function onPick(e: Event) {
    const input = e.currentTarget as HTMLInputElement;
    addFiles(fromFileList(input.files));
    input.value = ''; // so picking the same thing again still fires change
  }

  async function start() {
    if (picked.length === 0 || nameError) return;
    phase = 'uploading';
    uploading = true;
    ctrl = new AbortController();
    const signal = ctrl.signal;
    items = picked.map((p) => ({ ...p, target: targetFor(p.rel), status: 'queued', sent: 0, error: '' }));
    const total = Math.max(1, totalBytes);

    let next = 0;
    const one = async (it: Item) => {
      it.status = 'uploading';
      try {
        await dj.upload(it.target, it.file, {
          signal,
          overwrite,
          onProgress: (sent) => {
            it.sent = sent;
            progress = items.reduce((s, x) => s + x.sent, 0) / total;
          },
        });
        it.status = 'done';
        it.sent = it.file.size;
      } catch (err) {
        if ((err as Error).name === 'AbortError') it.status = 'cancelled';
        else if (err instanceof UploadError && err.status === 409) it.status = 'exists';
        else {
          it.status = 'failed';
          it.error = (err as Error).message;
        }
        it.sent = it.file.size; // count it as handled for the progress bar
      }
      progress = items.reduce((s, x) => s + x.sent, 0) / total;
    };
    // Two at a time: keeps the pipe full without making the server probe
    // a dozen files at once.
    const worker = async () => {
      while (next < items.length && !signal.aborted) await one(items[next++]);
    };
    await Promise.all([worker(), worker()]);
    for (const it of items) if (it.status === 'queued') it.status = 'cancelled';

    phase = 'done';
    progress = 1;
    uploading = false;
    ctrl = null;
    const n = (s: Status) => items.filter((i) => i.status === s).length;
    const parts = [`${n('done')} uploaded`];
    if (n('exists')) parts.push(`${n('exists')} already there`);
    if (n('failed')) parts.push(`${n('failed')} failed`);
    if (n('cancelled')) parts.push(`${n('cancelled')} cancelled`);
    addToast(`Upload finished: ${parts.join(', ')}`, n('failed') ? 'error' : 'success');
    // Close on its own unless something failed: then the per-file errors
    // stay on screen. The toast carries the summary either way.
    if (!n('failed')) open = false;
  }

  function cancel() {
    ctrl?.abort();
  }

  const counts = $derived({
    done: items.filter((i) => i.status === 'done').length,
    exists: items.filter((i) => i.status === 'exists').length,
    failed: items.filter((i) => i.status === 'failed').length,
  });

  const STATUS_LABEL: Record<Status, string> = {
    queued: 'Waiting', uploading: '', done: 'Done', exists: 'Already there', failed: 'Failed', cancelled: 'Cancelled',
  };

  const indent = (f: string) => '   '.repeat(Math.max(0, folderDepth(f) - 1));
</script>

<Modal bind:open title="Upload music" width="620px">
  {#if phase === 'pick'}
    <div
      class="drop"
      class:over={dragOver}
      role="region"
      aria-label="Drop zone"
      ondragover={(e) => { e.preventDefault(); e.stopPropagation(); dragOver = true; }}
      ondragleave={() => (dragOver = false)}
      ondrop={onDrop}
    >
      <p>Drop audio files or folders here</p>
      <div class="pick-btns">
        <button class="btn btn-sm" onclick={() => fileInput?.click()}>Choose files</button>
        <button class="btn btn-sm" onclick={() => folderInput?.click()}>Choose a folder</button>
      </div>
      <p class="tiny subtle">{AUDIO_EXTS.join(', ')}</p>
      <input bind:this={fileInput} type="file" multiple accept={AUDIO_EXTS.map((e) => '.' + e).join(',')} onchange={onPick} hidden />
      <input bind:this={folderInput} type="file" webkitdirectory onchange={onPick} hidden />
    </div>

    {#if picked.length > 0}
      <div class="summary">
        <strong>{picked.length} track{picked.length === 1 ? '' : 's'}</strong>
        <span class="subtle">{fmtBytes(totalBytes)}{commonRoot ? ` from "${commonRoot}"` : ''}</span>
        <button class="btn-ghost btn-sm" onclick={reset}>Clear</button>
      </div>

      <fieldset class="dest">
        <legend class="field-label">Where should they go?</legend>
        <label class="radio">
          <input type="radio" bind:group={mode} value="new" />
          <span>New playlist folder</span>
        </label>
        {#if mode === 'new'}
          <div class="sub">
            <label class="field"><span>Name</span>
              <input class="input" bind:value={newName} placeholder="Boss fights" />
            </label>
            {#if folders.length}
              <label class="field"><span>Inside</span>
                <select class="select" bind:value={newParent}>
                  <option value="">Music (top level)</option>
                  {#each folders as f (f)}<option value={f}>{indent(f)}{folderName(f)}</option>{/each}
                </select>
              </label>
            {/if}
            {#if nameError && newName}<p class="tiny err">{nameError}</p>{/if}
            {#if destExists}<p class="tiny subtle">That folder exists; the tracks will be added to it.</p>{/if}
          </div>
        {/if}

        <label class="radio">
          <input type="radio" bind:group={mode} value="existing" />
          <span>Existing folder</span>
        </label>
        {#if mode === 'existing'}
          <div class="sub">
            <select class="select" bind:value={existingFolder} aria-label="Destination folder">
              <option value="">Music (top level)</option>
              {#each folders as f (f)}<option value={f}>{indent(f)}{folderName(f)}</option>{/each}
            </select>
            {#if commonRoot}
              <Toggle bind:checked={keepRoot} label={`Keep "${commonRoot}" as a sub-folder`}
                hint={keepRoot ? `Tracks go in ${[existingFolder, commonRoot].filter(Boolean).join('/')}` : 'Tracks go straight into the chosen folder'} />
            {:else if existingFolder === ''}
              <p class="tiny subtle">Tracks at the top level aren't in any playlist folder, so no scene can pick them.</p>
            {/if}
          </div>
        {/if}
      </fieldset>

      <Toggle bind:checked={overwrite} label="Replace tracks that already exist" hint="Off: files with the same name are skipped" />

      <details class="preview">
        <summary class="tiny subtle">Show where each file goes</summary>
        <ul class="files">
          {#each picked.slice(0, 200) as p (p.rel)}<li class="tiny">{targetFor(p.rel)}</li>{/each}
          {#if picked.length > 200}<li class="tiny subtle">and {picked.length - 200} more</li>{/if}
        </ul>
      </details>
    {/if}

    {#if ignored.length}
      <p class="tiny subtle">Not audio, won't be uploaded: {ignored.slice(0, 5).join(', ')}{ignored.length > 5 ? ` and ${ignored.length - 5} more` : ''}</p>
    {/if}
  {:else}
    <div class="progress-head">
      <span>
        {phase === 'uploading' ? 'Uploading' : 'Finished'}: {counts.done + counts.exists + counts.failed} of {items.length} handled
      </span>
      <span class="subtle tiny">{Math.round(progress * 100)}%</span>
    </div>
    <div class="bar" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow={Math.round(progress * 100)}>
      <div style:width="{progress * 100}%"></div>
    </div>
    <ul class="files status-list">
      {#each items as it (it.rel)}
        <li class={it.status}>
          <span class="name tiny">{it.target}</span>
          <span class="st tiny">
            {#if it.status === 'uploading'}{Math.round((it.sent / Math.max(1, it.file.size)) * 100)}%{:else}{STATUS_LABEL[it.status]}{/if}
          </span>
          {#if it.error}<span class="tiny err full">{it.error}</span>{/if}
        </li>
      {/each}
    </ul>
  {/if}

  {#snippet footer()}
    {#if phase === 'pick'}
      <span class="spacer"></span>
      <button class="btn" onclick={() => (open = false)}>Cancel</button>
      <button class="btn-primary" onclick={start} disabled={picked.length === 0 || !!nameError}>
        Upload{picked.length ? ` ${picked.length}` : ''}
      </button>
    {:else if phase === 'uploading'}
      <button class="btn-ghost" onclick={cancel}>Stop uploading</button>
      <span class="spacer"></span>
      <button class="btn" onclick={() => (open = false)}>Hide</button>
    {:else}
      <span class="spacer"></span>
      <button class="btn" onclick={reset}>Upload more</button>
      <button class="btn-primary" onclick={() => (open = false)}>Done</button>
    {/if}
  {/snippet}
</Modal>

<style>
  .drop {
    border: 2px dashed var(--border-strong); border-radius: var(--radius-lg); padding: 22px 16px;
    display: flex; flex-direction: column; align-items: center; gap: 10px; text-align: center; color: var(--text-2);
    transition: background 0.15s, border-color 0.15s;
  }
  .drop.over { border-color: var(--accent); background: var(--accent-soft, var(--surface-hover)); }
  .drop p { margin: 0; }
  .pick-btns { display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; }

  .summary { display: flex; align-items: baseline; gap: 10px; margin: 16px 0 6px; }
  .summary .btn-ghost { margin-left: auto; }

  .dest { border: 0; padding: 0; margin: 10px 0 14px; display: flex; flex-direction: column; gap: 8px; }
  .dest legend { margin-bottom: 8px; font-weight: 600; }
  .radio { display: flex; align-items: center; gap: 10px; min-height: var(--touch); font-weight: 600; }
  .radio input { width: 20px; height: 20px; accent-color: var(--accent); }
  .sub { padding-left: 30px; display: flex; flex-direction: column; gap: 10px; }

  .preview { margin-top: 10px; }
  .preview summary { cursor: pointer; }
  .files { list-style: none; margin: 8px 0 0; padding: 0; max-height: 220px; overflow-y: auto; font-family: var(--font-mono); }
  .files li { padding: 3px 0; word-break: break-all; }

  .progress-head { display: flex; justify-content: space-between; margin-bottom: 8px; }
  .bar { height: 8px; border-radius: 4px; background: var(--surface-raised); overflow: hidden; }
  .bar > div { height: 100%; background: var(--accent); transition: width 0.2s; }
  .status-list { max-height: 320px; margin-top: 14px; }
  .status-list li { display: flex; flex-wrap: wrap; gap: 4px 10px; border-bottom: 1px solid var(--border-color); padding: 6px 0; }
  .status-list .name { flex: 1; min-width: 0; }
  .status-list .st { color: var(--text-2); }
  .status-list .done .st { color: var(--success); }
  .status-list .failed .st, .err { color: var(--danger, var(--warning)); }
  .status-list .full { width: 100%; }
  .spacer { flex: 1; }
</style>
