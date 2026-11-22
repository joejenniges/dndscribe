<script lang="ts">
  // Edit one soundboard pad: label, colour, image, volume, tab, delete.
  // The sound file stays linked whatever changes here.
  import Modal from '$lib/components/ui/Modal.svelte';
  import { dj } from '$lib/api';
  import { addToast } from '$lib/stores/toast';
  import type { SfxPad, SfxTab } from '$lib/types';
  import { SCENE_COLORS, fmtMs } from './format';

  let {
    open = $bindable(false),
    pad,
    tabs,
    onchanged,
  }: {
    open?: boolean;
    pad: SfxPad | null;
    tabs: SfxTab[];
    onchanged?: () => void;
  } = $props();

  let label = $state('');
  let color = $state('');
  let gain = $state(100);
  let tabId = $state(0);
  let saving = $state(false);
  let confirmDelete = $state(false);
  let imageBusy = $state(false);
  let imageInput: HTMLInputElement | undefined = $state();

  let seededFor: SfxPad | null | undefined;
  $effect(() => {
    if (!open || !pad) {
      seededFor = undefined;
      return;
    }
    if (seededFor?.id === pad.id) return;
    seededFor = pad;
    label = pad.label;
    color = pad.color;
    gain = Math.round(pad.gain * 100);
    tabId = pad.tabId;
    confirmDelete = false;
  });

  async function save() {
    if (!pad || !label.trim()) return;
    saving = true;
    try {
      await dj.sfx.pads.update(pad.id, { label: label.trim(), color, gain: gain / 100, tabId });
      open = false;
      onchanged?.();
    } catch (e) {
      addToast(`Couldn't save: ${(e as Error).message}`, 'error');
    } finally {
      saving = false;
    }
  }

  async function remove() {
    if (!pad) return;
    try {
      await dj.sfx.pads.delete(pad.id);
      open = false;
      onchanged?.();
    } catch (e) {
      addToast(`Couldn't delete: ${(e as Error).message}`, 'error');
    }
  }

  // Images save immediately (they're their own upload), unlike the form
  // fields, which wait for Save.
  async function pickImage(e: Event) {
    const input = e.currentTarget as HTMLInputElement;
    const file = input.files?.[0];
    input.value = '';
    if (!pad || !file) return;
    imageBusy = true;
    try {
      const { image } = await dj.sfx.pads.setImage(pad.id, file);
      pad = { ...pad, image };
      onchanged?.();
    } catch (err) {
      addToast(`Couldn't use that image: ${(err as Error).message}`, 'error');
    } finally {
      imageBusy = false;
    }
  }

  async function clearImage() {
    if (!pad) return;
    imageBusy = true;
    try {
      await dj.sfx.pads.clearImage(pad.id);
      pad = { ...pad, image: null };
      onchanged?.();
    } catch (err) {
      addToast(`Couldn't remove the image: ${(err as Error).message}`, 'error');
    } finally {
      imageBusy = false;
    }
  }
</script>

<Modal bind:open title="Edit sound" width="500px">
  {#if pad}
    <form class="form" onsubmit={(e) => { e.preventDefault(); save(); }}>
      <label class="field"><span>Name</span><input class="input" bind:value={label} required /></label>
      <p class="tiny subtle file">{pad.originalName} · {fmtMs(pad.durationMs)}</p>

      <div class="field">
        <span>Colour</span>
        <div class="swatches" role="radiogroup" aria-label="Pad colour">
          <button type="button" class="swatch none" class:on={color === ''} role="radio" aria-checked={color === ''}
            aria-label="Default" onclick={() => (color = '')}></button>
          {#each SCENE_COLORS as c}
            <button type="button" class="swatch" class:on={color === c} style:background={c}
              role="radio" aria-checked={color === c} aria-label={c} onclick={() => (color = c)}></button>
          {/each}
        </div>
      </div>

      <div class="field">
        <span>Image</span>
        <div class="image-row">
          {#if pad.image}
            <img src={dj.sfx.imageUrl(pad.image)} alt="" />
          {:else}
            <div class="noimg tiny subtle">None</div>
          {/if}
          <div class="image-btns">
            <button type="button" class="btn btn-sm" onclick={() => imageInput?.click()} disabled={imageBusy}>
              {pad.image ? 'Replace' : 'Upload image'}
            </button>
            {#if pad.image}
              <button type="button" class="btn-ghost btn-sm" onclick={clearImage} disabled={imageBusy}>Remove</button>
            {/if}
          </div>
          <input bind:this={imageInput} type="file" accept="image/png,image/jpeg,image/gif,image/webp" onchange={pickImage} hidden />
        </div>
        <span class="tiny subtle">PNG, JPEG, GIF or WebP, up to 5 MB.</span>
      </div>

      <label class="field">
        <span>Volume: {gain}%</span>
        <input type="range" min="0" max="100" bind:value={gain} />
      </label>

      {#if tabs.length > 1}
        <label class="field">
          <span>Tab</span>
          <select class="select" bind:value={tabId}>
            {#each tabs as t (t.id)}<option value={t.id}>{t.title}</option>{/each}
          </select>
        </label>
      {/if}
    </form>
  {/if}

  {#snippet footer()}
    {#if confirmDelete}
      <button class="btn-danger" onclick={remove}>Delete sound</button>
    {:else}
      <button class="btn-ghost" onclick={() => (confirmDelete = true)}>Delete</button>
    {/if}
    <span class="spacer"></span>
    <button class="btn" onclick={() => (open = false)}>Cancel</button>
    <button class="btn-primary" onclick={save} disabled={saving || !label.trim()}>{saving ? 'Saving…' : 'Save'}</button>
  {/snippet}
</Modal>

<style>
  .form { display: flex; flex-direction: column; gap: 14px; }
  .file { margin: -8px 0 0; word-break: break-all; }
  .swatches { display: flex; gap: 10px; flex-wrap: wrap; }
  .swatch { width: 36px; height: 36px; border-radius: 50%; border: 2px solid transparent; padding: 0; }
  .swatch.none { background: var(--surface-raised); border-color: var(--border-strong); }
  .swatch.on { border-color: var(--text); box-shadow: 0 0 0 2px var(--canvas) inset; }
  .image-row { display: flex; align-items: center; gap: 14px; }
  .image-row img, .noimg { width: 72px; height: 72px; border-radius: var(--radius); object-fit: cover; border: 1px solid var(--border-strong); }
  .noimg { display: grid; place-items: center; background: var(--surface-raised); }
  .image-btns { display: flex; gap: 8px; flex-wrap: wrap; }
  input[type='range'] { accent-color: var(--accent); height: 28px; }
  .spacer { flex: 1; }
</style>
