// Keyboard shortcuts for DJ Studio. One window-level handler, installed by
// the DJ page only. The list here is also what the help dialog shows.
//
// Tab is taken over to switch the selected deck, so on this page Tab no
// longer moves focus. That is a deliberate trade for live use; shortcuts can
// be switched off per device (the help dialog), which gives Tab back.
//
// Keys are ignored while typing (text fields, selects), while a dialog is
// open, and with Ctrl/Alt/Cmd held, so browser shortcuts keep working.
// On a focused fader or waveform the arrow keys stay with that control.
import { writable, get } from 'svelte/store';
import { djCmd } from './control';
import { crossfaderNow, local, DECK_NAMES } from './live';
import { addToast } from '$lib/stores/toast';
import type { DjState } from '$lib/types';

/** The deck Space, X and R act on. Per device, not shared. */
export const selectedDeck = writable<0 | 1>(0);

export const hotkeysOn = writable<boolean>(local.get('dj.hotkeys') !== 'off');
hotkeysOn.subscribe((on) => local.set('dj.hotkeys', on ? 'on' : 'off'));

export const HOTKEYS: { keys: string; what: string }[] = [
  { keys: 'Tab', what: 'Select the other deck' },
  { keys: 'Space', what: 'Play / pause the selected deck' },
  { keys: 'X', what: 'Crossfade to the selected deck (global fade time)' },
  { keys: 'R', what: 'Selected deck back to the start' },
  { keys: 'N', what: 'Next track (crossfade from the queue)' },
  { keys: 'A', what: 'Auto DJ on / off' },
  { keys: '← →', what: 'Nudge the crossfader toward A / B (Shift: all the way)' },
  { keys: '↑ ↓', what: 'Master volume up / down' },
  { keys: '1 – 9, 0', what: 'Sound effects 1 – 10 in the open soundboard tab' },
  { keys: 'Shift + 1 – 9', what: 'Play scene 1 – 9' },
  { keys: 'Esc', what: 'Stop sound effects' },
  { keys: 'Shift + Esc', what: 'Stop all (music and effects)' },
  { keys: '?', what: 'Show these shortcuts' },
];

export interface HotkeyTargets {
  state: () => DjState | null;
  live: () => boolean;
  scenes: () => { id: number; folders: string[] }[];
  /** Triggers pad i (0-based) of the visible soundboard tab; false if none. */
  pad: (i: number) => boolean;
  help: () => void;
}

const typing = (el: Element | null) =>
  !!el &&
  (el instanceof HTMLTextAreaElement ||
    el instanceof HTMLSelectElement ||
    (el instanceof HTMLInputElement && !['range', 'checkbox', 'radio', 'button'].includes(el.type)) ||
    (el as HTMLElement).isContentEditable);

const ownsArrows = (el: Element | null) =>
  !!el &&
  ((el instanceof HTMLInputElement && el.type === 'range') ||
    ['slider', 'separator'].includes(el.getAttribute('role') ?? ''));

function run(cmd: string, args: Record<string, unknown> = {}) {
  djCmd(cmd, args).catch((e) => addToast((e as Error).message, 'error'));
}

const clamp01 = (v: number) => Math.max(0, Math.min(1, v));

export function installHotkeys(t: HotkeyTargets): () => void {
  function onKey(e: KeyboardEvent) {
    if (!get(hotkeysOn) || e.ctrlKey || e.metaKey || e.altKey) return;
    if (document.querySelector('dialog[open]')) return;
    const el = document.activeElement;
    if (typing(el)) return;

    const handled = (fn: () => void, allowRepeat = false) => {
      e.preventDefault();
      e.stopPropagation();
      if (e.repeat && !allowRepeat) return;
      fn();
    };

    if (e.key === '?') return handled(t.help);
    if (e.key === 'Tab') {
      // Drop focus from whatever was clicked last, so Space can't also
      // "press" that button.
      if (el instanceof HTMLElement && el !== document.body) el.blur();
      return handled(() => selectedDeck.update((d) => (d ? 0 : 1)));
    }

    const live = t.live();
    const st = t.state();
    const deck = get(selectedDeck);
    const d = st?.decks?.[deck];
    const needLive = (fn: () => void, allowRepeat = false) =>
      handled(() => (live ? fn() : addToast('Not connected to the server', 'error')), allowRepeat);

    // Digits by physical key, so Shift+1 is "1" on any layout.
    const digit = /^Digit(\d)$/.exec(e.code)?.[1];
    if (digit !== undefined) {
      const n = digit === '0' ? 9 : Number(digit) - 1;
      if (e.shiftKey) {
        const s = t.scenes()[n];
        if (s && digit !== '0') return needLive(() => run('scene.play', { sceneId: s.id }));
        return;
      }
      return needLive(() => void t.pad(n));
    }

    switch (e.key) {
      case ' ':
        return needLive(() => {
          if (!d?.track) return addToast(`Deck ${DECK_NAMES[deck]} is empty`, 'info');
          run(d.playing ? 'deck.pause' : 'deck.play', { deck });
        });
      case 'x':
      case 'X':
        return needLive(() => run('mix.fadeTo', { deck }));
      case 'r':
      case 'R':
        if (!d?.track) return;
        return needLive(() => run('deck.seek', { deck, positionMs: 0 }));
      case 'n':
      case 'N':
        return needLive(() => run('mix.next'));
      case 'a':
      case 'A':
        return needLive(() => run('mix.autoDJ', { on: !st?.autoDJ }));
      case 'Escape':
        return needLive(() => run(e.shiftKey ? 'mix.stopAll' : 'sfx.stop'));
      case 'ArrowLeft':
      case 'ArrowRight': {
        if (ownsArrows(el)) return;
        const dir = e.key === 'ArrowLeft' ? -1 : 1;
        return needLive(() => {
          const v = e.shiftKey ? (dir < 0 ? 0 : 1) : clamp01(Math.round((crossfaderNow(st) + dir * 0.05) * 100) / 100);
          run('mix.crossfader', { value: v });
        }, true);
      }
      case 'ArrowUp':
      case 'ArrowDown': {
        if (ownsArrows(el)) return;
        const dir = e.key === 'ArrowDown' ? -1 : 1;
        return needLive(() => run('mix.master', { value: clamp01(Math.round(((st?.volume ?? 0.5) + dir * 0.05) * 100) / 100) }), true);
      }
    }
  }
  // Capture phase: runs before a focused button turns Space into a click.
  window.addEventListener('keydown', onKey, true);
  return () => window.removeEventListener('keydown', onKey, true);
}
