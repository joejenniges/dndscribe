import { writable } from 'svelte/store';

interface Toast {
  id: number;
  message: string;
  type: 'success' | 'error' | 'info';
}

let nextId = 0;

export const toasts = writable<Toast[]>([]);

export function addToast(message: string, type: Toast['type'] = 'info', durationMs = 3000): void {
  const id = nextId++;
  // Server error messages are lowercase by Go convention ("the bot is not in
  // a voice channel") and are often shown as-is, so capitalise here once
  // rather than at every call site.
  message = message.charAt(0).toUpperCase() + message.slice(1);
  toasts.update(t => [...t, { id, message, type }]);
  setTimeout(() => {
    toasts.update(t => t.filter(toast => toast.id !== id));
  }, durationMs);
}
