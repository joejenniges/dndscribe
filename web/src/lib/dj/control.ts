// Live DJ controls over the WebSocket (see docs/dj.md).
//
// Every user action is one command with a fresh id; the server answers the
// sender with dj.ack, and every device gets the new full state. Commands are
// never queued while the socket is down: a scene change or sound effect
// arriving a minute late would be worse than not at all.
import { send, subscribe } from '$lib/ws';

type Pending = { resolve: () => void; reject: (e: Error) => void; timer: ReturnType<typeof setTimeout> };
const pending = new Map<string, Pending>();

subscribe((msg) => {
  if (msg?.type === 'dj.ack') {
    const p = pending.get(msg.id);
    if (!p) return;
    pending.delete(msg.id);
    clearTimeout(p.timer);
    if (msg.ok) p.resolve();
    else p.reject(new Error(msg.error || 'The server refused that'));
  } else if (msg?.type === 'ws_closed') {
    // Whatever was in flight may or may not have run; say so rather than
    // leaving buttons spinning.
    for (const [id, p] of pending) {
      clearTimeout(p.timer);
      p.reject(new Error('Lost the connection to the server'));
      pending.delete(id);
    }
  }
});

// Not crypto.randomUUID: it's missing on plain-http LAN addresses.
let seq = 0;
const newId = () => `${Date.now().toString(36)}-${(seq++).toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

/** Sends one DJ command; resolves when the server has applied it. */
export function djCmd(cmd: string, args: Record<string, unknown> = {}, timeoutMs = 10_000): Promise<void> {
  const id = newId();
  if (!send({ type: 'dj.cmd', id, cmd, args })) {
    return Promise.reject(new Error('Not connected to the server'));
  }
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      pending.delete(id);
      reject(new Error('No answer from the server'));
    }, timeoutMs);
    pending.set(id, { resolve, reject, timer });
  });
}

/** Fire-and-forget form for fader steps: errors are reported once, by the
 *  final value's command. */
export function djSend(cmd: string, args: Record<string, unknown> = {}): boolean {
  return send({ type: 'dj.cmd', id: newId(), cmd, args });
}

/**
 * A throttled sender for a continuous control (fader, knob): at most one
 * command per `interval` ms while it moves, and always the final value.
 */
export function throttled(cmd: string, argsFor: (v: number) => Record<string, unknown>, interval = 33) {
  let last = 0;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let latest: number | null = null;
  const flush = () => {
    timer = undefined;
    if (latest === null) return;
    last = Date.now();
    djSend(cmd, argsFor(latest));
    latest = null;
  };
  return {
    /** While dragging. */
    input(v: number) {
      latest = v;
      const wait = interval - (Date.now() - last);
      if (wait <= 0) flush();
      else if (!timer) timer = setTimeout(flush, wait);
    },
    /** On release: the final value, acknowledged. */
    commit(v: number): Promise<void> {
      clearTimeout(timer);
      timer = undefined;
      latest = null;
      last = Date.now();
      return djCmd(cmd, argsFor(v));
    },
  };
}
