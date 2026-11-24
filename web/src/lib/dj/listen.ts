// Hear the test channel in this browser.
//
// While the bot is in the test channel the server sends the mixer's output
// (one 20 ms Opus frame per binary WebSocket message, 48 kHz stereo: what
// Discord would get) to /api/dj/listen. Frames are decoded here and played
// through Web Audio with a small jitter buffer.
//
// WHY opus-decoder (WASM) and not WebCodecs' AudioDecoder: WebCodecs only
// exists in secure contexts, and tablets reach this app over plain http on
// the LAN, where it would be missing. The WASM decoder works everywhere.
//
// Listening is per device (a tablet and a laptop in the same room would
// otherwise echo). The choice is remembered; browsers only start audio
// after a tap or key press, so a remembered "on" waits for the first one.
import { writable, get } from 'svelte/store';
import type { OpusDecoder as OpusDecoderT } from 'opus-decoder';

export type ListenState = 'off' | 'connecting' | 'blocked' | 'on' | 'error' | 'elsewhere' | 'unreachable';

const PREF = 'dj.listen';
/** Whether this device wants to hear the test channel (remembered). */
export const listenWanted = writable<boolean>(readPref());
export const listenState = writable<ListenState>('off');

function readPref(): boolean {
  try {
    return localStorage.getItem(PREF) === '1';
  } catch {
    return false;
  }
}
function writePref(on: boolean) {
  try {
    localStorage.setItem(PREF, on ? '1' : '0');
  } catch {
    /* private mode: just not remembered */
  }
}

// Jitter buffer: start playback LEAD behind the first frame; if frames pile
// up past MAX_AHEAD (browser clock slower than the server's), drop until
// back in range, so latency can't creep up over a long session.
const LEAD = 0.12;
const MAX_AHEAD = 0.35;
const RATE = 48000;

// One tab per browser. The "on" choice is in localStorage, so every DJ tab
// in a browser would otherwise play the stream, a few tens of ms apart,
// which sounds hollow and metallic (comb filtering). The tab that starts
// playing tells the others to stop (without changing the saved choice).
const tabId = Math.random().toString(36).slice(2);
const channel = typeof BroadcastChannel !== 'undefined' ? new BroadcastChannel('dj-listen') : null;
channel?.addEventListener('message', (e) => {
  if (e.data?.type === 'playing' && e.data.tab !== tabId && running) {
    stopListening(false);
    listenState.set('elsewhere');
  }
});

let ctx: AudioContext | null = null;
let gain: GainNode | null = null;
let decoder: OpusDecoderT<48000> | null = null;
let ws: WebSocket | null = null;
let next = 0;
let running = false;
// Bumped by every start and stop. A start that awaited (decoder load,
// resume) checks it afterwards and gives up if anything happened meanwhile;
// without this, start/stop/start in quick succession opened two sockets and
// every frame played twice.
let gen = 0;
let retry: ReturnType<typeof setTimeout> | null = null;
let backoff = 500;
// Socket attempts that closed without ever opening. After a few, say so
// ('unreachable') instead of showing "connecting" forever; retries go on.
let failures = 0;

function url(): string {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${location.host}/api/dj/listen`;
}

/** Turn listening on for this device. Call from a click so audio can start. */
export async function startListening(remember = true): Promise<void> {
  if (remember) {
    writePref(true);
    listenWanted.set(true);
  }
  if (running) {
    await resume();
    return;
  }
  running = true;
  const my = ++gen;
  listenState.set('connecting');
  try {
    if (!decoder) {
      const { OpusDecoder } = await import('opus-decoder');
      const d = new OpusDecoder({ channels: 2, sampleRate: 48000 as const });
      await d.ready;
      decoder = d;
    }
    if (!ctx) {
      ctx = new AudioContext({ sampleRate: RATE, latencyHint: 'interactive' });
      gain = ctx.createGain();
      gain.connect(ctx.destination);
    }
  } catch (e) {
    console.error('listen: audio setup failed', e);
    running = false;
    listenState.set('error');
    return;
  }
  if (my !== gen) return; // stopped (or restarted) while loading
  await resume();
  if (my !== gen) return;
  open();
}

/** Turn listening off for this device (remembered unless remember=false). */
export function stopListening(remember = true): void {
  if (remember) {
    writePref(false);
    listenWanted.set(false);
  }
  running = false;
  gen++;
  if (retry) clearTimeout(retry);
  retry = null;
  ws?.close();
  ws = null;
  next = 0;
  failures = 0;
  if (ctx && ctx.state === 'running') ctx.suspend().catch(() => {});
  removeUnlock();
  listenState.set('off');
}

async function resume() {
  if (!ctx) return;
  if (ctx.state !== 'running') {
    try {
      await ctx.resume();
    } catch {
      /* still blocked */
    }
  }
  if (ctx.state === 'running') {
    channel?.postMessage({ type: 'playing', tab: tabId });
    removeUnlock();
    if (get(listenState) === 'blocked') listenState.set(ws?.readyState === WebSocket.OPEN ? 'on' : 'connecting');
  } else {
    // No user gesture yet (a remembered "on" after a reload): start on the
    // first tap or key press anywhere on the page.
    listenState.set('blocked');
    addUnlock();
  }
}

const unlock = () => {
  void resume();
};
function addUnlock() {
  document.addEventListener('pointerdown', unlock, true);
  document.addEventListener('keydown', unlock, true);
}
function removeUnlock() {
  document.removeEventListener('pointerdown', unlock, true);
  document.removeEventListener('keydown', unlock, true);
}

function open() {
  if (!running) return;
  ws?.close();
  const sock = new WebSocket(url());
  sock.binaryType = 'arraybuffer';
  ws = sock;
  let opened = false;
  sock.onopen = () => {
    opened = true;
    failures = 0;
    backoff = 500;
    if (get(listenState) !== 'blocked') listenState.set('on');
  };
  sock.onmessage = (e) => {
    if (e.data instanceof ArrayBuffer) play(new Uint8Array(e.data));
  };
  sock.onclose = () => {
    if (ws !== sock) return;
    ws = null;
    if (!running) return;
    if (!opened) failures++;
    if (get(listenState) !== 'blocked') listenState.set(failures >= 3 ? 'unreachable' : 'connecting');
    retry = setTimeout(open, backoff);
    backoff = Math.min(backoff * 2, 8000);
  };
}

function play(frame: Uint8Array) {
  if (!ctx || !gain || !decoder || ctx.state !== 'running') return;
  let out;
  try {
    out = decoder.decodeFrame(frame);
  } catch {
    return;
  }
  const n = out.samplesDecoded;
  if (!n) return;
  const now = ctx.currentTime;
  // Start, or the stream paused (the engine sends nothing while silent) or
  // fell behind: rebuild the buffer rather than play late.
  if (next < now + 0.01) next = now + LEAD;
  else if (next > now + MAX_AHEAD) return;
  const buf = ctx.createBuffer(2, n, RATE);
  // (cast: the decoder's arrays are typed ArrayBufferLike; they're plain.)
  const left = out.channelData[0] as Float32Array<ArrayBuffer>;
  buf.copyToChannel(left, 0);
  buf.copyToChannel((out.channelData[1] ?? left) as Float32Array<ArrayBuffer>, 1);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(gain);
  src.start(next);
  next += n / RATE;
}
