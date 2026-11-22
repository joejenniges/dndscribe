/** WebSocket singleton with auto-reconnect. */

type MessageHandler = (msg: any) => void;

let ws: WebSocket | null = null;
let handlers: MessageHandler[] = [];
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
let currentCampaignId: number | null = null;
let attempts = 0;
let hello: { device: string; page: string } | null = null;

function getWsUrl(): string {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  // /ws so the Vite dev proxy can route it; the Go server upgrades on any path.
  return `${proto}//${location.host}/ws`;
}

export function connect(): void {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
    return;
  }

  ws = new WebSocket(getWsUrl());

  ws.onopen = () => {
    attempts = 0;
    // Re-subscribe to campaign if we were tracking one
    if (currentCampaignId != null) {
      setCampaign(currentCampaignId);
    }
    if (hello) send({ type: 'hello', ...hello });
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      for (const handler of handlers) {
        handler(msg);
      }
    } catch {
      // Ignore non-JSON messages
    }
  };

  ws.onclose = () => {
    ws = null;
    // Synthetic event so UI can show a disconnected state; the server's
    // 'init' message on reconnect flips it back.
    for (const handler of handlers) handler({ type: 'ws_closed' });
    scheduleReconnect();
  };

  ws.onerror = () => {
    ws?.close();
  };
}

// Reconnect with exponential backoff (1s, 2s, 4s, capped at 10s) plus up to
// 30% jitter, so a restarted server isn't hit by every tablet at once.
function scheduleReconnect(): void {
  if (reconnectTimer) return;
  const base = Math.min(10_000, 1000 * 2 ** attempts);
  attempts++;
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    connect();
  }, base * (1 + Math.random() * 0.3));
}

export function disconnect(): void {
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
  if (ws) {
    ws.onclose = null; // Don't trigger reconnect
    ws.close();
    ws = null;
  }
}

export function subscribe(handler: MessageHandler): () => void {
  handlers.push(handler);
  return () => {
    handlers = handlers.filter(h => h !== handler);
  };
}

/** Sends a message now. false if the socket isn't open: callers must not
 *  queue live actions for later (a sound effect firing a minute late is
 *  worse than not at all). */
export function send(msg: unknown): boolean {
  if (!ws || ws.readyState !== WebSocket.OPEN) return false;
  ws.send(JSON.stringify(msg));
  return true;
}

export function setCampaign(campaignId: number): void {
  currentCampaignId = campaignId;
  send({ type: 'set_campaign', campaignId });
}

/** Tells the server what kind of device this is and which page it's on
 *  (shown to other devices as presence). Re-sent after every reconnect. */
export function sayHello(device: string, page: string): void {
  if (hello && hello.device === device && hello.page === page) return;
  hello = { device, page };
  send({ type: 'hello', ...hello });
}

export function isConnected(): boolean {
  return ws != null && ws.readyState === WebSocket.OPEN;
}
