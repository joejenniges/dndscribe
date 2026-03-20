import http from 'http';
import fs from 'fs';
import { readFileSync } from 'fs';
import path from 'path';
import { WebSocketServer, WebSocket } from 'ws';
import { logger } from './logger';
import { TranscriptionWorker, TranscriptionLine } from './transcription';
import { getHotwords, addHotword, removeHotword, updateHotword } from './hotwords';
import {
  listSessions, deleteSessionFull, mergeSessions, getSessionPreview,
  bulkUpdateNicknames, searchTranscriptions,
  getNicknamePresets as getPresetsDB,
  addNicknamePreset, updateNicknamePreset, deleteNicknamePreset, movePresetToCategory,
  getNicknameCategories, addNicknameCategory, updateNicknameCategory, deleteNicknameCategory,
} from './db';
import type { BotController } from './bot';

let server: http.Server | null = null;
let wss: WebSocketServer | null = null;

function broadcast(data: object): void {
  if (!wss) return;
  const msg = JSON.stringify(data);
  for (const client of wss.clients) {
    if (client.readyState === WebSocket.OPEN) {
      client.send(msg);
    }
  }
}

async function broadcastPresets(): Promise<void> {
  const presets = await getPresetsDB();
  const categories = await getNicknameCategories();
  broadcast({ type: 'presets_updated', presets, categories });
}

function readBody(req: http.IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', (chunk: Buffer) => { body += chunk.toString(); });
    req.on('end', () => resolve(body));
    req.on('error', reject);
  });
}

function json(res: http.ServerResponse, status: number, data: unknown): void {
  res.writeHead(status, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(data));
}

export function startWebUI(worker: TranscriptionWorker, controller: BotController, port: number): http.Server {
  const htmlPath = path.join(__dirname, '..', 'src', 'public', 'index.html');
  let html: string;
  try {
    html = readFileSync(htmlPath, 'utf-8');
  } catch {
    const altPath = path.join(__dirname, 'public', 'index.html');
    html = readFileSync(altPath, 'utf-8');
  }

  let paletteHtml: string;
  try {
    paletteHtml = readFileSync(path.join(__dirname, '..', 'src', 'public', 'palette.html'), 'utf-8');
  } catch {
    try {
      paletteHtml = readFileSync(path.join(__dirname, 'public', 'palette.html'), 'utf-8');
    } catch {
      paletteHtml = '<html><body>Name Palette not found</body></html>';
    }
  }

  server = http.createServer(async (req, res) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    if (req.method === 'OPTIONS') {
      res.writeHead(204);
      res.end();
      return;
    }

    const url = new URL(req.url ?? '/', `http://localhost:${port}`);

    try {
      // Serve HTML
      if (req.method === 'GET' && url.pathname === '/') {
        res.writeHead(200, { 'Content-Type': 'text/html' });
        res.end(html);
        return;
      }

      // ---- Audio file serving ----
      if (req.method === 'GET' && url.pathname.startsWith('/recordings/raw/')) {
        const filename = decodeURIComponent(url.pathname.slice('/recordings/raw/'.length));
        // Path traversal protection
        if (filename.includes('..') || filename.includes('/') || filename.includes('\\')) {
          res.writeHead(400);
          res.end('Invalid filename');
          return;
        }
        const filePath = path.join('recordings', 'raw', filename);
        try {
          const stat = fs.statSync(filePath);
          const total = stat.size;
          const rangeHeader = req.headers.range;

          if (rangeHeader) {
            // Parse range request for seekable audio
            const match = rangeHeader.match(/bytes=(\d+)-(\d*)/);
            if (match) {
              const start = parseInt(match[1], 10);
              const end = match[2] ? parseInt(match[2], 10) : total - 1;
              res.writeHead(206, {
                'Content-Type': 'audio/wav',
                'Content-Range': `bytes ${start}-${end}/${total}`,
                'Accept-Ranges': 'bytes',
                'Content-Length': end - start + 1,
              });
              fs.createReadStream(filePath, { start, end }).pipe(res);
            } else {
              res.writeHead(416);
              res.end('Invalid range');
            }
          } else {
            res.writeHead(200, {
              'Content-Type': 'audio/wav',
              'Content-Length': total,
              'Accept-Ranges': 'bytes',
            });
            fs.createReadStream(filePath).pipe(res);
          }
        } catch {
          res.writeHead(404);
          res.end('Not found');
        }
        return;
      }

      // ---- Transcription endpoints ----

      if (req.method === 'GET' && url.pathname === '/api/lines') {
        const sessionParam = url.searchParams.get('session');
        const sessionId = sessionParam ? parseInt(sessionParam, 10) : undefined;
        const lines = await worker.getLines(sessionId);
        json(res, 200, lines);
        return;
      }

      const lineMatch = url.pathname.match(/^\/api\/lines\/(\d+)$/);
      if (req.method === 'POST' && lineMatch) {
        const id = parseInt(lineMatch[1], 10);
        const body = JSON.parse(await readBody(req));
        const newText = body.text?.trim();
        if (!newText) {
          json(res, 400, { error: 'text required' });
          return;
        }
        const autoAdded = await worker.updateLine(id, newText);
        broadcast({ type: 'edit', data: { id, text: newText, autoAdded } });
        json(res, 200, { success: true, autoAdded });
        return;
      }

      if (req.method === 'DELETE' && lineMatch) {
        const id = parseInt(lineMatch[1], 10);
        try {
          const deleted = await worker.deleteLine(id);
          if (deleted) {
            broadcast({ type: 'delete', data: { id } });
          }
          json(res, 200, { success: true });
        } catch (e) {
          json(res, 400, { error: String(e) });
        }
        return;
      }

      // ---- Session endpoints ----

      if (req.method === 'GET' && url.pathname === '/api/sessions') {
        const sessions = await listSessions();
        json(res, 200, sessions);
        return;
      }

      const sessionMatch = url.pathname.match(/^\/api\/sessions\/(\d+)$/);

      if (req.method === 'DELETE' && sessionMatch) {
        const id = parseInt(sessionMatch[1], 10);
        try {
          const audioFiles = await deleteSessionFull(id);
          // Delete audio files from disk
          for (const f of audioFiles) {
            const filePath = path.join('recordings', 'raw', f);
            fs.unlink(filePath, () => {});
          }
          const sessions = await listSessions();
          broadcast({ type: 'sessions', data: sessions });
          json(res, 200, { success: true, deletedAudioFiles: audioFiles.length });
        } catch (e) {
          json(res, 400, { error: String(e) });
        }
        return;
      }

      const previewMatch = url.pathname.match(/^\/api\/sessions\/(\d+)\/preview$/);
      if (req.method === 'GET' && previewMatch) {
        const id = parseInt(previewMatch[1], 10);
        const preview = await getSessionPreview(id);
        json(res, 200, preview);
        return;
      }

      if (req.method === 'POST' && url.pathname === '/api/sessions/merge') {
        const body = JSON.parse(await readBody(req));
        const sessionIds = body.sessionIds;
        const name = body.name?.trim();
        if (!sessionIds || !Array.isArray(sessionIds) || sessionIds.length < 2) {
          json(res, 400, { error: 'At least 2 sessionIds required' });
          return;
        }
        if (!name) {
          json(res, 400, { error: 'name required' });
          return;
        }
        try {
          const result = await mergeSessions(sessionIds.map(Number), name);
          const sessions = await listSessions();
          broadcast({ type: 'sessions', data: sessions });
          json(res, 200, { success: true, newSessionId: result.newSessionId });
        } catch (e) {
          json(res, 400, { error: String(e) });
        }
        return;
      }

      // ---- Search ----

      if (req.method === 'GET' && url.pathname === '/api/search') {
        const q = url.searchParams.get('q') || '';
        if (q.length < 2) {
          json(res, 200, []);
          return;
        }
        const results = await searchTranscriptions(q);
        json(res, 200, results);
        return;
      }

      // ---- Hotword endpoints ----

      if (req.method === 'GET' && url.pathname === '/api/hotwords') {
        json(res, 200, await getHotwords());
        return;
      }

      if (req.method === 'POST' && url.pathname === '/api/hotwords') {
        const body = JSON.parse(await readBody(req));
        const word = body.word?.trim();
        if (!word) {
          json(res, 400, { error: 'word required' });
          return;
        }
        const added = await addHotword(word);
        if (added) broadcast({ type: 'hotwords', data: await getHotwords() });
        json(res, 200, { added });
        return;
      }

      if (req.method === 'DELETE' && url.pathname === '/api/hotwords') {
        const body = JSON.parse(await readBody(req));
        const word = body.word?.trim();
        if (!word) {
          json(res, 400, { error: 'word required' });
          return;
        }
        const removed = await removeHotword(word);
        if (removed) broadcast({ type: 'hotwords', data: await getHotwords() });
        json(res, 200, { removed });
        return;
      }

      if (req.method === 'PUT' && url.pathname === '/api/hotwords') {
        const body = JSON.parse(await readBody(req));
        const oldWord = body.oldWord?.trim();
        const newWord = body.newWord?.trim();
        if (!oldWord || !newWord) {
          json(res, 400, { error: 'oldWord and newWord required' });
          return;
        }
        const updated = await updateHotword(oldWord, newWord);
        if (updated) broadcast({ type: 'hotwords', data: await getHotwords() });
        json(res, 200, { updated });
        return;
      }

      // ---- Voice control endpoints ----

      if (req.method === 'GET' && url.pathname === '/api/status') {
        json(res, 200, controller.getStatus());
        return;
      }

      if (req.method === 'GET' && url.pathname === '/api/channels') {
        json(res, 200, controller.getVoiceChannels());
        return;
      }

      if (req.method === 'GET' && url.pathname === '/api/members') {
        json(res, 200, controller.getVoiceMembers());
        return;
      }

      if (req.method === 'POST' && url.pathname === '/api/join') {
        const body = JSON.parse(await readBody(req));
        const channelId = body.channelId?.trim();
        if (!channelId) {
          json(res, 400, { error: 'channelId required' });
          return;
        }
        try {
          const channelName = await controller.joinChannel(channelId);
          json(res, 200, { success: true, channelName });
        } catch (e) {
          json(res, 400, { error: String(e) });
        }
        return;
      }

      if (req.method === 'POST' && url.pathname === '/api/leave') {
        try {
          const outFile = await controller.leaveChannel();
          json(res, 200, { success: true, outFile });
        } catch (e) {
          json(res, 400, { error: String(e) });
        }
        return;
      }

      if (req.method === 'POST' && url.pathname === '/api/character-name') {
        const body = JSON.parse(await readBody(req));
        const userId = body.userId?.trim();
        const name = body.name?.trim() || null;
        if (!userId) {
          json(res, 400, { error: 'userId required' });
          return;
        }
        controller.setCharacterName(userId, name);
        json(res, 200, { success: true });
        return;
      }

      // ---- Bulk nickname update ----

      if (req.method === 'POST' && url.pathname === '/api/bulk-nickname') {
        const body = JSON.parse(await readBody(req));
        const lineIds: number[] = body.lineIds;
        const nicknames: Record<string, string | null> = body.nicknames;
        if (!lineIds || !Array.isArray(lineIds) || !nicknames) {
          json(res, 400, { error: 'lineIds and nicknames required' });
          return;
        }
        try {
          await bulkUpdateNicknames(lineIds, nicknames);
          // Also update character names for future transcriptions
          for (const [userId, name] of Object.entries(nicknames)) {
            controller.setCharacterName(userId, name);
          }
          // Reload the affected session lines for the UI
          const lines = await worker.getLines();
          broadcast({ type: 'reload_lines', data: lines });
          json(res, 200, { success: true });
        } catch (e) {
          json(res, 400, { error: String(e) });
        }
        return;
      }

      // ---- Nickname presets (soundboard) ----

      if (req.method === 'GET' && url.pathname === '/api/presets') {
        const userId = url.searchParams.get('userId') || undefined;
        const presets = await controller.getNicknamePresets(userId);
        json(res, 200, presets);
        return;
      }

      if (req.method === 'POST' && url.pathname === '/api/presets') {
        const body = JSON.parse(await readBody(req));
        const { userId, label, position } = body;
        if (!userId || !label) { json(res, 400, { error: 'userId and label required' }); return; }
        const id = await addNicknamePreset(userId, label, position || 0);
        await broadcastPresets();
        json(res, 200, { id });
        return;
      }

      const presetMatch = url.pathname.match(/^\/api\/presets\/(\d+)$/);
      if (req.method === 'PUT' && presetMatch) {
        const id = parseInt(presetMatch[1], 10);
        const body = JSON.parse(await readBody(req));
        await updateNicknamePreset(id, body.label?.trim() || '', body.position);
        await broadcastPresets();
        json(res, 200, { success: true });
        return;
      }

      if (req.method === 'DELETE' && presetMatch) {
        const id = parseInt(presetMatch[1], 10);
        await deleteNicknamePreset(id);
        await broadcastPresets();
        json(res, 200, { success: true });
        return;
      }

      // Move preset to category
      if (req.method === 'POST' && url.pathname === '/api/presets/move') {
        const body = JSON.parse(await readBody(req));
        await movePresetToCategory(body.presetId, body.categoryId ?? null);
        await broadcastPresets();
        json(res, 200, { success: true });
        return;
      }

      // ---- Nickname categories ----

      if (req.method === 'GET' && url.pathname === '/api/categories') {
        const userId = url.searchParams.get('userId') || undefined;
        const categories = await getNicknameCategories(userId);
        json(res, 200, categories);
        return;
      }

      if (req.method === 'POST' && url.pathname === '/api/categories') {
        const body = JSON.parse(await readBody(req));
        const { userId, name, position } = body;
        if (!userId || !name) { json(res, 400, { error: 'userId and name required' }); return; }
        const id = await addNicknameCategory(userId, name, position || 0);
        await broadcastPresets();
        json(res, 200, { id });
        return;
      }

      const catMatch = url.pathname.match(/^\/api\/categories\/(\d+)$/);
      if (req.method === 'PUT' && catMatch) {
        const id = parseInt(catMatch[1], 10);
        const body = JSON.parse(await readBody(req));
        await updateNicknameCategory(id, body.name?.trim() || '', body.position);
        await broadcastPresets();
        json(res, 200, { success: true });
        return;
      }

      if (req.method === 'DELETE' && catMatch) {
        const id = parseInt(catMatch[1], 10);
        await deleteNicknameCategory(id);
        await broadcastPresets();
        json(res, 200, { success: true });
        return;
      }

      // Name Palette page (pop-out window)
      if (req.method === 'GET' && (url.pathname === '/soundboard' || url.pathname === '/palette')) {
        res.writeHead(200, { 'Content-Type': 'text/html' });
        res.end(paletteHtml);
        return;
      }

      // ---- Pause/resume transcription ----

      if (req.method === 'POST' && url.pathname === '/api/pause') {
        const paused = controller.togglePause();
        json(res, 200, { paused });
        return;
      }

      // ---- Save recordings toggle ----

      if (req.method === 'GET' && url.pathname === '/api/save-recordings') {
        json(res, 200, { enabled: controller.getSaveRecordings() });
        return;
      }

      if (req.method === 'POST' && url.pathname === '/api/save-recordings') {
        const body = JSON.parse(await readBody(req));
        const enabled = !!body.enabled;
        controller.setSaveRecordings(enabled);
        json(res, 200, { enabled });
        return;
      }

      // ---- Ignored users endpoints ----

      if (req.method === 'GET' && url.pathname === '/api/ignored-users') {
        const users = await controller.getIgnoredUsers();
        json(res, 200, users);
        return;
      }

      if (req.method === 'POST' && url.pathname === '/api/ignore') {
        const body = JSON.parse(await readBody(req));
        const userId = body.userId?.trim();
        const username = body.username?.trim() || userId;
        if (!userId) {
          json(res, 400, { error: 'userId required' });
          return;
        }
        const added = await controller.ignoreUser(userId, username);
        json(res, 200, { added });
        return;
      }

      if (req.method === 'POST' && url.pathname === '/api/unignore') {
        const body = JSON.parse(await readBody(req));
        const userId = body.userId?.trim();
        if (!userId) {
          json(res, 400, { error: 'userId required' });
          return;
        }
        const removed = await controller.unignoreUser(userId);
        json(res, 200, { removed });
        return;
      }

      // 404
      res.writeHead(404);
      res.end('Not found');
    } catch (e) {
      logger.error(`Web UI request error: ${e}`);
      json(res, 500, { error: String(e) });
    }
  });

  // WebSocket server
  wss = new WebSocketServer({ server });

  wss.on('connection', async (ws) => {
    logger.info('Web UI client connected');

    const lines = await worker.getLines();
    const hotwords = await getHotwords();
    const status = controller.getStatus();
    const members = controller.getVoiceMembers();
    const sessions = await listSessions();
    const saveRecordings = controller.getSaveRecordings();
    const ignoredUsers = await controller.getIgnoredUsers();
    const presets = await controller.getNicknamePresets();
    const categories = await getNicknameCategories();
    ws.send(JSON.stringify({ type: 'init', lines, hotwords, status, members, sessions, saveRecordings, ignoredUsers, presets, categories }));

    ws.on('close', () => {
      logger.info('Web UI client disconnected');
    });
  });

  // Forward transcription events to WebSocket clients
  worker.on('line', (line: TranscriptionLine) => {
    broadcast({ type: 'line', data: line });
  });


  worker.on('hotwords', (hotwords: string[]) => {
    broadcast({ type: 'hotwords', data: hotwords });
  });

  worker.on('finalized', () => {
    broadcast({ type: 'finalized' });
  });

  // Forward bot controller events
  controller.on('status', () => {
    broadcast({ type: 'status', data: controller.getStatus() });
  });

  controller.on('members', () => {
    broadcast({ type: 'members', data: controller.getVoiceMembers() });
  });

  controller.on('session_started', (session: any) => {
    broadcast({ type: 'session_started', data: session });
  });

  controller.on('save_recordings', (enabled: boolean) => {
    broadcast({ type: 'save_recordings', data: enabled });
  });

  controller.on('ignored_users', async () => {
    const users = await controller.getIgnoredUsers();
    broadcast({ type: 'ignored_users', data: users });
  });

  server.listen(port, () => {
    logger.info(`Web UI listening on http://localhost:${port}`);
  });

  return server;
}

export function stopWebUI(): void {
  if (wss) {
    for (const client of wss.clients) {
      client.close();
    }
    wss.close();
    wss = null;
  }
  if (server) {
    server.close();
    server = null;
  }
}
