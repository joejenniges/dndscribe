import { ChildProcess, spawn } from 'child_process';
import { writeFile, mkdir, unlink } from 'fs/promises';
import { createInterface, Interface } from 'readline';
import { EventEmitter } from 'events';
import path from 'path';
import { logger } from './logger';
import { detectNewHotwords, addHotword, getHotwords } from './hotwords';
import {
  insertTranscription, getTranscriptionsBySession, getTranscriptionById,
  updateTranscription, deleteTranscription, getLatestSession, endSession,
} from './db';

const RECORDINGS_DIR = 'recordings';

const HALLUCINATION_BLOCKLIST = new Set([
  'thank you',
  'thanks for watching',
  'thank you for watching',
  'subscribe',
  'please subscribe',
  'like and subscribe',
  'thanks for listening',
  'thank you for listening',
  'bye',
  'goodbye',
  'see you next time',
  'see you in the next video',
  'you',
  'the end',
  'i\'ll see you in the next one',
  'thanks',
]);

export interface TranscriptionTask {
  pcmData: Buffer;
  userId: string;
  username: string;
  timestamp: Date;
  audioFilename?: string;
  rms?: number;
  durationMs?: number;
  nickname?: string;
}

export interface TranscriptionLine {
  id: number;
  timestamp: string;
  discordUsername: string;
  nickname: string | null;
  text: string;
  audioFilenames: string[];
  rms?: number | null;
  confidence?: number | null;
}

/**
 * Convert 48kHz stereo PCM (s16le) to 16kHz mono WAV using ffmpeg.
 */
function pcmToWav(pcmData: Buffer, outputPath: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const ffmpeg = spawn('ffmpeg', [
      '-y',
      '-f', 's16le',
      '-ar', '48000',
      '-ac', '2',
      '-i', 'pipe:0',
      '-af', 'pan=mono|c0=0.5*c0+0.5*c1,lowpass=f=7500,aresample=16000:dither_method=triangular',
      outputPath,
    ], { stdio: ['pipe', 'pipe', 'pipe'] });

    let stderr = '';
    ffmpeg.stderr.on('data', (chunk: Buffer) => { stderr += chunk.toString(); });

    ffmpeg.on('close', (code) => {
      if (code === 0) resolve();
      else reject(new Error(`ffmpeg exited with code ${code}: ${stderr}`));
    });

    ffmpeg.on('error', reject);
    ffmpeg.stdin.write(pcmData);
    ffmpeg.stdin.end();
  });
}

/**
 * Manages a persistent Whisper Python process for fast transcription.
 */
export class TranscriptionWorker extends EventEmitter {
  private queue: TranscriptionTask[] = [];
  private processing = false;
  private active = true;
  private pythonProcess: ChildProcess | null = null;
  private readline: Interface | null = null;
  private ready = false;
  private pendingResolve: ((result: { text: string; confidence: number }) => void) | null = null;
  private pendingReject: ((err: Error) => void) | null = null;
  private pythonPath: string;
  private model: string;
  sessionId: number | null = null;
  // Track last transcription text per user for Whisper context conditioning
  private lastContext = new Map<string, string>();

  constructor(model = 'base', pythonPath?: string) {
    super();
    this.model = model;
    this.pythonPath = pythonPath ?? path.join(__dirname, '..', '.venv', 'Scripts', 'python.exe');
  }

  setSessionId(id: number | null): void {
    this.sessionId = id;
  }

  async init(): Promise<void> {
    await mkdir(RECORDINGS_DIR, { recursive: true });
    await mkdir(path.join(RECORDINGS_DIR, 'tmp'), { recursive: true });
    await this.startPythonProcess();
  }

  private startPythonProcess(): Promise<void> {
    return new Promise((resolve, reject) => {
      const scriptPath = path.join(__dirname, '..', 'src', 'whisper_server.py');

      logger.info(`Starting whisper server: ${this.pythonPath} ${scriptPath}`);

      this.pythonProcess = spawn(this.pythonPath, [scriptPath], {
        stdio: ['pipe', 'pipe', 'pipe'],
        env: {
          ...process.env,
          WHISPER_MODEL: this.model,
        },
      });

      this.pythonProcess.stderr!.on('data', (data: Buffer) => {
        logger.info(`[whisperx] ${data.toString().trim()}`);
      });

      this.readline = createInterface({ input: this.pythonProcess.stdout! });

      const onStartupLine = (line: string) => {
        try {
          const msg = JSON.parse(line);
          if (msg.ready) {
            this.ready = true;
            logger.info('Whisper server ready');
            this.readline!.removeListener('line', onStartupLine);
            this.readline!.on('line', this.onResponse.bind(this));
            resolve();
            return;
          }
        } catch {
          // Not JSON -- stray log line leaked to stdout, ignore
        }
        logger.info(`[whisperx stdout] ${line}`);
      };
      this.readline.on('line', onStartupLine);

      this.pythonProcess.on('exit', (code) => {
        this.ready = false;
        logger.error(`Whisper server exited with code ${code}`);
        if (this.pendingReject) {
          this.pendingReject(new Error(`Whisper process exited with code ${code}`));
          this.pendingResolve = null;
          this.pendingReject = null;
        }
        // Auto-restart if we're still active
        if (this.active) {
          logger.info('Restarting Whisper server...');
          this.startPythonProcess().catch((e) =>
            logger.error(`Failed to restart Whisper server: ${e}`)
          );
        }
      });

      this.pythonProcess.on('error', (err) => {
        logger.error(`Whisper server error: ${err.message}`);
        reject(err);
      });
    });
  }

  private onResponse(line: string): void {
    let msg: any;
    try {
      msg = JSON.parse(line);
    } catch {
      logger.debug(`[whisperx stdout noise] ${line}`);
      return;
    }

    if (!this.pendingResolve) return;

    if (msg.error) {
      logger.error(`Whisper error: ${msg.error}`);
    }
    this.pendingResolve({ text: msg.text ?? '', confidence: msg.confidence ?? 0 });
    this.pendingResolve = null;
    this.pendingReject = null;
  }

  private transcribe(wavPath: string, context?: string): Promise<{ text: string; confidence: number }> {
    return new Promise((resolve, reject) => {
      if (!this.ready || !this.pythonProcess?.stdin) {
        reject(new Error('Whisper server not ready'));
        return;
      }

      this.pendingResolve = resolve;
      this.pendingReject = reject;

      const reqObj: any = { wav: wavPath };
      if (context) reqObj.context = context;
      const req = JSON.stringify(reqObj) + '\n';
      this.pythonProcess.stdin.write(req);
    });
  }

  addTask(pcmData: Buffer, userId: string, username: string, timestamp: Date, opts?: {
    audioFilename?: string;
    rms?: number;
    durationMs?: number;
    nickname?: string;
  }): void {
    if (!this.active) return;

    // Skip very short audio (< 0.25s at 48kHz stereo 16-bit)
    if (pcmData.length < 48000) {
      logger.debug(`Skipping short audio for ${username}: ${pcmData.length} bytes`);
      return;
    }

    this.queue.push({
      pcmData, userId, username, timestamp,
      audioFilename: opts?.audioFilename,
      rms: opts?.rms,
      durationMs: opts?.durationMs,
      nickname: opts?.nickname,
    });
    logger.debug(`Queued transcription for ${username}, ${pcmData.length} bytes`);
    this.processNext();
  }

  private async processNext(): Promise<void> {
    if (this.processing || this.queue.length === 0) return;
    this.processing = true;

    const task = this.queue.shift()!;
    try {
      await this.processTask(task);
    } catch (e) {
      logger.error(`Transcription failed for ${task.username}: ${e}`);
    } finally {
      this.processing = false;
      if (this.queue.length > 0) this.processNext();
    }
  }

  private async processTask(task: TranscriptionTask): Promise<void> {
    const ts = task.timestamp.toISOString().replace(/[:.]/g, '-');
    const wavPath = path.join(RECORDINGS_DIR, 'tmp', `${task.userId}-${ts}.wav`);

    try {
      await pcmToWav(task.pcmData, wavPath);

      // Pass previous transcription text as context so Whisper maintains
      // sentence continuity across chunk boundaries.
      const prevContext = this.lastContext.get(task.userId);
      const result = await this.transcribe(wavPath, prevContext);
      if (!result.text || result.text.length < 2) return;

      // Reject known hallucinations
      if (HALLUCINATION_BLOCKLIST.has(result.text.toLowerCase().trim().replace(/[.!?,]/g, ''))) {
        logger.debug(`Filtered hallucination for ${task.username}: "${result.text}"`);
        return;
      }

      // Strip non-ASCII (same behavior as Python version)
      const asciiText = result.text.replace(/[^\x00-\x7F]/g, '');
      const confidence = result.confidence;
      const asciiUsername = task.username.replace(/[^\x00-\x7F]/g, '');

      if (!this.sessionId) {
        logger.error('No active session, dropping transcription');
        return;
      }

      // Update context for next transcription from this user
      this.lastContext.set(task.userId, asciiText.slice(-200));

      // Always INSERT -- visual merging of consecutive same-speaker lines
      // is done in the UI, not the DB. This preserves per-chunk metadata
      // (confidence, audio file, timestamp) for each individual chunk.
      const audioFilenames = task.audioFilename ? [task.audioFilename] : [];
      const dbId = await insertTranscription({
        sessionId: this.sessionId,
        discordUserId: task.userId,
        discordUsername: asciiUsername,
        nickname: task.nickname ?? null,
        spokenText: asciiText,
        audioFilenames,
        rms: task.rms ?? null,
        durationMs: task.durationMs ?? null,
        confidence,
      });

      this.emit('line', {
        id: dbId,
        timestamp: task.timestamp.toISOString(),
        discordUsername: asciiUsername,
        nickname: task.nickname ?? null,
        text: asciiText,
        audioFilenames,
        rms: task.rms ?? null,
        confidence,
      } as TranscriptionLine);

      logger.info(`Transcribed ${task.username}: ${asciiText.slice(0, 80)}...`);
    } finally {
      await unlink(wavPath).catch(() => {});
    }
  }

  /** Read transcription lines from the DB for a given session. */
  async getLines(sessionId?: number): Promise<TranscriptionLine[]> {
    let sid = sessionId;
    if (!sid) {
      // Default to current session, or latest session in DB
      if (this.sessionId) {
        sid = this.sessionId;
      } else {
        const latest = await getLatestSession();
        if (!latest) return [];
        sid = latest.id;
      }
    }

    const rows = await getTranscriptionsBySession(sid);
    return rows.map(r => ({
      id: r.id,
      timestamp: r.createdAt,
      discordUsername: r.discordUsername,
      nickname: r.nickname,
      text: r.spokenText,
      audioFilenames: r.audioFilenames,
      rms: r.rms,
      confidence: r.confidence,
    }));
  }

  /**
   * Update a transcription line by DB id. Returns auto-added hotwords.
   * Detects proper noun corrections and adds them to hotwords.txt.
   */
  async updateLine(id: number, newText: string): Promise<string[]> {
    const autoAdded: string[] = [];

    const row = await getTranscriptionById(id);
    if (!row) return autoAdded;

    const oldText = row.spokenText;
    const currentHotwords = await getHotwords();
    const candidates = detectNewHotwords(oldText, newText);
    for (const word of candidates) {
      if (!currentHotwords.some(h => h.toLowerCase() === word.toLowerCase())) {
        if (await addHotword(word)) {
          autoAdded.push(word);
        }
      }
    }

    await updateTranscription(id, newText);

    if (autoAdded.length > 0) {
      this.emit('hotwords', await getHotwords());
      logger.info(`Auto-added hotwords from correction: ${autoAdded.join(', ')}`);
    }

    return autoAdded;
  }

  /** Delete a transcription line and its audio file if one exists. */
  async deleteLine(id: number): Promise<boolean> {
    const audioFilenames = await deleteTranscription(id);
    if (audioFilenames === null) return false;

    // Delete all associated audio files
    for (const f of audioFilenames) {
      const filePath = path.join('recordings', 'raw', f);
      await unlink(filePath).catch(() => {});
    }
    logger.info(`Deleted transcription ${id} (${audioFilenames.length} audio files)`);
    return true;
  }

  /**
   * Wait for any in-flight transcription to finish, then finalize the
   * session: query DB, merge consecutive same-speaker lines, save export file.
   */
  async finalize(): Promise<string | null> {
    // Wait for queue to drain
    while (this.processing || this.queue.length > 0) {
      await new Promise((r) => setTimeout(r, 200));
    }

    if (!this.sessionId) {
      this.emit('finalized');
      return null;
    }

    const sessionId = this.sessionId;
    this.sessionId = null;
    this.lastContext.clear();

    const result = await finalizeSession(sessionId);
    this.emit('finalized');
    return result;
  }

  stop(): void {
    this.active = false;
    if (this.pythonProcess) {
      this.pythonProcess.stdin?.end();
      this.pythonProcess.kill();
      this.pythonProcess = null;
    }
    if (this.readline) {
      this.readline.close();
      this.readline = null;
    }
    logger.info(`Transcription worker stopped, ${this.queue.length} tasks remaining`);
  }
}

interface FinalizedLine {
  timestamp: Date;
  username: string;
  text: string;
}

/**
 * Query DB for session transcriptions, sort by timestamp, merge consecutive
 * lines from the same speaker, save to a dated export file, and end the session.
 */
async function finalizeSession(sessionId: number): Promise<string | null> {
  const rows = await getTranscriptionsBySession(sessionId);
  if (rows.length === 0) {
    await endSession(sessionId);
    return null;
  }

  // Map to finalization format -- already sorted by created_at from DB query
  const lines: FinalizedLine[] = rows.map(r => ({
    timestamp: new Date(r.createdAt),
    username: r.nickname ?? r.discordUsername,
    text: r.spokenText,
  }));

  // Merge consecutive lines from same speaker
  const merged: FinalizedLine[] = [];
  for (const line of lines) {
    const prev = merged[merged.length - 1];
    if (prev && prev.username === line.username) {
      let prevText = prev.text.replace(/\.{3,}$/, '').trimEnd();
      if (prevText.endsWith('.') && line.text[0] === line.text[0].toLowerCase()) {
        // Lowercase start suggests continuation, keep the period
      }
      prev.text = prevText + ' ' + line.text;
    } else {
      merged.push({ ...line, text: line.text.replace(/\.{3,}$/, '').trimEnd() });
    }
  }

  // Clean up final merged texts
  for (const line of merged) {
    line.text = line.text.replace(/\.{3,}$/, '').trimEnd();
    if (!/[.!?]$/.test(line.text)) {
      line.text += '.';
    }
  }

  // Build output
  const output = merged
    .map((l) => `${l.username}: ${l.text}`)
    .join('\n') + '\n';

  // Save to dated file
  const now = new Date();
  const dateStr = now.toISOString().replace(/[:.]/g, '-').replace('T', '_').slice(0, 19);
  const outFile = path.join(RECORDINGS_DIR, `transcription-${dateStr}.txt`);
  await writeFile(outFile, output, 'utf-8');

  await endSession(sessionId);

  logger.info(`Finalized transcription: ${merged.length} lines -> ${outFile}`);
  return outFile;
}
