import { OpusEncoder } from '@discordjs/opus';
import { writeFile, mkdir } from 'fs/promises';
import path from 'path';
import { logger } from './logger';
import { TranscriptionWorker } from './transcription';

const opus = new OpusEncoder(48000, 2);
const RECORDINGS_DIR = 'recordings';

interface UserBuffer {
  pcm: Buffer[];
  totalBytes: number;
  startTime: Date;
  lastPacketTime: number; // Date.now() ms
  lastTimestamp: number;  // Discord RTP timestamp for packet ordering
  packetCount: number;    // Total packets received for corruption detection
}

/**
 * Build a WAV file header for raw PCM data.
 */
function wavHeader(pcmLength: number): Buffer {
  const header = Buffer.alloc(44);
  const sampleRate = 48000;
  const channels = 2;
  const bitsPerSample = 16;
  const byteRate = sampleRate * channels * (bitsPerSample / 8);
  const blockAlign = channels * (bitsPerSample / 8);

  header.write('RIFF', 0);
  header.writeUInt32LE(36 + pcmLength, 4);
  header.write('WAVE', 8);
  header.write('fmt ', 12);
  header.writeUInt32LE(16, 16);          // fmt chunk size
  header.writeUInt16LE(1, 20);           // PCM format
  header.writeUInt16LE(channels, 22);
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(byteRate, 28);
  header.writeUInt16LE(blockAlign, 32);
  header.writeUInt16LE(bitsPerSample, 34);
  header.write('data', 36);
  header.writeUInt32LE(pcmLength, 40);
  return header;
}

export class AudioSink {
  private buffers = new Map<string, UserBuffer>();
  private ignoredUsers: Set<string>;
  private transcriptionWorker: TranscriptionWorker;
  private active = true;
  private paused = false;
  private silenceCheckInterval: ReturnType<typeof setInterval>;
  private userNames = new Map<string, string>();
  private characterNames = new Map<string, string>();
  private saveRecordings = false;
  sessionId: number | null = null;

  private readonly SILENCE_DURATION_MS = 800;
  private readonly MAX_BUFFER_DURATION_MS = 15000;
  // 48kHz * 2ch * 2bytes * 0.25s = 48000 bytes minimum
  private readonly MIN_BUFFER_BYTES = 48000;
  private readonly MIN_RMS_THRESHOLD = 50;

  constructor(
    transcriptionWorker: TranscriptionWorker,
    ignoredUsers: Set<string>,
  ) {
    this.transcriptionWorker = transcriptionWorker;
    this.ignoredUsers = ignoredUsers;

    // Check for silence periodically rather than on every packet
    this.silenceCheckInterval = setInterval(() => this.checkAllSilence(), 100);
  }

  setUserName(userId: string, name: string): void {
    this.userNames.set(userId, name);
  }

  setCharacterName(userId: string, name: string): void {
    this.characterNames.set(userId, name);
  }

  removeCharacterName(userId: string): void {
    this.characterNames.delete(userId);
  }

  setSessionId(id: number | null): void {
    this.sessionId = id;
  }

  /**
   * Called by the voice data stream event handler.
   * data is an Opus frame, userID is the speaking user.
   */
  onData(data: Buffer, userID: string, timestamp: number): void {
    if (!this.active) return;
    if (this.paused) return;
    if (this.ignoredUsers.has(userID)) return;

    // Detect corrupted packets (mostly zeros -- Cloudflare voice server issue)
    if (data.length > 0 && data[0] === 0) {
      let zeroCount = 0;
      for (let i = 0; i < data.length; i++) {
        if (data[i] === 0) zeroCount++;
      }
      if (zeroCount >= data.length - 1) {
        logger.debug(`Skipping corrupted zero-packet from ${userID}`);
        return;
      }
    }

    // Decode Opus -> PCM (48kHz stereo s16le)
    let pcm: Buffer;
    try {
      pcm = opus.decode(data);
    } catch (e) {
      logger.debug(`Opus decode failed for user ${userID}: ${e}`);
      return;
    }

    const now = Date.now();
    let buf = this.buffers.get(userID);

    if (!buf) {
      buf = {
        pcm: [],
        totalBytes: 0,
        startTime: new Date(),
        lastPacketTime: now,
        lastTimestamp: timestamp,
        packetCount: 0,
      };
      this.buffers.set(userID, buf);
    }

    buf.packetCount++;
    buf.lastPacketTime = now;

    if (timestamp < buf.lastTimestamp && buf.pcm.length > 0) {
      // Out-of-order packet -- insert at the beginning (simple approach since
      // most reordering is just 1-2 packets behind)
      buf.pcm.unshift(pcm);
      logger.debug(`Reordered packet from ${userID}: ts ${timestamp} < last ${buf.lastTimestamp}`);
    } else {
      buf.pcm.push(pcm);
      buf.lastTimestamp = timestamp;
    }

    buf.totalBytes += pcm.length;

    // Check max duration for this user
    if (now - buf.startTime.getTime() >= this.MAX_BUFFER_DURATION_MS) {
      this.flushUser(userID);
    }
  }

  private checkAllSilence(): void {
    const now = Date.now();
    for (const [userID, buf] of this.buffers) {
      if (now - buf.lastPacketTime >= this.SILENCE_DURATION_MS && buf.totalBytes > 0) {
        this.flushUser(userID);
      }
    }
  }

  private flushUser(userID: string): void {
    const buf = this.buffers.get(userID);
    if (!buf || buf.totalBytes === 0) return;

    const combined = Buffer.concat(buf.pcm);
    const timestamp = buf.startTime;
    // discord username is always the real discord name -- never the character name.
    // Character name goes in the nickname field separately.
    const username = this.userNames.get(userID) ?? 'Unknown';
    const nickname = this.characterNames.get(userID) ?? null;

    // Reset buffer
    buf.pcm = [];
    buf.totalBytes = 0;
    buf.startTime = new Date();

    const rms = this.computeRMS(combined);
    // Duration in ms: bytes / (sampleRate * channels * bytesPerSample) * 1000
    const durationMs = Math.round((combined.length / (48000 * 2 * 2)) * 1000);

    // Generate filename BEFORE both save and addTask for correlation
    const audioFilename = this.buildAudioFilename(rms, username, timestamp);

    // Save recording BEFORE any filtering -- user wants all audio saved.
    if (this.saveRecordings) {
      this.saveRecording(combined, audioFilename);
    }

    if (combined.length < this.MIN_BUFFER_BYTES) {
      logger.debug(`Skipping short buffer for ${username}: ${combined.length} bytes`);
      return;
    }

    // Check PCM energy -- reject near-silent buffers before they reach Whisper.
    if (rms < this.MIN_RMS_THRESHOLD) {
      logger.debug(`Skipping silent buffer for ${username}: RMS ${rms.toFixed(1)}`);
      return;
    }

    this.transcriptionWorker.addTask(combined, userID, username, timestamp, {
      audioFilename: this.saveRecordings ? audioFilename : undefined,
      rms,
      durationMs,
      nickname: nickname ?? undefined,
    });
  }

  private buildAudioFilename(rms: number, username: string, timestamp: Date): string {
    const pad = (n: number) => n.toString().padStart(2, '0');
    const d = timestamp;
    const ts = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}_${pad(d.getHours())}-${pad(d.getMinutes())}-${pad(d.getSeconds())}`;
    const safeUsername = username.replace(/[^a-zA-Z0-9_-]/g, '_');
    const rmsStr = Math.round(rms).toString();
    return `${ts}_${rmsStr}_${safeUsername}.wav`;
  }

  private saveRecording(pcm: Buffer, filename: string): void {
    const filePath = path.join(RECORDINGS_DIR, 'raw', filename);

    const header = wavHeader(pcm.length);
    const wav = Buffer.concat([header, pcm]);

    // Fire-and-forget -- don't block the audio pipeline
    mkdir(path.join(RECORDINGS_DIR, 'raw'), { recursive: true })
      .then(() => writeFile(filePath, wav))
      .then(() => logger.debug(`Saved recording: ${filename}`))
      .catch((e) => logger.error(`Failed to save recording ${filename}: ${e}`));
  }

  /**
   * Compute RMS (root mean square) amplitude of s16le PCM buffer.
   * Samples every 10th sample for speed -- good enough for energy detection.
   */
  private computeRMS(pcm: Buffer): number {
    const sampleCount = Math.floor(pcm.length / 2);
    if (sampleCount === 0) return 0;

    let sumSquares = 0;
    let count = 0;
    for (let i = 0; i < pcm.length - 1; i += 20) { // every 10th sample (2 bytes each)
      const sample = pcm.readInt16LE(i);
      sumSquares += sample * sample;
      count++;
    }

    return Math.sqrt(sumSquares / count);
  }

  /** Flush a specific user's audio buffer immediately. Used when nickname changes
   *  so speech before and after the change ends up in separate transcription lines. */
  flushUserBuffer(userID: string): void {
    this.flushUser(userID);
  }

  /** Flush all user buffers immediately. */
  flushAllBuffers(): void {
    for (const userID of this.buffers.keys()) {
      this.flushUser(userID);
    }
  }

  setPaused(paused: boolean): void {
    if (paused && !this.paused) {
      // Flush current buffers before pausing
      this.flushAllBuffers();
    }
    this.paused = paused;
    logger.info(`Transcription ${paused ? 'paused' : 'resumed'}`);
  }

  isPaused(): boolean {
    return this.paused;
  }

  setSaveRecordings(enabled: boolean): void {
    this.saveRecordings = enabled;
    logger.info(`Recording save ${enabled ? 'enabled' : 'disabled'}`);
  }

  getSaveRecordings(): boolean {
    return this.saveRecordings;
  }

  updateIgnoredUsers(users: Set<string>): void {
    this.ignoredUsers = users;
  }

  stop(): void {
    this.active = false;
    clearInterval(this.silenceCheckInterval);

    // Flush all remaining buffers
    for (const userID of this.buffers.keys()) {
      this.flushUser(userID);
    }
    this.buffers.clear();

    logger.info('AudioSink stopped, remaining buffers flushed');
  }
}
