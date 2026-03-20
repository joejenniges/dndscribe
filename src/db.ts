import { Pool } from 'pg';
import { logger } from './logger';

let pool: Pool;

export function getPool(): Pool {
  return pool;
}

export async function initDB(): Promise<void> {
  const connectionString = process.env.DATABASE_URL;
  if (!connectionString) throw new Error('DATABASE_URL env var required');

  pool = new Pool({ connectionString });

  // Connection test
  const client = await pool.connect();
  try {
    await client.query('SELECT 1');
    logger.info('Database connected');
  } finally {
    client.release();
  }

  // Create tables
  await pool.query(`
    CREATE TABLE IF NOT EXISTS sessions (
      id BIGSERIAL PRIMARY KEY,
      started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      ended_at TIMESTAMPTZ,
      channel_name TEXT,
      channel_id TEXT
    )
  `);

  await pool.query(`
    CREATE TABLE IF NOT EXISTS transcriptions (
      id BIGSERIAL PRIMARY KEY,
      session_id BIGINT NOT NULL REFERENCES sessions(id),
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      discord_user_id TEXT NOT NULL,
      discord_username TEXT NOT NULL,
      nickname TEXT,
      spoken_text TEXT NOT NULL,
      audio_filenames TEXT[] NOT NULL DEFAULT '{}',
      rms REAL,
      duration_ms INTEGER,
      confidence REAL,
      edited_at TIMESTAMPTZ
    )
  `);

  // Add confidence column if it doesn't exist (migration for existing DBs)
  try {
    await pool.query(`ALTER TABLE transcriptions ADD COLUMN IF NOT EXISTS confidence REAL`);
  } catch { /* already exists */ }

  // Migrate: if audio_filename column exists (old schema), move data to audio_filenames array
  try {
    await pool.query(`
      DO $$ BEGIN
        IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='transcriptions' AND column_name='audio_filename') THEN
          -- Add new column if it doesn't exist
          IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='transcriptions' AND column_name='audio_filenames') THEN
            ALTER TABLE transcriptions ADD COLUMN audio_filenames TEXT[] NOT NULL DEFAULT '{}';
          END IF;
          -- Migrate data
          UPDATE transcriptions SET audio_filenames = ARRAY[audio_filename] WHERE audio_filename IS NOT NULL AND audio_filenames = '{}';
          -- Drop old column
          ALTER TABLE transcriptions DROP COLUMN audio_filename;
        END IF;
      END $$;
    `);
  } catch (e) {
    // Migration may have already run
    logger.info(`Schema migration note: ${e}`);
  }

  await pool.query(`CREATE INDEX IF NOT EXISTS idx_transcriptions_session ON transcriptions(session_id)`);
  await pool.query(`CREATE INDEX IF NOT EXISTS idx_transcriptions_created ON transcriptions(created_at)`);

  await pool.query(`
    CREATE TABLE IF NOT EXISTS ignored_users (
      discord_user_id TEXT PRIMARY KEY,
      discord_username TEXT NOT NULL,
      ignored_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
  `);

  await pool.query(`
    CREATE TABLE IF NOT EXISTS hotwords (
      id BIGSERIAL PRIMARY KEY,
      word TEXT NOT NULL UNIQUE,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
  `);

  await pool.query(`
    CREATE TABLE IF NOT EXISTS character_names (
      discord_user_id TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
  `);

  await pool.query(`
    CREATE TABLE IF NOT EXISTS nickname_presets (
      id BIGSERIAL PRIMARY KEY,
      discord_user_id TEXT NOT NULL,
      label TEXT NOT NULL,
      position INT NOT NULL DEFAULT 0,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
  `);
  await pool.query(`CREATE INDEX IF NOT EXISTS idx_nickname_presets_user ON nickname_presets(discord_user_id)`);

  await pool.query(`
    CREATE TABLE IF NOT EXISTS nickname_categories (
      id BIGSERIAL PRIMARY KEY,
      discord_user_id TEXT NOT NULL,
      name TEXT NOT NULL,
      position INT NOT NULL DEFAULT 0,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
  `);
  await pool.query(`CREATE INDEX IF NOT EXISTS idx_nickname_categories_user ON nickname_categories(discord_user_id)`);

  // Add category_id to nickname_presets if it doesn't exist
  try {
    await pool.query(`ALTER TABLE nickname_presets ADD COLUMN IF NOT EXISTS category_id BIGINT REFERENCES nickname_categories(id) ON DELETE SET NULL`);
  } catch { /* already exists */ }

  logger.info('Database schema ready');
}

export async function shutdownDB(): Promise<void> {
  if (pool) {
    await pool.end();
    logger.info('Database pool closed');
  }
}

// ---- Sessions ----

export async function createSession(channelName: string, channelId: string): Promise<number> {
  const result = await pool.query(
    `INSERT INTO sessions (channel_name, channel_id) VALUES ($1, $2) RETURNING id`,
    [channelName, channelId]
  );
  const id = Number(result.rows[0].id);
  logger.info(`Created session ${id} for channel ${channelName}`);
  return id;
}

export async function endSession(sessionId: number): Promise<void> {
  await pool.query(
    `UPDATE sessions SET ended_at = NOW() WHERE id = $1`,
    [sessionId]
  );
  logger.info(`Ended session ${sessionId}`);
}

export interface SessionRow {
  id: number;
  startedAt: string;
  endedAt: string | null;
  channelName: string | null;
}

export async function listSessions(): Promise<SessionRow[]> {
  const result = await pool.query(
    `SELECT id, started_at, ended_at, channel_name FROM sessions ORDER BY id DESC`
  );
  return result.rows.map(r => ({
    id: Number(r.id),
    startedAt: r.started_at,
    endedAt: r.ended_at,
    channelName: r.channel_name,
  }));
}

export async function getLatestSession(): Promise<SessionRow | null> {
  const result = await pool.query(
    `SELECT id, started_at, ended_at, channel_name FROM sessions ORDER BY id DESC LIMIT 1`
  );
  if (result.rows.length === 0) return null;
  const r = result.rows[0];
  return {
    id: Number(r.id),
    startedAt: r.started_at,
    endedAt: r.ended_at,
    channelName: r.channel_name,
  };
}

// ---- Transcriptions ----

export interface InsertTranscriptionParams {
  sessionId: number;
  discordUserId: string;
  discordUsername: string;
  nickname: string | null;
  spokenText: string;
  audioFilenames: string[];
  rms: number | null;
  durationMs: number | null;
  confidence: number | null;
}

export async function insertTranscription(params: InsertTranscriptionParams): Promise<number> {
  const result = await pool.query(
    `INSERT INTO transcriptions
      (session_id, discord_user_id, discord_username, nickname, spoken_text, audio_filenames, rms, duration_ms, confidence)
     VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
     RETURNING id`,
    [
      params.sessionId,
      params.discordUserId,
      params.discordUsername,
      params.nickname,
      params.spokenText,
      params.audioFilenames,
      params.rms,
      params.durationMs,
      params.confidence,
    ]
  );
  return Number(result.rows[0].id);
}

export interface TranscriptionRow {
  id: number;
  createdAt: string;
  discordUserId: string;
  discordUsername: string;
  nickname: string | null;
  spokenText: string;
  audioFilenames: string[];
  rms: number | null;
  durationMs: number | null;
  confidence: number | null;
  editedAt: string | null;
}

export async function getTranscriptionsBySession(sessionId: number): Promise<TranscriptionRow[]> {
  const result = await pool.query(
    `SELECT id, created_at, discord_user_id, discord_username, nickname, spoken_text,
            audio_filenames, rms, duration_ms, confidence, edited_at
     FROM transcriptions WHERE session_id = $1 ORDER BY created_at ASC`,
    [sessionId]
  );
  return result.rows.map(mapTranscriptionRow);
}

export async function getTranscriptionById(id: number): Promise<TranscriptionRow | null> {
  const result = await pool.query(
    `SELECT id, created_at, discord_user_id, discord_username, nickname, spoken_text,
            audio_filenames, rms, duration_ms, confidence, edited_at
     FROM transcriptions WHERE id = $1`,
    [id]
  );
  if (result.rows.length === 0) return null;
  return mapTranscriptionRow(result.rows[0]);
}

export async function updateTranscription(id: number, spokenText: string): Promise<void> {
  await pool.query(
    `UPDATE transcriptions SET spoken_text = $1, edited_at = NOW() WHERE id = $2`,
    [spokenText, id]
  );
}

/** Bulk update nicknames on transcription rows by discord_user_id within given line IDs. */
export async function bulkUpdateNicknames(lineIds: number[], nicknameMap: Record<string, string | null>): Promise<void> {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');
    for (const [discordUserId, nickname] of Object.entries(nicknameMap)) {
      await client.query(
        `UPDATE transcriptions SET nickname = $1 WHERE id = ANY($2) AND discord_user_id = $3`,
        [nickname, lineIds, discordUserId]
      );
    }
    await client.query('COMMIT');
  } catch (e) {
    await client.query('ROLLBACK');
    throw e;
  } finally {
    client.release();
  }
}

/** Search transcriptions across all sessions. */
export async function searchTranscriptions(query: string, limit = 50): Promise<(TranscriptionRow & { sessionId: number })[]> {
  const result = await pool.query(
    `SELECT id, session_id, created_at, discord_user_id, discord_username, nickname, spoken_text,
            audio_filenames, rms, duration_ms, confidence, edited_at
     FROM transcriptions WHERE spoken_text ILIKE $1 ORDER BY created_at DESC LIMIT $2`,
    [`%${query}%`, limit]
  );
  return result.rows.map(r => ({
    ...mapTranscriptionRow(r),
    sessionId: Number(r.session_id),
  }));
}

/** Get the most recent transcription for a user in a session. */
export async function getLastTranscriptionForUser(sessionId: number, discordUserId: string): Promise<TranscriptionRow | null> {
  const result = await pool.query(
    `SELECT id, created_at, discord_user_id, discord_username, nickname, spoken_text,
            audio_filenames, rms, duration_ms, confidence, edited_at
     FROM transcriptions WHERE session_id = $1 AND discord_user_id = $2
     ORDER BY created_at DESC LIMIT 1`,
    [sessionId, discordUserId]
  );
  if (result.rows.length === 0) return null;
  return mapTranscriptionRow(result.rows[0]);
}

/** Append text to an existing transcription, add audio file to array, accumulate duration. */
export async function appendToTranscription(id: number, appendText: string, audioFilename: string | null, rms: number | null, durationMs: number | null): Promise<void> {
  let query = `UPDATE transcriptions SET spoken_text = spoken_text || ' ' || $1, edited_at = NOW()`;
  const params: any[] = [appendText];
  let paramIdx = 2;

  // Append audio filename to the array (not overwrite)
  if (audioFilename) {
    query += `, audio_filenames = audio_filenames || $${paramIdx}::text[]`;
    params.push([audioFilename]);
    paramIdx++;
  }
  if (rms != null) {
    query += `, rms = $${paramIdx}`;
    params.push(rms);
    paramIdx++;
  }
  if (durationMs != null) {
    query += `, duration_ms = COALESCE(duration_ms, 0) + $${paramIdx}`;
    params.push(durationMs);
    paramIdx++;
  }

  query += ` WHERE id = $${paramIdx}`;
  params.push(id);

  await pool.query(query, params);
}

export async function deleteTranscription(id: number): Promise<string[] | null> {
  const result = await pool.query(
    `DELETE FROM transcriptions WHERE id = $1 RETURNING audio_filenames`,
    [id]
  );
  if (result.rowCount === 0) return null;
  return result.rows[0].audio_filenames || [];
}

function mapTranscriptionRow(r: any): TranscriptionRow {
  return {
    id: Number(r.id),
    createdAt: r.created_at,
    discordUserId: r.discord_user_id,
    discordUsername: r.discord_username,
    nickname: r.nickname,
    spokenText: r.spoken_text,
    audioFilenames: r.audio_filenames || [],
    rms: r.rms != null ? Number(r.rms) : null,
    durationMs: r.duration_ms != null ? Number(r.duration_ms) : null,
    confidence: r.confidence != null ? Number(r.confidence) : null,
    editedAt: r.edited_at,
  };
}

// ---- Ignored Users ----

export interface IgnoredUserRow {
  discordUserId: string;
  discordUsername: string;
  ignoredAt: string;
}

export async function getIgnoredUsers(): Promise<IgnoredUserRow[]> {
  const result = await pool.query(
    `SELECT discord_user_id, discord_username, ignored_at FROM ignored_users ORDER BY ignored_at ASC`
  );
  return result.rows.map(r => ({
    discordUserId: r.discord_user_id,
    discordUsername: r.discord_username,
    ignoredAt: r.ignored_at,
  }));
}

export async function addIgnoredUser(discordUserId: string, discordUsername: string): Promise<boolean> {
  const result = await pool.query(
    `INSERT INTO ignored_users (discord_user_id, discord_username)
     VALUES ($1, $2) ON CONFLICT (discord_user_id) DO NOTHING`,
    [discordUserId, discordUsername]
  );
  return (result.rowCount ?? 0) > 0;
}

export async function removeIgnoredUser(discordUserId: string): Promise<boolean> {
  const result = await pool.query(
    `DELETE FROM ignored_users WHERE discord_user_id = $1`,
    [discordUserId]
  );
  return (result.rowCount ?? 0) > 0;
}

// ---- Hotwords ----

export async function getHotwordsDB(): Promise<string[]> {
  const result = await pool.query(`SELECT word FROM hotwords ORDER BY created_at ASC`);
  return result.rows.map(r => r.word);
}

export async function addHotwordDB(word: string): Promise<boolean> {
  try {
    await pool.query(
      `INSERT INTO hotwords (word) VALUES ($1) ON CONFLICT (word) DO NOTHING`,
      [word]
    );
    return true;
  } catch {
    return false;
  }
}

export async function removeHotwordDB(word: string): Promise<boolean> {
  const result = await pool.query(
    `DELETE FROM hotwords WHERE LOWER(word) = LOWER($1)`,
    [word]
  );
  return (result.rowCount ?? 0) > 0;
}

export async function updateHotwordDB(oldWord: string, newWord: string): Promise<boolean> {
  const result = await pool.query(
    `UPDATE hotwords SET word = $1 WHERE LOWER(word) = LOWER($2)`,
    [newWord, oldWord]
  );
  return (result.rowCount ?? 0) > 0;
}

// ---- Character Names ----

export async function getCharacterNamesDB(): Promise<Map<string, string>> {
  const result = await pool.query(`SELECT discord_user_id, name FROM character_names`);
  const map = new Map<string, string>();
  for (const r of result.rows) map.set(r.discord_user_id, r.name);
  return map;
}

export async function setCharacterNameDB(discordUserId: string, name: string | null): Promise<void> {
  if (name) {
    await pool.query(
      `INSERT INTO character_names (discord_user_id, name) VALUES ($1, $2)
       ON CONFLICT (discord_user_id) DO UPDATE SET name = $2, updated_at = NOW()`,
      [discordUserId, name]
    );
  } else {
    await pool.query(`DELETE FROM character_names WHERE discord_user_id = $1`, [discordUserId]);
  }
}

// ---- Nickname Presets (Soundboard) ----

export interface NicknamePreset {
  id: number;
  discordUserId: string;
  label: string;
  position: number;
  categoryId: number | null;
}

export async function getNicknamePresets(discordUserId?: string): Promise<NicknamePreset[]> {
  let query = `SELECT id, discord_user_id, label, position, category_id FROM nickname_presets`;
  const params: any[] = [];
  if (discordUserId) {
    query += ` WHERE discord_user_id = $1`;
    params.push(discordUserId);
  }
  query += ` ORDER BY discord_user_id, position`;
  const result = await pool.query(query, params);
  return result.rows.map(r => ({
    id: Number(r.id),
    discordUserId: r.discord_user_id,
    label: r.label,
    position: r.position,
    categoryId: r.category_id ? Number(r.category_id) : null,
  }));
}

export async function addNicknamePreset(discordUserId: string, label: string, position: number): Promise<number> {
  const result = await pool.query(
    `INSERT INTO nickname_presets (discord_user_id, label, position) VALUES ($1, $2, $3) RETURNING id`,
    [discordUserId, label, position]
  );
  return Number(result.rows[0].id);
}

export async function updateNicknamePreset(id: number, label: string, position?: number): Promise<void> {
  if (position != null) {
    await pool.query(`UPDATE nickname_presets SET label = $1, position = $2 WHERE id = $3`, [label, position, id]);
  } else {
    await pool.query(`UPDATE nickname_presets SET label = $1 WHERE id = $2`, [label, id]);
  }
}

export async function deleteNicknamePreset(id: number): Promise<void> {
  await pool.query(`DELETE FROM nickname_presets WHERE id = $1`, [id]);
}

// ---- Nickname Categories ----

export interface NicknameCategory {
  id: number;
  discordUserId: string;
  name: string;
  position: number;
}

export async function getNicknameCategories(discordUserId?: string): Promise<NicknameCategory[]> {
  let query = `SELECT id, discord_user_id, name, position FROM nickname_categories`;
  const params: any[] = [];
  if (discordUserId) {
    query += ` WHERE discord_user_id = $1`;
    params.push(discordUserId);
  }
  query += ` ORDER BY position`;
  const result = await pool.query(query, params);
  return result.rows.map(r => ({
    id: Number(r.id),
    discordUserId: r.discord_user_id,
    name: r.name,
    position: r.position,
  }));
}

export async function addNicknameCategory(discordUserId: string, name: string, position: number): Promise<number> {
  const result = await pool.query(
    `INSERT INTO nickname_categories (discord_user_id, name, position) VALUES ($1, $2, $3) RETURNING id`,
    [discordUserId, name, position]
  );
  return Number(result.rows[0].id);
}

export async function updateNicknameCategory(id: number, name: string, position?: number): Promise<void> {
  if (position != null) {
    await pool.query(`UPDATE nickname_categories SET name = $1, position = $2 WHERE id = $3`, [name, position, id]);
  } else {
    await pool.query(`UPDATE nickname_categories SET name = $1 WHERE id = $2`, [name, id]);
  }
}

export async function deleteNicknameCategory(id: number): Promise<void> {
  // Move presets back to uncategorized (null)
  await pool.query(`UPDATE nickname_presets SET category_id = NULL WHERE category_id = $1`, [id]);
  await pool.query(`DELETE FROM nickname_categories WHERE id = $1`, [id]);
}

export async function movePresetToCategory(presetId: number, categoryId: number | null): Promise<void> {
  await pool.query(`UPDATE nickname_presets SET category_id = $1 WHERE id = $2`, [categoryId, presetId]);
}

// ---- Session Management ----

/** Delete a session and all its transcriptions. Returns audio filenames for cleanup. */
export async function deleteSessionFull(sessionId: number): Promise<string[]> {
  // Get all audio filenames before deleting
  const audioResult = await pool.query(
    `SELECT audio_filenames FROM transcriptions WHERE session_id = $1`,
    [sessionId]
  );
  const audioFiles: string[] = audioResult.rows.flatMap(r => r.audio_filenames || []);

  await pool.query(`DELETE FROM transcriptions WHERE session_id = $1`, [sessionId]);
  await pool.query(`DELETE FROM sessions WHERE id = $1`, [sessionId]);

  return audioFiles;
}

/** Merge multiple sessions into a new one. Returns the new session ID and audio files from deleted sessions. */
export async function mergeSessions(sessionIds: number[], name: string): Promise<{ newSessionId: number }> {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    // Create new session
    const sessionResult = await client.query(
      `INSERT INTO sessions (channel_name, started_at, ended_at)
       VALUES ($1, NOW(), NOW()) RETURNING id`,
      [name]
    );
    const newSessionId = Number(sessionResult.rows[0].id);

    // Reassign transcriptions from each source session in order.
    // To preserve the user-specified session order, we adjust created_at
    // by adding a tiny offset per session so they sort correctly even if
    // sessions overlapped in time.
    for (let i = 0; i < sessionIds.length; i++) {
      const sid = sessionIds[i];
      // Add i milliseconds to created_at to guarantee session ordering
      await client.query(
        `UPDATE transcriptions SET session_id = $1, created_at = created_at + ($2 || ' milliseconds')::interval
         WHERE session_id = $3`,
        [newSessionId, i, sid]
      );
    }

    // Delete the now-empty source sessions
    for (const sid of sessionIds) {
      await client.query(`DELETE FROM sessions WHERE id = $1`, [sid]);
    }

    await client.query('COMMIT');
    return { newSessionId };
  } catch (e) {
    await client.query('ROLLBACK');
    throw e;
  } finally {
    client.release();
  }
}

/** Get first and last N lines of a session for preview. */
export async function getSessionPreview(sessionId: number, count = 3): Promise<{ first: TranscriptionRow[]; last: TranscriptionRow[] }> {
  const firstResult = await pool.query(
    `SELECT id, created_at, discord_user_id, discord_username, nickname, spoken_text,
            audio_filenames, rms, duration_ms, confidence, edited_at
     FROM transcriptions WHERE session_id = $1 ORDER BY created_at ASC LIMIT $2`,
    [sessionId, count]
  );
  const lastResult = await pool.query(
    `SELECT id, created_at, discord_user_id, discord_username, nickname, spoken_text,
            audio_filenames, rms, duration_ms, confidence, edited_at
     FROM transcriptions WHERE session_id = $1 ORDER BY created_at DESC LIMIT $2`,
    [sessionId, count]
  );

  return {
    first: firstResult.rows.map(mapTranscriptionRow),
    last: lastResult.rows.reverse().map(mapTranscriptionRow),
  };
}
