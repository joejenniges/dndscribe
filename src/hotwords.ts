import { writeFileSync } from 'fs';
import { logger } from './logger';
import { getHotwordsDB, addHotwordDB, removeHotwordDB, updateHotwordDB } from './db';

const HOTWORDS_FILE = 'hotwords.txt';

// Common English words that should never be auto-added as hotwords.
const COMMON_WORDS = new Set([
  'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
  'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her',
  'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
  'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
  'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
  'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
  'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
  'about', 'against', 'between', 'through', 'during', 'before', 'after', 'above',
  'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
  'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
  'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
  'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can',
  'will', 'just', 'don', 'should', 'now', 'also', 'back', 'could', 'would',
  'into', 'well', 'like', 'right', 'yeah', 'okay', 'oh', 'um', 'uh', 'yes',
  'no', 'got', 'get', 'go', 'going', 'gone', 'come', 'came', 'know', 'think',
  'want', 'look', 'make', 'say', 'said', 'tell', 'told', 'take', 'took', 'see',
  'saw', 'give', 'gave', 'put', 'let', 'keep', 'kept', 'still', 'try', 'tried',
  'thing', 'things', 'something', 'nothing', 'everything', 'anything', 'someone',
  'everyone', 'anyone', 'people', 'man', 'woman', 'one', 'two', 'three', 'four',
  'five', 'first', 'last', 'new', 'old', 'good', 'bad', 'great', 'little', 'big',
  'long', 'way', 'day', 'time', 'part', 'much', 'many', 'really', 'actually',
]);

/**
 * Sync hotwords from DB to the text file that whisper_server.py reads.
 * Called after every add/remove/update so the Python server picks up changes.
 */
async function syncToFile(): Promise<void> {
  try {
    const words = await getHotwordsDB();
    writeFileSync(HOTWORDS_FILE, words.length > 0 ? words.join('\n') + '\n' : '');
  } catch (e) {
    logger.error(`Error syncing hotwords to file: ${e}`);
  }
}

export async function getHotwords(): Promise<string[]> {
  return getHotwordsDB();
}

export async function addHotword(word: string): Promise<boolean> {
  const trimmed = word.trim();
  if (!trimmed) return false;
  const added = await addHotwordDB(trimmed);
  if (added) {
    await syncToFile();
    logger.info(`Added hotword: ${trimmed}`);
  }
  return added;
}

export async function removeHotword(word: string): Promise<boolean> {
  const trimmed = word.trim();
  const removed = await removeHotwordDB(trimmed);
  if (removed) {
    await syncToFile();
    logger.info(`Removed hotword: ${trimmed}`);
  }
  return removed;
}

export async function updateHotword(oldWord: string, newWord: string): Promise<boolean> {
  const updated = await updateHotwordDB(oldWord.trim(), newWord.trim());
  if (updated) {
    await syncToFile();
    logger.info(`Updated hotword: ${oldWord} -> ${newWord}`);
  }
  return updated;
}

/** Import existing hotwords.txt into DB on first run. */
export async function migrateHotwordsToDB(): Promise<void> {
  const { existsSync, readFileSync } = await import('fs');
  const dbWords = await getHotwordsDB();
  if (dbWords.length > 0) return; // Already have data in DB

  if (existsSync(HOTWORDS_FILE)) {
    const fileWords = readFileSync(HOTWORDS_FILE, 'utf-8').split('\n').map(l => l.trim()).filter(Boolean);
    for (const word of fileWords) {
      await addHotwordDB(word);
    }
    if (fileWords.length > 0) {
      logger.info(`Migrated ${fileWords.length} hotwords from file to DB`);
    }
  }
}

/**
 * Compare original and edited transcription text. Find words in the edited
 * version that look like proper noun corrections.
 */
export function detectNewHotwords(originalText: string, editedText: string): string[] {
  const strip = (s: string) => s.replace(/[.,!?;:"'()[\]{}]/g, '');
  const origWords = new Set(originalText.split(/\s+/).map(w => strip(w).toLowerCase()));

  const candidates: string[] = [];
  for (const raw of editedText.split(/\s+/)) {
    const word = strip(raw);
    if (!word || word.length < 2) continue;
    if (word[0] !== word[0].toUpperCase() || word[0] === word[0].toLowerCase()) continue;
    if (origWords.has(word.toLowerCase())) continue;
    if (COMMON_WORDS.has(word.toLowerCase())) continue;
    candidates.push(word);
  }

  const seen = new Set<string>();
  return candidates.filter(w => {
    const lower = w.toLowerCase();
    if (seen.has(lower)) return false;
    seen.add(lower);
    return true;
  });
}
