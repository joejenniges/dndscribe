// Small fuzzy matcher for short names (soundboard pads, tab titles).
//
// Each word of the query must match the text one of three ways, best wins:
//   1. substring     "slam" in "Door slam"          (strongest)
//   2. in order      "dslm" in "Door slam", or initials "ds"
//   3. typo-tolerant "dragn" ~ "dragon"  (1 edit for words of 4+ letters,
//                    2 for 7+; transpositions count as one edit)
// Words of the query are ANDed; a word that matches nothing rejects the
// text. Case, accents and _ - . separators are ignored, so file-derived
// names like "door_slam-02" match "door slam".
//
// WHY hand-rolled: the lists are tiny (hundreds of pads at most) and the
// rules above are all we need; a library would add weight and scoring we'd
// have to tune around anyway.

export function normalize(s: string): string {
  return s
    .normalize('NFKD')
    .replace(/[̀-ͯ]/g, '')
    .toLowerCase()
    .replace(/[_\-.,/\\()[\]]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

/** Optimal string alignment distance (Levenshtein + adjacent swaps),
 *  giving up early once it exceeds max. */
function editDistance(a: string, b: string, max: number): number {
  if (Math.abs(a.length - b.length) > max) return max + 1;
  const n = a.length, m = b.length;
  let prev2 = new Array<number>(m + 1).fill(0);
  let prev = Array.from({ length: m + 1 }, (_, j) => j);
  for (let i = 1; i <= n; i++) {
    const cur = new Array<number>(m + 1);
    cur[0] = i;
    let rowMin = cur[0];
    for (let j = 1; j <= m; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      let v = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost);
      if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) v = Math.min(v, prev2[j - 2] + 1);
      cur[j] = v;
      if (v < rowMin) rowMin = v;
    }
    if (rowMin > max) return max + 1;
    prev2 = prev;
    prev = cur;
  }
  return prev[m];
}

const isStart = (t: string, i: number) => i === 0 || t[i - 1] === ' ';

/** Score one query word against normalized text, or 0 for no match. */
function wordScore(q: string, text: string, words: string[]): number {
  // 1. Substring.
  const at = text.indexOf(q);
  if (at >= 0) {
    let s = 100;
    if (isStart(text, at)) s += 30;
    if (words.includes(q)) s += 20;
    return s;
  }
  // 2. Characters in order, scored by how tight and word-aligned they are.
  //    Short queries (2-3 letters) only count as initials or word-start
  //    runs ("fb" -> FireBall, "ds" -> Door Slam); anywhere-in-order is too
  //    loose at that length.
  if (q.length >= 2) {
    const initials = q.length <= 3;
    let ti = 0, s = 0, run = 0, first = -1, last = -1;
    for (const c of q) {
      let found: number;
      if (initials) {
        found = last >= 0 && text[last + 1] === c ? last + 1 : -1;
        for (let k = ti; found < 0 && k < text.length; k++) if (text[k] === c && isStart(text, k)) found = k;
      } else found = text.indexOf(c, ti);
      if (found < 0) {
        s = -1;
        break;
      }
      if (first < 0) first = found;
      run = found === last + 1 ? run + 1 : 0;
      s += 4 + run * 3 + (isStart(text, found) ? 8 : 0);
      last = found;
      ti = found + 1;
    }
    // A sprawling match ("ae" across a long title) is noise, not intent.
    if (s >= 0 && (initials || last - first + 1 <= q.length * 2 + 1)) return Math.min(90, 20 + s);
  }
  // 3. Typos: against each word, and each word's prefix of the same length
  //    (so a half-typed "dargo" still finds "dragons").
  if (q.length >= 4) {
    const max = q.length >= 7 ? 2 : 1;
    let best = max + 1;
    for (const w of words) {
      best = Math.min(best, editDistance(q, w, max));
      if (w.length > q.length) best = Math.min(best, editDistance(q, w.slice(0, q.length), max));
      if (best === 0) break;
    }
    if (best <= max) return 45 - best * 12;
  }
  return 0;
}

export interface FuzzyField {
  text: string;
  /** Multiplier for matches in this field (e.g. 0.6 for a tab name). */
  weight?: number;
}

/** Score a query against one or more fields; 0 means no match. Each query
 *  word may match any field. */
export function fuzzyScore(query: string, fields: FuzzyField[]): number {
  const qWords = normalize(query).split(' ').filter(Boolean);
  if (qWords.length === 0) return 0;
  const prepared = fields.map((f) => {
    const text = normalize(f.text);
    return { text, words: text.split(' ').filter(Boolean), weight: f.weight ?? 1 };
  });
  let total = 0;
  for (const q of qWords) {
    let best = 0;
    for (const f of prepared) best = Math.max(best, wordScore(q, f.text, f.words) * f.weight);
    if (best === 0) return 0;
    total += best;
  }
  // Prefer the shorter primary text on ties: "Slam" over "Door slam echo".
  return total - (prepared[0]?.text.length ?? 0) * 0.05;
}

/** Items matching the query, best first (stable for equal scores). */
export function fuzzyFilter<T>(items: T[], query: string, fields: (item: T) => FuzzyField[]): T[] {
  return items
    .map((item, i) => ({ item, i, s: fuzzyScore(query, fields(item)) }))
    .filter((r) => r.s > 0)
    .sort((a, b) => b.s - a.s || a.i - b.i)
    .map((r) => r.item);
}
