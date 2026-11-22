// Waveforms shared by both decks: a track moving from the queue to a deck,
// or between decks, is drawn without refetching.
import { dj } from '$lib/api';
import type { DjWaveform } from '$lib/types';

const cache = new Map<number, Promise<DjWaveform>>();
const MAX = 16;

export const waveCache = {
  load(id: number): Promise<DjWaveform> {
    let p = cache.get(id);
    if (!p) {
      p = dj.waveform(id);
      p.catch(() => cache.delete(id)); // let a failed one be retried
      cache.set(id, p);
      while (cache.size > MAX) cache.delete(cache.keys().next().value!);
    }
    return p;
  },
};
