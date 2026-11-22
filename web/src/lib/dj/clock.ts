// A shared 4Hz clock for time readouts (deck times, progress rings, the
// status bar). Waveforms and the crossfader animate on their own rAF loops;
// this only drives text, so 4 updates a second is plenty and cheap.
import { readable } from 'svelte/store';

export const clock = readable(Date.now(), (set) => {
  const t = setInterval(() => set(Date.now()), 250);
  return () => clearInterval(t);
});
