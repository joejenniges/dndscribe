import type { DjTrack } from '$lib/types';

export function fmtMs(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) ms = 0;
  const total = Math.floor(ms / 1000);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = String(total % 60).padStart(2, '0');
  return h > 0 ? `${h}:${String(m).padStart(2, '0')}:${s}` : `${m}:${s}`;
}

/** Display title: the tag if present, else the file name without extension. */
export function trackTitle(t: Pick<DjTrack, 'title' | 'path'>): string {
  if (t.title?.trim()) return t.title;
  const base = t.path.split('/').pop() ?? t.path;
  return base.replace(/\.[^.]+$/, '');
}

export const SCENE_COLORS = ['#f43f62', '#f2b45c', '#74d7a9', '#7cb6fa', '#b18cf5', '#9ca1aa'];

export const FADE_OPTIONS = [0, 2, 4, 6, 8, 12, 20];
