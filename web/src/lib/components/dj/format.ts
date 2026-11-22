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

/** Colours offered for scenes and pads, in hue order. Mid-to-light tones so
 *  they hold up both as a tint behind white text and as a border on the
 *  dark theme. */
export const PALETTE: { name: string; hex: string }[] = [
  { name: 'Red', hex: '#f43f62' },
  { name: 'Crimson', hex: '#c0263b' },
  { name: 'Orange', hex: '#f97316' },
  { name: 'Amber', hex: '#f2b45c' },
  { name: 'Yellow', hex: '#facc15' },
  { name: 'Lime', hex: '#a3e635' },
  { name: 'Mint', hex: '#74d7a9' },
  { name: 'Green', hex: '#16a34a' },
  { name: 'Teal', hex: '#14b8a6' },
  { name: 'Cyan', hex: '#22d3ee' },
  { name: 'Sky', hex: '#7cb6fa' },
  { name: 'Blue', hex: '#2563eb' },
  { name: 'Indigo', hex: '#6366f1' },
  { name: 'Violet', hex: '#b18cf5' },
  { name: 'Purple', hex: '#9333ea' },
  { name: 'Pink', hex: '#ec4899' },
  { name: 'Brown', hex: '#a0703c' },
  { name: 'Grey', hex: '#9ca1aa' },
];

/** The original six. New scenes start on the first, and pads with no
 *  colour cycle through these by id, so existing pads keep their colour. */
export const SCENE_COLORS = ['#f43f62', '#f2b45c', '#74d7a9', '#7cb6fa', '#b18cf5', '#9ca1aa'];

export const FADE_OPTIONS = [0, 2, 4, 6, 8, 12, 20];
