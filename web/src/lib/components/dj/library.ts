import type { DjTrack } from '$lib/types';

// Folder helpers. Folders under the server's music dir are the playlists;
// paths are music-dir-relative with forward slashes ("Combat/Boss").

export const AUDIO_EXTS = ['mp3', 'ogg', 'opus', 'flac', 'wav', 'm4a', 'aac', 'webm', 'wma'];

export function isAudioName(name: string): boolean {
  const ext = name.split('.').pop()?.toLowerCase() ?? '';
  return name.includes('.') && AUDIO_EXTS.includes(ext);
}

export const folderName = (p: string) => (p === '' ? 'Music' : p.split('/').pop()!);
export const folderDepth = (p: string) => (p === '' ? 0 : p.split('/').length);
export const parentOf = (p: string) => (p.includes('/') ? p.slice(0, p.lastIndexOf('/')) : '');
export const inFolder = (trackPath: string, folder: string) => folder === '' || trackPath.startsWith(folder + '/');

export interface FolderStats {
  count: number; // tracks in this folder and its sub-folders
  ms: number;
}

/** Recursive track count and duration for each folder. */
export function folderStats(folders: string[], tracks: DjTrack[]): Map<string, FolderStats> {
  const out = new Map<string, FolderStats>(folders.map((f) => [f, { count: 0, ms: 0 }]));
  for (const t of tracks) {
    // Credit the track to every ancestor folder.
    let dir = parentOf(t.path);
    while (dir !== '') {
      const s = out.get(dir);
      if (s) {
        s.count++;
        s.ms += t.durationMs;
      }
      dir = parentOf(dir);
    }
  }
  return out;
}

/** Tracks under any of folders, each once (folders may overlap). */
export function tracksIn(folders: string[], tracks: DjTrack[]): DjTrack[] {
  return tracks.filter((t) => folders.some((f) => inFolder(t.path, f)));
}

// ---- Collecting files for upload ----

/** A file picked or dropped, with its path relative to what was picked
 * ("Battle/Boss/x.mp3" for a dropped folder, "x.mp3" for a loose file). */
export interface PickedFile {
  file: File;
  rel: string;
}

/** From an <input type="file"> (webkitRelativePath is set for folder picks). */
export function fromFileList(list: FileList | null): PickedFile[] {
  return Array.from(list ?? []).map((file) => ({ file, rel: file.webkitRelativePath || file.name }));
}

/**
 * From a drop event, walking dropped folders. Entries must be taken from
 * the DataTransfer synchronously, inside the event handler: the browser
 * empties it as soon as the handler yields.
 */
export function fromDrop(dt: DataTransfer): Promise<PickedFile[]> {
  const entries: FileSystemEntry[] = [];
  const loose: File[] = [];
  for (const item of Array.from(dt.items)) {
    if (item.kind !== 'file') continue;
    const entry = item.webkitGetAsEntry?.();
    if (entry) entries.push(entry);
    else {
      const f = item.getAsFile();
      if (f) loose.push(f);
    }
  }
  return (async () => {
    const out: PickedFile[] = loose.map((file) => ({ file, rel: file.name }));
    for (const e of entries) await walk(e, '', out);
    return out;
  })();
}

async function walk(entry: FileSystemEntry, prefix: string, out: PickedFile[]) {
  const rel = prefix ? `${prefix}/${entry.name}` : entry.name;
  if (entry.isFile) {
    const file = await new Promise<File>((res, rej) => (entry as FileSystemFileEntry).file(res, rej));
    out.push({ file, rel });
    return;
  }
  if (!entry.isDirectory) return;
  const reader = (entry as FileSystemDirectoryEntry).createReader();
  // readEntries returns batches (100 in Chrome); call until it returns none.
  for (;;) {
    const batch = await new Promise<FileSystemEntry[]>((res, rej) => reader.readEntries(res, rej));
    if (batch.length === 0) break;
    for (const child of batch) await walk(child, rel, out);
  }
}

export function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(0)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}
