package dj

import (
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/joe/dndscribe-go/internal/db"
)

// File management for the music folder: uploads, folders, deletes. Folders
// are the playlists, so these are the library's only write paths besides a
// rescan.
//
// Every path that comes from a client goes through CleanRelPath and then
// l.abs, which re-checks that the result is inside the music dir.

// ErrExists is returned by SaveFile when the target exists and overwrite is off.
var ErrExists = errors.New("a file with that name already exists")

// IsAudioFile reports whether name has an extension the library scans.
func IsAudioFile(name string) bool {
	return audioExts[strings.ToLower(path.Ext(name))]
}

var windowsReserved = map[string]bool{
	"con": true, "prn": true, "aux": true, "nul": true,
	"com1": true, "com2": true, "com3": true, "com4": true, "com5": true, "com6": true, "com7": true, "com8": true, "com9": true,
	"lpt1": true, "lpt2": true, "lpt3": true, "lpt4": true, "lpt5": true, "lpt6": true, "lpt7": true, "lpt8": true, "lpt9": true,
}

// CleanRelPath validates a client-supplied, music-dir-relative path and
// returns it with forward slashes. "" (the music dir itself) is returned as
// "" and is only valid where the caller allows it.
//
// WHY so strict: names must mean the same thing on disk as in the database.
// Windows silently strips trailing dots and spaces and maps CON/NUL/... to
// devices, so a name like "Boss. " would be stored as one path and exist as
// another. Rejecting them with a clear message beats a track that can never
// be found.
func CleanRelPath(p string) (string, error) {
	p = strings.ReplaceAll(strings.TrimSpace(p), "\\", "/")
	p = strings.Trim(p, "/")
	if p == "" {
		return "", nil
	}
	parts := strings.Split(p, "/")
	for _, part := range parts {
		switch {
		case part == "" || part == "." || part == "..":
			return "", fmt.Errorf("invalid path %q", p)
		case strings.ContainsAny(part, `<>:"|?*`):
			return "", fmt.Errorf("%q contains a character Windows doesn't allow in names (< > : \" | ? *)", part)
		case strings.HasSuffix(part, ".") || strings.HasSuffix(part, " ") || strings.HasPrefix(part, " "):
			return "", fmt.Errorf("%q starts or ends with a space or ends with a dot, which Windows doesn't keep", part)
		case windowsReserved[strings.ToLower(strings.SplitN(part, ".", 2)[0])]:
			return "", fmt.Errorf("%q is a reserved name on Windows", part)
		case len(part) > 200:
			return "", fmt.Errorf("name too long: %q", part)
		}
		for _, r := range part {
			if r < 0x20 {
				return "", fmt.Errorf("%q contains a control character", part)
			}
		}
	}
	return strings.Join(parts, "/"), nil
}

// abs maps a cleaned relative path to an absolute one inside the music dir.
func (l *Library) abs(rel string) (string, error) {
	root, err := filepath.Abs(l.musicDir)
	if err != nil {
		return "", err
	}
	full := filepath.Join(root, filepath.FromSlash(rel))
	r, err := filepath.Rel(root, full)
	if err != nil || r == ".." || strings.HasPrefix(r, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("path escapes the music folder")
	}
	return full, nil
}

// Folders lists every folder under the music dir (not the root itself), as
// sorted relative paths. Empty folders are included: a folder made for an
// upload that hasn't happened yet is still a place to put things.
func (l *Library) Folders() ([]string, error) {
	if err := os.MkdirAll(l.musicDir, 0o755); err != nil {
		return nil, err
	}
	out := []string{}
	err := filepath.WalkDir(l.musicDir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			if d != nil && d.IsDir() && p != l.musicDir {
				return fs.SkipDir
			}
			return nil
		}
		if !d.IsDir() || p == l.musicDir {
			return nil
		}
		if strings.HasPrefix(d.Name(), ".") {
			return fs.SkipDir
		}
		rel, err := filepath.Rel(l.musicDir, p)
		if err == nil {
			out = append(out, filepath.ToSlash(rel))
		}
		return nil
	})
	sort.Slice(out, func(i, j int) bool { return strings.ToLower(out[i]) < strings.ToLower(out[j]) })
	return out, err
}

// MakeFolder creates a folder (and any missing parents).
func (l *Library) MakeFolder(rel string) error {
	if rel == "" {
		return errors.New("folder name is required")
	}
	full, err := l.abs(rel)
	if err != nil {
		return err
	}
	return os.MkdirAll(full, 0o755)
}

// SaveFile writes an uploaded track to rel and adds it to the library.
//
// The body is streamed to a ".part" file next to the target and renamed into
// place only when complete, so a scan never probes a half-written file and an
// aborted upload never leaves a truncated track behind. (".part" isn't an
// audio extension, so scans skip it.)
func (l *Library) SaveFile(ctx context.Context, rel string, body io.Reader, overwrite bool) (*db.DJTrack, error) {
	if rel == "" || !IsAudioFile(rel) {
		return nil, fmt.Errorf("not an audio file: %q", rel)
	}
	full, err := l.abs(rel)
	if err != nil {
		return nil, err
	}
	if !overwrite {
		if _, err := os.Stat(full); err == nil {
			return nil, ErrExists
		}
	}
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		return nil, err
	}
	tmp, err := os.CreateTemp(filepath.Dir(full), "."+filepath.Base(full)+".*.part")
	if err != nil {
		return nil, err
	}
	tmpName := tmp.Name()
	_, copyErr := io.Copy(tmp, body)
	closeErr := tmp.Close()
	if copyErr != nil || closeErr != nil {
		os.Remove(tmpName)
		if copyErr != nil {
			return nil, fmt.Errorf("upload interrupted: %w", copyErr)
		}
		return nil, closeErr
	}

	// Rename, probe and save under the write lock so a concurrent rescan
	// can't walk the folder before the rename and prune the new row after it.
	l.writeMu.Lock()
	defer l.writeMu.Unlock()
	if !overwrite {
		if _, err := os.Stat(full); err == nil {
			os.Remove(tmpName)
			return nil, ErrExists
		}
	}
	if err := os.Rename(tmpName, full); err != nil {
		os.Remove(tmpName)
		return nil, err
	}
	info, err := os.Stat(full)
	if err != nil {
		return nil, err
	}
	t := Probe(l.ffmpeg, full)
	t.Path = rel
	t.SizeBytes = info.Size()
	t.ModTime = info.ModTime().Truncate(time.Microsecond)
	if t.ID, err = db.UpsertDJTrack(ctx, t); err != nil {
		return nil, err
	}
	return &t, nil
}

// DeleteFile removes one track's file and its library row.
func (l *Library) DeleteFile(ctx context.Context, rel string) error {
	if rel == "" {
		return errors.New("path is required")
	}
	full, err := l.abs(rel)
	if err != nil {
		return err
	}
	l.writeMu.Lock()
	defer l.writeMu.Unlock()
	if err := os.Remove(full); err != nil && !errors.Is(err, fs.ErrNotExist) {
		return friendlyRemoveErr(err)
	}
	return db.DeleteDJTrackPaths(ctx, []string{rel})
}

// DeleteFolder removes a folder with everything in it.
func (l *Library) DeleteFolder(ctx context.Context, rel string) error {
	if rel == "" {
		return errors.New("can't delete the whole music folder")
	}
	full, err := l.abs(rel)
	if err != nil {
		return err
	}
	l.writeMu.Lock()
	defer l.writeMu.Unlock()
	rmErr := os.RemoveAll(full)
	// Drop rows even on a partial failure (a file in use on Windows); the
	// next scan re-adds whatever is still on disk.
	if err := db.DeleteDJTracksUnder(ctx, rel); err != nil {
		return err
	}
	if rmErr != nil {
		return friendlyRemoveErr(rmErr)
	}
	return nil
}

// RenameFolder renames (or moves) a folder and rewrites the stored paths
// under it. A case-only change ("curses" -> "Curses") is allowed even
// though the target "exists" on a case-insensitive filesystem.
func (l *Library) RenameFolder(ctx context.Context, from, to string) error {
	if from == "" || to == "" {
		return errors.New("folder name is required")
	}
	if from == to {
		return nil
	}
	if strings.HasPrefix(to+"/", from+"/") {
		return errors.New("can't move a folder inside itself")
	}
	src, err := l.abs(from)
	if err != nil {
		return err
	}
	dst, err := l.abs(to)
	if err != nil {
		return err
	}
	l.writeMu.Lock()
	defer l.writeMu.Unlock()
	if fi, err := os.Stat(src); err != nil || !fi.IsDir() {
		return fmt.Errorf("folder %q not found", from)
	}
	if !strings.EqualFold(from, to) {
		if _, err := os.Stat(dst); err == nil {
			return fmt.Errorf("%q already exists", to)
		}
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	if err := os.Rename(src, dst); err != nil {
		// Windows refuses while ffmpeg has a file inside open.
		if strings.Contains(strings.ToLower(err.Error()), "used by another process") ||
			strings.Contains(strings.ToLower(err.Error()), "access is denied") {
			return errors.New("a file in this folder is in use (loaded on a deck or playing); eject it and try again")
		}
		return err
	}
	if err := db.RenameDJFolder(ctx, from, to); err != nil {
		// Keep disk and database in step: put the folder back.
		if rerr := os.Rename(dst, src); rerr != nil {
			return fmt.Errorf("%v (and moving the folder back failed: %v; rescan to recover)", err, rerr)
		}
		return err
	}
	return nil
}

func friendlyRemoveErr(err error) error {
	// Windows won't delete a file ffmpeg has open.
	if strings.Contains(strings.ToLower(err.Error()), "used by another process") {
		return errors.New("a file is in use (probably playing right now); stop the music and try again")
	}
	return err
}
