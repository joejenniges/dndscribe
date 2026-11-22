package db

import (
	"context"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
)

// DJ storage: a track library scanned from the music folder, and scenes that
// pick one or more folders of it with playback settings. Folders are the
// playlists: the library is organised on disk, so the folder tree is the one
// source of truth and there is no separate playlist table to drift from it.
// These are global rather than per-campaign: the same battle music gets reused
// across campaigns, and the DJ page is a top-level route.

type DJTrack struct {
	ID         int64     `json:"id"`
	Path       string    `json:"path"` // relative to the music dir, forward slashes
	Title      string    `json:"title"`
	Artist     string    `json:"artist"`
	DurationMs int64     `json:"durationMs"`
	Missing    bool      `json:"missing"`
	SizeBytes  int64     `json:"-"`
	ModTime    time.Time `json:"-"`
}

type DJScene struct {
	ID   int64  `json:"id"`
	Name string `json:"name"`
	// Folders are music-dir-relative folder paths ("Combat", "Combat/Boss").
	// A folder includes its sub-folders.
	Folders     []string `json:"folders"`
	Color       string   `json:"color"`
	Volume      float64  `json:"volume"` // per-scene gain, 0..1
	Shuffle     bool     `json:"shuffle"`
	Loop        bool     `json:"loop"`
	FadeSeconds *float64 `json:"fadeSeconds"` // nil = use the global fade
	Position    int      `json:"position"`
	// ImageKey is an optional background image, stored with the soundboard
	// images and served at /api/dj/sfx/images/{key}. Set only through the
	// image endpoints, never by a scene save.
	ImageKey *string `json:"image"`
}

func migrateDJ(ctx context.Context) error {
	stmts := []string{
		`CREATE TABLE IF NOT EXISTS dj_tracks (
			id BIGSERIAL PRIMARY KEY,
			path TEXT NOT NULL UNIQUE,
			title TEXT NOT NULL DEFAULT '',
			artist TEXT NOT NULL DEFAULT '',
			duration_ms BIGINT NOT NULL DEFAULT 0,
			size_bytes BIGINT NOT NULL DEFAULT 0,
			mod_time TIMESTAMPTZ,
			missing BOOLEAN NOT NULL DEFAULT FALSE,
			created_at TIMESTAMPTZ DEFAULT NOW()
		)`,
		`CREATE TABLE IF NOT EXISTS dj_scenes (
			id BIGSERIAL PRIMARY KEY,
			name TEXT NOT NULL,
			folders TEXT[] NOT NULL DEFAULT '{}',
			color TEXT NOT NULL DEFAULT '',
			volume REAL NOT NULL DEFAULT 1,
			shuffle BOOLEAN NOT NULL DEFAULT FALSE,
			loop BOOLEAN NOT NULL DEFAULT TRUE,
			fade_seconds REAL,
			position INT NOT NULL DEFAULT 0,
			created_at TIMESTAMPTZ DEFAULT NOW()
		)`,
		// The first DJ build had hand-made playlists that scenes pointed at.
		// Folders replaced them; no install had real data in them.
		`ALTER TABLE dj_scenes ADD COLUMN IF NOT EXISTS folders TEXT[] NOT NULL DEFAULT '{}'`,
		`ALTER TABLE dj_scenes DROP COLUMN IF EXISTS playlist_id`,
		`ALTER TABLE dj_scenes ADD COLUMN IF NOT EXISTS image_key TEXT`,
		`DROP TABLE IF EXISTS dj_playlist_tracks`,
		`DROP TABLE IF EXISTS dj_playlists`,
	}
	for _, s := range stmts {
		if _, err := pool.Exec(ctx, s); err != nil {
			return err
		}
	}
	return nil
}

// ---- Tracks ----

const trackCols = `id, path, title, artist, duration_ms, missing, size_bytes, COALESCE(mod_time, 'epoch'::timestamptz)`

func scanTrack(row pgx.Row) (DJTrack, error) {
	var t DJTrack
	err := row.Scan(&t.ID, &t.Path, &t.Title, &t.Artist, &t.DurationMs, &t.Missing, &t.SizeBytes, &t.ModTime)
	return t, err
}

func ListDJTracks(ctx context.Context) ([]DJTrack, error) {
	rows, err := pool.Query(ctx, `SELECT `+trackCols+` FROM dj_tracks ORDER BY LOWER(path) COLLATE "C"`)
	if err != nil {
		return nil, fmt.Errorf("list dj tracks: %w", err)
	}
	defer rows.Close()
	out := []DJTrack{}
	for rows.Next() {
		t, err := scanTrack(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, t)
	}
	return out, rows.Err()
}

func GetDJTrack(ctx context.Context, id int64) (*DJTrack, error) {
	t, err := scanTrack(pool.QueryRow(ctx, `SELECT `+trackCols+` FROM dj_tracks WHERE id = $1`, id))
	if err == pgx.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &t, nil
}

// UpsertDJTrack inserts or refreshes a track by path, clears its missing
// flag, and returns its id.
func UpsertDJTrack(ctx context.Context, t DJTrack) (int64, error) {
	var id int64
	err := pool.QueryRow(ctx, `
		INSERT INTO dj_tracks (path, title, artist, duration_ms, size_bytes, mod_time, missing)
		VALUES ($1, $2, $3, $4, $5, $6, FALSE)
		ON CONFLICT (path) DO UPDATE SET
			title = EXCLUDED.title, artist = EXCLUDED.artist, duration_ms = EXCLUDED.duration_ms,
			size_bytes = EXCLUDED.size_bytes, mod_time = EXCLUDED.mod_time, missing = FALSE
		RETURNING id`,
		t.Path, t.Title, t.Artist, t.DurationMs, t.SizeBytes, t.ModTime).Scan(&id)
	return id, err
}

// PruneDJTracks deletes tracks whose path is not in present: the file is
// gone, and with folders as playlists nothing else refers to a track row.
func PruneDJTracks(ctx context.Context, present []string) error {
	_, err := pool.Exec(ctx, `DELETE FROM dj_tracks WHERE NOT (path = ANY($1))`, present)
	return err
}

// DeleteDJTrackPaths deletes the given track rows (after their files are removed).
func DeleteDJTrackPaths(ctx context.Context, paths []string) error {
	_, err := pool.Exec(ctx, `DELETE FROM dj_tracks WHERE path = ANY($1)`, paths)
	return err
}

// DeleteDJTracksUnder deletes track rows under a folder (after it is removed).
func DeleteDJTracksUnder(ctx context.Context, folder string) error {
	_, err := pool.Exec(ctx, `DELETE FROM dj_tracks WHERE starts_with(path, $1)`, folder+"/")
	return err
}

// ---- Scenes ----

const sceneCols = `id, name, folders, color, volume, shuffle, loop, fade_seconds, position, image_key`

func scanScene(row pgx.Row) (DJScene, error) {
	var s DJScene
	var vol float32
	var fade *float32
	err := row.Scan(&s.ID, &s.Name, &s.Folders, &s.Color, &vol, &s.Shuffle, &s.Loop, &fade, &s.Position, &s.ImageKey)
	s.Volume = float64(vol)
	if fade != nil {
		f := float64(*fade)
		s.FadeSeconds = &f
	}
	return s, err
}

func ListDJScenes(ctx context.Context) ([]DJScene, error) {
	rows, err := pool.Query(ctx, `SELECT `+sceneCols+` FROM dj_scenes ORDER BY position, id`)
	if err != nil {
		return nil, fmt.Errorf("list dj scenes: %w", err)
	}
	defer rows.Close()
	out := []DJScene{}
	for rows.Next() {
		s, err := scanScene(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, s)
	}
	return out, rows.Err()
}

func GetDJScene(ctx context.Context, id int64) (*DJScene, error) {
	s, err := scanScene(pool.QueryRow(ctx, `SELECT `+sceneCols+` FROM dj_scenes WHERE id = $1`, id))
	if err == pgx.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &s, nil
}

func CreateDJScene(ctx context.Context, s DJScene) (*DJScene, error) {
	err := pool.QueryRow(ctx, `
		INSERT INTO dj_scenes (name, folders, color, volume, shuffle, loop, fade_seconds, position)
		VALUES ($1, $2, $3, $4, $5, $6, $7, COALESCE((SELECT MAX(position) + 1 FROM dj_scenes), 0))
		RETURNING id, position`,
		s.Name, s.Folders, s.Color, s.Volume, s.Shuffle, s.Loop, s.FadeSeconds).Scan(&s.ID, &s.Position)
	if err != nil {
		return nil, err
	}
	return &s, nil
}

func UpdateDJScene(ctx context.Context, s DJScene) (bool, error) {
	tag, err := pool.Exec(ctx, `
		UPDATE dj_scenes SET name = $2, folders = $3, color = $4, volume = $5,
			shuffle = $6, loop = $7, fade_seconds = $8
		WHERE id = $1`,
		s.ID, s.Name, s.Folders, s.Color, s.Volume, s.Shuffle, s.Loop, s.FadeSeconds)
	return tag.RowsAffected() > 0, err
}

// DeleteDJScene deletes a scene and returns its image key (if any) so the
// caller can remove the file.
func DeleteDJScene(ctx context.Context, id int64) (image *string, err error) {
	err = pool.QueryRow(ctx, `DELETE FROM dj_scenes WHERE id = $1 RETURNING image_key`, id).Scan(&image)
	if err == pgx.ErrNoRows {
		return nil, nil
	}
	return image, err
}

// SetDJSceneImage sets (or clears, with nil) a scene's image and returns the
// previous key so the caller can delete that file.
func SetDJSceneImage(ctx context.Context, id int64, key *string) (old *string, err error) {
	err = pool.QueryRow(ctx, `
		UPDATE dj_scenes s SET image_key = $2
		FROM (SELECT image_key FROM dj_scenes WHERE id = $1) prev
		WHERE s.id = $1 RETURNING prev.image_key`, id, key).Scan(&old)
	if err == pgx.ErrNoRows {
		return nil, fmt.Errorf("scene not found")
	}
	return old, err
}

// ReorderDJScenes sets each scene's position to its index in ids.
func ReorderDJScenes(ctx context.Context, ids []int64) error {
	tx, err := pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)
	for i, id := range ids {
		if _, err := tx.Exec(ctx, `UPDATE dj_scenes SET position = $2 WHERE id = $1`, id, i); err != nil {
			return err
		}
	}
	return tx.Commit(ctx)
}
