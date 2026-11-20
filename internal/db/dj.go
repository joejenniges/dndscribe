package db

import (
	"context"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
)

// DJ storage: a track library scanned from the music folder, playlists that
// order tracks, and scenes that point at a playlist with playback settings.
// These are global rather than per-campaign: the same battle music gets reused
// across campaigns, and the DJ page is a top-level route.

type DJTrack struct {
	ID         int64  `json:"id"`
	Path       string `json:"path"` // relative to the music dir, forward slashes
	Title      string `json:"title"`
	Artist     string `json:"artist"`
	DurationMs int64  `json:"durationMs"`
	Missing    bool   `json:"missing"`
	SizeBytes  int64  `json:"-"`
	ModTime    time.Time `json:"-"`
}

type DJPlaylist struct {
	ID       int64   `json:"id"`
	Name     string  `json:"name"`
	TrackIDs []int64 `json:"trackIds"`
}

type DJScene struct {
	ID          int64    `json:"id"`
	Name        string   `json:"name"`
	PlaylistID  *int64   `json:"playlistId"`
	Color       string   `json:"color"`
	Volume      float64  `json:"volume"` // per-scene gain, 0..1
	Shuffle     bool     `json:"shuffle"`
	Loop        bool     `json:"loop"`
	FadeSeconds *float64 `json:"fadeSeconds"` // nil = use the global fade
	Position    int      `json:"position"`
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
		`CREATE TABLE IF NOT EXISTS dj_playlists (
			id BIGSERIAL PRIMARY KEY,
			name TEXT NOT NULL,
			created_at TIMESTAMPTZ DEFAULT NOW()
		)`,
		`CREATE TABLE IF NOT EXISTS dj_playlist_tracks (
			playlist_id BIGINT NOT NULL REFERENCES dj_playlists(id) ON DELETE CASCADE,
			track_id BIGINT NOT NULL REFERENCES dj_tracks(id) ON DELETE CASCADE,
			position INT NOT NULL,
			PRIMARY KEY (playlist_id, position)
		)`,
		`CREATE TABLE IF NOT EXISTS dj_scenes (
			id BIGSERIAL PRIMARY KEY,
			name TEXT NOT NULL,
			playlist_id BIGINT REFERENCES dj_playlists(id) ON DELETE SET NULL,
			color TEXT NOT NULL DEFAULT '',
			volume REAL NOT NULL DEFAULT 1,
			shuffle BOOLEAN NOT NULL DEFAULT FALSE,
			loop BOOLEAN NOT NULL DEFAULT TRUE,
			fade_seconds REAL,
			position INT NOT NULL DEFAULT 0,
			created_at TIMESTAMPTZ DEFAULT NOW()
		)`,
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
	rows, err := pool.Query(ctx, `SELECT `+trackCols+` FROM dj_tracks ORDER BY LOWER(path)`)
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

// UpsertDJTrack inserts or refreshes a track by path and clears its missing flag.
func UpsertDJTrack(ctx context.Context, t DJTrack) error {
	_, err := pool.Exec(ctx, `
		INSERT INTO dj_tracks (path, title, artist, duration_ms, size_bytes, mod_time, missing)
		VALUES ($1, $2, $3, $4, $5, $6, FALSE)
		ON CONFLICT (path) DO UPDATE SET
			title = EXCLUDED.title, artist = EXCLUDED.artist, duration_ms = EXCLUDED.duration_ms,
			size_bytes = EXCLUDED.size_bytes, mod_time = EXCLUDED.mod_time, missing = FALSE`,
		t.Path, t.Title, t.Artist, t.DurationMs, t.SizeBytes, t.ModTime)
	return err
}

// SetDJTracksMissing flags every track whose path is not in present.
// Missing tracks that a playlist still uses are kept, flagged, so playlists
// don't silently lose entries when a drive is unplugged or a folder renamed
// (a later rescan clears the flag). Missing tracks no playlist uses are
// deleted so the library doesn't fill up with dead rows.
func SetDJTracksMissing(ctx context.Context, present []string) error {
	if _, err := pool.Exec(ctx, `UPDATE dj_tracks SET missing = NOT (path = ANY($1))`, present); err != nil {
		return err
	}
	_, err := pool.Exec(ctx, `
		DELETE FROM dj_tracks t WHERE t.missing
		AND NOT EXISTS (SELECT 1 FROM dj_playlist_tracks pt WHERE pt.track_id = t.id)`)
	return err
}

// ---- Playlists ----

func ListDJPlaylists(ctx context.Context) ([]DJPlaylist, error) {
	rows, err := pool.Query(ctx, `
		SELECT p.id, p.name, COALESCE(array_agg(pt.track_id ORDER BY pt.position) FILTER (WHERE pt.track_id IS NOT NULL), '{}')
		FROM dj_playlists p
		LEFT JOIN dj_playlist_tracks pt ON pt.playlist_id = p.id
		GROUP BY p.id ORDER BY LOWER(p.name)`)
	if err != nil {
		return nil, fmt.Errorf("list dj playlists: %w", err)
	}
	defer rows.Close()
	out := []DJPlaylist{}
	for rows.Next() {
		var p DJPlaylist
		if err := rows.Scan(&p.ID, &p.Name, &p.TrackIDs); err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	return out, rows.Err()
}

func GetDJPlaylist(ctx context.Context, id int64) (*DJPlaylist, error) {
	var p DJPlaylist
	err := pool.QueryRow(ctx, `
		SELECT p.id, p.name, COALESCE(array_agg(pt.track_id ORDER BY pt.position) FILTER (WHERE pt.track_id IS NOT NULL), '{}')
		FROM dj_playlists p
		LEFT JOIN dj_playlist_tracks pt ON pt.playlist_id = p.id
		WHERE p.id = $1 GROUP BY p.id`, id).Scan(&p.ID, &p.Name, &p.TrackIDs)
	if err == pgx.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &p, nil
}

func CreateDJPlaylist(ctx context.Context, name string) (*DJPlaylist, error) {
	p := DJPlaylist{Name: name, TrackIDs: []int64{}}
	err := pool.QueryRow(ctx, `INSERT INTO dj_playlists (name) VALUES ($1) RETURNING id`, name).Scan(&p.ID)
	if err != nil {
		return nil, err
	}
	return &p, nil
}

func RenameDJPlaylist(ctx context.Context, id int64, name string) (bool, error) {
	tag, err := pool.Exec(ctx, `UPDATE dj_playlists SET name = $2 WHERE id = $1`, id, name)
	return tag.RowsAffected() > 0, err
}

func DeleteDJPlaylist(ctx context.Context, id int64) (bool, error) {
	tag, err := pool.Exec(ctx, `DELETE FROM dj_playlists WHERE id = $1`, id)
	return tag.RowsAffected() > 0, err
}

// SetDJPlaylistTracks replaces a playlist's tracks with trackIDs, in order.
// Duplicates are allowed (a short sting can appear twice in a rotation).
func SetDJPlaylistTracks(ctx context.Context, id int64, trackIDs []int64) error {
	tx, err := pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)
	if _, err := tx.Exec(ctx, `DELETE FROM dj_playlist_tracks WHERE playlist_id = $1`, id); err != nil {
		return err
	}
	for i, tid := range trackIDs {
		if _, err := tx.Exec(ctx,
			`INSERT INTO dj_playlist_tracks (playlist_id, track_id, position) VALUES ($1, $2, $3)`,
			id, tid, i); err != nil {
			return err
		}
	}
	return tx.Commit(ctx)
}

// ---- Scenes ----

const sceneCols = `id, name, playlist_id, color, volume, shuffle, loop, fade_seconds, position`

func scanScene(row pgx.Row) (DJScene, error) {
	var s DJScene
	var vol float32
	var fade *float32
	err := row.Scan(&s.ID, &s.Name, &s.PlaylistID, &s.Color, &vol, &s.Shuffle, &s.Loop, &fade, &s.Position)
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
		INSERT INTO dj_scenes (name, playlist_id, color, volume, shuffle, loop, fade_seconds, position)
		VALUES ($1, $2, $3, $4, $5, $6, $7, COALESCE((SELECT MAX(position) + 1 FROM dj_scenes), 0))
		RETURNING id, position`,
		s.Name, s.PlaylistID, s.Color, s.Volume, s.Shuffle, s.Loop, s.FadeSeconds).Scan(&s.ID, &s.Position)
	if err != nil {
		return nil, err
	}
	return &s, nil
}

func UpdateDJScene(ctx context.Context, s DJScene) (bool, error) {
	tag, err := pool.Exec(ctx, `
		UPDATE dj_scenes SET name = $2, playlist_id = $3, color = $4, volume = $5,
			shuffle = $6, loop = $7, fade_seconds = $8
		WHERE id = $1`,
		s.ID, s.Name, s.PlaylistID, s.Color, s.Volume, s.Shuffle, s.Loop, s.FadeSeconds)
	return tag.RowsAffected() > 0, err
}

func DeleteDJScene(ctx context.Context, id int64) (bool, error) {
	tag, err := pool.Exec(ctx, `DELETE FROM dj_scenes WHERE id = $1`, id)
	return tag.RowsAffected() > 0, err
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
