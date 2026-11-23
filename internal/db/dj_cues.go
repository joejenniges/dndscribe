package db

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/jackc/pgx/v5"
)

// Hotcues, goto triggers and beat analysis for DJ tracks (v4).
//
// All three hang off dj_tracks.id with ON DELETE CASCADE. A track's id
// survives renames and folder moves made in the app (RenameDJFolder updates
// the path in place), so cues follow a track wherever it's moved. Beat
// analysis is also keyed by size+mtime like the waveform cache, so a
// changed file is re-analysed but a renamed one isn't.

// HotcueSlots is the number of hotcue slots per track (1..8).
const HotcueSlots = 8

type DJHotcue struct {
	Slot       int    `json:"slot"` // 1..8
	PositionMs int64  `json:"positionMs"`
	Label      string `json:"label"`
	Color      string `json:"color"`
}

// DJGoto is a goto trigger: when playback crosses PositionMs and Enabled is
// set, the engine jumps to hotcue TargetSlot of the same track.
type DJGoto struct {
	ID         int64  `json:"id"`
	PositionMs int64  `json:"positionMs"`
	TargetSlot int    `json:"targetSlot"`
	Enabled    bool   `json:"enabled"`
	Label      string `json:"label"`
}

// DJTrackCues is everything cue-related for one track.
type DJTrackCues struct {
	TrackID int64      `json:"trackId"`
	Hotcues []DJHotcue `json:"hotcues"` // by slot
	Gotos   []DJGoto   `json:"gotos"`   // by position, then id
}

func migrateCues(ctx context.Context) error {
	stmts := []string{
		`CREATE TABLE IF NOT EXISTS dj_hotcues (
			track_id BIGINT NOT NULL REFERENCES dj_tracks(id) ON DELETE CASCADE,
			slot INT NOT NULL CHECK (slot BETWEEN 1 AND 8),
			position_ms BIGINT NOT NULL CHECK (position_ms >= 0),
			label TEXT NOT NULL DEFAULT '',
			color TEXT NOT NULL DEFAULT '',
			updated_at TIMESTAMPTZ DEFAULT NOW(),
			PRIMARY KEY (track_id, slot)
		)`,
		`CREATE TABLE IF NOT EXISTS dj_gotos (
			id BIGSERIAL PRIMARY KEY,
			track_id BIGINT NOT NULL REFERENCES dj_tracks(id) ON DELETE CASCADE,
			position_ms BIGINT NOT NULL CHECK (position_ms >= 0),
			target_slot INT NOT NULL CHECK (target_slot BETWEEN 1 AND 8),
			enabled BOOLEAN NOT NULL DEFAULT TRUE,
			label TEXT NOT NULL DEFAULT '',
			updated_at TIMESTAMPTZ DEFAULT NOW()
		)`,
		`CREATE INDEX IF NOT EXISTS dj_gotos_track ON dj_gotos (track_id, position_ms, id)`,
		`CREATE TABLE IF NOT EXISTS dj_track_analysis (
			track_id BIGINT PRIMARY KEY REFERENCES dj_tracks(id) ON DELETE CASCADE,
			size_bytes BIGINT NOT NULL,
			mtime TIMESTAMPTZ NOT NULL,
			version INT NOT NULL,
			status TEXT NOT NULL,
			source TEXT NOT NULL DEFAULT 'none',
			bpm REAL,
			confidence REAL,
			beats BYTEA,
			error TEXT NOT NULL DEFAULT '',
			analyzed_at TIMESTAMPTZ DEFAULT NOW()
		)`,
	}
	for _, s := range stmts {
		if _, err := pool.Exec(ctx, s); err != nil {
			return err
		}
	}
	return nil
}

// GetDJTrackCues returns a track's hotcues and gotos (empty lists when it
// has none).
func GetDJTrackCues(ctx context.Context, trackID int64) (*DJTrackCues, error) {
	c := &DJTrackCues{TrackID: trackID, Hotcues: []DJHotcue{}, Gotos: []DJGoto{}}
	rows, err := pool.Query(ctx, `SELECT slot, position_ms, label, color FROM dj_hotcues WHERE track_id = $1 ORDER BY slot`, trackID)
	if err != nil {
		return nil, fmt.Errorf("list hotcues: %w", err)
	}
	for rows.Next() {
		var h DJHotcue
		if err := rows.Scan(&h.Slot, &h.PositionMs, &h.Label, &h.Color); err != nil {
			rows.Close()
			return nil, err
		}
		c.Hotcues = append(c.Hotcues, h)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return nil, err
	}
	rows, err = pool.Query(ctx, `SELECT id, position_ms, target_slot, enabled, label FROM dj_gotos WHERE track_id = $1 ORDER BY position_ms, id`, trackID)
	if err != nil {
		return nil, fmt.Errorf("list gotos: %w", err)
	}
	defer rows.Close()
	for rows.Next() {
		var g DJGoto
		if err := rows.Scan(&g.ID, &g.PositionMs, &g.TargetSlot, &g.Enabled, &g.Label); err != nil {
			return nil, err
		}
		c.Gotos = append(c.Gotos, g)
	}
	return c, rows.Err()
}

// SetDJHotcue creates or replaces one hotcue slot.
func SetDJHotcue(ctx context.Context, trackID int64, h DJHotcue) error {
	_, err := pool.Exec(ctx, `
		INSERT INTO dj_hotcues (track_id, slot, position_ms, label, color) VALUES ($1, $2, $3, $4, $5)
		ON CONFLICT (track_id, slot) DO UPDATE SET position_ms = EXCLUDED.position_ms,
			label = EXCLUDED.label, color = EXCLUDED.color, updated_at = NOW()`,
		trackID, h.Slot, h.PositionMs, h.Label, h.Color)
	return err
}

// DeleteDJHotcue removes a hotcue slot. Gotos that targeted it are
// disabled (not deleted, not retargeted) in the same transaction: they show
// "target missing" until the slot is set again or they're repointed.
// Returns how many gotos were disabled.
func DeleteDJHotcue(ctx context.Context, trackID int64, slot int) (int, error) {
	tx, err := pool.Begin(ctx)
	if err != nil {
		return 0, err
	}
	defer tx.Rollback(ctx)
	if _, err := tx.Exec(ctx, `DELETE FROM dj_hotcues WHERE track_id = $1 AND slot = $2`, trackID, slot); err != nil {
		return 0, err
	}
	tag, err := tx.Exec(ctx, `UPDATE dj_gotos SET enabled = FALSE, updated_at = NOW()
		WHERE track_id = $1 AND target_slot = $2 AND enabled`, trackID, slot)
	if err != nil {
		return 0, err
	}
	return int(tag.RowsAffected()), tx.Commit(ctx)
}

// CreateDJGoto adds a goto trigger and returns it with its id.
func CreateDJGoto(ctx context.Context, trackID int64, g DJGoto) (*DJGoto, error) {
	err := pool.QueryRow(ctx, `
		INSERT INTO dj_gotos (track_id, position_ms, target_slot, enabled, label) VALUES ($1, $2, $3, $4, $5)
		RETURNING id`, trackID, g.PositionMs, g.TargetSlot, g.Enabled, g.Label).Scan(&g.ID)
	if err != nil {
		return nil, err
	}
	return &g, nil
}

// UpdateDJGoto saves a goto. Returns false if it doesn't exist on that track.
func UpdateDJGoto(ctx context.Context, trackID int64, g DJGoto) (bool, error) {
	tag, err := pool.Exec(ctx, `
		UPDATE dj_gotos SET position_ms = $3, target_slot = $4, enabled = $5, label = $6, updated_at = NOW()
		WHERE id = $1 AND track_id = $2`, g.ID, trackID, g.PositionMs, g.TargetSlot, g.Enabled, g.Label)
	return tag.RowsAffected() > 0, err
}

// SetDJGotoEnabled arms or disarms a goto. Returns false if not found.
func SetDJGotoEnabled(ctx context.Context, trackID, id int64, on bool) (bool, error) {
	tag, err := pool.Exec(ctx, `UPDATE dj_gotos SET enabled = $3, updated_at = NOW() WHERE id = $1 AND track_id = $2`, id, trackID, on)
	return tag.RowsAffected() > 0, err
}

func DeleteDJGoto(ctx context.Context, trackID, id int64) (bool, error) {
	tag, err := pool.Exec(ctx, `DELETE FROM dj_gotos WHERE id = $1 AND track_id = $2`, id, trackID)
	return tag.RowsAffected() > 0, err
}

// ---- Beat analysis ----

// DJAnalysis is a track's beat analysis. Beats are absolute positions in
// milliseconds (not BPM + offset: tempo drifts in live-played and
// orchestral music, and the grid has to follow it).
type DJAnalysis struct {
	TrackID    int64     `json:"trackId"`
	Version    int       `json:"version"`
	Status     string    `json:"status"` // pending, running, ready, unavailable, failed
	Source     string    `json:"source"` // auto, manual, none
	BPM        *float64  `json:"bpm"`
	Confidence *float64  `json:"confidence"`
	Beats      []int64   `json:"beats"`
	Error      string    `json:"error,omitempty"`
	AnalyzedAt time.Time `json:"analyzedAt"`
	// Current: computed from the file as it is now (same size and mtime).
	// A stale row is still returned so the UI can show the old grid while
	// the new analysis runs.
	Current bool `json:"current"`
}

// GetDJAnalysis returns the stored analysis for a track, or nil. Current
// says whether it matches the file's present size and mtime.
func GetDJAnalysis(ctx context.Context, trackID int64, size int64, mtime time.Time) (*DJAnalysis, error) {
	a := DJAnalysis{TrackID: trackID}
	var gotSize int64
	var gotMtime time.Time
	var bpm, conf *float32
	var beats []byte
	err := pool.QueryRow(ctx, `SELECT size_bytes, mtime, version, status, source, bpm, confidence, beats, error, analyzed_at
		FROM dj_track_analysis WHERE track_id = $1`, trackID).
		Scan(&gotSize, &gotMtime, &a.Version, &a.Status, &a.Source, &bpm, &conf, &beats, &a.Error, &a.AnalyzedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	// REAL storage: round off float32 noise (138.1 reads back as 138.10000610).
	if bpm != nil {
		v := math.Round(float64(*bpm)*100) / 100
		a.BPM = &v
	}
	if conf != nil {
		v := math.Round(float64(*conf)*1000) / 1000
		a.Confidence = &v
	}
	a.Beats = decodeBeats(beats)
	a.Current = gotSize == size && gotMtime.Truncate(time.Microsecond).Equal(mtime.Truncate(time.Microsecond))
	return &a, nil
}

// SaveDJAnalysis stores (replaces) a track's analysis, stamped with the
// file's size and mtime.
func SaveDJAnalysis(ctx context.Context, size int64, mtime time.Time, a DJAnalysis) error {
	var bpm, conf *float32
	if a.BPM != nil {
		v := float32(*a.BPM)
		bpm = &v
	}
	if a.Confidence != nil {
		v := float32(*a.Confidence)
		conf = &v
	}
	_, err := pool.Exec(ctx, `
		INSERT INTO dj_track_analysis (track_id, size_bytes, mtime, version, status, source, bpm, confidence, beats, error, analyzed_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NOW())
		ON CONFLICT (track_id) DO UPDATE SET size_bytes = EXCLUDED.size_bytes, mtime = EXCLUDED.mtime,
			version = EXCLUDED.version, status = EXCLUDED.status, source = EXCLUDED.source, bpm = EXCLUDED.bpm,
			confidence = EXCLUDED.confidence, beats = EXCLUDED.beats, error = EXCLUDED.error, analyzed_at = NOW()`,
		a.TrackID, size, mtime, a.Version, a.Status, a.Source, bpm, conf, encodeBeats(a.Beats), a.Error)
	return err
}

// SetDJAnalysisStatus records a status change (queued/running) without
// touching the stored grid, so the previous grid stays visible meanwhile.
func SetDJAnalysisStatus(ctx context.Context, trackID int64, size int64, mtime time.Time, version int, status string) error {
	_, err := pool.Exec(ctx, `
		INSERT INTO dj_track_analysis (track_id, size_bytes, mtime, version, status) VALUES ($1, $2, $3, $4, $5)
		ON CONFLICT (track_id) DO UPDATE SET status = EXCLUDED.status`,
		trackID, size, mtime, version, status)
	return err
}

// DJTracksNeedingAnalysis lists playable tracks with no analysis, a stale
// one (file changed or older algorithm version), or one left pending or
// running by a restart. Manual grids are kept even when the version moves
// on; only a file change invalidates them.
func DJTracksNeedingAnalysis(ctx context.Context, version int) ([]DJTrack, error) {
	all, err := ListDJTracks(ctx)
	if err != nil {
		return nil, err
	}
	rows, err := pool.Query(ctx, `SELECT track_id, size_bytes, mtime, version, status, source FROM dj_track_analysis`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	type stamp struct {
		size           int64
		mtime          time.Time
		version        int
		status, source string
	}
	have := map[int64]stamp{}
	for rows.Next() {
		var id int64
		var s stamp
		if err := rows.Scan(&id, &s.size, &s.mtime, &s.version, &s.status, &s.source); err != nil {
			return nil, err
		}
		have[id] = s
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	var out []DJTrack
	for _, t := range all {
		if t.Missing {
			continue
		}
		s, ok := have[t.ID]
		sameFile := ok && s.size == t.SizeBytes && s.mtime.Truncate(time.Microsecond).Equal(t.ModTime.Truncate(time.Microsecond))
		switch {
		case !sameFile:
			out = append(out, t)
		case s.status == "pending" || s.status == "running":
			out = append(out, t)
		case s.source != "manual" && s.version < version:
			out = append(out, t)
		}
	}
	return out, nil
}

// Beats are stored as little-endian uint32 deltas in ms (a 10 minute track
// at 180 BPM is 1800 beats = 7KB).
func encodeBeats(b []int64) []byte {
	if len(b) == 0 {
		return nil
	}
	out := make([]byte, 4*len(b))
	prev := int64(0)
	for i, v := range b {
		binary.LittleEndian.PutUint32(out[i*4:], uint32(max(0, v-prev)))
		prev = v
	}
	return out
}

func decodeBeats(raw []byte) []int64 {
	out := make([]int64, 0, len(raw)/4)
	acc := int64(0)
	for i := 0; i+4 <= len(raw); i += 4 {
		acc += int64(binary.LittleEndian.Uint32(raw[i:]))
		out = append(out, acc)
	}
	return out
}
