package db

import (
	"context"
	"errors"
	"time"

	"github.com/jackc/pgx/v5"
)

// Waveform cache. A track's waveform is computed once (a full ffmpeg decode)
// and kept here. It's stamped with the file size and mtime it was computed
// from, so a file replaced on disk under the same path gets a new one; the
// row goes with the track (ON DELETE CASCADE).

type DJWaveform struct {
	BinsPerSecond int
	Data          []byte // per bin: peak, rms (uint8 each, 0..255 of full scale)
}

func migrateWaveforms(ctx context.Context) error {
	_, err := pool.Exec(ctx, `CREATE TABLE IF NOT EXISTS dj_waveforms (
		track_id BIGINT PRIMARY KEY REFERENCES dj_tracks(id) ON DELETE CASCADE,
		size_bytes BIGINT NOT NULL,
		mtime TIMESTAMPTZ NOT NULL,
		bins_per_second INT NOT NULL,
		data BYTEA NOT NULL,
		created_at TIMESTAMPTZ DEFAULT NOW()
	)`)
	return err
}

// GetDJWaveform returns the cached waveform for a track if it was computed
// from the file as it is now (same size and mtime), else nil.
func GetDJWaveform(ctx context.Context, trackID, size int64, mtime time.Time) (*DJWaveform, error) {
	var w DJWaveform
	var gotSize int64
	var gotMtime time.Time
	err := pool.QueryRow(ctx,
		`SELECT size_bytes, mtime, bins_per_second, data FROM dj_waveforms WHERE track_id = $1`, trackID).
		Scan(&gotSize, &gotMtime, &w.BinsPerSecond, &w.Data)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	// Postgres keeps microseconds; compare at that precision.
	if gotSize != size || !gotMtime.Truncate(time.Microsecond).Equal(mtime.Truncate(time.Microsecond)) {
		return nil, nil
	}
	return &w, nil
}

func SaveDJWaveform(ctx context.Context, trackID, size int64, mtime time.Time, w DJWaveform) error {
	_, err := pool.Exec(ctx, `
		INSERT INTO dj_waveforms (track_id, size_bytes, mtime, bins_per_second, data)
		VALUES ($1, $2, $3, $4, $5)
		ON CONFLICT (track_id) DO UPDATE SET size_bytes = EXCLUDED.size_bytes, mtime = EXCLUDED.mtime,
			bins_per_second = EXCLUDED.bins_per_second, data = EXCLUDED.data, created_at = NOW()`,
		trackID, size, mtime, w.BinsPerSecond, w.Data)
	return err
}

// DJTracksWithoutWaveform lists playable tracks with no up-to-date waveform,
// for filling the cache in the background.
func DJTracksWithoutWaveform(ctx context.Context) ([]DJTrack, error) {
	all, err := ListDJTracks(ctx)
	if err != nil {
		return nil, err
	}
	rows, err := pool.Query(ctx, `SELECT track_id, size_bytes, mtime FROM dj_waveforms`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	type stamp struct {
		size  int64
		mtime time.Time
	}
	have := map[int64]stamp{}
	for rows.Next() {
		var id int64
		var s stamp
		if err := rows.Scan(&id, &s.size, &s.mtime); err != nil {
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
		if ok && s.size == t.SizeBytes && s.mtime.Truncate(time.Microsecond).Equal(t.ModTime.Truncate(time.Microsecond)) {
			continue
		}
		out = append(out, t)
	}
	return out, nil
}
