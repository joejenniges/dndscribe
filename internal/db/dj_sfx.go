package db

import (
	"context"
	"fmt"
	"math"

	"github.com/jackc/pgx/v5"
)

// Soundboard storage: tabs that hold pads. Each pad owns one uploaded sound
// file (stored under a server-generated key, never the client's file name)
// and optionally an image. A pad lives in exactly one tab; moving it between
// tabs changes tab_id. Like the rest of the DJ data this is global, not
// per-campaign.

type SfxTab struct {
	ID       int64  `json:"id"`
	Title    string `json:"title"`
	Position int    `json:"position"`
}

type SfxPad struct {
	ID           int64   `json:"id"`
	TabID        int64   `json:"tabId"`
	Label        string  `json:"label"`
	FileKey      string  `json:"-"`
	OriginalName string  `json:"originalName"`
	DurationMs   int64   `json:"durationMs"`
	Color        string  `json:"color"`
	ImageKey     *string `json:"image"` // nil = no image; served at /api/dj/sfx/images/{key}
	Gain         float64 `json:"gain"`  // 0..1
	Position     int     `json:"position"`
}

func migrateSfx(ctx context.Context) error {
	stmts := []string{
		`CREATE TABLE IF NOT EXISTS dj_sfx_tabs (
			id BIGSERIAL PRIMARY KEY,
			title TEXT NOT NULL,
			position INT NOT NULL DEFAULT 0,
			created_at TIMESTAMPTZ DEFAULT NOW()
		)`,
		`CREATE TABLE IF NOT EXISTS dj_sfx_pads (
			id BIGSERIAL PRIMARY KEY,
			tab_id BIGINT NOT NULL REFERENCES dj_sfx_tabs(id) ON DELETE CASCADE,
			label TEXT NOT NULL,
			file_key TEXT NOT NULL UNIQUE,
			original_name TEXT NOT NULL DEFAULT '',
			duration_ms BIGINT NOT NULL DEFAULT 0,
			color TEXT NOT NULL DEFAULT '',
			image_key TEXT,
			gain REAL NOT NULL DEFAULT 1,
			position INT NOT NULL DEFAULT 0,
			created_at TIMESTAMPTZ DEFAULT NOW()
		)`,
		`CREATE INDEX IF NOT EXISTS dj_sfx_pads_tab ON dj_sfx_pads (tab_id, position)`,
	}
	for _, s := range stmts {
		if _, err := pool.Exec(ctx, s); err != nil {
			return err
		}
	}
	return nil
}

// ---- Tabs ----

func ListSfxTabs(ctx context.Context) ([]SfxTab, error) {
	rows, err := pool.Query(ctx, `SELECT id, title, position FROM dj_sfx_tabs ORDER BY position, id`)
	if err != nil {
		return nil, fmt.Errorf("list sfx tabs: %w", err)
	}
	defer rows.Close()
	out := []SfxTab{}
	for rows.Next() {
		var t SfxTab
		if err := rows.Scan(&t.ID, &t.Title, &t.Position); err != nil {
			return nil, err
		}
		out = append(out, t)
	}
	return out, rows.Err()
}

func CreateSfxTab(ctx context.Context, title string) (*SfxTab, error) {
	t := SfxTab{Title: title}
	err := pool.QueryRow(ctx, `
		INSERT INTO dj_sfx_tabs (title, position)
		VALUES ($1, COALESCE((SELECT MAX(position) + 1 FROM dj_sfx_tabs), 0))
		RETURNING id, position`, title).Scan(&t.ID, &t.Position)
	if err != nil {
		return nil, err
	}
	return &t, nil
}

func RenameSfxTab(ctx context.Context, id int64, title string) (bool, error) {
	tag, err := pool.Exec(ctx, `UPDATE dj_sfx_tabs SET title = $2 WHERE id = $1`, id, title)
	return tag.RowsAffected() > 0, err
}

// DeleteSfxTab deletes a tab and its pads, returning the deleted pads so the
// caller can remove their files.
func DeleteSfxTab(ctx context.Context, id int64) ([]SfxPad, error) {
	pads, err := listSfxPads(ctx, `WHERE tab_id = $1`, id)
	if err != nil {
		return nil, err
	}
	if _, err := pool.Exec(ctx, `DELETE FROM dj_sfx_tabs WHERE id = $1`, id); err != nil {
		return nil, err
	}
	return pads, nil
}

func ReorderSfxTabs(ctx context.Context, ids []int64) error {
	tx, err := pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)
	for i, id := range ids {
		if _, err := tx.Exec(ctx, `UPDATE dj_sfx_tabs SET position = $2 WHERE id = $1`, id, i); err != nil {
			return err
		}
	}
	return tx.Commit(ctx)
}

// ---- Pads ----

const padCols = `id, tab_id, label, file_key, original_name, duration_ms, color, image_key, gain, position`

func scanPad(row pgx.Row) (SfxPad, error) {
	var p SfxPad
	var gain float32
	err := row.Scan(&p.ID, &p.TabID, &p.Label, &p.FileKey, &p.OriginalName, &p.DurationMs, &p.Color, &p.ImageKey, &gain, &p.Position)
	p.Gain = math.Round(float64(gain)*1000) / 1000 // REAL storage: 0.7 reads back as 0.69999
	return p, err
}

func listSfxPads(ctx context.Context, where string, args ...any) ([]SfxPad, error) {
	rows, err := pool.Query(ctx, `SELECT `+padCols+` FROM dj_sfx_pads `+where+` ORDER BY tab_id, position, id`, args...)
	if err != nil {
		return nil, fmt.Errorf("list sfx pads: %w", err)
	}
	defer rows.Close()
	out := []SfxPad{}
	for rows.Next() {
		p, err := scanPad(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	return out, rows.Err()
}

func ListSfxPads(ctx context.Context) ([]SfxPad, error) { return listSfxPads(ctx, "") }

func GetSfxPad(ctx context.Context, id int64) (*SfxPad, error) {
	p, err := scanPad(pool.QueryRow(ctx, `SELECT `+padCols+` FROM dj_sfx_pads WHERE id = $1`, id))
	if err == pgx.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &p, nil
}

// CreateSfxPad adds a pad at the end of its tab.
func CreateSfxPad(ctx context.Context, p SfxPad) (*SfxPad, error) {
	err := pool.QueryRow(ctx, `
		INSERT INTO dj_sfx_pads (tab_id, label, file_key, original_name, duration_ms, color, gain, position)
		VALUES ($1, $2, $3, $4, $5, $6, $7,
			COALESCE((SELECT MAX(position) + 1 FROM dj_sfx_pads WHERE tab_id = $1), 0))
		RETURNING id, position`,
		p.TabID, p.Label, p.FileKey, p.OriginalName, p.DurationMs, p.Color, p.Gain).Scan(&p.ID, &p.Position)
	if err != nil {
		return nil, err
	}
	return &p, nil
}

// UpdateSfxPad saves label, color and gain. Moving a pad to another tab puts
// it at the end of that tab.
func UpdateSfxPad(ctx context.Context, p SfxPad) (bool, error) {
	tag, err := pool.Exec(ctx, `
		UPDATE dj_sfx_pads SET label = $2, color = $3, gain = $4,
			position = CASE WHEN tab_id = $5 THEN position
				ELSE COALESCE((SELECT MAX(position) + 1 FROM dj_sfx_pads WHERE tab_id = $5), 0) END,
			tab_id = $5
		WHERE id = $1`,
		p.ID, p.Label, p.Color, p.Gain, p.TabID)
	return tag.RowsAffected() > 0, err
}

// SetSfxPadSound points a pad at a new sound file and returns the previous
// file key so the caller can delete that file.
func SetSfxPadSound(ctx context.Context, id int64, key, originalName string, durationMs int64) (old string, err error) {
	err = pool.QueryRow(ctx, `
		UPDATE dj_sfx_pads p SET file_key = $2, original_name = $3, duration_ms = $4
		FROM (SELECT file_key FROM dj_sfx_pads WHERE id = $1) prev
		WHERE p.id = $1 RETURNING prev.file_key`, id, key, originalName, durationMs).Scan(&old)
	if err == pgx.ErrNoRows {
		return "", fmt.Errorf("pad not found")
	}
	return old, err
}

// SetSfxPadImage sets (or clears, with nil) a pad's image and returns the
// previous image key so the caller can delete that file.
func SetSfxPadImage(ctx context.Context, id int64, key *string) (old *string, err error) {
	err = pool.QueryRow(ctx, `
		UPDATE dj_sfx_pads p SET image_key = $2
		FROM (SELECT image_key FROM dj_sfx_pads WHERE id = $1) prev
		WHERE p.id = $1 RETURNING prev.image_key`, id, key).Scan(&old)
	if err == pgx.ErrNoRows {
		return nil, fmt.Errorf("pad not found")
	}
	return old, err
}

func DeleteSfxPad(ctx context.Context, id int64) (*SfxPad, error) {
	p, err := scanPad(pool.QueryRow(ctx, `DELETE FROM dj_sfx_pads WHERE id = $1 RETURNING `+padCols, id))
	if err == pgx.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &p, nil
}

// ArrangeSfxTab makes ids the contents of tab, in order. Pads listed from
// another tab move into this one, which is how drag between tabs persists.
func ArrangeSfxTab(ctx context.Context, tabID int64, ids []int64) error {
	tx, err := pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)
	for i, id := range ids {
		if _, err := tx.Exec(ctx, `UPDATE dj_sfx_pads SET tab_id = $2, position = $3 WHERE id = $1`, id, tabID, i); err != nil {
			return err
		}
	}
	return tx.Commit(ctx)
}
