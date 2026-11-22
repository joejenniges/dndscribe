package db

import (
	"context"
	"encoding/json"
	"errors"

	"github.com/jackc/pgx/v5"
)

// App settings changed from the web UI. config.yaml supplies the defaults;
// a value saved here wins over it, so a setting survives a restart without
// anyone editing the server's config file.

// Setting keys.
const (
	SettingDJFadeSeconds = "dj.fade_seconds"
)

func migrateSettings(ctx context.Context) error {
	_, err := pool.Exec(ctx, `CREATE TABLE IF NOT EXISTS app_settings (
		key TEXT PRIMARY KEY,
		value JSONB NOT NULL,
		updated_at TIMESTAMPTZ DEFAULT NOW()
	)`)
	return err
}

// GetSetting decodes a saved setting into out. found is false when the
// setting was never saved, leaving out untouched.
func GetSetting(ctx context.Context, key string, out any) (found bool, err error) {
	var raw []byte
	err = pool.QueryRow(ctx, `SELECT value FROM app_settings WHERE key = $1`, key).Scan(&raw)
	if errors.Is(err, pgx.ErrNoRows) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	return true, json.Unmarshal(raw, out)
}

func SetSetting(ctx context.Context, key string, value any) error {
	raw, err := json.Marshal(value)
	if err != nil {
		return err
	}
	_, err = pool.Exec(ctx, `
		INSERT INTO app_settings (key, value) VALUES ($1, $2)
		ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()`, key, raw)
	return err
}
