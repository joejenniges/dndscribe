package db

import (
	"context"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

var pool *pgxpool.Pool

// Init creates the connection pool and runs migrations.
func Init(databaseURL string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	cfg, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		return fmt.Errorf("parse database URL: %w", err)
	}

	p, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		return fmt.Errorf("create connection pool: %w", err)
	}

	if err := p.Ping(ctx); err != nil {
		p.Close()
		return fmt.Errorf("ping database: %w", err)
	}

	pool = p

	if err := runMigrations(ctx); err != nil {
		pool.Close()
		pool = nil
		return fmt.Errorf("run migrations: %w", err)
	}

	return nil
}

// Close shuts down the connection pool.
func Close() {
	if pool != nil {
		pool.Close()
	}
}

// Pool returns the underlying pgxpool for direct use if needed.
func Pool() *pgxpool.Pool {
	return pool
}

func runMigrations(ctx context.Context) error {
	// Core tables
	tables := []string{
		`CREATE TABLE IF NOT EXISTS campaigns (
			id BIGSERIAL PRIMARY KEY,
			name TEXT NOT NULL,
			description TEXT,
			created_at TIMESTAMPTZ DEFAULT NOW(),
			updated_at TIMESTAMPTZ DEFAULT NOW()
		)`,

		`CREATE TABLE IF NOT EXISTS sessions (
			id BIGSERIAL PRIMARY KEY,
			campaign_id BIGINT REFERENCES campaigns(id),
			started_at TIMESTAMPTZ DEFAULT NOW(),
			ended_at TIMESTAMPTZ,
			channel_name TEXT,
			channel_id TEXT
		)`,

		`CREATE TABLE IF NOT EXISTS transcriptions (
			id BIGSERIAL PRIMARY KEY,
			session_id BIGINT REFERENCES sessions(id),
			created_at TIMESTAMPTZ DEFAULT NOW(),
			discord_user_id TEXT,
			discord_username TEXT,
			nickname TEXT,
			spoken_text TEXT,
			audio_filenames TEXT[],
			rms REAL,
			duration_ms INTEGER,
			confidence REAL,
			edited_at TIMESTAMPTZ
		)`,

		`CREATE TABLE IF NOT EXISTS hotwords (
			id BIGSERIAL PRIMARY KEY,
			campaign_id BIGINT REFERENCES campaigns(id),
			word TEXT,
			created_at TIMESTAMPTZ DEFAULT NOW()
		)`,

		// ignored_users and character_names originally had single-column PKs.
		// These DO $$ blocks migrate them to composite PKs if needed.
		`CREATE TABLE IF NOT EXISTS ignored_users (
			campaign_id BIGINT,
			discord_user_id TEXT,
			discord_username TEXT,
			ignored_at TIMESTAMPTZ DEFAULT NOW()
		)`,

		`CREATE TABLE IF NOT EXISTS character_names (
			campaign_id BIGINT,
			discord_user_id TEXT,
			name TEXT,
			updated_at TIMESTAMPTZ DEFAULT NOW()
		)`,

		`CREATE TABLE IF NOT EXISTS nickname_categories (
			id BIGSERIAL PRIMARY KEY,
			campaign_id BIGINT REFERENCES campaigns(id),
			discord_user_id TEXT,
			name TEXT,
			position INT,
			created_at TIMESTAMPTZ DEFAULT NOW()
		)`,

		`CREATE TABLE IF NOT EXISTS nickname_presets (
			id BIGSERIAL PRIMARY KEY,
			campaign_id BIGINT REFERENCES campaigns(id),
			discord_user_id TEXT,
			label TEXT,
			position INT,
			category_id BIGINT REFERENCES nickname_categories(id) ON DELETE SET NULL,
			created_at TIMESTAMPTZ DEFAULT NOW()
		)`,
	}

	for _, ddl := range tables {
		if _, err := pool.Exec(ctx, ddl); err != nil {
			return fmt.Errorf("create table: %w", err)
		}
	}

	// Unique index on hotwords
	_, err := pool.Exec(ctx, `
		CREATE UNIQUE INDEX IF NOT EXISTS hotwords_campaign_word_unique
		ON hotwords (campaign_id, LOWER(word))
	`)
	if err != nil {
		return fmt.Errorf("create hotwords index: %w", err)
	}

	// PK migration for ignored_users: add composite PK if not present
	_, err = pool.Exec(ctx, `
		DO $$ BEGIN
			IF NOT EXISTS (
				SELECT 1 FROM information_schema.table_constraints
				WHERE table_name = 'ignored_users' AND constraint_type = 'PRIMARY KEY'
			) THEN
				ALTER TABLE ignored_users ADD PRIMARY KEY (campaign_id, discord_user_id);
			END IF;
		END $$
	`)
	if err != nil {
		return fmt.Errorf("ignored_users PK migration: %w", err)
	}

	// PK migration for character_names: add composite PK if not present
	_, err = pool.Exec(ctx, `
		DO $$ BEGIN
			IF NOT EXISTS (
				SELECT 1 FROM information_schema.table_constraints
				WHERE table_name = 'character_names' AND constraint_type = 'PRIMARY KEY'
			) THEN
				ALTER TABLE character_names ADD PRIMARY KEY (campaign_id, discord_user_id);
			END IF;
		END $$
	`)
	if err != nil {
		return fmt.Errorf("character_names PK migration: %w", err)
	}

	// ALTER TABLE migrations for columns added after initial schema
	alterMigrations := []string{
		`ALTER TABLE sessions ADD COLUMN IF NOT EXISTS channel_id TEXT`,
		`ALTER TABLE transcriptions ADD COLUMN IF NOT EXISTS nickname TEXT`,
		`ALTER TABLE transcriptions ADD COLUMN IF NOT EXISTS audio_filenames TEXT[]`,
		`ALTER TABLE transcriptions ADD COLUMN IF NOT EXISTS rms REAL`,
		`ALTER TABLE transcriptions ADD COLUMN IF NOT EXISTS duration_ms INTEGER`,
		`ALTER TABLE transcriptions ADD COLUMN IF NOT EXISTS confidence REAL`,
		`ALTER TABLE transcriptions ADD COLUMN IF NOT EXISTS edited_at TIMESTAMPTZ`,
		`ALTER TABLE ignored_users ADD COLUMN IF NOT EXISTS campaign_id BIGINT`,
		`ALTER TABLE character_names ADD COLUMN IF NOT EXISTS campaign_id BIGINT`,
	}

	for _, alter := range alterMigrations {
		if _, err := pool.Exec(ctx, alter); err != nil {
			return fmt.Errorf("alter table: %w", err)
		}
	}

	// Seed default campaign
	_, err = pool.Exec(ctx, `
		INSERT INTO campaigns (id, name, description)
		VALUES (1, 'Default Campaign', 'Default campaign')
		ON CONFLICT (id) DO NOTHING
	`)
	if err != nil {
		return fmt.Errorf("seed default campaign: %w", err)
	}

	return nil
}
