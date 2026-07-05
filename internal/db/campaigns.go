package db

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
)

type Campaign struct {
	ID          int64     `json:"id"`
	Name        string    `json:"name"`
	Description *string   `json:"description"`
	CreatedAt   time.Time `json:"createdAt"`
	UpdatedAt   time.Time `json:"updatedAt"`
}

func ListCampaigns(ctx context.Context) ([]Campaign, error) {
	rows, err := pool.Query(ctx,
		`SELECT id, name, description, created_at, updated_at
		 FROM campaigns ORDER BY id`)
	if err != nil {
		return nil, fmt.Errorf("list campaigns: %w", err)
	}
	defer rows.Close()

	var campaigns []Campaign
	for rows.Next() {
		var c Campaign
		if err := rows.Scan(&c.ID, &c.Name, &c.Description, &c.CreatedAt, &c.UpdatedAt); err != nil {
			return nil, fmt.Errorf("scan campaign: %w", err)
		}
		campaigns = append(campaigns, c)
	}
	return campaigns, rows.Err()
}

func GetCampaign(ctx context.Context, id int64) (*Campaign, error) {
	var c Campaign
	err := pool.QueryRow(ctx,
		`SELECT id, name, description, created_at, updated_at
		 FROM campaigns WHERE id = $1`, id).
		Scan(&c.ID, &c.Name, &c.Description, &c.CreatedAt, &c.UpdatedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("get campaign: %w", err)
	}
	return &c, nil
}

func CreateCampaign(ctx context.Context, name string, description *string) (*Campaign, error) {
	var c Campaign
	err := pool.QueryRow(ctx,
		`INSERT INTO campaigns (name, description)
		 VALUES ($1, $2)
		 RETURNING id, name, description, created_at, updated_at`,
		name, description).
		Scan(&c.ID, &c.Name, &c.Description, &c.CreatedAt, &c.UpdatedAt)
	if err != nil {
		return nil, fmt.Errorf("create campaign: %w", err)
	}
	return &c, nil
}

func UpdateCampaign(ctx context.Context, id int64, name string, description *string) (*Campaign, error) {
	var c Campaign
	err := pool.QueryRow(ctx,
		`UPDATE campaigns SET name = $2, description = $3, updated_at = NOW()
		 WHERE id = $1
		 RETURNING id, name, description, created_at, updated_at`,
		id, name, description).
		Scan(&c.ID, &c.Name, &c.Description, &c.CreatedAt, &c.UpdatedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("update campaign: %w", err)
	}
	return &c, nil
}

func DeleteCampaign(ctx context.Context, id int64) error {
	if id == 1 {
		return fmt.Errorf("cannot delete the default campaign")
	}

	tx, err := pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin tx: %w", err)
	}
	defer tx.Rollback(ctx)

	// Cascade delete: transcriptions -> sessions -> hotwords -> ignored_users -> character_names -> presets -> categories -> campaign
	_, err = tx.Exec(ctx,
		`DELETE FROM transcriptions WHERE session_id IN (SELECT id FROM sessions WHERE campaign_id = $1)`, id)
	if err != nil {
		return fmt.Errorf("delete transcriptions: %w", err)
	}

	_, err = tx.Exec(ctx, `DELETE FROM sessions WHERE campaign_id = $1`, id)
	if err != nil {
		return fmt.Errorf("delete sessions: %w", err)
	}

	_, err = tx.Exec(ctx, `DELETE FROM hotwords WHERE campaign_id = $1`, id)
	if err != nil {
		return fmt.Errorf("delete hotwords: %w", err)
	}

	_, err = tx.Exec(ctx, `DELETE FROM ignored_users WHERE campaign_id = $1`, id)
	if err != nil {
		return fmt.Errorf("delete ignored_users: %w", err)
	}

	_, err = tx.Exec(ctx, `DELETE FROM character_names WHERE campaign_id = $1`, id)
	if err != nil {
		return fmt.Errorf("delete character_names: %w", err)
	}

	_, err = tx.Exec(ctx, `DELETE FROM nickname_presets WHERE campaign_id = $1`, id)
	if err != nil {
		return fmt.Errorf("delete nickname_presets: %w", err)
	}

	_, err = tx.Exec(ctx, `DELETE FROM nickname_categories WHERE campaign_id = $1`, id)
	if err != nil {
		return fmt.Errorf("delete nickname_categories: %w", err)
	}

	_, err = tx.Exec(ctx, `DELETE FROM campaigns WHERE id = $1`, id)
	if err != nil {
		return fmt.Errorf("delete campaign: %w", err)
	}

	return tx.Commit(ctx)
}
