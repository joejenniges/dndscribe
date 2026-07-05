package db

import (
	"context"
	"fmt"
	"time"
)

type NicknamePreset struct {
	ID            int64     `json:"id"`
	DiscordUserID string    `json:"discordUserId"`
	Label         string    `json:"label"`
	Position      int       `json:"position"`
	CategoryID    *int64    `json:"categoryId"`
	CampaignID    int64     `json:"campaignId"`
	CreatedAt     time.Time `json:"createdAt"`
}

type NicknameCategory struct {
	ID            int64     `json:"id"`
	DiscordUserID string    `json:"discordUserId"`
	Name          string    `json:"name"`
	Position      int       `json:"position"`
	CampaignID    int64     `json:"campaignId"`
	CreatedAt     time.Time `json:"createdAt"`
}

func GetNicknamePresets(ctx context.Context, campaignID int64, discordUserID *string) ([]NicknamePreset, error) {
	var query string
	var args []any

	if discordUserID != nil {
		query = `SELECT id, discord_user_id, label, position, category_id, campaign_id, created_at
		         FROM nickname_presets WHERE campaign_id = $1 AND discord_user_id = $2
		         ORDER BY position`
		args = []any{campaignID, *discordUserID}
	} else {
		query = `SELECT id, discord_user_id, label, position, category_id, campaign_id, created_at
		         FROM nickname_presets WHERE campaign_id = $1
		         ORDER BY position`
		args = []any{campaignID}
	}

	rows, err := pool.Query(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("get nickname presets: %w", err)
	}
	defer rows.Close()

	var presets []NicknamePreset
	for rows.Next() {
		var p NicknamePreset
		if err := rows.Scan(&p.ID, &p.DiscordUserID, &p.Label, &p.Position, &p.CategoryID, &p.CampaignID, &p.CreatedAt); err != nil {
			return nil, fmt.Errorf("scan nickname preset: %w", err)
		}
		presets = append(presets, p)
	}
	return presets, rows.Err()
}

func AddNicknamePreset(ctx context.Context, campaignID int64, discordUserID, label string, position int) (int64, error) {
	var id int64
	err := pool.QueryRow(ctx,
		`INSERT INTO nickname_presets (campaign_id, discord_user_id, label, position)
		 VALUES ($1, $2, $3, $4)
		 RETURNING id`,
		campaignID, discordUserID, label, position).Scan(&id)
	if err != nil {
		return 0, fmt.Errorf("add nickname preset: %w", err)
	}
	return id, nil
}

func UpdateNicknamePreset(ctx context.Context, id int64, label string, position *int) error {
	if position != nil {
		_, err := pool.Exec(ctx,
			`UPDATE nickname_presets SET label = $2, position = $3 WHERE id = $1`,
			id, label, *position)
		if err != nil {
			return fmt.Errorf("update nickname preset: %w", err)
		}
	} else {
		_, err := pool.Exec(ctx,
			`UPDATE nickname_presets SET label = $2 WHERE id = $1`,
			id, label)
		if err != nil {
			return fmt.Errorf("update nickname preset: %w", err)
		}
	}
	return nil
}

func DeleteNicknamePreset(ctx context.Context, id int64) error {
	_, err := pool.Exec(ctx, `DELETE FROM nickname_presets WHERE id = $1`, id)
	if err != nil {
		return fmt.Errorf("delete nickname preset: %w", err)
	}
	return nil
}

func MovePresetToCategory(ctx context.Context, presetID int64, categoryID *int64) error {
	_, err := pool.Exec(ctx,
		`UPDATE nickname_presets SET category_id = $2 WHERE id = $1`,
		presetID, categoryID)
	if err != nil {
		return fmt.Errorf("move preset to category: %w", err)
	}
	return nil
}

func GetNicknameCategories(ctx context.Context, campaignID int64, discordUserID *string) ([]NicknameCategory, error) {
	var query string
	var args []any

	if discordUserID != nil {
		query = `SELECT id, discord_user_id, name, position, campaign_id, created_at
		         FROM nickname_categories WHERE campaign_id = $1 AND discord_user_id = $2
		         ORDER BY position`
		args = []any{campaignID, *discordUserID}
	} else {
		query = `SELECT id, discord_user_id, name, position, campaign_id, created_at
		         FROM nickname_categories WHERE campaign_id = $1
		         ORDER BY position`
		args = []any{campaignID}
	}

	rows, err := pool.Query(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("get nickname categories: %w", err)
	}
	defer rows.Close()

	var categories []NicknameCategory
	for rows.Next() {
		var c NicknameCategory
		if err := rows.Scan(&c.ID, &c.DiscordUserID, &c.Name, &c.Position, &c.CampaignID, &c.CreatedAt); err != nil {
			return nil, fmt.Errorf("scan nickname category: %w", err)
		}
		categories = append(categories, c)
	}
	return categories, rows.Err()
}

func AddNicknameCategory(ctx context.Context, campaignID int64, discordUserID, name string, position int) (int64, error) {
	var id int64
	err := pool.QueryRow(ctx,
		`INSERT INTO nickname_categories (campaign_id, discord_user_id, name, position)
		 VALUES ($1, $2, $3, $4)
		 RETURNING id`,
		campaignID, discordUserID, name, position).Scan(&id)
	if err != nil {
		return 0, fmt.Errorf("add nickname category: %w", err)
	}
	return id, nil
}

func UpdateNicknameCategory(ctx context.Context, id int64, name string, position *int) error {
	if position != nil {
		_, err := pool.Exec(ctx,
			`UPDATE nickname_categories SET name = $2, position = $3 WHERE id = $1`,
			id, name, *position)
		if err != nil {
			return fmt.Errorf("update nickname category: %w", err)
		}
	} else {
		_, err := pool.Exec(ctx,
			`UPDATE nickname_categories SET name = $2 WHERE id = $1`,
			id, name)
		if err != nil {
			return fmt.Errorf("update nickname category: %w", err)
		}
	}
	return nil
}

func DeleteNicknameCategory(ctx context.Context, id int64) error {
	_, err := pool.Exec(ctx, `DELETE FROM nickname_categories WHERE id = $1`, id)
	if err != nil {
		return fmt.Errorf("delete nickname category: %w", err)
	}
	return nil
}
