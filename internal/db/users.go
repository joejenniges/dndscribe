package db

import (
	"context"
	"fmt"
	"time"
)

type IgnoredUser struct {
	DiscordUserID   string    `json:"discordUserId"`
	DiscordUsername  string    `json:"discordUsername"`
	IgnoredAt       time.Time `json:"ignoredAt"`
}

func GetIgnoredUsers(ctx context.Context, campaignID int64) ([]IgnoredUser, error) {
	rows, err := pool.Query(ctx,
		`SELECT discord_user_id, discord_username, ignored_at
		 FROM ignored_users WHERE campaign_id = $1
		 ORDER BY ignored_at`, campaignID)
	if err != nil {
		return nil, fmt.Errorf("get ignored users: %w", err)
	}
	defer rows.Close()

	var users []IgnoredUser
	for rows.Next() {
		var u IgnoredUser
		if err := rows.Scan(&u.DiscordUserID, &u.DiscordUsername, &u.IgnoredAt); err != nil {
			return nil, fmt.Errorf("scan ignored user: %w", err)
		}
		users = append(users, u)
	}
	return users, rows.Err()
}

// AddIgnoredUser adds a user to the ignore list. Returns true if actually inserted.
func AddIgnoredUser(ctx context.Context, campaignID int64, discordUserID, discordUsername string) (bool, error) {
	tag, err := pool.Exec(ctx,
		`INSERT INTO ignored_users (campaign_id, discord_user_id, discord_username)
		 VALUES ($1, $2, $3)
		 ON CONFLICT (campaign_id, discord_user_id) DO NOTHING`,
		campaignID, discordUserID, discordUsername)
	if err != nil {
		return false, fmt.Errorf("add ignored user: %w", err)
	}
	return tag.RowsAffected() > 0, nil
}

// RemoveIgnoredUser removes a user from the ignore list. Returns true if actually deleted.
func RemoveIgnoredUser(ctx context.Context, campaignID int64, discordUserID string) (bool, error) {
	tag, err := pool.Exec(ctx,
		`DELETE FROM ignored_users WHERE campaign_id = $1 AND discord_user_id = $2`,
		campaignID, discordUserID)
	if err != nil {
		return false, fmt.Errorf("remove ignored user: %w", err)
	}
	return tag.RowsAffected() > 0, nil
}

// GetCharacterNames returns a map of discord_user_id -> character name for a campaign.
func GetCharacterNames(ctx context.Context, campaignID int64) (map[string]string, error) {
	rows, err := pool.Query(ctx,
		`SELECT discord_user_id, name FROM character_names WHERE campaign_id = $1`,
		campaignID)
	if err != nil {
		return nil, fmt.Errorf("get character names: %w", err)
	}
	defer rows.Close()

	names := make(map[string]string)
	for rows.Next() {
		var userID, name string
		if err := rows.Scan(&userID, &name); err != nil {
			return nil, fmt.Errorf("scan character name: %w", err)
		}
		names[userID] = name
	}
	return names, rows.Err()
}

// SetCharacterName sets or removes a character name for a user in a campaign.
// Pass nil to remove the character name.
func SetCharacterName(ctx context.Context, campaignID int64, discordUserID string, name *string) error {
	if name == nil {
		_, err := pool.Exec(ctx,
			`DELETE FROM character_names WHERE campaign_id = $1 AND discord_user_id = $2`,
			campaignID, discordUserID)
		if err != nil {
			return fmt.Errorf("delete character name: %w", err)
		}
		return nil
	}

	_, err := pool.Exec(ctx,
		`INSERT INTO character_names (campaign_id, discord_user_id, name, updated_at)
		 VALUES ($1, $2, $3, NOW())
		 ON CONFLICT (campaign_id, discord_user_id)
		 DO UPDATE SET name = $3, updated_at = NOW()`,
		campaignID, discordUserID, *name)
	if err != nil {
		return fmt.Errorf("set character name: %w", err)
	}
	return nil
}
