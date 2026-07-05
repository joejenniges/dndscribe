package db

import (
	"context"
	"fmt"
	"strings"
)

func GetHotwords(ctx context.Context, campaignID int64) ([]string, error) {
	rows, err := pool.Query(ctx,
		`SELECT word FROM hotwords WHERE campaign_id = $1 ORDER BY word`, campaignID)
	if err != nil {
		return nil, fmt.Errorf("get hotwords: %w", err)
	}
	defer rows.Close()

	var words []string
	for rows.Next() {
		var w string
		if err := rows.Scan(&w); err != nil {
			return nil, fmt.Errorf("scan hotword: %w", err)
		}
		words = append(words, w)
	}
	return words, rows.Err()
}

// GetAllHotwords returns the union of hotwords across all campaigns.
func GetAllHotwords(ctx context.Context) ([]string, error) {
	rows, err := pool.Query(ctx,
		`SELECT DISTINCT word FROM hotwords ORDER BY word`)
	if err != nil {
		return nil, fmt.Errorf("get all hotwords: %w", err)
	}
	defer rows.Close()

	var words []string
	for rows.Next() {
		var w string
		if err := rows.Scan(&w); err != nil {
			return nil, fmt.Errorf("scan hotword: %w", err)
		}
		words = append(words, w)
	}
	return words, rows.Err()
}

// AddHotword adds a hotword for a campaign. Returns true if it was actually inserted (not a duplicate).
func AddHotword(ctx context.Context, campaignID int64, word string) (bool, error) {
	word = strings.TrimSpace(word)
	if word == "" {
		return false, fmt.Errorf("hotword cannot be empty")
	}

	tag, err := pool.Exec(ctx,
		`INSERT INTO hotwords (campaign_id, word)
		 VALUES ($1, $2)
		 ON CONFLICT (campaign_id, LOWER(word)) DO NOTHING`,
		campaignID, word)
	if err != nil {
		return false, fmt.Errorf("add hotword: %w", err)
	}
	return tag.RowsAffected() > 0, nil
}

// RemoveHotword removes a hotword. Returns true if a row was actually deleted.
func RemoveHotword(ctx context.Context, campaignID int64, word string) (bool, error) {
	tag, err := pool.Exec(ctx,
		`DELETE FROM hotwords WHERE campaign_id = $1 AND LOWER(word) = LOWER($2)`,
		campaignID, word)
	if err != nil {
		return false, fmt.Errorf("remove hotword: %w", err)
	}
	return tag.RowsAffected() > 0, nil
}

// UpdateHotword renames a hotword. Returns true if a row was actually updated.
func UpdateHotword(ctx context.Context, campaignID int64, oldWord, newWord string) (bool, error) {
	newWord = strings.TrimSpace(newWord)
	if newWord == "" {
		return false, fmt.Errorf("hotword cannot be empty")
	}

	tag, err := pool.Exec(ctx,
		`UPDATE hotwords SET word = $3
		 WHERE campaign_id = $1 AND LOWER(word) = LOWER($2)`,
		campaignID, oldWord, newWord)
	if err != nil {
		return false, fmt.Errorf("update hotword: %w", err)
	}
	return tag.RowsAffected() > 0, nil
}
