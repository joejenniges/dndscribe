package db

import (
	"context"
	"fmt"
	"time"
)

// SpeakerRow is one diarization speaker profile: a distinct voice detected on
// a Discord user's audio stream. Embedding is the L2-normalized running-mean
// centroid used for cosine matching.
type SpeakerRow struct {
	ID            int64     `json:"id"`
	CampaignID    int64     `json:"campaignId"`
	DiscordUserID string    `json:"discordUserId"`
	Label         string    `json:"label"`
	DisplayName   *string   `json:"displayName"`
	Embedding     []float32 `json:"-"` // internal; not exposed over the API
	SampleCount   int       `json:"sampleCount"`
	CreatedAt     time.Time `json:"createdAt"`
}

// GetSpeakersByCampaign returns all speaker profiles for a campaign,
// embeddings included (used to seed the diarizer's in-memory cache).
func GetSpeakersByCampaign(ctx context.Context, campaignID int64) ([]SpeakerRow, error) {
	rows, err := pool.Query(ctx,
		`SELECT id, campaign_id, discord_user_id, label, display_name, embedding, sample_count, created_at
		 FROM speakers WHERE campaign_id = $1
		 ORDER BY discord_user_id, id`, campaignID)
	if err != nil {
		return nil, fmt.Errorf("query speakers: %w", err)
	}
	defer rows.Close()

	var result []SpeakerRow
	for rows.Next() {
		var s SpeakerRow
		if err := rows.Scan(&s.ID, &s.CampaignID, &s.DiscordUserID, &s.Label,
			&s.DisplayName, &s.Embedding, &s.SampleCount, &s.CreatedAt); err != nil {
			return nil, fmt.Errorf("scan speaker: %w", err)
		}
		result = append(result, s)
	}
	return result, rows.Err()
}

// InsertSpeaker creates a new speaker profile and returns its ID.
func InsertSpeaker(ctx context.Context, campaignID int64, discordUserID, label string, embedding []float32) (int64, error) {
	var id int64
	err := pool.QueryRow(ctx,
		`INSERT INTO speakers (campaign_id, discord_user_id, label, embedding)
		 VALUES ($1, $2, $3, $4) RETURNING id`,
		campaignID, discordUserID, label, embedding).Scan(&id)
	if err != nil {
		return 0, fmt.Errorf("insert speaker: %w", err)
	}
	return id, nil
}

// UpdateSpeakerEmbedding persists an updated centroid and sample count.
func UpdateSpeakerEmbedding(ctx context.Context, id int64, embedding []float32, sampleCount int) error {
	_, err := pool.Exec(ctx,
		`UPDATE speakers SET embedding = $2, sample_count = $3, updated_at = NOW() WHERE id = $1`,
		id, embedding, sampleCount)
	if err != nil {
		return fmt.Errorf("update speaker embedding: %w", err)
	}
	return nil
}

// UpdateSpeakerDisplayName sets the human-assigned name for a speaker
// (phase-2 UI). Empty name clears it back to the auto label.
func UpdateSpeakerDisplayName(ctx context.Context, id int64, name string) (bool, error) {
	var displayName *string
	if name != "" {
		displayName = &name
	}
	tag, err := pool.Exec(ctx,
		`UPDATE speakers SET display_name = $2, updated_at = NOW() WHERE id = $1`,
		id, displayName)
	if err != nil {
		return false, fmt.Errorf("update speaker name: %w", err)
	}
	return tag.RowsAffected() > 0, nil
}

// DeleteSpeaker removes a speaker profile. Transcription rows keep their
// speaker_id (dangling reference reads as no label — harmless).
func DeleteSpeaker(ctx context.Context, id int64) (bool, error) {
	tag, err := pool.Exec(ctx, `DELETE FROM speakers WHERE id = $1`, id)
	if err != nil {
		return false, fmt.Errorf("delete speaker: %w", err)
	}
	return tag.RowsAffected() > 0, nil
}
