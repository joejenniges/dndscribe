package db

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
)

type Session struct {
	ID          int64      `json:"id"`
	CampaignID  int64      `json:"campaignId"`
	StartedAt   time.Time  `json:"startedAt"`
	EndedAt     *time.Time `json:"endedAt"`
	ChannelName *string    `json:"channelName"`
	ChannelID   *string    `json:"channelId"`
}

func CreateSession(ctx context.Context, campaignID int64, channelName, channelID string) (int64, error) {
	var id int64
	err := pool.QueryRow(ctx,
		`INSERT INTO sessions (campaign_id, channel_name, channel_id)
		 VALUES ($1, $2, $3)
		 RETURNING id`,
		campaignID, channelName, channelID).Scan(&id)
	if err != nil {
		return 0, fmt.Errorf("create session: %w", err)
	}
	return id, nil
}

func EndSession(ctx context.Context, id int64) error {
	_, err := pool.Exec(ctx,
		`UPDATE sessions SET ended_at = NOW() WHERE id = $1`, id)
	if err != nil {
		return fmt.Errorf("end session: %w", err)
	}
	return nil
}

func GetSession(ctx context.Context, id int64) (*Session, error) {
	var s Session
	err := pool.QueryRow(ctx,
		`SELECT id, campaign_id, started_at, ended_at, channel_name, channel_id
		 FROM sessions WHERE id = $1`, id).
		Scan(&s.ID, &s.CampaignID, &s.StartedAt, &s.EndedAt, &s.ChannelName, &s.ChannelID)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, nil
		}
		return nil, fmt.Errorf("get session: %w", err)
	}
	return &s, nil
}

func ListSessions(ctx context.Context, campaignID int64) ([]Session, error) {
	rows, err := pool.Query(ctx,
		`SELECT id, campaign_id, started_at, ended_at, channel_name, channel_id
		 FROM sessions WHERE campaign_id = $1 ORDER BY started_at DESC`, campaignID)
	if err != nil {
		return nil, fmt.Errorf("list sessions: %w", err)
	}
	defer rows.Close()

	var sessions []Session
	for rows.Next() {
		var s Session
		if err := rows.Scan(&s.ID, &s.CampaignID, &s.StartedAt, &s.EndedAt, &s.ChannelName, &s.ChannelID); err != nil {
			return nil, fmt.Errorf("scan session: %w", err)
		}
		sessions = append(sessions, s)
	}
	return sessions, rows.Err()
}

func UpdateSessionName(ctx context.Context, id int64, name string) error {
	_, err := pool.Exec(ctx,
		`UPDATE sessions SET channel_name = $2 WHERE id = $1`, id, name)
	if err != nil {
		return fmt.Errorf("update session name: %w", err)
	}
	return nil
}

// DeleteSessionFull deletes a session and its transcriptions, returning audio filenames for cleanup.
func DeleteSessionFull(ctx context.Context, id int64) ([]string, error) {
	tx, err := pool.Begin(ctx)
	if err != nil {
		return nil, fmt.Errorf("begin tx: %w", err)
	}
	defer tx.Rollback(ctx)

	// Collect audio filenames before deletion
	rows, err := tx.Query(ctx,
		`SELECT audio_filenames FROM transcriptions WHERE session_id = $1 AND audio_filenames IS NOT NULL`, id)
	if err != nil {
		return nil, fmt.Errorf("query audio filenames: %w", err)
	}

	var allFiles []string
	for rows.Next() {
		var filenames []string
		if err := rows.Scan(&filenames); err != nil {
			rows.Close()
			return nil, fmt.Errorf("scan audio filenames: %w", err)
		}
		allFiles = append(allFiles, filenames...)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate audio filenames: %w", err)
	}

	_, err = tx.Exec(ctx, `DELETE FROM transcriptions WHERE session_id = $1`, id)
	if err != nil {
		return nil, fmt.Errorf("delete transcriptions: %w", err)
	}

	_, err = tx.Exec(ctx, `DELETE FROM sessions WHERE id = $1`, id)
	if err != nil {
		return nil, fmt.Errorf("delete session: %w", err)
	}

	if err := tx.Commit(ctx); err != nil {
		return nil, fmt.Errorf("commit: %w", err)
	}
	return allFiles, nil
}

// MergeSessions merges multiple sessions into a new one, reassigning transcriptions ordered by created_at.
func MergeSessions(ctx context.Context, campaignID int64, sessionIDs []int64, name string) (int64, error) {
	if len(sessionIDs) < 2 {
		return 0, fmt.Errorf("need at least 2 sessions to merge")
	}

	tx, err := pool.Begin(ctx)
	if err != nil {
		return 0, fmt.Errorf("begin tx: %w", err)
	}
	defer tx.Rollback(ctx)

	// Create new merged session
	var newID int64
	err = tx.QueryRow(ctx,
		`INSERT INTO sessions (campaign_id, channel_name, started_at)
		 VALUES ($1, $2, NOW())
		 RETURNING id`,
		campaignID, name).Scan(&newID)
	if err != nil {
		return 0, fmt.Errorf("create merged session: %w", err)
	}

	// Reassign all transcriptions to new session
	_, err = tx.Exec(ctx,
		`UPDATE transcriptions SET session_id = $1
		 WHERE session_id = ANY($2)`,
		newID, sessionIDs)
	if err != nil {
		return 0, fmt.Errorf("reassign transcriptions: %w", err)
	}

	// Delete old sessions
	_, err = tx.Exec(ctx,
		`DELETE FROM sessions WHERE id = ANY($1)`, sessionIDs)
	if err != nil {
		return 0, fmt.Errorf("delete old sessions: %w", err)
	}

	// Set started_at and ended_at from the transcription range
	_, err = tx.Exec(ctx,
		`UPDATE sessions SET
			started_at = COALESCE((SELECT MIN(created_at) FROM transcriptions WHERE session_id = $1), NOW()),
			ended_at = (SELECT MAX(created_at) FROM transcriptions WHERE session_id = $1)
		 WHERE id = $1`, newID)
	if err != nil {
		return 0, fmt.Errorf("update merged session timestamps: %w", err)
	}

	if err := tx.Commit(ctx); err != nil {
		return 0, fmt.Errorf("commit: %w", err)
	}
	return newID, nil
}

// GetSessionPreview returns the first and last N transcriptions for a session.
func GetSessionPreview(ctx context.Context, id int64, count int) (first []TranscriptionRow, last []TranscriptionRow, err error) {
	first, err = queryTranscriptions(ctx,
		`SELECT id, session_id, created_at, discord_user_id, discord_username, nickname,
		        spoken_text, audio_filenames, rms, duration_ms, confidence, edited_at
		 FROM transcriptions WHERE session_id = $1
		 ORDER BY created_at ASC LIMIT $2`, id, count)
	if err != nil {
		return nil, nil, fmt.Errorf("get first transcriptions: %w", err)
	}

	last, err = queryTranscriptions(ctx,
		`SELECT id, session_id, created_at, discord_user_id, discord_username, nickname,
		        spoken_text, audio_filenames, rms, duration_ms, confidence, edited_at
		 FROM transcriptions WHERE session_id = $1
		 ORDER BY created_at DESC LIMIT $2`, id, count)
	if err != nil {
		return nil, nil, fmt.Errorf("get last transcriptions: %w", err)
	}

	// Reverse last slice so it's in chronological order
	for i, j := 0, len(last)-1; i < j; i, j = i+1, j-1 {
		last[i], last[j] = last[j], last[i]
	}

	return first, last, nil
}

func GetLatestSession(ctx context.Context, campaignID *int64) (*Session, error) {
	var s Session
	var err error

	if campaignID != nil {
		err = pool.QueryRow(ctx,
			`SELECT id, campaign_id, started_at, ended_at, channel_name, channel_id
			 FROM sessions WHERE campaign_id = $1
			 ORDER BY started_at DESC LIMIT 1`, *campaignID).
			Scan(&s.ID, &s.CampaignID, &s.StartedAt, &s.EndedAt, &s.ChannelName, &s.ChannelID)
	} else {
		err = pool.QueryRow(ctx,
			`SELECT id, campaign_id, started_at, ended_at, channel_name, channel_id
			 FROM sessions ORDER BY started_at DESC LIMIT 1`).
			Scan(&s.ID, &s.CampaignID, &s.StartedAt, &s.EndedAt, &s.ChannelName, &s.ChannelID)
	}

	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("get latest session: %w", err)
	}
	return &s, nil
}
