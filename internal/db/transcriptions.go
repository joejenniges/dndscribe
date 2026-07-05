package db

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
)

// TranscriptionRow JSON tags must match the frontend's TranscriptionLine type
// (and SearchResult which extends it with sessionId).
// WHY: The session preview and search endpoints serialize this struct directly
// to JSON. The frontend expects "text" not "spokenText", "timestamp" not
// "createdAt". Go field names stay descriptive; JSON tags match the API contract.
type TranscriptionRow struct {
	ID              int64      `json:"id"`
	SessionID       int64      `json:"sessionId"`
	CreatedAt       time.Time  `json:"timestamp"`
	DiscordUserID   string     `json:"discordUserId"`
	DiscordUsername  string     `json:"discordUsername"`
	Nickname        *string    `json:"nickname"`
	SpokenText      string     `json:"text"`
	AudioFilenames  []string   `json:"audioFilenames"`
	RMS             *float32   `json:"rms"`
	DurationMs      *int       `json:"durationMs"`
	Confidence      *float32   `json:"confidence"`
	EditedAt        *time.Time `json:"editedAt"`
}

type InsertTranscriptionParams struct {
	SessionID       int64
	DiscordUserID   string
	DiscordUsername  string
	Nickname        *string
	SpokenText      string
	AudioFilenames  []string
	RMS             *float32
	DurationMs      *int
	Confidence      *float32
}

func InsertTranscription(ctx context.Context, params InsertTranscriptionParams) (int64, error) {
	var id int64
	err := pool.QueryRow(ctx,
		`INSERT INTO transcriptions (session_id, discord_user_id, discord_username, nickname,
		                             spoken_text, audio_filenames, rms, duration_ms, confidence)
		 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
		 RETURNING id`,
		params.SessionID, params.DiscordUserID, params.DiscordUsername, params.Nickname,
		params.SpokenText, params.AudioFilenames, params.RMS, params.DurationMs, params.Confidence).
		Scan(&id)
	if err != nil {
		return 0, fmt.Errorf("insert transcription: %w", err)
	}
	return id, nil
}

func GetTranscriptionsBySession(ctx context.Context, sessionID int64) ([]TranscriptionRow, error) {
	return queryTranscriptions(ctx,
		`SELECT id, session_id, created_at, discord_user_id, discord_username, nickname,
		        spoken_text, audio_filenames, rms, duration_ms, confidence, edited_at
		 FROM transcriptions WHERE session_id = $1
		 ORDER BY created_at ASC`, sessionID)
}

func GetTranscriptionByID(ctx context.Context, id int64) (*TranscriptionRow, error) {
	rows, err := queryTranscriptions(ctx,
		`SELECT id, session_id, created_at, discord_user_id, discord_username, nickname,
		        spoken_text, audio_filenames, rms, duration_ms, confidence, edited_at
		 FROM transcriptions WHERE id = $1`, id)
	if err != nil {
		return nil, err
	}
	if len(rows) == 0 {
		return nil, nil
	}
	return &rows[0], nil
}

func UpdateTranscription(ctx context.Context, id int64, text string) error {
	_, err := pool.Exec(ctx,
		`UPDATE transcriptions SET spoken_text = $2, edited_at = NOW() WHERE id = $1`,
		id, text)
	if err != nil {
		return fmt.Errorf("update transcription: %w", err)
	}
	return nil
}

// DeleteTranscription deletes a transcription and returns its audio filenames for cleanup.
func DeleteTranscription(ctx context.Context, id int64) ([]string, error) {
	var filenames []string
	err := pool.QueryRow(ctx,
		`DELETE FROM transcriptions WHERE id = $1
		 RETURNING COALESCE(audio_filenames, ARRAY[]::TEXT[])`, id).
		Scan(&filenames)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("delete transcription: %w", err)
	}
	return filenames, nil
}

// BulkUpdateNicknames updates nicknames for specific transcription lines and/or by user ID.
func BulkUpdateNicknames(ctx context.Context, lineIDs []int64, nicknameMap map[string]*string) error {
	tx, err := pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin tx: %w", err)
	}
	defer tx.Rollback(ctx)

	// Update specific lines by ID
	if len(lineIDs) > 0 && len(nicknameMap) > 0 {
		// For bulk update, we iterate the map and update lines matching both the ID list and user
		for userID, nickname := range nicknameMap {
			_, err := tx.Exec(ctx,
				`UPDATE transcriptions SET nickname = $1
				 WHERE id = ANY($2) AND discord_user_id = $3`,
				nickname, lineIDs, userID)
			if err != nil {
				return fmt.Errorf("bulk update nicknames: %w", err)
			}
		}
	}

	return tx.Commit(ctx)
}

func SearchTranscriptions(ctx context.Context, campaignID int64, query string, limit int) ([]TranscriptionRow, error) {
	if limit <= 0 {
		limit = 50
	}
	return queryTranscriptions(ctx,
		`SELECT t.id, t.session_id, t.created_at, t.discord_user_id, t.discord_username,
		        t.nickname, t.spoken_text, t.audio_filenames, t.rms, t.duration_ms, t.confidence, t.edited_at
		 FROM transcriptions t
		 JOIN sessions s ON s.id = t.session_id
		 WHERE s.campaign_id = $1 AND t.spoken_text ILIKE '%' || $2 || '%'
		 ORDER BY t.created_at DESC
		 LIMIT $3`,
		campaignID, query, limit)
}

func GetLastTranscriptionForUser(ctx context.Context, sessionID int64, discordUserID string) (*TranscriptionRow, error) {
	rows, err := queryTranscriptions(ctx,
		`SELECT id, session_id, created_at, discord_user_id, discord_username, nickname,
		        spoken_text, audio_filenames, rms, duration_ms, confidence, edited_at
		 FROM transcriptions
		 WHERE session_id = $1 AND discord_user_id = $2
		 ORDER BY created_at DESC LIMIT 1`,
		sessionID, discordUserID)
	if err != nil {
		return nil, err
	}
	if len(rows) == 0 {
		return nil, nil
	}
	return &rows[0], nil
}

// AppendToTranscription appends text (and optionally audio metadata) to an existing transcription.
func AppendToTranscription(ctx context.Context, id int64, text string, audioFilename *string, rms *float32, durationMs *int) error {
	// Build the update dynamically based on which optional fields are provided
	setClauses := []string{"spoken_text = spoken_text || ' ' || $2"}
	args := []any{id, text}
	argIdx := 3

	if audioFilename != nil {
		setClauses = append(setClauses, fmt.Sprintf("audio_filenames = array_append(COALESCE(audio_filenames, ARRAY[]::TEXT[]), $%d)", argIdx))
		args = append(args, *audioFilename)
		argIdx++
	}

	if rms != nil {
		// Use the latest RMS value
		setClauses = append(setClauses, fmt.Sprintf("rms = $%d", argIdx))
		args = append(args, *rms)
		argIdx++
	}

	if durationMs != nil {
		// Add to existing duration
		setClauses = append(setClauses, fmt.Sprintf("duration_ms = COALESCE(duration_ms, 0) + $%d", argIdx))
		args = append(args, *durationMs)
		argIdx++
	}

	query := fmt.Sprintf("UPDATE transcriptions SET %s WHERE id = $1", strings.Join(setClauses, ", "))
	_, err := pool.Exec(ctx, query, args...)
	if err != nil {
		return fmt.Errorf("append to transcription: %w", err)
	}
	return nil
}

// queryTranscriptions is a helper that scans rows into TranscriptionRow slices.
func queryTranscriptions(ctx context.Context, query string, args ...any) ([]TranscriptionRow, error) {
	rows, err := pool.Query(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("query transcriptions: %w", err)
	}
	defer rows.Close()

	var result []TranscriptionRow
	for rows.Next() {
		var t TranscriptionRow
		if err := rows.Scan(
			&t.ID, &t.SessionID, &t.CreatedAt, &t.DiscordUserID, &t.DiscordUsername,
			&t.Nickname, &t.SpokenText, &t.AudioFilenames, &t.RMS, &t.DurationMs,
			&t.Confidence, &t.EditedAt,
		); err != nil {
			return nil, fmt.Errorf("scan transcription: %w", err)
		}
		// WHY: Frontend expects audioFilenames as [] not null. DB returns nil
		// for NULL array columns, but JSON null breaks frontend iteration.
		if t.AudioFilenames == nil {
			t.AudioFilenames = []string{}
		}
		result = append(result, t)
	}
	return result, rows.Err()
}
