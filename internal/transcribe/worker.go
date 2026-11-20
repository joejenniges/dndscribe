package transcribe

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/joe/dndscribe-go/internal/db"
	"github.com/joe/dndscribe-go/internal/web"
)

// Compile-time interface check.
var _ web.TranscriptionWorker = (*Worker)(nil)

// Minimum PCM size to process: 0.25s at 48kHz mono 16-bit = 24000 bytes.
const minPCMBytes = 24000

// Task represents a queued transcription job from the voice recorder.
type Task struct {
	PCM           []byte
	UserID        string
	Username      string
	Timestamp     time.Time
	AudioFilename string
	RMS           float32
	DurationMs    int
	Nickname      string
}

// Worker processes audio chunks through the whisper engine and manages
// transcription state. It implements web.TranscriptionWorker.
type Worker struct {
	engine       Engine          // batch engine (whisper); nil in streaming mode
	streamEngine StreamingEngine // streaming engine (sherpa); nil in batch mode
	diarizer     *Diarizer       // nil when diarization is disabled
	queue        chan *Task
	done         chan struct{}
	wg           sync.WaitGroup
	hotwordsFile string
	recordingDir string

	mu          sync.RWMutex
	sessionID   *int64
	campaignID  *int64
	lastContext map[string]string // userID -> last 200 chars of transcription

	// Callbacks for the web/bot layers.
	OnLine      func(line web.TranscriptionLine)
	OnHotwords  func(hotwords []string)
	OnFinalized func()
	// OnPartial fires for provisional streaming hypotheses (streaming engine
	// only). Ephemeral — not persisted; the web layer renders and supersedes
	// these live. speakerLabel is the diarization label ("S1"...), or the
	// speaker's display name once assigned; empty when unknown.
	OnPartial func(userID, username, nickname, text, speakerLabel string)
}

// streaming reports whether the worker is running a streaming engine.
func (w *Worker) streaming() bool { return w.streamEngine != nil }

// NewWorker creates a new transcription worker. Call Init to load the model
// and Start to begin processing.
func NewWorker(recordingDir string) *Worker {
	return &Worker{
		queue:        make(chan *Task, 256),
		done:         make(chan struct{}),
		lastContext:  make(map[string]string),
		recordingDir: recordingDir,
		hotwordsFile: "hotwords.txt",
	}
}

// SetEngine sets the batch transcription engine (whisper). Mutually exclusive
// with SetStreamEngine.
func (w *Worker) SetEngine(e Engine) {
	w.engine = e
}

// SetStreamEngine sets the streaming engine (sherpa) and wires its result
// callback into the worker's final-insert / partial-broadcast paths. Mutually
// exclusive with SetEngine.
func (w *Worker) SetStreamEngine(e StreamingEngine) {
	w.streamEngine = e
	e.SetCallback(w.handleStreamResult)
}

// SetDiarizer attaches speaker identification (nil = off). Engine-independent:
// the whisper path calls identifySpeaker per flushed chunk; the sherpa engine
// gets it as its identify callback (wired in Init).
func (w *Worker) SetDiarizer(d *Diarizer) { w.diarizer = d }

// identifySpeaker resolves the speaker for a segment of 16kHz audio in the
// context of the active campaign. Returns (0, "") when diarization is off,
// no campaign is active, or the segment is too short.
func (w *Worker) identifySpeaker(userID string, samples16k []float32) (int64, string) {
	if w.diarizer == nil {
		return 0, ""
	}
	w.mu.RLock()
	cid := w.campaignID
	w.mu.RUnlock()
	if cid == nil {
		return 0, ""
	}
	return w.diarizer.Identify(context.Background(), *cid, userID, samples16k)
}

// Start launches the background processing goroutine.
func (w *Worker) Start() {
	w.wg.Add(1)
	go w.processLoop()
	log.Println("Transcription worker started")
}

// Stop signals the worker to stop and waits for it to finish.
func (w *Worker) Stop() {
	close(w.done)
	w.wg.Wait()
	if w.engine != nil {
		w.engine.Close()
	}
	if w.streamEngine != nil {
		w.streamEngine.Close()
	}
	w.diarizer.Close() // nil-safe
	log.Printf("Transcription worker stopped, %d tasks remaining in queue", len(w.queue))
}

// FeedFrame supplies a single decoded mono 48kHz s16le frame for a user. In
// streaming mode it feeds the streaming engine; in batch mode it is a no-op
// (the batch path uses silence-flushed chunks via AddTask instead). Called per
// voice packet, so it must stay cheap.
func (w *Worker) FeedFrame(userID, username, nickname string, mono48k []int16) {
	if w.streamEngine == nil {
		return
	}
	w.streamEngine.Feed(userID, username, nickname, mono48k)
}

// AddTask is called by the voice layer's silence flush. In batch mode it
// enqueues the utterance chunk for whisper. In streaming mode the chunk itself
// is unused (the engine already consumed the audio frame-by-frame via
// FeedFrame) — the call is repurposed as the utterance-boundary signal to
// finalize and reset that user's stream. This signature matches the
// bot.TranscriptionWorker interface.
func (w *Worker) AddTask(userID string, pcm []byte, timestamp time.Time, username, nickname string, rms float32, durationMs int, audioFilename string) {
	if w.streaming() {
		w.streamEngine.Flush(userID)
		return
	}
	if len(pcm) < minPCMBytes {
		log.Printf("Skipping short audio for %s: %d bytes", username, len(pcm))
		return
	}

	task := &Task{
		PCM:           pcm,
		UserID:        userID,
		Username:      username,
		Timestamp:     timestamp,
		RMS:           rms,
		DurationMs:    durationMs,
		Nickname:      nickname,
		AudioFilename: audioFilename,
	}

	select {
	case w.queue <- task:
		log.Printf("Queued transcription for %s, %d bytes", username, len(pcm))
	default:
		log.Printf("WARNING: transcription queue full, dropping task for %s", username)
	}
}

// AddTaskWithFile enqueues an audio chunk with an associated audio filename.
func (w *Worker) AddTaskWithFile(task *Task) {
	if len(task.PCM) < minPCMBytes {
		log.Printf("Skipping short audio for %s: %d bytes", task.Username, len(task.PCM))
		return
	}

	select {
	case w.queue <- task:
		log.Printf("Queued transcription for %s, %d bytes", task.Username, len(task.PCM))
	default:
		log.Printf("WARNING: transcription queue full, dropping task for %s", task.Username)
	}
}

// --- web.TranscriptionWorker interface ---

// GetLines returns transcription lines for a session. If sessionID is nil,
// uses the worker's active session or the latest session from DB.
func (w *Worker) GetLines(sessionID *int64) ([]web.TranscriptionLine, error) {
	ctx := context.Background()

	sid := sessionID
	if sid == nil {
		w.mu.RLock()
		sid = w.sessionID
		w.mu.RUnlock()
	}
	if sid == nil {
		latest, err := db.GetLatestSession(ctx, nil)
		if err != nil {
			return nil, err
		}
		if latest == nil {
			return nil, nil
		}
		sid = &latest.ID
	}

	rows, err := db.GetTranscriptionsBySession(ctx, *sid)
	if err != nil {
		return nil, err
	}

	lines := make([]web.TranscriptionLine, len(rows))
	for i, r := range rows {
		lines[i] = rowToLine(r)
	}
	return lines, nil
}

// UpdateLine updates a transcription line's text and auto-detects new hotwords
// from the edit. Returns the list of auto-added hotword strings.
func (w *Worker) UpdateLine(id int64, text string) ([]string, error) {
	ctx := context.Background()

	row, err := db.GetTranscriptionByID(ctx, id)
	if err != nil {
		return nil, err
	}
	if row == nil {
		return nil, nil
	}

	w.mu.RLock()
	campaignID := w.campaignID
	w.mu.RUnlock()

	cid := int64(1) // default campaign
	if campaignID != nil {
		cid = *campaignID
	}

	// Detect new hotwords from the correction.
	oldText := row.SpokenText
	candidates := DetectNewHotwords(oldText, text)

	var autoAdded []string
	if len(candidates) > 0 {
		currentHotwords, err := db.GetHotwords(ctx, cid)
		if err != nil {
			log.Printf("Failed to get hotwords for auto-add: %v", err)
		} else {
			existing := make(map[string]bool)
			for _, h := range currentHotwords {
				existing[strings.ToLower(h)] = true
			}
			for _, word := range candidates {
				if existing[strings.ToLower(word)] {
					continue
				}
				added, err := db.AddHotword(ctx, cid, word)
				if err != nil {
					log.Printf("Failed to auto-add hotword %q: %v", word, err)
					continue
				}
				if added {
					autoAdded = append(autoAdded, word)
				}
			}
		}
	}

	if err := db.UpdateTranscription(ctx, id, text); err != nil {
		return nil, err
	}

	if len(autoAdded) > 0 {
		log.Printf("Auto-added hotwords from correction: %s", strings.Join(autoAdded, ", "))
		// Sync hotwords file and notify listeners.
		if err := SyncHotwordsFile(w.hotwordsFile); err != nil {
			log.Printf("Failed to sync hotwords file: %v", err)
		}
		if w.OnHotwords != nil {
			hotwords, _ := db.GetHotwords(ctx, cid)
			w.OnHotwords(hotwords)
		}
	}

	return autoAdded, nil
}

// DeleteLine deletes a transcription line and cleans up its audio files.
func (w *Worker) DeleteLine(id int64) (bool, error) {
	ctx := context.Background()

	filenames, err := db.DeleteTranscription(ctx, id)
	if err != nil {
		return false, err
	}
	if filenames == nil {
		return false, nil
	}

	// Clean up audio files.
	for _, f := range filenames {
		path := filepath.Join(w.recordingDir, "raw", f)
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			log.Printf("Failed to remove audio file %s: %v", path, err)
		}
	}

	log.Printf("Deleted transcription %d (%d audio files)", id, len(filenames))
	return true, nil
}

// GetSessionID returns the active session ID, or nil if no session.
func (w *Worker) GetSessionID() *int64 {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return w.sessionID
}

// GetCampaignID returns the active campaign ID, or nil if no session.
func (w *Worker) GetCampaignID() *int64 {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return w.campaignID
}

// SetCampaignID sets the active campaign ID.
func (w *Worker) SetCampaignID(id *int64) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.campaignID = id
}

// UpdateSpeakerName persists a diarization speaker's display name and pushes
// it into the live diarizer cache so subsequent lines/partials use it.
func (w *Worker) UpdateSpeakerName(id int64, name string) (bool, error) {
	ok, err := db.UpdateSpeakerDisplayName(context.Background(), id, name)
	if err != nil || !ok {
		return ok, err
	}
	w.diarizer.SetDisplayName(id, name) // nil-safe
	return true, nil
}

// RemoveSpeaker deletes a speaker profile from the DB and the live diarizer
// cache, so that voice re-enrolls fresh on its next utterance.
func (w *Worker) RemoveSpeaker(id int64) (bool, error) {
	ok, err := db.DeleteSpeaker(context.Background(), id)
	if err != nil || !ok {
		return ok, err
	}
	w.diarizer.Remove(id) // nil-safe
	return true, nil
}

// SetSessionID sets the active session ID.
func (w *Worker) SetSessionID(id *int64) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.sessionID = id
	if id == nil {
		w.lastContext = make(map[string]string)
	}
}

// Finalize waits for the queue to drain, merges consecutive same-speaker lines,
// exports a transcription file, and ends the session in DB.
// This signature matches bot.TranscriptionWorker interface.
func (w *Worker) Finalize() {
	result, err := w.FinalizeSession()
	if err != nil {
		log.Printf("Finalize error: %v", err)
	}
	if result != nil {
		log.Printf("Session transcription saved to %s", *result)
	}
}

// FinalizeSession is like Finalize but returns the output file path and error.
func (w *Worker) FinalizeSession() (*string, error) {
	// Wait for queue to drain.
	for len(w.queue) > 0 {
		time.Sleep(200 * time.Millisecond)
	}
	// Give in-flight task a moment to finish.
	time.Sleep(500 * time.Millisecond)

	w.mu.Lock()
	sid := w.sessionID
	w.sessionID = nil
	w.campaignID = nil
	w.lastContext = make(map[string]string)
	w.mu.Unlock()

	if sid == nil {
		if w.OnFinalized != nil {
			w.OnFinalized()
		}
		return nil, nil
	}

	ctx := context.Background()
	result, err := w.finalizeSession(ctx, *sid)
	if err != nil {
		if w.OnFinalized != nil {
			w.OnFinalized()
		}
		return nil, err
	}

	if w.OnFinalized != nil {
		w.OnFinalized()
	}

	return result, nil
}

// --- Internal processing ---

func (w *Worker) processLoop() {
	defer w.wg.Done()
	for {
		select {
		case task := <-w.queue:
			w.processTask(task)
		case <-w.done:
			// Exit immediately on shutdown. Queued audio tails (the last
			// ~15s per speaker from Recorder.Stop's final flushes) are
			// intentionally dropped: whisper inference on each can take
			// several seconds, and making shutdown wait for all of them
			// was visibly blocking the process for 10-20s. The session's
			// main transcript is already persisted; a few trailing chunks
			// at session-end are acceptable loss in exchange for a fast exit.
			return
		}
	}
}

var nonASCII = regexp.MustCompile(`[^\x00-\x7F]`)

func (w *Worker) processTask(task *Task) {
	if w.engine == nil {
		log.Printf("No transcription engine, dropping task for %s", task.Username)
		return
	}

	// Resample 48kHz mono s16le -> 16kHz float32.
	samples := Resample48to16(task.PCM)
	if len(samples) == 0 {
		return
	}

	// Build initial prompt with hotwords and previous context.
	w.mu.RLock()
	sid := w.sessionID
	cid := w.campaignID
	prevContext := w.lastContext[task.UserID]
	w.mu.RUnlock()

	if sid == nil {
		log.Printf("No active session, dropping transcription for %s", task.Username)
		return
	}

	var hotwords []string
	if cid != nil {
		var err error
		hotwords, err = db.GetHotwords(context.Background(), *cid)
		if err != nil {
			log.Printf("Failed to load hotwords: %v", err)
		}
	}
	prompt := BuildInitialPrompt(hotwords, prevContext)

	// Transcribe.
	text, confidence, err := w.engine.Transcribe(samples, prompt)
	if err != nil {
		log.Printf("Transcription error for %s: %v", task.Username, err)
		return
	}
	if text == "" {
		return
	}

	// Filter hallucinations.
	if IsHallucination(text, task.RMS, confidence) {
		log.Printf("Filtered hallucination for %s: %q (rms=%.0f, conf=%.2f)",
			task.Username, text, task.RMS, confidence)
		return
	}

	// Strip non-ASCII.
	asciiText := nonASCII.ReplaceAllString(text, "")
	asciiText = strings.TrimSpace(asciiText)
	if asciiText == "" {
		return
	}
	asciiUsername := nonASCII.ReplaceAllString(task.Username, "")

	// Update last context for this user.
	w.mu.Lock()
	ctx200 := asciiText
	if len(ctx200) > 200 {
		ctx200 = ctx200[len(ctx200)-200:]
	}
	w.lastContext[task.UserID] = ctx200
	w.mu.Unlock()

	// Diarization: the resampled 16k samples are exactly what the embedding
	// model wants, so identification is free of extra audio work here.
	speakerID, speakerLabel := w.identifySpeaker(task.UserID, samples)

	// Insert into DB.
	var nickname *string
	if task.Nickname != "" {
		nickname = &task.Nickname
	}
	audioFilenames := []string{}
	if task.AudioFilename != "" {
		audioFilenames = []string{task.AudioFilename}
	}

	rms := task.RMS
	durationMs := task.DurationMs
	var speakerIDPtr *int64
	var speakerLabelPtr *string
	if speakerID != 0 {
		speakerIDPtr = &speakerID
		speakerLabelPtr = &speakerLabel
	}

	dbID, err := db.InsertTranscription(context.Background(), db.InsertTranscriptionParams{
		SessionID:      *sid,
		DiscordUserID:  task.UserID,
		DiscordUsername: asciiUsername,
		Nickname:       nickname,
		SpokenText:     asciiText,
		AudioFilenames: audioFilenames,
		RMS:            &rms,
		DurationMs:     &durationMs,
		Confidence:     &confidence,
		SpeakerID:      speakerIDPtr,
	})
	if err != nil {
		log.Printf("Failed to insert transcription: %v", err)
		return
	}

	log.Printf("Transcribed %s: %s", task.Username, truncate(asciiText, 80))

	// Notify listeners.
	if w.OnLine != nil {
		w.OnLine(web.TranscriptionLine{
			ID:             dbID,
			Timestamp:      task.Timestamp.Format(time.RFC3339Nano),
			DiscordUsername: asciiUsername,
			Nickname:       nickname,
			Text:           asciiText,
			AudioFilenames: audioFilenames,
			RMS:            &rms,
			Confidence:     &confidence,
			SpeakerID:      speakerIDPtr,
			SpeakerLabel:   speakerLabelPtr,
		})
	}
}

// handleStreamResult is the callback the streaming engine invokes. Partials are
// forwarded to OnPartial (ephemeral, not persisted). Finals are stripped,
// inserted into the DB, and broadcast via OnLine — the same persistence path
// the batch engine uses, so session history / export / search all keep working.
//
// May be called concurrently from per-user engine goroutines; the DB pool and
// the broadcast channel are safe for that, and sessionID is read under w.mu.
func (w *Worker) handleStreamResult(r StreamingResult) {
	if !r.IsFinal {
		if w.OnPartial != nil {
			text := strings.TrimSpace(nonASCII.ReplaceAllString(r.Text, ""))
			if text != "" {
				w.OnPartial(r.UserID, r.Username, r.Nickname, text, r.SpeakerLabel)
			}
		}
		return
	}

	asciiText := strings.TrimSpace(nonASCII.ReplaceAllString(r.Text, ""))
	if asciiText == "" {
		return
	}

	w.mu.RLock()
	sid := w.sessionID
	w.mu.RUnlock()
	if sid == nil {
		log.Printf("No active session, dropping streaming final for %s", r.Username)
		return
	}

	asciiUsername := nonASCII.ReplaceAllString(r.Username, "")
	var nickname *string
	if r.Nickname != "" {
		nickname = &r.Nickname
	}
	rms := r.RMS
	durationMs := r.DurationMs
	var speakerIDPtr *int64
	var speakerLabelPtr *string
	if r.SpeakerID != 0 {
		sID, sLabel := r.SpeakerID, r.SpeakerLabel
		speakerIDPtr = &sID
		speakerLabelPtr = &sLabel
	}

	dbID, err := db.InsertTranscription(context.Background(), db.InsertTranscriptionParams{
		SessionID:       *sid,
		DiscordUserID:   r.UserID,
		DiscordUsername: asciiUsername,
		Nickname:        nickname,
		SpokenText:      asciiText,
		AudioFilenames:  []string{},
		RMS:             &rms,
		DurationMs:      &durationMs,
		SpeakerID:       speakerIDPtr,
	})
	if err != nil {
		log.Printf("Failed to insert streaming transcription: %v", err)
		return
	}

	log.Printf("Transcribed (stream) %s: %s", r.Username, truncate(asciiText, 80))

	if w.OnLine != nil {
		ts := r.Timestamp
		if ts.IsZero() {
			ts = time.Now()
		}
		w.OnLine(web.TranscriptionLine{
			ID:              dbID,
			Timestamp:       ts.Format(time.RFC3339Nano),
			DiscordUsername: asciiUsername,
			Nickname:        nickname,
			Text:            asciiText,
			AudioFilenames:  []string{},
			RMS:             &rms,
			DurationMs:      &durationMs,
			SpeakerID:       speakerIDPtr,
			SpeakerLabel:    speakerLabelPtr,
		})
	}
}

type mergedLine struct {
	timestamp time.Time
	username  string
	text      string
}

func (w *Worker) finalizeSession(ctx context.Context, sessionID int64) (*string, error) {
	rows, err := db.GetTranscriptionsBySession(ctx, sessionID)
	if err != nil {
		return nil, fmt.Errorf("get transcriptions for finalize: %w", err)
	}

	if len(rows) == 0 {
		if err := db.EndSession(ctx, sessionID); err != nil {
			log.Printf("EndSession error: %v", err)
		}
		return nil, nil
	}

	// Count distinct diarization speakers per discord user: the speaker
	// suffix is only worth showing when a user actually has more than one
	// voice (shared mic) — solo users would just get a noisy "(S1)".
	speakersPerUser := make(map[string]map[int64]bool)
	for _, r := range rows {
		if r.SpeakerID == nil {
			continue
		}
		if speakersPerUser[r.DiscordUserID] == nil {
			speakersPerUser[r.DiscordUserID] = make(map[int64]bool)
		}
		speakersPerUser[r.DiscordUserID][*r.SpeakerID] = true
	}

	// Merge consecutive same-speaker lines. The merge key includes the
	// diarization speaker so two people on a shared mic export as distinct
	// voices ("joe (S2)") instead of collapsing into one paragraph.
	var merged []mergedLine
	for _, r := range rows {
		username := r.DiscordUsername
		if r.Nickname != nil && *r.Nickname != "" {
			username = *r.Nickname
		}
		if len(speakersPerUser[r.DiscordUserID]) > 1 {
			if r.SpeakerName != nil && *r.SpeakerName != "" {
				username = fmt.Sprintf("%s (%s)", username, *r.SpeakerName)
			} else if r.SpeakerLabel != nil && *r.SpeakerLabel != "" {
				username = fmt.Sprintf("%s (%s)", username, *r.SpeakerLabel)
			}
		}
		text := strings.TrimRight(r.SpokenText, ".")
		text = strings.TrimSpace(text)

		if len(merged) > 0 && merged[len(merged)-1].username == username {
			prev := &merged[len(merged)-1]
			prev.text = strings.TrimRight(prev.text, ".") + " " + text
		} else {
			merged = append(merged, mergedLine{
				timestamp: r.CreatedAt,
				username:  username,
				text:      text,
			})
		}
	}

	// Ensure each line ends with punctuation.
	for i := range merged {
		merged[i].text = strings.TrimRight(merged[i].text, ".")
		merged[i].text = strings.TrimSpace(merged[i].text)
		if !endsWithPunctuation(merged[i].text) {
			merged[i].text += "."
		}
	}

	// Build output.
	var sb strings.Builder
	for _, line := range merged {
		sb.WriteString(line.username)
		sb.WriteString(": ")
		sb.WriteString(line.text)
		sb.WriteString("\n")
	}

	// Write file.
	now := time.Now()
	dateStr := now.Format("2006-01-02_15-04-05")
	outFile := filepath.Join(w.recordingDir, fmt.Sprintf("transcription-%s.txt", dateStr))

	if err := os.MkdirAll(w.recordingDir, 0o755); err != nil {
		return nil, fmt.Errorf("create recording dir: %w", err)
	}
	if err := os.WriteFile(outFile, []byte(sb.String()), 0o644); err != nil {
		return nil, fmt.Errorf("write transcription file: %w", err)
	}

	// End session in DB.
	if err := db.EndSession(ctx, sessionID); err != nil {
		log.Printf("EndSession error: %v", err)
	}

	log.Printf("Finalized transcription: %d lines -> %s", len(merged), outFile)
	return &outFile, nil
}

// --- Helpers ---

func rowToLine(r db.TranscriptionRow) web.TranscriptionLine {
	filenames := r.AudioFilenames
	if filenames == nil {
		filenames = []string{}
	}
	return web.TranscriptionLine{
		ID:             r.ID,
		Timestamp:      r.CreatedAt.Format(time.RFC3339Nano),
		DiscordUsername: r.DiscordUsername,
		Nickname:       r.Nickname,
		Text:           r.SpokenText,
		AudioFilenames: filenames,
		RMS:            r.RMS,
		Confidence:     r.Confidence,
		DurationMs:     r.DurationMs,
		SpeakerID:      r.SpeakerID,
		SpeakerLabel:   r.SpeakerLabel,
		SpeakerName:    r.SpeakerName,
	}
}

func endsWithPunctuation(s string) bool {
	if s == "" {
		return false
	}
	last := s[len(s)-1]
	return last == '.' || last == '!' || last == '?'
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
