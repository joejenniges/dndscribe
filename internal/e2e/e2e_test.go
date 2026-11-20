// Package e2e runs the four standard capture WAVs (testdata/) through the
// whole pipeline exactly as live Discord audio flows: WAV -> stereo 20ms
// opus packets -> UserStream (opus decode, buffering, silence flush) ->
// transcription worker (sherpa streaming or whisper batch, punctuation,
// diarization) -> Postgres. Assertions are fuzzy keyword checks, not exact
// text — model accuracy drift (or a swapped model) shouldn't fail the suite,
// but dropped words at utterance heads (the 2026-07-23 bug class) will.
//
// Requirements (each skips the test when missing):
//   - DNDSCRIBE_E2E=1 and running from bin\ next to the sherpa DLLs
//     (System32's stale onnxruntime crashes the process otherwise) — use
//     ./run-e2e.ps1, which sets all of this up.
//   - Postgres reachable (database.url default, or DNDSCRIBE_TEST_DB).
//   - Models downloaded: streaming zipformer en-2023-06-26, punctuation,
//     speaker embedding (download-model.ps1), and recordings/models/
//     ggml-small.bin for the whisper variant.
//
// The test creates its own campaign + session and deletes them afterwards.
package e2e

import (
	"context"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/joe/dndscribe-go/internal/config"
	"github.com/joe/dndscribe-go/internal/db"
	"github.com/joe/dndscribe-go/internal/transcribe"
	"github.com/joe/dndscribe-go/internal/voice"

	"gopkg.in/hraban/opus.v2"
)

// wavCase is one standard capture and the phrases its transcription must
// contain (matched on punctuation-stripped lowercase text). Phrases are
// chosen to hold for BOTH engines' known-good output; proper nouns like
// "catio" are deliberately not asserted.
type wavCase struct {
	file    string
	phrases []string
}

var wavCases = []wavCase{
	{"catio.wav", []string{"gave them", "earlier", "want it"}},
	{"closed-noon.wav", []string{"around noon", "warm"}},
	{"say-about.wav", []string{"say about that"}},
	{"supporting.wav", []string{"supporting you"}},
}

func TestPipelineSherpa(t *testing.T) {
	runPipeline(t, "sherpa")
}

func TestPipelineWhisper(t *testing.T) {
	if testing.Short() {
		t.Skip("whisper variant skipped in -short mode")
	}
	runPipeline(t, "whisper")
}

func runPipeline(t *testing.T, engine string) {
	if os.Getenv("DNDSCRIBE_E2E") != "1" {
		t.Skip("set DNDSCRIBE_E2E=1 (and run from bin\\ for DLLs) — use ./run-e2e.ps1")
	}
	root := repoRoot(t)

	// --- Engine + diarization config (absolute paths; CWD is bin\) ---
	tcfg := &config.TranscribeConfig{
		Engine:  engine,
		Model:   "small",
		Threads: 4,
		Sherpa: config.SherpaConfig{
			ModelDir:   filepath.Join(root, "models", "sherpa-onnx-streaming-zipformer-en-2023-06-26"),
			NumThreads: 4,
			Punctuation: config.PunctuationConfig{
				ModelDir: filepath.Join(root, "models", "sherpa-onnx-online-punct-en-2024-08-06"),
			},
		},
		Diarization: config.DiarizationConfig{
			Enabled: true,
			Model:   filepath.Join(root, "models", "wespeaker_en_voxceleb_CAM++.onnx"),
		},
	}
	config.ApplySherpaDefaults(&tcfg.Sherpa)
	config.ApplyDiarizationDefaults(&tcfg.Diarization)

	requireFile(t, filepath.Join(tcfg.Sherpa.ModelDir, "tokens.txt"), "ASR model (download-model.ps1)")
	requireFile(t, tcfg.Diarization.Model, "speaker model (download-model.ps1 -Speaker)")
	if engine == "whisper" {
		requireFile(t, filepath.Join(root, "recordings", "models", "ggml-small.bin"), "whisper small model")
	}

	// --- DB sandbox: own campaign + session, deleted afterwards ---
	initDB(t, root)
	ctx := context.Background()
	camp, err := db.CreateCampaign(ctx, fmt.Sprintf("e2e-%s-%d", engine, time.Now().Unix()), nil)
	if err != nil {
		t.Fatalf("create campaign: %v", err)
	}
	t.Cleanup(func() { cleanupCampaign(t, camp.ID) })
	sessionID, err := db.CreateSession(ctx, camp.ID, "e2e-test", "e2e")
	if err != nil {
		t.Fatalf("create session: %v", err)
	}

	// --- Worker wired exactly like the bot ---
	worker := transcribe.NewWorker(filepath.Join(root, "recordings"))
	if err := worker.Init(tcfg); err != nil {
		t.Fatalf("worker init: %v", err)
	}
	worker.Start()
	t.Cleanup(worker.Stop)
	worker.SetCampaignID(&camp.ID)
	worker.SetSessionID(&sessionID)

	// --- Stream each WAV through the voice layer as opus packets ---
	for _, c := range wavCases {
		streamWavAsDiscordPackets(t, worker, engine, filepath.Join(root, "internal", "e2e", "testdata", c.file))
	}

	// --- Collect rows (whisper transcribes async via the task queue) ---
	rows := waitForRows(t, ctx, sessionID, len(wavCases), 180*time.Second)
	var joined strings.Builder
	for _, r := range rows {
		t.Logf("line: [%s spk=%v] %s", r.DiscordUsername, ptrOr(r.SpeakerLabel, "-"), r.SpokenText)
		joined.WriteString(r.SpokenText)
		joined.WriteString(" ")
	}
	transcript := normalizeText(joined.String())

	// --- Fuzzy content assertions ---
	for _, c := range wavCases {
		for _, phrase := range c.phrases {
			if !strings.Contains(transcript, phrase) {
				t.Errorf("%s: transcript missing %q\nfull: %s", c.file, phrase, transcript)
			}
		}
	}

	// --- Diarization assertions: every line got a speaker, and one voice
	// across four clips must not fragment into more than two profiles ---
	speakerIDs := map[int64]bool{}
	for _, r := range rows {
		if r.DurationMs != nil && *r.DurationMs >= 1500 && r.SpeakerID == nil {
			t.Errorf("line %d (%.1fs) has no speaker assigned: %q", r.ID, float64(*r.DurationMs)/1000, r.SpokenText)
		}
		if r.SpeakerID != nil {
			speakerIDs[*r.SpeakerID] = true
		}
	}
	if len(speakerIDs) == 0 {
		t.Error("diarization assigned no speakers at all")
	}
	if len(speakerIDs) > 2 {
		t.Errorf("one voice fragmented into %d speaker profiles (threshold too high?)", len(speakerIDs))
	}
	t.Logf("distinct speakers: %d", len(speakerIDs))
}

// streamWavAsDiscordPackets encodes a 48kHz mono WAV into stereo 20ms opus
// packets and pushes them through a fresh UserStream, mirroring the live
// wiring (onFrame -> FeedFrame in sherpa mode; onFlush -> AddTask). Close()
// drains the packet queue and fires the flush, i.e. the utterance boundary.
func streamWavAsDiscordPackets(t *testing.T, worker *transcribe.Worker, engine, wavPath string) {
	t.Helper()

	mono, sampleRate, err := readWavMonoInt16(wavPath)
	if err != nil {
		t.Fatalf("read %s: %v", wavPath, err)
	}
	if sampleRate != 48000 {
		t.Fatalf("%s: expected 48kHz, got %d", wavPath, sampleRate)
	}

	onFlush := func(userID string, pcm []byte, ts time.Time, username, nickname string, rms float32, durationMs int, audioFilename string) {
		worker.AddTask(userID, pcm, ts, username, nickname, rms, durationMs, audioFilename)
	}
	var onFrame voice.FrameCallback
	if engine == "sherpa" {
		onFrame = func(userID, username, nickname string, mono48k []int16) {
			worker.FeedFrame(userID, username, nickname, mono48k)
		}
	}

	us, err := voice.NewUserStream("e2e-user", nil, onFlush, onFrame)
	if err != nil {
		t.Fatalf("new user stream: %v", err)
	}
	us.SetUsername("e2etester")
	us.EnableDirectOpusForTest()

	enc, err := opus.NewEncoder(48000, 2, opus.AppVoIP)
	if err != nil {
		t.Fatalf("opus encoder: %v", err)
	}

	const frame = 960 // 20ms at 48kHz per channel, Discord's packet cadence
	pktBuf := make([]byte, 4000)
	stereo := make([]int16, frame*2)
	var seq uint16
	var ts uint32
	for off := 0; off < len(mono); off += frame {
		// Duplicate mono into interleaved stereo, zero-padding the tail frame
		// (opus requires whole 20ms frames).
		for i := 0; i < frame; i++ {
			var s int16
			if off+i < len(mono) {
				s = mono[off+i]
			}
			stereo[i*2] = s
			stereo[i*2+1] = s
		}
		n, err := enc.Encode(stereo, pktBuf)
		if err != nil {
			t.Fatalf("opus encode: %v", err)
		}
		pkt := append([]byte(nil), pktBuf[:n]...)
		// SendPacket is non-blocking and drops when the decode queue is full;
		// we feed much faster than real time, so retry instead of losing audio.
		for !us.SendPacket(pkt, ts, seq) {
			time.Sleep(time.Millisecond)
		}
		seq++
		ts += frame
	}

	// Close drains the queue, then flushes the buffer — the same utterance
	// boundary the 1.5s silence timeout produces live.
	us.Close()
}

// waitForRows polls the session's transcriptions until at least want rows
// exist and the count has stopped growing (sherpa can split a clip into
// multiple lines; whisper inserts asynchronously from the task queue).
func waitForRows(t *testing.T, ctx context.Context, sessionID int64, want int, timeout time.Duration) []db.TranscriptionRow {
	t.Helper()
	deadline := time.Now().Add(timeout)
	prev := -1
	for time.Now().Before(deadline) {
		rows, err := db.GetTranscriptionsBySession(ctx, sessionID)
		if err != nil {
			t.Fatalf("get transcriptions: %v", err)
		}
		if len(rows) >= want && len(rows) == prev {
			return rows
		}
		prev = len(rows)
		time.Sleep(2 * time.Second)
	}
	rows, _ := db.GetTranscriptionsBySession(ctx, sessionID)
	if len(rows) < want {
		t.Fatalf("timed out: %d/%d transcription rows after %s", len(rows), want, timeout)
	}
	return rows
}

func initDB(t *testing.T, root string) {
	t.Helper()
	url := os.Getenv("DNDSCRIBE_TEST_DB")
	if url == "" {
		cfg, err := config.Load(filepath.Join(root, "config.yaml"))
		if err != nil || cfg.Database.URL == "" {
			url = "postgresql://dev:dev@localhost:5432/dndscribe"
		} else {
			url = cfg.Database.URL
		}
	}
	if err := db.Init(url); err != nil {
		t.Skipf("postgres not reachable (%v) — start shared-postgres or set DNDSCRIBE_TEST_DB", err)
	}
}

func cleanupCampaign(t *testing.T, campaignID int64) {
	ctx := context.Background()
	sessions, err := db.ListSessions(ctx, campaignID)
	if err == nil {
		for _, s := range sessions {
			if _, err := db.DeleteSessionFull(ctx, s.ID); err != nil {
				t.Logf("cleanup session %d: %v", s.ID, err)
			}
		}
	}
	speakers, err := db.GetSpeakersByCampaign(ctx, campaignID)
	if err == nil {
		for _, sp := range speakers {
			if _, err := db.DeleteSpeaker(ctx, sp.ID); err != nil {
				t.Logf("cleanup speaker %d: %v", sp.ID, err)
			}
		}
	}
	if err := db.DeleteCampaign(ctx, campaignID); err != nil {
		t.Logf("cleanup campaign %d: %v", campaignID, err)
	}
}

// --- Helpers ---

func repoRoot(t *testing.T) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	return filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
}

func requireFile(t *testing.T, path, hint string) {
	t.Helper()
	if _, err := os.Stat(path); err != nil {
		t.Skipf("%s missing at %s", hint, path)
	}
}

var nonWord = regexp.MustCompile(`[^a-z0-9' ]+`)
var multiSpace = regexp.MustCompile(` +`)

// normalizeText lowercases and strips punctuation so phrase assertions hold
// across engines regardless of casing/punctuation differences.
func normalizeText(s string) string {
	s = strings.ToLower(s)
	s = nonWord.ReplaceAllString(s, " ")
	return strings.TrimSpace(multiSpace.ReplaceAllString(s, " "))
}

func ptrOr(p *string, def string) string {
	if p == nil {
		return def
	}
	return *p
}

// readWavMonoPCM parses a 16-bit mono WAV into samples + sample rate.
func readWavMonoInt16(path string) ([]int16, int, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, 0, err
	}
	if len(data) < 44 || string(data[0:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return nil, 0, fmt.Errorf("%q is not a RIFF/WAVE file", path)
	}
	var (
		channels   int
		sampleRate int
		bits       int
		pcm        []byte
	)
	for off := 12; off+8 <= len(data); {
		id := string(data[off : off+4])
		size := int(binary.LittleEndian.Uint32(data[off+4 : off+8]))
		body := off + 8
		if body+size > len(data) {
			size = len(data) - body
		}
		switch id {
		case "fmt ":
			if size >= 16 {
				channels = int(binary.LittleEndian.Uint16(data[body+2 : body+4]))
				sampleRate = int(binary.LittleEndian.Uint32(data[body+4 : body+8]))
				bits = int(binary.LittleEndian.Uint16(data[body+14 : body+16]))
			}
		case "data":
			pcm = data[body : body+size]
		}
		off = body + size
		if size%2 == 1 {
			off++
		}
	}
	if bits != 16 || channels != 1 {
		return nil, 0, fmt.Errorf("%q: expected 16-bit mono PCM, got %d-bit %dch", path, bits, channels)
	}
	out := make([]int16, len(pcm)/2)
	for i := range out {
		out[i] = int16(binary.LittleEndian.Uint16(pcm[i*2:]))
	}
	return out, sampleRate, nil
}
