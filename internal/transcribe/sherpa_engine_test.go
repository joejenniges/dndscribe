package transcribe

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/joe/dndscribe-go/internal/config"
)

// defaultTestModelDir is the model download-model.ps1 fetches by default.
const defaultTestModelDir = "../../models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"

// TestSherpaEngineEndToEnd exercises the full engine wrapper — resampling,
// per-user stream, decode loop, and finalize/callback — with real inference.
// It feeds the model's own 16k test wav, upsampled to the 48k mono int16 the
// engine expects (mirroring the Discord input format), then Flushes and asserts
// a non-empty final came back through the callback.
//
// Skips when the model isn't downloaded so it doesn't block CI. Requires the
// sherpa DLLs on PATH (see build.md); without them the package won't load.
func TestSherpaEngineEndToEnd(t *testing.T) {
	modelDir := defaultTestModelDir
	if env := os.Getenv("SHERPA_MODEL_DIR"); env != "" {
		modelDir = env
	}
	if _, err := os.Stat(filepath.Join(modelDir, "tokens.txt")); err != nil {
		t.Skipf("model not present at %s (run download-model.ps1); skipping", modelDir)
	}

	// Mirror config.Load's defaults (applySherpaDefaults is unexported).
	cfg := &config.SherpaConfig{
		ModelDir:                modelDir,
		Variant:                 "int8",
		Provider:                "cpu",
		NumThreads:              1,
		DecodingMethod:          "greedy_search",
		MaxActivePaths:          4,
		Rule1MinTrailingSilence: 2.4,
		Rule2MinTrailingSilence: 1.2,
		Rule3MinUtteranceLength: 20,
	}

	eng, err := NewSherpaEngine(cfg, nil)
	if err != nil {
		t.Fatalf("NewSherpaEngine: %v", err)
	}
	defer eng.Close()

	var (
		mu       sync.Mutex
		partials int
		finals   []string
	)
	eng.SetCallback(func(r StreamingResult) {
		mu.Lock()
		defer mu.Unlock()
		if r.IsFinal {
			finals = append(finals, r.Text)
		} else {
			partials++
		}
	})

	in48k := upsample3x(readTestWav16k(t, filepath.Join(modelDir, "test_wavs", "0.wav")))

	// Feed in ~20ms frames (960 samples @48k) to mimic Discord packet cadence.
	const frame = 960
	for off := 0; off < len(in48k); off += frame {
		end := off + frame
		if end > len(in48k) {
			end = len(in48k)
		}
		eng.Feed("user1", "alice", "Rogue", in48k[off:end])
	}
	eng.Flush("user1")

	mu.Lock()
	defer mu.Unlock()
	if len(finals) == 0 {
		t.Fatalf("no final emitted (partials=%d)", partials)
	}
	got := finals[len(finals)-1]
	if got == "" {
		t.Fatalf("final text empty (partials=%d, finals=%d)", partials, len(finals))
	}
	t.Logf("partials=%d finals=%d last=%q", partials, len(finals), got)
}

// TestSherpaEnginePunctuatedFinal runs the engine WITH a punctuator and asserts
// the committed final is truecased + punctuated (not the raw ALL-CAPS). Skips if
// either model is missing.
func TestSherpaEnginePunctuatedFinal(t *testing.T) {
	asrDir := defaultTestModelDir
	if env := os.Getenv("SHERPA_MODEL_DIR"); env != "" {
		asrDir = env
	}
	punctDir := defaultPunctModelDir
	if env := os.Getenv("SHERPA_PUNCT_DIR"); env != "" {
		punctDir = env
	}
	if _, err := os.Stat(filepath.Join(asrDir, "tokens.txt")); err != nil {
		t.Skipf("ASR model missing at %s; skipping", asrDir)
	}
	if _, err := os.Stat(filepath.Join(punctDir, "bpe.vocab")); err != nil {
		t.Skipf("punct model missing at %s; skipping", punctDir)
	}

	punct, err := NewPunctuator(&config.PunctuationConfig{ModelDir: punctDir})
	if err != nil {
		t.Fatalf("NewPunctuator: %v", err)
	}
	cfg := &config.SherpaConfig{
		ModelDir: asrDir, Variant: "int8", Provider: "cpu", NumThreads: 1,
		DecodingMethod: "greedy_search", MaxActivePaths: 4,
		Rule1MinTrailingSilence: 2.4, Rule2MinTrailingSilence: 1.2, Rule3MinUtteranceLength: 30,
		SoftCapSeconds: 12,
	}
	eng, err := NewSherpaEngine(cfg, punct)
	if err != nil {
		t.Fatalf("NewSherpaEngine: %v", err)
	}
	defer eng.Close()

	var (
		mu     sync.Mutex
		finals []string
	)
	eng.SetCallback(func(r StreamingResult) {
		if r.IsFinal {
			mu.Lock()
			finals = append(finals, r.Text)
			mu.Unlock()
		}
	})

	in48k := upsample3x(readTestWav16k(t, filepath.Join(asrDir, "test_wavs", "0.wav")))
	const frame = 960
	for off := 0; off < len(in48k); off += frame {
		end := off + frame
		if end > len(in48k) {
			end = len(in48k)
		}
		eng.Feed("u", "alice", "Rogue", in48k[off:end])
	}
	eng.Flush("u")

	mu.Lock()
	defer mu.Unlock()
	if len(finals) == 0 {
		t.Fatal("no final emitted")
	}
	got := finals[len(finals)-1]
	t.Logf("punctuated final: %q", got)
	// Assert the punctuator ran end-to-end: raw ASR is ALL-CAPS, so a truecased
	// final contains lowercase letters. (Terminal punctuation isn't asserted —
	// this test wav is a single sentence fragment with no natural sentence end;
	// punctuation insertion is covered by TestPunctuatorRealModel.)
	if got == strings.ToUpper(got) {
		t.Errorf("final still ALL-CAPS (punctuator did not run): %q", got)
	}
}

// readTestWav16k reads a 16-bit mono WAV's samples as int16 (assumes 16k mono,
// which the sherpa test_wavs are). Minimal parser — fixed 44-byte header.
func readTestWav16k(t *testing.T, path string) []int16 {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read wav: %v", err)
	}
	if len(data) < 44 {
		t.Fatalf("wav too short: %d bytes", len(data))
	}
	pcm := data[44:]
	out := make([]int16, len(pcm)/2)
	for i := range out {
		out[i] = int16(binary.LittleEndian.Uint16(pcm[i*2:]))
	}
	return out
}

// upsample3x converts 16k -> 48k via linear interpolation. Good enough to feed
// the engine (which downsamples back to 16k); not production audio quality.
func upsample3x(in []int16) []int16 {
	if len(in) == 0 {
		return nil
	}
	out := make([]int16, len(in)*3)
	for i := 0; i < len(in); i++ {
		a := float64(in[i])
		b := a
		if i+1 < len(in) {
			b = float64(in[i+1])
		}
		out[i*3] = int16(a)
		out[i*3+1] = int16(a + (b-a)/3)
		out[i*3+2] = int16(a + 2*(b-a)/3)
	}
	return out
}
