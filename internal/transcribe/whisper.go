package transcribe

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"

	whisper "github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
)

const modelURLBase = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

// WhisperEngine wraps the whisper.cpp Go bindings for speech-to-text.
type WhisperEngine struct {
	model   whisper.Model
	threads int
	mu      sync.Mutex
}

// NewWhisperEngine loads (or downloads) a whisper model and returns an engine
// ready for transcription. modelName is e.g. "base", "small", "medium".
// modelDir is the directory where model files are stored/cached.
func NewWhisperEngine(modelName string, modelDir string, threads int) (*WhisperEngine, error) {
	modelPath := filepath.Join(modelDir, fmt.Sprintf("ggml-%s.bin", modelName))

	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		if err := downloadModel(modelName, modelPath); err != nil {
			return nil, fmt.Errorf("download model %s: %w", modelName, err)
		}
	}

	model, err := whisper.New(modelPath)
	if err != nil {
		return nil, fmt.Errorf("load whisper model %s: %w", modelPath, err)
	}

	log.Printf("Loaded whisper model %q (%d threads)", modelName, threads)
	return &WhisperEngine{
		model:   model,
		threads: threads,
	}, nil
}

// Transcribe processes float32 samples (16kHz mono) and returns the
// transcribed text and average confidence score.
// prompt is the initial prompt for vocabulary biasing and context.
func (w *WhisperEngine) Transcribe(samples []float32, prompt string) (text string, confidence float32, err error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	ctx, err := w.model.NewContext()
	if err != nil {
		return "", 0, fmt.Errorf("create whisper context: %w", err)
	}

	if err := ctx.SetLanguage("en"); err != nil {
		return "", 0, fmt.Errorf("set language: %w", err)
	}
	ctx.SetThreads(uint(w.threads))
	if prompt != "" {
		ctx.SetInitialPrompt(prompt)
	}

	if err := ctx.Process(samples, nil, nil, nil); err != nil {
		return "", 0, fmt.Errorf("whisper process: %w", err)
	}

	var segments []string
	for {
		seg, err := ctx.NextSegment()
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", 0, fmt.Errorf("read segment: %w", err)
		}
		t := strings.TrimSpace(seg.Text)
		if t == "" {
			continue
		}
		segments = append(segments, t)
	}

	if len(segments) == 0 {
		return "", 0, nil
	}

	fullText := strings.Join(segments, " ")

	// Confidence heuristic based on word count.
	// WHY: The whisper.cpp Go bindings don't expose per-token log probabilities.
	// Longer utterances are more likely to be real speech than hallucinations.
	// Short utterances from sensitive mics are filtered by the RMS heuristic
	// in filter.go regardless of this confidence value.
	wordCount := len(strings.Fields(fullText))
	var conf float32
	switch {
	case wordCount >= 10:
		conf = 0.85
	case wordCount >= 5:
		conf = 0.70
	case wordCount >= 3:
		conf = 0.55
	default:
		conf = 0.40
	}

	return fullText, conf, nil
}

// Close releases the whisper model resources.
func (w *WhisperEngine) Close() error {
	return w.model.Close()
}

func downloadModel(name, destPath string) error {
	url := fmt.Sprintf("%s/ggml-%s.bin", modelURLBase, name)
	log.Printf("Downloading whisper model %q from %s ...", name, url)

	if err := os.MkdirAll(filepath.Dir(destPath), 0o755); err != nil {
		return err
	}

	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("download: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download returned HTTP %d", resp.StatusCode)
	}

	tmpPath := destPath + ".tmp"
	f, err := os.Create(tmpPath)
	if err != nil {
		return err
	}

	n, err := io.Copy(f, resp.Body)
	f.Close()
	if err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("write model: %w", err)
	}

	if err := os.Rename(tmpPath, destPath); err != nil {
		os.Remove(tmpPath)
		return err
	}

	log.Printf("Downloaded whisper model %q (%d MB)", name, n/1024/1024)
	return nil
}
