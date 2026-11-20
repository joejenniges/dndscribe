package transcribe

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/joe/dndscribe-go/internal/config"
)

// Init constructs the transcription engine selected by cfg.Engine and prepares
// the worker. Both engines are linked into every build; the choice is made here
// at runtime. Requires CGo (whisper.cpp + sherpa-onnx/onnxruntime native libs).
func (w *Worker) Init(cfg *config.TranscribeConfig) error {
	// Diarization is engine-independent: whisper identifies per flushed chunk,
	// sherpa per streaming segment via the engine's identify callback.
	diarizer, err := NewDiarizer(&cfg.Diarization)
	if err != nil {
		return fmt.Errorf("init diarization: %w", err)
	}
	w.SetDiarizer(diarizer)

	switch strings.ToLower(cfg.Engine) {
	case "sherpa":
		punct, err := NewPunctuator(&cfg.Sherpa.Punctuation)
		if err != nil {
			return fmt.Errorf("init punctuation: %w", err)
		}
		engine, err := NewSherpaEngine(&cfg.Sherpa, punct)
		if err != nil {
			punct.Close()
			return fmt.Errorf("init sherpa engine: %w", err)
		}
		if diarizer != nil {
			engine.SetIdentify(w.identifySpeaker)
		}
		w.SetStreamEngine(engine)
		log.Printf("Transcription engine: sherpa (streaming, punctuation=%v, diarization=%v)",
			punct != nil, diarizer != nil)
		return nil
	case "", "whisper":
		modelDir := filepath.Join(w.recordingDir, "models")
		if err := os.MkdirAll(modelDir, 0o755); err != nil {
			return fmt.Errorf("create model dir: %w", err)
		}
		engine, err := NewWhisperEngine(cfg.Model, modelDir, cfg.Threads)
		if err != nil {
			return fmt.Errorf("init whisper engine: %w", err)
		}
		w.engine = engine
		log.Printf("Transcription engine: whisper (batch)")
		return nil
	default:
		return fmt.Errorf("unknown transcribe.engine %q (want whisper or sherpa)", cfg.Engine)
	}
}
