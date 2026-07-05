package transcribe

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/joe/dndscribe-go/internal/config"
	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

// Punctuator adds punctuation and truecasing to raw ASR text using sherpa-onnx's
// online punctuation model (Edge-Punct-Casing CNN-BiLSTM). The streaming
// Zipformer emits ALL-CAPS, unpunctuated tokens; this turns
// "the yellow lamps would light up" into "The yellow lamps would light up." so
// streaming lines read like the whisper engine's output.
//
// AddPunct is a single text->text call that does both punctuation and casing.
type Punctuator struct {
	mu   sync.Mutex // AddPunct treated as non-reentrant; calls are cheap
	impl *sherpa.OnlinePunctuation
}

// NewPunctuator loads the punctuation model described by cfg. Returns nil (no
// error) when punctuation is disabled, so callers can treat a nil *Punctuator as
// "pass text through unchanged".
func NewPunctuator(cfg *config.PunctuationConfig) (*Punctuator, error) {
	if !cfg.IsEnabled() {
		return nil, nil
	}
	model, bpe, err := resolvePunctFiles(cfg)
	if err != nil {
		return nil, err
	}

	c := sherpa.OnlinePunctuationConfig{
		Model: sherpa.OnlinePunctuationModelConfig{
			CnnBilstm:  model,
			BpeVocab:   bpe,
			NumThreads: 1,
			Provider:   "cpu",
			Debug:      0,
		},
	}
	impl := sherpa.NewOnlinePunctuation(&c)
	if impl == nil {
		return nil, fmt.Errorf("sherpa NewOnlinePunctuation returned nil (model=%s bpe=%s): check files and native libs", model, bpe)
	}
	return &Punctuator{impl: impl}, nil
}

// Punctuate returns text with punctuation and casing restored. A nil Punctuator
// or empty input returns the input unchanged, so call sites never need a guard.
func (p *Punctuator) Punctuate(text string) string {
	if p == nil || text == "" {
		return text
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.impl.AddPunct(text)
}

// Close frees the native model. Safe on a nil Punctuator.
func (p *Punctuator) Close() error {
	if p == nil || p.impl == nil {
		return nil
	}
	sherpa.DeleteOnlinePunctuation(p.impl)
	p.impl = nil
	return nil
}

// resolvePunctFiles returns the cnn-bilstm model and bpe-vocab paths. Explicit
// cfg paths win; otherwise they're discovered in cfg.ModelDir (prefer int8).
func resolvePunctFiles(cfg *config.PunctuationConfig) (model, bpe string, err error) {
	if cfg.CnnBilstm != "" && cfg.BpeVocab != "" {
		return cfg.CnnBilstm, cfg.BpeVocab, nil
	}
	if cfg.ModelDir == "" {
		return "", "", fmt.Errorf("punctuation: set transcribe.sherpa.punctuation.model_dir or explicit cnn_bilstm/bpe_vocab")
	}
	int8 := filepath.Join(cfg.ModelDir, "model.int8.onnx")
	fp32 := filepath.Join(cfg.ModelDir, "model.onnx")
	switch {
	case fileExists(int8):
		model = int8
	case fileExists(fp32):
		model = fp32
	default:
		return "", "", fmt.Errorf("punctuation: no model.onnx/model.int8.onnx in %q", cfg.ModelDir)
	}
	bpe = filepath.Join(cfg.ModelDir, "bpe.vocab")
	if !fileExists(bpe) {
		return "", "", fmt.Errorf("punctuation: bpe.vocab not found in %q", cfg.ModelDir)
	}
	return model, bpe, nil
}

func fileExists(p string) bool {
	info, err := os.Stat(p)
	return err == nil && !info.IsDir()
}
