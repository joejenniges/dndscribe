package transcribe

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/joe/dndscribe-go/internal/config"
)

const defaultPunctModelDir = "../../models/sherpa-onnx-online-punct-en-2024-08-06"

func TestUncommittedTail(t *testing.T) {
	cases := []struct {
		raw       string
		committed int
		want      string
	}{
		{"a b c d", 2, "c d"},
		{"a b c d", 0, "a b c d"},
		{"a b c d", 4, ""},
		{"a b c d", 9, ""},  // shrank past end -> empty
		{"a b c d", -1, "a b c d"}, // negative clamps to 0
		{"   ", 0, ""},
	}
	for _, c := range cases {
		if got := uncommittedTail(c.raw, c.committed); got != c.want {
			t.Errorf("uncommittedTail(%q,%d)=%q want %q", c.raw, c.committed, got, c.want)
		}
	}
}

func TestSplitCommittable(t *testing.T) {
	cases := []struct {
		in        string
		want      string
		wantOK    bool
	}{
		{"How are you? I am fine", "How are you?", true},
		{"One. Two. Three", "One. Two.", true},
		{"How are you?", "", false},            // ends at boundary, nothing pending after
		{"hello world", "", false},             // no sentence end yet
		{"Wait... what is that", "Wait...", true}, // last boundary-with-text-after
		{"", "", false},
	}
	for _, c := range cases {
		got, ok := splitCommittable(c.in)
		if ok != c.wantOK || got != c.want {
			t.Errorf("splitCommittable(%q)=(%q,%v) want (%q,%v)", c.in, got, ok, c.want, c.wantOK)
		}
	}
}

// TestPunctuatorRealModel validates punctuation + truecasing with the real
// model. Skips when the model isn't downloaded. Requires sherpa DLLs (build the
// test binary into bin/ — see build.md).
func TestPunctuatorRealModel(t *testing.T) {
	modelDir := defaultPunctModelDir
	if env := os.Getenv("SHERPA_PUNCT_DIR"); env != "" {
		modelDir = env
	}
	if _, err := os.Stat(filepath.Join(modelDir, "bpe.vocab")); err != nil {
		t.Skipf("punct model not present at %s (run download-model.ps1); skipping", modelDir)
	}

	p, err := NewPunctuator(&config.PunctuationConfig{ModelDir: modelDir})
	if err != nil {
		t.Fatalf("NewPunctuator: %v", err)
	}
	defer p.Close()

	got := p.Punctuate("how are you i am fine thank you")
	t.Logf("punctuated: %q", got)

	// Truecasing: first letter should be uppercase.
	if got == "" || got[0] < 'A' || got[0] > 'Z' {
		t.Errorf("expected truecased output starting uppercase, got %q", got)
	}
	// Punctuation: should contain at least one sentence-ending mark.
	if !strings.ContainsAny(got, ".?!") {
		t.Errorf("expected sentence punctuation, got %q", got)
	}
}

// TestPunctuatorNil: a nil Punctuator passes text through unchanged.
func TestPunctuatorNil(t *testing.T) {
	var p *Punctuator
	if got := p.Punctuate("hello"); got != "hello" {
		t.Errorf("nil Punctuate = %q want %q", got, "hello")
	}
	if err := p.Close(); err != nil {
		t.Errorf("nil Close err: %v", err)
	}
}
