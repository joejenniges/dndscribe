package transcribe

import (
	"time"

	"github.com/joe/dndscribe-go/internal/web"
)

// TranscriptionLine is an alias for the shared type in the web package.
type TranscriptionLine = web.TranscriptionLine

// Engine is the interface for batch speech-to-text backends.
// WhisperEngine in whisper.go satisfies this. The interface allows for
// testing with mock engines or alternative backends.
//
// Batch engines receive a complete utterance (one silence-delimited chunk) and
// return final text. This is whisper's natural shape.
type Engine interface {
	Transcribe(samples []float32, prompt string) (text string, confidence float32, err error)
	Close() error
}

// StreamingResult is one emission from a StreamingEngine: either a provisional
// partial hypothesis (IsFinal=false) that supersedes the previous partial for
// the same user, or a committed final (IsFinal=true) at an utterance boundary.
type StreamingResult struct {
	UserID   string
	Username string
	Nickname string
	Text     string
	IsFinal  bool

	// Timestamp is the wall-clock start of the utterance. Set on finals.
	Timestamp time.Time
	// RMS and DurationMs describe the utterance audio; set on finals (computed
	// by the engine from the audio it was fed). Zero on partials.
	RMS        float32
	DurationMs int
}

// StreamingEngine is the interface for streaming-native speech-to-text backends
// (sherpa-onnx). Audio is fed continuously per user; results arrive via the
// callback set with SetCallback as partials update and finals commit.
//
// WHY this is separate from Engine: whisper is sequence-to-sequence and only
// produces output once it has a whole utterance, so it cannot emit partials.
// The two shapes don't unify cleanly, so the Worker holds one or the other and
// routes audio accordingly (see worker.go mode handling).
type StreamingEngine interface {
	// SetCallback registers where results are delivered. Called once at wiring
	// time before any Feed.
	SetCallback(fn func(StreamingResult))
	// Feed supplies mono s16le 48kHz samples for a user's stream. Identity is
	// passed through so results can be attributed without the engine knowing
	// anything Discord-specific. The engine downmix/resample/decode internally.
	Feed(userID, username, nickname string, mono48k []int16)
	// Flush signals the user stopped speaking (silence flush) so the engine
	// finalizes and resets that user's stream.
	Flush(userID string)
	Close() error
}
