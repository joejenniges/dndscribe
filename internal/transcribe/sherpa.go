package transcribe

import (
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/joe/dndscribe-go/internal/config"
	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

// Compile-time interface check.
var _ StreamingEngine = (*SherpaEngine)(nil)

// recognizerSampleRate is what the streaming Zipformer models expect.
const recognizerSampleRate = 16000

// SherpaEngine is a streaming speech-to-text engine backed by sherpa-onnx's
// online recognizer. One recognizer is shared across all users; each user gets
// a lazily-created OnlineStream fed continuously, producing partial hypotheses
// that update live and finals on utterance boundaries.
type SherpaEngine struct {
	recognizer *sherpa.OnlineRecognizer
	endpointOn bool
	callback   func(StreamingResult)

	// punct adds punctuation + truecasing to committed text. Nil = pass raw
	// (caps) through. softCap is the per-turn age past which we start committing
	// completed sentences as their own lines (sentence-aware long-turn break).
	punct   *Punctuator
	softCap time.Duration

	// recMu serializes calls into the shared recognizer. sherpa/onnxruntime can
	// decode concurrent streams, but serializing keeps v1 unambiguously correct
	// for the handful of simultaneous speakers a D&D session has; the locked
	// section (one chunk decode) is short. Can be relaxed to per-stream later.
	recMu sync.Mutex

	mu      sync.Mutex // guards streams map
	streams map[string]*sherpaStream
}

// sherpaStream is one user's streaming state. Its own mutex serializes the
// per-user Feed goroutine (voice decodeLoop) against Flush (recorder silence
// loop), which run on different goroutines.
//
// Line model: the recognizer accumulates one rolling hypothesis per "turn"
// (until a full reset on final). committedWords tracks how many words of that
// turn have already been emitted as lines via mid-turn sentence breaks, so the
// recognizer keeps full context (better accuracy) while we slice its output.
type sherpaStream struct {
	mu        sync.Mutex
	stream    *sherpa.OnlineStream
	resampler *StreamResampler

	username   string
	nickname   string
	lastPartial string // dedupe of emitted partial text

	turnStart         time.Time // start of current turn (since last full reset)
	committedWords    int       // words of the turn already emitted as lines
	lastSentenceCheck time.Time // throttle for the long-turn punctuation check

	// Per-segment audio accounting (a segment = audio since the last committed
	// line) for that line's RMS + duration. Reset on every commit, mid-turn or
	// final.
	segStart  time.Time
	segSamples int
	segSumSq   float64
}

// NewSherpaEngine resolves model files from cfg, loads the online recognizer,
// and returns a ready engine. punct may be nil (raw caps output). Call
// SetCallback before feeding audio.
func NewSherpaEngine(cfg *config.SherpaConfig, punct *Punctuator) (*SherpaEngine, error) {
	enc, dec, join, tokens, err := resolveModelFiles(cfg)
	if err != nil {
		return nil, err
	}

	rc := sherpa.OnlineRecognizerConfig{
		FeatConfig: sherpa.FeatureConfig{SampleRate: recognizerSampleRate, FeatureDim: 80},
		ModelConfig: sherpa.OnlineModelConfig{
			Transducer: sherpa.OnlineTransducerModelConfig{Encoder: enc, Decoder: dec, Joiner: join},
			Tokens:     tokens,
			NumThreads: cfg.NumThreads,
			Provider:   cfg.Provider,
			// ModelType is normally empty so sherpa auto-detects from the encoder
			// onnx metadata. Hardcoding "zipformer2" hard-aborts on first-gen
			// zipformer models that lack the query_head_dims field.
			ModelType: cfg.ModelType,
			Debug:     0,
		},
		DecodingMethod: cfg.DecodingMethod,
		MaxActivePaths: cfg.MaxActivePaths,
	}
	if cfg.EndpointEnabled() {
		rc.EnableEndpoint = 1
		rc.Rule1MinTrailingSilence = cfg.Rule1MinTrailingSilence
		rc.Rule2MinTrailingSilence = cfg.Rule2MinTrailingSilence
		rc.Rule3MinUtteranceLength = cfg.Rule3MinUtteranceLength
	}

	recognizer := sherpa.NewOnlineRecognizer(&rc)
	if recognizer == nil {
		return nil, fmt.Errorf("sherpa NewOnlineRecognizer returned nil (encoder=%s tokens=%s): check model files and native libs", enc, tokens)
	}

	softCap := time.Duration(cfg.SoftCapSeconds * float64(time.Second))

	log.Printf("Loaded sherpa streaming engine (encoder=%s, provider=%s, endpoint=%v, punct=%v, softCap=%s)",
		filepath.Base(enc), cfg.Provider, cfg.EndpointEnabled(), punct != nil, softCap)

	return &SherpaEngine{
		recognizer: recognizer,
		endpointOn: cfg.EndpointEnabled(),
		punct:      punct,
		softCap:    softCap,
		streams:    make(map[string]*sherpaStream),
	}, nil
}

func (e *SherpaEngine) SetCallback(fn func(StreamingResult)) { e.callback = fn }

// Feed downmixes-already (mono48k), resamples to 16k, decodes, and emits a
// partial when the rolling hypothesis text changes. If endpoint detection is
// enabled and fires, it commits a final and resets the user's stream.
func (e *SherpaEngine) Feed(userID, username, nickname string, mono48k []int16) {
	if len(mono48k) == 0 {
		return
	}
	s := e.getOrCreate(userID)

	s.mu.Lock()
	defer s.mu.Unlock()

	s.username = username
	s.nickname = nickname
	now := time.Now()
	if s.turnStart.IsZero() {
		s.turnStart = now
		s.segStart = now
	}
	s.segSamples += len(mono48k)
	for _, v := range mono48k {
		f := float64(v)
		s.segSumSq += f * f
	}

	samples16 := s.resampler.Process(mono48k)
	if len(samples16) == 0 {
		return
	}

	e.recMu.Lock()
	s.stream.AcceptWaveform(recognizerSampleRate, samples16)
	for e.recognizer.IsReady(s.stream) {
		e.recognizer.Decode(s.stream)
	}
	raw := strings.TrimSpace(e.recognizer.GetResult(s.stream).Text)
	endpoint := e.endpointOn && e.recognizer.IsEndpoint(s.stream)
	e.recMu.Unlock()

	// Emit the uncommitted tail as a lowercased partial (cheap — no punctuation
	// per partial; full truecasing happens on commit). Lowercasing is a free
	// readability win over the model's ALL-CAPS output.
	tail := uncommittedTail(raw, s.committedWords)
	if partial := strings.ToLower(tail); partial != "" && partial != s.lastPartial {
		s.lastPartial = partial
		e.emit(StreamingResult{UserID: userID, Username: username, Nickname: nickname, Text: partial})
	}

	if endpoint {
		e.finalizeLocked(userID, s, false)
		return
	}

	// Long-turn sentence-aware break: once a turn runs past softCap, commit any
	// completed sentences as their own lines so a monologue doesn't grow into one
	// huge line. Throttled to ~1/sec and only when punctuation is available
	// (no punctuation model => no sentence boundaries to split on).
	if e.punct != nil && e.softCap > 0 && now.Sub(s.turnStart) > e.softCap &&
		now.Sub(s.lastSentenceCheck) > time.Second {
		s.lastSentenceCheck = now
		e.commitSentencesLocked(userID, s, raw)
	}
}

// Flush finalizes a user's current utterance (called on voice-layer silence).
// It pads the encoder via InputFinished so the trailing chunk is decoded, then
// commits the final and recreates the stream for a clean next utterance.
func (e *SherpaEngine) Flush(userID string) {
	e.mu.Lock()
	s := e.streams[userID]
	e.mu.Unlock()
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	e.finalizeLocked(userID, s, true)
}

// finalizeLocked emits the remaining (uncommitted) hypothesis tail as the final
// line and fully resets the turn. Caller must hold s.mu. drain=true is the
// silence/Flush path (pad + InputFinished + fresh stream); drain=false is the
// endpoint path (plain recognizer.Reset).
func (e *SherpaEngine) finalizeLocked(userID string, s *sherpaStream, drain bool) {
	e.recMu.Lock()
	if drain {
		// Push the resampler tail and signal end-of-input so the encoder flushes
		// its last chunk, then drain the decoder.
		if tail := s.resampler.Drain(); len(tail) > 0 {
			s.stream.AcceptWaveform(recognizerSampleRate, tail)
		}
		s.stream.InputFinished()
	}
	for e.recognizer.IsReady(s.stream) {
		e.recognizer.Decode(s.stream)
	}
	raw := strings.TrimSpace(e.recognizer.GetResult(s.stream).Text)
	if drain {
		// Recreate the stream: after InputFinished a stream can't take more
		// audio, so swap in a fresh one for the next turn.
		sherpa.DeleteOnlineStream(s.stream)
		s.stream = sherpa.NewOnlineStream(e.recognizer)
	} else {
		e.recognizer.Reset(s.stream)
	}
	e.recMu.Unlock()

	tail := uncommittedTail(raw, s.committedWords)
	// Lowercase before punctuating: the Edge-Punct-Casing model expects
	// lowercase input (truecasing is its job) and passes ALL-CAPS through
	// unchanged. With punctuation disabled, emit the raw caps tail (documented
	// fallback).
	text := tail
	if e.punct != nil {
		text = e.punct.Punctuate(strings.ToLower(tail))
	}
	rms, durationMs := segMetrics(s)
	ts := s.segStart

	// Full turn reset.
	s.resampler.Reset()
	s.lastPartial = ""
	s.committedWords = 0
	s.turnStart = time.Time{}
	s.segStart = time.Time{}
	s.segSamples = 0
	s.segSumSq = 0
	s.lastSentenceCheck = time.Time{}

	if strings.TrimSpace(text) == "" {
		return
	}
	e.emit(StreamingResult{
		UserID:     userID,
		Username:   s.username,
		Nickname:   s.nickname,
		Text:       text,
		IsFinal:    true,
		Timestamp:  ts,
		RMS:        rms,
		DurationMs: durationMs,
	})
}

// commitSentencesLocked punctuates the uncommitted tail and, if it contains any
// completed sentence(s), emits them as a final line and advances committedWords.
// The recognizer is NOT reset — it keeps full context; we just slice its rolling
// hypothesis. Caller must hold s.mu and have verified e.punct != nil.
//
// WHY this can occasionally revise: greedy decoding may change earlier tokens as
// more audio arrives, so a word already committed could differ from a later
// re-decode. With enough trailing context this is rare, and for a transcript the
// cosmetic risk is acceptable — the alternative (resetting mid-turn) loses
// recognizer context and hurts accuracy.
func (e *SherpaEngine) commitSentencesLocked(userID string, s *sherpaStream, raw string) {
	tail := uncommittedTail(raw, s.committedWords)
	if tail == "" {
		return
	}
	// Lowercase first — the punctuation/truecasing model expects lowercase input.
	punctuated := e.punct.Punctuate(strings.ToLower(tail))
	committed, ok := splitCommittable(punctuated)
	if !ok {
		return
	}
	n := len(strings.Fields(committed))
	if n == 0 {
		return
	}

	rms, durationMs := segMetrics(s)
	ts := s.segStart

	s.committedWords += n
	// Start a fresh segment for the next line; turn + recognizer continue.
	s.segStart = time.Now()
	s.segSamples = 0
	s.segSumSq = 0
	s.lastPartial = ""

	e.emit(StreamingResult{
		UserID:     userID,
		Username:   s.username,
		Nickname:   s.nickname,
		Text:       committed,
		IsFinal:    true,
		Timestamp:  ts,
		RMS:        rms,
		DurationMs: durationMs,
	})
}

// uncommittedTail returns the words of raw beyond the first committedWords,
// rejoined. Clamps if the hypothesis shrank (greedy revision).
func uncommittedTail(raw string, committedWords int) string {
	words := strings.Fields(raw)
	if committedWords >= len(words) {
		return ""
	}
	if committedWords < 0 {
		committedWords = 0
	}
	return strings.Join(words[committedWords:], " ")
}

// splitCommittable returns the leading run of completed sentences in punctuated
// text — everything up to and including the last sentence-ending mark that still
// has non-space text after it (the trailing fragment is still being spoken, so
// it stays pending). ok=false when there is no completed sentence yet.
func splitCommittable(text string) (string, bool) {
	lastIdx := -1
	for i, r := range text {
		if r == '.' || r == '!' || r == '?' {
			if strings.TrimSpace(text[i+1:]) != "" {
				lastIdx = i
			}
		}
	}
	if lastIdx < 0 {
		return "", false
	}
	return strings.TrimSpace(text[:lastIdx+1]), true
}

// segMetrics computes the RMS and duration of the current (uncommitted) audio
// segment.
func segMetrics(s *sherpaStream) (rms float32, durationMs int) {
	if s.segSamples > 0 {
		rms = float32(math.Sqrt(s.segSumSq / float64(s.segSamples)))
	}
	durationMs = int(math.Round(float64(s.segSamples) / 48000.0 * 1000))
	return rms, durationMs
}

func (e *SherpaEngine) emit(r StreamingResult) {
	if e.callback != nil {
		e.callback(r)
	}
}

func (e *SherpaEngine) getOrCreate(userID string) *sherpaStream {
	e.mu.Lock()
	defer e.mu.Unlock()
	if s, ok := e.streams[userID]; ok {
		return s
	}
	s := &sherpaStream{
		stream:    sherpa.NewOnlineStream(e.recognizer),
		resampler: NewStreamResampler(),
	}
	e.streams[userID] = s
	return s
}

// Close frees all per-user streams and the recognizer.
func (e *SherpaEngine) Close() error {
	e.mu.Lock()
	for _, s := range e.streams {
		s.mu.Lock()
		if s.stream != nil {
			sherpa.DeleteOnlineStream(s.stream)
			s.stream = nil
		}
		s.mu.Unlock()
	}
	e.streams = make(map[string]*sherpaStream)
	e.mu.Unlock()

	if e.recognizer != nil {
		sherpa.DeleteOnlineRecognizer(e.recognizer)
		e.recognizer = nil
	}
	return e.punct.Close() // nil-safe
}

// resolveModelFiles returns encoder/decoder/joiner/tokens paths. Explicit paths
// in cfg win; otherwise they're discovered in cfg.ModelDir, preferring the
// configured Variant ("int8" or "fp32") and keeping all three the same variant.
func resolveModelFiles(cfg *config.SherpaConfig) (enc, dec, join, tokens string, err error) {
	if cfg.Encoder != "" && cfg.Decoder != "" && cfg.Joiner != "" && cfg.Tokens != "" {
		return cfg.Encoder, cfg.Decoder, cfg.Joiner, cfg.Tokens, nil
	}
	if cfg.ModelDir == "" {
		return "", "", "", "", fmt.Errorf("sherpa: set transcribe.sherpa.model_dir or explicit encoder/decoder/joiner/tokens paths")
	}
	entries, err := os.ReadDir(cfg.ModelDir)
	if err != nil {
		return "", "", "", "", fmt.Errorf("sherpa: read model_dir %q: %w", cfg.ModelDir, err)
	}

	wantInt8 := cfg.Variant != "fp32"
	pick := func(prefix string) string {
		var fp32, int8 string
		for _, ent := range entries {
			name := ent.Name()
			lower := strings.ToLower(name)
			if !strings.HasPrefix(lower, prefix) || !strings.HasSuffix(lower, ".onnx") {
				continue
			}
			if strings.Contains(lower, ".int8.") {
				int8 = filepath.Join(cfg.ModelDir, name)
			} else {
				fp32 = filepath.Join(cfg.ModelDir, name)
			}
		}
		if wantInt8 && int8 != "" {
			return int8
		}
		if !wantInt8 && fp32 != "" {
			return fp32
		}
		// Fall back to whatever variant exists.
		if fp32 != "" {
			return fp32
		}
		return int8
	}

	enc = pick("encoder")
	dec = pick("decoder")
	join = pick("joiner")
	tokens = filepath.Join(cfg.ModelDir, "tokens.txt")

	if enc == "" || dec == "" || join == "" {
		return "", "", "", "", fmt.Errorf("sherpa: model_dir %q missing encoder/decoder/joiner .onnx (got enc=%q dec=%q join=%q)", cfg.ModelDir, enc, dec, join)
	}
	if _, statErr := os.Stat(tokens); statErr != nil {
		return "", "", "", "", fmt.Errorf("sherpa: tokens.txt not found in model_dir %q", cfg.ModelDir)
	}
	return enc, dec, join, tokens, nil
}
