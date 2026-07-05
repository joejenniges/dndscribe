package config

import (
	"os"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Discord    DiscordConfig    `yaml:"discord"`
	Database   DatabaseConfig   `yaml:"database"`
	Web        WebConfig        `yaml:"web"`
	Transcribe TranscribeConfig `yaml:"transcribe"`
	Recordings RecordingsConfig `yaml:"recordings"`
}

type DiscordConfig struct {
	Token   string `yaml:"token"`
	GuildID string `yaml:"guild_id"`
}

type DatabaseConfig struct {
	URL string `yaml:"url"`
}

type WebConfig struct {
	Port int `yaml:"port"`
}

type TranscribeConfig struct {
	Engine  string `yaml:"engine"` // whisper (batch) | sherpa (streaming)
	Model   string `yaml:"model"`  // whisper model name
	Threads int    `yaml:"threads"`

	Sherpa SherpaConfig `yaml:"sherpa"`
}

// SherpaConfig configures the streaming sherpa-onnx engine. ModelDir points at
// an extracted streaming-zipformer release (download-model.ps1); the engine
// finds encoder/decoder/joiner/tokens inside it. Explicit Encoder/Decoder/
// Joiner/Tokens paths override the dir-based discovery if set.
type SherpaConfig struct {
	ModelDir string `yaml:"model_dir"`
	Encoder  string `yaml:"encoder"`
	Decoder  string `yaml:"decoder"`
	Joiner   string `yaml:"joiner"`
	Tokens   string `yaml:"tokens"`

	// Variant selects which onnx files to prefer inside ModelDir: "int8"
	// (smaller/faster, default) or "fp32". Ignored when explicit paths are set.
	Variant string `yaml:"variant"`

	// ModelType is normally left empty so sherpa auto-detects from the encoder
	// onnx metadata. WHY: hardcoding "zipformer2" hard-aborts on first-gen
	// zipformer models whose metadata lacks query_head_dims. Only set this if
	// auto-detection picks wrong.
	ModelType  string `yaml:"model_type"`
	Provider   string `yaml:"provider"` // cpu (default), cuda, coreml
	NumThreads int    `yaml:"num_threads"`

	DecodingMethod string `yaml:"decoding_method"` // greedy_search (default) | modified_beam_search
	MaxActivePaths int    `yaml:"max_active_paths"`

	// Endpoint detection (trailing-silence based). Enabled by default (nil) so
	// long monologues commit intermediate finals; the voice-layer silence flush
	// is the primary utterance boundary regardless. Pointer so an explicit
	// `enable_endpoint: false` is distinguishable from "unset".
	EnableEndpoint          *bool   `yaml:"enable_endpoint"`
	Rule1MinTrailingSilence float32 `yaml:"rule1_min_trailing_silence"`
	Rule2MinTrailingSilence float32 `yaml:"rule2_min_trailing_silence"`
	Rule3MinUtteranceLength float32 `yaml:"rule3_min_utterance_length"`

	// SoftCapSeconds: once a single spoken turn exceeds this, the engine starts
	// committing completed sentences as their own lines (sentence-aware break)
	// rather than letting one turn grow until Rule3. Default 12.
	SoftCapSeconds float64 `yaml:"soft_cap_seconds"`

	Punctuation PunctuationConfig `yaml:"punctuation"`
}

// PunctuationConfig configures the streaming punctuation + truecasing pass
// applied to sherpa finals. ModelDir points at an extracted
// sherpa-onnx-online-punct-* release (model.onnx/model.int8.onnx + bpe.vocab);
// explicit CnnBilstm/BpeVocab paths override discovery.
type PunctuationConfig struct {
	Enabled   *bool  `yaml:"enabled"` // default true (nil); set false to ship raw caps
	ModelDir  string `yaml:"model_dir"`
	CnnBilstm string `yaml:"cnn_bilstm"`
	BpeVocab  string `yaml:"bpe_vocab"`
}

// IsEnabled reports whether punctuation should run, defaulting to true when unset.
func (p *PunctuationConfig) IsEnabled() bool {
	return p.Enabled == nil || *p.Enabled
}

type RecordingsConfig struct {
	Dir     string `yaml:"dir"`
	SaveRaw bool   `yaml:"save_raw"`
}

func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}
	if cfg.Web.Port == 0 {
		cfg.Web.Port = 3001
	}
	if cfg.Transcribe.Engine == "" {
		cfg.Transcribe.Engine = "whisper"
	}
	if cfg.Transcribe.Model == "" {
		cfg.Transcribe.Model = "base"
	}
	if cfg.Transcribe.Threads == 0 {
		cfg.Transcribe.Threads = 4
	}
	applySherpaDefaults(&cfg.Transcribe.Sherpa)
	if cfg.Recordings.Dir == "" {
		cfg.Recordings.Dir = "recordings"
	}
	return &cfg, nil
}

func applySherpaDefaults(s *SherpaConfig) {
	if s.Variant == "" {
		s.Variant = "int8"
	}
	if s.Provider == "" {
		s.Provider = "cpu"
	}
	if s.NumThreads == 0 {
		s.NumThreads = 2
	}
	if s.DecodingMethod == "" {
		s.DecodingMethod = "greedy_search"
	}
	if s.MaxActivePaths == 0 {
		s.MaxActivePaths = 4
	}
	// Endpoint rule defaults match sherpa-onnx upstream defaults.
	if s.Rule1MinTrailingSilence == 0 {
		s.Rule1MinTrailingSilence = 2.4
	}
	if s.Rule2MinTrailingSilence == 0 {
		s.Rule2MinTrailingSilence = 1.2
	}
	if s.Rule3MinUtteranceLength == 0 {
		// Backstop only: sentence-aware breaking (SoftCapSeconds) is the primary
		// long-turn splitter now. Rule3 just catches a run-on with no detectable
		// sentence end, so it's raised from the 20s default to 30s.
		s.Rule3MinUtteranceLength = 30
	}
	if s.SoftCapSeconds == 0 {
		s.SoftCapSeconds = 12
	}
}

// EndpointEnabled reports whether endpoint detection should be on, defaulting to
// true when unset.
func (s *SherpaConfig) EndpointEnabled() bool {
	return s.EnableEndpoint == nil || *s.EnableEndpoint
}
