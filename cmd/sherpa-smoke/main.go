// Command sherpa-smoke validates the sherpa-onnx-go native linkage and DLL
// bundling BEFORE any Discord wiring exists. It loads a streaming Zipformer
// recognizer, feeds a 16kHz mono WAV in small chunks (simulating real-time
// audio arrival), and prints partial hypotheses as they update plus the final
// text on endpoint.
//
// WHY this exists as a standalone command: the project builds under MSYS2
// UCRT64 (libopus + libwhisper) while sherpa-onnx-go-windows ships prebuilt
// x86_64-pc-windows-gnu DLLs (onnxruntime.dll, sherpa-onnx-c-api.dll). Proving
// the cgo link + runtime DLL load works here is the cheapest possible gate; if
// it fails, we learn it without having touched the voice pipeline.
//
// Usage:
//
//	sherpa-smoke <model-dir> [wav-path]
//
// model-dir must contain encoder-*.onnx, decoder-*.onnx, joiner-*.onnx and
// tokens.txt (the layout of a sherpa-onnx-streaming-zipformer-* release). If
// wav-path is omitted, the first *.wav under <model-dir>/test_wavs is used.
package main

import (
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: sherpa-smoke <model-dir> [wav-path]")
		os.Exit(2)
	}
	modelDir := os.Args[1]

	enc, dec, join, tokens, err := findModelFiles(modelDir)
	if err != nil {
		fatal(err)
	}

	var wavPath string
	if len(os.Args) >= 3 {
		wavPath = os.Args[2]
	} else {
		wavPath, err = firstTestWav(modelDir)
		if err != nil {
			fatal(err)
		}
	}

	fmt.Printf("encoder: %s\ndecoder: %s\njoiner:  %s\ntokens:  %s\nwav:     %s\n\n",
		enc, dec, join, tokens, wavPath)

	cfg := sherpa.OnlineRecognizerConfig{
		FeatConfig: sherpa.FeatureConfig{SampleRate: 16000, FeatureDim: 80},
		ModelConfig: sherpa.OnlineModelConfig{
			Transducer: sherpa.OnlineTransducerModelConfig{
				Encoder: enc,
				Decoder: dec,
				Joiner:  join,
			},
			Tokens:     tokens,
			NumThreads: 1,
			Provider:   "cpu",
			// ModelType left empty: sherpa auto-detects from the encoder onnx
			// metadata. Hardcoding "zipformer2" breaks first-gen streaming
			// zipformer models (e.g. the en-20M-2023-02-17 release), whose
			// metadata predates the zipformer2 'query_head_dims' field.
			Debug: 0,
		},
		DecodingMethod:          "greedy_search",
		EnableEndpoint:          1,
		Rule1MinTrailingSilence: 2.4,
		Rule2MinTrailingSilence: 1.2,
		Rule3MinUtteranceLength: 20,
	}

	recognizer := sherpa.NewOnlineRecognizer(&cfg)
	if recognizer == nil {
		fatal(fmt.Errorf("NewOnlineRecognizer returned nil (check model paths / native libs)"))
	}
	defer sherpa.DeleteOnlineRecognizer(recognizer)

	samples, sampleRate, err := readWavMonoFloat32(wavPath)
	if err != nil {
		fatal(err)
	}
	fmt.Printf("loaded %d samples @ %d Hz\n\n", len(samples), sampleRate)

	stream := sherpa.NewOnlineStream(recognizer)
	defer sherpa.DeleteOnlineStream(stream)

	// Feed in ~100ms chunks to simulate streaming arrival.
	chunk := sampleRate / 10
	if chunk < 1 {
		chunk = 1
	}
	last := ""
	for off := 0; off < len(samples); off += chunk {
		end := off + chunk
		if end > len(samples) {
			end = len(samples)
		}
		stream.AcceptWaveform(sampleRate, samples[off:end])

		for recognizer.IsReady(stream) {
			recognizer.Decode(stream)
		}
		text := strings.TrimSpace(recognizer.GetResult(stream).Text)
		if text != "" && text != last {
			fmt.Printf("  partial: %s\n", text)
			last = text
		}
		if recognizer.IsEndpoint(stream) {
			if text != "" {
				fmt.Printf("FINAL:   %s\n", text)
			}
			recognizer.Reset(stream)
			last = ""
		}
	}

	// Flush trailing audio with tail padding so the last utterance finalizes.
	stream.InputFinished()
	for recognizer.IsReady(stream) {
		recognizer.Decode(stream)
	}
	final := strings.TrimSpace(recognizer.GetResult(stream).Text)
	if final != "" {
		fmt.Printf("FINAL:   %s\n", final)
	}

	fmt.Println("\nsmoke test OK: native linkage + DLLs load and inference runs")
}

// findModelFiles locates the encoder/decoder/joiner/tokens in a model dir,
// matching whatever filenames the release shipped rather than hardcoding them.
func findModelFiles(dir string) (enc, dec, join, tokens string, err error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return "", "", "", "", fmt.Errorf("read model dir %q: %w", dir, err)
	}
	for _, e := range entries {
		name := e.Name()
		lower := strings.ToLower(name)
		full := filepath.Join(dir, name)
		switch {
		case strings.HasPrefix(lower, "encoder") && strings.HasSuffix(lower, ".onnx"):
			enc = full
		case strings.HasPrefix(lower, "decoder") && strings.HasSuffix(lower, ".onnx"):
			dec = full
		case strings.HasPrefix(lower, "joiner") && strings.HasSuffix(lower, ".onnx"):
			join = full
		case lower == "tokens.txt":
			tokens = full
		}
	}
	if enc == "" || dec == "" || join == "" || tokens == "" {
		return "", "", "", "", fmt.Errorf("model dir %q missing one of encoder/decoder/joiner/tokens (got enc=%q dec=%q join=%q tokens=%q)", dir, enc, dec, join, tokens)
	}
	return enc, dec, join, tokens, nil
}

func firstTestWav(modelDir string) (string, error) {
	wavDir := filepath.Join(modelDir, "test_wavs")
	entries, err := os.ReadDir(wavDir)
	if err != nil {
		return "", fmt.Errorf("no wav given and %q unreadable: %w", wavDir, err)
	}
	var wavs []string
	for _, e := range entries {
		if strings.HasSuffix(strings.ToLower(e.Name()), ".wav") {
			wavs = append(wavs, filepath.Join(wavDir, e.Name()))
		}
	}
	if len(wavs) == 0 {
		return "", fmt.Errorf("no .wav files in %q", wavDir)
	}
	sort.Strings(wavs)
	return wavs[0], nil
}

// readWavMonoFloat32 parses a 16-bit PCM WAV and returns mono float32 samples
// in [-1, 1]. Stereo input is downmixed by averaging. Minimal RIFF parser —
// enough for sherpa's test_wavs, not a general-purpose decoder.
func readWavMonoFloat32(path string) ([]float32, int, error) {
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
	// Walk chunks starting after the 12-byte RIFF header.
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
			off++ // chunks are word-aligned
		}
	}

	if bits != 16 {
		return nil, 0, fmt.Errorf("%q: only 16-bit PCM supported, got %d-bit", path, bits)
	}
	if channels < 1 {
		channels = 1
	}

	n := len(pcm) / 2
	frames := n / channels
	out := make([]float32, frames)
	for i := 0; i < frames; i++ {
		var acc int32
		for c := 0; c < channels; c++ {
			s := int16(binary.LittleEndian.Uint16(pcm[(i*channels+c)*2:]))
			acc += int32(s)
		}
		out[i] = float32(acc) / float32(channels) / 32768.0
	}
	return out, sampleRate, nil
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "error:", err)
	os.Exit(1)
}
