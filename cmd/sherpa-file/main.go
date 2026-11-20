// Command sherpa-file drives the PRODUCTION SherpaEngine (warmup padding,
// resampler, finalize path — everything the bot uses) over a 48kHz mono WAV,
// fed in 20ms frames to mimic Discord packet cadence. Diagnostic counterpart
// to wav-transcribe (whisper): run both on the same recordings/raw WAV to
// compare engines on identical audio.
//
// Usage:
//
//	sherpa-file <model-dir> <wav-path> [wav-path...]
package main

import (
	"encoding/binary"
	"fmt"
	"os"

	"github.com/joe/dndscribe-go/internal/config"
	"github.com/joe/dndscribe-go/internal/transcribe"
)

func main() {
	if len(os.Args) < 3 {
		fmt.Fprintln(os.Stderr, "usage: sherpa-file <model-dir> <wav-path> [wav-path...]")
		os.Exit(2)
	}

	cfg := &config.SherpaConfig{ModelDir: os.Args[1]}
	// SHERPA_DECODING=modified_beam_search overrides the default greedy_search,
	// for comparing decoding methods on the same audio.
	cfg.DecodingMethod = os.Getenv("SHERPA_DECODING")
	config.ApplySherpaDefaults(cfg)

	eng, err := transcribe.NewSherpaEngine(cfg, nil)
	if err != nil {
		fatal(err)
	}
	defer eng.Close()

	for _, wavPath := range os.Args[2:] {
		pcm, sampleRate, err := readWavMonoInt16(wavPath)
		if err != nil {
			fatal(err)
		}
		if sampleRate != 48000 {
			fatal(fmt.Errorf("%s: expected 48kHz mono (voice-layer format), got %d Hz", wavPath, sampleRate))
		}

		var finals []string
		eng.SetCallback(func(r transcribe.StreamingResult) {
			if r.IsFinal {
				finals = append(finals, r.Text)
			}
		})

		const frame = 960 // 20ms at 48kHz, matching Discord packets
		for off := 0; off < len(pcm); off += frame {
			end := off + frame
			if end > len(pcm) {
				end = len(pcm)
			}
			eng.Feed("diag", "diag", "", pcm[off:end])
		}
		eng.Flush("diag")

		fmt.Printf("%s\n", wavPath)
		for i, f := range finals {
			fmt.Printf("  final[%d]: %q\n", i, f)
		}
		if len(finals) == 0 {
			fmt.Println("  (no finals)")
		}
	}
}

// readWavMonoInt16 returns the samples and sample rate of a 16-bit mono WAV.
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

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "error:", err)
	os.Exit(1)
}
