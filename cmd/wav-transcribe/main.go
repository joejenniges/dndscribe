// Command wav-transcribe runs the bot's whisper engine on a saved raw WAV.
// Diagnostic tool: the recordings/raw WAVs are exactly the audio the voice
// layer buffered, so whisper's output here is the ground truth for "did the
// audio contain those words" when comparing engines.
//
// Usage:
//
//	wav-transcribe <model-name> <model-dir> <wav-path> [wav-path...]
package main

import (
	"encoding/binary"
	"fmt"
	"os"

	"github.com/joe/dndscribe-go/internal/transcribe"
)

func main() {
	if len(os.Args) < 4 {
		fmt.Fprintln(os.Stderr, "usage: wav-transcribe <model-name> <model-dir> <wav-path> [wav-path...]")
		os.Exit(2)
	}

	engine, err := transcribe.NewWhisperEngine(os.Args[1], os.Args[2], 4)
	if err != nil {
		fatal(err)
	}
	defer engine.Close()

	for _, wavPath := range os.Args[3:] {
		pcm, sampleRate, err := readWavMonoPCM(wavPath)
		if err != nil {
			fatal(err)
		}
		if sampleRate != 48000 {
			fatal(fmt.Errorf("%s: expected 48kHz mono (voice-layer format), got %d Hz", wavPath, sampleRate))
		}
		samples := transcribe.Resample48to16(pcm)
		text, conf, err := engine.Transcribe(samples, "")
		if err != nil {
			fatal(err)
		}
		fmt.Printf("%s\n  conf=%.2f text=%q\n", wavPath, conf, text)
	}
}

// readWavMonoPCM returns the raw s16le PCM bytes and sample rate of a 16-bit
// mono WAV. Minimal RIFF parser for the voice layer's own output.
func readWavMonoPCM(path string) ([]byte, int, error) {
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
	return pcm, sampleRate, nil
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "error:", err)
	os.Exit(1)
}
