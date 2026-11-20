package dj

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"math"
	"os/exec"
	"strings"
	"sync"
)

// Discord voice audio: 48kHz stereo, 20ms opus frames.
const (
	SampleRate   = 48000
	Channels     = 2
	FrameSamples = 960                     // per channel, 20ms
	FrameLen     = FrameSamples * Channels // interleaved int16 values per frame
	FrameMs      = 20
)

// Source yields 20ms frames of interleaved stereo PCM.
//
// Next never blocks: the mixer runs on a 20ms clock and must not stall on a
// slow decoder. ready=false means no frame is available yet (decoder still
// starting or briefly behind), and the mixer plays silence for this source for
// that tick. done=true means the source is exhausted.
type Source interface {
	Next() (frame []int16, ready bool, done bool)
	Close()
}

// ffmpegSource decodes a file to s16le 48k stereo through an ffmpeg
// subprocess. A goroutine reads ahead into a buffered channel so the mixer
// only ever does a non-blocking receive.
type ffmpegSource struct {
	cmd    *exec.Cmd
	frames chan []int16
	stop   chan struct{}
	once   sync.Once
}

// readAheadFrames is how much decoded audio is buffered per source: 100
// frames = 2s. Enough to ride out ffmpeg startup and disk hiccups; small enough
// that a prefetched next track holds only ~384KB.
const readAheadFrames = 100

func openFFmpeg(ffmpegPath, file string, startMs int64) (Source, error) {
	args := []string{"-hide_banner", "-loglevel", "error", "-nostdin"}
	if startMs > 0 {
		args = append(args, "-ss", fmt.Sprintf("%.3f", float64(startMs)/1000))
	}
	args = append(args, "-i", file, "-vn", "-f", "s16le", "-acodec", "pcm_s16le",
		"-ac", fmt.Sprint(Channels), "-ar", fmt.Sprint(SampleRate), "pipe:1")
	cmd := exec.Command(ffmpegPath, args...)
	hideWindow(cmd)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start ffmpeg: %w", err)
	}
	s := &ffmpegSource{
		cmd:    cmd,
		frames: make(chan []int16, readAheadFrames),
		stop:   make(chan struct{}),
	}
	go func() {
		sc := bufio.NewScanner(stderr)
		for sc.Scan() {
			if line := strings.TrimSpace(sc.Text()); line != "" {
				log.Printf("DJ ffmpeg %s: %s", file, line)
			}
		}
	}()
	go s.readLoop(stdout)
	return s, nil
}

func (s *ffmpegSource) readLoop(r io.Reader) {
	defer close(s.frames)
	br := bufio.NewReaderSize(r, FrameLen*2*8)
	raw := make([]byte, FrameLen*2)
	for {
		n, err := io.ReadFull(br, raw)
		if n > 0 {
			// Zero-pad a short final frame so the track's tail still plays.
			for i := n; i < len(raw); i++ {
				raw[i] = 0
			}
			frame := make([]int16, FrameLen)
			for i := range frame {
				frame[i] = int16(binary.LittleEndian.Uint16(raw[i*2:]))
			}
			select {
			case s.frames <- frame:
			case <-s.stop:
				return
			}
		}
		if err != nil {
			return
		}
	}
}

func (s *ffmpegSource) Next() ([]int16, bool, bool) {
	select {
	case f, ok := <-s.frames:
		if !ok {
			return nil, false, true
		}
		return f, true, false
	default:
		return nil, false, false
	}
}

func (s *ffmpegSource) Close() {
	s.once.Do(func() {
		close(s.stop)
		if s.cmd.Process != nil {
			_ = s.cmd.Process.Kill()
		}
		go func() { _ = s.cmd.Wait() }()
	})
}

// toneSource is a sine wave, used by the "test tone" button to check the voice
// send path without needing any music files.
type toneSource struct {
	freq   float64
	frames int // remaining
	phase  float64
}

func newTone(freq float64, seconds float64) Source {
	return &toneSource{freq: freq, frames: int(seconds * 1000 / FrameMs)}
}

func (t *toneSource) Next() ([]int16, bool, bool) {
	if t.frames <= 0 {
		return nil, false, true
	}
	t.frames--
	frame := make([]int16, FrameLen)
	step := 2 * math.Pi * t.freq / SampleRate
	for i := 0; i < FrameSamples; i++ {
		v := int16(math.Sin(t.phase) * 0.3 * math.MaxInt16)
		frame[i*2], frame[i*2+1] = v, v
		t.phase += step
		if t.phase > 2*math.Pi {
			t.phase -= 2 * math.Pi
		}
	}
	return frame, true, false
}

func (t *toneSource) Close() {}
