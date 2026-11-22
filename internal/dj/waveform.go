package dj

import (
	"bufio"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"os/exec"
	"path/filepath"
	"sync"

	"github.com/joe/dndscribe-go/internal/db"
)

// Waveforms computes and caches per-track waveforms for the DJ page.
//
// A waveform is a list of bins, each holding the peak and RMS level of that
// slice of the track, as one byte each. 100 bins per second is fine enough
// for the zoomed, scrolling view; the whole-track overview is drawn by the
// browser from the same data, so only one resolution is stored.
//
// Computing one is a full decode of the file (around a second for a typical
// track), so results go into dj_waveforms and are only recomputed when the
// file changes. Tracks are filled in the background after a scan or upload;
// a request for a track not done yet computes it on the spot.
type Waveforms struct {
	ffmpeg   string
	musicDir string

	mu       sync.Mutex
	inflight map[int64]*waveCall
	filling  bool
	refill   bool
}

type waveCall struct {
	done chan struct{}
	w    *db.DJWaveform
	err  error
}

const (
	waveBinsPerSecond = 100
	// waveMaxBins caps very long files (an hour of rain) at 2 bytes x 360k
	// bins = 720KB; they get coarser bins instead.
	waveMaxBins = 360_000
	// waveDecodeRate: the decode for analysis runs at 8kHz mono. Plenty for
	// a picture of loudness, and much less to read through than 48k stereo.
	waveDecodeRate = 8000
)

func NewWaveforms(ffmpegPath, musicDir string) *Waveforms {
	return &Waveforms{ffmpeg: ffmpegPath, musicDir: musicDir, inflight: map[int64]*waveCall{}}
}

// Get returns a track's waveform, computing and caching it if needed.
// Concurrent requests for the same track share one computation.
func (wf *Waveforms) Get(ctx context.Context, t db.DJTrack) (*db.DJWaveform, error) {
	if w, err := db.GetDJWaveform(ctx, t.ID, t.SizeBytes, t.ModTime); err != nil || w != nil {
		return w, err
	}
	wf.mu.Lock()
	c, ok := wf.inflight[t.ID]
	if !ok {
		c = &waveCall{done: make(chan struct{})}
		wf.inflight[t.ID] = c
		go func() {
			// Detached from the request: if the browser gives up, the
			// result is still worth keeping.
			c.w, c.err = wf.compute(context.Background(), t)
			wf.mu.Lock()
			delete(wf.inflight, t.ID)
			wf.mu.Unlock()
			close(c.done)
		}()
	}
	wf.mu.Unlock()
	select {
	case <-c.done:
		return c.w, c.err
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (wf *Waveforms) compute(ctx context.Context, t db.DJTrack) (*db.DJWaveform, error) {
	bps := waveBinsPerSecond
	if secs := t.DurationMs / 1000; secs > 0 && secs*int64(bps) > waveMaxBins {
		bps = max(1, int(waveMaxBins/secs))
	}
	full := filepath.Join(wf.musicDir, filepath.FromSlash(t.Path))
	data, err := analyse(ctx, wf.ffmpeg, full, waveDecodeRate/bps)
	if err != nil {
		return nil, err
	}
	w := &db.DJWaveform{BinsPerSecond: bps, Data: data}
	if err := db.SaveDJWaveform(ctx, t.ID, t.SizeBytes, t.ModTime, *w); err != nil {
		// Still usable this time; it'll be computed again next time.
		log.Printf("DJ waveform: save %s: %v", t.Path, err)
	}
	return w, nil
}

// analyse decodes file to mono 8kHz and returns [peak, rms] bytes per bin of
// binSamples samples. Levels are normalised to the track's loudest peak, so
// a quiet ambient track still shows its shape rather than a flat line.
func analyse(ctx context.Context, ffmpegPath, file string, binSamples int) ([]byte, error) {
	cmd := exec.CommandContext(ctx, ffmpegPath, "-hide_banner", "-loglevel", "error", "-nostdin",
		"-i", file, "-vn", "-ac", "1", "-ar", fmt.Sprint(waveDecodeRate), "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1")
	hideWindow(cmd)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	var stderr limitedBuffer
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start ffmpeg: %w", err)
	}

	var peaks, rmss []float64
	br := bufio.NewReaderSize(stdout, 64<<10)
	buf := make([]byte, binSamples*2)
	for {
		n, rerr := io.ReadFull(br, buf)
		if n >= 2 {
			var peak, sum float64
			cnt := n / 2
			for i := 0; i < cnt; i++ {
				v := float64(int16(binary.LittleEndian.Uint16(buf[i*2:]))) / 32768
				sum += v * v
				if a := math.Abs(v); a > peak {
					peak = a
				}
			}
			peaks = append(peaks, peak)
			rmss = append(rmss, math.Sqrt(sum/float64(cnt)))
		}
		if rerr != nil {
			break
		}
	}
	werr := cmd.Wait()
	if len(peaks) == 0 {
		if werr != nil {
			return nil, fmt.Errorf("decode: %v: %s", werr, stderr.String())
		}
		return nil, errors.New("decode produced no audio")
	}

	top := 0.0
	for _, p := range peaks {
		top = math.Max(top, p)
	}
	if top < 1e-4 {
		top = 1 // silence: draw it flat rather than amplifying noise
	}
	out := make([]byte, len(peaks)*2)
	for i := range peaks {
		out[i*2] = byte(math.Round(math.Min(1, peaks[i]/top) * 255))
		out[i*2+1] = byte(math.Round(math.Min(1, rmss[i]/top) * 255))
	}
	return out, nil
}

// Fill computes waveforms for every track that lacks one, one at a time, in
// the background. Calling it while it runs makes it go round once more when
// done, so tracks uploaded mid-fill are picked up.
func (wf *Waveforms) Fill(ctx context.Context) {
	wf.mu.Lock()
	if wf.filling {
		wf.refill = true
		wf.mu.Unlock()
		return
	}
	wf.filling = true
	wf.mu.Unlock()

	go func() {
		for {
			wf.fillOnce(ctx)
			wf.mu.Lock()
			if !wf.refill || ctx.Err() != nil {
				wf.filling = false
				wf.mu.Unlock()
				return
			}
			wf.refill = false
			wf.mu.Unlock()
		}
	}()
}

func (wf *Waveforms) fillOnce(ctx context.Context) {
	tracks, err := db.DJTracksWithoutWaveform(ctx)
	if err != nil {
		log.Printf("DJ waveform: list: %v", err)
		return
	}
	done := 0
	for _, t := range tracks {
		if ctx.Err() != nil {
			return
		}
		if _, err := wf.Get(ctx, t); err != nil {
			log.Printf("DJ waveform: %s: %v", t.Path, err)
			continue
		}
		done++
	}
	if done > 0 {
		log.Printf("DJ waveform: computed %d", done)
	}
}

// limitedBuffer keeps the first 2KB written to it (ffmpeg's error output).
type limitedBuffer struct{ b []byte }

func (l *limitedBuffer) Write(p []byte) (int, error) {
	if room := 2048 - len(l.b); room > 0 {
		l.b = append(l.b, p[:min(room, len(p))]...)
	}
	return len(p), nil
}

func (l *limitedBuffer) String() string { return string(l.b) }
