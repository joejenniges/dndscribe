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
	"time"

	"github.com/joe/dndscribe-go/internal/db"
)

// Analyzer runs beat analysis (beatdetect.go) in the background, one track
// at a time, in this process. It never touches the mixer: decoding is an
// ffmpeg subprocess and the DSP runs on the worker goroutine.
//
// Results are cached in dj_track_analysis keyed by size+mtime (like the
// waveform cache), so a changed file is re-analysed and a renamed one isn't.
// Every track gets analysed after a library scan or upload; a track someone
// asks about (GET, or Analyze) jumps the queue.
type Analyzer struct {
	ffmpeg   string
	musicDir string

	// OnChanged is called when a track's analysis status or result changes,
	// so clients showing it refetch.
	OnChanged func(trackID int64)

	mu      sync.Mutex
	queue   []int64
	queued  map[int64]bool
	running bool
	filling bool
	refill  bool
}

// maxAnalyseSecs: longer files (an hour of rain) aren't analysed. They
// rarely have a beat, and the decode would hold ~80MB per 15 minutes.
const maxAnalyseSecs = 15 * 60

func NewAnalyzer(ffmpegPath, musicDir string) *Analyzer {
	return &Analyzer{ffmpeg: ffmpegPath, musicDir: musicDir, queued: map[int64]bool{}}
}

// Enqueue schedules a track. front puts it ahead of the background fill
// (someone is waiting for it). Already queued tracks just move up if front.
func (a *Analyzer) Enqueue(ctx context.Context, t db.DJTrack, front bool) {
	a.mu.Lock()
	if a.queued[t.ID] {
		if front {
			for i, id := range a.queue {
				if id == t.ID {
					a.queue = append(a.queue[:i], a.queue[i+1:]...)
					break
				}
			}
			a.queue = append([]int64{t.ID}, a.queue...)
		}
		a.mu.Unlock()
		return
	}
	a.queued[t.ID] = true
	if front {
		a.queue = append([]int64{t.ID}, a.queue...)
	} else {
		a.queue = append(a.queue, t.ID)
	}
	start := !a.running
	a.running = true
	a.mu.Unlock()

	if err := db.SetDJAnalysisStatus(ctx, t.ID, t.SizeBytes, t.ModTime, beatAlgoVersion, "pending"); err != nil {
		log.Printf("DJ beats: mark %d pending: %v", t.ID, err)
	}
	a.changed(t.ID)
	if start {
		go a.work()
	}
}

func (a *Analyzer) changed(id int64) {
	if a.OnChanged != nil {
		a.OnChanged(id)
	}
}

func (a *Analyzer) work() {
	for {
		a.mu.Lock()
		if len(a.queue) == 0 {
			a.running = false
			a.mu.Unlock()
			return
		}
		id := a.queue[0]
		a.queue = a.queue[1:]
		delete(a.queued, id)
		a.mu.Unlock()
		a.analyse(id)
	}
}

func (a *Analyzer) analyse(id int64) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	t, err := db.GetDJTrack(ctx, id)
	if err != nil || t == nil || t.Missing {
		return
	}
	_ = db.SetDJAnalysisStatus(ctx, t.ID, t.SizeBytes, t.ModTime, beatAlgoVersion, "running")
	a.changed(t.ID)

	res := db.DJAnalysis{TrackID: t.ID, Version: beatAlgoVersion, Source: "none"}
	start := time.Now()
	if t.DurationMs > maxAnalyseSecs*1000 {
		res.Status, res.Error = "unavailable", "too long to analyse"
	} else if x, err := decodeMono(ctx, a.ffmpeg, filepath.Join(a.musicDir, filepath.FromSlash(t.Path))); err != nil {
		res.Status, res.Error = "failed", err.Error()
	} else {
		r := detectBeats(x)
		if r.unavailable != "" {
			res.Status, res.Error = "unavailable", r.unavailable
		} else {
			bpm, conf := r.bpm, r.confidence
			res.Status, res.Source, res.BPM, res.Confidence, res.Beats = "ready", "auto", &bpm, &conf, r.beatsMs
		}
	}
	if err := db.SaveDJAnalysis(ctx, t.SizeBytes, t.ModTime, res); err != nil {
		log.Printf("DJ beats: save %s: %v", t.Path, err)
	}
	if res.Status == "failed" {
		log.Printf("DJ beats: %s: %s", t.Path, res.Error)
	} else if res.BPM != nil {
		log.Printf("DJ beats: %s: %.1f BPM, confidence %.2f, %d beats (%s)", t.Path, *res.BPM, *res.Confidence, len(res.Beats), time.Since(start).Round(time.Millisecond))
	}
	a.changed(t.ID)
}

// Fill queues every track without an up-to-date analysis. Calls while it
// runs make it go round once more.
func (a *Analyzer) Fill(ctx context.Context) {
	a.mu.Lock()
	if a.filling {
		a.refill = true
		a.mu.Unlock()
		return
	}
	a.filling = true
	a.mu.Unlock()
	go func() {
		for {
			tracks, err := db.DJTracksNeedingAnalysis(ctx, beatAlgoVersion)
			if err != nil {
				log.Printf("DJ beats: list: %v", err)
			}
			for _, t := range tracks {
				a.Enqueue(ctx, t, false)
			}
			a.mu.Lock()
			if !a.refill || ctx.Err() != nil {
				a.filling = false
				a.mu.Unlock()
				return
			}
			a.refill = false
			a.mu.Unlock()
		}
	}()
}

// decodeMono decodes file to mono float samples at beatRate.
func decodeMono(ctx context.Context, ffmpegPath, file string) ([]float64, error) {
	cmd := exec.CommandContext(ctx, ffmpegPath, "-hide_banner", "-loglevel", "error", "-nostdin",
		"-i", file, "-vn", "-ac", "1", "-ar", fmt.Sprint(beatRate), "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1")
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
	br := bufio.NewReaderSize(stdout, 64<<10)
	var x []float64
	buf := make([]byte, 8192)
	limit := maxAnalyseSecs * beatRate
	for {
		n, rerr := io.ReadFull(br, buf)
		for i := 0; i+1 < n; i += 2 {
			x = append(x, float64(int16(binary.LittleEndian.Uint16(buf[i:])))/32768)
		}
		if rerr != nil || len(x) > limit {
			break
		}
	}
	_ = stdout.Close()
	werr := cmd.Wait()
	if len(x) == 0 {
		if werr != nil {
			return nil, fmt.Errorf("decode: %v: %s", werr, stderr.String())
		}
		return nil, errors.New("decode produced no audio")
	}
	return x, nil
}

// ---- Manual grid edits ----

// GridEdit is a manual correction to a track's beat grid.
type GridEdit struct {
	Op          string  `json:"op"` // half, double, shift, grid, off
	Phase       int     `json:"phase"`
	ShiftMs     int64   `json:"shiftMs"`
	BPM         float64 `json:"bpm"`
	FirstBeatMs int64   `json:"firstBeatMs"`
}

// applyGridEdit returns the corrected beats. durationMs bounds generated
// grids.
func applyGridEdit(beats []int64, e GridEdit, durationMs int64) ([]int64, error) {
	switch e.Op {
	case "half":
		if len(beats) < 4 {
			return nil, errors.New("not enough beats to halve")
		}
		var out []int64
		for i := e.Phase & 1; i < len(beats); i += 2 {
			out = append(out, beats[i])
		}
		return out, nil
	case "double":
		if len(beats) < 2 {
			return nil, errors.New("not enough beats to double")
		}
		out := make([]int64, 0, 2*len(beats))
		for i, b := range beats {
			out = append(out, b)
			if i+1 < len(beats) {
				out = append(out, (b+beats[i+1])/2)
			}
		}
		return out, nil
	case "shift":
		if e.ShiftMs == 0 || e.ShiftMs < -2000 || e.ShiftMs > 2000 {
			return nil, errors.New("shiftMs must be between -2000 and 2000, not 0")
		}
		var out []int64
		for _, b := range beats {
			if v := b + e.ShiftMs; v >= 0 && (durationMs <= 0 || v <= durationMs) {
				out = append(out, v)
			}
		}
		return out, nil
	case "grid":
		if e.BPM < 30 || e.BPM > 300 {
			return nil, errors.New("bpm must be 30-300")
		}
		if durationMs <= 0 {
			return nil, errors.New("the track's length is unknown")
		}
		step := 60000 / e.BPM
		first := math.Mod(float64(e.FirstBeatMs), step)
		if first < 0 {
			first += step
		}
		var out []int64
		for t := first; t <= float64(durationMs); t += step {
			out = append(out, int64(math.Round(t)))
		}
		return out, nil
	case "off":
		return []int64{}, nil
	}
	return nil, fmt.Errorf("unknown grid op %q", e.Op)
}
