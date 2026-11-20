package dj

import (
	"bytes"
	"context"
	"fmt"
	"io/fs"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"gopkg.in/hraban/opus.v2"

	"github.com/joe/dndscribe-go/internal/db"
)

var audioExts = map[string]bool{
	".mp3": true, ".ogg": true, ".opus": true, ".flac": true, ".wav": true,
	".m4a": true, ".aac": true, ".webm": true, ".wma": true,
}

// Library keeps dj_tracks in sync with the music folder.
type Library struct {
	ffmpeg   string
	musicDir string

	mu       sync.Mutex
	scanning bool
}

func NewLibrary(ffmpegPath, musicDir string) *Library {
	return &Library{ffmpeg: ffmpegPath, musicDir: musicDir}
}

// ScanResult summarises a library scan.
type ScanResult struct {
	Found   int `json:"found"`
	Probed  int `json:"probed"`
	Missing int `json:"missing"`
}

// Scan walks the music folder, probes new or changed files for duration and
// tags, and flags tracks whose files are gone. Unchanged files (same size and
// mtime) are not re-probed, so a rescan of a big folder is quick.
func (l *Library) Scan(ctx context.Context) (*ScanResult, error) {
	l.mu.Lock()
	if l.scanning {
		l.mu.Unlock()
		return nil, fmt.Errorf("a scan is already running")
	}
	l.scanning = true
	l.mu.Unlock()
	defer func() {
		l.mu.Lock()
		l.scanning = false
		l.mu.Unlock()
	}()

	if err := os.MkdirAll(l.musicDir, 0o755); err != nil {
		return nil, err
	}

	existing, err := db.ListDJTracks(ctx)
	if err != nil {
		return nil, err
	}
	known := make(map[string]db.DJTrack, len(existing))
	for _, t := range existing {
		known[t.Path] = t
	}

	type job struct {
		rel  string
		full string
		info fs.FileInfo
	}
	var jobs []job
	var present []string
	err = filepath.WalkDir(l.musicDir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil // unreadable entry: skip, keep scanning
		}
		if d.IsDir() || !audioExts[strings.ToLower(filepath.Ext(p))] {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return nil
		}
		rel, err := filepath.Rel(l.musicDir, p)
		if err != nil {
			return nil
		}
		rel = filepath.ToSlash(rel)
		present = append(present, rel)
		if k, ok := known[rel]; ok && !k.Missing && k.SizeBytes == info.Size() && k.ModTime.Equal(info.ModTime().Truncate(time.Microsecond)) {
			return nil
		}
		jobs = append(jobs, job{rel: rel, full: p, info: info})
		return nil
	})
	if err != nil {
		return nil, err
	}

	// Probe in parallel: each probe is an ffmpeg process start, which is the
	// slow part on Windows.
	var wg sync.WaitGroup
	sem := make(chan struct{}, 4)
	var probeErrs int
	var errMu sync.Mutex
	for _, j := range jobs {
		wg.Add(1)
		sem <- struct{}{}
		go func(j job) {
			defer wg.Done()
			defer func() { <-sem }()
			t := Probe(l.ffmpeg, j.full)
			t.Path = j.rel
			t.SizeBytes = j.info.Size()
			t.ModTime = j.info.ModTime().Truncate(time.Microsecond)
			if err := db.UpsertDJTrack(ctx, t); err != nil {
				errMu.Lock()
				probeErrs++
				errMu.Unlock()
				log.Printf("DJ: save track %s: %v", j.rel, err)
			}
		}(j)
	}
	wg.Wait()

	if present == nil {
		present = []string{}
	}
	if err := db.SetDJTracksMissing(ctx, present); err != nil {
		return nil, err
	}
	presentSet := make(map[string]bool, len(present))
	for _, p := range present {
		presentSet[p] = true
	}
	missing := 0
	for p := range known {
		if !presentSet[p] {
			missing++
		}
	}
	res := &ScanResult{Found: len(present), Probed: len(jobs) - probeErrs, Missing: missing}
	log.Printf("DJ: library scan of %s: %d files, %d probed, %d missing", l.musicDir, res.Found, res.Probed, res.Missing)
	return res, nil
}

var (
	reDuration = regexp.MustCompile(`Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)`)
	reTag      = regexp.MustCompile(`^\s+(title|artist)\s*:\s*(.*)$`)
)

// Probe reads duration and title/artist tags by running `ffmpeg -i file` and
// parsing its banner. WHY not ffprobe: the only ffmpeg on the target machine
// is a bundled copy without ffprobe, and the banner has everything we need.
func Probe(ffmpegPath, file string) db.DJTrack {
	var t db.DJTrack
	cmd := exec.Command(ffmpegPath, "-hide_banner", "-nostdin", "-i", file)
	hideWindow(cmd)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	_ = cmd.Run() // exits non-zero because no output is given; the banner is what we want

	for _, line := range strings.Split(stderr.String(), "\n") {
		line = strings.TrimRight(line, "\r")
		// Only container-level metadata: stream sections can repeat tags
		// with different values (e.g. an encoder title).
		if strings.Contains(line, "Stream #") {
			break
		}
		if m := reTag.FindStringSubmatch(line); m != nil {
			switch m[1] {
			case "title":
				if t.Title == "" {
					t.Title = strings.TrimSpace(m[2])
				}
			case "artist":
				if t.Artist == "" {
					t.Artist = strings.TrimSpace(m[2])
				}
			}
		}
		if m := reDuration.FindStringSubmatch(line); m != nil && t.DurationMs == 0 {
			h, _ := strconv.Atoi(m[1])
			mi, _ := strconv.Atoi(m[2])
			s, _ := strconv.ParseFloat(m[3], 64)
			t.DurationMs = int64((float64(h*3600+mi*60) + s) * 1000)
		}
	}
	return t
}

// opusEncoder adapts hraban/opus to Encoder.
type opusEncoder struct{ enc *opus.Encoder }

// NewOpusEncoder creates a 48kHz stereo music encoder.
func NewOpusEncoder(bitrate int) (Encoder, error) {
	enc, err := opus.NewEncoder(SampleRate, Channels, opus.AppAudio)
	if err != nil {
		return nil, err
	}
	if err := enc.SetBitrate(bitrate); err != nil {
		return nil, err
	}
	return &opusEncoder{enc: enc}, nil
}

func (o *opusEncoder) Encode(pcm []int16, out []byte) (int, error) {
	return o.enc.Encode(pcm, out)
}
