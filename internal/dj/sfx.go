package dj

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"os/exec"
	"sync"
	"time"
)

// Soundboard playback: a sound-effect bus summed with the music before the
// master volume and limiter, on the same mixer clock and voice connection.
//
// Semantics (from the soundboard spec):
//   - Each pad has at most one playing instance. Triggering a pad that is
//     playing cuts the old instance (2ms de-click ramp) and starts the sound
//     again from sample zero. Repeated taps never stack copies.
//   - Different pads overlap, up to maxSfxVoices; past that the oldest is cut.
//   - Effects ignore music pause and scene changes. A lost voice connection
//     drops them; they don't resume on reconnect.
//
// Effects are decoded once to PCM and kept in memory (sfxCache), so a tap
// starts on the next mixer tick instead of after an ffmpeg process start
// (~100ms on Windows), which is too slow for a sound effect.

const (
	maxSfxVoices = 8
	// deClickSamples: ramp length at the start and cut of an effect. 2ms is
	// short enough to keep the attack of an explosion or a sword hit intact
	// and long enough to avoid a click when a waveform is cut mid-cycle.
	deClickSamples = 96
	// MaxSfxDuration caps one effect. Longer loops belong in the music library.
	MaxSfxDuration = 5 * time.Minute
)

type sfxVoice struct {
	padID    int64
	instance int64
	label    string
	pcm      []int16 // interleaved stereo, shared with the cache
	pos      int     // next sample frame (per channel) to play
	gain     float64
	stopping int // >0: samples left in the cut ramp
	started  time.Time
}

func (v *sfxVoice) frames() int { return len(v.pcm) / Channels }

// SfxCache holds decoded effects, bounded by total size. Effects are short,
// so in practice the whole board fits; the bound is for a careless 5-minute
// upload or a very large board.
type SfxCache struct {
	ffmpeg string
	limit  int // bytes

	mu      sync.Mutex
	entries map[string]*cacheEntry
	size    int
}

type cacheEntry struct {
	pcm      []int16
	lastUsed time.Time
}

func NewSfxCache(ffmpegPath string, limitBytes int) *SfxCache {
	return &SfxCache{ffmpeg: ffmpegPath, limit: limitBytes, entries: map[string]*cacheEntry{}}
}

// Load returns the decoded PCM for file, decoding it on first use.
func (c *SfxCache) Load(ctx context.Context, file string) ([]int16, error) {
	c.mu.Lock()
	if e, ok := c.entries[file]; ok {
		e.lastUsed = time.Now()
		c.mu.Unlock()
		return e.pcm, nil
	}
	c.mu.Unlock()

	pcm, err := decodeAll(ctx, c.ffmpeg, file)
	if err != nil {
		return nil, err
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	if e, ok := c.entries[file]; ok { // decoded concurrently
		return e.pcm, nil
	}
	c.entries[file] = &cacheEntry{pcm: pcm, lastUsed: time.Now()}
	c.size += len(pcm) * 2
	for c.size > c.limit && len(c.entries) > 1 {
		var oldest string
		var at time.Time
		for k, e := range c.entries {
			if k != file && (oldest == "" || e.lastUsed.Before(at)) {
				oldest, at = k, e.lastUsed
			}
		}
		c.size -= len(c.entries[oldest].pcm) * 2
		delete(c.entries, oldest) // playing voices keep their own slice reference
	}
	return pcm, nil
}

// Forget drops a file from the cache (after it is deleted or replaced).
func (c *SfxCache) Forget(file string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if e, ok := c.entries[file]; ok {
		c.size -= len(e.pcm) * 2
		delete(c.entries, file)
	}
}

// decodeAll decodes a whole file to 48k stereo s16 PCM.
func decodeAll(ctx context.Context, ffmpegPath, file string) ([]int16, error) {
	maxBytes := int(MaxSfxDuration.Seconds()) * SampleRate * Channels * 2
	cmd := exec.CommandContext(ctx, ffmpegPath, "-hide_banner", "-loglevel", "error", "-nostdin",
		"-i", file, "-vn", "-t", fmt.Sprint(int(MaxSfxDuration.Seconds())),
		"-f", "s16le", "-acodec", "pcm_s16le", "-ac", fmt.Sprint(Channels), "-ar", fmt.Sprint(SampleRate), "pipe:1")
	hideWindow(cmd)
	var out, stderr bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		msg := bytes.TrimSpace(stderr.Bytes())
		if len(msg) > 200 {
			msg = msg[:200]
		}
		return nil, fmt.Errorf("can't decode: %v %s", err, msg)
	}
	b := out.Bytes()
	if len(b) > maxBytes {
		b = b[:maxBytes]
	}
	n := len(b) / 2
	n -= n % Channels
	if n == 0 {
		return nil, errors.New("the file has no audio")
	}
	pcm := make([]int16, n)
	for i := range pcm {
		pcm[i] = int16(binary.LittleEndian.Uint16(b[i*2:]))
	}
	return pcm, nil
}

// ---- Engine side ----

// TriggerPad plays pcm as pad padID, restarting it if it is already playing.
// requestID makes a retried HTTP request idempotent: the same requestID
// twice triggers once, while two separate taps (two requestIDs) restart.
func (e *Engine) TriggerPad(padID int64, label string, pcm []int16, gain float64, requestID string) error {
	if !e.isConnected() {
		return ErrNotConnected
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if requestID != "" {
		if e.sfxSeen[requestID] {
			return nil
		}
		e.rememberRequestLocked(requestID)
	}
	kept := e.sfx[:0]
	for _, v := range e.sfx {
		if v.padID == padID && v.stopping == 0 {
			// Retrigger: cut this pad's previous instance. It finishes its
			// 2ms ramp in the next frame and is then dropped.
			v.stopping = deClickSamples
		}
		kept = append(kept, v)
	}
	e.sfx = kept
	// Polyphony ceiling: cut the oldest still-sounding effects.
	sounding := 0
	for _, v := range e.sfx {
		if v.stopping == 0 {
			sounding++
		}
	}
	for i := 0; sounding >= maxSfxVoices && i < len(e.sfx); i++ {
		if e.sfx[i].stopping == 0 {
			e.sfx[i].stopping = deClickSamples
			sounding--
		}
	}
	e.sfxInstance++
	e.sfx = append(e.sfx, &sfxVoice{
		padID: padID, instance: e.sfxInstance, label: label,
		pcm: pcm, gain: clamp01(gain), started: time.Now(),
	})
	e.notify()
	return nil
}

func (e *Engine) rememberRequestLocked(id string) {
	if e.sfxSeen == nil {
		e.sfxSeen = map[string]bool{}
	}
	e.sfxSeen[id] = true
	e.sfxSeenOrder = append(e.sfxSeenOrder, id)
	if len(e.sfxSeenOrder) > 256 {
		delete(e.sfxSeen, e.sfxSeenOrder[0])
		e.sfxSeenOrder = e.sfxSeenOrder[1:]
	}
}

// StopPad cuts a pad's playing instance. padID 0 stops every effect.
func (e *Engine) StopPad(padID int64) {
	e.mu.Lock()
	defer e.mu.Unlock()
	for _, v := range e.sfx {
		if (padID == 0 || v.padID == padID) && v.stopping == 0 {
			v.stopping = deClickSamples
		}
	}
	e.notify()
}

// SetSfxGain sets the soundboard bus gain, 0..1.
func (e *Engine) SetSfxGain(g float64) {
	e.mu.Lock()
	e.sfxGain = clamp01(g)
	e.mu.Unlock()
	e.notify()
}

// mixSfxLocked adds one frame of every effect into acc (already scaled by
// the bus gain). Returns whether any effect produced audio.
func (e *Engine) mixSfxLocked(acc []float64) bool {
	if len(e.sfx) == 0 {
		return false
	}
	bus := e.sfxGain
	contributed := false
	finished := false
	kept := e.sfx[:0]
	for _, v := range e.sfx {
		total := v.frames()
		for i := 0; i < FrameSamples && v.pos < total; i++ {
			g := v.gain * bus
			if v.pos < deClickSamples {
				g *= float64(v.pos+1) / deClickSamples
			}
			if v.stopping > 0 {
				g *= float64(v.stopping) / deClickSamples
				v.stopping--
				if v.stopping == 0 {
					v.pos = total // ramp done: end this instance
					break
				}
			}
			for c := 0; c < Channels; c++ {
				acc[i*Channels+c] += float64(v.pcm[v.pos*Channels+c]) * g
			}
			v.pos++
		}
		contributed = true
		if v.pos >= total {
			finished = true
			continue
		}
		kept = append(kept, v)
	}
	for i := len(kept); i < len(e.sfx); i++ {
		e.sfx[i] = nil
	}
	e.sfx = kept
	if finished {
		e.notify()
	}
	return contributed
}

// SfxPlaying describes one playing effect, for the UI.
type SfxPlaying struct {
	PadID      int64  `json:"padId"`
	Instance   int64  `json:"instance"` // changes on every retrigger
	Label      string `json:"label"`
	PositionMs int64  `json:"positionMs"`
	DurationMs int64  `json:"durationMs"`
}

func (e *Engine) sfxStateLocked() []SfxPlaying {
	out := []SfxPlaying{}
	for _, v := range e.sfx {
		if v.stopping > 0 {
			continue
		}
		out = append(out, SfxPlaying{
			PadID:      v.padID,
			Instance:   v.instance,
			Label:      v.label,
			PositionMs: int64(v.pos) * 1000 / SampleRate,
			DurationMs: int64(v.frames()) * 1000 / SampleRate,
		})
	}
	return out
}
