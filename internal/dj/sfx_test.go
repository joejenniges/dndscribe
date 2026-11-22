package dj

import (
	"context"
	"fmt"
	"math"
	"os/exec"
	"path/filepath"
	"testing"
)

// constPCM is n frames of a constant stereo sample.
func constPCM(val int16, frames int) []int16 {
	p := make([]int16, frames*FrameLen)
	for i := range p {
		p[i] = val
	}
	return p
}

// frames mixes exactly n frames: the output looks one short of full before
// each tick, so the engine tops it up by a single frame.
func frames(e *Engine, out *fakeOutput, n int) {
	buf := make([]byte, 4000)
	for i := 0; i < n; i++ {
		out.pending = targetQueue - 1
		e.tick(buf)
	}
}

// lastSample is the final left sample of the most recent mixed frame.
func lastSample(enc *fakeEncoder) int16 {
	f := enc.frames[len(enc.frames)-1]
	return f[len(f)-2]
}

func TestSfxRetriggerRestartsWithoutStacking(t *testing.T) {
	e, out, enc := newTestEngine()
	frames(e, out, 1) // pick up the connection
	pcm := constPCM(1000, 10)

	if err := e.TriggerPad(1, "A", pcm, 1, ""); err != nil {
		t.Fatal(err)
	}
	frames(e, out, 5)
	if got := lastSample(enc); got != 1000 {
		t.Fatalf("pad A level = %d, want 1000", got)
	}

	// Retrigger mid-sound: one copy, from the beginning.
	_ = e.TriggerPad(1, "A", pcm, 1, "")
	frames(e, out, 1)
	if got := lastSample(enc); got != 1000 {
		t.Fatalf("after retrigger level = %d, want 1000 (copies must not stack)", got)
	}
	st := e.State()
	if len(st.Sfx) != 1 || st.Sfx[0].Instance != 2 || st.Sfx[0].PositionMs != FrameMs {
		t.Fatalf("state after retrigger = %+v, want one instance (2) at 20ms", st.Sfx)
	}

	// It restarted, so it plays the full 10 frames from the retrigger.
	before := len(enc.frames)
	frames(e, out, 20)
	if played := len(enc.frames) - before; played != 9 {
		t.Fatalf("played %d more frames after the first, want 9", played)
	}
}

func TestSfxDifferentPadsOverlapAndRampIn(t *testing.T) {
	e, out, enc := newTestEngine()
	frames(e, out, 1)
	_ = e.TriggerPad(1, "A", constPCM(1000, 10), 1, "")
	_ = e.TriggerPad(2, "B", constPCM(3000, 10), 1, "")
	frames(e, out, 1)
	f := enc.frames[0]
	if f[0] >= 100 {
		t.Fatalf("first sample = %d, want a de-click ramp from near zero", f[0])
	}
	if got := lastSample(enc); got != 4000 {
		t.Fatalf("A+B = %d, want 4000", got)
	}
	if n := len(e.State().Sfx); n != 2 {
		t.Fatalf("%d effects playing, want 2", n)
	}
}

func TestSfxPlaysWhileMusicPausedAndKeepsMusicPosition(t *testing.T) {
	e, out, enc := newTestEngine()
	_ = e.PlayTone(5)
	frames(e, out, 5)
	e.SetPaused(true)
	frames(e, out, 20) // ramp out, then silent
	pos := e.prog.cur.framesPlayed
	n := len(enc.frames)

	_ = e.TriggerPad(1, "A", constPCM(1000, 5), 1, "")
	frames(e, out, 10)
	if len(enc.frames)-n != 5 {
		t.Fatalf("effect sent %d frames while music paused, want 5", len(enc.frames)-n)
	}
	if got := e.prog.cur.framesPlayed; got != pos {
		t.Fatalf("music advanced from %d to %d while paused", pos, got)
	}
	if !e.State().Paused {
		t.Fatal("effect unpaused the music")
	}
}

func TestSfxRequestIDIsIdempotent(t *testing.T) {
	e, out, _ := newTestEngine()
	runTicks(e, out, 1)
	pcm := constPCM(1000, 50)
	_ = e.TriggerPad(1, "A", pcm, 1, "req-1")
	_ = e.TriggerPad(1, "A", pcm, 1, "req-1") // network retry of the same tap
	if e.sfxInstance != 1 {
		t.Fatalf("retried request triggered again (instance %d)", e.sfxInstance)
	}
	_ = e.TriggerPad(1, "A", pcm, 1, "req-2") // a second tap
	if e.sfxInstance != 2 {
		t.Fatalf("second tap didn't retrigger (instance %d)", e.sfxInstance)
	}
}

func TestSfxPolyphonyCeiling(t *testing.T) {
	e, out, _ := newTestEngine()
	runTicks(e, out, 1)
	for i := int64(1); i <= 12; i++ {
		_ = e.TriggerPad(i, fmt.Sprint(i), constPCM(100, 50), 1, "")
	}
	runTicks(e, out, 1)
	st := e.State()
	if len(st.Sfx) != maxSfxVoices {
		t.Fatalf("%d effects playing, want the ceiling %d", len(st.Sfx), maxSfxVoices)
	}
	if st.Sfx[0].PadID != 5 {
		t.Fatalf("oldest surviving pad = %d, want 5 (1-4 cut)", st.Sfx[0].PadID)
	}
}

func TestSfxDroppedOnDisconnect(t *testing.T) {
	e, out, enc := newTestEngine()
	runTicks(e, out, 1)
	_ = e.TriggerPad(1, "A", constPCM(1000, 100), 1, "")
	runTicks(e, out, 2)
	out.ready = false
	runTicks(e, out, 1)
	if len(e.sfx) != 0 {
		t.Fatal("effect survived a disconnect")
	}
	if err := e.TriggerPad(1, "A", constPCM(1000, 100), 1, ""); err != ErrNotConnected {
		t.Fatalf("trigger while disconnected: %v, want ErrNotConnected", err)
	}
	out.ready = true
	n := len(enc.frames)
	runTicks(e, out, 5)
	if len(enc.frames) != n {
		t.Fatal("effect resumed after reconnect")
	}
}

func TestSoftClip(t *testing.T) {
	if softClip(1000) != 1000 || softClip(-20000) != -20000 {
		t.Fatal("soft clip changed a sample below the knee")
	}
	prev := 0.0
	for s := 0.0; s < 200000; s += 500 {
		y := softClip(s)
		if y > math.MaxInt16 || y < prev {
			t.Fatalf("softClip(%v) = %v: must stay in range and rise monotonically", s, y)
		}
		prev = y
	}
	if softClip(-200000) < math.MinInt16 {
		t.Fatal("negative overflow")
	}
}

// TestSfxCacheDecode needs ffmpeg on PATH; skipped otherwise.
func TestSfxCacheDecode(t *testing.T) {
	ff, err := exec.LookPath("ffmpeg")
	if err != nil {
		t.Skip("ffmpeg not on PATH")
	}
	file := filepath.Join(t.TempDir(), "hit.ogg")
	gen := exec.Command(ff, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=880:duration=0.5", file)
	if out, err := gen.CombinedOutput(); err != nil {
		t.Fatalf("generate: %v\n%s", err, out)
	}
	c := NewSfxCache(ff, 64<<20)
	pcm, err := c.Load(context.Background(), file)
	if err != nil {
		t.Fatal(err)
	}
	ms := len(pcm) / Channels * 1000 / SampleRate
	if ms < 450 || ms > 560 {
		t.Fatalf("decoded %dms, want about 500", ms)
	}
	again, _ := c.Load(context.Background(), file)
	if &again[0] != &pcm[0] {
		t.Fatal("second load decoded again instead of using the cache")
	}
}
