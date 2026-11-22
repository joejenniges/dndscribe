package dj

import (
	"testing"

	"github.com/joe/dndscribe-go/internal/db"
)

// seekProgram is one 60s "track" whose openAt records where it was opened.
func seekProgram(opened *[]int64) (*program, *constSource) {
	first := &constSource{val: 1000, n: 3000}
	p := &program{name: "t", scale: 1, items: []item{{
		track: &db.DJTrack{ID: 1, DurationMs: 60_000},
		label: "t",
		open:  func() (Source, error) { return first, nil },
		openAt: func(ms int64) (Source, error) {
			*opened = append(*opened, ms)
			return &constSource{val: 2000, n: 3000}, nil
		},
	}}}
	return p, first
}

func TestSeekSwapsDecoderAndPosition(t *testing.T) {
	e, out, _ := newTestEngine()
	var opened []int64
	p, first := seekProgram(&opened)
	if err := e.startProgram(p, 0); err != nil {
		t.Fatal(err)
	}
	frames(e, out, 10)
	if err := e.SeekTo(30_000); err != nil {
		t.Fatal(err)
	}
	if len(opened) != 1 || opened[0] != 30_000 {
		t.Fatalf("opened at %v, want [30000]", opened)
	}
	if got := e.State().PositionMs; got != 30_000 {
		t.Fatalf("position %d right after seek, want 30000", got)
	}
	frames(e, out, 5)
	if !first.closed {
		t.Fatal("old decoder still open after its fade-out")
	}
	if got := e.State().PositionMs; got != 30_000+5*FrameMs {
		t.Fatalf("position %d, want %d", got, 30_000+5*FrameMs)
	}
}

func TestSeekClampsAndRejects(t *testing.T) {
	e, _, _ := newTestEngine()
	if err := e.SeekTo(1000); err != ErrNotPlaying {
		t.Fatalf("seek with nothing playing: %v", err)
	}
	var opened []int64
	p, _ := seekProgram(&opened)
	if err := e.startProgram(p, 0); err != nil {
		t.Fatal(err)
	}
	_ = e.SeekTo(-5)
	_ = e.SeekTo(10 * 60_000)
	if opened[0] != 0 || opened[1] != 59_500 {
		t.Fatalf("clamped to %v, want [0 59500]", opened)
	}

	tone := &program{name: "tone", scale: 1, items: []item{{label: "tone",
		open: func() (Source, error) { return newTone(440, 5), nil }}}}
	if err := e.startProgram(tone, 0); err != nil {
		t.Fatal(err)
	}
	if err := e.SeekTo(1000); err == nil {
		t.Fatal("seeking the test tone should fail")
	}
}

func TestSeekWhilePausedDropsOldVoice(t *testing.T) {
	e, out, _ := newTestEngine()
	var opened []int64
	p, first := seekProgram(&opened)
	if err := e.startProgram(p, 0); err != nil {
		t.Fatal(err)
	}
	frames(e, out, 5)
	e.SetPaused(true)
	if err := e.SeekTo(20_000); err != nil {
		t.Fatal(err)
	}
	if !first.closed || len(e.fading) != 0 {
		t.Fatal("paused seek left the old voice to play on resume")
	}
}
