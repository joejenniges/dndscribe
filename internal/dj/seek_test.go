package dj

import (
	"testing"

	"github.com/joe/dndscribe-go/internal/db"
)

// seekItem is one 60s "track" whose openAt records where it was opened.
func seekItem(opened *[]int64) (item, *constSource) {
	first := &constSource{val: 1000, n: 3000}
	return item{
		track: &db.DJTrack{ID: 1, DurationMs: 60_000},
		label: "t",
		scale: 1,
		open:  func() (Source, error) { return first, nil },
		openAt: func(ms int64) (Source, error) {
			*opened = append(*opened, ms)
			return &constSource{val: 2000, n: 3000}, nil
		},
	}, first
}

func TestSeekSwapsDecoderAndPosition(t *testing.T) {
	e, out, _ := newTestEngine()
	var opened []int64
	it, first := seekItem(&opened)
	if err := e.startNow(it, 0, "track"); err != nil {
		t.Fatal(err)
	}
	frames(e, out, 10)
	if err := e.SeekDeck(0, 30_000); err != nil {
		t.Fatal(err)
	}
	if len(opened) != 1 || opened[0] != 30_000 {
		t.Fatalf("opened at %v, want [30000]", opened)
	}
	if got := e.State().Decks[0].PositionMs; got != 30_000 {
		t.Fatalf("position %d right after seek, want 30000", got)
	}
	frames(e, out, 5)
	if !first.closed {
		t.Fatal("old decoder still open after its fade-out")
	}
	if got := e.State().Decks[0].PositionMs; got != 30_000+5*FrameMs {
		t.Fatalf("position %d, want %d", got, 30_000+5*FrameMs)
	}
}

func TestSeekClampsAndRejects(t *testing.T) {
	e, _, _ := newTestEngine()
	if err := e.SeekDeck(0, 1000); err != ErrNotPlaying {
		t.Fatalf("seek with nothing loaded: %v", err)
	}
	if err := e.SeekDeck(2, 1000); err == nil {
		t.Fatal("deck 2 accepted")
	}
	var opened []int64
	it, _ := seekItem(&opened)
	if err := e.startNow(it, 0, "track"); err != nil {
		t.Fatal(err)
	}
	_ = e.SeekDeck(0, -5)
	_ = e.SeekDeck(0, 10*60_000)
	if opened[0] != 0 || opened[1] != 59_500 {
		t.Fatalf("clamped to %v, want [0 59500]", opened)
	}

	noSeek, _ := constItem("tone", 1, 100, 0)
	if err := e.openOnDeck(1, &noSeek); err != nil {
		t.Fatal(err)
	}
	if err := e.SeekDeck(1, 1000); err == nil {
		t.Fatal("seeking an item without openAt should fail")
	}
}

func TestSeekWhilePausedDropsOldVoice(t *testing.T) {
	e, out, _ := newTestEngine()
	var opened []int64
	it, first := seekItem(&opened)
	if err := e.startNow(it, 0, "track"); err != nil {
		t.Fatal(err)
	}
	frames(e, out, 5)
	_ = e.PauseDeck(0)
	frames(e, out, 15)
	if err := e.SeekDeck(0, 20_000); err != nil {
		t.Fatal(err)
	}
	if !first.closed || len(e.decks[0].tails) != 0 {
		t.Fatal("paused seek left the old voice to play on resume")
	}
}
