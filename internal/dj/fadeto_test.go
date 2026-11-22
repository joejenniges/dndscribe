package dj

import "testing"

// Crossfading to a cued deck starts it, moves the fader over the global fade
// and unloads the deck it left.
func TestFadeToCuedDeck(t *testing.T) {
	e, out, _ := newTestEngine() // global fade 1s = 50 frames
	e.SetAutoDJ(false)
	a, sa := constItem("a", 1000, 1000, 0)
	if err := e.startNow(a, 0, "scene"); err != nil {
		t.Fatal(err)
	}
	frames(e, out, 60)
	b, _ := constItem("b", 1000, 1000, 0)
	if err := e.openOnDeck(1, &b); err != nil {
		t.Fatal(err)
	}
	if e.State().Decks[1].Playing {
		t.Fatal("a cued deck should not be playing")
	}
	if err := e.FadeToDeck(1); err != nil {
		t.Fatal(err)
	}
	st := e.State()
	if tr := st.Transition; tr == nil || tr.ToDeck != 1 || tr.DurationMs != 1000 || !st.Decks[1].Playing {
		t.Fatalf("transition %+v, deck B playing %v", st.Transition, st.Decks[1].Playing)
	}
	frames(e, out, 60)
	st = e.State()
	if st.Crossfader != 1 || st.Decks[0].Track != nil || !sa.closed {
		t.Fatalf("after fade: fader %.2f, deck A %+v", st.Crossfader, st.Decks[0].Track)
	}
	if err := e.FadeToDeck(0); err == nil {
		t.Fatal("fading to an empty deck should fail")
	}
}
