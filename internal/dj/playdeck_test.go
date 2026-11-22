package dj

import "testing"

// Play on a deck with the other one silent moves the crossfader to it; with
// the other deck playing, the crossfader stays where it is.
func TestPlayDeckTakesCrossfaderOnlyFromSilence(t *testing.T) {
	e, out, _ := newTestEngine()
	e.SetAutoDJ(false)
	a, _ := constItem("a", 1000, 1000, 0)
	b, _ := constItem("b", 1000, 1000, 0)
	if err := e.openOnDeck(0, &a); err != nil {
		t.Fatal(err)
	}
	if err := e.openOnDeck(1, &b); err != nil {
		t.Fatal(err)
	}
	if e.State().Crossfader != 0 {
		t.Fatal("fader should start on A")
	}
	if err := e.PlayDeck(1); err != nil {
		t.Fatal(err)
	}
	if got := e.State().Crossfader; got != 1 {
		t.Fatalf("playing B from silence: fader %.2f, want 1", got)
	}
	frames(e, out, 5)
	if err := e.PlayDeck(0); err != nil {
		t.Fatal(err)
	}
	st := e.State()
	if st.Crossfader != 1 || !st.Decks[0].Playing || !st.Decks[1].Playing {
		t.Fatalf("starting A under B: fader %.2f, playing %v %v", st.Crossfader, st.Decks[0].Playing, st.Decks[1].Playing)
	}
}
