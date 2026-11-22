package dj

import "testing"

// FadeToDeck only glides the fader: both decks keep playing and stay loaded.
func TestFadeToDeckMovesOnlyTheFader(t *testing.T) {
	e, out, _ := newTestEngine() // global fade 1s = 50 frames
	e.SetAutoDJ(false)
	a, _ := constItem("a", 1000, 1000, 0)
	b, _ := constItem("b", 1000, 1000, 0)
	_ = e.openOnDeck(0, &a)
	_ = e.openOnDeck(1, &b)
	_ = e.PlayDeck(0)
	_ = e.PlayDeck(1)
	if err := e.FadeToDeck(1); err != nil {
		t.Fatal(err)
	}
	if tr := e.State().Transition; tr == nil || tr.Kind != "manual" || tr.DurationMs != 1000 {
		t.Fatalf("transition = %+v", tr)
	}
	frames(e, out, 25)
	if x := e.State().Crossfader; x < 0.3 || x > 0.7 {
		t.Fatalf("halfway: fader %.2f", x)
	}
	frames(e, out, 40)
	st := e.State()
	if st.Crossfader != 1 || st.Transition != nil {
		t.Fatalf("after: fader %.2f transition %+v", st.Crossfader, st.Transition)
	}
	if !st.Decks[0].Playing || st.Decks[0].Track == nil || !st.Decks[1].Playing {
		t.Fatal("a manual fade must not stop or unload a deck")
	}
}

// With nothing playing the glide still runs at the fade speed (the engine
// tick steps it), and a glide never stays "in progress" once everything has
// stopped.
func TestFadeToDeckWhenSilent(t *testing.T) {
	e, out, _ := newTestEngine()
	e.SetAutoDJ(false)
	if err := e.FadeToDeck(1); err != nil {
		t.Fatal(err)
	}
	if tr := e.State().Transition; tr == nil || tr.Kind != "manual" {
		t.Fatalf("idle: transition %+v", tr)
	}
	frames(e, out, 25)
	if x := e.State().Crossfader; x < 0.3 || x > 0.7 {
		t.Fatalf("idle halfway: fader %.2f", x)
	}
	frames(e, out, 40)
	if st := e.State(); st.Crossfader != 1 || st.Transition != nil {
		t.Fatalf("idle after: fader %.2f transition %+v", st.Crossfader, st.Transition)
	}
	a, _ := constItem("a", 1000, 1000, 0)
	_ = e.openOnDeck(0, &a)
	_ = e.PlayDeck(0)
	frames(e, out, 5)
	_ = e.FadeToDeck(0)
	frames(e, out, 10)
	_ = e.PauseDeck(0)
	frames(e, out, 30) // ramp down, then silence
	// (Pausing cancels the glide where it is; either way nothing is left
	// showing a fade that will never finish.)
	if st := e.State(); st.Transition != nil {
		t.Fatalf("after stop: transition still running %+v", st.Transition)
	}
}
