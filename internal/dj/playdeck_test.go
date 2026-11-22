package dj

import "testing"

// Play never moves the crossfader: crossfades are always manual.
func TestPlayDeckLeavesCrossfader(t *testing.T) {
	e, _, _ := newTestEngine()
	e.SetAutoDJ(false)
	b, _ := constItem("b", 1000, 1000, 0)
	if err := e.openOnDeck(1, &b); err != nil {
		t.Fatal(err)
	}
	if err := e.PlayDeck(1); err != nil {
		t.Fatal(err)
	}
	if st := e.State(); st.Crossfader != 0 || st.Transition != nil {
		t.Fatalf("fader %.2f transition %+v, want untouched", st.Crossfader, st.Transition)
	}
}
