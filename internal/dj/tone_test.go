package dj

import "testing"

// The test tone is 2s by default and ramps in and out so it doesn't click.
func TestToneLengthAndRamps(t *testing.T) {
	src := newTone(440, 2)
	var frames [][]int16
	for {
		f, ready, done := src.Next()
		if done {
			break
		}
		if ready {
			frames = append(frames, f)
		}
	}
	if len(frames) != 100 {
		t.Fatalf("%d frames, want 100 (2s)", len(frames))
	}
	if v := frames[0][0]; v != 0 {
		t.Fatalf("first sample %d, want 0", v)
	}
	last := frames[99]
	if v := last[len(last)-2]; v < -40 || v > 40 {
		t.Fatalf("last sample %d, want near 0", v)
	}
	peak := int16(0)
	for _, v := range frames[50] {
		peak = max(peak, v)
	}
	if peak < 9000 {
		t.Fatalf("mid-tone peak %d, want full level (~9830)", peak)
	}
}
