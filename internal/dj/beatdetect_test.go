package dj

import (
	"math"
	"math/rand"
	"testing"
)

// click adds a short decaying noise burst (a drum-like hit) at sample s.
func click(x []float64, s int, amp float64, rng *rand.Rand) {
	n := beatRate / 25 // 40ms
	for i := 0; i < n && s+i < len(x); i++ {
		if s+i < 0 {
			continue
		}
		x[s+i] += amp * (rng.Float64()*2 - 1) * math.Exp(-float64(i)/float64(n/6))
	}
}

// clickTrack: hits at the given times (seconds), plus quiet noise.
func clickTrack(secs float64, times []float64, amp func(i int) float64, seed int64) []float64 {
	rng := rand.New(rand.NewSource(seed))
	x := make([]float64, int(secs*beatRate))
	for i := range x {
		x[i] = 0.01 * (rng.Float64()*2 - 1)
	}
	for i, t := range times {
		click(x, int(t*beatRate), amp(i), rng)
	}
	return x
}

func steadyTimes(bpm, offset, secs float64) []float64 {
	var t []float64
	for s := offset; s < secs; s += 60 / bpm {
		t = append(t, s)
	}
	return t
}

// matchBeats: fraction of true beats (after the first 2s) that have a
// detected beat within tolMs.
func matchBeats(got []int64, truth []float64, tolMs float64) float64 {
	hit, total := 0, 0
	for _, t := range truth {
		if t < 2 || t > truth[len(truth)-1]-1 {
			continue
		}
		total++
		for _, g := range got {
			if math.Abs(float64(g)-t*1000) <= tolMs {
				hit++
				break
			}
		}
	}
	if total == 0 {
		return 0
	}
	return float64(hit) / float64(total)
}

func one(int) float64 { return 0.8 }

func TestBeatsSteadyClickTrack(t *testing.T) {
	for _, bpm := range []float64{80, 120, 128, 160} {
		truth := steadyTimes(bpm, 0.37, 40)
		r := detectBeats(clickTrack(40, truth, one, 1))
		if r.unavailable != "" {
			t.Fatalf("%v BPM: unavailable: %s", bpm, r.unavailable)
		}
		if math.Abs(r.bpm-bpm) > 1 {
			t.Errorf("%v BPM: estimated %.2f", bpm, r.bpm)
		}
		if m := matchBeats(r.beatsMs, truth, 25); m < 0.95 {
			t.Errorf("%v BPM: %.0f%% of beats within 25ms", bpm, m*100)
		}
		if r.confidence < BeatConfidenceOK {
			t.Errorf("%v BPM: confidence %.2f, want >= %.2f", bpm, r.confidence, BeatConfidenceOK)
		}
		t.Logf("%v BPM -> %.2f, confidence %.2f, %d beats", bpm, r.bpm, r.confidence, len(r.beatsMs))
	}
}

// Kick on the beat with a quieter hi-hat between: the hi-hat must not
// double the tempo.
func TestBeatsKickAndOffbeatHat(t *testing.T) {
	bpm := 90.0
	kicks := steadyTimes(bpm, 0.5, 40)
	var times []float64
	for _, k := range kicks {
		times = append(times, k, k+30/bpm)
	}
	r := detectBeats(clickTrack(40, times, func(i int) float64 {
		if i%2 == 0 {
			return 0.9
		}
		return 0.3
	}, 2))
	if math.Abs(r.bpm-bpm) > 1.5 {
		t.Fatalf("estimated %.2f BPM, want %v (not the hi-hat's %v)", r.bpm, bpm, 2*bpm)
	}
	if m := matchBeats(r.beatsMs, kicks, 25); m < 0.9 {
		t.Fatalf("only %.0f%% of kicks matched", m*100)
	}
}

// Tempo drifting 100 -> 124 BPM over a minute: the grid has to follow
// rather than settle on one BPM.
func TestBeatsFollowTempoDrift(t *testing.T) {
	var times []float64
	for s := 0.5; s < 60; {
		times = append(times, s)
		bpm := 100 + 24*(s/60)
		s += 60 / bpm
	}
	r := detectBeats(clickTrack(60, times, one, 3))
	if m := matchBeats(r.beatsMs, times, 30); m < 0.9 {
		t.Fatalf("only %.0f%% of drifting beats matched within 30ms", m*100)
	}
	t.Logf("drift: median %.1f BPM, confidence %.2f", r.bpm, r.confidence)
}

func TestBeatsNoiseIsUncertain(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	x := make([]float64, 40*beatRate)
	for i := range x {
		x[i] = 0.5 * (rng.Float64()*2 - 1)
	}
	r := detectBeats(x)
	if r.unavailable == "" && r.confidence >= BeatConfidenceOK {
		t.Fatalf("white noise: confidence %.2f, should be uncertain", r.confidence)
	}
	t.Logf("noise: %s confidence %.2f", r.unavailable, r.confidence)
}

// An ambient pad: chords swelling in and out every few seconds, no pulse.
func TestBeatsAmbientPadIsUncertain(t *testing.T) {
	x := make([]float64, 60*beatRate)
	freqs := [][]float64{{220, 277, 330}, {196, 247, 294}, {174, 220, 262}, {247, 311, 370}}
	for i := range x {
		ts := float64(i) / beatRate
		chord := freqs[int(ts/5.3)%len(freqs)]
		phase := math.Mod(ts, 5.3) / 5.3
		env := math.Sin(math.Pi * phase) // slow swell
		for _, f := range chord {
			x[i] += 0.2 * env * math.Sin(2*math.Pi*f*ts)
		}
	}
	r := detectBeats(x)
	if r.unavailable == "" && r.confidence >= BeatConfidenceOK {
		t.Fatalf("ambient pad: confidence %.2f, should be uncertain", r.confidence)
	}
	t.Logf("pad: %q confidence %.2f bpm %.1f", r.unavailable, r.confidence, r.bpm)
}

func TestBeatsSilenceAndShortClips(t *testing.T) {
	if r := detectBeats(make([]float64, 30*beatRate)); r.unavailable == "" {
		t.Fatal("silence produced a grid")
	}
	truth := steadyTimes(120, 0.2, 3)
	if r := detectBeats(clickTrack(3, truth, one, 5)); r.unavailable == "" {
		t.Fatal("a 3 second clip was analysed")
	}
}

func TestFFTMatchesDFT(t *testing.T) {
	rng := rand.New(rand.NewSource(6))
	n := 64
	a := make([]complex128, n)
	for i := range a {
		a[i] = complex(rng.Float64(), 0)
	}
	want := make([]complex128, n)
	for k := 0; k < n; k++ {
		for j := 0; j < n; j++ {
			ang := -2 * math.Pi * float64(j*k) / float64(n)
			want[k] += a[j] * complex(math.Cos(ang), math.Sin(ang))
		}
	}
	fft(a)
	for k := range a {
		if d := a[k] - want[k]; math.Hypot(real(d), imag(d)) > 1e-9 {
			t.Fatalf("bin %d: %v, want %v", k, a[k], want[k])
		}
	}
}

func TestGridEdits(t *testing.T) {
	beats := []int64{100, 600, 1100, 1600, 2100, 2600}
	got, _ := applyGridEdit(beats, GridEdit{Op: "half", Phase: 1}, 10_000)
	if len(got) != 3 || got[0] != 600 || got[2] != 2600 {
		t.Fatalf("half: %v", got)
	}
	got, _ = applyGridEdit(beats, GridEdit{Op: "double"}, 10_000)
	if len(got) != 11 || got[1] != 350 {
		t.Fatalf("double: %v", got)
	}
	got, _ = applyGridEdit(beats, GridEdit{Op: "shift", ShiftMs: -200}, 10_000)
	if got[0] != 400 || len(got) != 5 { // 100-200 falls off the start
		t.Fatalf("shift: %v", got)
	}
	got, _ = applyGridEdit(nil, GridEdit{Op: "grid", BPM: 120, FirstBeatMs: 1250}, 3_000)
	if len(got) != 6 || got[0] != 250 || got[5] != 2750 {
		t.Fatalf("grid: %v", got)
	}
	if _, err := applyGridEdit(beats, GridEdit{Op: "grid", BPM: 5}, 3_000); err == nil {
		t.Fatal("5 BPM accepted")
	}
	if _, err := applyGridEdit(beats, GridEdit{Op: "nope"}, 3_000); err == nil {
		t.Fatal("unknown op accepted")
	}
}
