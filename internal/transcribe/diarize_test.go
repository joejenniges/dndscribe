package transcribe

import (
	"math"
	"testing"
)

func TestCosineSim(t *testing.T) {
	cases := []struct {
		name string
		a, b []float32
		want float32
	}{
		{"identical", []float32{1, 0, 0}, []float32{1, 0, 0}, 1},
		{"orthogonal", []float32{1, 0, 0}, []float32{0, 1, 0}, 0},
		{"opposite", []float32{1, 0}, []float32{-1, 0}, -1},
		{"scale invariant", []float32{2, 2}, []float32{5, 5}, 1},
		{"length mismatch", []float32{1}, []float32{1, 2}, -1},
		{"zero vector", []float32{0, 0}, []float32{1, 0}, -1},
	}
	for _, c := range cases {
		if got := cosineSim(c.a, c.b); math.Abs(float64(got-c.want)) > 1e-6 {
			t.Errorf("%s: cosineSim = %v, want %v", c.name, got, c.want)
		}
	}
}

func TestL2Normalize(t *testing.T) {
	v := l2normalize([]float32{3, 4})
	if math.Abs(float64(v[0]-0.6)) > 1e-6 || math.Abs(float64(v[1]-0.8)) > 1e-6 {
		t.Errorf("l2normalize = %v, want [0.6 0.8]", v)
	}
	// Zero vector passes through without NaN.
	z := l2normalize([]float32{0, 0})
	if z[0] != 0 || z[1] != 0 {
		t.Errorf("l2normalize zero = %v", z)
	}
}

func TestBestMatch(t *testing.T) {
	p1 := &speakerProfile{id: 1, label: "S1", centroid: l2normalize([]float32{1, 0, 0})}
	p2 := &speakerProfile{id: 2, label: "S2", centroid: l2normalize([]float32{0, 1, 0})}

	best, sim := bestMatch([]*speakerProfile{p1, p2}, l2normalize([]float32{0.9, 0.1, 0}))
	if best == nil || best.id != 1 {
		t.Fatalf("bestMatch picked %+v, want S1", best)
	}
	if sim < 0.9 {
		t.Errorf("similarity = %v, want >0.9", sim)
	}

	best, sim = bestMatch(nil, []float32{1, 0, 0})
	if best != nil || sim != -1 {
		t.Errorf("empty profiles: got %+v sim=%v, want nil/-1", best, sim)
	}
}

func TestUpdateCentroid(t *testing.T) {
	// Centroid of 1 sample at [1,0], new sample [0,1] -> mean [0.5,0.5],
	// normalized to [1/sqrt2, 1/sqrt2].
	c := updateCentroid([]float32{1, 0}, []float32{0, 1}, 1)
	want := float32(1 / math.Sqrt2)
	if math.Abs(float64(c[0]-want)) > 1e-6 || math.Abs(float64(c[1]-want)) > 1e-6 {
		t.Errorf("updateCentroid = %v, want [%v %v]", c, want, want)
	}

	// Heavy prior (n=99) barely moves toward the newcomer.
	c = updateCentroid([]float32{1, 0}, []float32{0, 1}, 99)
	if c[0] < 0.99 {
		t.Errorf("heavy prior moved too much: %v", c)
	}

	// Mismatched lengths return the original centroid untouched.
	orig := []float32{1, 0}
	if got := updateCentroid(orig, []float32{1}, 1); &got[0] != &orig[0] {
		t.Errorf("length mismatch should return original slice")
	}
}
