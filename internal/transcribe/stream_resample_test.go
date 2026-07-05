package transcribe

import (
	"encoding/binary"
	"math"
	"testing"
)

// int16ToBytes mirrors the s16le byte layout Resample48to16 expects.
func int16ToBytes(s []int16) []byte {
	b := make([]byte, len(s)*2)
	for i, v := range s {
		binary.LittleEndian.PutUint16(b[i*2:], uint16(v))
	}
	return b
}

// sine48k generates n samples of a tone at freq Hz, 48kHz, int16.
func sine48k(n int, freq float64) []int16 {
	out := make([]int16, n)
	for i := 0; i < n; i++ {
		out[i] = int16(0.6 * 32767 * math.Sin(2*math.Pi*freq*float64(i)/inputRate))
	}
	return out
}

func maxAbsDiff(a, b []float32) (idx int, diff float64) {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	worst := -1.0
	for i := 0; i < n; i++ {
		d := math.Abs(float64(a[i]) - float64(b[i]))
		if d > worst {
			worst, idx = d, i
		}
	}
	return idx, worst
}

// TestStreamResamplerMatchesBlock: feeding the signal in chunks then draining
// must reproduce the stateless block resampler over the concatenated signal.
// This is the core correctness + continuity guarantee — if chunk boundaries
// injected discontinuities, the outputs would diverge.
func TestStreamResamplerMatchesBlock(t *testing.T) {
	in := sine48k(48000, 440) // 1s @ 48k
	want := Resample48to16(int16ToBytes(in))

	// Feed in irregular chunk sizes (not multiples of the decimation factor) to
	// stress the cross-call boundary handling.
	chunks := []int{960, 1, 7, 1920, 333, 480, 2, 5000}
	r := NewStreamResampler()
	var got []float32
	pos := 0
	ci := 0
	for pos < len(in) {
		sz := chunks[ci%len(chunks)]
		ci++
		end := pos + sz
		if end > len(in) {
			end = len(in)
		}
		got = append(got, r.Process(in[pos:end])...)
		pos = end
	}
	got = append(got, r.Drain()...)

	if len(got) != len(want) {
		t.Fatalf("length mismatch: streaming=%d block=%d", len(got), len(want))
	}
	idx, diff := maxAbsDiff(got, want)
	if diff > 1e-5 {
		t.Fatalf("streaming output diverges from block at sample %d: diff=%g", idx, diff)
	}
}

// TestStreamResamplerChunkInvariant: output must not depend on how the input is
// chunked. Sample-by-sample feeding and whole-buffer feeding must agree.
func TestStreamResamplerChunkInvariant(t *testing.T) {
	in := sine48k(24000, 1000)

	whole := NewStreamResampler()
	a := append(whole.Process(in), whole.Drain()...)

	single := NewStreamResampler()
	var b []float32
	for i := range in {
		b = append(b, single.Process(in[i:i+1])...)
	}
	b = append(b, single.Drain()...)

	if len(a) != len(b) {
		t.Fatalf("length mismatch: whole=%d single=%d", len(a), len(b))
	}
	if idx, diff := maxAbsDiff(a, b); diff > 1e-6 {
		t.Fatalf("chunking changed output at %d: diff=%g", idx, diff)
	}
}

// TestStreamResamplerRatio: 48k->16k is 3:1, so output length is input/3.
func TestStreamResamplerRatio(t *testing.T) {
	in := sine48k(30003, 300)
	r := NewStreamResampler()
	out := append(r.Process(in), r.Drain()...)
	want := len(in) / decimationFactor
	if len(out) != want {
		t.Fatalf("ratio wrong: got %d outputs for %d inputs, want %d", len(out), len(in), want)
	}
}

// TestStreamResamplerReset: a reset instance behaves like a fresh one.
func TestStreamResamplerReset(t *testing.T) {
	in := sine48k(9600, 500)

	fresh := NewStreamResampler()
	want := append(fresh.Process(in), fresh.Drain()...)

	reused := NewStreamResampler()
	reused.Process(sine48k(5000, 123))
	reused.Drain()
	reused.Reset()
	got := append(reused.Process(in), reused.Drain()...)

	if len(got) != len(want) {
		t.Fatalf("length mismatch after reset: got=%d want=%d", len(got), len(want))
	}
	if idx, diff := maxAbsDiff(got, want); diff > 1e-6 {
		t.Fatalf("reset instance diverged at %d: diff=%g", idx, diff)
	}
}
