package transcribe

// StreamResampler converts mono s16le 48kHz audio to mono float32 16kHz one
// chunk at a time, carrying filter state across calls. One instance per user
// stream.
//
// WHY this exists separately from Resample48to16: that function is a stateless
// block FIR — it convolves a whole utterance and zero-pads both edges. Calling
// it independently per 20ms Discord frame would zero-pad every frame boundary,
// injecting a discontinuity (audible click / spectral splatter) ~50 times a
// second and degrading streaming recognition. StreamResampler instead keeps a
// sliding window of past input so the convolution is continuous; its output for
// a signal fed in chunks (then Drain'd) is identical to Resample48to16 over the
// same signal concatenated.
//
// It reuses the same Blackman-windowed sinc kernel (resampleKernel) and 3:1
// decimation (decimationFactor) as the block path, so audio quality matches.
type StreamResampler struct {
	half int // kernel half-width (len(kernel)/2)

	// buf is a sliding window of float32 input samples. buf[0] is absolute
	// input index `base`. Samples no future output needs are trimmed off.
	buf  []float32
	base int

	totalIn int // absolute count of input samples accepted so far
	emitted int // count of output samples produced so far (next output at emitted*decimationFactor)
}

// NewStreamResampler returns a resampler ready to accept 48kHz mono s16le.
func NewStreamResampler() *StreamResampler {
	return &StreamResampler{
		half: len(resampleKernel) / 2,
		buf:  make([]float32, 0, 4096),
	}
}

// Process accepts more 48kHz mono int16 samples and returns the 16kHz float32
// samples that became fully resolvable with the added context. Trailing samples
// near the input frontier are held back until the next Process (or Drain)
// supplies enough forward context for the centered kernel.
func (r *StreamResampler) Process(in []int16) []float32 {
	for _, s := range in {
		r.buf = append(r.buf, float32(s)/32768.0)
	}
	r.totalIn += len(in)

	var out []float32
	for {
		o := r.emitted * decimationFactor
		// Need forward context up to o+half; only emit once it has arrived.
		if o+r.half >= r.totalIn {
			break
		}
		out = append(out, r.computeAt(o))
		r.emitted++
	}
	r.trim()
	return out
}

// Drain emits any remaining outputs at the end of the stream, zero-padding the
// forward edge exactly as the block resampler does. Total outputs over the
// stream equal totalIn/decimationFactor. Call on Flush/stream reset; cheap to
// call when nothing is pending.
func (r *StreamResampler) Drain() []float32 {
	want := r.totalIn / decimationFactor
	var out []float32
	for r.emitted < want {
		o := r.emitted * decimationFactor
		out = append(out, r.computeAt(o))
		r.emitted++
	}
	r.trim()
	return out
}

// Reset clears all state so the resampler can be reused for a fresh utterance.
func (r *StreamResampler) Reset() {
	r.buf = r.buf[:0]
	r.base = 0
	r.totalIn = 0
	r.emitted = 0
}

// computeAt evaluates the centered FIR at absolute input index o, matching
// applyFilter's index math exactly: out = sum_j input[o-j+half]*kernel[j].
// Out-of-range indices (<0 or >=totalIn) are treated as zero, like the block
// path's bounds check.
func (r *StreamResampler) computeAt(o int) float32 {
	var acc float64
	for idx := o - r.half; idx <= o+r.half; idx++ {
		if idx < 0 || idx >= r.totalIn {
			continue
		}
		j := o + r.half - idx
		acc += float64(r.buf[idx-r.base]) * resampleKernel[j]
	}
	return float32(acc)
}

// trim drops leading buffered samples no future output can reference. The next
// output (at emitted*decimationFactor) reaches back to o-half, so anything
// before that is safe to discard.
func (r *StreamResampler) trim() {
	keepFrom := r.emitted*decimationFactor - r.half
	if keepFrom <= r.base {
		return
	}
	drop := keepFrom - r.base
	if drop > len(r.buf) {
		drop = len(r.buf)
	}
	r.buf = append(r.buf[:0], r.buf[drop:]...)
	r.base = keepFrom
}
