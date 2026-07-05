package transcribe

import (
	"encoding/binary"
	"math"
)

const (
	inputRate        = 48000
	outputRate       = 16000
	decimationFactor = inputRate / outputRate // 3
	filterTaps       = 32
	cutoff           = float64(outputRate/2) / float64(inputRate) // 8000/48000
)

// resampleKernel is precomputed at init time.
var resampleKernel []float64

func init() {
	resampleKernel = buildSincKernel(filterTaps, cutoff)
}

// Resample48to16 converts 48kHz mono int16 PCM (little-endian bytes) to
// 16kHz float32 samples in the range [-1.0, 1.0].
//
// Uses a 32-tap Blackman-windowed sinc low-pass filter before 3:1 decimation,
// matching the approach in discord-rpg-summariser.
func Resample48to16(pcm []byte) []float32 {
	numSamples := len(pcm) / 2
	if numSamples == 0 {
		return nil
	}

	// Convert int16 PCM to float32.
	samples := make([]float32, numSamples)
	for i := 0; i < numSamples; i++ {
		s := int16(binary.LittleEndian.Uint16(pcm[i*2 : i*2+2]))
		samples[i] = float32(s) / 32768.0
	}

	// Apply low-pass filter then decimate.
	filtered := applyFilter(samples, resampleKernel)
	return decimate(filtered, decimationFactor)
}

// buildSincKernel generates a Blackman-windowed sinc low-pass FIR filter.
func buildSincKernel(taps int, fc float64) []float64 {
	kernel := make([]float64, taps+1)
	mid := float64(taps) / 2.0
	sum := 0.0

	for i := 0; i <= taps; i++ {
		x := float64(i) - mid
		var sinc float64
		if x == 0 {
			sinc = 2.0 * math.Pi * fc
		} else {
			sinc = math.Sin(2.0*math.Pi*fc*x) / x
		}
		// Blackman window.
		w := 0.42 - 0.5*math.Cos(2.0*math.Pi*float64(i)/float64(taps)) +
			0.08*math.Cos(4.0*math.Pi*float64(i)/float64(taps))
		kernel[i] = sinc * w
		sum += kernel[i]
	}

	// Normalise for unity DC gain.
	for i := range kernel {
		kernel[i] /= sum
	}
	return kernel
}

// applyFilter convolves the input with the FIR kernel.
func applyFilter(samples []float32, kernel []float64) []float32 {
	n := len(samples)
	kLen := len(kernel)
	out := make([]float32, n)

	for i := 0; i < n; i++ {
		var acc float64
		for j := 0; j < kLen; j++ {
			idx := i - j + kLen/2
			if idx >= 0 && idx < n {
				acc += float64(samples[idx]) * kernel[j]
			}
		}
		out[i] = float32(acc)
	}
	return out
}

// decimate picks every factor-th sample.
func decimate(samples []float32, factor int) []float32 {
	outLen := len(samples) / factor
	out := make([]float32, outLen)
	for i := 0; i < outLen; i++ {
		out[i] = samples[i*factor]
	}
	return out
}
