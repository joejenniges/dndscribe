package audio

import "math"

// DownmixStereoToMono converts interleaved stereo s16le samples to mono s16le
// by averaging left and right channels.
func DownmixStereoToMono(stereo []int16) []int16 {
	return DownmixStereoToMonoInto(nil, stereo)
}

// DownmixStereoToMonoInto downmixes into dst (reused across calls to avoid
// per-frame allocation on the hot per-packet streaming path). dst is grown as
// needed; pass nil for a freshly allocated result. Returns the filled slice.
func DownmixStereoToMonoInto(dst, stereo []int16) []int16 {
	monoLen := len(stereo) / 2
	if cap(dst) < monoLen {
		dst = make([]int16, monoLen)
	}
	dst = dst[:monoLen]
	for i := 0; i < monoLen; i++ {
		left := int32(stereo[i*2])
		right := int32(stereo[i*2+1])
		avg := (left + right) / 2
		// Clamp to int16 range
		if avg > 32767 {
			avg = 32767
		} else if avg < -32768 {
			avg = -32768
		}
		dst[i] = int16(avg)
	}
	return dst
}

// ComputeRMS computes the root mean square of mono s16le samples.
// Samples every 10th sample for performance.
func ComputeRMS(mono []int16) float32 {
	if len(mono) == 0 {
		return 0
	}

	var sumSquares float64
	var count int
	for i := 0; i < len(mono); i += 10 {
		s := float64(mono[i])
		sumSquares += s * s
		count++
	}
	if count == 0 {
		return 0
	}
	return float32(math.Sqrt(sumSquares / float64(count)))
}
