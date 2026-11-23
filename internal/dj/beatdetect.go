package dj

import (
	"math"
	"math/cmplx"
)

// Beat detection (v4): estimated tempo, a confidence, and absolute beat
// positions. Used only for drawing beat ticks and snapping cues; hotcues
// and gotos work without it.
//
// Pipeline, on mono audio at beatRate:
//
//  1. Onset strength: log-magnitude spectral flux (512-sample Hann frames,
//     128 hop = 11.6ms), with a moving average subtracted so slow swells
//     don't count, half-wave rectified, normalised.
//  2. Tempo: autocorrelation of the onset strength over 60-200 BPM,
//     weighted by a log-normal prior around 120 BPM (one octave wide) to
//     settle half/double ambiguity toward the usual range.
//  3. Beats: dynamic programming (Ellis 2007, "Beat Tracking by Dynamic
//     Programming"): each beat is placed on strong onsets while paying a
//     penalty for straying from the period. Because each interval is chosen
//     locally, the grid follows tempo drift instead of being BPM + offset.
//  4. Confidence (0..1) from how strongly the onset strength repeats at the
//     chosen period and how much stronger it is on the beats than overall.
//     Ambient and rubato orchestral music scores low; the UI treats that as
//     "beat analysis uncertain" (no ticks drawn by default, no snapping).
//
// It's an estimate. Steady percussive music gets a usable grid; music
// without a pulse gets a low confidence rather than an invented grid.

const (
	beatRate    = 11025
	beatWin     = 512
	beatHop     = 128
	beatMinBPM  = 60.0
	beatMaxBPM  = 200.0
	beatPrior   = 120.0
	tightness   = 100.0
	minBeatSecs = 6.0 // shorter clips aren't analysed
	// BeatConfidenceOK: at or above this the grid is used for snapping and
	// drawn normally. Below it the UI says the analysis is uncertain.
	BeatConfidenceOK = 0.35
	// beatAlgoVersion: bump when the algorithm changes, to re-analyse
	// automatic grids (manual ones are kept).
	beatAlgoVersion = 1
)

type beatResult struct {
	bpm        float64
	confidence float64
	beatsMs    []int64
	// unavailable: no usable rhythm at all (silence, too short).
	unavailable string
}

// fps of the onset envelope.
const onsetFPS = float64(beatRate) / beatHop

// detectBeats analyses mono samples in [-1, 1] at beatRate.
func detectBeats(x []float64) beatResult {
	if float64(len(x))/beatRate < minBeatSecs {
		return beatResult{unavailable: "too short to analyse"}
	}
	env, scale := onsetEnvelope(x)
	if env == nil {
		return beatResult{unavailable: "silent"}
	}
	period, acStrength := estimatePeriod(env)
	if period <= 0 {
		return beatResult{unavailable: "no repeating pulse found"}
	}
	beats := trackBeats(env, period)
	beats = halveIfAlternating(env, beats, period)
	if len(beats) < 4 {
		return beatResult{unavailable: "no repeating pulse found"}
	}

	// Times: frame i covers samples [i*hop, i*hop+win); its centre is where
	// an onset it detects sits on average.
	ms := make([]int64, len(beats))
	for i, f := range beats {
		ms[i] = int64(math.Round((float64(f*beatHop) + beatWin/2) * 1000 / beatRate))
	}
	// Average tempo over the whole grid. Each beat sits on the 11.6ms hop
	// grid, so a single interval (or their median) is quantised; the span
	// over many beats isn't.
	bpm := 60000 * float64(len(ms)-1) / float64(ms[len(ms)-1]-ms[0])

	// Onset strength on the beats vs everywhere.
	var onBeat, all float64
	for _, f := range beats {
		onBeat += env[f]
	}
	onBeat /= float64(len(beats))
	for _, v := range env {
		all += v
	}
	all /= float64(len(env))
	contrast := 0.0
	if all > 0 {
		contrast = onBeat / all // 1 = beats no stronger than anywhere
	}
	// Three 0..1 scores, combined by geometric mean so any one failing
	// sinks the confidence:
	//   c1 periodicity: autocorrelation at the period (clicks ~1, noise ~0)
	//   c2 contrast: onsets on the beats vs everywhere
	//   c3 salience: absolute onset size on the beats; a smooth pad's
	//      periodic FFT ripple passes c1 and c2 but measures ~0.02 here,
	//      noise ~0.04, quiet click tracks ~0.37, loud ones ~1.9.
	c1 := clamp01((acStrength - 0.05) / 0.35)
	c2 := clamp01((contrast - 1.2) / 2.3)
	c3 := clamp01((onBeat*scale - 0.08) / 0.25)
	conf := math.Cbrt(c1 * c2 * c3)
	return beatResult{bpm: math.Round(bpm*100) / 100, confidence: math.Round(conf*1000) / 1000, beatsMs: ms}
}

// halveIfAlternating fixes the usual double-tempo error: a busy hi-hat or
// shaker between the beats makes the onset strength repeat at twice the
// tempo. If every other tracked beat is clearly weaker, the strong ones are
// the beat. Only when the halved tempo stays in range.
func halveIfAlternating(env []float64, beats []int, period float64) []int {
	if len(beats) < 8 || onsetFPS*60/(2*period) < beatMinBPM+10 {
		return beats
	}
	var sum [2]float64
	var n [2]int
	for i, f := range beats {
		sum[i%2] += env[f]
		n[i%2]++
	}
	a, b := sum[0]/float64(n[0]), sum[1]/float64(n[1])
	strong := 0
	if b > a {
		a, b = b, a
		strong = 1
	}
	// 1.4: the off-beats are ~6 dB+ weaker (a hi-hat under a kick measures
	// ~1.6; a mild accent pattern like kick/snare ~1.0-1.2 stays as is).
	if b > 0 && a/b < 1.4 {
		return beats
	}
	out := make([]int, 0, len(beats)/2+1)
	for i := strong; i < len(beats); i += 2 {
		out = append(out, beats[i])
	}
	return out
}

// onsetEnvelope returns the normalised onset strength per hop, or nil for
// silence.
func onsetEnvelope(x []float64) ([]float64, float64) {
	nFrames := (len(x) - beatWin) / beatHop
	if nFrames < 8 {
		return nil, 0
	}
	win := make([]float64, beatWin)
	for i := range win {
		win[i] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/beatWin)
	}
	half := beatWin/2 + 1
	prev := make([]float64, half)
	cur := make([]float64, half)
	buf := make([]complex128, beatWin)
	flux := make([]float64, nFrames)
	peak := 0.0
	for _, v := range x {
		peak = math.Max(peak, math.Abs(v))
	}
	if peak < 1e-4 {
		return nil, 0
	}
	for f := 0; f < nFrames; f++ {
		off := f * beatHop
		for i := 0; i < beatWin; i++ {
			buf[i] = complex(x[off+i]*win[i]/peak, 0)
		}
		fft(buf)
		sum := 0.0
		for k := 0; k < half; k++ {
			cur[k] = math.Log1p(100 * cmplx.Abs(buf[k]))
			if f > 0 {
				if d := cur[k] - prev[k]; d > 0 {
					sum += d
				}
			}
		}
		flux[f] = sum
		prev, cur = cur, prev
	}
	// Remove the local mean (~0.4 s) and keep the rises.
	w := int(math.Round(0.4 * onsetFPS))
	env := make([]float64, nFrames)
	var acc float64
	for i := 0; i < nFrames; i++ {
		acc += flux[i]
		if i >= w {
			acc -= flux[i-w]
		}
		mean := acc / float64(min(i+1, w))
		env[i] = math.Max(0, flux[i]-mean)
	}
	// Normalise to unit standard deviation.
	var m, s float64
	for _, v := range env {
		m += v
	}
	m /= float64(nFrames)
	for _, v := range env {
		s += (v - m) * (v - m)
	}
	s = math.Sqrt(s / float64(nFrames))
	if s < 1e-9 {
		return nil, 0
	}
	for i := range env {
		env[i] /= s
	}
	// scale converts the normalised envelope back to absolute onset size
	// per spectral bin. Everything else here is scale-free, so without it
	// a near-stationary pad, whose tiny frame-to-frame ripple (tones
	// between FFT bins) is periodic, would look like a strong pulse.
	return env, s / float64(half)
}

// estimatePeriod returns the beat period in onset frames (fractional) and
// the normalised autocorrelation strength at it.
func estimatePeriod(env []float64) (float64, float64) {
	n := len(env)
	var mean float64
	for _, v := range env {
		mean += v
	}
	mean /= float64(n)
	z := make([]float64, n)
	for i, v := range env {
		z[i] = v - mean
	}
	ac := func(lag int) float64 {
		s := 0.0
		for i := lag; i < n; i++ {
			s += z[i] * z[i-lag]
		}
		return s / float64(n-lag)
	}
	ac0 := ac(0)
	if ac0 <= 0 {
		return 0, 0
	}
	lo := int(math.Floor(onsetFPS * 60 / beatMaxBPM))
	hi := int(math.Ceil(onsetFPS * 60 / beatMinBPM))
	if hi+2 >= n {
		return 0, 0
	}
	acs := make([]float64, hi+2)
	for l := max(1, lo-1); l <= hi+1; l++ {
		acs[l] = ac(l)
	}
	best, bestScore := 0, math.Inf(-1)
	for l := lo; l <= hi; l++ {
		if acs[l] <= 0 {
			continue
		}
		bpm := onsetFPS * 60 / float64(l)
		oct := math.Log2(bpm / beatPrior)
		w := math.Exp(-0.5 * oct * oct)
		// A true beat period also repeats at twice the lag; reward that a
		// little so a strong off-beat doesn't win.
		score := w * (acs[l] + 0.5*ac(min(2*l, n-1)))
		if score > bestScore {
			best, bestScore = l, score
		}
	}
	if best == 0 {
		return 0, 0
	}
	// Parabolic interpolation around the peak for a fractional period.
	p := float64(best)
	if best > 1 && best < len(acs)-1 {
		a, b, c := acs[best-1], acs[best], acs[best+1]
		if d := a - 2*b + c; d < 0 {
			p += 0.5 * (a - c) / d
		}
	}
	return p, acs[best] / ac0
}

// trackBeats places beats by dynamic programming and returns onset frame
// indices.
func trackBeats(env []float64, period float64) []int {
	n := len(env)
	// Local score: onset strength smoothed by a Gaussian a 32nd of the
	// period wide, as in Ellis's paper.
	sd := period / 32
	rad := int(math.Ceil(4 * sd))
	kern := make([]float64, 2*rad+1)
	for i := range kern {
		d := float64(i-rad) / sd
		kern[i] = math.Exp(-0.5 * d * d)
	}
	local := make([]float64, n)
	for i := 0; i < n; i++ {
		s := 0.0
		for k := -rad; k <= rad; k++ {
			if j := i + k; j >= 0 && j < n {
				s += env[j] * kern[k+rad]
			}
		}
		local[i] = s
	}
	score := make([]float64, n)
	back := make([]int, n)
	minPrev := int(math.Round(period / 2))
	maxPrev := int(math.Round(2 * period))
	for t := 0; t < n; t++ {
		best, from := math.Inf(-1), -1
		for p := t - maxPrev; p <= t-minPrev; p++ {
			if p < 0 {
				continue
			}
			r := math.Log(float64(t-p) / period)
			s := score[p] - tightness*r*r
			if s > best {
				best, from = s, p
			}
		}
		if from < 0 {
			best, from = 0, -1
		}
		score[t] = local[t] + best
		back[t] = from
	}
	// Start from the best-scoring frame in the last period that has a real
	// onset, then follow the back links.
	thresh := 0.0
	for _, v := range local {
		thresh += v * v
	}
	thresh = 0.5 * math.Sqrt(thresh/float64(n))
	end, endScore := -1, math.Inf(-1)
	for t := max(0, n-int(math.Ceil(period))-1); t < n; t++ {
		if local[t] >= thresh && score[t] > endScore {
			end, endScore = t, score[t]
		}
	}
	if end < 0 {
		end = n - 1
	}
	var beats []int
	for t := end; t >= 0; t = back[t] {
		beats = append(beats, t)
		if back[t] < 0 {
			break
		}
	}
	for i, j := 0, len(beats)-1; i < j; i, j = i+1, j-1 {
		beats[i], beats[j] = beats[j], beats[i]
	}
	// Trim weak beats at the very start and end (fade-ins, silence).
	for len(beats) > 0 && local[beats[0]] < thresh {
		beats = beats[1:]
	}
	for len(beats) > 0 && local[beats[len(beats)-1]] < thresh {
		beats = beats[:len(beats)-1]
	}
	return beats
}

// fft is an in-place radix-2 Cooley-Tukey FFT (len must be a power of 2).
func fft(a []complex128) {
	n := len(a)
	for i, j := 1, 0; i < n; i++ {
		bit := n >> 1
		for ; j&bit != 0; bit >>= 1 {
			j ^= bit
		}
		j ^= bit
		if i < j {
			a[i], a[j] = a[j], a[i]
		}
	}
	for size := 2; size <= n; size <<= 1 {
		ang := -2 * math.Pi / float64(size)
		wn := complex(math.Cos(ang), math.Sin(ang))
		for start := 0; start < n; start += size {
			w := complex(1, 0)
			for k := 0; k < size/2; k++ {
				u := a[start+k]
				v := a[start+k+size/2] * w
				a[start+k] = u + v
				a[start+k+size/2] = u - v
				w *= wn
			}
		}
	}
}
