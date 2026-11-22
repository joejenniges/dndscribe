package dj

import (
	"context"
	"os/exec"
	"path/filepath"
	"testing"
)

// TestAnalyse needs ffmpeg on PATH; skipped otherwise. One second of silence
// then one second of a quiet tone: the bins should show both halves, and the
// tone should be normalised up to full height.
func TestAnalyse(t *testing.T) {
	ff, err := exec.LookPath("ffmpeg")
	if err != nil {
		t.Skip("ffmpeg not on PATH")
	}
	file := filepath.Join(t.TempDir(), "t.wav")
	gen := exec.Command(ff, "-hide_banner", "-loglevel", "error", "-f", "lavfi",
		"-i", `aevalsrc=if(gte(t\,1)\,0.25*sin(2*PI*440*t)\,0):d=2:s=48000`, file)
	if out, err := gen.CombinedOutput(); err != nil {
		t.Fatalf("generate: %v\n%s", err, out)
	}
	data, err := analyse(context.Background(), ff, file, waveDecodeRate/waveBinsPerSecond)
	if err != nil {
		t.Fatal(err)
	}
	bins := len(data) / 2
	if bins < 195 || bins > 205 {
		t.Fatalf("%d bins for 2s, want about 200", bins)
	}
	for b := 5; b < 95; b++ {
		if data[b*2] > 3 {
			t.Fatalf("bin %d in the silent half has peak %d", b, data[b*2])
		}
	}
	for b := 105; b < 195; b++ {
		pk, rms := data[b*2], data[b*2+1]
		if pk < 240 {
			t.Fatalf("bin %d: peak %d, want near 255 after normalising", b, pk)
		}
		if rms < 165 || rms > 195 { // sine RMS = peak / sqrt 2 = ~180
			t.Fatalf("bin %d: rms %d, want about 180", b, rms)
		}
	}
}
