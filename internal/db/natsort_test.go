package db

import (
	"sort"
	"testing"
)

func TestNaturalLess(t *testing.T) {
	in := []string{
		"Boss/Boss Fight 10.mp3", "Boss/Boss Fight 2.mp3", "Boss/Boss Fight 1.mp3",
		"Boss/boss fight 11.mp3", "Boss/Boss Fight 20.mp3", "Boss/Boss Fight 19.mp3",
		"Boss Extra/a.mp3", "Boss/Boss Fight 02.mp3", "Ambience/x.mp3",
	}
	want := []string{
		"Ambience/x.mp3",
		"Boss/Boss Fight 1.mp3", "Boss/Boss Fight 2.mp3", "Boss/Boss Fight 02.mp3",
		"Boss/Boss Fight 10.mp3", "Boss/boss fight 11.mp3", "Boss/Boss Fight 19.mp3", "Boss/Boss Fight 20.mp3",
		"Boss Extra/a.mp3",
	}
	sort.SliceStable(in, func(i, j int) bool { return NaturalLess(in[i], in[j]) })
	for i := range want {
		if in[i] != want[i] {
			t.Fatalf("order:\n got %q\nwant %q", in, want)
		}
	}
	for _, s := range want {
		if NaturalLess(s, s) {
			t.Fatalf("NaturalLess(%q, %q) = true", s, s)
		}
	}
}
