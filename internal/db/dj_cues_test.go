package db

import (
	"context"
	"os"
	"testing"
	"time"
)

// Runs against a real Postgres when DNDSCRIBE_TEST_DB is set (the dev
// database is fine: it works on one throwaway track row, zz-cue-test/...,
// and deletes it, which cascades its cues and analysis).
func cueTestDB(t *testing.T) (context.Context, int64) {
	t.Helper()
	url := os.Getenv("DNDSCRIBE_TEST_DB")
	if url == "" {
		t.Skip("DNDSCRIBE_TEST_DB not set")
	}
	if pool == nil {
		if err := Init(url); err != nil {
			t.Fatal(err)
		}
	}
	ctx := context.Background()
	path := "zz-cue-test/" + t.Name() + ".mp3"
	id, err := UpsertDJTrack(ctx, DJTrack{Path: path, Title: "zz cue test", DurationMs: 200_000, SizeBytes: 123, ModTime: time.Unix(1700000000, 0)})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := DeleteDJTrackPaths(ctx, []string{path}); err != nil {
			t.Errorf("cleanup: %v", err)
		}
	})
	return ctx, id
}

func TestHotcuesPersistAndDeleteDisablesGotos(t *testing.T) {
	ctx, id := cueTestDB(t)
	for slot := 1; slot <= HotcueSlots; slot++ {
		if err := SetDJHotcue(ctx, id, DJHotcue{Slot: slot, PositionMs: int64(slot) * 10_000, Label: "c"}); err != nil {
			t.Fatal(err)
		}
	}
	if err := SetDJHotcue(ctx, id, DJHotcue{Slot: 1, PositionMs: 12_000, Label: "intro"}); err != nil { // move
		t.Fatal(err)
	}
	if err := SetDJHotcue(ctx, id, DJHotcue{Slot: 9, PositionMs: 1}); err == nil {
		t.Fatal("slot 9 stored")
	}
	g, err := CreateDJGoto(ctx, id, DJGoto{PositionMs: 188_000, TargetSlot: 1, Enabled: true, Label: "G1"})
	if err != nil {
		t.Fatal(err)
	}
	c, err := GetDJTrackCues(ctx, id)
	if err != nil {
		t.Fatal(err)
	}
	if len(c.Hotcues) != 8 || c.Hotcues[0].PositionMs != 12_000 || c.Hotcues[0].Label != "intro" {
		t.Fatalf("hotcues %+v", c.Hotcues)
	}
	if len(c.Gotos) != 1 || !c.Gotos[0].Enabled || c.Gotos[0].ID != g.ID {
		t.Fatalf("gotos %+v", c.Gotos)
	}

	if ok, err := SetDJGotoEnabled(ctx, id, g.ID, false); err != nil || !ok {
		t.Fatalf("disarm: %v %v", ok, err)
	}
	if ok, _ := SetDJGotoEnabled(ctx, id+1_000_000, g.ID, true); ok {
		t.Fatal("goto toggled through the wrong track id")
	}
	_, _ = SetDJGotoEnabled(ctx, id, g.ID, true)

	n, err := DeleteDJHotcue(ctx, id, 1)
	if err != nil || n != 1 {
		t.Fatalf("delete HC1: disabled %d gotos, err %v", n, err)
	}
	c, _ = GetDJTrackCues(ctx, id)
	if len(c.Hotcues) != 7 || c.Gotos[0].Enabled || c.Gotos[0].TargetSlot != 1 {
		t.Fatalf("after deleting HC1: hotcues %d, goto %+v (want disabled, still aimed at 1)", len(c.Hotcues), c.Gotos[0])
	}
}

func TestAnalysisRoundTripAndInvalidation(t *testing.T) {
	ctx, id := cueTestDB(t)
	mt := time.Unix(1700000000, 0)
	bpm, conf := 120.0, 0.8
	beats := []int64{250, 750, 1250, 1751, 2250}
	if err := SaveDJAnalysis(ctx, 123, mt, DJAnalysis{TrackID: id, Version: 1, Status: "ready", Source: "auto", BPM: &bpm, Confidence: &conf, Beats: beats}); err != nil {
		t.Fatal(err)
	}
	a, err := GetDJAnalysis(ctx, id, 123, mt)
	if err != nil || a == nil {
		t.Fatal(a, err)
	}
	if !a.Current || len(a.Beats) != 5 || a.Beats[3] != 1751 || *a.BPM != 120 {
		t.Fatalf("got %+v", a)
	}
	// Same id, different content: stale.
	a, _ = GetDJAnalysis(ctx, id, 124, mt)
	if a.Current {
		t.Fatal("changed file still current")
	}
	need, err := DJTracksNeedingAnalysis(ctx, 1)
	if err != nil {
		t.Fatal(err)
	}
	for _, tr := range need {
		if tr.ID == id {
			t.Fatal("up-to-date track listed for analysis")
		}
	}
	need, _ = DJTracksNeedingAnalysis(ctx, 2) // algorithm version moved on
	found := false
	for _, tr := range need {
		found = found || tr.ID == id
	}
	if !found {
		t.Fatal("old-version analysis not listed")
	}
}
