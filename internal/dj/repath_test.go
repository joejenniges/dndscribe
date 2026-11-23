package dj

import (
	"testing"

	"github.com/joe/dndscribe-go/internal/db"
)

// After a folder rename, queued items from it must point at the new path
// (items capture the path when built), and the queue's folder follows.
func TestRepathFolder(t *testing.T) {
	e, _, _ := newTestEngine()
	in := e.trackItem(db.DJTrack{ID: 1, Path: "Combat/Boss/a.mp3"})
	in.sceneID = 7
	out := e.trackItem(db.DJTrack{ID: 2, Path: "Combatant/b.mp3"})
	f := "Combat"
	e.mu.Lock()
	e.q.replace([]item{in, out}, nil, &f, "Combat", false, false)
	e.mu.Unlock()

	e.RepathFolder("Combat", "Fights")

	e.mu.Lock()
	defer e.mu.Unlock()
	got := map[int64]string{}
	for _, qe := range e.q.entries {
		got[qe.it.track.ID] = qe.it.track.Path
		if qe.it.track.ID == 1 && qe.it.sceneID != 7 {
			t.Errorf("sceneID lost: %d", qe.it.sceneID)
		}
	}
	if got[1] != "Fights/Boss/a.mp3" || got[2] != "Combatant/b.mp3" {
		t.Fatalf("paths = %v", got)
	}
	if e.q.source[0].track.Path != "Fights/Boss/a.mp3" {
		t.Fatalf("source path = %s", e.q.source[0].track.Path)
	}
	if e.q.folder == nil || *e.q.folder != "Fights" || e.q.name != "Fights" {
		t.Fatalf("folder = %v name = %q", e.q.folder, e.q.name)
	}
}
