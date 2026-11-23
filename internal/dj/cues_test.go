package dj

import (
	"sync"
	"testing"
	"time"

	"github.com/joe/dndscribe-go/internal/db"
)

// cueTrack is a 60s seekable track whose decoders are counted by start
// position. gate, when set, blocks openAt until closed (a slow disk).
type cueTrack struct {
	mu     sync.Mutex
	opened map[int64]int
	gate   chan struct{}
}

func (c *cueTrack) count(ms int64) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.opened[ms]
}

func newCueItem(id int64, durMs int64) (item, *cueTrack) {
	ct := &cueTrack{opened: map[int64]int{}}
	frames := int(durMs / FrameMs)
	return item{
		track: &db.DJTrack{ID: id, Title: "cue track", DurationMs: durMs},
		label: "cue track",
		scale: 1,
		open:  func() (Source, error) { return &constSource{val: 1000, n: frames}, nil },
		openAt: func(ms int64) (Source, error) {
			ct.mu.Lock()
			g := ct.gate
			ct.mu.Unlock()
			if g != nil {
				<-g
			}
			ct.mu.Lock()
			ct.opened[ms]++
			ct.mu.Unlock()
			return &constSource{val: 2000, n: frames - int(ms/FrameMs)}, nil
		},
	}, ct
}

func cuesFor(id int64, hot map[int]int64, gotos ...db.DJGoto) *db.DJTrackCues {
	c := &db.DJTrackCues{TrackID: id, Hotcues: []db.DJHotcue{}, Gotos: gotos}
	for slot := 1; slot <= 8; slot++ {
		if p, ok := hot[slot]; ok {
			c.Hotcues = append(c.Hotcues, db.DJHotcue{Slot: slot, PositionMs: p})
		}
	}
	return c
}

// waitPrep waits until deck i has a decoder ready at posMs.
func waitPrep(t *testing.T, e *Engine, i int, posMs int64) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		e.mu.Lock()
		p := e.decks[i].prep
		ok := p != nil && p.src != nil && p.posMs == posMs
		e.mu.Unlock()
		if ok {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("deck %d: no decoder prepared at %d", i, posMs)
}

// waitSeek waits for background seeks/loads to settle (seekAsync).
func waitPos(t *testing.T, e *Engine, i int, want int64) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if e.State().Decks[i].PositionMs == want {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("deck %d position %d, want %d", i, e.State().Decks[i].PositionMs, want)
}

func pos(e *Engine, i int) int64 { return e.State().Decks[i].PositionMs }

// startAt puts it on deck 0 playing, positioned at ms.
func startAt(t *testing.T, e *Engine, it item, ms int64) {
	t.Helper()
	if err := e.startNow(it, 0, "track"); err != nil {
		t.Fatal(err)
	}
	if ms > 0 {
		if err := e.SeekDeck(0, ms); err != nil {
			t.Fatal(err)
		}
	}
}

// The spec's example scaled down: HC1 at 12s, G1 at 18.8s targeting it.
const (
	hc1Ms = 12_000
	g1Ms  = 18_800
)

func TestGotoLoopsSeamlesslyForManyLaps(t *testing.T) {
	e, out, _ := newTestEngine()
	it, ct := newCueItem(101, 60_000)
	e.SetTrackCues(cuesFor(101, map[int]int64{1: hc1Ms}, db.DJGoto{ID: 1, PositionMs: g1Ms, TargetSlot: 1, Enabled: true}))
	startAt(t, e, it, 18_000)

	frames(e, out, 1) // prepares the target decoder
	waitPrep(t, e, 0, hc1Ms)
	frames(e, out, 39) // 18000+40 frames*20 = 18800: the 40th frame crosses
	if got := pos(e, 0); got != hc1Ms {
		t.Fatalf("after crossing G1: position %d, want %d", got, hc1Ms)
	}
	j := e.State().Decks[0].LastJump
	if j == nil || j.Kind != "goto" || j.GotoID != 1 || !j.Seamless || j.ToMs != hc1Ms {
		t.Fatalf("last jump %+v", j)
	}

	// Three more laps of (18800-12000)/20 = 340 frames each.
	lap := int((g1Ms - hc1Ms) / FrameMs)
	for l := 0; l < 3; l++ {
		frames(e, out, 1)
		waitPrep(t, e, 0, hc1Ms)
		frames(e, out, lap-1)
		if got := pos(e, 0); got != hc1Ms {
			t.Fatalf("lap %d: position %d, want %d", l+2, got, hc1Ms)
		}
	}
	// Every jump came from a prepared decoder: one per lap plus the one
	// being prepared now.
	frames(e, out, 1)
	waitPrep(t, e, 0, hc1Ms)
	if n := ct.count(hc1Ms); n != 5 {
		t.Fatalf("decoders opened at HC1: %d, want 5", n)
	}
}

func TestGotoDisabledPlaysThrough(t *testing.T) {
	e, out, _ := newTestEngine()
	it, _ := newCueItem(102, 60_000)
	e.SetTrackCues(cuesFor(102, map[int]int64{1: hc1Ms}, db.DJGoto{ID: 2, PositionMs: g1Ms, TargetSlot: 1, Enabled: false}))
	startAt(t, e, it, 18_000)
	frames(e, out, 60)
	if got := pos(e, 0); got != 18_000+60*FrameMs {
		t.Fatalf("position %d, want %d (no jump)", got, 18_000+60*FrameMs)
	}
	if e.State().Decks[0].LastJump != nil {
		t.Fatal("a disabled goto jumped")
	}
	e.mu.Lock()
	prep := e.decks[0].prep
	e.mu.Unlock()
	if prep != nil {
		t.Fatal("decoder prepared for a disabled goto")
	}
}

func TestGotoReArmedWhilePlayingTakesEffect(t *testing.T) {
	e, out, _ := newTestEngine()
	it, _ := newCueItem(103, 60_000)
	off := cuesFor(103, map[int]int64{1: hc1Ms}, db.DJGoto{ID: 3, PositionMs: g1Ms, TargetSlot: 1, Enabled: false})
	e.SetTrackCues(off)
	startAt(t, e, it, 18_000)
	frames(e, out, 10)
	on := cuesFor(103, map[int]int64{1: hc1Ms}, db.DJGoto{ID: 3, PositionMs: g1Ms, TargetSlot: 1, Enabled: true})
	e.SetTrackCues(on) // a tablet arms it mid-track
	frames(e, out, 1)
	waitPrep(t, e, 0, hc1Ms)
	frames(e, out, 29)
	if got := pos(e, 0); got != hc1Ms {
		t.Fatalf("position %d, want %d", got, hc1Ms)
	}
}

func TestSeekOverGotoDoesNotFire(t *testing.T) {
	e, out, _ := newTestEngine()
	it, _ := newCueItem(104, 60_000)
	e.SetTrackCues(cuesFor(104, map[int]int64{1: hc1Ms}, db.DJGoto{ID: 4, PositionMs: g1Ms, TargetSlot: 1, Enabled: true}))
	startAt(t, e, it, 5_000)
	frames(e, out, 5)
	if err := e.SeekDeck(0, 19_000); err != nil { // jumps past G1
		t.Fatal(err)
	}
	frames(e, out, 10)
	if got := pos(e, 0); got != 19_000+10*FrameMs {
		t.Fatalf("position %d after seeking past G1, want %d", got, 19_000+10*FrameMs)
	}
	if e.State().Decks[0].LastJump != nil {
		t.Fatal("skipped-over goto fired")
	}
}

func TestSeekExactlyOntoGotoWaitsForNextCrossing(t *testing.T) {
	e, out, _ := newTestEngine()
	it, _ := newCueItem(105, 60_000)
	e.SetTrackCues(cuesFor(105, map[int]int64{1: hc1Ms}, db.DJGoto{ID: 5, PositionMs: g1Ms, TargetSlot: 1, Enabled: true}))
	startAt(t, e, it, g1Ms)
	frames(e, out, 5)
	if got := pos(e, 0); got != g1Ms+5*FrameMs {
		t.Fatalf("position %d, want %d: landing on a goto must not fire it", got, g1Ms+5*FrameMs)
	}
	// From before it, the next natural crossing does fire.
	_ = e.SeekDeck(0, g1Ms-200)
	frames(e, out, 1)
	waitPrep(t, e, 0, hc1Ms)
	frames(e, out, 9)
	if got := pos(e, 0); got != hc1Ms {
		t.Fatalf("position %d, want %d", got, hc1Ms)
	}
}

func TestGotoTargetMissingFailsOpen(t *testing.T) {
	e, out, _ := newTestEngine()
	it, _ := newCueItem(106, 60_000)
	e.SetTrackCues(cuesFor(106, map[int]int64{}, db.DJGoto{ID: 6, PositionMs: g1Ms, TargetSlot: 1, Enabled: true}))
	startAt(t, e, it, 18_000)
	frames(e, out, 60)
	if got := pos(e, 0); got != 18_000+60*FrameMs {
		t.Fatalf("position %d: a goto with no target must not jump", got)
	}
	g := e.State().Decks[0].Gotos
	if len(g) != 1 || g[0].Problem != "target missing" {
		t.Fatalf("goto state %+v, want problem 'target missing'", g)
	}
}

func TestGotoTooCloseIsRejectedAndIgnored(t *testing.T) {
	c := cuesFor(107, map[int]int64{1: 10_000})
	if err := ValidateGoto(c, db.DJGoto{PositionMs: 10_300, TargetSlot: 1}, 60_000); err == nil {
		t.Fatal("goto 300ms after its target accepted")
	}
	if err := ValidateGoto(c, db.DJGoto{PositionMs: 10_000, TargetSlot: 1}, 60_000); err == nil {
		t.Fatal("goto on its own target accepted")
	}
	if err := ValidateGoto(c, db.DJGoto{PositionMs: 9_000, TargetSlot: 1}, 60_000); err != nil {
		t.Fatalf("forward jump rejected: %v", err)
	}
	if err := ValidateGoto(c, db.DJGoto{PositionMs: 61_000, TargetSlot: 1}, 60_000); err == nil {
		t.Fatal("goto past the end accepted")
	}

	// A hotcue moved onto its goto after the goto was saved: ignored at
	// runtime (no 20ms loop), reported in state.
	e, out, _ := newTestEngine()
	it, _ := newCueItem(107, 60_000)
	e.SetTrackCues(cuesFor(107, map[int]int64{1: g1Ms - 100}, db.DJGoto{ID: 7, PositionMs: g1Ms, TargetSlot: 1, Enabled: true}))
	startAt(t, e, it, 18_000)
	frames(e, out, 60)
	if got := pos(e, 0); got != 18_000+60*FrameMs {
		t.Fatalf("position %d: an invalid goto jumped", got)
	}
	if p := e.State().Decks[0].Gotos[0].Problem; p != "too close to its target" {
		t.Fatalf("problem %q", p)
	}
}

func TestForwardGotoSkipsAhead(t *testing.T) {
	e, out, _ := newTestEngine()
	it, _ := newCueItem(108, 60_000)
	// G at 10s jumps forward to HC2 at 40s; a second goto in between is
	// skipped over by the jump and never fires.
	e.SetTrackCues(cuesFor(108, map[int]int64{1: 5_000, 2: 40_000},
		db.DJGoto{ID: 8, PositionMs: 10_000, TargetSlot: 2, Enabled: true},
		db.DJGoto{ID: 9, PositionMs: 20_000, TargetSlot: 1, Enabled: true}))
	startAt(t, e, it, 9_000)
	frames(e, out, 1)
	waitPrep(t, e, 0, 40_000)
	frames(e, out, 49)
	if got := pos(e, 0); got != 40_000 {
		t.Fatalf("position %d, want 40000", got)
	}
	frames(e, out, 20)
	if got := pos(e, 0); got != 40_000+20*FrameMs {
		t.Fatalf("position %d after the forward jump", got)
	}
}

func TestPausedAtGotoFiresAfterResume(t *testing.T) {
	e, out, _ := newTestEngine()
	it, _ := newCueItem(109, 60_000)
	e.SetTrackCues(cuesFor(109, map[int]int64{1: hc1Ms}, db.DJGoto{ID: 10, PositionMs: g1Ms, TargetSlot: 1, Enabled: true}))
	startAt(t, e, it, 18_700)
	frames(e, out, 1)
	waitPrep(t, e, 0, hc1Ms)
	frames(e, out, 3) // 18780
	if err := e.PauseDeck(0); err != nil {
		t.Fatal(err)
	}
	frames(e, out, 50) // the 200ms pause ramp plays a few frames out, then stops
	p := pos(e, 0)
	if p > g1Ms {
		// The ramp itself carried it over G1: then it must have jumped.
		if p >= g1Ms+FrameMs && e.State().Decks[0].LastJump == nil {
			t.Fatalf("crossed G1 during the pause ramp without jumping (pos %d)", p)
		}
		return
	}
	frames(e, out, 50)
	if pos(e, 0) != p {
		t.Fatal("paused deck advanced")
	}
	if err := e.PlayDeck(0); err != nil {
		t.Fatal(err)
	}
	frames(e, out, 10)
	if j := e.State().Decks[0].LastJump; j == nil || j.Kind != "goto" {
		t.Fatalf("no goto after resuming before G1 (pos was %d): %+v", p, j)
	}
}

func TestGotoFallsBackToSeekWhenNotPrepared(t *testing.T) {
	e, out, _ := newTestEngine()
	it, ct := newCueItem(110, 60_000)
	e.SetTrackCues(cuesFor(110, map[int]int64{1: hc1Ms}, db.DJGoto{ID: 11, PositionMs: g1Ms, TargetSlot: 1, Enabled: true}))
	startAt(t, e, it, 18_700)
	gate := make(chan struct{})
	ct.mu.Lock()
	ct.gate = gate // every decoder open now stalls
	ct.mu.Unlock()
	frames(e, out, 6) // crosses G1 at the 5th frame with nothing prepared
	j := e.State().Decks[0].LastJump
	if j == nil || j.Seamless {
		t.Fatalf("expected a non-seamless jump, got %+v", j)
	}
	close(gate)
	waitPos(t, e, 0, hc1Ms)
}

func TestHotcueJumpOnlyMovesItsDeck(t *testing.T) {
	e, out, _ := newTestEngine()
	a, _ := newCueItem(111, 60_000)
	b, _ := newCueItem(111, 60_000) // the same track on both decks
	e.SetTrackCues(cuesFor(111, map[int]int64{1: hc1Ms, 3: 30_000}))
	if err := e.openOnDeck(0, &a); err != nil {
		t.Fatal(err)
	}
	if err := e.openOnDeck(1, &b); err != nil {
		t.Fatal(err)
	}
	_ = e.PlayDeck(0)
	_ = e.PlayDeck(1)
	frames(e, out, 5)
	if err := e.HotcueDeck(1, 3); err != nil {
		t.Fatal(err)
	}
	waitPos(t, e, 1, 30_000)
	if got := pos(e, 0); got != 5*FrameMs {
		t.Fatalf("deck A moved to %d by deck B's hotcue", got)
	}
	if err := e.HotcueDeck(0, 2); err == nil {
		t.Fatal("empty hotcue slot jumped")
	}
	if err := e.HotcueDeck(0, 9); err == nil {
		t.Fatal("slot 9 accepted")
	}
	// Paused deck: jumps, stays paused.
	_ = e.PauseDeck(1)
	frames(e, out, 20)
	if err := e.HotcueDeck(1, 1); err != nil {
		t.Fatal(err)
	}
	waitPos(t, e, 1, hc1Ms)
	if e.State().Decks[1].Playing {
		t.Fatal("hotcue started a paused deck")
	}
}

func TestGotoSuppressedOnDeckBeingFadedOut(t *testing.T) {
	e, out, _ := newTestEngine()
	it, _ := newCueItem(112, 60_000)
	e.SetTrackCues(cuesFor(112, map[int]int64{1: hc1Ms}, db.DJGoto{ID: 12, PositionMs: g1Ms, TargetSlot: 1, Enabled: true}))
	startAt(t, e, it, 18_000)
	frames(e, out, 1)
	// A scene change starts a 2s crossfade to deck B.
	next, _ := constItem("next", 500, 1000, 20_000)
	if err := e.startNow(next, 2, "scene"); err != nil {
		t.Fatal(err)
	}
	frames(e, out, 45) // crosses G1 mid-fade
	if j := e.State().Decks[0].LastJump; j != nil {
		t.Fatalf("goto fired on the deck being faded out: %+v", j)
	}
}

func TestArmedLoopHoldsOffAutoDJ(t *testing.T) {
	e, out, _ := newTestEngine()
	e.SetFadeSeconds(1)
	it, _ := newCueItem(113, 20_000)
	// Loop 15s -> 5s, well before the track's end.
	e.SetTrackCues(cuesFor(113, map[int]int64{1: 5_000}, db.DJGoto{ID: 13, PositionMs: 15_000, TargetSlot: 1, Enabled: true}))
	startAt(t, e, it, 14_000)
	n1, _ := constItem("q1", 300, 2000, 40_000)
	e.mu.Lock()
	e.q.replace([]item{n1}, nil, nil, "test", false, false)
	e.mu.Unlock()
	for i := 0; i < 6; i++ { // 6 x 10s of playback
		frames(e, out, 1)
		settle(t, e)
		waitPrep(t, e, 0, 5_000)
		frames(e, out, 499)
	}
	st := e.State()
	if st.Transition != nil || st.OnAir != 0 || !st.Decks[0].Playing {
		t.Fatalf("Auto DJ moved on from an armed loop: transition=%+v onAir=%d", st.Transition, st.OnAir)
	}
	// Disarm: the track now runs to its end and Auto DJ takes over.
	e.SetTrackCues(cuesFor(113, map[int]int64{1: 5_000}, db.DJGoto{ID: 13, PositionMs: 15_000, TargetSlot: 1, Enabled: false}))
	frames(e, out, 1000)
	settle(t, e)
	if st := e.State(); st.OnAir != 1 {
		t.Fatalf("after disarming, on air %d, want deck B", st.OnAir)
	}
}

func TestUnloadDropsGotoState(t *testing.T) {
	e, out, _ := newTestEngine()
	it, _ := newCueItem(114, 60_000)
	e.SetTrackCues(cuesFor(114, map[int]int64{1: hc1Ms}, db.DJGoto{ID: 14, PositionMs: g1Ms, TargetSlot: 1, Enabled: true}))
	startAt(t, e, it, 18_000)
	frames(e, out, 1)
	waitPrep(t, e, 0, hc1Ms)
	prep := func() *prepared { e.mu.Lock(); defer e.mu.Unlock(); return e.decks[0].prep }
	p := prep()
	if err := e.EjectDeck(0); err != nil {
		t.Fatal(err)
	}
	if prep() != nil || !p.src.(*constSource).closed {
		t.Fatal("prepared decoder not closed on unload")
	}
	if ds := e.State().Decks[0]; len(ds.Gotos) != 0 || len(ds.Hotcues) != 0 {
		t.Fatal("unloaded deck still reports cues")
	}
}

// Concurrent edits from several devices while the mixer runs: run with
// -race. Ends with a consistent state (the last write wins).
func TestConcurrentCueEditsWhilePlaying(t *testing.T) {
	e, out, _ := newTestEngine()
	it, _ := newCueItem(115, 60_000)
	startAt(t, e, it, 1_000)
	var wg sync.WaitGroup
	stop := make(chan struct{})
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
				frames(e, out, 1)
			}
		}
	}()
	var devs sync.WaitGroup
	for dev := 0; dev < 3; dev++ {
		devs.Add(1)
		go func(dev int) {
			defer devs.Done()
			for i := 0; i < 200; i++ {
				e.SetTrackCues(cuesFor(115, map[int]int64{1: int64(1000 * (dev + 1))},
					db.DJGoto{ID: 15, PositionMs: 30_000, TargetSlot: 1, Enabled: i%2 == 0}))
				_ = e.State()
			}
		}(dev)
	}
	devs.Wait()
	e.SetTrackCues(cuesFor(115, map[int]int64{1: 7_000}))
	close(stop)
	wg.Wait()
	if hc := e.State().Decks[0].Hotcues; len(hc) != 1 || hc[0].PositionMs != 7_000 {
		t.Fatalf("final hotcues %+v", hc)
	}
}
