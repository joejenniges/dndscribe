package dj

import (
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/bwmarrin/discordgo"
	"github.com/joe/dndscribe-go/internal/db"
)

// fakeOutput models the library's OpusSend: frames sit in a queue until the
// test drains them (the real sender drains one per 20ms).
type fakeOutput struct {
	ready    bool
	pending  int
	sent     [][]byte
	speaking []bool
}

func (f *fakeOutput) Ready() bool         { return f.ready }
func (f *fakeOutput) Queued() int         { return f.pending }
func (f *fakeOutput) SetSpeaking(on bool) { f.speaking = append(f.speaking, on) }
func (f *fakeOutput) Send(b []byte) bool {
	f.pending++
	f.sent = append(f.sent, b)
	return true
}
func (f *fakeOutput) drain() { f.pending = 0 }

// fakeEncoder records the PCM it was given so tests can inspect the mix.
type fakeEncoder struct{ frames [][]int16 }

func (f *fakeEncoder) Encode(pcm []int16, out []byte) (int, error) {
	c := make([]int16, len(pcm))
	copy(c, pcm)
	f.frames = append(f.frames, c)
	out[0] = 0x01
	return 1, nil
}

// constSource yields n frames of a constant sample value.
type constSource struct {
	val    int16
	n      int
	closed bool
}

func (c *constSource) Next() ([]int16, bool, bool) {
	if c.n <= 0 {
		return nil, false, true
	}
	c.n--
	f := make([]int16, FrameLen)
	for i := range f {
		f[i] = c.val
	}
	return f, true, false
}
func (c *constSource) Close() { c.closed = true }

func newTestEngine() (*Engine, *fakeOutput, *fakeEncoder) {
	out := &fakeOutput{ready: true}
	enc := &fakeEncoder{}
	e := NewEngine("ffmpeg", ".", out, enc, 1, 1)
	e.connected = true
	return e, out, enc
}

var constTrackID int64

// constItem is a playable item backed by a constSource of n frames. durMs > 0
// gives it a library track with that duration (Auto DJ timing needs one).
func constItem(label string, val int16, n int, durMs int64) (item, *constSource) {
	src := &constSource{val: val, n: n}
	it := item{label: label, scale: 1, open: func() (Source, error) { return src, nil }}
	if durMs > 0 {
		constTrackID++
		it.track = &db.DJTrack{ID: constTrackID, Title: label, DurationMs: durMs}
	}
	return it, src
}

// runTicks runs n mixer ticks, draining the output between them, so each
// tick mixes targetQueue frames.
func runTicks(e *Engine, out *fakeOutput, n int) {
	buf := make([]byte, 4000)
	for i := 0; i < n; i++ {
		e.tick(buf)
		out.drain()
	}
}

// settle waits for background deck loads (Auto DJ preloads) to finish.
func settle(t *testing.T, e *Engine) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		e.mu.Lock()
		busy := e.decks[0].loading || e.decks[1].loading
		e.mu.Unlock()
		if !busy {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("deck load did not finish")
}

func mid(f []int16) int16 { return f[FrameLen/2] }

func TestToneUsesEffectsBusAndStopsSpeaking(t *testing.T) {
	e, out, enc := newTestEngine()
	if err := e.PlayTone(0.2); err != nil { // 10 frames
		t.Fatal(err)
	}
	runTicks(e, out, 30)

	if len(enc.frames) != 10 {
		t.Fatalf("encoded %d frames, want 10", len(enc.frames))
	}
	silence := 0
	for _, b := range out.sent {
		if len(b) == 3 && b[0] == 0xF8 && b[1] == 0xFF && b[2] == 0xFE {
			silence++
		}
	}
	if silence != 5 {
		t.Fatalf("sent %d opus silence frames, want 5", silence)
	}
	if len(out.speaking) != 1 || out.speaking[0] {
		t.Fatalf("speaking calls = %v, want one Speaking(false) after the tail", out.speaking)
	}
	st := e.State()
	if st.Decks[0].Track != nil || st.Decks[1].Track != nil {
		t.Fatal("the tone touched a deck")
	}
}

func TestQueueStaysShallow(t *testing.T) {
	e, out, _ := newTestEngine()
	_ = e.PlayTone(5)
	e.tick(make([]byte, 4000))
	if out.pending != targetQueue {
		t.Fatalf("queued %d frames in one tick, want %d (control latency)", out.pending, targetQueue)
	}
}

func TestNoConnectionHoldsPosition(t *testing.T) {
	e, out, enc := newTestEngine()
	it, _ := constItem("a", 1000, 500, 0)
	if err := e.startNow(it, 0, "track"); err != nil {
		t.Fatal(err)
	}
	frames(e, out, 2)
	n := len(enc.frames)
	pos := e.decks[0].v.framesPlayed
	out.ready = false
	runTicks(e, out, 20)
	if len(enc.frames) != n || e.decks[0].v.framesPlayed != pos {
		t.Fatalf("mixed %d frames while disconnected", len(enc.frames)-n)
	}
	out.ready = true
	frames(e, out, 2)
	if len(enc.frames) <= n {
		t.Fatal("playback did not resume after reconnect")
	}
}

// From silence, music starts on the deck the crossfader points at, without
// moving the fader.
func TestStartFromSilenceUsesFavouredDeck(t *testing.T) {
	e, out, _ := newTestEngine()
	e.SetCrossfader(1)
	frames(e, out, 1)
	it, _ := constItem("b", 1000, 500, 0)
	if err := e.startNow(it, 6, "scene"); err != nil {
		t.Fatal(err)
	}
	st := e.State()
	if st.Decks[1].Track == nil || !st.Decks[1].Playing || st.Transition != nil || st.Crossfader != 1 {
		t.Fatalf("want deck B playing, no transition, fader at B; got %+v", st)
	}
}

// Auto DJ with no known durations: the next queue track is preloaded on the
// other deck and cut to in the same frame the first one ends.
func TestAutoDJGaplessCut(t *testing.T) {
	e, out, enc := newTestEngine()
	a, sa := constItem("a", 1000, 20, 0)
	b, _ := constItem("b", 2000, 10, 0)
	e.q.replace([]item{a, b}, nil, nil, "two", false, false)
	if err := e.popAndStart(0, "scene"); err != nil {
		t.Fatal(err)
	}
	frames(e, out, 1) // Auto DJ starts preloading b
	settle(t, e)
	frames(e, out, 40)
	// 20 frames of a then 10 of b, back to back, no silent frame between.
	if len(enc.frames) != 30 {
		t.Fatalf("encoded %d frames, want 30", len(enc.frames))
	}
	if got := mid(enc.frames[19]); got != 1000 {
		t.Fatalf("frame 19 = %d, want 1000 (a)", got)
	}
	for i := 20; i < 30; i++ {
		if got := mid(enc.frames[i]); got != 2000 {
			t.Fatalf("frame %d = %d, want 2000 (b, full level)", i, got)
		}
	}
	if !sa.closed {
		t.Fatal("finished source not closed")
	}
}

func TestCrossfaderEndpointsAndCentre(t *testing.T) {
	e, out, enc := newTestEngine()
	a, _ := constItem("a", 10000, 1000, 0)
	b, _ := constItem("b", 6000, 1000, 0)
	if err := e.openOnDeck(0, &a); err != nil {
		t.Fatal(err)
	}
	if err := e.openOnDeck(1, &b); err != nil {
		t.Fatal(err)
	}
	_ = e.PlayDeck(0)
	_ = e.PlayDeck(1)
	check := func(x float64, want float64) {
		t.Helper()
		e.SetCrossfader(x)
		frames(e, out, 15) // past the play ramp and the one-frame move
		got := float64(mid(enc.frames[len(enc.frames)-1]))
		if math.Abs(got-want) > 2 {
			t.Fatalf("x=%.1f: sample %.0f, want %.0f", x, got, want)
		}
	}
	check(0, 10000)
	check(1, 6000)
	check(0.5, (10000+6000)*math.Sqrt2/2) // both at -3dB
}

// A scene change moves the crossfader across over the fade, then unloads the
// old deck so Auto DJ can use it.
func TestTransitionMovesFaderThenRetiresOldDeck(t *testing.T) {
	e, out, enc := newTestEngine()
	e.SetAutoDJ(false)
	a, sa := constItem("a", 10000, 1000, 0)
	if err := e.startNow(a, 0, "scene"); err != nil {
		t.Fatal(err)
	}
	frames(e, out, 15)
	b, _ := constItem("b", 10000, 1000, 0)
	start := len(enc.frames)
	if err := e.startNow(b, 1, "scene"); err != nil { // 50 frames
		t.Fatal(err)
	}
	if tr := e.State().Transition; tr == nil || tr.ToDeck != 1 || tr.DurationMs != 1000 {
		t.Fatalf("transition = %+v, want to deck B over 1000ms", tr)
	}
	frames(e, out, 25)
	// Two equal, correlated signals under the equal-power law peak at
	// sqrt(2) halfway.
	if got := float64(mid(enc.frames[start+24])); math.Abs(got-10000*math.Sqrt2) > 400 {
		t.Fatalf("mid-fade sample %.0f, want about %.0f", got, 10000*math.Sqrt2)
	}
	frames(e, out, 30)
	st := e.State()
	if st.Transition != nil || st.Crossfader != 1 {
		t.Fatalf("after fade: transition %+v, fader %.2f", st.Transition, st.Crossfader)
	}
	if st.Decks[0].Track != nil || !sa.closed {
		t.Fatal("old deck not unloaded after the fade")
	}
	if got := mid(enc.frames[len(enc.frames)-1]); got != 10000 {
		t.Fatalf("after fade sample %d, want 10000", got)
	}
}

func TestManualCrossfaderCancelsAutomation(t *testing.T) {
	e, out, _ := newTestEngine()
	e.SetAutoDJ(false)
	a, _ := constItem("a", 1000, 1000, 0)
	_ = e.startNow(a, 0, "scene")
	frames(e, out, 15)
	b, _ := constItem("b", 1000, 1000, 0)
	_ = e.startNow(b, 2, "scene")
	frames(e, out, 20)
	e.SetCrossfader(0.3)
	frames(e, out, 150)
	st := e.State()
	if st.Transition != nil || math.Abs(st.Crossfader-0.3) > 1e-9 {
		t.Fatalf("fader %.2f transition %+v, want 0.3 and none", st.Crossfader, st.Transition)
	}
	if !st.Decks[0].Playing || !st.Decks[1].Playing {
		t.Fatal("a cancelled transition should leave both decks playing")
	}
}

// Auto DJ starts the next track before the current one ends, by the fade
// time, and the old deck is unloaded when the fade completes.
func TestAutoDJCrossfadesBeforeTheEnd(t *testing.T) {
	e, out, enc := newTestEngine()
	e.SetFadeSeconds(0.4) // 20 frames
	a, _ := constItem("a", 1000, 100, 2000)
	b, _ := constItem("b", 3000, 1000, 20000)
	e.q.replace([]item{a, b}, nil, nil, "two", false, false)
	if err := e.popAndStart(0, "scene"); err != nil {
		t.Fatal(err)
	}
	frames(e, out, 1)
	settle(t, e)
	frames(e, out, 78) // 79 frames = 1580ms played, 420ms left
	if e.State().Transition != nil {
		t.Fatal("transition started too early")
	}
	frames(e, out, 2)
	if tr := e.State().Transition; tr == nil || tr.Kind != "auto" || tr.ToDeck != 1 {
		t.Fatalf("want an auto transition to B at 1.6s, got %+v", tr)
	}
	frames(e, out, 30)
	st := e.State()
	if st.OnAir != 1 || st.Decks[0].Track != nil || st.Transition != nil {
		t.Fatalf("after: onAir %d deckA %+v transition %+v", st.OnAir, st.Decks[0].Track, st.Transition)
	}
	if len(st.History) != 1 || st.History[0].Title != "a" {
		t.Fatalf("history = %+v, want [a]", st.History)
	}
	if got := mid(enc.frames[len(enc.frames)-1]); got != 3000 {
		t.Fatalf("now playing %d, want b (3000)", got)
	}
}

// A new scene during a transition replaces it: one automation, ending on
// the newest track only.
func TestLatestTransitionWins(t *testing.T) {
	e, out, _ := newTestEngine()
	e.SetAutoDJ(false)
	a, _ := constItem("a", 1000, 1000, 0)
	_ = e.startNow(a, 0, "scene")
	frames(e, out, 15)
	b, _ := constItem("b", 2000, 1000, 0)
	_ = e.startNow(b, 1, "scene")
	frames(e, out, 10)
	c, _ := constItem("c", 3000, 1000, 0)
	_ = e.startNow(c, 1, "scene")
	frames(e, out, 120)
	st := e.State()
	var loaded []string
	for _, d := range st.Decks {
		if d.Track != nil {
			loaded = append(loaded, d.Track.Title)
		}
	}
	if len(loaded) != 1 || loaded[0] != "c" || st.Transition != nil {
		t.Fatalf("loaded %v, transition %+v; want only c", loaded, st.Transition)
	}
}

func TestPauseDeckRampsAndHoldsPosition(t *testing.T) {
	e, out, enc := newTestEngine()
	it, _ := constItem("a", 1000, 1000, 0)
	_ = e.startNow(it, 0, "track")
	frames(e, out, 20)
	_ = e.PauseDeck(0)
	frames(e, out, 30)
	n := len(enc.frames)
	pos := e.decks[0].v.framesPlayed
	frames(e, out, 10)
	if len(enc.frames) != n || e.decks[0].v.framesPlayed != pos {
		t.Fatal("still mixing while paused")
	}
	if !e.State().Paused {
		t.Fatal("legacy paused flag not set")
	}
	_ = e.PlayDeck(0)
	frames(e, out, 5)
	if len(enc.frames) == n {
		t.Fatal("did not resume")
	}
}

func TestStopAllFadesDecksAndEffectsAndClearsQueue(t *testing.T) {
	e, out, enc := newTestEngine()
	a, _ := constItem("a", 1000, 1000, 0)
	b, _ := constItem("b", 1000, 1000, 0)
	e.q.replace([]item{a, b}, nil, nil, "two", true, false)
	_ = e.popAndStart(0, "scene")
	_ = e.TriggerPad(1, "fx", constPCM(1000, 500), 1, "")
	frames(e, out, 20)
	settle(t, e)
	e.Stop(nil)
	frames(e, out, 30)
	st := e.State()
	if st.Decks[0].Track != nil || st.Decks[1].Track != nil || st.QueueTotal != 0 || len(st.Sfx) != 0 {
		t.Fatalf("after stop: decks %+v queue %d sfx %d", st.Decks, st.QueueTotal, len(st.Sfx))
	}
	n := len(enc.frames)
	frames(e, out, 5)
	if len(enc.frames) != n {
		t.Fatal("still sending after stop")
	}
}

func TestQueueLoopRefillAndShuffleAvoidsRepeat(t *testing.T) {
	var q queue
	var items []item
	for i := 0; i < 4; i++ {
		it, _ := constItem(string(rune('a'+i)), 0, 1, 1000)
		items = append(items, it)
	}
	q.replace(items, nil, nil, "x", true, false)
	for i := 0; i < 4; i++ {
		q.popLocked()
	}
	if it, ok := q.popLocked(); !ok || it.label != "a" {
		t.Fatalf("loop refill gave %q %v, want a", it.label, ok)
	}
	q.shuffle = true
	for i := 0; i < 200; i++ {
		q.entries = nil
		q.fill(items[2].track.ID)
		if q.entries[0].it.track.ID == items[2].track.ID {
			t.Fatal("shuffle refill repeated the last track")
		}
	}
}

func TestQueueMoveRemoveUnshuffle(t *testing.T) {
	e, _, _ := newTestEngine()
	var items []item
	for i := 0; i < 5; i++ {
		it, _ := constItem(string(rune('a'+i)), 0, 1, 1000)
		items = append(items, it)
	}
	e.q.replace(items, nil, nil, "x", false, false)
	titles := func() string {
		s := ""
		for _, q := range e.State().Queue {
			s += q.Title
		}
		return s
	}
	st := e.State()
	if err := e.QueueMove(st.Queue[4].QID, 0); err != nil {
		t.Fatal(err)
	}
	if err := e.QueueRemove(st.Queue[1].QID); err != nil {
		t.Fatal(err)
	}
	if got := titles(); got != "eacd" {
		t.Fatalf("after move+remove %s, want eacd", got)
	}
	e.SetShuffle(true)
	e.SetShuffle(false)
	if got := titles(); got != "acde" {
		t.Fatalf("unshuffle gave %s, want source order acde", got)
	}
}

func TestSendOnClosedChannelDoesNotPanic(t *testing.T) {
	ch := make(chan []byte, 16)
	vc := &discordgo.VoiceConnection{Cond: sync.NewCond(&sync.Mutex{}), Status: discordgo.VoiceConnectionStatusReady, OpusSend: ch}
	o := &VoiceOutput{Conn: func() *discordgo.VoiceConnection { return vc }}
	if !o.Send([]byte{1}) {
		t.Fatal("send to open channel failed")
	}
	close(ch)
	if o.Send([]byte{1}) {
		t.Fatal("send to closed channel reported success")
	}
}

// TestFFmpegDecodeAndProbe needs ffmpeg on PATH; skipped otherwise.
func TestFFmpegDecodeAndProbe(t *testing.T) {
	ff, err := exec.LookPath("ffmpeg")
	if err != nil {
		t.Skip("ffmpeg not on PATH")
	}
	dir := t.TempDir()
	file := filepath.Join(dir, "tone.mp3")
	gen := exec.Command(ff, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
		"-metadata", "title=Test Sine", "-metadata", "artist=dndscribe", file)
	if out, err := gen.CombinedOutput(); err != nil {
		t.Fatalf("generate mp3: %v\n%s", err, out)
	}
	if _, err := os.Stat(file); err != nil {
		t.Fatal(err)
	}

	tr := Probe(ff, file)
	if tr.Title != "Test Sine" || tr.Artist != "dndscribe" {
		t.Fatalf("tags = %q / %q", tr.Title, tr.Artist)
	}
	if tr.DurationMs < 1900 || tr.DurationMs > 2200 {
		t.Fatalf("duration = %dms, want about 2000", tr.DurationMs)
	}

	src, err := openFFmpeg(ff, file, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	frames := 0
	var peak int16
	for {
		f, ready, done := src.Next()
		if done {
			break
		}
		if !ready {
			continue
		}
		frames++
		for _, s := range f {
			if s > peak {
				peak = s
			}
		}
	}
	// 2s = 100 frames; mp3 encoder padding adds a few.
	if frames < 98 || frames > 110 {
		t.Fatalf("decoded %d frames, want about 100", frames)
	}
	if peak < 1000 {
		t.Fatalf("decoded audio is near silent (peak %d)", peak)
	}
}

// Decks can be cued before the bot joins voice; only music that was playing
// when the connection dropped is cleared after the grace period.
func TestDisconnectGraceKeepsCuedDecks(t *testing.T) {
	e, out, _ := newTestEngine()
	out.ready = false
	e.connected = false
	cued, _ := constItem("cued", 1000, 100, 0)
	if err := e.openOnDeck(1, &cued); err != nil {
		t.Fatal(err)
	}
	runTicks(e, out, 3)
	if e.State().Decks[1].Track == nil {
		t.Fatal("a deck loaded while not in voice was unloaded")
	}

	out.ready = true
	playing, _ := constItem("playing", 1000, 1000, 0)
	_ = e.startNow(playing, 0, "track")
	runTicks(e, out, 2)
	out.ready = false
	runTicks(e, out, 1)
	e.mu.Lock()
	e.disconnectedAt = time.Now().Add(-2 * disconnectGrace)
	e.mu.Unlock()
	runTicks(e, out, 1)
	st := e.State()
	if st.Decks[0].Track != nil {
		t.Fatal("playing deck survived the grace period")
	}
	if st.Decks[1].Track == nil {
		t.Fatal("cued (paused) deck was dropped with the playing one")
	}
}
