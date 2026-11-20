package dj

import (
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"testing"

	"github.com/bwmarrin/discordgo"
)

// fakeOutput models the library's OpusSend: frames sit in a queue until the
// test drains them (the real sender drains one per 20ms).
type fakeOutput struct {
	ready    bool
	pending  int
	sent     [][]byte
	speaking []bool
}

func (f *fakeOutput) Ready() bool  { return f.ready }
func (f *fakeOutput) Queued() int  { return f.pending }
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
	return e, out, enc
}

func constProgram(name string, vals ...int16) (*program, []*constSource) {
	p := &program{name: name, scale: 1}
	var srcs []*constSource
	for _, v := range vals {
		s := &constSource{val: v, n: 10}
		srcs = append(srcs, s)
		p.items = append(p.items, item{label: name, open: func() (Source, error) { return s, nil }})
	}
	return p, srcs
}

// runTicks runs n mixer ticks, draining the output between them.
func runTicks(e *Engine, out *fakeOutput, n int) {
	buf := make([]byte, 4000)
	for i := 0; i < n; i++ {
		e.tick(buf)
		out.drain()
	}
}

func TestToneSendsFramesThenSilenceAndStopsSpeaking(t *testing.T) {
	e, out, enc := newTestEngine()
	e.connected = true
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
	if e.State().Playing {
		t.Fatal("still playing after the tone ended")
	}
}

func TestQueueStaysShallow(t *testing.T) {
	e, out, _ := newTestEngine()
	e.connected = true
	_ = e.PlayTone(5)
	e.tick(make([]byte, 4000))
	if out.pending != targetQueue {
		t.Fatalf("queued %d frames in one tick, want %d (control latency)", out.pending, targetQueue)
	}
}

func TestNoConnectionHoldsPosition(t *testing.T) {
	e, out, enc := newTestEngine()
	e.connected = true
	_ = e.PlayTone(1)
	runTicks(e, out, 2)
	n := len(enc.frames)
	out.ready = false
	runTicks(e, out, 20)
	if len(enc.frames) != n {
		t.Fatalf("mixed %d frames while disconnected", len(enc.frames)-n)
	}
	if !e.State().Playing {
		t.Fatal("program dropped before the grace period")
	}
	out.ready = true
	runTicks(e, out, 2)
	if len(enc.frames) <= n {
		t.Fatal("playback did not resume after reconnect")
	}
}

func TestGaplessAdvance(t *testing.T) {
	e, out, enc := newTestEngine()
	e.connected = true
	p, srcs := constProgram("two", 1000, 2000)
	if err := e.startProgram(p, 0); err != nil {
		t.Fatal(err)
	}
	runTicks(e, out, 30)
	// 10 frames of each, back to back, no silent frame between.
	if len(enc.frames) != 20 {
		t.Fatalf("encoded %d frames, want 20", len(enc.frames))
	}
	for i, f := range enc.frames {
		want := int16(1000)
		if i >= 10 {
			want = 2000
		}
		if f[FrameLen/2] != want {
			t.Fatalf("frame %d mid sample = %d, want %d", i, f[FrameLen/2], want)
		}
	}
	if !srcs[0].closed {
		t.Fatal("finished source not closed")
	}
}

func TestEqualPowerCrossfade(t *testing.T) {
	e, out, enc := newTestEngine()
	e.connected = true
	a := &program{name: "a", scale: 1, loop: true}
	sa := &constSource{val: 10000, n: 1000}
	a.items = []item{{label: "a", open: func() (Source, error) { return sa, nil }}}
	if err := e.startProgram(a, 0); err != nil {
		t.Fatal(err)
	}
	runTicks(e, out, 20) // past the 0.3s start fade

	b := &program{name: "b", scale: 1, loop: true}
	sb := &constSource{val: 10000, n: 1000}
	b.items = []item{{label: "b", open: func() (Source, error) { return sb, nil }}}
	start := len(enc.frames)
	if err := e.startProgram(b, 1); err != nil { // 50-frame crossfade
		t.Fatal(err)
	}
	runTicks(e, out, 60)

	// Two equal, correlated signals under an equal-power fade sum to
	// 10000*(sin+cos), which peaks at sqrt(2) halfway. Uncorrelated music
	// stays near constant power; here we check the curve shape itself.
	mid := enc.frames[start+25][FrameLen/2]
	want := 10000 * math.Sqrt2
	if math.Abs(float64(mid)-want) > 400 {
		t.Fatalf("mid-fade sample = %d, want about %.0f", mid, want)
	}
	end := enc.frames[start+55][FrameLen/2]
	if end != 10000 {
		t.Fatalf("after fade sample = %d, want 10000", end)
	}
	if !sa.closed {
		t.Fatal("outgoing source not closed after fade")
	}
	if e.State().ProgramName != "b" {
		t.Fatalf("program = %q, want b", e.State().ProgramName)
	}
}

func TestPauseRampsAndResumes(t *testing.T) {
	e, out, enc := newTestEngine()
	e.connected = true
	_ = e.PlayTone(5)
	runTicks(e, out, 20)
	e.SetPaused(true)
	runTicks(e, out, 30)
	n := len(enc.frames)
	runTicks(e, out, 10)
	if len(enc.frames) != n {
		t.Fatal("still mixing while paused")
	}
	e.SetPaused(false)
	runTicks(e, out, 5)
	if len(enc.frames) == n {
		t.Fatal("did not resume")
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
