package dj

import (
	"context"
	"errors"
	"log"
	"math"
	"path"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/joe/dndscribe-go/internal/db"
)

// Engine is the server-side DJ: it owns what is playing, mixes it, encodes it
// to opus and feeds the voice connection. Browsers are only controllers, so
// music keeps playing with every page closed, and any number of devices (the
// GM's desktop and tablet) can drive it at once and always agree on state.
//
// Model (two-deck, like DJ software):
//
//		Deck A --gain--\
//		                crossfader (equal power) --+
//		Deck B --gain--/                           +-- master -- limiter -- opus
//		Soundboard effects -- effects gain --------+
//
//	  - A deck holds one loaded track and plays/pauses/seeks independently.
//	  - The crossfader x (0 = A, 1 = B) is server state. A user can move it;
//	    automated transitions (scene changes, Auto DJ, Next) move it over time.
//	    Touching it during an automated transition cancels the automation.
//	  - A shared queue says what plays next. A scene or folder fills it.
//	  - Auto DJ: near the end of the on-air deck's track, the other deck (which
//	    Auto DJ has already loaded with the next queue item) starts and the
//	    crossfader moves across.
//	  - Effects are mixed after the crossfader, so they never get faded out.
type Engine struct {
	mu sync.Mutex

	ffmpeg   string
	musicDir string
	out      Output
	enc      Encoder

	volume      float64 // master target, 0..1
	curVolume   float64 // smoothed master, ramps to volume
	fadeSeconds float64 // default transition length (scene change, Auto DJ)

	decks [2]*deck

	xfade   float64   // crossfader position as last mixed, 0 (A) .. 1 (B)
	xTarget float64   // where a manual move wants it; reached in one frame
	auto    *autoFade // automated crossfader move in progress, nil = none
	autoDJ  bool

	q       queue
	history []HistoryEntry

	pausedByPauseAll []int // decks the legacy "pause" stopped, for "resume"

	connected      bool
	disconnectedAt time.Time
	sending        bool
	speakingOffIn  int // ticks until Speaking(false) after the silence tail

	lastErr string

	// Soundboard bus (sfx.go).
	sfx          []*sfxVoice
	sfxInstance  int64
	sfxGain      float64
	sfxSeen      map[string]bool // recent trigger requestIDs, for idempotent retries
	sfxSeenOrder []string

	masterPeak float64 // meter accumulator, reset when read

	rev      atomic.Int64 // bumped on every state change
	notifyCh chan struct{}
	OnState  func(State)
	OnMeters func(Meters)
}

// Encoder turns one 20ms stereo PCM frame into an opus packet.
type Encoder interface {
	Encode(pcm []int16, out []byte) (int, error)
}

// item is something a deck can play: a library track, or (in tests) any
// Source.
type item struct {
	track   *db.DJTrack
	label   string
	sceneID int64 // scene whose queue this came from, 0 = none (for live volume changes)
	scale   float64
	open    func() (Source, error)
	// openAt opens the item part-way through, for seeking. nil = can't seek.
	openAt func(startMs int64) (Source, error)
}

func (it *item) durationMs() int64 {
	if it.track == nil {
		return 0
	}
	return it.track.DurationMs
}

type voice struct {
	src          Source
	item         *item
	env          envelope
	framesPlayed int64
	done         bool
}

// envelope is a gain curve over a number of frames. Fades are equal-power
// (sin in, cos out) so a crossfade doesn't dip in loudness halfway through,
// which a linear fade does.
type envelope struct {
	kind   int // envSteady, envIn, envOut
	pos    int // frames elapsed
	length int // frames
}

const (
	envSteady = iota
	envIn
	envOut
)

// gain at fractional frame position f (pos + fraction within the frame).
func (e *envelope) gain(f float64) float64 {
	if e.kind == envSteady || e.length <= 0 {
		if e.kind == envOut {
			return 0
		}
		return 1
	}
	p := f / float64(e.length)
	if p >= 1 {
		if e.kind == envIn {
			return 1
		}
		return 0
	}
	if p < 0 {
		p = 0
	}
	if e.kind == envIn {
		return math.Sin(p * math.Pi / 2)
	}
	return math.Cos(p * math.Pi / 2)
}

func (e *envelope) finishedOut() bool { return e.kind == envOut && e.pos >= e.length }

func fadeIn(frames int) envelope {
	if frames <= 0 {
		return envelope{kind: envSteady}
	}
	return envelope{kind: envIn, length: frames}
}

func fadeOut(frames int) envelope { return envelope{kind: envOut, length: frames} }

func secondsToFrames(s float64) int { return int(s * 1000 / FrameMs) }

// NewEngine builds an engine. Call Run to start the output clock.
func NewEngine(ffmpegPath, musicDir string, out Output, enc Encoder, volume, fadeSeconds float64) *Engine {
	e := &Engine{
		ffmpeg:      ffmpegPath,
		musicDir:    musicDir,
		out:         out,
		enc:         enc,
		volume:      clamp01(volume),
		curVolume:   clamp01(volume),
		fadeSeconds: fadeSeconds,
		sfxGain:     1,
		autoDJ:      true,
		notifyCh:    make(chan struct{}, 1),
	}
	for i := range e.decks {
		e.decks[i] = &deck{idx: i, gain: 1, curGain: 1}
	}
	return e
}

func clamp01(v float64) float64 { return math.Max(0, math.Min(1, v)) }

// ---- Output clock ----

// targetQueue is how many frames we keep queued in OpusSend. The library's
// sender drains one per 20ms on its own ticker; 3 frames (60ms) covers timer
// jitter while keeping control latency (tap a scene -> hear it) low. Filling
// the whole 16-frame channel would add 320ms to every action.
const targetQueue = 3

// disconnectGrace: how long music survives with no voice connection before
// it is dropped. Covers voice reconnects without music resuming unexpectedly
// at the start of next week's session.
const disconnectGrace = 10 * time.Second

// Run drives the mixer until ctx is done.
func (e *Engine) Run(ctx context.Context) {
	go e.notifyLoop(ctx)
	go e.meterLoop(ctx)
	ticker := time.NewTicker(FrameMs * time.Millisecond)
	defer ticker.Stop()
	buf := make([]byte, 4000)
	for {
		select {
		case <-ctx.Done():
			e.mu.Lock()
			e.stopAllNowLocked()
			e.sfx = nil
			e.mu.Unlock()
			return
		case <-ticker.C:
		}
		e.tick(buf)
	}
}

func (e *Engine) tick(buf []byte) {
	e.mu.Lock()
	defer e.mu.Unlock()

	ready := e.out.Ready()
	if ready != e.connected {
		e.connected = ready
		if !ready {
			e.disconnectedAt = time.Now()
			e.sending = false
			e.speakingOffIn = 0
			// Effects are moments: one cut off by a voice drop shouldn't
			// play late after the reconnect.
			e.sfx = nil
		}
		e.notify()
	}
	if !ready {
		// Hold position while there's nowhere to send: nothing is pulled from
		// the decoders, so the music picks up where it was after a reconnect.
		// After the grace period, music that was playing is dropped. Decks
		// that are only loaded (cued up before the bot joins) stay.
		if !e.disconnectedAt.IsZero() && time.Since(e.disconnectedAt) > disconnectGrace {
			e.disconnectedAt = time.Time{}
			if e.musicPlayingLocked() {
				log.Printf("DJ: no voice connection for %s, stopping playback", disconnectGrace)
				for _, d := range e.decks {
					if d.playing || d.run > 0 {
						e.ejectLocked(d)
					}
				}
				e.auto = nil
				e.q.clear()
				e.notify()
			}
		}
		return
	}

	if e.speakingOffIn > 0 {
		e.speakingOffIn--
		if e.speakingOffIn == 0 && !e.sending {
			e.out.SetSpeaking(false)
		}
	}

	for e.out.Queued() < targetQueue {
		pcm, active := e.mixFrameLocked()
		if !active {
			if e.sending {
				// Discord asks for five frames of opus silence when audio stops,
				// to avoid interpolation artifacts on the receiving end.
				for i := 0; i < 5; i++ {
					e.out.Send([]byte{0xF8, 0xFF, 0xFE})
				}
				e.sending = false
				// WHY the delay: the library's sender re-asserts Speaking(true)
				// whenever it sends a frame while not speaking. Turning it off
				// now, with the silence frames still queued, would just be
				// undone. 8 ticks lets the queue drain first.
				e.speakingOffIn = 8
			}
			return
		}
		n, err := e.enc.Encode(pcm, buf)
		if err != nil {
			log.Printf("DJ: opus encode: %v", err)
			return
		}
		pkt := make([]byte, n)
		copy(pkt, buf[:n])
		if !e.out.Send(pkt) {
			return
		}
		e.sending = true
		e.speakingOffIn = 0
	}
}

func (e *Engine) musicPlayingLocked() bool {
	return e.decks[0].playing || e.decks[1].playing || e.decks[0].run > 0 || e.decks[1].run > 0 || e.auto != nil
}

// mixFrameLocked produces one frame. active=false means there is nothing to
// play and the caller should go quiet.
func (e *Engine) mixFrameLocked() ([]int16, bool) {
	musicOn := e.decks[0].sounding() || e.decks[1].sounding()
	if !musicOn && len(e.sfx) == 0 {
		return nil, false
	}

	e.autoStepLocked()

	var bus [2][FrameLen]float64
	var contributed bool
	for i, d := range e.decks {
		was := d.playing
		contributed = e.mixDeck(d, bus[i][:]) || contributed
		if was && !d.playing && d.v != nil && d.v.done {
			// The track ran out this frame. With Auto DJ, cut straight to
			// the other deck in this same frame, so there's no gap. (A deck
			// later in this loop gets mixed by the loop itself.)
			if o := e.trackEndedLocked(i); o >= 0 && o < i {
				contributed = e.mixDeck(e.decks[o], bus[o][:]) || contributed
			}
		}
	}

	// Crossfader: from where it was to where it's going this frame.
	x0 := e.xfade
	x1 := e.xTarget
	if a := e.auto; a != nil {
		a.pos++
		x1 = a.from + (a.target-a.from)*math.Min(1, float64(a.pos)/float64(max(1, a.frames)))
		e.xTarget = x1
	}
	wa0, wb0 := xfadeWeights(x0)
	wa1, wb1 := xfadeWeights(x1)

	var fx [FrameLen]float64
	fxOn := e.mixSfxLocked(fx[:])

	if !contributed && !fxOn && !e.decks[0].sounding() && !e.decks[1].sounding() && len(e.sfx) == 0 {
		e.xfade = x1
		e.finishAutoLocked()
		return nil, false
	}

	m0, m1 := e.curVolume, e.volume
	out := make([]int16, FrameLen)
	peak := 0.0
	for i := 0; i < FrameSamples; i++ {
		t := float64(i) / FrameSamples
		wa := wa0 + (wa1-wa0)*t
		wb := wb0 + (wb1-wb0)*t
		gm := m0 + (m1-m0)*t
		for c := 0; c < Channels; c++ {
			j := i*Channels + c
			s := softClip((bus[0][j]*wa+bus[1][j]*wb)*gm + fx[j]*gm)
			out[j] = int16(s)
			peak = math.Max(peak, math.Abs(s))
		}
	}
	e.masterPeak = math.Max(e.masterPeak, peak/math.MaxInt16)
	e.curVolume = e.volume
	e.xfade = x1
	e.finishAutoLocked()
	return out, true
}

// xfadeWeights is the equal-power crossfader law: at the centre both decks
// are at -3dB (0.707), so two uncorrelated tracks sum to the same loudness as
// one at the edge.
func xfadeWeights(x float64) (a, b float64) {
	return math.Cos(x * math.Pi / 2), math.Sin(x * math.Pi / 2)
}

// softClip limits a sample to int16 range. Below the knee it is untouched;
// above, it bends smoothly toward full scale instead of hard clipping. WHY:
// an explosion on top of loud battle music sums past full scale, and hard
// clipping that is harsh distortion, where this is a gentle squash.
func softClip(s float64) float64 {
	const full = math.MaxInt16
	const knee = 0.85 * full
	a := math.Abs(s)
	if a <= knee {
		return s
	}
	r := full - knee
	y := knee + r*math.Tanh((a-knee)/r)
	if s < 0 {
		return -y
	}
	return y
}

// mixVoice adds one frame of v, scaled by g0..g1 across the frame, into acc.
// Returns true if v produced audio.
func mixVoice(v *voice, acc []float64, scale float64) bool {
	if v.done {
		return false
	}
	frame, ready, done := v.src.Next()
	if done {
		v.done = true
		return false
	}
	g0 := v.env.gain(float64(v.env.pos)) * scale
	g1 := v.env.gain(float64(v.env.pos+1)) * scale
	if v.env.kind != envSteady {
		v.env.pos++
	}
	if !ready {
		return false // decoder behind: silence for this voice this tick
	}
	v.framesPlayed++
	for i := 0; i < FrameSamples; i++ {
		g := g0 + (g1-g0)*float64(i)/FrameSamples
		for c := 0; c < Channels; c++ {
			acc[i*Channels+c] += float64(frame[i*Channels+c]) * g
		}
	}
	if v.env.kind == envIn && v.env.pos >= v.env.length {
		v.env = envelope{kind: envSteady}
	}
	return true
}

func (e *Engine) setErrLocked(msg string) {
	log.Printf("DJ: %s", msg)
	e.lastErr = msg
	e.notify()
}

// ErrNotConnected is returned by play commands when the bot isn't in a voice channel.
var ErrNotConnected = errors.New("the bot is not in a voice channel")

func (e *Engine) isConnected() bool {
	return e.out.Ready()
}

func (e *Engine) trackItem(t db.DJTrack) item {
	full := filepath.Join(e.musicDir, filepath.FromSlash(t.Path))
	ff := e.ffmpeg
	return item{
		track:  &t,
		label:  trackTitle(t),
		scale:  1,
		open:   func() (Source, error) { return openFFmpeg(ff, full, 0) },
		openAt: func(ms int64) (Source, error) { return openFFmpeg(ff, full, ms) },
	}
}

func trackTitle(t db.DJTrack) string {
	if strings.TrimSpace(t.Title) != "" {
		return t.Title
	}
	base := path.Base(t.Path)
	return strings.TrimSuffix(base, path.Ext(base))
}

// tracksInFolders returns the playable library tracks under any of folders
// (sub-folders included), in path order, each track once even when folders
// overlap ("Combat" and "Combat/Boss"). "" means the whole library.
func tracksInFolders(ctx context.Context, folders []string) ([]db.DJTrack, error) {
	all, err := db.ListDJTracks(ctx)
	if err != nil {
		return nil, err
	}
	var out []db.DJTrack
	for _, t := range all {
		if t.Missing {
			continue
		}
		for _, f := range folders {
			if inFolder(t.Path, f) {
				out = append(out, t)
				break
			}
		}
	}
	return out, nil
}

func inFolder(trackPath, folder string) bool {
	return folder == "" || strings.HasPrefix(trackPath, folder+"/")
}

// PlayTone plays a 440Hz test tone on the effects bus, for checking the voice
// send path. It goes through the effects bus so it never disturbs the decks,
// the queue or the crossfader.
func (e *Engine) PlayTone(seconds float64) error {
	if seconds <= 0 {
		seconds = 2 // long enough to hear, short enough not to annoy the table
	}
	seconds = min(seconds, 30)
	src := newTone(440, seconds)
	var pcm []int16
	for {
		f, _, done := src.Next()
		if done {
			break
		}
		pcm = append(pcm, f...)
	}
	return e.TriggerPad(TonePadID, "Test tone", pcm, 1, "")
}

// TonePadID is the effects-bus id the test tone plays under.
const TonePadID = -1

func (e *Engine) SetVolume(v float64) {
	e.mu.Lock()
	e.volume = clamp01(v)
	e.mu.Unlock()
	e.notify()
}

func (e *Engine) SetFadeSeconds(s float64) {
	e.mu.Lock()
	e.fadeSeconds = math.Max(0, math.Min(30, s))
	e.mu.Unlock()
	e.notify()
}

func (e *Engine) FadeSeconds() float64 {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.fadeSeconds
}

// ---- Notifications ----

// Rev is the current state revision. It increases on every change, so a
// client that got an acknowledgement at revision R knows any state snapshot
// with rev >= R includes its command.
func (e *Engine) Rev() int64 { return e.rev.Load() }

// notify bumps the revision and asks the notify loop to publish state.
// Non-blocking; bursts coalesce.
func (e *Engine) notify() {
	e.rev.Add(1)
	select {
	case e.notifyCh <- struct{}{}:
	default:
	}
}

// minStateInterval caps snapshot broadcasts at 25/s. A crossfader drag from
// two devices sends ~60 commands a second; every client re-rendering 60 full
// snapshots a second is wasted work, and the last one always wins anyway.
const minStateInterval = 40 * time.Millisecond

// notifyLoop publishes state on change, plus every 5s while playing so
// clients correct any drift in their extrapolated positions.
func (e *Engine) notifyLoop(ctx context.Context) {
	t := time.NewTicker(5 * time.Second)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-e.notifyCh:
		case <-t.C:
			e.mu.Lock()
			active := e.connected && (e.decks[0].playing || e.decks[1].playing || len(e.sfx) > 0)
			e.mu.Unlock()
			if !active {
				continue
			}
		}
		if e.OnState != nil {
			e.OnState(e.State())
		}
		select {
		case <-ctx.Done():
			return
		case <-time.After(minStateInterval):
		}
	}
}

// Meters are peak levels (0..1 of full scale) since the last report: each
// deck after its gain (before the crossfader, as on a DJ mixer), and the
// final mix.
type Meters struct {
	Decks  [2]float64 `json:"decks"`
	Master float64    `json:"master"`
}

// meterInterval: 10 reports a second is smooth enough for a level bar and
// light on the socket. Sent separately from state so a meter never causes a
// full-state re-render.
const meterInterval = 100 * time.Millisecond

func (e *Engine) meterLoop(ctx context.Context) {
	t := time.NewTicker(meterInterval)
	defer t.Stop()
	wasActive := false
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
		e.mu.Lock()
		m := Meters{Master: e.masterPeak}
		for i, d := range e.decks {
			m.Decks[i] = d.peak
			d.peak = 0
		}
		e.masterPeak = 0
		active := m.Master > 0 || m.Decks[0] > 0 || m.Decks[1] > 0
		e.mu.Unlock()
		// One all-zero report after activity, so bars drop to nothing.
		if (active || wasActive) && e.OnMeters != nil {
			e.OnMeters(m)
		}
		wasActive = active
	}
}
