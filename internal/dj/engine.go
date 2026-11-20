package dj

import (
	"context"
	"errors"
	"fmt"
	"log"
	"math"
	"math/rand"
	"path"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/joe/dndscribe-go/internal/db"
)

// Engine is the server-side DJ: it owns what is playing, mixes it, encodes it
// to opus and feeds the voice connection. The browser is only a remote
// control, so music keeps playing with every page closed and the monitor and
// tablet always agree on state.
//
// Model: a "program" is what was asked for (a scene, a playlist, a single
// track, or the test tone): an ordered list of items plus loop/shuffle. The
// program's current item plays as a voice. Voices being faded out (the old
// scene during a crossfade, a skipped track) live in e.fading until silent.
type Engine struct {
	mu sync.Mutex

	ffmpeg   string
	musicDir string
	out      Output
	enc      Encoder

	volume      float64 // master target, 0..1
	curVolume   float64 // smoothed master, ramps to volume
	fadeSeconds float64

	paused    bool
	pauseGain float64 // ramps 0..1 so pause/resume doesn't click

	prog   *program
	fading []*voice
	progSeq int64

	connected      bool
	disconnectedAt time.Time
	sending        bool
	speakingOffIn  int // ticks until Speaking(false) after the silence tail

	lastErr string

	notifyCh chan struct{}
	OnState  func(State)
}

// Encoder turns one 20ms stereo PCM frame into an opus packet.
type Encoder interface {
	Encode(pcm []int16, out []byte) (int, error)
}

type item struct {
	track *db.DJTrack // nil for the test tone
	label string
	open  func() (Source, error)
}

type program struct {
	seq        int64
	sceneID    *int64
	playlistID *int64
	name       string
	items      []item
	order      []int // play order, indices into items
	nextOrder  []int // order for the next loop pass, built lazily when shuffling
	pos        int   // index into order of the current item
	loop       bool
	shuffle    bool
	scale      float64 // per-scene volume

	cur *voice

	// Prefetch: the next item's decoder is started as soon as the current one
	// begins, so track changes are gapless.
	next     Source
	nextItem int // item index next was opened for, -1 = none
}

type voice struct {
	src          Source
	item         *item
	scale        float64
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
	return &Engine{
		ffmpeg:      ffmpegPath,
		musicDir:    musicDir,
		out:         out,
		enc:         enc,
		volume:      clamp01(volume),
		curVolume:   clamp01(volume),
		fadeSeconds: fadeSeconds,
		pauseGain:   1,
		notifyCh:    make(chan struct{}, 1),
	}
}

func clamp01(v float64) float64 { return math.Max(0, math.Min(1, v)) }

// ---- Output clock ----

// targetQueue is how many frames we keep queued in OpusSend. The library's
// sender drains one per 20ms on its own ticker; 3 frames (60ms) covers timer
// jitter while keeping control latency (tap a scene -> hear it) low. Filling
// the whole 16-frame channel would add 320ms to every action.
const targetQueue = 3

// disconnectGrace: how long a program survives with no voice connection
// before it is dropped. Covers voice reconnects without music resuming
// unexpectedly at the start of next week's session.
const disconnectGrace = 10 * time.Second

// Run drives the mixer until ctx is done.
func (e *Engine) Run(ctx context.Context) {
	go e.notifyLoop(ctx)
	ticker := time.NewTicker(FrameMs * time.Millisecond)
	defer ticker.Stop()
	buf := make([]byte, 4000)
	for {
		select {
		case <-ctx.Done():
			e.mu.Lock()
			e.stopAllLocked()
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
		}
		e.notify()
	}
	if !ready {
		// Hold position while there's nowhere to send: nothing is pulled from
		// the decoders, so the music picks up where it was after a reconnect.
		if (e.prog != nil || len(e.fading) > 0) && time.Since(e.disconnectedAt) > disconnectGrace {
			log.Printf("DJ: no voice connection for %s, stopping playback", disconnectGrace)
			e.stopAllLocked()
			e.notify()
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

// mixFrameLocked produces one frame. active=false means there is nothing to
// play (or playback is fully paused) and the caller should go quiet.
func (e *Engine) mixFrameLocked() ([]int16, bool) {
	if e.prog == nil && len(e.fading) == 0 {
		return nil, false
	}
	pauseTarget := 1.0
	if e.paused {
		pauseTarget = 0
	}
	if e.paused && e.pauseGain <= 0 {
		return nil, false
	}

	var acc [FrameLen]float64
	contributed := false

	if p := e.prog; p != nil {
		// Two attempts: if the current track ends this tick, roll straight
		// into the next one so there's no 20ms hole between tracks.
		for attempt := 0; attempt < 2 && e.prog == p && p.cur != nil; attempt++ {
			contributed = e.mixVoice(p.cur, acc[:]) || contributed
			if !p.cur.done {
				break
			}
			e.advanceLocked(p, 0)
			e.notify()
		}
	}

	kept := e.fading[:0]
	for _, v := range e.fading {
		contributed = e.mixVoice(v, acc[:]) || contributed
		if v.done || v.env.finishedOut() {
			v.src.Close()
			continue
		}
		kept = append(kept, v)
	}
	for i := len(kept); i < len(e.fading); i++ {
		e.fading[i] = nil
	}
	e.fading = kept

	// The last track ended this tick and nothing else is playing: go quiet
	// now rather than sending one frame of silence first.
	if !contributed && e.prog == nil && len(e.fading) == 0 {
		return nil, false
	}

	// Master volume and pause gain, ramped across the frame so changes
	// don't produce zipper noise or clicks.
	pauseStep := 1.0 / 10 // 200ms pause/resume ramp
	nextPause := e.pauseGain
	if pauseTarget > e.pauseGain {
		nextPause = math.Min(pauseTarget, e.pauseGain+pauseStep)
	} else if pauseTarget < e.pauseGain {
		nextPause = math.Max(pauseTarget, e.pauseGain-pauseStep)
	}
	v0, v1 := e.curVolume*e.pauseGain, e.volume*nextPause
	out := make([]int16, FrameLen)
	for i := 0; i < FrameSamples; i++ {
		g := v0 + (v1-v0)*float64(i)/FrameSamples
		for c := 0; c < Channels; c++ {
			s := acc[i*Channels+c] * g
			if s > math.MaxInt16 {
				s = math.MaxInt16
			} else if s < math.MinInt16 {
				s = math.MinInt16
			}
			out[i*Channels+c] = int16(s)
		}
	}
	e.curVolume = e.volume
	e.pauseGain = nextPause
	return out, true
}

// mixVoice adds one frame of v into acc. Returns true if v produced audio.
func (e *Engine) mixVoice(v *voice, acc []float64) bool {
	if v.done {
		return false
	}
	frame, ready, done := v.src.Next()
	if done {
		v.done = true
		return false
	}
	g0 := v.env.gain(float64(v.env.pos)) * v.scale
	g1 := v.env.gain(float64(v.env.pos+1)) * v.scale
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

// ---- Program navigation ----

// peekNext returns the item index that plays after the current one, or -1.
func (p *program) peekNext() int {
	if p.pos+1 < len(p.order) {
		return p.order[p.pos+1]
	}
	if !p.loop || len(p.order) == 0 {
		return -1
	}
	if p.shuffle {
		if p.nextOrder == nil {
			p.nextOrder = shuffledOrder(len(p.items), p.order[len(p.order)-1])
		}
		return p.nextOrder[0]
	}
	return p.order[0]
}

// shuffledOrder returns a random permutation of n, avoiding `avoidFirst` as
// the first entry so a looped shuffle never plays the same track twice in a
// row.
func shuffledOrder(n, avoidFirst int) []int {
	o := rand.Perm(n)
	if n > 1 && o[0] == avoidFirst {
		j := 1 + rand.Intn(n-1)
		o[0], o[j] = o[j], o[0]
	}
	return o
}

// advanceLocked moves p to its next item. fadeFrames > 0 crossfades from the
// current voice (used by Next); 0 is a straight, gapless change (track end).
func (e *Engine) advanceLocked(p *program, fadeFrames int) {
	old := p.cur
	p.cur = nil
	if old != nil {
		if fadeFrames > 0 && !old.done {
			old.env = fadeOut(fadeFrames)
			e.fading = append(e.fading, old)
		} else {
			old.src.Close()
		}
	}

	// Try each remaining item at most once so a playlist of missing files
	// can't spin forever.
	for tries := 0; tries < len(p.items)+1; tries++ {
		idx := p.peekNext()
		if idx < 0 {
			e.endProgramLocked(p)
			return
		}
		if p.pos+1 < len(p.order) {
			p.pos++
		} else {
			if p.nextOrder != nil {
				p.order, p.nextOrder = p.nextOrder, nil
			}
			p.pos = 0
		}
		var src Source
		if p.next != nil && p.nextItem == idx {
			src = p.next
		} else {
			if p.next != nil {
				p.next.Close()
			}
			s, err := p.items[idx].open()
			if err != nil {
				e.setErrLocked(fmt.Sprintf("Can't play %s: %v", p.items[idx].label, err))
				continue
			}
			src = s
		}
		p.next, p.nextItem = nil, -1
		p.cur = &voice{src: src, item: &p.items[idx], scale: p.scale, env: fadeIn(fadeFrames)}
		e.prefetchLocked(p)
		return
	}
	e.endProgramLocked(p)
}

func (e *Engine) endProgramLocked(p *program) {
	if p.next != nil {
		p.next.Close()
		p.next = nil
	}
	if p.cur != nil {
		p.cur.src.Close()
		p.cur = nil
	}
	if e.prog == p {
		e.prog = nil
	}
}

// prefetchLocked starts decoding the item after the current one in the
// background. Opening ffmpeg happens off the mixer clock; the result is only
// kept if the program hasn't moved on in the meantime.
func (e *Engine) prefetchLocked(p *program) {
	idx := p.peekNext()
	if idx < 0 || (p.next != nil && p.nextItem == idx) {
		return
	}
	if p.next != nil {
		p.next.Close()
		p.next = nil
	}
	p.nextItem = idx
	open := p.items[idx].open
	go func() {
		src, err := open()
		if err != nil {
			return // advanceLocked retries synchronously and reports the error
		}
		e.mu.Lock()
		defer e.mu.Unlock()
		if e.prog == p && p.nextItem == idx && p.next == nil {
			p.next = src
		} else {
			src.Close()
		}
	}()
}

func (e *Engine) stopAllLocked() {
	if e.prog != nil {
		e.endProgramLocked(e.prog)
	}
	for _, v := range e.fading {
		v.src.Close()
	}
	e.fading = nil
	e.paused = false
	e.pauseGain = 1
}

func (e *Engine) setErrLocked(msg string) {
	log.Printf("DJ: %s", msg)
	e.lastErr = msg
	e.notify()
}

// ---- Commands ----

// ErrNotConnected is returned by play commands when the bot isn't in a voice channel.
var ErrNotConnected = errors.New("the bot is not in a voice channel")

// startProgram swaps in a new program, crossfading from whatever is playing.
// The first item is opened before taking the lock so a slow ffmpeg start
// never stalls the mixer.
func (e *Engine) startProgram(p *program, fadeSeconds float64) error {
	if len(p.items) == 0 {
		return errors.New("nothing to play: the playlist is empty or its files are missing")
	}
	if p.shuffle {
		p.order = shuffledOrder(len(p.items), -1)
	} else {
		p.order = make([]int, len(p.items))
		for i := range p.order {
			p.order[i] = i
		}
	}
	p.nextItem = -1

	var src Source
	var err error
	for p.pos = 0; p.pos < len(p.order); p.pos++ {
		it := p.items[p.order[p.pos]]
		if src, err = it.open(); err == nil {
			break
		}
		log.Printf("DJ: can't open %s: %v", it.label, err)
	}
	if src == nil {
		return fmt.Errorf("couldn't open any track: %v", err)
	}

	e.mu.Lock()
	defer e.mu.Unlock()
	e.progSeq++
	p.seq = e.progSeq

	fadeFrames := secondsToFrames(fadeSeconds)
	if e.prog == nil && len(e.fading) == 0 || e.paused {
		// Nothing audible to crossfade from: short fade-in only, so a battle
		// cue lands immediately instead of swelling in over several seconds.
		fadeFrames = min(fadeFrames, secondsToFrames(0.3))
	}
	if old := e.prog; old != nil {
		if old.next != nil {
			old.next.Close()
			old.next = nil
		}
		if old.cur != nil {
			old.cur.env = fadeOut(fadeFrames)
			if fadeFrames == 0 {
				old.cur.src.Close()
			} else {
				e.fading = append(e.fading, old.cur)
			}
			old.cur = nil
		}
	}
	if e.paused {
		// Playing something new while paused drops the paused audio.
		for _, v := range e.fading {
			v.src.Close()
		}
		e.fading = nil
		e.paused = false
		e.pauseGain = 1
	}
	p.cur = &voice{src: src, item: &p.items[p.order[p.pos]], scale: p.scale, env: fadeIn(fadeFrames)}
	e.prog = p
	e.lastErr = ""
	e.prefetchLocked(p)
	e.notify()
	return nil
}

func (e *Engine) trackItem(t db.DJTrack) item {
	full := filepath.Join(e.musicDir, filepath.FromSlash(t.Path))
	ff := e.ffmpeg
	return item{
		track: &t,
		label: trackTitle(t),
		open:  func() (Source, error) { return openFFmpeg(ff, full, 0) },
	}
}

func trackTitle(t db.DJTrack) string {
	if strings.TrimSpace(t.Title) != "" {
		return t.Title
	}
	base := path.Base(t.Path)
	return strings.TrimSuffix(base, path.Ext(base))
}

// playableTracks resolves track IDs to library tracks, dropping missing files.
func playableTracks(ctx context.Context, ids []int64) ([]db.DJTrack, error) {
	all, err := db.ListDJTracks(ctx)
	if err != nil {
		return nil, err
	}
	byID := make(map[int64]db.DJTrack, len(all))
	for _, t := range all {
		byID[t.ID] = t
	}
	var out []db.DJTrack
	for _, id := range ids {
		if t, ok := byID[id]; ok && !t.Missing {
			out = append(out, t)
		}
	}
	return out, nil
}

// PlayScene crossfades to a scene's playlist. Tapping the scene that is
// already playing does nothing, so a double tap on the tablet doesn't restart
// the music.
func (e *Engine) PlayScene(ctx context.Context, sceneID int64, fadeOverride *float64) error {
	if !e.isConnected() {
		return ErrNotConnected
	}
	e.mu.Lock()
	if e.prog != nil && e.prog.sceneID != nil && *e.prog.sceneID == sceneID && !e.paused {
		e.mu.Unlock()
		return nil
	}
	e.mu.Unlock()

	scene, err := db.GetDJScene(ctx, sceneID)
	if err != nil {
		return err
	}
	if scene == nil {
		return errors.New("scene not found")
	}
	if scene.PlaylistID == nil {
		return fmt.Errorf("scene %q has no playlist", scene.Name)
	}
	pl, err := db.GetDJPlaylist(ctx, *scene.PlaylistID)
	if err != nil {
		return err
	}
	if pl == nil {
		return fmt.Errorf("scene %q has no playlist", scene.Name)
	}
	tracks, err := playableTracks(ctx, pl.TrackIDs)
	if err != nil {
		return err
	}
	p := &program{
		sceneID:    &scene.ID,
		playlistID: &pl.ID,
		name:       scene.Name,
		loop:       scene.Loop,
		shuffle:    scene.Shuffle,
		scale:      scene.Volume,
	}
	for _, t := range tracks {
		p.items = append(p.items, e.trackItem(t))
	}
	fade := e.FadeSeconds()
	if scene.FadeSeconds != nil {
		fade = *scene.FadeSeconds
	}
	if fadeOverride != nil {
		fade = *fadeOverride
	}
	return e.startProgram(p, fade)
}

// PlayPlaylist plays a playlist in order, looping, optionally starting at a
// given track.
func (e *Engine) PlayPlaylist(ctx context.Context, playlistID int64, startTrackID *int64) error {
	if !e.isConnected() {
		return ErrNotConnected
	}
	pl, err := db.GetDJPlaylist(ctx, playlistID)
	if err != nil {
		return err
	}
	if pl == nil {
		return errors.New("playlist not found")
	}
	tracks, err := playableTracks(ctx, pl.TrackIDs)
	if err != nil {
		return err
	}
	if startTrackID != nil {
		for i, t := range tracks {
			if t.ID == *startTrackID {
				tracks = append(tracks[i:], tracks[:i]...)
				break
			}
		}
	}
	p := &program{playlistID: &pl.ID, name: pl.Name, loop: true, scale: 1}
	for _, t := range tracks {
		p.items = append(p.items, e.trackItem(t))
	}
	return e.startProgram(p, e.FadeSeconds())
}

// PlayTrack plays one library track once.
func (e *Engine) PlayTrack(ctx context.Context, trackID int64) error {
	if !e.isConnected() {
		return ErrNotConnected
	}
	t, err := db.GetDJTrack(ctx, trackID)
	if err != nil {
		return err
	}
	if t == nil || t.Missing {
		return errors.New("track not found")
	}
	p := &program{name: trackTitle(*t), scale: 1, items: []item{e.trackItem(*t)}}
	return e.startProgram(p, e.FadeSeconds())
}

// PlayTone plays a 440Hz test tone, for checking the voice send path.
func (e *Engine) PlayTone(seconds float64) error {
	if !e.isConnected() {
		return ErrNotConnected
	}
	if seconds <= 0 {
		seconds = 10
	}
	p := &program{name: "Test tone", scale: 1, items: []item{{
		label: fmt.Sprintf("440 Hz tone, %.0fs", seconds),
		open:  func() (Source, error) { return newTone(440, seconds), nil },
	}}}
	return e.startProgram(p, 0)
}

// Stop fades out everything. fadeSeconds nil = the global fade.
func (e *Engine) Stop(fadeSeconds *float64) {
	e.mu.Lock()
	defer e.mu.Unlock()
	fade := e.fadeSeconds
	if fadeSeconds != nil {
		fade = *fadeSeconds
	}
	frames := secondsToFrames(fade)
	if e.paused {
		frames = 0
	}
	if p := e.prog; p != nil {
		if p.next != nil {
			p.next.Close()
			p.next = nil
		}
		if p.cur != nil {
			if frames > 0 {
				p.cur.env = fadeOut(frames)
				e.fading = append(e.fading, p.cur)
			} else {
				p.cur.src.Close()
			}
			p.cur = nil
		}
		e.prog = nil
	}
	if frames == 0 {
		for _, v := range e.fading {
			v.src.Close()
		}
		e.fading = nil
		e.paused = false
		e.pauseGain = 1
	}
	e.notify()
}

// Next skips to the next item with a short crossfade.
func (e *Engine) Next() {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.prog == nil {
		return
	}
	e.advanceLocked(e.prog, secondsToFrames(1))
	e.notify()
}

func (e *Engine) SetPaused(paused bool) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.prog == nil && len(e.fading) == 0 {
		paused = false
	}
	e.paused = paused
	e.notify()
}

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

func (e *Engine) isConnected() bool {
	return e.out.Ready()
}

// SceneUpdated applies live-adjustable scene settings (volume, loop) to the
// playing program if it is that scene.
func (e *Engine) SceneUpdated(s db.DJScene) {
	e.mu.Lock()
	defer e.mu.Unlock()
	p := e.prog
	if p == nil || p.sceneID == nil || *p.sceneID != s.ID {
		return
	}
	p.name = s.Name
	p.scale = s.Volume
	if p.cur != nil {
		p.cur.scale = s.Volume
	}
	if p.loop != s.Loop {
		p.loop = s.Loop
		p.nextOrder = nil
		e.prefetchLocked(p)
	}
	e.notify()
}

// ---- State ----

type TrackInfo struct {
	ID         int64  `json:"id"`
	Title      string `json:"title"`
	Artist     string `json:"artist"`
	DurationMs int64  `json:"durationMs"`
}

type State struct {
	Connected   bool        `json:"connected"`
	Playing     bool        `json:"playing"`
	Paused      bool        `json:"paused"`
	SceneID     *int64      `json:"sceneId"`
	PlaylistID  *int64      `json:"playlistId"`
	ProgramName string      `json:"programName"`
	Track       *TrackInfo  `json:"track"`
	PositionMs  int64       `json:"positionMs"`
	At          int64       `json:"at"` // unix ms when PositionMs was sampled
	UpNext      []TrackInfo `json:"upNext"`
	Loop        bool        `json:"loop"`
	Shuffle     bool        `json:"shuffle"`
	Volume      float64     `json:"volume"`
	FadeSeconds float64     `json:"fadeSeconds"`
	Error       string      `json:"error,omitempty"`
}

func itemInfo(it *item) TrackInfo {
	ti := TrackInfo{Title: it.label}
	if it.track != nil {
		ti.ID = it.track.ID
		ti.Artist = it.track.Artist
		ti.DurationMs = it.track.DurationMs
	}
	return ti
}

func (e *Engine) State() State {
	e.mu.Lock()
	defer e.mu.Unlock()
	st := State{
		Connected:   e.connected,
		Paused:      e.paused,
		Volume:      e.volume,
		FadeSeconds: e.fadeSeconds,
		Error:       e.lastErr,
		At:          time.Now().UnixMilli(),
		UpNext:      []TrackInfo{},
	}
	p := e.prog
	if p == nil {
		return st
	}
	st.Playing = true
	st.SceneID = p.sceneID
	st.PlaylistID = p.playlistID
	st.ProgramName = p.name
	st.Loop = p.loop
	st.Shuffle = p.shuffle
	if p.cur != nil {
		ti := itemInfo(p.cur.item)
		st.Track = &ti
		st.PositionMs = p.cur.framesPlayed * FrameMs
	}
	// Up next: the rest of this pass, then (when looping) the next pass.
	for i := p.pos + 1; i < len(p.order) && len(st.UpNext) < 10; i++ {
		st.UpNext = append(st.UpNext, itemInfo(&p.items[p.order[i]]))
	}
	if p.loop && len(st.UpNext) < 10 && len(p.order) > 1 {
		next := p.order
		if p.shuffle {
			if p.nextOrder == nil {
				p.nextOrder = shuffledOrder(len(p.items), p.order[len(p.order)-1])
			}
			next = p.nextOrder
		}
		for i := 0; i < len(next) && len(st.UpNext) < 10; i++ {
			st.UpNext = append(st.UpNext, itemInfo(&p.items[next[i]]))
		}
	}
	return st
}

// notify asks the notify loop to publish state. Non-blocking; bursts coalesce.
func (e *Engine) notify() {
	select {
	case e.notifyCh <- struct{}{}:
	default:
	}
}

// notifyLoop publishes state on change, plus every 5s while playing so
// clients correct any drift in their interpolated position.
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
			active := e.prog != nil && !e.paused && e.connected
			e.mu.Unlock()
			if !active {
				continue
			}
		}
		if e.OnState != nil {
			e.OnState(e.State())
		}
	}
}
