package dj

import (
	"context"
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/joe/dndscribe-go/internal/db"
)

// deck is one music transport: a loaded track that plays, pauses and seeks
// on its own. What reaches the mix also depends on the crossfader.
type deck struct {
	idx   int
	v     *voice   // loaded track; nil = empty
	tails []*voice // old decoders fading out after a seek or reload

	playing bool
	run     float64 // play ramp 0..1, so play/pause doesn't click
	runStep float64 // ramp change per frame; 0 = default (200ms)

	gain    float64 // channel gain target, 0..1
	curGain float64 // smoothed

	loading bool
	loadSeq int64
	seekSeq int64

	// auto: loaded by Auto DJ from the queue (rather than by hand).
	auto bool
	// playOnLoad: Auto DJ needed this deck to start the moment its track
	// finished loading (the other deck ended before the preload was ready).
	playOnLoad bool
	// ejectWhenSilent: Stop all — unload once the fade-out reaches zero.
	ejectWhenSilent bool

	err  string
	peak float64 // meter accumulator
}

const defaultRunStep = 1.0 / 10 // 200ms play/pause ramp

// sounding reports whether the deck needs mixing this frame.
func (d *deck) sounding() bool {
	return (d.v != nil && (d.playing || d.run > 0)) || len(d.tails) > 0
}

// audible: playing a track (the crossfader may still have it at zero).
func (d *deck) audible() bool { return d.v != nil && d.playing && !d.v.done }

func deckName(i int) string { return string(rune('A' + i)) }

// mixDeck adds one frame of d (after its gain and play ramp) into acc.
func (e *Engine) mixDeck(d *deck, acc []float64) bool {
	var tmp [FrameLen]float64
	contributed := false

	target := 0.0
	if d.playing {
		target = 1
	}
	step := d.runStep
	if step <= 0 {
		step = defaultRunStep
	}
	r0 := d.run
	r1 := r0
	if target > r0 {
		r1 = math.Min(target, r0+step)
	} else if target < r0 {
		r1 = math.Max(target, r0-step)
	}

	if d.v != nil && (d.playing || r0 > 0) {
		scale := 1.0
		if d.v.item != nil && d.v.item.scale > 0 {
			scale = d.v.item.scale
		}
		contributed = mixVoice(d.v, tmp[:], scale)
		if d.v.done && d.playing {
			d.playing = false
			r1 = 0
			e.notify()
		}
	}
	kept := d.tails[:0]
	for _, t := range d.tails {
		scale := 1.0
		if t.item != nil && t.item.scale > 0 {
			scale = t.item.scale
		}
		contributed = mixVoice(t, tmp[:], scale) || contributed
		if t.done || t.env.finishedOut() {
			t.src.Close()
			continue
		}
		kept = append(kept, t)
	}
	for i := len(kept); i < len(d.tails); i++ {
		d.tails[i] = nil
	}
	d.tails = kept

	g0, g1 := d.curGain*r0, d.gain*r1
	peak := 0.0
	if contributed {
		for i := 0; i < FrameSamples; i++ {
			g := g0 + (g1-g0)*float64(i)/FrameSamples
			for c := 0; c < Channels; c++ {
				j := i*Channels + c
				s := tmp[j] * g
				acc[j] += s
				peak = math.Max(peak, math.Abs(s))
			}
		}
	}
	d.peak = math.Max(d.peak, math.Min(1, peak/math.MaxInt16))
	d.curGain = d.gain
	d.run = r1
	if r1 == 0 && !d.playing {
		d.runStep = 0
		if d.ejectWhenSilent {
			e.ejectLocked(d)
			e.notify()
		}
	}
	return contributed
}

// ---- Crossfader automation and Auto DJ ----

// autoFade is an automated crossfader move toward deck `to`.
type autoFade struct {
	kind      string // "scene", "auto", "next", "track"
	to        int
	from      float64
	target    float64
	frames    int
	pos       int
	startedAt time.Time
}

// onAirLocked is the deck the audience is mainly hearing: the playing one, or
// with both playing, the one the crossfader favours. -1 = neither.
func (e *Engine) onAirLocked() int {
	a, b := e.decks[0].audible(), e.decks[1].audible()
	switch {
	case a && !b:
		return 0
	case b && !a:
		return 1
	case a && b:
		if e.xfade <= 0.5 {
			return 0
		}
		return 1
	}
	return -1
}

// favouredDeck is where the crossfader currently points, used to start music
// from silence without moving the fader.
func (e *Engine) favouredDeck() int {
	if e.xTarget <= 0.5 {
		return 0
	}
	return 1
}

func deckSide(i int) float64 { return float64(i) }

// startTransitionLocked starts deck `to` and moves the crossfader to it over
// frames. frames <= 0 cuts over at once.
func (e *Engine) startTransitionLocked(to int, frames int, kind string) {
	d := e.decks[to]
	d.playing = true
	d.run = 1 // the crossfader does the fading in
	d.runStep = 0
	e.auto = nil
	target := deckSide(to)
	if frames <= 0 || e.xTarget == target {
		e.xTarget = target
		e.xfade = target
		e.retireOtherLocked(to)
		e.notify()
		return
	}
	e.auto = &autoFade{kind: kind, to: to, from: e.xTarget, target: target, frames: frames, startedAt: time.Now()}
	e.notify()
}

// finishAutoLocked completes an automated move once it reaches its target:
// the deck faded away from is stopped and unloaded (into history).
func (e *Engine) finishAutoLocked() {
	a := e.auto
	if a == nil || a.pos < a.frames {
		return
	}
	e.auto = nil
	e.xTarget, e.xfade = a.target, a.target
	e.retireOtherLocked(a.to)
	e.notify()
}

// retireOtherLocked unloads the deck that isn't `keep`, once it's been
// crossfaded out, so Auto DJ can load the next track there.
func (e *Engine) retireOtherLocked(keep int) {
	o := e.decks[1-keep]
	if o.v != nil && o.playing {
		o.playing = false
		o.run = 0
		e.ejectLocked(o)
	}
}

// trackEndedLocked handles deck i's track running out. With Auto DJ it cuts
// to the other deck at once (gapless) and returns that deck's index, or -1.
func (e *Engine) trackEndedLocked(i int) int {
	d := e.decks[i]
	o := e.decks[1-i]
	if !e.autoDJ {
		return -1 // stays loaded at its end; Play starts it again
	}
	if e.auto != nil && e.auto.to == 1-i {
		// Already crossfading to the other deck: just let this one go.
		e.ejectLocked(d)
		e.notify()
		return -1
	}
	if o.audible() {
		e.ejectLocked(d)
		e.notify()
		return -1
	}
	e.ejectLocked(d)
	if o.v != nil && !o.loading {
		e.startTransitionLocked(1-i, 0, "auto")
		return 1 - i
	}
	// The next track isn't ready (a very short track, or a slow disk):
	// play it the moment it loads.
	if o.loading {
		o.playOnLoad = true
	} else if it, ok := e.q.popLocked(); ok {
		e.cueAsyncLocked(1-i, it, true)
	}
	e.notify()
	return -1
}

// autoLeadFrames: how long before the on-air track ends Auto DJ starts the
// next one. The configured fade, but never more than a third of either track
// so short tracks still get most of their play time.
func (e *Engine) autoLeadMs(on, next *item) int64 {
	ms := int64(e.fadeSeconds * 1000)
	if d := on.durationMs(); d > 0 {
		ms = min(ms, d/3)
	}
	if d := next.durationMs(); d > 0 {
		ms = min(ms, d/3)
	}
	return ms
}

// autoStepLocked runs once per frame: Auto DJ preloads the next queue track
// onto the idle deck, and starts the crossfade when the on-air track is
// nearly over.
func (e *Engine) autoStepLocked() {
	if !e.autoDJ || e.auto != nil {
		return
	}
	on := e.onAirLocked()
	if on < 0 {
		return
	}
	d, o := e.decks[on], e.decks[1-on]
	if o.v == nil && !o.loading {
		if it, ok := e.q.popLocked(); ok {
			e.cueAsyncLocked(1-on, it, false)
		}
		return
	}
	if o.v == nil || o.loading || o.playing || d.v.item.durationMs() <= 0 {
		return // no duration known: the gapless cut at the end handles it
	}
	lead := e.autoLeadMs(d.v.item, o.v.item)
	if lead <= 0 {
		return
	}
	remaining := d.v.item.durationMs() - d.v.framesPlayed*FrameMs
	if remaining <= lead {
		e.startTransitionLocked(1-on, int(remaining/FrameMs), "auto")
	}
}

// ---- Loading ----

// replaceLocked puts a freshly opened source on deck d, fading out whatever
// was there (40ms, just enough not to click).
func (e *Engine) replaceLocked(d *deck, it *item, src Source) {
	// The 40ms fade-in is only for cutting in over audio this deck was
	// already playing. A track starting from silence starts clean, so a
	// gapless change between tracks has no dip.
	env := envelope{kind: envSteady}
	if d.playing || d.run > 0 {
		env = fadeIn(seekFadeFrames)
	}
	if d.v != nil {
		if d.playing || d.run > 0 {
			old := d.v
			old.env = fadeOut(seekFadeFrames)
			d.tails = append(d.tails, old)
		} else {
			d.v.src.Close()
		}
		e.pushHistoryLocked(d.v)
	}
	d.v = &voice{src: src, item: it, env: env}
	d.err = ""
	d.ejectWhenSilent = false
}

// ejectLocked unloads deck d.
func (e *Engine) ejectLocked(d *deck) {
	if d.v != nil {
		e.pushHistoryLocked(d.v)
		d.v.src.Close()
		d.v = nil
	}
	for _, t := range d.tails {
		t.src.Close()
	}
	d.tails = nil
	d.playing = false
	d.run = 0
	d.runStep = 0
	d.auto = false
	d.playOnLoad = false
	d.ejectWhenSilent = false
	d.loadSeq++ // abandon any load in flight
	d.loading = false
}

// cueAsyncLocked opens it in the background and loads it onto deck i (Auto
// DJ preload). Opening ffmpeg takes ~100ms, far too long to hold the mixer.
func (e *Engine) cueAsyncLocked(i int, it item, playOnLoad bool) {
	d := e.decks[i]
	d.loadSeq++
	seq := d.loadSeq
	d.loading = true
	d.playOnLoad = playOnLoad
	itp := &it
	go func() {
		src, err := itp.open()
		e.mu.Lock()
		defer e.mu.Unlock()
		if d.loadSeq != seq {
			if src != nil {
				src.Close()
			}
			return
		}
		d.loading = false
		if err != nil {
			d.err = fmt.Sprintf("Can't play %s: %v", itp.label, err)
			e.setErrLocked(d.err)
			return
		}
		e.replaceLocked(d, itp, src)
		d.auto = true
		if d.playOnLoad {
			d.playOnLoad = false
			e.startTransitionLocked(i, 0, "auto")
		}
		e.notify()
	}()
}

// openOnDeck opens it (outside the lock) and loads it onto deck i, replacing
// what's there. The deck is left paused at the start.
func (e *Engine) openOnDeck(i int, it *item) error {
	e.mu.Lock()
	d := e.decks[i]
	d.loadSeq++
	seq := d.loadSeq
	d.loading = true
	e.notify()
	e.mu.Unlock()

	src, err := it.open()

	e.mu.Lock()
	defer e.mu.Unlock()
	if d.loadSeq != seq {
		if src != nil {
			src.Close()
		}
		return errors.New("superseded by another load")
	}
	d.loading = false
	if err != nil {
		d.err = fmt.Sprintf("Can't play %s: %v", it.label, err)
		e.notify()
		return errors.New(d.err)
	}
	if d.playing {
		d.playing = false
		d.run = 0
	}
	e.replaceLocked(d, it, src)
	d.auto = false
	e.notify()
	return nil
}

func (e *Engine) pushHistoryLocked(v *voice) {
	if v == nil || v.item == nil || v.item.track == nil || v.framesPlayed < 50 {
		return // never really played (under a second)
	}
	h := HistoryEntry{TrackInfo: itemInfo(v.item), PlayedAt: time.Now().UnixMilli()}
	e.history = append([]HistoryEntry{h}, e.history...)
	if len(e.history) > maxHistory {
		e.history = e.history[:maxHistory]
	}
}

const maxHistory = 30

// ---- Deck commands ----

func checkDeck(i int) error {
	if i != 0 && i != 1 {
		return fmt.Errorf("deck must be 0 (A) or 1 (B)")
	}
	return nil
}

// LoadDeck loads a library track onto a deck, paused at the start.
func (e *Engine) LoadDeck(ctx context.Context, i int, trackID int64) error {
	if err := checkDeck(i); err != nil {
		return err
	}
	t, err := db.GetDJTrack(ctx, trackID)
	if err != nil {
		return err
	}
	if t == nil || t.Missing {
		return errors.New("track not found")
	}
	it := e.trackItem(*t)
	return e.openOnDeck(i, &it)
}

// PlayDeck starts a deck. A deck whose track has ended starts it again.
func (e *Engine) PlayDeck(i int) error {
	if err := checkDeck(i); err != nil {
		return err
	}
	if !e.isConnected() {
		return ErrNotConnected
	}
	e.mu.Lock()
	d := e.decks[i]
	if d.v == nil {
		e.mu.Unlock()
		if d.loading {
			return fmt.Errorf("deck %s is still loading", deckName(i))
		}
		return fmt.Errorf("deck %s is empty", deckName(i))
	}
	if d.v.done {
		it := d.v.item
		e.mu.Unlock()
		if err := e.openOnDeck(i, it); err != nil {
			return err
		}
		e.mu.Lock()
	}
	d.playing = true
	d.runStep = 0
	d.ejectWhenSilent = false
	e.pausedByPauseAll = nil
	e.mu.Unlock()
	e.notify()
	return nil
}

func (e *Engine) PauseDeck(i int) error {
	if err := checkDeck(i); err != nil {
		return err
	}
	e.mu.Lock()
	d := e.decks[i]
	d.playing = false
	d.playOnLoad = false
	if e.auto != nil && e.auto.to == i {
		e.auto = nil // a paused deck can't be faded to
	}
	e.mu.Unlock()
	e.notify()
	return nil
}

func (e *Engine) EjectDeck(i int) error {
	if err := checkDeck(i); err != nil {
		return err
	}
	e.mu.Lock()
	d := e.decks[i]
	if d.playing && d.v != nil {
		d.tails = append(d.tails, &voice{src: d.v.src, item: d.v.item, env: fadeOut(seekFadeFrames)})
		e.pushHistoryLocked(d.v)
		d.v = nil
	}
	tails := d.tails
	d.tails = nil
	e.ejectLocked(d)
	d.tails = tails // let the cut finish its 40ms ramp
	if e.auto != nil && e.auto.to == i {
		e.auto = nil
	}
	e.mu.Unlock()
	e.notify()
	return nil
}

// SetDeckGain sets a deck's channel gain, 0..1.
func (e *Engine) SetDeckGain(i int, g float64) error {
	if err := checkDeck(i); err != nil {
		return err
	}
	e.mu.Lock()
	e.decks[i].gain = clamp01(g)
	e.mu.Unlock()
	e.notify()
	return nil
}

// SetCrossfader moves the crossfader by hand. It cancels any automated
// transition at once, so the automation never fights the user.
func (e *Engine) SetCrossfader(x float64) {
	e.mu.Lock()
	e.auto = nil
	e.xTarget = clamp01(x)
	e.mu.Unlock()
	e.notify()
}

func (e *Engine) SetAutoDJ(on bool) {
	e.mu.Lock()
	e.autoDJ = on
	if !on {
		for _, d := range e.decks {
			d.playOnLoad = false
		}
	}
	e.mu.Unlock()
	e.notify()
}

// ErrNotPlaying is returned by a seek when the deck is empty.
var ErrNotPlaying = errors.New("nothing is loaded")

// seekFadeFrames: a seek swaps decoders with a 40ms crossfade, just enough
// to avoid a click at the cut.
const seekFadeFrames = 2

// SeekDeck jumps deck i to positionMs. It restarts the decoder at that point
// (ffmpeg -ss), quick for the formats we play but not instant, so there can
// be a few tens of ms of silence after the cut. Overlapping seeks (dragging
// on a slow machine) are fine: only the latest is applied.
func (e *Engine) SeekDeck(i int, positionMs int64) error {
	if err := checkDeck(i); err != nil {
		return err
	}
	e.mu.Lock()
	d := e.decks[i]
	if d.v == nil {
		e.mu.Unlock()
		return ErrNotPlaying
	}
	it := d.v.item
	if it.openAt == nil {
		e.mu.Unlock()
		return errors.New("this can't be seeked")
	}
	if dur := it.durationMs(); dur > 0 {
		// Past the end would just end the track; land just before it.
		positionMs = min(positionMs, max(0, dur-500))
	}
	positionMs = max(0, positionMs)
	d.seekSeq++
	seq := d.seekSeq
	e.mu.Unlock()

	src, err := it.openAt(positionMs) // ffmpeg start, off the mixer lock
	if err != nil {
		return fmt.Errorf("seek: %w", err)
	}

	e.mu.Lock()
	defer e.mu.Unlock()
	if seq != d.seekSeq || d.v == nil || d.v.item != it {
		src.Close() // superseded, or the track changed meanwhile
		return nil
	}
	old := d.v
	if d.playing || d.run > 0 {
		old.env = fadeOut(seekFadeFrames)
		d.tails = append(d.tails, old)
	} else {
		// A paused deck isn't mixed, so a fading voice would hang around
		// and play its tail on resume.
		old.src.Close()
	}
	// Seeking the on-air track into Auto DJ's lead time starts the
	// transition on the next frame (autoStepLocked).
	d.v = &voice{src: src, item: it, env: fadeIn(seekFadeFrames), framesPlayed: positionMs / FrameMs}
	e.notify()
	return nil
}
