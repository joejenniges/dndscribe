package dj

import (
	"context"
	"errors"
	"fmt"
	"log"
	"time"

	"github.com/joe/dndscribe-go/internal/db"
)

// Hotcues and goto triggers (v4).
//
// Definitions live in the database and are edited over HTTP; the engine
// keeps a copy per track (e.cues) that the mixer reads every frame without
// touching the database. An edit replaces the copy (SetTrackCues) and takes
// effect on the next frame. A track's cues are loaded off the mixer lock
// when it's put on a deck (ensureCues).
//
// Goto runtime rules (all evaluated on the mixer's own frame clock, never
// browser timers):
//
//   - A goto fires when forward playback crosses its position: the frame
//     that advances the deck from p0 to p0+20ms fires it if p0 < pos <= p0+20.
//     Half-open on purpose: landing exactly on a goto by a seek (or a jump)
//     doesn't fire it; the next natural crossing does.
//   - Seeks and jumps are not crossings, so triggers skipped over by a
//     seek never fire.
//   - Earliest goto in the frame wins (position, then id); one jump per
//     frame at most, then gotoHoldFrames of normal playback before any
//     goto on that deck can fire again.
//   - A goto is only live if it's enabled, its target hotcue exists, both
//     positions are within the track, and target and trigger are at least
//     minGotoSpanMs apart. Anything else is reported in the deck state and
//     ignored by the mixer (fail open: the track just keeps playing).
//   - A deck being faded out by an automated transition (scene change,
//     Auto DJ, Next) doesn't fire gotos: it's on its way out.
//   - Paused decks produce no frames, so nothing fires (the 200ms pause
//     ramp still counts as playback); unloading drops the deck's pending
//     state.
//
// Seamless loops: a goto jump needs a decoder already running at the
// target, because opening ffmpeg takes ~100ms. While a deck plays, the next
// armed goto's target is opened ahead of time (d.prep) and swapped in on
// the crossing frame with the usual 40ms declick crossfade. If it isn't
// ready (just armed, slow disk), the jump falls back to a normal seek.

// minGotoSpanMs: a goto and its target must be at least this far apart. A
// smaller backward loop would replay a fraction of a second forever.
const minGotoSpanMs = 500

// gotoHoldFrames: after a goto jump, this many frames (200ms) play before
// any goto on that deck can fire again. Guards against chains of gotos
// bouncing between each other within one moment.
const gotoHoldFrames = 10

// CueLoader reads a track's cue definitions (db.GetDJTrackCues in the app).
type CueLoader func(ctx context.Context, trackID int64) (*db.DJTrackCues, error)

// SetCueLoader sets where the engine reads cue definitions from. Without one
// (tests), tracks have no cues unless SetTrackCues provides them.
func (e *Engine) SetCueLoader(f CueLoader) {
	e.mu.Lock()
	e.cueLoader = f
	e.mu.Unlock()
}

// ensureCues loads a track's cues into the engine's copy if it doesn't have
// them yet. Called off the mixer lock while a track is opened. An edit that
// lands meanwhile (SetTrackCues) wins: a load only fills a gap.
func (e *Engine) ensureCues(it *item) {
	if it == nil || it.track == nil {
		return
	}
	id := it.track.ID
	e.mu.Lock()
	_, have := e.cues[id]
	load := e.cueLoader
	e.mu.Unlock()
	if have || load == nil {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	c, err := load(ctx, id)
	if err != nil {
		// The track still plays; it just has no cues this time.
		log.Printf("DJ: load cues for track %d: %v", id, err)
		return
	}
	e.mu.Lock()
	if _, ok := e.cues[id]; !ok {
		e.cues[id] = c
	}
	e.mu.Unlock()
}

// SetTrackCues replaces the engine's copy of a track's cues after an edit
// has been saved. Decks playing the track use it from the next frame.
func (e *Engine) SetTrackCues(c *db.DJTrackCues) {
	if c == nil {
		return
	}
	e.mu.Lock()
	e.cues[c.TrackID] = c
	e.mu.Unlock()
	e.notify()
}

// ForgetTrackCues drops a cached copy (track deleted).
func (e *Engine) ForgetTrackCues(trackID int64) {
	e.mu.Lock()
	delete(e.cues, trackID)
	e.mu.Unlock()
}

func (e *Engine) deckCuesLocked(d *deck) *db.DJTrackCues {
	if d.v == nil || d.v.item == nil || d.v.item.track == nil {
		return nil
	}
	return e.cues[d.v.item.track.ID]
}

func hotcuePos(c *db.DJTrackCues, slot int) (int64, bool) {
	if c == nil {
		return 0, false
	}
	for _, h := range c.Hotcues {
		if h.Slot == slot {
			return h.PositionMs, true
		}
	}
	return 0, false
}

// gotoProblem says why a goto can't fire ("" = it can, if enabled).
func gotoProblem(c *db.DJTrackCues, g db.DJGoto, durationMs int64) string {
	target, ok := hotcuePos(c, g.TargetSlot)
	if !ok {
		return "target missing"
	}
	if durationMs > 0 && (g.PositionMs > durationMs || target > durationMs) {
		return "past the end of the track"
	}
	if d := g.PositionMs - target; d < minGotoSpanMs && d > -minGotoSpanMs {
		return "too close to its target"
	}
	return ""
}

// ValidateGoto checks a goto against a track's hotcues before it's saved.
func ValidateGoto(c *db.DJTrackCues, g db.DJGoto, durationMs int64) error {
	if g.TargetSlot < 1 || g.TargetSlot > db.HotcueSlots {
		return fmt.Errorf("target must be hotcue 1-%d", db.HotcueSlots)
	}
	if g.PositionMs < 0 || (durationMs > 0 && g.PositionMs > durationMs) {
		return errors.New("the goto is outside the track")
	}
	if p := gotoProblem(c, g, durationMs); p != "" {
		if p == "too close to its target" {
			return fmt.Errorf("a goto must be at least %d ms from its target hotcue", minGotoSpanMs)
		}
		return fmt.Errorf("goto %s", p)
	}
	return nil
}

// gotosSuppressedLocked: deck i is being faded away from by an automated
// transition, so its gotos don't fire.
func (e *Engine) gotosSuppressedLocked(d *deck) bool {
	a := e.auto
	return a != nil && a.retire && a.to != d.idx
}

// checkGotoLocked runs after deck d's voice produced the frame that took it
// from frame p0 to p0+1. It fires the earliest live goto crossed.
func (e *Engine) checkGotoLocked(d *deck, p0 int64) {
	if d.gotoHold > 0 {
		d.gotoHold--
		return
	}
	// d.run > 0: still sounding through the pause ramp. That audio is real
	// forward playback, so a crossing during it fires too; otherwise a
	// pause just before a goto would let the ramp carry past it unnoticed.
	if (!d.playing && d.run <= 0) || d.v == nil || e.gotosSuppressedLocked(d) {
		return
	}
	c := e.deckCuesLocked(d)
	if c == nil || len(c.Gotos) == 0 {
		return
	}
	lo, hi := p0*FrameMs, (p0+1)*FrameMs
	dur := d.v.item.durationMs()
	for _, g := range c.Gotos { // sorted by position, then id
		if g.PositionMs <= lo {
			continue
		}
		if g.PositionMs > hi {
			break
		}
		if !g.Enabled || gotoProblem(c, g, dur) != "" {
			continue
		}
		target, _ := hotcuePos(c, g.TargetSlot)
		e.jumpLocked(d, target, CueJump{Kind: "goto", Slot: g.TargetSlot, GotoID: g.ID, FromMs: g.PositionMs, ToMs: target})
		d.gotoHold = gotoHoldFrames
		return
	}
}

// CueJump describes the last hotcue or goto jump on a deck, for the UI.
type CueJump struct {
	Kind   string `json:"kind"` // hotcue, goto
	Slot   int    `json:"slot"`
	GotoID int64  `json:"gotoId,omitempty"`
	FromMs int64  `json:"fromMs"`
	ToMs   int64  `json:"toMs"`
	At     int64  `json:"at"` // unix ms
	// Seamless: served from the pre-opened decoder (no gap). False means
	// it fell back to a normal seek.
	Seamless bool `json:"seamless"`
}

// jumpLocked moves deck d to posMs now: from the pre-opened decoder if it's
// at that position, otherwise by a background seek.
func (e *Engine) jumpLocked(d *deck, posMs int64, j CueJump) {
	j.At = time.Now().UnixMilli()
	if p := d.prep; p != nil && p.src != nil && p.posMs == posMs && d.v != nil && p.item == d.v.item {
		d.prep = nil
		e.swapVoiceLocked(d, p.src, posMs)
		j.Seamless = true
	} else {
		e.seekAsyncLocked(d, posMs)
	}
	d.lastJump = &j
	e.notify()
}

// swapVoiceLocked replaces d's voice with src (already positioned at posMs),
// with the same 40ms declick as a seek.
func (e *Engine) swapVoiceLocked(d *deck, src Source, posMs int64) {
	old := d.v
	if d.playing || d.run > 0 {
		old.env = fadeOut(seekFadeFrames)
		d.tails = append(d.tails, old)
	} else {
		old.src.Close()
	}
	d.v = &voice{src: src, item: old.item, env: fadeIn(seekFadeFrames), framesPlayed: posMs / FrameMs}
}

// seekAsyncLocked seeks d in the background (the jump fallback). Same
// supersede rules as SeekDeck: only the latest seek is applied.
func (e *Engine) seekAsyncLocked(d *deck, posMs int64) {
	if d.v == nil || d.v.item.openAt == nil {
		return
	}
	it := d.v.item
	d.seekSeq++
	seq := d.seekSeq
	go func() {
		src, err := it.openAt(posMs)
		e.mu.Lock()
		defer e.mu.Unlock()
		if err != nil {
			e.setErrLocked(fmt.Sprintf("Cue jump on deck %s failed: %v", deckName(d.idx), err))
			return
		}
		if seq != d.seekSeq || d.v == nil || d.v.item != it {
			src.Close()
			return
		}
		e.swapVoiceLocked(d, src, posMs)
		e.notify()
	}()
}

// prepared is a decoder opened ahead at a goto's target position.
type prepared struct {
	item  *item
	posMs int64
	src   Source // nil while opening
}

func (d *deck) dropPrep() {
	if d.prep != nil {
		if d.prep.src != nil {
			d.prep.src.Close()
		}
		d.prep = nil
	}
}

// nextGotoTargetLocked is the target position of the next armed, live goto
// ahead of the deck's position, or -1.
func (e *Engine) nextGotoTargetLocked(d *deck) int64 {
	c := e.deckCuesLocked(d)
	if c == nil {
		return -1
	}
	pos := d.v.framesPlayed * FrameMs
	dur := d.v.item.durationMs()
	for _, g := range c.Gotos {
		if g.PositionMs <= pos || !g.Enabled || gotoProblem(c, g, dur) != "" {
			continue
		}
		t, _ := hotcuePos(c, g.TargetSlot)
		return t
	}
	return -1
}

// ensurePrepLocked keeps a decoder open at the next goto's target while the
// deck plays, and drops it when there's no goto ahead.
func (e *Engine) ensurePrepLocked(d *deck) {
	if d.v == nil || d.v.item.openAt == nil || !d.playing {
		return
	}
	target := e.nextGotoTargetLocked(d)
	if target < 0 {
		d.dropPrep()
		return
	}
	if p := d.prep; p != nil && p.item == d.v.item && p.posMs == target {
		return // ready, or opening
	}
	d.dropPrep()
	p := &prepared{item: d.v.item, posMs: target}
	d.prep = p
	it := d.v.item
	go func() {
		src, err := it.openAt(target)
		e.mu.Lock()
		defer e.mu.Unlock()
		if err != nil {
			if d.prep == p {
				d.prep = nil
			}
			log.Printf("DJ: prepare goto target on deck %s: %v", deckName(d.idx), err)
			return
		}
		if d.prep != p {
			src.Close() // no longer wanted
			return
		}
		p.src = src
	}()
}

// HotcueDeck jumps deck i to hotcue slot of its loaded track. Play state is
// unchanged: a paused deck stays paused at the cue.
func (e *Engine) HotcueDeck(i, slot int) error {
	if err := checkDeck(i); err != nil {
		return err
	}
	if slot < 1 || slot > db.HotcueSlots {
		return fmt.Errorf("hotcue must be 1-%d", db.HotcueSlots)
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	d := e.decks[i]
	if d.v == nil {
		return ErrNotPlaying
	}
	pos, ok := hotcuePos(e.deckCuesLocked(d), slot)
	if !ok {
		return fmt.Errorf("hotcue %d isn't set on this track", slot)
	}
	if d.v.item.openAt == nil {
		return errors.New("this can't be seeked")
	}
	from := d.v.framesPlayed * FrameMs
	e.jumpLocked(d, pos, CueJump{Kind: "hotcue", Slot: slot, FromMs: from, ToMs: pos})
	return nil
}
