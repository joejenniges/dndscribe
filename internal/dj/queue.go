package dj

import (
	"context"
	"errors"
	"fmt"
	"log"
	"math/rand"
	"path"
	"sort"
	"time"

	"github.com/joe/dndscribe-go/internal/db"
)

// queue is what plays next. A scene or folder fills it; tracks can be added,
// moved and removed by hand. With loop on, an emptied queue refills from the
// source's track list (reshuffled if shuffle is on).
type queue struct {
	entries []queueEntry
	nextID  int64

	sceneID *int64  // scene that filled it
	folder  *string // folder that filled it (Play on a folder)
	name    string  // what to call the source in the UI
	source  []item  // full track list, for loop refills
	loop    bool
	shuffle bool
	lastKey int64 // track id of the last item popped, so a reshuffle doesn't repeat it
}

type queueEntry struct {
	id   int64 // stable, for move/remove from the UI
	it   item
	rank int // position in the source order, for turning shuffle off
}

func (q *queue) push(it item, rank int) {
	q.nextID++
	q.entries = append(q.entries, queueEntry{id: q.nextID, it: it, rank: rank})
}

func itemKey(it *item) int64 {
	if it.track != nil {
		return it.track.ID
	}
	return 0
}

// popLocked takes the next item, refilling from the source when looping.
func (q *queue) popLocked() (item, bool) {
	if len(q.entries) == 0 && q.loop && len(q.source) > 0 {
		q.fill(q.lastKey)
	}
	if len(q.entries) == 0 {
		return item{}, false
	}
	en := q.entries[0]
	q.entries = q.entries[1:]
	q.lastKey = itemKey(&en.it)
	return en.it, true
}

// fill appends one pass over the source, shuffled if shuffle is on, avoiding
// avoidKey as the first entry so a looped shuffle never plays the same track
// twice in a row.
func (q *queue) fill(avoidKey int64) {
	order := make([]int, len(q.source))
	for i := range order {
		order[i] = i
	}
	if q.shuffle {
		order = rand.Perm(len(q.source))
		if len(order) > 1 && itemKey(&q.source[order[0]]) == avoidKey && avoidKey != 0 {
			j := 1 + rand.Intn(len(order)-1)
			order[0], order[j] = order[j], order[0]
		}
	}
	for _, i := range order {
		q.push(q.source[i], i)
	}
}

// replace makes the queue a fresh pass over items.
func (q *queue) replace(items []item, sceneID *int64, folder *string, name string, loop, shuffle bool) {
	q.entries = nil
	q.source = items
	q.sceneID, q.folder, q.name = sceneID, folder, name
	q.loop, q.shuffle = loop, shuffle
	q.lastKey = 0
	q.fill(0)
}

func (q *queue) clear() {
	*q = queue{nextID: q.nextID}
}

// ---- Scene / folder / track: start music now ----

// startNow opens it and makes it the on-air music: on the idle deck with a
// crossfade of fadeSeconds, or straight onto the deck the crossfader points
// at if nothing is playing. From silence only the deck's 200ms play ramp
// applies, so a battle cue lands immediately instead of swelling in. Opening happens before the lock so a slow
// ffmpeg start never stalls the mixer. A later call supersedes an earlier
// one still in progress (latest command wins).
func (e *Engine) startNow(it item, fadeSeconds float64, kind string) error {
	src, err := it.open()
	if err != nil {
		return fmt.Errorf("can't play %s: %w", it.label, err)
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	e.auto = nil // the latest transition replaces any unfinished one

	on := e.onAirLocked()
	itp := &it
	if on < 0 {
		target := e.favouredDeck()
		d := e.decks[target]
		e.replaceLocked(d, itp, src)
		d.auto = true
		d.loading = false
		d.loadSeq++
		d.playing = true
		d.run = 0
		d.runStep = 0
		// An Auto DJ preload on the other deck came from the old queue;
		// drop it. A track the GM loaded by hand stays (and plays next).
		if o := e.decks[1-target]; !o.playing && o.auto {
			e.ejectLocked(o)
		}
		e.lastErr = ""
		e.notify()
		return nil
	}
	target := 1 - on
	d := e.decks[target]
	e.replaceLocked(d, itp, src)
	d.auto = true
	d.loading = false
	d.loadSeq++
	d.playOnLoad = false
	e.startTransitionLocked(target, secondsToFrames(fadeSeconds), kind)
	e.lastErr = ""
	return nil
}

// popAndStart pops the queue (skipping tracks that fail to open) and starts
// the first that opens.
func (e *Engine) popAndStart(fadeSeconds float64, kind string) error {
	var lastErr error
	for tries := 0; tries < 20; tries++ {
		e.mu.Lock()
		it, ok := e.q.popLocked()
		e.mu.Unlock()
		if !ok {
			if lastErr != nil {
				return lastErr
			}
			return errors.New("nothing to play: no tracks in the chosen folders")
		}
		if err := e.startNow(it, fadeSeconds, kind); err != nil {
			log.Printf("DJ: %v", err)
			lastErr = err
			continue
		}
		return nil
	}
	return lastErr
}

// PlayScene makes a scene's tracks the queue and crossfades to the first.
// Tapping the scene that is already playing does nothing, so a double tap on
// the tablet doesn't restart the music.
func (e *Engine) PlayScene(ctx context.Context, sceneID int64, fadeOverride *float64) error {
	if !e.isConnected() {
		return ErrNotConnected
	}
	e.mu.Lock()
	if e.q.sceneID != nil && *e.q.sceneID == sceneID && e.onAirLocked() >= 0 {
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
	if len(scene.Folders) == 0 {
		return fmt.Errorf("scene %q has no folders", scene.Name)
	}
	tracks, err := tracksInFolders(ctx, scene.Folders)
	if err != nil {
		return err
	}
	if len(tracks) == 0 {
		return fmt.Errorf("scene %q has no tracks in its folders", scene.Name)
	}
	items := make([]item, len(tracks))
	for i, t := range tracks {
		items[i] = e.trackItem(t)
		items[i].sceneID = scene.ID
		items[i].scale = scene.Volume
	}
	fade := e.FadeSeconds()
	if scene.FadeSeconds != nil {
		fade = *scene.FadeSeconds
	}
	if fadeOverride != nil {
		fade = *fadeOverride
	}
	id := scene.ID
	e.mu.Lock()
	e.q.replace(items, &id, nil, scene.Name, scene.Loop, scene.Shuffle)
	e.mu.Unlock()
	return e.popAndStart(fade, "scene")
}

// PlayFolder makes a folder (with its sub-folders) the queue, looping, and
// starts it, optionally at a given track.
func (e *Engine) PlayFolder(ctx context.Context, folder string, startTrackID *int64) error {
	if !e.isConnected() {
		return ErrNotConnected
	}
	tracks, err := tracksInFolders(ctx, []string{folder})
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
	name := path.Base(folder)
	if folder == "" {
		name = "All music"
	}
	items := make([]item, len(tracks))
	for i, t := range tracks {
		items[i] = e.trackItem(t)
	}
	f := folder
	e.mu.Lock()
	e.q.replace(items, nil, &f, name, true, false)
	e.mu.Unlock()
	return e.popAndStart(e.FadeSeconds(), "scene")
}

// PlayTrack crossfades to one library track now. The queue carries on
// after it.
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
	return e.startNow(e.trackItem(*t), e.FadeSeconds(), "track")
}

// nextFadeSeconds: Next is a skip, so it's quick, but still a crossfade.
const nextFadeSeconds = 1

// Next crossfades to the next track: whatever is loaded on the idle deck,
// else the head of the queue.
func (e *Engine) Next() error {
	if !e.isConnected() {
		return ErrNotConnected
	}
	e.mu.Lock()
	on := e.onAirLocked()
	if on >= 0 {
		o := e.decks[1-on]
		if o.v != nil && !o.loading && !o.v.done {
			// Already cued (or, with both playing, the other one): go to it.
			e.startTransitionLocked(1-on, secondsToFrames(nextFadeSeconds), "next")
			e.mu.Unlock()
			return nil
		}
	}
	e.mu.Unlock()
	return e.popAndStart(nextFadeSeconds, "next")
}

// stopFadeSeconds is how long Stop all takes by default. Short enough to
// feel instant, long enough not to click.
const stopFadeSeconds = 0.25

// Stop fades out both decks and every effect, unloads the decks and clears
// the queue. fadeSeconds nil = stopFadeSeconds.
func (e *Engine) Stop(fadeSeconds *float64) {
	e.mu.Lock()
	defer e.mu.Unlock()
	fade := stopFadeSeconds
	if fadeSeconds != nil {
		fade = *fadeSeconds
	}
	frames := secondsToFrames(fade)
	e.auto = nil
	for _, d := range e.decks {
		d.playOnLoad = false
		if d.v == nil || d.run <= 0 || frames <= 0 {
			e.ejectLocked(d)
			continue
		}
		d.loadSeq++
		d.loading = false
		d.playing = false
		d.runStep = d.run / float64(frames)
		d.ejectWhenSilent = true
	}
	for _, v := range e.sfx {
		if v.stopping == 0 {
			v.stopping = deClickSamples
		}
	}
	e.q.clear()
	e.pausedByPauseAll = nil
	e.notify()
}

func (e *Engine) stopAllNowLocked() {
	for _, d := range e.decks {
		e.ejectLocked(d)
	}
	e.auto = nil
	e.q.clear()
	e.pausedByPauseAll = nil
}

// SetPaused is the old single-transport pause: pause pauses every playing
// deck; resume restarts those (or the deck the crossfader points at).
func (e *Engine) SetPaused(paused bool) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if paused {
		e.pausedByPauseAll = nil
		for i, d := range e.decks {
			if d.playing {
				d.playing = false
				e.pausedByPauseAll = append(e.pausedByPauseAll, i)
			}
		}
		e.auto = nil
	} else {
		resume := e.pausedByPauseAll
		if len(resume) == 0 {
			resume = []int{e.favouredDeck()}
		}
		for _, i := range resume {
			if d := e.decks[i]; d.v != nil && !d.v.done {
				d.playing = true
				d.runStep = 0
			}
		}
		e.pausedByPauseAll = nil
	}
	e.notify()
}

// SeekTo seeks the on-air deck (the old single-transport seek).
func (e *Engine) SeekTo(positionMs int64) error {
	e.mu.Lock()
	i := e.onAirLocked()
	if i < 0 {
		i = e.favouredDeck()
	}
	e.mu.Unlock()
	return e.SeekDeck(i, positionMs)
}

// SceneUpdated applies live-adjustable scene settings (volume, loop) if that
// scene is playing or queued.
func (e *Engine) SceneUpdated(s db.DJScene) {
	e.mu.Lock()
	defer e.mu.Unlock()
	for _, d := range e.decks {
		for _, v := range append([]*voice{d.v}, d.tails...) {
			if v != nil && v.item != nil && v.item.sceneID == s.ID {
				v.item.scale = s.Volume
			}
		}
	}
	if e.q.sceneID == nil || *e.q.sceneID != s.ID {
		e.notify()
		return
	}
	e.q.name = s.Name
	e.q.loop = s.Loop
	for i := range e.q.entries {
		e.q.entries[i].it.scale = s.Volume
	}
	for i := range e.q.source {
		e.q.source[i].scale = s.Volume
	}
	e.notify()
}

// ---- Queue commands ----

// QueueAdd adds library tracks to the queue: at the front (next=true) or the end.
func (e *Engine) QueueAdd(ctx context.Context, trackIDs []int64, next bool) error {
	var items []item
	for _, id := range trackIDs {
		t, err := db.GetDJTrack(ctx, id)
		if err != nil {
			return err
		}
		if t == nil || t.Missing {
			continue
		}
		items = append(items, e.trackItem(*t))
	}
	if len(items) == 0 {
		return errors.New("no playable tracks")
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	start := len(e.q.entries)
	for _, it := range items {
		e.q.push(it, len(e.q.source)+1)
	}
	if next {
		added := append([]queueEntry(nil), e.q.entries[start:]...)
		e.q.entries = append(added, e.q.entries[:start]...)
	}
	e.notify()
	return nil
}

// QueueMove moves entry id to index to.
func (e *Engine) QueueMove(id int64, to int) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	from := e.q.find(id)
	if from < 0 {
		return errors.New("not in the queue")
	}
	en := e.q.entries[from]
	rest := append(e.q.entries[:from:from], e.q.entries[from+1:]...)
	to = max(0, min(to, len(rest)))
	e.q.entries = append(rest[:to:to], append([]queueEntry{en}, rest[to:]...)...)
	e.notify()
	return nil
}

func (e *Engine) QueueRemove(id int64) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	i := e.q.find(id)
	if i < 0 {
		return errors.New("not in the queue")
	}
	e.q.entries = append(e.q.entries[:i], e.q.entries[i+1:]...)
	e.notify()
	return nil
}

func (e *Engine) QueueClear() {
	e.mu.Lock()
	e.q.entries = nil
	e.q.loop = false // otherwise it would refill straight away
	e.mu.Unlock()
	e.notify()
}

// SetShuffle reorders what's left: shuffled, or back in source order.
func (e *Engine) SetShuffle(on bool) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.q.shuffle = on
	if on {
		rand.Shuffle(len(e.q.entries), func(i, j int) {
			e.q.entries[i], e.q.entries[j] = e.q.entries[j], e.q.entries[i]
		})
	} else {
		sort.SliceStable(e.q.entries, func(i, j int) bool { return e.q.entries[i].rank < e.q.entries[j].rank })
	}
	e.notify()
}

func (e *Engine) SetLoop(on bool) {
	e.mu.Lock()
	e.q.loop = on
	e.mu.Unlock()
	e.notify()
}

func (q *queue) find(id int64) int {
	for i, en := range q.entries {
		if en.id == id {
			return i
		}
	}
	return -1
}

// ---- State ----

type TrackInfo struct {
	ID         int64  `json:"id"`
	Title      string `json:"title"`
	Artist     string `json:"artist"`
	DurationMs int64  `json:"durationMs"`
	Path       string `json:"path,omitempty"` // music-dir-relative, for showing the folder
}

type HistoryEntry struct {
	TrackInfo
	PlayedAt int64 `json:"playedAt"` // unix ms it left the deck
}

type QueueItem struct {
	QID int64 `json:"qid"`
	TrackInfo
}

type DeckState struct {
	Track      *TrackInfo `json:"track"`
	Playing    bool       `json:"playing"`
	Ended      bool       `json:"ended"`
	PositionMs int64      `json:"positionMs"`
	Gain       float64    `json:"gain"`
	Loading    bool       `json:"loading"`
	Auto       bool       `json:"auto"` // loaded by Auto DJ / a scene, not by hand
	Error      string     `json:"error,omitempty"`
}

// Transition describes an automated crossfader move so clients can animate
// the fader themselves instead of receiving a position every frame.
type Transition struct {
	Kind       string  `json:"kind"` // scene, auto, next, track, manual (A/B buttons: fader only)
	ToDeck     int     `json:"toDeck"`
	From       float64 `json:"from"`
	To         float64 `json:"to"`
	StartedAt  int64   `json:"startedAt"` // unix ms
	DurationMs int64   `json:"durationMs"`
}

type State struct {
	Rev       int64 `json:"rev"`
	At        int64 `json:"at"` // unix ms when positions were sampled
	Connected bool  `json:"connected"`

	Decks      [2]DeckState `json:"decks"`
	OnAir      int          `json:"onAir"` // deck the audience mainly hears, -1 = none
	Crossfader float64      `json:"crossfader"`
	Transition *Transition  `json:"transition"`
	AutoDJ     bool         `json:"autoDJ"`

	Volume      float64 `json:"volume"` // master
	FadeSeconds float64 `json:"fadeSeconds"`

	SceneID    *int64         `json:"sceneId"`
	Folder     *string        `json:"folder"`
	SourceName string         `json:"sourceName"`
	Queue      []QueueItem    `json:"queue"`
	QueueTotal int            `json:"queueTotal"`
	Loop       bool           `json:"loop"`
	Shuffle    bool           `json:"shuffle"`
	History    []HistoryEntry `json:"history"`

	Error string `json:"error,omitempty"`

	Sfx     []SfxPlaying `json:"sfx"`
	SfxGain float64      `json:"sfxGain"`

	// Single-transport view of the above, for the pre-v3 page.
	Playing     bool        `json:"playing"`
	Paused      bool        `json:"paused"`
	ProgramName string      `json:"programName"`
	Track       *TrackInfo  `json:"track"`
	PositionMs  int64       `json:"positionMs"`
	UpNext      []TrackInfo `json:"upNext"`
}

func itemInfo(it *item) TrackInfo {
	ti := TrackInfo{Title: it.label}
	if it.track != nil {
		ti.ID = it.track.ID
		ti.Artist = it.track.Artist
		ti.DurationMs = it.track.DurationMs
		ti.Path = it.track.Path
	}
	return ti
}

// maxQueueInState: a whole-library queue can be hundreds of tracks; the
// snapshot carries the first 100 and the total.
const maxQueueInState = 100

func (e *Engine) State() State {
	e.mu.Lock()
	defer e.mu.Unlock()
	st := State{
		Rev:         e.rev.Load(),
		At:          time.Now().UnixMilli(),
		Connected:   e.connected,
		OnAir:       e.onAirLocked(),
		Crossfader:  e.xTarget,
		AutoDJ:      e.autoDJ,
		Volume:      e.volume,
		FadeSeconds: e.fadeSeconds,
		SceneID:     e.q.sceneID,
		Folder:      e.q.folder,
		SourceName:  e.q.name,
		Queue:       []QueueItem{},
		QueueTotal:  len(e.q.entries),
		Loop:        e.q.loop,
		Shuffle:     e.q.shuffle,
		History:     append([]HistoryEntry{}, e.history...),
		Error:       e.lastErr,
		Sfx:         e.sfxStateLocked(),
		SfxGain:     e.sfxGain,
		UpNext:      []TrackInfo{},
	}
	for i, d := range e.decks {
		ds := DeckState{Playing: d.playing, Gain: d.gain, Loading: d.loading, Auto: d.auto, Error: d.err}
		if d.v != nil {
			ti := itemInfo(d.v.item)
			ds.Track = &ti
			ds.PositionMs = d.v.framesPlayed * FrameMs
			ds.Ended = d.v.done
		}
		st.Decks[i] = ds
	}
	if a := e.auto; a != nil {
		st.Transition = &Transition{
			Kind: a.kind, ToDeck: a.to, From: a.from, To: a.target,
			StartedAt:  a.startedAt.UnixMilli(),
			DurationMs: int64(a.frames) * FrameMs,
		}
		// The fader position as of now, so a client joining mid-move
		// starts from the right place.
		st.Crossfader = e.xfade
	}
	for i, en := range e.q.entries {
		if i >= maxQueueInState {
			break
		}
		st.Queue = append(st.Queue, QueueItem{QID: en.id, TrackInfo: itemInfo(&en.it)})
	}

	// Legacy single-transport fields.
	main := st.OnAir
	if main < 0 {
		for i, d := range e.decks {
			if d.v != nil {
				main = i
				break
			}
		}
	}
	if main >= 0 && e.decks[main].v != nil {
		ds := st.Decks[main]
		st.Track = ds.Track
		st.PositionMs = ds.PositionMs
		st.Playing = true
		st.Paused = st.OnAir < 0
	}
	st.ProgramName = e.q.name
	if st.ProgramName == "" && st.Track != nil {
		st.ProgramName = st.Track.Title
	}
	if main >= 0 {
		if o := e.decks[1-main]; o.v != nil && !o.playing {
			st.UpNext = append(st.UpNext, itemInfo(o.v.item))
		}
	}
	for i := 0; i < len(e.q.entries) && len(st.UpNext) < 10; i++ {
		st.UpNext = append(st.UpNext, itemInfo(&e.q.entries[i].it))
	}
	return st
}
