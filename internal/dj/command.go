package dj

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"time"
)

// Live DJ controls over the WebSocket.
//
// Client -> server:
//
//	{"type":"dj.cmd","id":"<unique per press>","cmd":"deck.play","args":{"deck":0}}
//
// Server -> that client only:
//
//	{"type":"dj.ack","id":"...","ok":true,"rev":1234}
//	{"type":"dj.ack","id":"...","ok":false,"error":"the bot is not in a voice channel","rev":1234}
//
// Everyone then gets the full dj_state snapshot (rev >= the ack's rev) via
// the normal broadcast. There are no per-field events: a snapshot is the
// whole truth, so a client that missed messages just uses the next one.
//
// WHY WebSocket for these and HTTP for editing: faders are sent ~30 times a
// second while dragging and pads must fire with as little delay as possible;
// a request per fader step from a tablet on Wi-Fi is wasteful. Library,
// scene and soundboard editing and uploads stay on HTTP.
//
// ids make retries safe: the same id twice runs once and returns the first
// result. A new press always has a new id, so retriggering a pad works.

type wsCommand struct {
	ID   string          `json:"id"`
	Cmd  string          `json:"cmd"`
	Args json.RawMessage `json:"args"`
}

type wsAck struct {
	Type  string `json:"type"`
	ID    string `json:"id"`
	OK    bool   `json:"ok"`
	Error string `json:"error,omitempty"`
	Rev   int64  `json:"rev"`
}

// ackCache remembers recent results by command id.
type ackCache struct {
	mu    sync.Mutex
	acks  map[string]wsAck
	order []string
}

const ackCacheSize = 512

func (c *ackCache) get(id string) (wsAck, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	a, ok := c.acks[id]
	return a, ok
}

func (c *ackCache) put(a wsAck) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.acks == nil {
		c.acks = map[string]wsAck{}
	}
	if _, ok := c.acks[a.ID]; ok {
		return
	}
	c.acks[a.ID] = a
	c.order = append(c.order, a.ID)
	if len(c.order) > ackCacheSize {
		delete(c.acks, c.order[0])
		c.order = c.order[1:]
	}
}

// commandTimeout bounds one command (a scene start opens ffmpeg and reads
// the database).
const commandTimeout = 15 * time.Second

// HandleWS runs one dj.cmd message and returns the ack for the sender.
func (h *Handler) HandleWS(raw []byte) any {
	var c wsCommand
	if err := json.Unmarshal(raw, &c); err != nil || c.ID == "" || c.Cmd == "" {
		return wsAck{Type: "dj.ack", ID: c.ID, Error: "a command needs an id and a cmd", Rev: h.engine.Rev()}
	}
	if a, ok := h.acks.get(c.ID); ok {
		return a
	}
	ctx, cancel := context.WithTimeout(context.Background(), commandTimeout)
	defer cancel()
	err := h.runCommand(ctx, c)
	a := wsAck{Type: "dj.ack", ID: c.ID, OK: err == nil, Rev: h.engine.Rev()}
	if err != nil {
		a.Error = err.Error()
	}
	h.acks.put(a)
	return a
}

func (h *Handler) runCommand(ctx context.Context, c wsCommand) error {
	var a struct {
		Deck       *int     `json:"deck"`
		TrackID    *int64   `json:"trackId"`
		TrackIDs   []int64  `json:"trackIds"`
		SceneID    *int64   `json:"sceneId"`
		PadID      *int64   `json:"padId"`
		Folder     *string  `json:"folder"`
		PositionMs *int64   `json:"positionMs"`
		Value      *float64 `json:"value"`
		On         *bool    `json:"on"`
		QID        *int64   `json:"qid"`
		To         *int     `json:"to"`
		Next       bool     `json:"next"`
		Seconds    *float64 `json:"seconds"`
	}
	if len(c.Args) > 0 {
		if err := json.Unmarshal(c.Args, &a); err != nil {
			return fmt.Errorf("bad args for %s: %v", c.Cmd, err)
		}
	}
	need := func(ok bool, what string) error {
		if !ok {
			return fmt.Errorf("%s needs %s", c.Cmd, what)
		}
		return nil
	}
	e := h.engine
	switch c.Cmd {
	// Decks
	case "deck.load":
		if err := need(a.Deck != nil && a.TrackID != nil, "deck and trackId"); err != nil {
			return err
		}
		return e.LoadDeck(ctx, *a.Deck, *a.TrackID)
	case "deck.play":
		if err := need(a.Deck != nil, "deck"); err != nil {
			return err
		}
		return e.PlayDeck(*a.Deck)
	case "deck.pause":
		if err := need(a.Deck != nil, "deck"); err != nil {
			return err
		}
		return e.PauseDeck(*a.Deck)
	case "deck.eject":
		if err := need(a.Deck != nil, "deck"); err != nil {
			return err
		}
		return e.EjectDeck(*a.Deck)
	case "deck.seek":
		if err := need(a.Deck != nil && a.PositionMs != nil, "deck and positionMs"); err != nil {
			return err
		}
		return e.SeekDeck(*a.Deck, *a.PositionMs)
	case "deck.gain":
		if err := need(a.Deck != nil && a.Value != nil, "deck and value"); err != nil {
			return err
		}
		return e.SetDeckGain(*a.Deck, *a.Value)

	// Mixer
	case "mix.crossfader":
		if err := need(a.Value != nil, "value"); err != nil {
			return err
		}
		e.SetCrossfader(*a.Value)
	case "mix.master":
		if err := need(a.Value != nil, "value"); err != nil {
			return err
		}
		e.SetVolume(*a.Value)
	case "mix.sfxGain":
		if err := need(a.Value != nil, "value"); err != nil {
			return err
		}
		e.SetSfxGain(*a.Value)
	case "mix.autoDJ":
		if err := need(a.On != nil, "on"); err != nil {
			return err
		}
		e.SetAutoDJ(*a.On)
	case "mix.next":
		return e.Next()
	case "mix.stopAll":
		e.Stop(nil)

	// Music sources
	case "scene.play":
		if err := need(a.SceneID != nil, "sceneId"); err != nil {
			return err
		}
		return e.PlayScene(ctx, *a.SceneID, nil)
	case "folder.play":
		if err := need(a.Folder != nil, "folder"); err != nil {
			return err
		}
		rel, err := CleanRelPath(*a.Folder)
		if err != nil {
			return err
		}
		return e.PlayFolder(ctx, rel, a.TrackID)
	case "track.play":
		if err := need(a.TrackID != nil, "trackId"); err != nil {
			return err
		}
		return e.PlayTrack(ctx, *a.TrackID)
	case "tone.play":
		secs := 0.0
		if a.Seconds != nil {
			secs = *a.Seconds
		}
		return e.PlayTone(secs)

	// Queue
	case "queue.add":
		ids := a.TrackIDs
		if a.TrackID != nil {
			ids = append(ids, *a.TrackID)
		}
		if err := need(len(ids) > 0, "trackIds"); err != nil {
			return err
		}
		return e.QueueAdd(ctx, ids, a.Next)
	case "queue.move":
		if err := need(a.QID != nil && a.To != nil, "qid and to"); err != nil {
			return err
		}
		return e.QueueMove(*a.QID, *a.To)
	case "queue.remove":
		if err := need(a.QID != nil, "qid"); err != nil {
			return err
		}
		return e.QueueRemove(*a.QID)
	case "queue.clear":
		e.QueueClear()
	case "queue.shuffle":
		if err := need(a.On != nil, "on"); err != nil {
			return err
		}
		e.SetShuffle(*a.On)
	case "queue.loop":
		if err := need(a.On != nil, "on"); err != nil {
			return err
		}
		e.SetLoop(*a.On)

	// Soundboard
	case "sfx.trigger":
		if err := need(a.PadID != nil, "padId"); err != nil {
			return err
		}
		return h.sb.Trigger(ctx, *a.PadID, c.ID)
	case "sfx.stop":
		var pad int64
		if a.PadID != nil {
			pad = *a.PadID
		}
		e.StopPad(pad)

	default:
		return errors.New("unknown command " + c.Cmd)
	}
	return nil
}
