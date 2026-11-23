package dj

import (
	"context"
	"errors"
	"net/http"
	"strconv"
	"strings"

	"github.com/joe/dndscribe-go/internal/db"
)

// Cue definitions over HTTP (v4). Live actions (jump to a hotcue, arm or
// disarm a goto) are WebSocket commands (command.go); creating, moving,
// renaming and deleting cues is here. After every change the engine gets
// the new definitions (so a deck playing the track uses them on the next
// frame) and every client is told which track changed (dj_cues), so pages
// with that track open refetch. Decks carry their loaded track's cues in
// the normal state snapshot as well.
//
//	GET    /api/dj/tracks/{id}/cues
//	PUT    /api/dj/tracks/{id}/hotcues/{slot}   {positionMs, label, color}
//	DELETE /api/dj/tracks/{id}/hotcues/{slot}   (disables gotos aimed at it)
//	POST   /api/dj/tracks/{id}/gotos            {positionMs, targetSlot, enabled, label}
//	PUT    /api/dj/tracks/{id}/gotos/{gid}      same body
//	DELETE /api/dj/tracks/{id}/gotos/{gid}

func (h *Handler) mountCues() {
	m := h.mux
	m.HandleFunc("GET /api/dj/tracks/{id}/cues", func(w http.ResponseWriter, r *http.Request) {
		t, ok := h.cueTrack(w, r)
		if !ok {
			return
		}
		c, err := db.GetDJTrackCues(r.Context(), t.ID)
		if err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, c)
	})

	m.HandleFunc("PUT /api/dj/tracks/{id}/hotcues/{slot}", func(w http.ResponseWriter, r *http.Request) {
		t, ok := h.cueTrack(w, r)
		if !ok {
			return
		}
		slot, ok := cueSlot(w, r)
		if !ok {
			return
		}
		var body struct {
			PositionMs *int64 `json:"positionMs"`
			Label      string `json:"label"`
			Color      string `json:"color"`
		}
		if err := readBody(r, &body); err != nil || body.PositionMs == nil {
			writeErr(w, http.StatusBadRequest, "positionMs is required")
			return
		}
		if *body.PositionMs < 0 || (t.DurationMs > 0 && *body.PositionMs > t.DurationMs) {
			writeErr(w, http.StatusBadRequest, "the hotcue is outside the track")
			return
		}
		hc := db.DJHotcue{Slot: slot, PositionMs: *body.PositionMs, Label: clip(body.Label, 60), Color: clip(body.Color, 20)}
		if err := db.SetDJHotcue(r.Context(), t.ID, hc); err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		h.cuesChanged(r.Context(), w, t.ID)
	})

	m.HandleFunc("DELETE /api/dj/tracks/{id}/hotcues/{slot}", func(w http.ResponseWriter, r *http.Request) {
		t, ok := h.cueTrack(w, r)
		if !ok {
			return
		}
		slot, ok := cueSlot(w, r)
		if !ok {
			return
		}
		if _, err := db.DeleteDJHotcue(r.Context(), t.ID, slot); err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		h.cuesChanged(r.Context(), w, t.ID)
	})

	saveGoto := func(w http.ResponseWriter, r *http.Request, id int64) {
		t, ok := h.cueTrack(w, r)
		if !ok {
			return
		}
		var body struct {
			PositionMs *int64 `json:"positionMs"`
			TargetSlot *int   `json:"targetSlot"`
			Enabled    *bool  `json:"enabled"`
			Label      string `json:"label"`
		}
		if err := readBody(r, &body); err != nil || body.PositionMs == nil || body.TargetSlot == nil {
			writeErr(w, http.StatusBadRequest, "positionMs and targetSlot are required")
			return
		}
		g := db.DJGoto{ID: id, PositionMs: *body.PositionMs, TargetSlot: *body.TargetSlot, Enabled: true, Label: clip(body.Label, 60)}
		if body.Enabled != nil {
			g.Enabled = *body.Enabled
		}
		cur, err := db.GetDJTrackCues(r.Context(), t.ID)
		if err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		if err := ValidateGoto(cur, g, t.DurationMs); err != nil {
			writeErr(w, http.StatusBadRequest, err.Error())
			return
		}
		if id == 0 {
			_, err = db.CreateDJGoto(r.Context(), t.ID, g)
		} else {
			var found bool
			found, err = db.UpdateDJGoto(r.Context(), t.ID, g)
			if err == nil && !found {
				writeErr(w, http.StatusNotFound, "goto not found")
				return
			}
		}
		if err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		h.cuesChanged(r.Context(), w, t.ID)
	}
	m.HandleFunc("POST /api/dj/tracks/{id}/gotos", func(w http.ResponseWriter, r *http.Request) {
		saveGoto(w, r, 0)
	})
	m.HandleFunc("PUT /api/dj/tracks/{id}/gotos/{gid}", func(w http.ResponseWriter, r *http.Request) {
		gid, err := strconv.ParseInt(r.PathValue("gid"), 10, 64)
		if err != nil || gid <= 0 {
			writeErr(w, http.StatusBadRequest, "invalid goto id")
			return
		}
		saveGoto(w, r, gid)
	})
	m.HandleFunc("DELETE /api/dj/tracks/{id}/gotos/{gid}", func(w http.ResponseWriter, r *http.Request) {
		t, ok := h.cueTrack(w, r)
		if !ok {
			return
		}
		gid, err := strconv.ParseInt(r.PathValue("gid"), 10, 64)
		if err != nil {
			writeErr(w, http.StatusBadRequest, "invalid goto id")
			return
		}
		if _, err := db.DeleteDJGoto(r.Context(), t.ID, gid); err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		h.cuesChanged(r.Context(), w, t.ID)
	})
}

func (h *Handler) cueTrack(w http.ResponseWriter, r *http.Request) (*db.DJTrack, bool) {
	id, ok := pathID(w, r)
	if !ok {
		return nil, false
	}
	t, err := db.GetDJTrack(r.Context(), id)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return nil, false
	}
	if t == nil {
		writeErr(w, http.StatusNotFound, "track not found")
		return nil, false
	}
	return t, true
}

func cueSlot(w http.ResponseWriter, r *http.Request) (int, bool) {
	slot, err := strconv.Atoi(r.PathValue("slot"))
	if err != nil || slot < 1 || slot > db.HotcueSlots {
		writeErr(w, http.StatusBadRequest, "slot must be 1-8")
		return 0, false
	}
	return slot, true
}

// cuesChanged reloads a track's cues into the engine, tells clients, and
// replies with the new definitions.
func (h *Handler) cuesChanged(ctx context.Context, w http.ResponseWriter, trackID int64) {
	c, err := h.reloadCues(ctx, trackID)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, c)
}

func (h *Handler) reloadCues(ctx context.Context, trackID int64) (*db.DJTrackCues, error) {
	c, err := db.GetDJTrackCues(ctx, trackID)
	if err != nil {
		return nil, err
	}
	h.engine.SetTrackCues(c)
	if h.OnCuesChanged != nil {
		h.OnCuesChanged(trackID)
	}
	return c, nil
}

// toggleGoto arms or disarms a goto (WS command goto.toggle). Persisted:
// the armed state is part of the definition, shared by every device and
// kept across restarts.
func (h *Handler) toggleGoto(ctx context.Context, trackID, gotoID int64, on bool) error {
	if on {
		t, err := db.GetDJTrack(ctx, trackID)
		if err != nil {
			return err
		}
		if t == nil {
			return errors.New("track not found")
		}
		c, err := db.GetDJTrackCues(ctx, trackID)
		if err != nil {
			return err
		}
		for _, g := range c.Gotos {
			if g.ID == gotoID {
				if p := gotoProblem(c, g, t.DurationMs); p != "" {
					return errors.New("can't arm this goto: " + p)
				}
			}
		}
	}
	found, err := db.SetDJGotoEnabled(ctx, trackID, gotoID, on)
	if err != nil {
		return err
	}
	if !found {
		return errors.New("goto not found")
	}
	_, err = h.reloadCues(ctx, trackID)
	return err
}

func clip(s string, n int) string {
	s = strings.TrimSpace(s)
	if r := []rune(s); len(r) > n {
		return string(r[:n])
	}
	return s
}
