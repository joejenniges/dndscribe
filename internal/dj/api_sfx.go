package dj

import (
	"errors"
	"net/http"
	"strconv"
	"strings"

	"github.com/joe/dndscribe-go/internal/db"
)

// Soundboard routes, under /api/dj/sfx. Layout edits notify other clients
// through the same dj_library event as the music library; playback state
// (which pads are playing) rides on dj_state.
func (h *Handler) mountSoundboard() {
	m, sb := h.mux, h.sb

	m.HandleFunc("GET /api/dj/sfx", func(w http.ResponseWriter, r *http.Request) {
		tabs, err := db.ListSfxTabs(r.Context())
		if err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		pads, err := db.ListSfxPads(r.Context())
		if err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"tabs": tabs, "pads": pads})
	})

	// Playback
	m.HandleFunc("POST /api/dj/sfx/pads/{id}/trigger", func(w http.ResponseWriter, r *http.Request) {
		id, ok := pathID(w, r)
		if !ok {
			return
		}
		var body struct {
			RequestID string `json:"requestId"`
		}
		_ = readBody(r, &body)
		if err := sb.Trigger(r.Context(), id, body.RequestID); err != nil {
			writePlayErr(w, err)
			return
		}
		writeJSON(w, http.StatusOK, h.engine.State())
	})
	m.HandleFunc("POST /api/dj/sfx/pads/{id}/stop", func(w http.ResponseWriter, r *http.Request) {
		id, ok := pathID(w, r)
		if !ok {
			return
		}
		h.engine.StopPad(id)
		writeJSON(w, http.StatusOK, h.engine.State())
	})
	m.HandleFunc("POST /api/dj/sfx/stop", func(w http.ResponseWriter, r *http.Request) {
		h.engine.StopPad(0)
		writeJSON(w, http.StatusOK, h.engine.State())
	})
	m.HandleFunc("POST /api/dj/sfx/gain", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Gain float64 `json:"gain"`
		}
		if err := readBody(r, &body); err != nil {
			writeErr(w, http.StatusBadRequest, "invalid body")
			return
		}
		h.engine.SetSfxGain(body.Gain)
		writeJSON(w, http.StatusOK, h.engine.State())
	})

	// Tabs
	m.HandleFunc("POST /api/dj/sfx/tabs", func(w http.ResponseWriter, r *http.Request) {
		title, ok := readTitle(w, r)
		if !ok {
			return
		}
		t, err := db.CreateSfxTab(r.Context(), title)
		if err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		h.changed()
		writeJSON(w, http.StatusCreated, t)
	})
	m.HandleFunc("PUT /api/dj/sfx/tabs/{id}", func(w http.ResponseWriter, r *http.Request) {
		id, ok := pathID(w, r)
		if !ok {
			return
		}
		title, ok := readTitle(w, r)
		if !ok {
			return
		}
		found, err := db.RenameSfxTab(r.Context(), id, title)
		if err != nil || !found {
			writeErr(w, http.StatusNotFound, "tab not found")
			return
		}
		h.changed()
		writeJSON(w, http.StatusOK, map[string]bool{"success": true})
	})
	m.HandleFunc("DELETE /api/dj/sfx/tabs/{id}", func(w http.ResponseWriter, r *http.Request) {
		id, ok := pathID(w, r)
		if !ok {
			return
		}
		if err := sb.DeleteTab(r.Context(), id); err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		h.changed()
		writeJSON(w, http.StatusOK, map[string]bool{"success": true})
	})
	m.HandleFunc("POST /api/dj/sfx/tabs/order", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			IDs []int64 `json:"ids"`
		}
		if err := readBody(r, &body); err != nil {
			writeErr(w, http.StatusBadRequest, "invalid body")
			return
		}
		if err := db.ReorderSfxTabs(r.Context(), body.IDs); err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		h.changed()
		writeJSON(w, http.StatusOK, map[string]bool{"success": true})
	})
	// Arrange sets a tab's pads and their order; listing a pad from another
	// tab moves it here.
	m.HandleFunc("POST /api/dj/sfx/tabs/{id}/arrange", func(w http.ResponseWriter, r *http.Request) {
		id, ok := pathID(w, r)
		if !ok {
			return
		}
		var body struct {
			IDs []int64 `json:"ids"`
		}
		if err := readBody(r, &body); err != nil {
			writeErr(w, http.StatusBadRequest, "invalid body")
			return
		}
		if err := db.ArrangeSfxTab(r.Context(), id, body.IDs); err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		h.changed()
		writeJSON(w, http.StatusOK, map[string]bool{"success": true})
	})

	// Pads. Upload is one raw-body request per file (same reasons as music
	// uploads); the file name in ?name= becomes the default label.
	m.HandleFunc("PUT /api/dj/sfx/upload", func(w http.ResponseWriter, r *http.Request) {
		tabID, err := strconv.ParseInt(r.URL.Query().Get("tab"), 10, 64)
		if err != nil {
			writeErr(w, http.StatusBadRequest, "tab is required")
			return
		}
		body := http.MaxBytesReader(w, r.Body, maxSfxUploadBytes)
		name := r.URL.Query().Get("name")
		var pad *db.SfxPad
		// ?replace=<pad id>: overwrite that pad's sound, keeping its
		// label, colour, image and place (re-uploading a folder).
		if rep := r.URL.Query().Get("replace"); rep != "" {
			padID, perr := strconv.ParseInt(rep, 10, 64)
			if perr != nil {
				writeErr(w, http.StatusBadRequest, "invalid replace id")
				return
			}
			pad, err = sb.ReplaceSound(r.Context(), padID, name, body)
		} else {
			pad, err = sb.AddSound(r.Context(), tabID, name, body)
		}
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			writeErr(w, http.StatusRequestEntityTooLarge, "sound effects are limited to 50 MB")
			return
		}
		if err != nil {
			writeErr(w, http.StatusBadRequest, err.Error())
			return
		}
		h.changedSoon()
		writeJSON(w, http.StatusCreated, pad)
	})
	m.HandleFunc("PUT /api/dj/sfx/pads/{id}", func(w http.ResponseWriter, r *http.Request) {
		id, ok := pathID(w, r)
		if !ok {
			return
		}
		cur, err := db.GetSfxPad(r.Context(), id)
		if err != nil || cur == nil {
			writeErr(w, http.StatusNotFound, "pad not found")
			return
		}
		// Decode over the current values: fields the client leaves out keep
		// theirs.
		p := *cur
		if err := readBody(r, &p); err != nil {
			writeErr(w, http.StatusBadRequest, "invalid body")
			return
		}
		p.ID = id
		p.Label = strings.TrimSpace(p.Label)
		if p.Label == "" {
			writeErr(w, http.StatusBadRequest, "label is required")
			return
		}
		p.Gain = clamp01(p.Gain)
		if _, err := db.UpdateSfxPad(r.Context(), p); err != nil {
			writeErr(w, http.StatusBadRequest, err.Error())
			return
		}
		updated, _ := db.GetSfxPad(r.Context(), id)
		h.changed()
		writeJSON(w, http.StatusOK, updated)
	})
	m.HandleFunc("DELETE /api/dj/sfx/pads/{id}", func(w http.ResponseWriter, r *http.Request) {
		id, ok := pathID(w, r)
		if !ok {
			return
		}
		if err := sb.DeletePad(r.Context(), id); err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		h.changed()
		writeJSON(w, http.StatusOK, map[string]bool{"success": true})
	})
	m.HandleFunc("PUT /api/dj/sfx/pads/{id}/image", func(w http.ResponseWriter, r *http.Request) {
		id, ok := pathID(w, r)
		if !ok {
			return
		}
		key, err := sb.SetImage(r.Context(), id, r.Body)
		if err != nil {
			writeErr(w, http.StatusBadRequest, err.Error())
			return
		}
		h.changed()
		writeJSON(w, http.StatusOK, map[string]string{"image": key})
	})
	m.HandleFunc("DELETE /api/dj/sfx/pads/{id}/image", func(w http.ResponseWriter, r *http.Request) {
		id, ok := pathID(w, r)
		if !ok {
			return
		}
		if err := sb.ClearImage(r.Context(), id); err != nil {
			writeErr(w, http.StatusBadRequest, err.Error())
			return
		}
		h.changed()
		writeJSON(w, http.StatusOK, map[string]bool{"success": true})
	})
	m.HandleFunc("GET /api/dj/sfx/images/{key}", func(w http.ResponseWriter, r *http.Request) {
		sb.ServeImage(w, r, r.PathValue("key"))
	})
}

func readTitle(w http.ResponseWriter, r *http.Request) (string, bool) {
	var body struct {
		Title string `json:"title"`
	}
	if err := readBody(r, &body); err != nil || strings.TrimSpace(body.Title) == "" {
		writeErr(w, http.StatusBadRequest, "title is required")
		return "", false
	}
	return strings.TrimSpace(body.Title), true
}
