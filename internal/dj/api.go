package dj

import (
	"encoding/json"
	"errors"
	"net/http"
	"strconv"
	"strings"

	"github.com/joe/dndscribe-go/internal/db"
)

// Handler serves /api/dj/*.
type Handler struct {
	engine  *Engine
	library *Library
	mux     *http.ServeMux

	// OnLibraryChanged is called after tracks, playlists or scenes change so
	// other open clients (the table monitor, a tablet) can refetch.
	OnLibraryChanged func()
}

func NewHandler(engine *Engine, library *Library) *Handler {
	h := &Handler{engine: engine, library: library, mux: http.NewServeMux()}
	m := h.mux

	m.HandleFunc("GET /api/dj/state", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, h.engine.State())
	})

	// Transport
	m.HandleFunc("POST /api/dj/play", h.play)
	m.HandleFunc("POST /api/dj/stop", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			FadeSeconds *float64 `json:"fadeSeconds"`
		}
		_ = readBody(r, &body)
		h.engine.Stop(body.FadeSeconds)
		writeJSON(w, http.StatusOK, h.engine.State())
	})
	m.HandleFunc("POST /api/dj/pause", func(w http.ResponseWriter, r *http.Request) {
		h.engine.SetPaused(true)
		writeJSON(w, http.StatusOK, h.engine.State())
	})
	m.HandleFunc("POST /api/dj/resume", func(w http.ResponseWriter, r *http.Request) {
		h.engine.SetPaused(false)
		writeJSON(w, http.StatusOK, h.engine.State())
	})
	m.HandleFunc("POST /api/dj/next", func(w http.ResponseWriter, r *http.Request) {
		h.engine.Next()
		writeJSON(w, http.StatusOK, h.engine.State())
	})
	m.HandleFunc("POST /api/dj/volume", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Volume float64 `json:"volume"`
		}
		if err := readBody(r, &body); err != nil {
			writeErr(w, http.StatusBadRequest, "invalid body")
			return
		}
		h.engine.SetVolume(body.Volume)
		writeJSON(w, http.StatusOK, h.engine.State())
	})
	m.HandleFunc("POST /api/dj/fade", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Seconds float64 `json:"seconds"`
		}
		if err := readBody(r, &body); err != nil {
			writeErr(w, http.StatusBadRequest, "invalid body")
			return
		}
		h.engine.SetFadeSeconds(body.Seconds)
		writeJSON(w, http.StatusOK, h.engine.State())
	})
	m.HandleFunc("POST /api/dj/tone", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Seconds float64 `json:"seconds"`
		}
		_ = readBody(r, &body)
		if err := h.engine.PlayTone(body.Seconds); err != nil {
			writePlayErr(w, err)
			return
		}
		writeJSON(w, http.StatusOK, h.engine.State())
	})

	// Library
	m.HandleFunc("GET /api/dj/tracks", func(w http.ResponseWriter, r *http.Request) {
		tracks, err := db.ListDJTracks(r.Context())
		if err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, tracks)
	})
	m.HandleFunc("POST /api/dj/tracks/rescan", func(w http.ResponseWriter, r *http.Request) {
		res, err := h.library.Scan(r.Context())
		if err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		h.changed()
		writeJSON(w, http.StatusOK, res)
	})

	// Playlists
	m.HandleFunc("GET /api/dj/playlists", func(w http.ResponseWriter, r *http.Request) {
		pls, err := db.ListDJPlaylists(r.Context())
		if err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, pls)
	})
	m.HandleFunc("POST /api/dj/playlists", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Name     string  `json:"name"`
			TrackIDs []int64 `json:"trackIds"`
		}
		if err := readBody(r, &body); err != nil || strings.TrimSpace(body.Name) == "" {
			writeErr(w, http.StatusBadRequest, "name is required")
			return
		}
		pl, err := db.CreateDJPlaylist(r.Context(), strings.TrimSpace(body.Name))
		if err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		if len(body.TrackIDs) > 0 {
			if err := db.SetDJPlaylistTracks(r.Context(), pl.ID, body.TrackIDs); err != nil {
				writeErr(w, http.StatusInternalServerError, err.Error())
				return
			}
			pl.TrackIDs = body.TrackIDs
		}
		h.changed()
		writeJSON(w, http.StatusCreated, pl)
	})
	m.HandleFunc("PUT /api/dj/playlists/{id}", func(w http.ResponseWriter, r *http.Request) {
		id, ok := pathID(w, r)
		if !ok {
			return
		}
		var body struct {
			Name     *string  `json:"name"`
			TrackIDs *[]int64 `json:"trackIds"`
		}
		if err := readBody(r, &body); err != nil {
			writeErr(w, http.StatusBadRequest, "invalid body")
			return
		}
		if body.Name != nil {
			name := strings.TrimSpace(*body.Name)
			if name == "" {
				writeErr(w, http.StatusBadRequest, "name is required")
				return
			}
			found, err := db.RenameDJPlaylist(r.Context(), id, name)
			if err != nil {
				writeErr(w, http.StatusInternalServerError, err.Error())
				return
			}
			if !found {
				writeErr(w, http.StatusNotFound, "playlist not found")
				return
			}
		}
		if body.TrackIDs != nil {
			if err := db.SetDJPlaylistTracks(r.Context(), id, *body.TrackIDs); err != nil {
				writeErr(w, http.StatusInternalServerError, err.Error())
				return
			}
		}
		pl, err := db.GetDJPlaylist(r.Context(), id)
		if err != nil || pl == nil {
			writeErr(w, http.StatusNotFound, "playlist not found")
			return
		}
		h.changed()
		writeJSON(w, http.StatusOK, pl)
	})
	m.HandleFunc("DELETE /api/dj/playlists/{id}", func(w http.ResponseWriter, r *http.Request) {
		id, ok := pathID(w, r)
		if !ok {
			return
		}
		if _, err := db.DeleteDJPlaylist(r.Context(), id); err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		h.changed()
		writeJSON(w, http.StatusOK, map[string]bool{"success": true})
	})

	// Scenes
	m.HandleFunc("GET /api/dj/scenes", func(w http.ResponseWriter, r *http.Request) {
		scenes, err := db.ListDJScenes(r.Context())
		if err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, scenes)
	})
	m.HandleFunc("POST /api/dj/scenes", func(w http.ResponseWriter, r *http.Request) {
		s, ok := readScene(w, r)
		if !ok {
			return
		}
		created, err := db.CreateDJScene(r.Context(), s)
		if err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		h.changed()
		writeJSON(w, http.StatusCreated, created)
	})
	m.HandleFunc("POST /api/dj/scenes/order", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			IDs []int64 `json:"ids"`
		}
		if err := readBody(r, &body); err != nil {
			writeErr(w, http.StatusBadRequest, "invalid body")
			return
		}
		if err := db.ReorderDJScenes(r.Context(), body.IDs); err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		h.changed()
		writeJSON(w, http.StatusOK, map[string]bool{"success": true})
	})
	m.HandleFunc("PUT /api/dj/scenes/{id}", func(w http.ResponseWriter, r *http.Request) {
		id, ok := pathID(w, r)
		if !ok {
			return
		}
		s, ok := readScene(w, r)
		if !ok {
			return
		}
		s.ID = id
		found, err := db.UpdateDJScene(r.Context(), s)
		if err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		if !found {
			writeErr(w, http.StatusNotFound, "scene not found")
			return
		}
		updated, _ := db.GetDJScene(r.Context(), id)
		if updated != nil {
			h.engine.SceneUpdated(*updated)
		}
		h.changed()
		writeJSON(w, http.StatusOK, updated)
	})
	m.HandleFunc("DELETE /api/dj/scenes/{id}", func(w http.ResponseWriter, r *http.Request) {
		id, ok := pathID(w, r)
		if !ok {
			return
		}
		if _, err := db.DeleteDJScene(r.Context(), id); err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		h.changed()
		writeJSON(w, http.StatusOK, map[string]bool{"success": true})
	})

	return h
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	h.mux.ServeHTTP(w, r)
}

func (h *Handler) changed() {
	if h.OnLibraryChanged != nil {
		h.OnLibraryChanged()
	}
}

func (h *Handler) play(w http.ResponseWriter, r *http.Request) {
	var body struct {
		SceneID     *int64   `json:"sceneId"`
		PlaylistID  *int64   `json:"playlistId"`
		TrackID     *int64   `json:"trackId"`
		FadeSeconds *float64 `json:"fadeSeconds"`
	}
	if err := readBody(r, &body); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid body")
		return
	}
	var err error
	switch {
	case body.SceneID != nil:
		err = h.engine.PlayScene(r.Context(), *body.SceneID, body.FadeSeconds)
	case body.PlaylistID != nil:
		err = h.engine.PlayPlaylist(r.Context(), *body.PlaylistID, body.TrackID)
	case body.TrackID != nil:
		err = h.engine.PlayTrack(r.Context(), *body.TrackID)
	default:
		writeErr(w, http.StatusBadRequest, "sceneId, playlistId or trackId is required")
		return
	}
	if err != nil {
		writePlayErr(w, err)
		return
	}
	writeJSON(w, http.StatusOK, h.engine.State())
}

func readScene(w http.ResponseWriter, r *http.Request) (db.DJScene, bool) {
	// Defaults for fields the client leaves out; decoding only overwrites
	// what's present. Without this an omitted volume would be a silent scene.
	s := db.DJScene{Volume: 1, Loop: true}
	if err := readBody(r, &s); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid body")
		return s, false
	}
	s.Name = strings.TrimSpace(s.Name)
	if s.Name == "" {
		writeErr(w, http.StatusBadRequest, "name is required")
		return s, false
	}
	s.Volume = clamp01(s.Volume)
	if s.FadeSeconds != nil && (*s.FadeSeconds < 0 || *s.FadeSeconds > 30) {
		writeErr(w, http.StatusBadRequest, "fadeSeconds must be between 0 and 30")
		return s, false
	}
	return s, true
}

func pathID(w http.ResponseWriter, r *http.Request) (int64, bool) {
	id, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil {
		writeErr(w, http.StatusBadRequest, "invalid id")
		return 0, false
	}
	return id, true
}

func writePlayErr(w http.ResponseWriter, err error) {
	if errors.Is(err, ErrNotConnected) {
		writeErr(w, http.StatusConflict, err.Error())
		return
	}
	writeErr(w, http.StatusBadRequest, err.Error())
}

func readBody(r *http.Request, v any) error {
	if r.Body == nil || r.ContentLength == 0 {
		return nil
	}
	return json.NewDecoder(r.Body).Decode(v)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}
