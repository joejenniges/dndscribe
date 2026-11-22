package dj

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/joe/dndscribe-go/internal/db"
)

// Handler serves /api/dj/*.
type Handler struct {
	engine  *Engine
	library *Library
	sb      *Soundboard
	waves   *Waveforms
	mux     *http.ServeMux

	// OnLibraryChanged is called after tracks, folders or scenes change so
	// other open clients (the table monitor, a tablet) can refetch.
	OnLibraryChanged func()

	changeMu    sync.Mutex
	changeTimer *time.Timer

	acks ackCache // recent WebSocket command results, by id
}

// maxUploadBytes caps one uploaded file. Generous: an hour-long FLAC
// ambience track is a few hundred MB.
const maxUploadBytes = 2 << 30

func NewHandler(engine *Engine, library *Library, sb *Soundboard, waves *Waveforms) *Handler {
	h := &Handler{engine: engine, library: library, sb: sb, waves: waves, mux: http.NewServeMux()}
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
	m.HandleFunc("POST /api/dj/seek", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			PositionMs *int64 `json:"positionMs"`
		}
		if err := readBody(r, &body); err != nil || body.PositionMs == nil {
			writeErr(w, http.StatusBadRequest, "positionMs is required")
			return
		}
		if err := h.engine.SeekTo(*body.PositionMs); err != nil {
			status := http.StatusBadRequest
			if errors.Is(err, ErrNotPlaying) {
				status = http.StatusConflict
			}
			writeErr(w, status, err.Error())
			return
		}
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
		if err := h.engine.Next(); err != nil {
			writePlayErr(w, err)
			return
		}
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
		if err := h.setFade(r, body.Seconds); err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, h.engine.State())
	})

	// Settings saved from the Settings page. Kept in the database so they
	// survive a restart; config.yaml only supplies the starting defaults.
	m.HandleFunc("GET /api/dj/settings", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, h.settings())
	})
	m.HandleFunc("PUT /api/dj/settings", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			FadeSeconds *float64 `json:"fadeSeconds"`
		}
		if err := readBody(r, &body); err != nil {
			writeErr(w, http.StatusBadRequest, "invalid body")
			return
		}
		if body.FadeSeconds != nil {
			if *body.FadeSeconds < 0 || *body.FadeSeconds > 30 {
				writeErr(w, http.StatusBadRequest, "fadeSeconds must be between 0 and 30")
				return
			}
			if err := h.setFade(r, *body.FadeSeconds); err != nil {
				writeErr(w, http.StatusInternalServerError, err.Error())
				return
			}
		}
		writeJSON(w, http.StatusOK, h.settings())
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

	m.HandleFunc("GET /api/dj/tracks/{id}/waveform", h.waveform)

	m.HandleFunc("DELETE /api/dj/tracks/{id}", func(w http.ResponseWriter, r *http.Request) {
		id, ok := pathID(w, r)
		if !ok {
			return
		}
		t, err := db.GetDJTrack(r.Context(), id)
		if err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		if t == nil {
			writeErr(w, http.StatusNotFound, "track not found")
			return
		}
		if err := h.library.DeleteFile(r.Context(), t.Path); err != nil {
			writeErr(w, http.StatusConflict, err.Error())
			return
		}
		h.changed()
		writeJSON(w, http.StatusOK, map[string]bool{"success": true})
	})

	// Folders. Each folder under the music dir is a playlist; scenes pick
	// folders. Paths are music-dir-relative with forward slashes.
	m.HandleFunc("GET /api/dj/folders", func(w http.ResponseWriter, r *http.Request) {
		folders, err := h.library.Folders()
		if err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, folders)
	})
	m.HandleFunc("POST /api/dj/folders", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Path string `json:"path"`
		}
		if err := readBody(r, &body); err != nil {
			writeErr(w, http.StatusBadRequest, "invalid body")
			return
		}
		rel, err := CleanRelPath(body.Path)
		if err == nil {
			err = h.library.MakeFolder(rel)
		}
		if err != nil {
			writeErr(w, http.StatusBadRequest, err.Error())
			return
		}
		h.changed()
		writeJSON(w, http.StatusCreated, map[string]string{"path": rel})
	})
	m.HandleFunc("DELETE /api/dj/folders", func(w http.ResponseWriter, r *http.Request) {
		rel, err := CleanRelPath(r.URL.Query().Get("path"))
		if err == nil {
			err = h.library.DeleteFolder(r.Context(), rel)
		}
		if err != nil {
			writeErr(w, http.StatusConflict, err.Error())
			return
		}
		h.changed()
		writeJSON(w, http.StatusOK, map[string]bool{"success": true})
	})

	// Upload one file. The body is the raw file; the target path (folders
	// included) is in ?path=. One request per file keeps uploads streaming
	// (no multipart buffering), lets the browser show per-file progress and
	// retry or skip a single file, and means a dropped connection loses one
	// track rather than the batch.
	m.HandleFunc("PUT /api/dj/files", func(w http.ResponseWriter, r *http.Request) {
		rel, err := CleanRelPath(r.URL.Query().Get("path"))
		if err != nil {
			writeErr(w, http.StatusBadRequest, err.Error())
			return
		}
		if rel == "" || !IsAudioFile(rel) {
			writeErr(w, http.StatusBadRequest, "only audio files can be uploaded (mp3, ogg, opus, flac, wav, m4a, aac, webm, wma)")
			return
		}
		body := http.MaxBytesReader(w, r.Body, maxUploadBytes)
		t, err := h.library.SaveFile(r.Context(), rel, body, r.URL.Query().Get("overwrite") == "1")
		if errors.Is(err, ErrExists) {
			writeErr(w, http.StatusConflict, err.Error())
			return
		}
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			writeErr(w, http.StatusRequestEntityTooLarge, "file is larger than 2 GB")
			return
		}
		if err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		// Debounced: a 200-file folder upload sends one refetch, not 200.
		h.changedSoon()
		writeJSON(w, http.StatusCreated, t)
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
		img, err := db.DeleteDJScene(r.Context(), id)
		if err != nil {
			writeErr(w, http.StatusInternalServerError, err.Error())
			return
		}
		h.sb.RemoveImage(img)
		h.changed()
		writeJSON(w, http.StatusOK, map[string]bool{"success": true})
	})
	m.HandleFunc("PUT /api/dj/scenes/{id}/image", func(w http.ResponseWriter, r *http.Request) {
		id, ok := pathID(w, r)
		if !ok {
			return
		}
		key, err := h.sb.SetSceneImage(r.Context(), id, r.Body)
		if err != nil {
			writeErr(w, http.StatusBadRequest, err.Error())
			return
		}
		h.changed()
		writeJSON(w, http.StatusOK, map[string]string{"image": key})
	})
	m.HandleFunc("DELETE /api/dj/scenes/{id}/image", func(w http.ResponseWriter, r *http.Request) {
		id, ok := pathID(w, r)
		if !ok {
			return
		}
		if err := h.sb.ClearSceneImage(r.Context(), id); err != nil {
			writeErr(w, http.StatusBadRequest, err.Error())
			return
		}
		h.changed()
		writeJSON(w, http.StatusOK, map[string]bool{"success": true})
	})

	h.mountSoundboard()
	return h
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	h.mux.ServeHTTP(w, r)
}

func (h *Handler) changed() {
	if h.OnLibraryChanged != nil {
		h.OnLibraryChanged()
	}
	// New or replaced tracks get their waveforms worked out now rather than
	// when first played. Cheap when there's nothing to do.
	h.waves.Fill(context.Background())
}

type djSettings struct {
	FadeSeconds float64 `json:"fadeSeconds"`
}

func (h *Handler) settings() djSettings {
	return djSettings{FadeSeconds: h.engine.FadeSeconds()}
}

func (h *Handler) setFade(r *http.Request, seconds float64) error {
	h.engine.SetFadeSeconds(seconds)
	return db.SetSetting(r.Context(), db.SettingDJFadeSeconds, h.engine.FadeSeconds())
}

// waveform serves a track's waveform as raw bytes: [peak, rms] per bin,
// X-Bins-Per-Second bins per second. The ETag is the file's size and mtime,
// so the browser keeps its copy until the file changes.
func (h *Handler) waveform(w http.ResponseWriter, r *http.Request) {
	id, ok := pathID(w, r)
	if !ok {
		return
	}
	t, err := db.GetDJTrack(r.Context(), id)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	if t == nil || t.Missing {
		writeErr(w, http.StatusNotFound, "track not found")
		return
	}
	etag := fmt.Sprintf(`"wf1-%d-%d-%d"`, t.ID, t.SizeBytes, t.ModTime.UnixMicro())
	w.Header().Set("ETag", etag)
	w.Header().Set("Cache-Control", "private, no-cache")
	if r.Header.Get("If-None-Match") == etag {
		w.WriteHeader(http.StatusNotModified)
		return
	}
	wf, err := h.waves.Get(r.Context(), *t)
	if err != nil {
		if r.Context().Err() == nil {
			log.Printf("DJ waveform: %s: %v", t.Path, err)
			writeErr(w, http.StatusInternalServerError, "couldn't read the track to draw its waveform")
		}
		return
	}
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("X-Bins-Per-Second", strconv.Itoa(wf.BinsPerSecond))
	w.Header().Set("Content-Length", strconv.Itoa(len(wf.Data)))
	_, _ = w.Write(wf.Data)
}

// changedSoon coalesces bursts of changes into one notification 1s after
// the last one.
func (h *Handler) changedSoon() {
	h.changeMu.Lock()
	defer h.changeMu.Unlock()
	if h.changeTimer != nil {
		h.changeTimer.Stop()
	}
	h.changeTimer = time.AfterFunc(time.Second, h.changed)
}

func (h *Handler) play(w http.ResponseWriter, r *http.Request) {
	var body struct {
		SceneID     *int64   `json:"sceneId"`
		Folder      *string  `json:"folder"`
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
	case body.Folder != nil:
		var rel string
		if rel, err = CleanRelPath(*body.Folder); err == nil {
			err = h.engine.PlayFolder(r.Context(), rel, body.TrackID)
		}
	case body.TrackID != nil:
		err = h.engine.PlayTrack(r.Context(), *body.TrackID)
	default:
		writeErr(w, http.StatusBadRequest, "sceneId, folder or trackId is required")
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
	folders := []string{} // never nil: the column is NOT NULL
	seen := map[string]bool{}
	for _, f := range s.Folders {
		rel, err := CleanRelPath(f)
		if err != nil {
			writeErr(w, http.StatusBadRequest, err.Error())
			return s, false
		}
		if !seen[rel] {
			seen[rel] = true
			folders = append(folders, rel)
		}
	}
	s.Folders = folders
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
