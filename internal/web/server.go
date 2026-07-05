package web

import (
	"fmt"
	"log"
	"mime"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/joe/dndscribe-go/internal/config"
)

func init() {
	// Register MIME types that may not be in the OS registry
	mime.AddExtensionType(".js", "application/javascript")
	mime.AddExtensionType(".css", "text/css")
	mime.AddExtensionType(".html", "text/html")
	mime.AddExtensionType(".json", "application/json")
	mime.AddExtensionType(".svg", "image/svg+xml")
	mime.AddExtensionType(".png", "image/png")
	mime.AddExtensionType(".ico", "image/x-icon")
	mime.AddExtensionType(".woff", "font/woff")
	mime.AddExtensionType(".woff2", "font/woff2")
}

// Start creates and starts the HTTP server. It returns the *http.Server so the
// caller can shut it down gracefully. The hub and worker are created externally
// so the caller can wire event callbacks before starting the server.
func Start(cfg *config.Config, ctrl BotController, hub *Hub, worker TranscriptionWorker) *http.Server {
	go hub.Run()

	api := &APIHandler{
		ctrl:         ctrl,
		hub:          hub,
		tw:           worker,
		recordingDir: cfg.Recordings.Dir,
	}

	mux := http.NewServeMux()

	// Audio file serving with Range support
	mux.HandleFunc("/recordings/raw/", func(w http.ResponseWriter, r *http.Request) {
		serveAudioFile(w, r, cfg.Recordings.Dir)
	})

	// API routes
	mux.HandleFunc("/api/", func(w http.ResponseWriter, r *http.Request) {
		setCORS(w)
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		api.ServeHTTP(w, r)
	})

	// Static files with SPA fallback + WebSocket upgrade on root
	staticDir := "web/build"
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		// Check for WebSocket upgrade request on any path
		if r.Header.Get("Upgrade") == "websocket" {
			hub.HandleWS(w, r)
			return
		}
		setCORS(w)
		serveStatic(w, r, staticDir)
	})

	addr := fmt.Sprintf(":%d", cfg.Web.Port)
	srv := &http.Server{
		Addr:    addr,
		Handler: mux,
	}

	go func() {
		log.Printf("Web server listening on %s", addr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Printf("HTTP server error: %v", err)
		}
	}()

	return srv
}

// GetHub returns the WebSocket hub from a running server. This is a convenience
// for callers that need to broadcast messages. Typically you'd hold onto the hub
// reference from Start, but this provides an alternative path.
// NOTE: For now, callers should keep a reference to the hub via the APIHandler.
// This function is a placeholder if we need a global accessor later.

func setCORS(w http.ResponseWriter) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
}

func serveStatic(w http.ResponseWriter, r *http.Request, staticDir string) {
	// Clean the path
	urlPath := r.URL.Path
	if urlPath == "/" {
		urlPath = "/200.html"
	}

	filePath := filepath.Join(staticDir, filepath.FromSlash(urlPath))

	// Check if file exists
	info, err := os.Stat(filePath)
	if err != nil || info.IsDir() {
		// SPA fallback: serve 200.html for any path that doesn't match a file
		filePath = filepath.Join(staticDir, "200.html")
		info, err = os.Stat(filePath)
		if err != nil {
			http.NotFound(w, r)
			return
		}
	}

	// Set content type from extension
	ext := filepath.Ext(filePath)
	ct := mime.TypeByExtension(ext)
	if ct != "" {
		w.Header().Set("Content-Type", ct)
	}

	http.ServeFile(w, r, filePath)
	_ = info // used for existence check
}

func serveAudioFile(w http.ResponseWriter, r *http.Request, recordingDir string) {
	setCORS(w)
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusNoContent)
		return
	}

	// Extract filename from /recordings/raw/{filename}
	prefix := "/recordings/raw/"
	if !strings.HasPrefix(r.URL.Path, prefix) {
		http.NotFound(w, r)
		return
	}
	filename := r.URL.Path[len(prefix):]
	if filename == "" || strings.Contains(filename, "..") || strings.Contains(filename, "/") {
		http.NotFound(w, r)
		return
	}

	filePath := filepath.Join(recordingDir, "raw", filename)
	f, err := os.Open(filePath)
	if err != nil {
		http.NotFound(w, r)
		return
	}
	defer f.Close()

	stat, err := f.Stat()
	if err != nil {
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}

	// Determine content type
	ct := "application/octet-stream"
	ext := filepath.Ext(filename)
	switch ext {
	case ".ogg":
		ct = "audio/ogg"
	case ".opus":
		ct = "audio/opus"
	case ".wav":
		ct = "audio/wav"
	case ".mp3":
		ct = "audio/mpeg"
	case ".webm":
		ct = "audio/webm"
	}
	w.Header().Set("Content-Type", ct)

	// http.ServeContent handles Range requests (HTTP 206) automatically
	http.ServeContent(w, r, filename, stat.ModTime(), f)
}
