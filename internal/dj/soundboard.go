package dj

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"image"
	_ "image/gif" // register decoders for image.DecodeConfig
	_ "image/jpeg"
	_ "image/png"
	"io"
	"log"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/joe/dndscribe-go/internal/db"
)

// Soundboard manages sound-effect files and pad images on disk. Layout:
//
//	<dir>/sounds/<key><ext>   one file per pad
//	<dir>/images/<key><ext>   optional pad and scene images
//
// Keys are random and server-generated; the client's file name is kept only
// as a label. That keeps client input out of filesystem paths entirely,
// unlike the music library, where the folder layout is the point.
type Soundboard struct {
	dir    string
	ffmpeg string
	cache  *SfxCache
	engine *Engine
}

const (
	maxSfxUploadBytes   = 50 << 20
	maxImageUploadBytes = 5 << 20
	maxImagePixels      = 40_000_000
	sfxCacheBytes       = 512 << 20
)

func NewSoundboard(dir, ffmpegPath string, engine *Engine) *Soundboard {
	return &Soundboard{dir: dir, ffmpeg: ffmpegPath, cache: NewSfxCache(ffmpegPath, sfxCacheBytes), engine: engine}
}

func (s *Soundboard) soundPath(key string) string { return filepath.Join(s.dir, "sounds", key) }
func (s *Soundboard) imagePath(key string) string { return filepath.Join(s.dir, "images", key) }

func newKey(ext string) string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b) + ext
}

// Warm decodes every pad into the cache so the first tap of each is instant.
// Run in the background at startup.
func (s *Soundboard) Warm(ctx context.Context) {
	pads, err := db.ListSfxPads(ctx)
	if err != nil {
		log.Printf("Soundboard: warm cache: %v", err)
		return
	}
	for _, p := range pads {
		if _, err := s.cache.Load(ctx, s.soundPath(p.FileKey)); err != nil {
			log.Printf("Soundboard: %q (pad %d): %v", p.Label, p.ID, err)
		}
	}
	if len(pads) > 0 {
		log.Printf("Soundboard: %d effects loaded", len(pads))
	}
}

// AddSound stores an uploaded effect and creates its pad at the end of tab.
// The file is decoded before the pad is created: a file ffmpeg can't read
// is rejected now, not discovered mid-game when the pad is tapped.
func (s *Soundboard) AddSound(ctx context.Context, tabID int64, name string, body io.Reader) (*db.SfxPad, error) {
	name = path.Base(strings.ReplaceAll(name, "\\", "/"))
	ext := strings.ToLower(path.Ext(name))
	if !audioExts[ext] {
		return nil, fmt.Errorf("%q isn't an audio file this server can play", name)
	}
	if err := os.MkdirAll(filepath.Join(s.dir, "sounds"), 0o755); err != nil {
		return nil, err
	}
	key := newKey(ext)
	full := s.soundPath(key)
	f, err := os.Create(full)
	if err != nil {
		return nil, err
	}
	_, err = io.Copy(f, body)
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		os.Remove(full)
		return nil, fmt.Errorf("upload interrupted: %w", err)
	}

	probe := Probe(s.ffmpeg, full)
	if probe.DurationMs > MaxSfxDuration.Milliseconds() {
		os.Remove(full)
		return nil, fmt.Errorf("%q is longer than %d minutes; put long tracks in the music library", name, int(MaxSfxDuration.Minutes()))
	}
	pcm, err := s.cache.Load(ctx, full)
	if err != nil {
		os.Remove(full)
		// ffmpeg's message is for the log: it's noisy and names server paths.
		log.Printf("Soundboard: rejected upload %q: %v", name, err)
		return nil, fmt.Errorf("%q isn't a playable audio file", name)
	}
	label := strings.TrimSuffix(name, path.Ext(name))
	label = strings.TrimSpace(strings.ReplaceAll(strings.ReplaceAll(label, "_", " "), "-", " "))
	if label == "" {
		label = "Sound"
	}
	pad, err := db.CreateSfxPad(ctx, db.SfxPad{
		TabID: tabID, Label: label, FileKey: key, OriginalName: name,
		DurationMs: int64(len(pcm)/Channels) * 1000 / SampleRate, Gain: 1,
	})
	if err != nil {
		s.cache.Forget(full)
		os.Remove(full)
		return nil, err
	}
	return pad, nil
}

// DeletePad stops the pad, removes it and its files.
func (s *Soundboard) DeletePad(ctx context.Context, id int64) error {
	p, err := db.DeleteSfxPad(ctx, id)
	if err != nil || p == nil {
		return err
	}
	s.removeFiles(*p)
	return nil
}

// DeleteTab removes a tab with its pads and their files.
func (s *Soundboard) DeleteTab(ctx context.Context, id int64) error {
	pads, err := db.DeleteSfxTab(ctx, id)
	if err != nil {
		return err
	}
	for _, p := range pads {
		s.removeFiles(p)
	}
	return nil
}

func (s *Soundboard) removeFiles(p db.SfxPad) {
	s.engine.StopPad(p.ID)
	full := s.soundPath(p.FileKey)
	s.cache.Forget(full)
	// The engine holds the decoded PCM, not the file, so this works even
	// while the pad is still finishing its de-click ramp.
	if err := os.Remove(full); err != nil && !errors.Is(err, os.ErrNotExist) {
		log.Printf("Soundboard: remove %s: %v", full, err)
	}
	if p.ImageKey != nil {
		os.Remove(s.imagePath(*p.ImageKey))
	}
}

// Trigger plays a pad, restarting it if it's already playing.
func (s *Soundboard) Trigger(ctx context.Context, id int64, requestID string) error {
	p, err := db.GetSfxPad(ctx, id)
	if err != nil {
		return err
	}
	if p == nil {
		return errors.New("pad not found")
	}
	if !s.engine.isConnected() {
		return ErrNotConnected
	}
	pcm, err := s.cache.Load(ctx, s.soundPath(p.FileKey))
	if err != nil {
		log.Printf("Soundboard: pad %d: %v", p.ID, err)
		return fmt.Errorf("can't play %q: its sound file is missing or unreadable", p.Label)
	}
	return s.engine.TriggerPad(p.ID, p.Label, pcm, p.Gain, requestID)
}

var imageExts = map[string]string{
	"image/png": ".png", "image/jpeg": ".jpg", "image/gif": ".gif", "image/webp": ".webp",
}

// SetImage stores a pad image, replacing any previous one.
func (s *Soundboard) SetImage(ctx context.Context, padID int64, body io.Reader) (string, error) {
	return s.setImage(ctx, body, func(key *string) (*string, error) { return db.SetSfxPadImage(ctx, padID, key) })
}

// SetSceneImage stores a scene's background image, replacing any previous one.
func (s *Soundboard) SetSceneImage(ctx context.Context, sceneID int64, body io.Reader) (string, error) {
	return s.setImage(ctx, body, func(key *string) (*string, error) { return db.SetDJSceneImage(ctx, sceneID, key) })
}

func (s *Soundboard) ClearSceneImage(ctx context.Context, sceneID int64) error {
	old, err := db.SetDJSceneImage(ctx, sceneID, nil)
	if err != nil {
		return err
	}
	s.RemoveImage(old)
	return nil
}

// RemoveImage deletes a stored image file; nil is a no-op.
func (s *Soundboard) RemoveImage(key *string) {
	if key != nil && reImageKey.MatchString(*key) {
		os.Remove(s.imagePath(*key))
	}
}

// setImage validates and stores an image, then calls save to record the new
// key against its owner (which returns the key it replaced).
//
// Validation is by content, not the file name: the bytes must sniff as
// PNG/JPEG/GIF/WebP, and PNG/JPEG/GIF must decode a header within the pixel
// limit. SVG and anything else is refused, so an uploaded file can never be
// served as a script-capable document. (WebP has no decoder in the standard
// library; it is accepted on its signature and only ever served as
// image/webp.)
func (s *Soundboard) setImage(ctx context.Context, body io.Reader, save func(*string) (*string, error)) (string, error) {
	data, err := io.ReadAll(io.LimitReader(body, maxImageUploadBytes+1))
	if err != nil {
		return "", err
	}
	if len(data) > maxImageUploadBytes {
		return "", fmt.Errorf("image is larger than %d MB", maxImageUploadBytes>>20)
	}
	ctype := http.DetectContentType(data)
	ext, ok := imageExts[ctype]
	if !ok {
		return "", errors.New("use a PNG, JPEG, GIF or WebP image")
	}
	if ctype != "image/webp" {
		cfg, _, err := image.DecodeConfig(bytes.NewReader(data))
		if err != nil {
			return "", fmt.Errorf("can't read the image: %v", err)
		}
		if cfg.Width*cfg.Height > maxImagePixels {
			return "", errors.New("image has too many pixels")
		}
	}
	if err := os.MkdirAll(filepath.Join(s.dir, "images"), 0o755); err != nil {
		return "", err
	}
	key := newKey(ext)
	if err := os.WriteFile(s.imagePath(key), data, 0o644); err != nil {
		return "", err
	}
	old, err := save(&key)
	if err != nil {
		os.Remove(s.imagePath(key))
		return "", err
	}
	s.RemoveImage(old)
	return key, nil
}

func (s *Soundboard) ClearImage(ctx context.Context, padID int64) error {
	old, err := db.SetSfxPadImage(ctx, padID, nil)
	if err != nil {
		return err
	}
	if old != nil {
		os.Remove(s.imagePath(*old))
	}
	return nil
}

var reImageKey = regexp.MustCompile(`^[0-9a-f]{32}\.(png|jpg|gif|webp)$`)

// ServeImage serves a stored pad or scene image by key.
func (s *Soundboard) ServeImage(w http.ResponseWriter, r *http.Request, key string) {
	if !reImageKey.MatchString(key) {
		http.NotFound(w, r)
		return
	}
	f, err := os.Open(s.imagePath(key))
	if err != nil {
		http.NotFound(w, r)
		return
	}
	defer f.Close()
	ctype := map[string]string{".png": "image/png", ".jpg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp"}[path.Ext(key)]
	w.Header().Set("Content-Type", ctype)
	w.Header().Set("X-Content-Type-Options", "nosniff")
	// Keys are never reused (a replaced image gets a new key), so the
	// browser can cache forever.
	w.Header().Set("Cache-Control", "public, max-age=31536000, immutable")
	http.ServeContent(w, r, "", time.Time{}, f)
}
