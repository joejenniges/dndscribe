package transcribe

import (
	"context"
	"fmt"
	"log"
	"math"
	"os"
	"sync"

	"github.com/joe/dndscribe-go/internal/config"
	"github.com/joe/dndscribe-go/internal/db"
	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

// Diarizer identifies which person is speaking within a single Discord user's
// audio (shared-microphone case). Each utterance segment is embedded with a
// speaker-verification model; the embedding is cosine-matched against that
// user's known speaker profiles. Unmatched voices auto-enroll as "S1", "S2"...
// up to maxSpeakers. Profiles persist in the speakers table so labels are
// stable across sessions; display names are assigned via the API (phase-2 UI).
//
// Engine-independent: whisper hands it flushed utterance chunks, sherpa hands
// it streaming segments. Both are 16kHz float32 mono.
type Diarizer struct {
	ex          *sherpa.SpeakerEmbeddingExtractor
	threshold   float32
	maxSpeakers int
	minSamples  int // segments shorter than this are skipped (unreliable)

	// mu serializes extractor use AND profile map access. Identify is called
	// from per-user engine goroutines and the whisper worker; embedding takes
	// ~tens of ms so one lock for both keeps the invariants simple.
	mu       sync.Mutex
	profiles map[string][]*speakerProfile // key: campaignID:discordUserID
	loaded   map[int64]bool               // campaigns whose profiles are in memory
}

type speakerProfile struct {
	id          int64
	label       string // auto-assigned "S1", "S2"...
	displayName string // human-assigned (phase-2 UI); "" when unset
	centroid    []float32 // L2-normalized running mean
	sampleCount int
}

// displayLabel is what callers show for this speaker: the assigned name when
// set, otherwise the auto label.
func (p *speakerProfile) displayLabel() string {
	if p.displayName != "" {
		return p.displayName
	}
	return p.label
}

// NewDiarizer loads the speaker-embedding model. Returns (nil, nil) when
// diarization is disabled so callers can treat a nil *Diarizer as "off".
func NewDiarizer(cfg *config.DiarizationConfig) (*Diarizer, error) {
	if !cfg.Enabled {
		return nil, nil
	}
	if _, err := os.Stat(cfg.Model); err != nil {
		return nil, fmt.Errorf("diarization model %q not found (run download-model.ps1 -Speaker): %w", cfg.Model, err)
	}

	ex := sherpa.NewSpeakerEmbeddingExtractor(&sherpa.SpeakerEmbeddingExtractorConfig{
		Model:      cfg.Model,
		NumThreads: cfg.NumThreads,
		Provider:   cfg.Provider,
		Debug:      0,
	})
	if ex == nil {
		return nil, fmt.Errorf("sherpa NewSpeakerEmbeddingExtractor returned nil (model=%s): check file and native libs", cfg.Model)
	}

	log.Printf("Loaded diarization model %s (dim=%d, threshold=%.2f, max_speakers=%d)",
		cfg.Model, ex.Dim(), cfg.Threshold, cfg.MaxSpeakers)

	return &Diarizer{
		ex:          ex,
		threshold:   cfg.Threshold,
		maxSpeakers: cfg.MaxSpeakers,
		minSamples:  int(cfg.MinSeconds * recognizerSampleRate),
		profiles:    make(map[string][]*speakerProfile),
		loaded:      make(map[int64]bool),
	}, nil
}

// Identify returns the (speakerID, label) for the voice in samples16k, or
// (0, "") when identification is skipped (too short, or diarizer nil).
// New voices are enrolled and persisted; matched profiles have their centroid
// updated with the new embedding (running mean, persisted async).
func (d *Diarizer) Identify(ctx context.Context, campaignID int64, discordUserID string, samples16k []float32) (int64, string) {
	if d == nil || len(samples16k) < d.minSamples {
		return 0, ""
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	if err := d.ensureLoadedLocked(ctx, campaignID); err != nil {
		log.Printf("Diarization: load profiles for campaign %d: %v", campaignID, err)
		return 0, ""
	}

	emb := d.embedLocked(samples16k)
	if emb == nil {
		return 0, ""
	}

	key := profileKey(campaignID, discordUserID)
	best, bestSim := bestMatch(d.profiles[key], emb)

	// Match: fold the embedding into the centroid (running mean) so profiles
	// track drift (mic position, cold voice). Persist async — a lost update
	// only costs one sample of centroid refinement.
	if best != nil && (bestSim >= d.threshold || len(d.profiles[key]) >= d.maxSpeakers) {
		best.centroid = updateCentroid(best.centroid, emb, best.sampleCount)
		best.sampleCount++
		go func(id int64, c []float32, n int) {
			if err := db.UpdateSpeakerEmbedding(context.Background(), id, c, n); err != nil {
				log.Printf("Diarization: persist centroid for speaker %d: %v", id, err)
			}
		}(best.id, append([]float32(nil), best.centroid...), best.sampleCount)
		return best.id, best.displayLabel()
	}

	// New voice: enroll. Insert synchronously — the ID must go on this line.
	label := fmt.Sprintf("S%d", len(d.profiles[key])+1)
	id, err := db.InsertSpeaker(ctx, campaignID, discordUserID, label, emb)
	if err != nil {
		log.Printf("Diarization: enroll speaker for %s: %v", discordUserID, err)
		return 0, ""
	}
	d.profiles[key] = append(d.profiles[key], &speakerProfile{
		id: id, label: label, centroid: emb, sampleCount: 1,
	})
	log.Printf("Diarization: new speaker %s for user %s (campaign %d, similarity to closest %.2f)",
		label, discordUserID, campaignID, bestSim)
	return id, label
}

// embedLocked computes an L2-normalized embedding. Caller must hold d.mu.
func (d *Diarizer) embedLocked(samples16k []float32) []float32 {
	stream := d.ex.CreateStream()
	defer sherpa.DeleteOnlineStream(stream)

	stream.AcceptWaveform(recognizerSampleRate, samples16k)
	stream.InputFinished()
	if !d.ex.IsReady(stream) {
		return nil
	}
	return l2normalize(d.ex.Compute(stream))
}

// ensureLoadedLocked lazily loads a campaign's speaker profiles from the DB.
func (d *Diarizer) ensureLoadedLocked(ctx context.Context, campaignID int64) error {
	if d.loaded[campaignID] {
		return nil
	}
	rows, err := db.GetSpeakersByCampaign(ctx, campaignID)
	if err != nil {
		return err
	}
	for _, r := range rows {
		key := profileKey(campaignID, r.DiscordUserID)
		name := ""
		if r.DisplayName != nil {
			name = *r.DisplayName
		}
		d.profiles[key] = append(d.profiles[key], &speakerProfile{
			id: r.ID, label: r.Label, displayName: name,
			centroid: l2normalize(r.Embedding), sampleCount: r.SampleCount,
		})
	}
	d.loaded[campaignID] = true
	return nil
}

// SetDisplayName updates the cached display name for a speaker (call after
// persisting via db.UpdateSpeakerDisplayName so live labels pick it up
// without a restart). No-op for unknown IDs or a nil Diarizer.
func (d *Diarizer) SetDisplayName(speakerID int64, name string) {
	if d == nil {
		return
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	for _, profiles := range d.profiles {
		for _, p := range profiles {
			if p.id == speakerID {
				p.displayName = name
				return
			}
		}
	}
}

// Remove evicts a speaker profile from the in-memory cache (call after
// db.DeleteSpeaker). The voice re-enrolls as a new speaker on its next
// utterance. No-op on nil.
func (d *Diarizer) Remove(speakerID int64) {
	if d == nil {
		return
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	for key, profiles := range d.profiles {
		for i, p := range profiles {
			if p.id == speakerID {
				d.profiles[key] = append(profiles[:i], profiles[i+1:]...)
				return
			}
		}
	}
}

// Close frees the native extractor. Safe on nil.
func (d *Diarizer) Close() error {
	if d == nil || d.ex == nil {
		return nil
	}
	sherpa.DeleteSpeakerEmbeddingExtractor(d.ex)
	d.ex = nil
	return nil
}

func profileKey(campaignID int64, userID string) string {
	return fmt.Sprintf("%d:%s", campaignID, userID)
}

// bestMatch returns the profile with the highest cosine similarity to emb.
func bestMatch(profiles []*speakerProfile, emb []float32) (*speakerProfile, float32) {
	var best *speakerProfile
	bestSim := float32(-1)
	for _, p := range profiles {
		if sim := cosineSim(p.centroid, emb); sim > bestSim {
			best, bestSim = p, sim
		}
	}
	return best, bestSim
}

// cosineSim computes cosine similarity; inputs need not be normalized.
func cosineSim(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return -1
	}
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return -1
	}
	return float32(dot / (math.Sqrt(na) * math.Sqrt(nb)))
}

// l2normalize returns v scaled to unit length (copy; v is not modified).
func l2normalize(v []float32) []float32 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 {
		return append([]float32(nil), v...)
	}
	inv := float32(1 / math.Sqrt(sum))
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = x * inv
	}
	return out
}

// updateCentroid folds a new (normalized) embedding into a running-mean
// centroid of n prior samples, re-normalizing the result. Weighting by n keeps
// early samples from being washed out by a noisy newcomer while still letting
// the profile drift with the voice.
func updateCentroid(centroid, emb []float32, n int) []float32 {
	if len(centroid) != len(emb) || n <= 0 {
		return centroid
	}
	out := make([]float32, len(centroid))
	fn := float32(n)
	for i := range centroid {
		out[i] = (centroid[i]*fn + emb[i]) / (fn + 1)
	}
	return l2normalize(out)
}
