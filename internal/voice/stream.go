package voice

import (
	"encoding/binary"
	"log"
	"math"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	"github.com/joe/dndscribe-go/internal/audio"

	"github.com/bwmarrin/discordgo"
	"gopkg.in/hraban/opus.v2"
)

const (
	// Discord sends stereo opus at 48kHz.
	sampleRate   = 48000
	channels     = 2 // Decode as stereo, downmix to mono at flush time.
	frameSamples = 960 // 20ms at 48kHz per channel
	// Max opus frame size at 48kHz (120ms).
	maxPCMFrameSize = 5760 * channels

	// Silence detection: flush after 1.5s of no packets OR no loud frames.
	// WHY 1.5s not 800ms: shorter timeouts cause false flushes during natural
	// speech pauses and network jitter, splitting utterances. Longer (e.g.
	// 5s) adds too much transcription latency.
	silenceTimeout = 1500 * time.Millisecond
	// Max buffer duration before forced flush.
	maxBufferDuration = 15 * time.Second
	// Minimum buffer size after stereo-to-mono downmix: 48kHz * 1ch * 2bytes * 0.25s = 24000 bytes.
	minBufferBytes = 24000
	// Minimum RMS threshold -- reject near-silent buffers.
	minRMSThreshold = 50
)

// FlushCallback is called when a user's audio buffer is flushed.
// pcm is mono s16le at 48kHz. audioFilename is the saved WAV filename
// (basename only, relative to the recordings dir) or empty if saving is disabled.
type FlushCallback func(userID string, pcm []byte, timestamp time.Time, username, nickname string, rms float32, durationMs int, audioFilename string)

// FrameCallback is called per decoded voice packet with mono s16le 48kHz
// samples, for streaming engines that consume audio continuously. Only wired in
// streaming mode; nil otherwise so the batch (whisper) path has zero overhead.
// The slice is reused across calls — consume it synchronously, do not retain it.
type FrameCallback func(userID, username, nickname string, mono48k []int16)

// pktInput is one voice packet queued for the stream's decode goroutine.
type pktInput struct {
	opus      []byte
	timestamp uint32
	seq       uint16
}

// streamPktChanSize bounds per-stream packet backlog. At 50pps, 256 packets is
// ~5s of audio for one speaker. WHY this size: enough to absorb transient
// decode stalls (opus first-frame warmup, scheduler hiccup, disk flush) without
// silently dropping packets, but small enough to surface real backpressure via
// the drop counter rather than unbounded memory growth.
const streamPktChanSize = 256

// UserStream manages a single user's audio: DAVE decryption, opus decoding,
// buffer accumulation, silence detection, and flush. Each stream runs its own
// decode goroutine fed by pktCh, so slow work on one speaker's stream cannot
// stall packet receipt or decoding for any other speaker.
type UserStream struct {
	mu sync.Mutex

	decoder *opus.Decoder

	// pcmScratch is reused across decode calls to avoid per-packet allocations
	// (~23KB per packet × 50pps × N speakers was driving GC pressure that
	// caused STW pauses visible as choppy audio under multi-speaker load).
	pcmScratch []int16

	// Stereo PCM sample buffer (interleaved L/R). No phantom-data fill —
	// packets are decoded and appended directly, matching the original
	// dndscribe Python project (src/bufferwavesink.py). Gaps in the RTP
	// timeline that would previously be PLC/zero-filled now just compress
	// the buffer timeline; Discord's behavior of pausing packet delivery
	// during user silence means those gaps correspond to utterance
	// boundaries that silenceTimeout will catch and flush.
	buffer    []int16
	startTime time.Time
	lastPacket time.Time
	// lastSpeechTime is the wall-clock time of the most recent decoded frame
	// whose amplitude was above the speech threshold. Used to detect "user
	// stopped talking" independently of "packets stopped arriving" — useful
	// if any Discord client does continue sending DTX comfort-noise frames
	// during quiet periods (most don't, but the check is cheap insurance).
	lastSpeechTime time.Time

	userID   string
	username string
	nickname string

	// DAVE state for this user.
	daveState  *discordgo.ReceiverState
	daveActive bool

	// Save raw recordings.
	saveRaw    bool
	saveDir    string
	sessionID  int64

	onFlush FlushCallback
	// onFrame, when set (streaming mode), receives every decoded mono frame.
	onFrame     FrameCallback
	monoScratch []int16 // reused downmix buffer for onFrame, avoids per-packet alloc

	// Per-stream packet pipeline.
	// closeSignal closes on shutdown; decodeLoop watches it to exit.
	// pktCh is NEVER closed: SendPacket can be called from the recorder's
	// receive goroutine concurrently with Close, and closing pktCh while a
	// send is in flight would panic. Using a separate close signal lets us
	// signal "stop" without creating a send-on-closed-channel race.
	pktCh       chan *pktInput
	closeSignal chan struct{}
	closeOnce   sync.Once
	decodeDone  chan struct{}
	dropped     uint64 // count of packets dropped due to full channel

	// vc is the voice connection this stream belongs to. Used on garbage
	// detection to re-derive the DAVE receiver key from the current session
	// (recovers from wrong-key decryption during epoch transitions without
	// waiting for the OnDAVETransition callback to fire).
	vc *discordgo.VoiceConnection
}

// NewUserStream creates a new per-user audio stream with opus decoder and
// spawns the decode goroutine. vc is retained for garbage-triggered DAVE key
// re-derivation; pass nil only in tests or when re-derivation isn't wanted.
func NewUserStream(userID string, vc *discordgo.VoiceConnection, onFlush FlushCallback, onFrame FrameCallback) (*UserStream, error) {
	dec, err := opus.NewDecoder(sampleRate, channels)
	if err != nil {
		return nil, err
	}

	now := time.Now()
	s := &UserStream{
		decoder:     dec,
		pcmScratch:  make([]int16, maxPCMFrameSize),
		buffer:      make([]int16, 0, sampleRate*channels*2), // pre-allocate ~2s
		startTime:   now,
		lastPacket:  now,
		userID:      userID,
		onFlush:     onFlush,
		onFrame:     onFrame,
		pktCh:       make(chan *pktInput, streamPktChanSize),
		closeSignal: make(chan struct{}),
		decodeDone:  make(chan struct{}),
		vc:          vc,
	}
	go s.decodeLoop()
	return s, nil
}

// decodeLoop is the per-stream worker that consumes packets from pktCh and
// performs DAVE decryption + opus decode off the recorder's critical path.
// Exits when closeSignal fires; drains any packets still queued first so no
// audio that was already accepted by SendPacket is silently discarded.
func (s *UserStream) decodeLoop() {
	defer close(s.decodeDone)
	for {
		select {
		case p := <-s.pktCh:
			s.processPacket(p.opus, p.timestamp, p.seq)
		case <-s.closeSignal:
			// Drain remaining queued packets before exiting.
			for {
				select {
				case p := <-s.pktCh:
					s.processPacket(p.opus, p.timestamp, p.seq)
				default:
					return
				}
			}
		}
	}
}

// SendPacket enqueues a packet for the decode goroutine. Returns false if the
// channel is full or the stream is closed. Non-blocking by design: dropping a
// single packet on an overloaded stream is strictly better than stalling the
// recorder and losing packets across ALL streams.
//
// Note: packets sent after close() may be accepted into pktCh and never
// consumed. That's a deliberate tradeoff — we never close pktCh, so the send
// can't panic. Leaked packets are a transient shutdown-only condition.
func (s *UserStream) SendPacket(opus []byte, timestamp uint32, seq uint16) bool {
	select {
	case <-s.closeSignal:
		return false
	default:
	}
	select {
	case s.pktCh <- &pktInput{opus: opus, timestamp: timestamp, seq: seq}:
		return true
	default:
		dropped := atomic.AddUint64(&s.dropped, 1)
		if dropped%50 == 1 {
			log.Printf("Stream %s: packet channel full, dropped %d packets total", s.userID, dropped)
		}
		return false
	}
}

// SetDAVEState sets the DAVE receiver state for decrypting this user's frames.
func (s *UserStream) SetDAVEState(rs *discordgo.ReceiverState) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.daveState = rs
	s.daveActive = false
}

// ResetDAVE updates the DAVE state and resets the active flag.
// Called on SSRC reconnection to prevent stale key decryption.
func (s *UserStream) ResetDAVE(rs *discordgo.ReceiverState) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.daveState = rs
	s.daveActive = false
	log.Printf("Reset DAVE state for user %s", s.userID)
}

// FlushOnDisconnect flushes any pre-disconnect audio as a complete utterance.
// Called when a user reconnects with a new SSRC.
//
// WHY this replaces the previous InsertSilenceForDisconnect: the old version
// appended up to 30s of wall-clock silence onto the existing buffer, which
// could push bufferDuration well past maxBufferDuration (15s) and produce
// transcription records with startTime tens of seconds in the past. Flushing
// instead preserves the pre-disconnect utterance with its correct timestamp
// and lets the next packet post-reconnect start a fresh buffer. For a
// transcription use case, treating disconnect as an utterance boundary is
// both simpler and more correct.
func (s *UserStream) FlushOnDisconnect() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.flushLocked()
}

// SetUsername sets the display name for this user.
func (s *UserStream) SetUsername(name string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.username = name
}

// SetNickname sets the character name for this user.
func (s *UserStream) SetNickname(name string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.nickname = name
}

// Nickname returns the current character name.
func (s *UserStream) Nickname() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.nickname
}

// SetSaveRaw enables/disables raw WAV saving.
func (s *UserStream) SetSaveRaw(enabled bool, dir string, sessionID int64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.saveRaw = enabled
	s.saveDir = dir
	s.sessionID = sessionID
}

// findDAVEFrame scans the last few bytes for the 0xFAFA DAVE secure frame
// trailer. RTP extension stripping can leave extra trailing bytes, so we scan
// up to 8 positions rather than checking only the final two bytes.
func findDAVEFrame(data []byte) ([]byte, bool) {
	if len(data) < 13 {
		return data, false
	}
	limit := 8
	if maxLimit := len(data) - 12; maxLimit < limit {
		limit = maxLimit
	}
	for offset := 0; offset < limit; offset++ {
		pos := len(data) - 2 - offset
		if data[pos] == 0xFA && data[pos+1] == 0xFA {
			ss := int(data[pos-1])
			frameLen := pos + 2
			if ss >= 12 && frameLen-ss >= 1 {
				return data[:frameLen], true
			}
		}
	}
	return data, false
}

// bufferDurationLocked converts a stereo sample count into its wall-clock
// duration. Factored as a helper because the conversion shows up in the
// force-flush check and the silent-discard gate.
func bufferDurationLocked(stereoSamples int) time.Duration {
	return time.Duration(stereoSamples/channels) * time.Second / sampleRate
}

// isOpusSilence returns true for raw (unencrypted) opus DTX silence frames.
// Discord sends these outside of DAVE encryption even during active sessions.
func isOpusSilence(data []byte) bool {
	return len(data) > 0 && len(data) <= 3 && data[0] == 0xF8
}

// garbageRMSThreshold is the int16 RMS above which we treat a decoded frame
// as likely wrong-key output and discard it. Rationale: normal loud speech
// tops out around RMS 6000-8000; decoder output from wrong-key AES-CTR
// plaintext is near-uniform noise at RMS 15000-19000. 12000 is comfortably
// between those. See DAVE-transition WHY note at the call site.
const garbageRMSThreshold = 12000

// retryWithRederivedKey attempts to recover from a garbage decode by
// re-deriving the DAVE receiver key from the current voice session and
// retrying the decrypt + decode. On success it returns the new PCM (a slice
// into s.pcmScratch — caller must consume before the scratch is reused) and
// also persists the new key to s.daveState so subsequent packets benefit.
// On any failure (no VC, re-derive error, still-garbage output) returns nil,
// false, and the caller should discard the frame.
//
// Caller must hold s.mu.
func (s *UserStream) retryWithRederivedKey(daveFrame []byte) ([]int16, bool) {
	if s.vc == nil {
		return nil, false
	}
	dave := s.vc.DAVESession()
	if dave == nil {
		return nil, false
	}
	rs, err := dave.DeriveReceiverKey(s.userID)
	if err != nil {
		return nil, false
	}

	decrypted, err := discordgo.DecryptFrame(rs, daveFrame)
	if err != nil {
		return nil, false
	}

	n, err := s.decoder.Decode(decrypted, s.pcmScratch)
	if err != nil || n == 0 {
		return nil, false
	}
	pcm := s.pcmScratch[:n*channels]
	if isGarbageAmplitude(pcm) {
		// New key also produced garbage — probably we re-derived the same
		// (still-wrong) key because the fork's DAVESession hasn't processed
		// the transition yet. Discard; the next packet will retry.
		return nil, false
	}

	// Retry succeeded: persist the new key so following packets don't hit
	// this same retry path. Only log once per recovery burst to keep output
	// readable during multi-packet transitions.
	s.daveState = rs
	log.Printf("Recovered DAVE key for %s via re-derive (wrong-key decryption detected)", s.userID)
	return pcm, true
}

// speechAmplitudeThreshold is the peak int16 amplitude above which we consider
// a frame to contain real speech (as opposed to DTX comfort noise / room tone).
// Opus DTX output is typically well under 200 peak; normal speech peaks
// several thousand. 500 reliably distinguishes without tripping on whispers.
const speechAmplitudeThreshold = 500

// isFrameLoudEnough returns true if the frame has samples above the speech
// threshold, meaning the user is actually talking (not just sending DTX).
func isFrameLoudEnough(pcm []int16) bool {
	for i := 0; i < len(pcm); i += 16 { // stride 16 for speed
		v := pcm[i]
		if v > speechAmplitudeThreshold || v < -speechAmplitudeThreshold {
			return true
		}
	}
	return false
}

// isGarbageAmplitude returns true if the frame's RMS is so high it's almost
// certainly not a real opus decode. Used to reject frames decoded from
// wrong-key-decrypted ciphertext during DAVE epoch transitions.
func isGarbageAmplitude(pcm []int16) bool {
	if len(pcm) == 0 {
		return false
	}
	var sumSq int64
	// Sample every 8th element for speed — garbage is uniformly noisy, a
	// sparse sample is enough to distinguish from speech.
	var count int
	for i := 0; i < len(pcm); i += 8 {
		v := int64(pcm[i])
		sumSq += v * v
		count++
	}
	if count == 0 {
		return false
	}
	rms := math.Sqrt(float64(sumSq) / float64(count))
	return rms > garbageRMSThreshold
}

// processPacket decrypts, decodes, and buffers a single voice packet. Called
// only by the stream's own decodeLoop goroutine, so s.mu does not need to
// guard against concurrent decodes — but we still lock for coordination with
// CheckSilence / Flush / ResetDAVE / setters which run on other goroutines.
//
// WHY there's no gap-fill or PLC: we now do what the original Python project
// does (src/bufferwavesink.py) — take raw decoded PCM from each packet and
// append it. If Discord stops sending packets during user silence, the
// timeline just "compresses" across those gaps, and silenceTimeout catches
// the user-stopped-talking case and flushes. Phantom data (PLC frames,
// zero-fill) created more problems than it solved: decoder state degradation,
// wrong timestamps, complexity for little audible benefit.
func (s *UserStream) processPacket(opusData []byte, timestamp uint32, seq uint16) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// DAVE decryption.
	daveFrame, isDave := findDAVEFrame(opusData)
	if isDave {
		if s.daveState == nil {
			return
		}
		decrypted, err := discordgo.DecryptFrame(s.daveState, daveFrame)
		if err != nil {
			log.Printf("DAVE decrypt failed for %s (seq=%d, %d bytes): %v",
				s.userID, seq, len(daveFrame), err)
			return
		}
		opusData = decrypted
		s.daveActive = true
	} else if s.daveActive {
		if isOpusSilence(opusData) {
			// Fall through to normal opus decode — short 0xF8 frames are raw
			// opus DTX silence indicators, opus decodes them to quiet comfort
			// noise.
		} else {
			// Non-DAVE, non-silence packet after DAVE was active = lost or
			// corrupt. Skip it.
			return
		}
	} else {
		// Skip everything before DAVE is active.
		return
	}

	// Decode opus into the reusable scratch buffer (avoids per-packet alloc).
	n, err := s.decoder.Decode(opusData, s.pcmScratch)
	if err != nil {
		log.Printf("Opus decode error for %s: %v", s.userID, err)
		return
	}
	pcm := s.pcmScratch[:n*channels] // n samples per channel, interleaved

	// Recover from wrong-key garbage decodes during DAVE epoch transitions —
	// the fork's DecryptFrame skips GCM tag verification (Go stdlib won't do
	// DAVE's 8-byte truncated tags), so a post-transition packet decrypted
	// under our stale key produces random bytes that opus decodes as
	// full-scale noise. RMS > 12000 reliably indicates this; re-derive the
	// key and retry. If the retry is also garbage, discard the frame.
	if isDave && isGarbageAmplitude(pcm) {
		if retryPCM, ok := s.retryWithRederivedKey(daveFrame); ok {
			pcm = retryPCM
		} else {
			return
		}
	}

	s.lastPacket = time.Now()

	// Track last time we saw real speech (vs DTX comfort noise). Used by
	// CheckSilence to flush when the user has stopped talking even if some
	// Discord client is still forwarding DTX packets. Defensive — most
	// clients stop sending entirely, in which case silenceTimeout via
	// lastPacket is the primary trigger.
	if isFrameLoudEnough(pcm) {
		s.lastSpeechTime = time.Now()
	}

	// Streaming tap: feed every decoded frame to the streaming engine for live
	// partials. Downmix into the reused scratch to avoid a per-packet alloc.
	// Whisper mode leaves onFrame nil, so this is skipped entirely.
	if s.onFrame != nil {
		s.monoScratch = audio.DownmixStereoToMonoInto(s.monoScratch, pcm)
		s.onFrame(s.userID, s.username, s.nickname, s.monoScratch)
	}

	// Initialize start time on first actual audio.
	if len(s.buffer) == 0 {
		s.startTime = time.Now()
	}

	// append copies from pcm — safe to reuse s.pcmScratch on the next call.
	s.buffer = append(s.buffer, pcm...)

	// Force flush if buffer exceeds max duration.
	if bufferDurationLocked(len(s.buffer)) >= maxBufferDuration {
		s.flushLocked()
	}
}

// CheckSilence checks if the user has been silent long enough to flush.
// Returns true if a flush occurred.
func (s *UserStream) CheckSilence(now time.Time) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.buffer) == 0 {
		return false
	}

	timeSincePacket := now.Sub(s.lastPacket)

	// Silent-buffer-discard: if the buffer is mostly silence AND has
	// accumulated at least 500ms worth of samples, drop it and reset the
	// timestamp. Prevents startTime from drifting when Discord keeps
	// forwarding DTX comfort-noise packets (which keep timeSincePacket low
	// forever, so a time-gated check would never fire).
	// WHY the 500ms buffer minimum: avoids discarding brief pre-speech DTX
	// that would be stripped by leading-silence trim anyway; also avoids a
	// tight loop of discard-append-discard on fresh silent packets.
	if bufferDurationLocked(len(s.buffer)) >= 500*time.Millisecond && s.isBufferSilent() {
		s.buffer = s.buffer[:0]
		s.startTime = time.Now()
		s.lastSpeechTime = time.Time{}
		return false
	}

	if timeSincePacket >= silenceTimeout {
		s.flushLocked()
		return true
	}

	// Speech-timeout: flush if the user has actually stopped talking, even if
	// Discord is still forwarding DTX comfort-noise packets. Without this, a
	// short utterance followed by DTX would sit in the buffer until the 15s
	// force-flush and the record would claim a startTime far older than the
	// real end of speech.
	if !s.lastSpeechTime.IsZero() && now.Sub(s.lastSpeechTime) >= silenceTimeout {
		s.flushLocked()
		return true
	}
	return false
}

// isBufferSilent checks if the buffer is predominantly silence.
// Returns true if the RMS of the buffer is below the threshold.
func (s *UserStream) isBufferSilent() bool {
	if len(s.buffer) == 0 {
		return true
	}
	// Sample every 20th sample for speed
	var sumSquares float64
	var count int
	for i := 0; i < len(s.buffer); i += 20 {
		sample := float64(s.buffer[i])
		sumSquares += sample * sample
		count++
	}
	if count == 0 {
		return true
	}
	rms := math.Sqrt(sumSquares / float64(count))
	return rms < float64(minRMSThreshold)
}

// Flush forces a flush of the current buffer.
func (s *UserStream) Flush() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.flushLocked()
}

// Close shuts down the decode goroutine and flushes any remaining audio.
// Safe to call multiple times; second and later calls are no-ops.
func (s *UserStream) Close() {
	s.closeOnce.Do(func() {
		close(s.closeSignal)
		<-s.decodeDone
		s.mu.Lock()
		s.flushLocked()
		s.mu.Unlock()
	})
}

// flushLocked performs the actual flush. Caller must hold s.mu.
func (s *UserStream) flushLocked() {
	if len(s.buffer) == 0 {
		return
	}

	stereo := s.buffer
	timestamp := s.startTime
	username := s.username
	nickname := s.nickname

	// Reset buffer and per-utterance timers. Zeroing lastSpeechTime means the
	// next utterance starts with a clean speech-timeout window and won't
	// misfire a flush on a stale timer before any real speech arrives.
	s.buffer = make([]int16, 0, sampleRate*channels*2)
	s.startTime = time.Now()
	s.lastSpeechTime = time.Time{}

	// Trim leading silence -- the buffer often starts with a long silent
	// section from the RTP timestamp gap between stream creation and first
	// actual speech. We scan forward in stereo frame pairs until we find
	// samples above a threshold.
	trimThreshold := int16(100) // ~0.3% of full scale
	trimmed := 0
	for i := 0; i+1 < len(stereo); i += channels {
		if stereo[i] > trimThreshold || stereo[i] < -trimThreshold ||
			stereo[i+1] > trimThreshold || stereo[i+1] < -trimThreshold {
			break
		}
		trimmed += channels
	}
	if trimmed > 0 && trimmed < len(stereo) {
		stereo = stereo[trimmed:]
		// Adjust timestamp forward by the trimmed duration.
		trimDuration := time.Duration(trimmed/channels) * time.Second / time.Duration(sampleRate)
		timestamp = timestamp.Add(trimDuration)
	}

	if len(stereo) == 0 {
		return
	}

	// Downmix stereo to mono.
	mono := audio.DownmixStereoToMono(stereo)

	// Convert mono int16 samples to bytes for the callback and WAV saving.
	monoBytes := int16SliceToBytes(mono)

	// Compute RMS.
	rms := audio.ComputeRMS(mono)

	// Duration in ms.
	durationMs := int(math.Round(float64(len(mono)) / float64(sampleRate) * 1000))

	// Save raw recording if enabled. Compute filename deterministically before
	// the async goroutine so we can pass it to the flush callback.
	// WHY: The frontend needs the audio filename in TranscriptionLine to show
	// play buttons. We compute the filename synchronously, save async, and pass
	// the basename to the callback. By the time the user clicks play, the file
	// will be written.
	var audioFilename string
	if s.saveRaw && s.saveDir != "" {
		dir := filepath.Join(s.saveDir, "raw")
		ts := timestamp.Format("2006-01-02_15-04-05")
		rmsStr := int(math.Round(float64(rms)))
		safeName := username
		if safeName == "" {
			safeName = s.userID
		}
		basename := ts + "_" + itoa(rmsStr) + "_" + safeName + ".wav"
		fullPath := filepath.Join(dir, basename)
		// Store just the basename for the DB. The server adds the "raw/" path
		// prefix when serving audio files.
		audioFilename = basename

		go func() {
			if err := os.MkdirAll(dir, 0o755); err != nil {
				log.Printf("Failed to create recordings dir: %v", err)
				return
			}
			if err := audio.WriteWAV(fullPath, monoBytes, sampleRate, 1, 16); err != nil {
				log.Printf("Failed to save recording %s: %v", fullPath, err)
			}
		}()
	}

	// Apply filters before sending to transcription.
	if len(monoBytes) < minBufferBytes {
		return
	}
	if rms < minRMSThreshold {
		return
	}

	if s.onFlush != nil {
		s.onFlush(s.userID, monoBytes, timestamp, username, nickname, rms, durationMs, audioFilename)
	}
}

// int16SliceToBytes converts a slice of int16 to little-endian bytes.
func int16SliceToBytes(samples []int16) []byte {
	buf := make([]byte, len(samples)*2)
	for i, s := range samples {
		binary.LittleEndian.PutUint16(buf[i*2:], uint16(s))
	}
	return buf
}

// itoa is a simple int-to-string helper.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	s := ""
	neg := false
	if n < 0 {
		neg = true
		n = -n
	}
	for n > 0 {
		s = string(rune('0'+n%10)) + s
		n /= 10
	}
	if neg {
		s = "-" + s
	}
	return s
}
