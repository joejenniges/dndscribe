package voice

import (
	"log"
	"sync"
	"time"

	"github.com/bwmarrin/discordgo"
)

const (
	maxPendingPackets = 100 // ~2 seconds at 50 packets/sec
	silenceCheckHz    = 100 * time.Millisecond
)

// Recorder manages per-user audio streams for a voice session. It maps SSRCs
// to users, handles DAVE key derivation, and routes packets to per-user streams.
type Recorder struct {
	mu sync.RWMutex

	streams        map[uint32]*UserStream  // SSRC -> stream
	ssrcToUser     map[uint32]string       // SSRC -> userID
	userToSSRC     map[string]uint32       // userID -> current SSRC
	pendingPackets map[uint32][]*discordgo.Packet

	ignoredUsers   map[string]bool
	characterNames map[string]string
	userNames      map[string]string

	sessionID  int64
	campaignID int64
	saveRaw    bool
	saveDir    string
	paused     bool

	done chan struct{}

	onFlush FlushCallback
	onFrame FrameCallback
}

// NewRecorder creates a new voice recorder for a session. onFrame is optional
// (nil in batch/whisper mode); when set, each user's decoded frames are tapped
// for the streaming engine.
func NewRecorder(sessionID, campaignID int64, ignoredUsers map[string]bool, characterNames map[string]string, onFlush FlushCallback, onFrame FrameCallback) *Recorder {
	if ignoredUsers == nil {
		ignoredUsers = make(map[string]bool)
	}
	if characterNames == nil {
		characterNames = make(map[string]string)
	}
	return &Recorder{
		streams:        make(map[uint32]*UserStream),
		ssrcToUser:     make(map[uint32]string),
		userToSSRC:     make(map[string]uint32),
		pendingPackets: make(map[uint32][]*discordgo.Packet),
		ignoredUsers:   ignoredUsers,
		characterNames: characterNames,
		userNames:      make(map[string]string),
		sessionID:      sessionID,
		campaignID:     campaignID,
		done:           make(chan struct{}),
		onFlush:        onFlush,
		onFrame:        onFrame,
	}
}

// Start registers voice handlers and begins packet processing.
func (r *Recorder) Start(vc *discordgo.VoiceConnection) {
	log.Printf("Recorder starting (OpusRecv=%v)", vc.OpusRecv != nil)

	// Register speaking update handler.
	vc.AddHandler(func(vc *discordgo.VoiceConnection, vs *discordgo.VoiceSpeakingUpdate) {
		log.Printf("Speaking update: user=%s ssrc=%d", vs.UserID, vs.SSRC)
		r.HandleSpeakingUpdate(vc, uint32(vs.SSRC), vs.UserID)
	})

	// Packet receive loop.
	go func() {
		var pktCount int64
		for {
			select {
			case <-r.done:
				log.Printf("Recorder stopped after %d packets", pktCount)
				return
			case pkt, ok := <-vc.OpusRecv:
				if !ok {
					log.Printf("OpusRecv channel closed after %d packets", pktCount)
					return
				}
				if pkt != nil {
					pktCount++
					if pktCount == 1 {
						log.Printf("First voice packet received (SSRC=%d)", pkt.SSRC)
					}
					r.HandlePacket(pkt)
				}
			}
		}
	}()

	// Silence detection loop.
	go func() {
		ticker := time.NewTicker(silenceCheckHz)
		defer ticker.Stop()
		for {
			select {
			case <-r.done:
				return
			case now := <-ticker.C:
				r.checkSilence(now)
			}
		}
	}()
}

// HandleSpeakingUpdate maps an SSRC to a user ID and creates/reuses a UserStream.
//
// WHY the DAVE key derivation runs OUTSIDE r.mu: DeriveReceiverKey performs
// HKDF and allocations that can take milliseconds. Previously we held the
// recorder lock across it, which blocked the packet receive loop for every
// speaking update (= every user join/leave during active multi-speaker use).
// Under a DAVE transition that fires ReDeriveAllDAVEKeys across N streams,
// the stall is multiplied by N. The fix is to do the crypto with no lock,
// then take the lock only long enough to apply the result.
func (r *Recorder) HandleSpeakingUpdate(vc *discordgo.VoiceConnection, ssrc uint32, userID string) {
	// Derive DAVE key without holding r.mu.
	var daveState *discordgo.ReceiverState
	if dave := vc.DAVESession(); dave != nil {
		rs, err := dave.DeriveReceiverKey(userID)
		if err != nil {
			log.Printf("DAVE key derivation FAILED for %s (ssrc=%d): %v", userID, ssrc, err)
		} else {
			daveState = rs
		}
	} else {
		log.Printf("No DAVE session available for speaking update (user=%s, ssrc=%d)", userID, ssrc)
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	prevUser, hadSSRC := r.ssrcToUser[ssrc]
	r.ssrcToUser[ssrc] = userID

	if daveState != nil {
		if hadSSRC {
			log.Printf("DAVE key re-derived for existing user %s (ssrc=%d, prev=%s)", userID, ssrc, prevUser)
		} else {
			log.Printf("DAVE key derived for new user %s (ssrc=%d)", userID, ssrc)
		}
	}

	// Stream already exists for this exact SSRC -- update DAVE state and return.
	if existing, ok := r.streams[ssrc]; ok {
		log.Printf("Updating DAVE state for existing stream (user=%s, ssrc=%d)", userID, ssrc)
		existing.ResetDAVE(daveState)
		return
	}

	// Check if this user already has a stream under a previous SSRC (reconnect).
	// WHY: Discord reassigns SSRCs when users rejoin or when new users join.
	// We must: (1) insert silence for the disconnect gap, (2) reset DAVE state
	// so the first packet of the new session isn't decrypted with a stale key.
	if oldSSRC, ok := r.userToSSRC[userID]; ok && oldSSRC != ssrc {
		if existing, ok := r.streams[oldSSRC]; ok {
			existing.FlushOnDisconnect()
			existing.ResetDAVE(daveState)

			r.streams[ssrc] = existing
			delete(r.streams, oldSSRC)
			r.userToSSRC[userID] = ssrc
			log.Printf("User %s reconnected with new SSRC %d (was %d)", userID, ssrc, oldSSRC)

			// Route pending packets through the stream's channel.
			if pending, ok := r.pendingPackets[ssrc]; ok {
				for _, pkt := range pending {
					existing.SendPacket(pkt.Opus, pkt.Timestamp, pkt.Sequence)
				}
				delete(r.pendingPackets, ssrc)
			}
			return
		}
	}

	r.userToSSRC[userID] = ssrc

	us, err := NewUserStream(userID, vc, r.onFlush, r.onFrame)
	if err != nil {
		log.Printf("Failed to create stream for user %s: %v", userID, err)
		return
	}
	us.SetDAVEState(daveState)

	if name, ok := r.userNames[userID]; ok {
		us.SetUsername(name)
	}
	if name, ok := r.characterNames[userID]; ok {
		us.SetNickname(name)
	}
	if r.saveRaw {
		us.SetSaveRaw(true, r.saveDir, r.sessionID)
	}

	r.streams[ssrc] = us
	log.Printf("Recording user %s (SSRC=%d)", userID, ssrc)

	if pending, ok := r.pendingPackets[ssrc]; ok {
		for _, pkt := range pending {
			us.SendPacket(pkt.Opus, pkt.Timestamp, pkt.Sequence)
		}
		delete(r.pendingPackets, ssrc)
	}
}

// HandlePacket routes a voice packet to the correct UserStream. It only holds
// r.mu long enough to look up the target stream; the decode + decrypt happens
// on the stream's own goroutine so slow work on one speaker can't stall
// receipt for any other speaker.
func (r *Recorder) HandlePacket(pkt *discordgo.Packet) {
	r.mu.Lock()

	if r.paused {
		r.mu.Unlock()
		return
	}

	if userID, ok := r.ssrcToUser[pkt.SSRC]; ok {
		if r.ignoredUsers[userID] {
			r.mu.Unlock()
			return
		}
	}

	us, ok := r.streams[pkt.SSRC]
	if ok {
		r.mu.Unlock()
		us.SendPacket(pkt.Opus, pkt.Timestamp, pkt.Sequence)
		return
	}

	// Unknown SSRC: buffer until a speaking update maps it.
	buf := r.pendingPackets[pkt.SSRC]
	if len(buf) >= maxPendingPackets {
		buf = buf[1:]
	}
	r.pendingPackets[pkt.SSRC] = append(buf, pkt)
	r.mu.Unlock()
}

// checkSilence runs the silence check on all active streams.
func (r *Recorder) checkSilence(now time.Time) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	for _, us := range r.streams {
		us.CheckSilence(now)
	}
}

// SetUserName sets the display name for a user.
func (r *Recorder) SetUserName(userID, username string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.userNames[userID] = username

	// Update any active stream.
	if ssrc, ok := r.userToSSRC[userID]; ok {
		if us, ok := r.streams[ssrc]; ok {
			us.SetUsername(username)
		}
	}
}

// SetCharacterName sets the character name for a user.
func (r *Recorder) SetCharacterName(userID, name string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.characterNames[userID] = name

	if ssrc, ok := r.userToSSRC[userID]; ok {
		if us, ok := r.streams[ssrc]; ok {
			us.SetNickname(name)
		}
	}
}

// RemoveCharacterName removes the character name for a user.
func (r *Recorder) RemoveCharacterName(userID string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	delete(r.characterNames, userID)

	if ssrc, ok := r.userToSSRC[userID]; ok {
		if us, ok := r.streams[ssrc]; ok {
			us.SetNickname("")
		}
	}
}

// UpdateIgnoredUsers replaces the set of ignored users.
func (r *Recorder) UpdateIgnoredUsers(ignored map[string]bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.ignoredUsers = ignored
}

// FlushUserBuffer forces a flush of a specific user's audio buffer.
func (r *Recorder) FlushUserBuffer(userID string) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if ssrc, ok := r.userToSSRC[userID]; ok {
		if us, ok := r.streams[ssrc]; ok {
			us.Flush()
		}
	}
}

// SetSaveRecordings enables or disables raw WAV saving for all streams.
func (r *Recorder) SetSaveRecordings(enabled bool) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.saveRaw = enabled
	for _, us := range r.streams {
		us.SetSaveRaw(enabled, r.saveDir, r.sessionID)
	}
}

// SetSaveDir sets the directory for saving raw recordings.
func (r *Recorder) SetSaveDir(dir string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.saveDir = dir
}

// SetPaused pauses or resumes recording.
func (r *Recorder) SetPaused(paused bool) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if paused && !r.paused {
		// Flush all buffers before pausing.
		for _, us := range r.streams {
			us.Flush()
		}
	}
	r.paused = paused
	log.Printf("Recording %s", map[bool]string{true: "paused", false: "resumed"}[paused])
}

// IsPaused returns whether recording is paused.
func (r *Recorder) IsPaused() bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.paused
}

// Stop signals the recorder to stop and flushes all remaining buffers.
// ReDeriveAllDAVEKeys re-derives DAVE receiver keys for all active streams.
// Called when a DAVE transition occurs (user joins/leaves triggering key renegotiation).
//
// WHY two passes with the lock released between them: derivation is the slow
// part (HKDF + allocations per user). Previously we held r.mu for the entire
// loop, which stalled packet receipt for the duration of N derivations. Now
// we briefly lock to snapshot the (ssrc, userID) set, derive with no lock,
// then briefly lock to apply. Stream.ResetDAVE takes its own per-stream lock
// so it's safe to call without r.mu held.
func (r *Recorder) ReDeriveAllDAVEKeys(vc *discordgo.VoiceConnection) {
	dave := vc.DAVESession()
	if dave == nil {
		log.Println("ReDeriveAllDAVEKeys: no DAVE session available")
		return
	}

	// Snapshot the user set under lock.
	type pair struct {
		ssrc   uint32
		userID string
		stream *UserStream
	}
	r.mu.RLock()
	pairs := make([]pair, 0, len(r.streams))
	for ssrc, us := range r.streams {
		userID, ok := r.ssrcToUser[ssrc]
		if !ok {
			continue
		}
		pairs = append(pairs, pair{ssrc: ssrc, userID: userID, stream: us})
	}
	r.mu.RUnlock()

	// Derive keys off-lock, then apply per-stream (each stream uses its own mutex).
	count := 0
	for _, p := range pairs {
		rs, err := dave.DeriveReceiverKey(p.userID)
		if err != nil {
			log.Printf("ReDeriveAllDAVEKeys: failed for user %s (ssrc=%d): %v", p.userID, p.ssrc, err)
			continue
		}
		p.stream.ResetDAVE(rs)
		count++
	}
	log.Printf("ReDeriveAllDAVEKeys: updated %d streams", count)
}

func (r *Recorder) Stop() {
	close(r.done)

	// Snapshot streams under lock, then close them without holding r.mu — each
	// stream's Close blocks on its decode goroutine draining, which can take a
	// moment, and we don't want to block other recorder operations.
	r.mu.Lock()
	streams := make([]*UserStream, 0, len(r.streams))
	for _, us := range r.streams {
		streams = append(streams, us)
	}
	r.mu.Unlock()

	for _, us := range streams {
		us.Close()
	}
	log.Println("Recorder stopped, all streams closed and buffers flushed")
}
