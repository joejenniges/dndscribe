package bot

import (
	"context"
	"fmt"
	"log"
	"strings"
	"sync"
	"time"

	"github.com/joe/dndscribe-go/internal/config"
	"github.com/joe/dndscribe-go/internal/db"
	"github.com/joe/dndscribe-go/internal/voice"
	"github.com/joe/dndscribe-go/internal/web"

	"github.com/bwmarrin/discordgo"
)

// TranscriptionWorker is the subset of the transcription worker that the bot
// needs. Avoids importing the transcribe package directly.
type TranscriptionWorker interface {
	AddTask(userID string, pcm []byte, timestamp time.Time, username, nickname string, rms float32, durationMs int, audioFilename string)
	FeedFrame(userID, username, nickname string, mono48k []int16)
	Finalize()
	SetSessionID(id *int64)
	SetCampaignID(id *int64)
}

// Bot manages the Discord session and coordinates voice recording.
type Bot struct {
	session *discordgo.Session
	guildID string
	config  *config.Config
	worker  TranscriptionWorker

	// Per-campaign state loaded from DB at startup.
	ignoredUsers   map[int64]map[string]bool   // campaignID -> set of userIDs
	characterNames map[int64]map[string]string  // campaignID -> userID -> name
	mu             sync.RWMutex

	// Active voice state.
	voiceState   *VoiceState
	voiceStateMu sync.Mutex

	// Save recordings preference (persists across sessions).
	saveRecordings bool

	// Registered slash commands for cleanup on shutdown.
	registeredCmds []*discordgo.ApplicationCommand

	// Event callbacks for the web layer.
	OnStatus         func()
	OnMembers        func()
	OnSessionStarted func(id int64, channelName string, campaignID int64)
	OnSaveRecordings func(enabled bool)
	OnIgnoredUsers   func(campaignID int64)
}

// VoiceState holds the state of an active voice recording session.
type VoiceState struct {
	Connection *discordgo.VoiceConnection
	Recorder   *voice.Recorder
	ChannelID  string
	CampaignID int64
	SessionID  int64
}

// New creates a new Bot. Call Start() to connect to Discord.
// The worker parameter can be nil if transcription is not yet available.
func New(cfg *config.Config, worker TranscriptionWorker) (*Bot, error) {
	dg, err := discordgo.New("Bot " + cfg.Discord.Token)
	if err != nil {
		return nil, fmt.Errorf("create discord session: %w", err)
	}

	dg.Identify.Intents = discordgo.IntentsGuilds |
		discordgo.IntentsGuildVoiceStates |
		discordgo.IntentsGuildMembers

	return &Bot{
		session:        dg,
		guildID:        cfg.Discord.GuildID,
		config:         cfg,
		worker:         worker,
		ignoredUsers:   make(map[int64]map[string]bool),
		characterNames: make(map[int64]map[string]string),
	}, nil
}

// Start opens the Discord connection, loads campaign data, and registers
// slash commands and event handlers.
func (b *Bot) Start() error {
	ctx := context.Background()

	// Load all campaign data from DB.
	if err := b.loadAllCampaignData(ctx); err != nil {
		log.Printf("Warning: failed to load campaign data: %v", err)
	}

	b.session.AddHandler(b.handleInteraction)
	b.session.AddHandler(b.handleVoiceStateUpdate)

	if err := b.session.Open(); err != nil {
		return fmt.Errorf("open discord session: %w", err)
	}

	if err := b.registerCommands(); err != nil {
		return fmt.Errorf("register commands: %w", err)
	}

	log.Println("Bot is running.")
	return nil
}

// Stop disconnects from voice, removes slash commands, and closes the session.
//
// WHY the bounded voice Disconnect: the underlying waitUntilStatus blocks on
// a sync.Cond.Wait() until Discord acknowledges the disconnect. On flaky
// network or during Discord-side degradation this can hang indefinitely, and
// we don't want a shutdown to wait forever for a best-effort cleanup.
func (b *Bot) Stop() error {
	b.voiceStateMu.Lock()
	if b.voiceState != nil {
		b.voiceState.Recorder.Stop()
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		if err := b.voiceState.Connection.Disconnect(ctx); err != nil {
			log.Printf("Voice disconnect: %v (continuing shutdown)", err)
		}
		cancel()
		b.voiceState = nil
	}
	b.voiceStateMu.Unlock()

	for _, cmd := range b.registeredCmds {
		if err := b.session.ApplicationCommandDelete(b.session.State.User.ID, b.guildID, cmd.ID); err != nil {
			log.Printf("Error removing command %s: %v", cmd.Name, err)
		}
	}

	return b.session.Close()
}

// loadAllCampaignData loads ignored users and character names for all campaigns.
func (b *Bot) loadAllCampaignData(ctx context.Context) error {
	campaigns, err := db.ListCampaigns(ctx)
	if err != nil {
		return err
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	for _, c := range campaigns {
		// Load ignored users.
		ignored, err := db.GetIgnoredUsers(ctx, c.ID)
		if err != nil {
			log.Printf("Failed to load ignored users for campaign %d: %v", c.ID, err)
			continue
		}
		set := make(map[string]bool, len(ignored))
		for _, u := range ignored {
			set[u.DiscordUserID] = true
		}
		b.ignoredUsers[c.ID] = set
		log.Printf("Loaded %d ignored users for campaign %d", len(set), c.ID)

		// Load character names.
		names, err := db.GetCharacterNames(ctx, c.ID)
		if err != nil {
			log.Printf("Failed to load character names for campaign %d: %v", c.ID, err)
			continue
		}
		b.characterNames[c.ID] = names
		log.Printf("Loaded %d character names for campaign %d", len(names), c.ID)
	}

	return nil
}

// getIgnoredSet returns the ignored user set for a campaign, creating it if needed.
func (b *Bot) getIgnoredSet(campaignID int64) map[string]bool {
	if set, ok := b.ignoredUsers[campaignID]; ok {
		return set
	}
	set := make(map[string]bool)
	b.ignoredUsers[campaignID] = set
	return set
}

// getCharacterNames returns the character name map for a campaign, creating it if needed.
func (b *Bot) getCharacterNameMap(campaignID int64) map[string]string {
	if m, ok := b.characterNames[campaignID]; ok {
		return m
	}
	m := make(map[string]string)
	b.characterNames[campaignID] = m
	return m
}

// --- BotController interface implementation ---

// GetVoiceChannels returns voice channels in the guild with their members.
func (b *Bot) GetVoiceChannels() []web.VoiceChannelInfo {
	channels, err := b.session.GuildChannels(b.guildID)
	if err != nil {
		log.Printf("Failed to get guild channels: %v", err)
		return nil
	}

	guild, err := b.session.State.Guild(b.guildID)
	if err != nil {
		log.Printf("Failed to get guild state: %v", err)
		return nil
	}

	// Build a map of channel ID -> member usernames from voice states.
	voiceMembers := make(map[string][]string)
	botUserID := b.session.State.User.ID
	for _, vs := range guild.VoiceStates {
		if vs.UserID == botUserID {
			continue
		}
		member, _ := b.session.GuildMember(b.guildID, vs.UserID)
		name := vs.UserID
		if member != nil && member.User != nil {
			name = member.User.Username
		}
		voiceMembers[vs.ChannelID] = append(voiceMembers[vs.ChannelID], name)
	}

	var result []web.VoiceChannelInfo
	for _, ch := range channels {
		if ch.Type != discordgo.ChannelTypeGuildVoice && ch.Type != discordgo.ChannelTypeGuildStageVoice {
			continue
		}
		members := voiceMembers[ch.ID]
		if members == nil {
			members = []string{}
		}
		result = append(result, web.VoiceChannelInfo{
			ID:      ch.ID,
			Name:    ch.Name,
			Members: members,
		})
	}
	return result
}

// GetVoiceMembers returns members in the currently active voice channel.
func (b *Bot) GetVoiceMembers() []web.VoiceMemberInfo {
	b.voiceStateMu.Lock()
	vs := b.voiceState
	b.voiceStateMu.Unlock()

	if vs == nil {
		return nil
	}

	guild, err := b.session.State.Guild(b.guildID)
	if err != nil {
		return nil
	}

	b.mu.RLock()
	charNames := b.characterNames[vs.CampaignID]
	if charNames == nil {
		charNames = make(map[string]string)
	}
	b.mu.RUnlock()

	botUserID := b.session.State.User.ID
	var members []web.VoiceMemberInfo
	for _, voiceState := range guild.VoiceStates {
		if voiceState.ChannelID != vs.ChannelID {
			continue
		}
		if voiceState.UserID == botUserID {
			continue
		}

		username := voiceState.UserID
		member, _ := b.session.GuildMember(b.guildID, voiceState.UserID)
		if member != nil && member.User != nil {
			username = member.User.Username
		}

		var charName *string
		if name, ok := charNames[voiceState.UserID]; ok {
			charName = &name
		}

		members = append(members, web.VoiceMemberInfo{
			ID:            voiceState.UserID,
			Username:      username,
			CharacterName: charName,
		})
	}
	return members
}

// JoinChannel joins a voice channel and starts recording.
func (b *Bot) JoinChannel(channelID string, campaignID int64) (string, error) {
	// Quick check without holding lock during the blocking voice join.
	b.voiceStateMu.Lock()
	if b.voiceState != nil {
		b.voiceStateMu.Unlock()
		return "", fmt.Errorf("already recording")
	}
	b.voiceStateMu.Unlock()

	// Resolve channel name.
	channel, err := b.session.Channel(channelID)
	if err != nil {
		return "", fmt.Errorf("channel not found: %w", err)
	}

	// Create DB session.
	ctx := context.Background()
	sessionID, err := db.CreateSession(ctx, campaignID, channel.Name, channelID)
	if err != nil {
		return "", fmt.Errorf("create session: %w", err)
	}

	// Join voice channel -- this blocks waiting for the voice connection.
	// Do NOT hold voiceStateMu during this call or it deadlocks GetStatus/GetVoiceMembers.
	vc, err := b.session.ChannelVoiceJoin(ctx, b.guildID, channelID, false, false)
	if err != nil {
		return "", fmt.Errorf("join voice: %w", err)
	}

	// Get ignored users and character names for this campaign.
	b.mu.Lock()
	ignored := copyBoolMap(b.getIgnoredSet(campaignID))
	charNames := copyStringMap(b.getCharacterNameMap(campaignID))
	b.mu.Unlock()

	// Create recorder. Only wire the per-frame tap in streaming mode so the
	// whisper path keeps its exact prior behaviour with zero per-packet overhead.
	var onFrame voice.FrameCallback
	if strings.EqualFold(b.config.Transcribe.Engine, "sherpa") {
		onFrame = b.handleAudioFrame
	}
	rec := voice.NewRecorder(sessionID, campaignID, ignored, charNames, b.handleAudioFlush, onFrame)
	if b.saveRecordings {
		rec.SetSaveDir(b.config.Recordings.Dir)
		rec.SetSaveRecordings(true)
	}

	// Set known usernames for voice members.
	guild, _ := b.session.State.Guild(b.guildID)
	if guild != nil {
		for _, vs := range guild.VoiceStates {
			if vs.ChannelID == channelID {
				if member, err := b.session.GuildMember(b.guildID, vs.UserID); err == nil && member.User != nil {
					rec.SetUserName(vs.UserID, member.User.Username)
				}
			}
		}
	}

	// Hook DAVE transition events to re-derive keys for all active streams.
	// WHY: When a user joins or leaves, Discord renegotiates DAVE E2EE keys.
	// Without re-derivation, existing streams decrypt with stale keys → garbage audio.
	vc.OnDAVETransition = func() {
		log.Println("DAVE transition detected, re-deriving all receiver keys")
		rec.ReDeriveAllDAVEKeys(vc)
	}

	rec.Start(vc)

	b.voiceStateMu.Lock()
	b.voiceState = &VoiceState{
		Connection: vc,
		Recorder:   rec,
		ChannelID:  channelID,
		CampaignID: campaignID,
		SessionID:  sessionID,
	}
	b.voiceStateMu.Unlock()

	log.Printf("Started recording in %s (session #%d, campaign %d)", channel.Name, sessionID, campaignID)

	// Tell the transcription worker about the new session.
	if b.worker != nil {
		b.worker.SetSessionID(&sessionID)
		b.worker.SetCampaignID(&campaignID)
	}

	if b.OnStatus != nil {
		b.OnStatus()
	}
	if b.OnMembers != nil {
		b.OnMembers()
	}
	if b.OnSessionStarted != nil {
		b.OnSessionStarted(sessionID, channel.Name, campaignID)
	}

	return channel.Name, nil
}

// LeaveChannel stops recording and disconnects from voice.
func (b *Bot) LeaveChannel() (*string, error) {
	// Grab state reference quickly, then unlock before blocking calls.
	// WHY: Same deadlock pattern as JoinChannel had -- holding voiceStateMu
	// during Recorder.Stop() / Disconnect() / Finalize() blocks GetStatus
	// and GetVoiceMembers which also need the lock.
	b.voiceStateMu.Lock()
	vs := b.voiceState
	if vs == nil {
		b.voiceStateMu.Unlock()
		return nil, fmt.Errorf("not in a voice channel")
	}
	b.voiceState = nil
	b.voiceStateMu.Unlock()

	// Blocking work happens without the lock held.
	vs.Recorder.Stop()
	vs.Connection.Disconnect(context.Background())

	// Finalize transcription and clear worker session state.
	if b.worker != nil {
		b.worker.Finalize()
		b.worker.SetSessionID(nil)
		b.worker.SetCampaignID(nil)
	}

	// End session in DB.
	ctx := context.Background()
	if err := db.EndSession(ctx, vs.SessionID); err != nil {
		log.Printf("EndSession error: %v", err)
	}

	if b.OnStatus != nil {
		b.OnStatus()
	}
	if b.OnMembers != nil {
		b.OnMembers()
	}

	return nil, nil
}

// SetCharacterName updates a character name in memory and DB, flushing the
// user's audio buffer first so speech gets attributed to the old name.
func (b *Bot) SetCharacterName(campaignID int64, userID string, name *string) {
	// Flush audio buffer before changing the name.
	b.voiceStateMu.Lock()
	if b.voiceState != nil && b.voiceState.CampaignID == campaignID {
		b.voiceState.Recorder.FlushUserBuffer(userID)
	}
	b.voiceStateMu.Unlock()

	b.mu.Lock()
	charNames := b.getCharacterNameMap(campaignID)
	if name != nil && *name != "" {
		charNames[userID] = *name
	} else {
		delete(charNames, userID)
	}
	b.mu.Unlock()

	// Persist to DB.
	ctx := context.Background()
	if err := db.SetCharacterName(ctx, campaignID, userID, name); err != nil {
		log.Printf("Failed to save character name: %v", err)
	}

	// Update active recorder if recording for this campaign.
	b.voiceStateMu.Lock()
	if b.voiceState != nil && b.voiceState.CampaignID == campaignID {
		if name != nil && *name != "" {
			b.voiceState.Recorder.SetCharacterName(userID, *name)
		} else {
			b.voiceState.Recorder.RemoveCharacterName(userID)
		}
	}
	b.voiceStateMu.Unlock()

	if b.OnMembers != nil {
		b.OnMembers()
	}
}

// IsRecording returns true if the bot is currently recording.
func (b *Bot) IsRecording() bool {
	b.voiceStateMu.Lock()
	defer b.voiceStateMu.Unlock()
	return b.voiceState != nil
}

// GetStatus returns the current bot status.
func (b *Bot) GetStatus() web.BotStatus {
	b.voiceStateMu.Lock()
	vs := b.voiceState
	b.voiceStateMu.Unlock()

	if vs == nil {
		return web.BotStatus{Recording: false}
	}

	var channelName *string
	ch, err := b.session.Channel(vs.ChannelID)
	if err == nil {
		channelName = &ch.Name
	}

	return web.BotStatus{
		Recording:   true,
		ChannelName: channelName,
		CampaignID:  &vs.CampaignID,
	}
}

// GetActiveCampaignID returns the campaign ID of the active recording session.
func (b *Bot) GetActiveCampaignID() *int64 {
	b.voiceStateMu.Lock()
	vs := b.voiceState
	b.voiceStateMu.Unlock()

	if vs == nil {
		return nil
	}
	return &vs.CampaignID
}

// GetSaveRecordings returns the save-recordings preference.
func (b *Bot) GetSaveRecordings() bool {
	b.voiceStateMu.Lock()
	defer b.voiceStateMu.Unlock()

	if b.voiceState != nil {
		// Not stored on recorder, use bot-level preference.
	}
	return b.saveRecordings
}

// SetSaveRecordings updates the save-recordings preference.
func (b *Bot) SetSaveRecordings(enabled bool) {
	b.saveRecordings = enabled

	b.voiceStateMu.Lock()
	if b.voiceState != nil {
		b.voiceState.Recorder.SetSaveDir(b.config.Recordings.Dir)
		b.voiceState.Recorder.SetSaveRecordings(enabled)
	}
	b.voiceStateMu.Unlock()

	if b.OnSaveRecordings != nil {
		b.OnSaveRecordings(enabled)
	}
}

// TogglePause toggles the recording pause state. Returns the new paused state.
func (b *Bot) TogglePause() bool {
	b.voiceStateMu.Lock()
	defer b.voiceStateMu.Unlock()

	if b.voiceState == nil {
		return false
	}
	newState := !b.voiceState.Recorder.IsPaused()
	b.voiceState.Recorder.SetPaused(newState)
	return newState
}

// GetIgnoredUsers returns ignored users for a campaign from DB.
func (b *Bot) GetIgnoredUsers(campaignID int64) ([]db.IgnoredUser, error) {
	return db.GetIgnoredUsers(context.Background(), campaignID)
}

// IgnoreUser adds a user to the ignore list.
func (b *Bot) IgnoreUser(campaignID int64, userID, username string) (bool, error) {
	ctx := context.Background()
	added, err := db.AddIgnoredUser(ctx, campaignID, userID, username)
	if err != nil {
		return false, err
	}
	if !added {
		return false, nil
	}

	b.mu.Lock()
	set := b.getIgnoredSet(campaignID)
	set[userID] = true
	setCopy := copyBoolMap(set)
	b.mu.Unlock()

	// Update active recorder.
	b.voiceStateMu.Lock()
	if b.voiceState != nil && b.voiceState.CampaignID == campaignID {
		b.voiceState.Recorder.UpdateIgnoredUsers(setCopy)
	}
	b.voiceStateMu.Unlock()

	if b.OnIgnoredUsers != nil {
		b.OnIgnoredUsers(campaignID)
	}
	log.Printf("Ignoring user %s (%s) in campaign %d", username, userID, campaignID)
	return true, nil
}

// UnignoreUser removes a user from the ignore list.
func (b *Bot) UnignoreUser(campaignID int64, userID string) (bool, error) {
	ctx := context.Background()
	removed, err := db.RemoveIgnoredUser(ctx, campaignID, userID)
	if err != nil {
		return false, err
	}
	if !removed {
		return false, nil
	}

	b.mu.Lock()
	set := b.getIgnoredSet(campaignID)
	delete(set, userID)
	setCopy := copyBoolMap(set)
	b.mu.Unlock()

	// Update active recorder.
	b.voiceStateMu.Lock()
	if b.voiceState != nil && b.voiceState.CampaignID == campaignID {
		b.voiceState.Recorder.UpdateIgnoredUsers(setCopy)
	}
	b.voiceStateMu.Unlock()

	if b.OnIgnoredUsers != nil {
		b.OnIgnoredUsers(campaignID)
	}
	log.Printf("Unignored user %s in campaign %d", userID, campaignID)
	return true, nil
}

// GetNicknamePresets returns nickname presets for a campaign.
func (b *Bot) GetNicknamePresets(campaignID int64, userID *string) ([]db.NicknamePreset, error) {
	return db.GetNicknamePresets(context.Background(), campaignID, userID)
}

// handleAudioFlush is called when a user's audio buffer is flushed by the recorder.
// It forwards the PCM audio to the transcription worker for speech-to-text.
func (b *Bot) handleAudioFlush(userID string, pcm []byte, timestamp time.Time, username, nickname string, rms float32, durationMs int, audioFilename string) {
	log.Printf("Audio flush: user=%s username=%s nickname=%s rms=%.1f duration=%dms bytes=%d audio=%s",
		userID, username, nickname, rms, durationMs, len(pcm), audioFilename)
	if b.worker != nil {
		b.worker.AddTask(userID, pcm, timestamp, username, nickname, rms, durationMs, audioFilename)
	}
}

// handleAudioFrame forwards each decoded voice frame to the worker. In batch
// (whisper) mode the worker's FeedFrame is a no-op; in streaming (sherpa) mode
// it feeds the recognizer for live partials. The recorder only invokes this
// when an onFrame callback is wired, so it's free in batch mode.
func (b *Bot) handleAudioFrame(userID, username, nickname string, mono48k []int16) {
	if b.worker != nil {
		b.worker.FeedFrame(userID, username, nickname, mono48k)
	}
}

// handleVoiceStateUpdate detects when the voice channel empties and auto-leaves.
// It also populates the recorder's username map for users who join while we're
// recording — without this, the recorder would create their stream via the
// speaking-update path with an empty username, and transcriptions for them
// land in the DB with DiscordUsername="".
func (b *Bot) handleVoiceStateUpdate(s *discordgo.Session, vsu *discordgo.VoiceStateUpdate) {
	// Emit members event for web UI.
	if b.OnMembers != nil && b.IsRecording() {
		b.OnMembers()
	}

	b.voiceStateMu.Lock()
	vs := b.voiceState
	b.voiceStateMu.Unlock()

	if vs == nil {
		return
	}

	// Joined (or moved into) our channel: push their username to the recorder
	// so the stream created on their first speaking update has a name.
	// WHY this matters: the /join-time loop only iterates users ALREADY in the
	// channel. Without this, late joiners had empty DiscordUsername in the DB.
	joined := vsu.ChannelID == vs.ChannelID &&
		(vsu.BeforeUpdate == nil || vsu.BeforeUpdate.ChannelID != vs.ChannelID)
	if joined && vsu.UserID != s.State.User.ID {
		if member, err := s.GuildMember(b.guildID, vsu.UserID); err == nil && member.User != nil {
			vs.Recorder.SetUserName(vsu.UserID, member.User.Username)
		}
	}

	// Only care about users leaving our channel.
	if vsu.BeforeUpdate == nil || vsu.BeforeUpdate.ChannelID != vs.ChannelID {
		return
	}
	if vsu.ChannelID == vs.ChannelID {
		return
	}

	// Check if any non-bot users remain.
	guild, err := s.State.Guild(b.guildID)
	if err != nil {
		return
	}

	botUserID := s.State.User.ID
	for _, voiceState := range guild.VoiceStates {
		if voiceState.ChannelID != vs.ChannelID {
			continue
		}
		if voiceState.UserID == botUserID {
			continue
		}
		// At least one non-bot user remains.
		return
	}

	log.Println("Voice channel emptied, auto-stopping session.")
	b.LeaveChannel()
}

// --- Helpers ---

func copyBoolMap(m map[string]bool) map[string]bool {
	c := make(map[string]bool, len(m))
	for k, v := range m {
		c[k] = v
	}
	return c
}

func copyStringMap(m map[string]string) map[string]string {
	c := make(map[string]string, len(m))
	for k, v := range m {
		c[k] = v
	}
	return c
}
