package bot

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/joe/dndscribe-go/internal/db"

	"github.com/bwmarrin/discordgo"
)

// Slash command definitions matching the TypeScript bot.
var commands = []*discordgo.ApplicationCommand{
	{
		Name:        "transcribe",
		Description: "Start recording and transcribing the voice channel",
	},
	{
		Name:        "leave",
		Description: "Leave the voice channel and stop recording",
	},
	{
		Name:        "ignore",
		Description: "Ignore a user's audio",
		Options: []*discordgo.ApplicationCommandOption{
			{
				Type:        discordgo.ApplicationCommandOptionUser,
				Name:        "user",
				Description: "The user to ignore",
				Required:    true,
			},
		},
	},
	{
		Name:        "unignore",
		Description: "Stop ignoring a user's audio",
		Options: []*discordgo.ApplicationCommandOption{
			{
				Type:        discordgo.ApplicationCommandOptionUser,
				Name:        "user",
				Description: "The user to stop ignoring",
				Required:    true,
			},
		},
	},
	{
		Name:        "list_ignored",
		Description: "List all ignored users",
	},
	{
		Name:        "sync",
		Description: "Re-sync slash commands",
	},
	{
		Name:        "save_recordings",
		Description: "Toggle saving raw audio recordings to disk",
	},
}

// registerCommands registers all slash commands with the guild.
func (b *Bot) registerCommands() error {
	b.registeredCmds = make([]*discordgo.ApplicationCommand, 0, len(commands))
	for _, cmd := range commands {
		registered, err := b.session.ApplicationCommandCreate(b.session.State.User.ID, b.guildID, cmd)
		if err != nil {
			return fmt.Errorf("register command %s: %w", cmd.Name, err)
		}
		b.registeredCmds = append(b.registeredCmds, registered)
	}
	log.Printf("Registered %d slash commands", len(b.registeredCmds))
	return nil
}

// handleInteraction is the top-level interaction dispatcher.
func (b *Bot) handleInteraction(s *discordgo.Session, i *discordgo.InteractionCreate) {
	if i.Type != discordgo.InteractionApplicationCommand {
		return
	}

	data := i.ApplicationCommandData()

	switch data.Name {
	case "transcribe":
		b.handleTranscribe(s, i)
	case "leave":
		b.handleLeave(s, i)
	case "ignore":
		b.handleIgnoreCmd(s, i)
	case "unignore":
		b.handleUnignoreCmd(s, i)
	case "list_ignored":
		b.handleListIgnored(s, i)
	case "sync":
		b.handleSync(s, i)
	case "save_recordings":
		b.handleSaveRecordings(s, i)
	}
}

// --- Command handlers ---

func (b *Bot) handleTranscribe(s *discordgo.Session, i *discordgo.InteractionCreate) {
	guildID := i.GuildID
	if guildID == "" {
		respondEphemeral(s, i, "This command must be used in a server.")
		return
	}

	guild, err := s.State.Guild(guildID)
	if err != nil {
		respondEphemeral(s, i, "Could not find guild.")
		return
	}

	// Find the invoking user's voice channel.
	userID := interactionUserID(i)
	var voiceChannelID string
	for _, vs := range guild.VoiceStates {
		if vs.UserID == userID {
			voiceChannelID = vs.ChannelID
			break
		}
	}

	if voiceChannelID == "" {
		respondEphemeral(s, i, "You are not in a voice channel.")
		return
	}

	if b.IsRecording() {
		respondEphemeral(s, i, "Already recording in this server.")
		return
	}

	// Default to campaign 1 for slash commands.
	channelName, err := b.JoinChannel(voiceChannelID, 1)
	if err != nil {
		respondEphemeral(s, i, fmt.Sprintf("Failed to join: %v", err))
		return
	}

	respond(s, i, fmt.Sprintf("Recording in %s...", channelName))
}

func (b *Bot) handleLeave(s *discordgo.Session, i *discordgo.InteractionCreate) {
	if !b.IsRecording() {
		respondEphemeral(s, i, "Not in a voice channel.")
		return
	}

	_, err := b.LeaveChannel()
	if err != nil {
		respondEphemeral(s, i, fmt.Sprintf("Error: %v", err))
		return
	}

	respond(s, i, "Left voice channel.")
}

func (b *Bot) handleIgnoreCmd(s *discordgo.Session, i *discordgo.InteractionCreate) {
	data := i.ApplicationCommandData()
	if len(data.Options) == 0 {
		respondEphemeral(s, i, "No user specified.")
		return
	}

	targetID := fmt.Sprintf("%v", data.Options[0].Value)
	displayName := targetID
	if data.Resolved != nil {
		if targetUser, ok := data.Resolved.Users[targetID]; ok && targetUser != nil {
			displayName = targetUser.Username
		}
	}

	// Use active campaign or default to 1.
	campaignID := int64(1)
	if cid := b.GetActiveCampaignID(); cid != nil {
		campaignID = *cid
	}

	b.mu.RLock()
	alreadyIgnored := false
	if set, ok := b.ignoredUsers[campaignID]; ok {
		alreadyIgnored = set[targetID]
	}
	b.mu.RUnlock()

	if alreadyIgnored {
		respondEphemeral(s, i, fmt.Sprintf("Already ignoring %s.", displayName))
		return
	}

	added, err := b.IgnoreUser(campaignID, targetID, displayName)
	if err != nil || !added {
		respondEphemeral(s, i, fmt.Sprintf("Failed to ignore %s.", displayName))
		return
	}

	respondEphemeral(s, i, fmt.Sprintf("Now ignoring %s.", displayName))
}

func (b *Bot) handleUnignoreCmd(s *discordgo.Session, i *discordgo.InteractionCreate) {
	data := i.ApplicationCommandData()
	if len(data.Options) == 0 {
		respondEphemeral(s, i, "No user specified.")
		return
	}

	targetID := fmt.Sprintf("%v", data.Options[0].Value)
	displayName := targetID
	if data.Resolved != nil {
		if targetUser, ok := data.Resolved.Users[targetID]; ok && targetUser != nil {
			displayName = targetUser.Username
		}
	}

	campaignID := int64(1)
	if cid := b.GetActiveCampaignID(); cid != nil {
		campaignID = *cid
	}

	b.mu.RLock()
	isIgnored := false
	if set, ok := b.ignoredUsers[campaignID]; ok {
		isIgnored = set[targetID]
	}
	b.mu.RUnlock()

	if !isIgnored {
		respondEphemeral(s, i, fmt.Sprintf("Wasn't ignoring %s.", displayName))
		return
	}

	removed, err := b.UnignoreUser(campaignID, targetID)
	if err != nil || !removed {
		respondEphemeral(s, i, fmt.Sprintf("Failed to unignore %s.", displayName))
		return
	}

	respondEphemeral(s, i, fmt.Sprintf("No longer ignoring %s.", displayName))
}

func (b *Bot) handleListIgnored(s *discordgo.Session, i *discordgo.InteractionCreate) {
	campaignID := int64(1)
	if cid := b.GetActiveCampaignID(); cid != nil {
		campaignID = *cid
	}

	ctx := context.Background()
	users, err := db.GetIgnoredUsers(ctx, campaignID)
	if err != nil {
		respondEphemeral(s, i, "Failed to get ignored users.")
		return
	}

	if len(users) == 0 {
		respondEphemeral(s, i, "No users are being ignored.")
		return
	}

	var sb strings.Builder
	sb.WriteString("Currently ignored users:\n")
	for _, u := range users {
		sb.WriteString(fmt.Sprintf("- %s (%s)\n", u.DiscordUsername, u.DiscordUserID))
	}

	respondEphemeral(s, i, sb.String())
}

func (b *Bot) handleSync(s *discordgo.Session, i *discordgo.InteractionCreate) {
	if err := b.registerCommands(); err != nil {
		respondEphemeral(s, i, fmt.Sprintf("Failed to sync commands: %v", err))
		return
	}
	respondEphemeral(s, i, fmt.Sprintf("Synced %d commands.", len(b.registeredCmds)))
}

func (b *Bot) handleSaveRecordings(s *discordgo.Session, i *discordgo.InteractionCreate) {
	if !b.IsRecording() {
		respondEphemeral(s, i, "Not currently recording in this server.")
		return
	}

	current := b.GetSaveRecordings()
	b.SetSaveRecordings(!current)

	status := "enabled"
	if current {
		status = "disabled"
	}
	respondEphemeral(s, i, fmt.Sprintf("Recording save %s.", status))
}

// --- Helpers ---

func respond(s *discordgo.Session, i *discordgo.InteractionCreate, content string) {
	s.InteractionRespond(i.Interaction, &discordgo.InteractionResponse{
		Type: discordgo.InteractionResponseChannelMessageWithSource,
		Data: &discordgo.InteractionResponseData{Content: content},
	})
}

func respondEphemeral(s *discordgo.Session, i *discordgo.InteractionCreate, content string) {
	s.InteractionRespond(i.Interaction, &discordgo.InteractionResponse{
		Type: discordgo.InteractionResponseChannelMessageWithSource,
		Data: &discordgo.InteractionResponseData{
			Content: content,
			Flags:   discordgo.MessageFlagsEphemeral,
		},
	})
}

func interactionUserID(i *discordgo.InteractionCreate) string {
	if i.Member != nil {
		return i.Member.User.ID
	}
	return i.User.ID
}
