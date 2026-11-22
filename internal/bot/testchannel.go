package bot

import (
	"errors"
	"log"

	"github.com/joe/dndscribe-go/internal/web"
)

// The test channel is a pretend voice channel for trying the DJ without
// Discord: it appears in the channel list when switched on (Settings page),
// "joining" it never touches Discord, and while the bot is in it the DJ
// engine mixes as normal into an output that discards the audio. It is
// music-only; recording needs a real channel.

const (
	TestChannelID   = "test-channel"
	TestChannelName = "Test channel (no Discord)"
)

var errTestChannelRecording = errors.New("the test channel is for the DJ only; recording needs a real voice channel")

// SetTestChannelEnabled shows or hides the test channel. Hiding it while the
// bot is in it leaves the channel.
func (b *Bot) SetTestChannelEnabled(on bool) {
	b.testChannelOn.Store(on)
	if !on && b.InTestChannel() {
		b.LeaveMusic()
	}
}

func (b *Bot) TestChannelEnabled() bool { return b.testChannelOn.Load() }

// InTestChannel reports whether the bot is "in" the test channel. The DJ
// engine's output checks this every tick.
func (b *Bot) InTestChannel() bool {
	b.voiceStateMu.Lock()
	defer b.voiceStateMu.Unlock()
	return b.inTest
}

func (b *Bot) joinTestChannel() (string, error) {
	if !b.TestChannelEnabled() {
		return "", errors.New("the test channel is switched off in Settings")
	}
	b.voiceStateMu.Lock()
	if b.voiceState != nil {
		b.voiceStateMu.Unlock()
		return "", errors.New("the bot is recording in another channel")
	}
	already := b.inTest
	hasMusic := b.music != nil
	b.voiceStateMu.Unlock()
	if already {
		return TestChannelName, nil
	}
	if hasMusic {
		b.LeaveMusic()
	}
	b.voiceStateMu.Lock()
	b.inTest = true
	b.voiceStateMu.Unlock()
	log.Printf("Joined the test channel for music only")
	if b.OnStatus != nil {
		b.OnStatus()
	}
	return TestChannelName, nil
}

// leaveTestChannelLocked clears the test channel; returns whether it was set.
// Caller holds voiceStateMu.
func (b *Bot) leaveTestChannelLocked() bool {
	was := b.inTest
	b.inTest = false
	return was
}

func testChannelInfo() web.VoiceChannelInfo {
	return web.VoiceChannelInfo{ID: TestChannelID, Name: TestChannelName, Members: []string{}}
}
