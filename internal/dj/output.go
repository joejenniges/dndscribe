package dj

import (
	"sync"
	"time"

	"github.com/bwmarrin/discordgo"
)

// Output is where encoded opus frames go. VoiceOutput is the real one; tests
// use a fake.
type Output interface {
	// Ready reports whether a voice connection is up and can take frames.
	Ready() bool
	// Queued is how many frames are waiting to be sent.
	Queued() int
	// Send offers one opus frame without blocking. false = not accepted.
	Send(opus []byte) bool
	// SetSpeaking toggles the speaking indicator.
	SetSpeaking(on bool)
}

// VoiceOutput sends into the bot's current voice connection.
//
// WHY Conn is a func and not a *VoiceConnection: the connection the bot records
// on comes and goes (join/leave, reconnects), and the library closes OpusSend
// when a connection dies. Looking it up on every tick means the mixer follows
// whatever connection is live and never holds a dead channel.
type VoiceOutput struct {
	Conn func() *discordgo.VoiceConnection
}

func (o *VoiceOutput) conn() (*discordgo.VoiceConnection, chan []byte) {
	if o.Conn == nil {
		return nil, nil
	}
	vc := o.Conn()
	if vc == nil || vc.Cond == nil {
		return nil, nil
	}
	vc.Cond.L.Lock()
	defer vc.Cond.L.Unlock()
	if vc.Status != discordgo.VoiceConnectionStatusReady || vc.OpusSend == nil {
		return nil, nil
	}
	return vc, vc.OpusSend
}

func (o *VoiceOutput) Ready() bool {
	vc, _ := o.conn()
	return vc != nil
}

func (o *VoiceOutput) Queued() int {
	_, ch := o.conn()
	if ch == nil {
		return 0
	}
	return len(ch)
}

func (o *VoiceOutput) Send(frame []byte) (ok bool) {
	_, ch := o.conn()
	if ch == nil {
		return false
	}
	// WHY recover: the library closes OpusSend ~100ms after a connection dies
	// (VoiceConnection.Kill). Between our Ready check and this send the channel
	// can close, and a send on a closed channel panics.
	defer func() {
		if recover() != nil {
			ok = false
		}
	}()
	select {
	case ch <- frame:
		return true
	default:
		return false
	}
}

func (o *VoiceOutput) SetSpeaking(on bool) {
	vc, _ := o.conn()
	if vc == nil {
		return
	}
	_ = vc.Speaking(on)
}

// NullOutput stands in for a voice connection when the bot is in the test
// channel: it accepts frames and drains them at the real 20ms rate, so the
// mixer runs at normal speed (positions, fades and Auto DJ timing behave as
// in Discord) but the audio goes nowhere.
type NullOutput struct {
	mu     sync.Mutex
	queued int
	last   time.Time
}

const nullOutputCap = 16 // same depth as discordgo's OpusSend

func (o *NullOutput) drainLocked() {
	now := time.Now()
	if o.last.IsZero() {
		o.last = now
		return
	}
	n := int(now.Sub(o.last) / (FrameMs * time.Millisecond))
	if n > 0 {
		o.queued = max(0, o.queued-n)
		o.last = o.last.Add(time.Duration(n) * FrameMs * time.Millisecond)
	}
}

func (o *NullOutput) Ready() bool { return true }

func (o *NullOutput) Queued() int {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.drainLocked()
	return o.queued
}

func (o *NullOutput) Send([]byte) bool {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.drainLocked()
	if o.queued >= nullOutputCap {
		return false
	}
	o.queued++
	return true
}

func (o *NullOutput) SetSpeaking(bool) {}

// SwitchOutput sends to Test while UseTest reports true, otherwise to Real.
// Checked on every call, like VoiceOutput's connection lookup, so joining or
// leaving the test channel takes effect on the next tick.
type SwitchOutput struct {
	Real    Output
	Test    Output
	UseTest func() bool
}

func (o *SwitchOutput) pick() Output {
	if o.UseTest != nil && o.UseTest() {
		return o.Test
	}
	return o.Real
}

func (o *SwitchOutput) Ready() bool         { return o.pick().Ready() }
func (o *SwitchOutput) Queued() int         { return o.pick().Queued() }
func (o *SwitchOutput) Send(f []byte) bool  { return o.pick().Send(f) }
func (o *SwitchOutput) SetSpeaking(on bool) { o.pick().SetSpeaking(on) }
