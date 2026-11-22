package dj

import (
	"encoding/json"
	"testing"
)

func ackOf(t *testing.T, v any) wsAck {
	t.Helper()
	a, ok := v.(wsAck)
	if !ok {
		t.Fatalf("reply %T, want wsAck", v)
	}
	return a
}

func TestWSCommandAckAndDedupe(t *testing.T) {
	e, _, _ := newTestEngine()
	h := NewHandler(e, nil, nil, nil)

	msg := func(id, cmd, args string) []byte {
		return []byte(`{"type":"dj.cmd","id":"` + id + `","cmd":"` + cmd + `","args":` + args + `}`)
	}
	a := ackOf(t, h.HandleWS(msg("c1", "mix.crossfader", `{"value":0.25}`)))
	if !a.OK || a.ID != "c1" || a.Rev < 1 {
		t.Fatalf("ack %+v", a)
	}
	if got := e.State().Crossfader; got != 0.25 {
		t.Fatalf("crossfader %v, want 0.25", got)
	}

	// The same id again is not run again (a retry), and returns the same ack.
	e.SetCrossfader(0.9)
	again := ackOf(t, h.HandleWS(msg("c1", "mix.crossfader", `{"value":0.25}`)))
	if again != a || e.State().Crossfader != 0.9 {
		t.Fatalf("duplicate id re-ran: ack %+v fader %v", again, e.State().Crossfader)
	}

	bad := ackOf(t, h.HandleWS(msg("c2", "nope", `{}`)))
	if bad.OK || bad.Error == "" {
		t.Fatalf("unknown command acked ok: %+v", bad)
	}
	missing := ackOf(t, h.HandleWS(msg("c3", "deck.play", `{}`)))
	if missing.OK || missing.Error != "deck.play needs deck" {
		t.Fatalf("missing arg: %+v", missing)
	}
	empty := ackOf(t, h.HandleWS(msg("c4", "deck.play", `{"deck":1}`)))
	if empty.OK || empty.Error != "deck B is empty" {
		t.Fatalf("empty deck: %+v", empty)
	}
	noID := ackOf(t, h.HandleWS([]byte(`{"type":"dj.cmd","cmd":"mix.next"}`)))
	if noID.OK {
		t.Fatal("command without id accepted")
	}

	// Acks are JSON the client can read.
	b, _ := json.Marshal(a)
	var round map[string]any
	_ = json.Unmarshal(b, &round)
	if round["type"] != "dj.ack" || round["ok"] != true {
		t.Fatalf("ack json %s", b)
	}
}
