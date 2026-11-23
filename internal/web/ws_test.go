package web

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

func dialHub(t *testing.T, url string) *websocket.Conn {
	t.Helper()
	c, _, err := websocket.DefaultDialer.Dial(url, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { c.Close() })
	return c
}

// readType reads messages until one of the given type arrives.
func readType(t *testing.T, c *websocket.Conn, typ string) map[string]any {
	t.Helper()
	_ = c.SetReadDeadline(time.Now().Add(2 * time.Second))
	for {
		_, data, err := c.ReadMessage()
		if err != nil {
			t.Fatalf("waiting for %s: %v", typ, err)
		}
		var m map[string]any
		_ = json.Unmarshal(data, &m)
		if m["type"] == typ {
			return m
		}
	}
}

func TestHubRoutesRepliesToSenderAndBroadcastsPresence(t *testing.T) {
	h := NewHub()
	h.Handle("dj.", func(raw []byte) any {
		return map[string]any{"type": "dj.ack", "echo": string(raw)}
	})
	go h.Run()
	srv := httptest.NewServer(http.HandlerFunc(h.HandleWS))
	defer srv.Close()
	url := "ws" + strings.TrimPrefix(srv.URL, "http")

	a := dialHub(t, url)
	b := dialHub(t, url)

	_ = a.WriteJSON(map[string]any{"type": "hello", "device": "desktop", "page": "dj"})
	p := readType(t, b, "presence")
	clients := p["data"].(map[string]any)["clients"].([]any)
	if len(clients) != 1 || clients[0].(map[string]any)["device"] != "desktop" {
		t.Fatalf("presence %v", p)
	}

	_ = a.WriteJSON(map[string]any{"type": "dj.cmd", "id": "x"})
	ack := readType(t, a, "dj.ack")
	if !strings.Contains(ack["echo"].(string), `"id":"x"`) {
		t.Fatalf("ack %v", ack)
	}
	// b must not get a's ack: send b a command of its own and check the
	// first ack b sees is its own.
	_ = b.WriteJSON(map[string]any{"type": "dj.cmd", "id": "y"})
	if got := readType(t, b, "dj.ack"); !strings.Contains(got["echo"].(string), `"id":"y"`) {
		t.Fatalf("b received someone else's ack: %v", got)
	}

	// a leaving updates presence for b.
	a.Close()
	p = readType(t, b, "presence")
	if n := len(p["data"].(map[string]any)["clients"].([]any)); n != 0 {
		t.Fatalf("after a left, %d clients in presence", n)
	}
}

func TestCheckOrigin(t *testing.T) {
	cases := []struct {
		origin, host string
		want         bool
	}{
		{"", "x:3002", true},
		{"http://10.0.0.5:3002", "10.0.0.5:3002", true},
		{"http://joe.tail.ts.net:3002", "joe.tail.ts.net:3002", true},
		{"http://localhost:5173", "localhost:3002", true},
		{"http://evil.example", "10.0.0.5:3002", false},
		{"http://10.0.0.5:9999", "10.0.0.5:3002", false},
	}
	for _, c := range cases {
		r := httptest.NewRequest("GET", "/", nil)
		r.Host = c.host
		if c.origin != "" {
			r.Header.Set("Origin", c.origin)
		}
		if got := CheckOrigin(r); got != c.want {
			t.Errorf("origin %q host %q: %v, want %v", c.origin, c.host, got, c.want)
		}
	}
}
