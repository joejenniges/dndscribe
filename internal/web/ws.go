package web

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"net/url"
	"strings"
	"sync"

	"github.com/gorilla/websocket"
	"github.com/joe/dndscribe-go/internal/db"
)

var upgrader = websocket.Upgrader{CheckOrigin: checkOrigin}

// checkOrigin accepts the app's own pages (whatever host they were opened
// on: localhost, a LAN IP, the Tailscale name) and the Vite dev server on
// localhost. It refuses other websites, which could otherwise drive the DJ
// and read transcripts from a browser on the same network. There is no auth;
// this is the one cheap guard against cross-site WebSocket use.
func checkOrigin(r *http.Request) bool {
	origin := r.Header.Get("Origin")
	if origin == "" {
		return true // not a browser
	}
	u, err := url.Parse(origin)
	if err != nil {
		return false
	}
	if strings.EqualFold(u.Host, r.Host) {
		return true
	}
	switch u.Hostname() {
	case "localhost", "127.0.0.1", "::1":
		return true
	}
	log.Printf("WebSocket: refused origin %q (host %q)", origin, r.Host)
	return false
}

// wsClient represents a single WebSocket connection.
type wsClient struct {
	conn       *websocket.Conn
	send       chan []byte
	campaignID *int64
}

// Hub maintains the set of active WebSocket clients and broadcasts messages.
type Hub struct {
	mu         sync.RWMutex
	clients    map[*wsClient]struct{}
	register   chan *wsClient
	unregister chan *wsClient
	broadcast  chan broadcastMsg
	initFunc   InitMessageFunc
}

type broadcastMsg struct {
	data       []byte
	campaignID *int64
}

// InitMessageFunc returns the JSON bytes to send to a new WS client on connect.
type InitMessageFunc func() []byte

// NewHub creates a new Hub.
func NewHub() *Hub {
	return &Hub{
		clients:    make(map[*wsClient]struct{}),
		register:   make(chan *wsClient),
		unregister: make(chan *wsClient),
		broadcast:  make(chan broadcastMsg, 256),
	}
}

// SetInitFunc sets the function that generates the init message for new clients.
func (h *Hub) SetInitFunc(fn InitMessageFunc) {
	h.initFunc = fn
}

// Run starts the hub's main loop. Call this in a goroutine.
func (h *Hub) Run() {
	for {
		select {
		case client := <-h.register:
			h.mu.Lock()
			h.clients[client] = struct{}{}
			h.mu.Unlock()

		case client := <-h.unregister:
			h.mu.Lock()
			if _, ok := h.clients[client]; ok {
				delete(h.clients, client)
				close(client.send)
			}
			h.mu.Unlock()

		case msg := <-h.broadcast:
			h.mu.RLock()
			for client := range h.clients {
				// If message is campaign-scoped, only send to clients subscribed to that campaign
				if msg.campaignID != nil && client.campaignID != nil && *client.campaignID != *msg.campaignID {
					continue
				}
				select {
				case client.send <- msg.data:
				default:
					// Client buffer full, drop them
					go func(c *wsClient) {
						h.unregister <- c
					}(client)
				}
			}
			h.mu.RUnlock()
		}
	}
}

// HandleWS upgrades an HTTP connection to WebSocket and registers the client.
func (h *Hub) HandleWS(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("WebSocket upgrade error: %v", err)
		return
	}

	client := &wsClient{
		conn: conn,
		send: make(chan []byte, 256),
	}

	h.register <- client

	// Send init message with current state.
	if h.initFunc != nil {
		if data := h.initFunc(); data != nil {
			select {
			case client.send <- data:
			default:
			}
		}
	}

	go h.writePump(client)
	go h.readPump(client)
}

// readPump reads messages from the WebSocket client. Handles set_campaign subscriptions.
func (h *Hub) readPump(client *wsClient) {
	defer func() {
		h.unregister <- client
		client.conn.Close()
	}()

	for {
		_, message, err := client.conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseNormalClosure) {
				log.Printf("WebSocket read error: %v", err)
			}
			break
		}

		var msg struct {
			Type       string `json:"type"`
			CampaignID *int64 `json:"campaignId"`
		}
		if err := json.Unmarshal(message, &msg); err != nil {
			continue
		}

		switch msg.Type {
		case "set_campaign":
			h.mu.Lock()
			client.campaignID = msg.CampaignID
			h.mu.Unlock()
		}
	}
}

// writePump writes messages to the WebSocket client from its send channel.
func (h *Hub) writePump(client *wsClient) {
	defer client.conn.Close()

	for message := range client.send {
		if err := client.conn.WriteMessage(websocket.TextMessage, message); err != nil {
			break
		}
	}
}

// Broadcast sends a JSON message to all connected clients, optionally filtered by campaign ID.
func (h *Hub) Broadcast(msg interface{}, campaignID *int64) {
	data, err := json.Marshal(msg)
	if err != nil {
		log.Printf("WebSocket broadcast marshal error: %v", err)
		return
	}

	h.broadcast <- broadcastMsg{
		data:       data,
		campaignID: campaignID,
	}
}

// BroadcastPresets fetches presets and categories for a campaign and broadcasts them.
func (h *Hub) BroadcastPresets(ctx context.Context, campaignID int64) {
	presets, err := db.GetNicknamePresets(ctx, campaignID, nil)
	if err != nil {
		log.Printf("BroadcastPresets: get presets: %v", err)
		return
	}
	categories, err := db.GetNicknameCategories(ctx, campaignID, nil)
	if err != nil {
		log.Printf("BroadcastPresets: get categories: %v", err)
		return
	}

	// Ensure non-nil slices for JSON
	if presets == nil {
		presets = []db.NicknamePreset{}
	}
	if categories == nil {
		categories = []db.NicknameCategory{}
	}

	h.Broadcast(map[string]interface{}{
		"type":       "presets_updated",
		"presets":    presets,
		"categories": categories,
	}, &campaignID)
}
