package dj

import (
	"net/http"
	"time"

	"github.com/gorilla/websocket"
)

// listen streams the test channel to a browser: one binary WebSocket
// message per 20 ms Opus frame (48 kHz stereo, exactly what Discord would
// get). The page decodes them with WebCodecs and plays them through Web
// Audio.
//
// WHY a separate socket and not the hub: the hub is JSON text shared by
// every page; this is ~50 binary messages a second that only the devices
// that asked to listen want. WHY raw Opus and not an <audio> stream: an
// HTTP Ogg stream needs a muxer and browsers buffer it for seconds, which
// makes pads and scene changes feel broken; frames over a socket play
// ~100 ms behind.
//
// Frames only flow while the bot is in the test channel (the engine's
// SwitchOutput sends elsewhere otherwise), so a listener connected at
// other times just idles.
func (h *Handler) listen(w http.ResponseWriter, r *http.Request) {
	if h.Listen == nil {
		writeErr(w, http.StatusNotFound, "listening is not available")
		return
	}
	up := websocket.Upgrader{CheckOrigin: h.CheckOrigin}
	conn, err := up.Upgrade(w, r, nil)
	if err != nil {
		return // Upgrade has already replied
	}
	defer conn.Close()

	frames, stop := h.Listen()
	defer stop()

	// Read loop: handles pings/close and tells us when the page goes away.
	gone := make(chan struct{})
	go func() {
		defer close(gone)
		for {
			if _, _, err := conn.ReadMessage(); err != nil {
				return
			}
		}
	}()

	ping := time.NewTicker(20 * time.Second)
	defer ping.Stop()
	for {
		select {
		case <-gone:
			return
		case f := <-frames:
			conn.SetWriteDeadline(time.Now().Add(5 * time.Second))
			if err := conn.WriteMessage(websocket.BinaryMessage, f); err != nil {
				return
			}
		case <-ping.C:
			conn.SetWriteDeadline(time.Now().Add(5 * time.Second))
			if err := conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}
