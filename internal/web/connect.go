package web

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"os/exec"
	"strings"
	"time"

	qrcode "github.com/skip2/go-qrcode"
)

// "Connect a device": the addresses another device (the GM's tablet) can
// open this app on, each shown as a QR code. The app is LAN / Tailscale only,
// so the desktop's own URL (often localhost) is no use to a tablet; the
// server works out its real addresses instead.

type connectURL struct {
	Label string `json:"label"`
	URL   string `json:"url"`
}

var tailnet = mustCIDR("100.64.0.0/10") // Tailscale's CGNAT range

func mustCIDR(s string) *net.IPNet {
	_, n, err := net.ParseCIDR(s)
	if err != nil {
		panic(err)
	}
	return n
}

// connectURLs lists Tailscale first (works on and off the home network),
// then the home-network address. IPv4 only: easier to read and type.
func connectURLs(ctx context.Context, port int) []connectURL {
	var out []connectURL
	if name := tailscaleDNSName(ctx); name != "" {
		out = append(out, connectURL{"Tailscale", fmt.Sprintf("http://%s:%d", name, port)})
	}

	var tsIP, lanIP net.IP
	ifaces, _ := net.Interfaces()
	for _, ifc := range ifaces {
		if ifc.Flags&net.FlagUp == 0 {
			continue
		}
		as, _ := ifc.Addrs()
		for _, a := range as {
			if ipn, ok := a.(*net.IPNet); ok && ipn.IP.To4() != nil && tailnet.Contains(ipn.IP) {
				tsIP = ipn.IP.To4()
			}
		}
	}
	if tsIP != nil {
		out = append(out, connectURL{"Tailscale IP", fmt.Sprintf("http://%s:%d", tsIP, port)})
	}
	// WHY a UDP "dial" rather than listing interfaces: a dev machine has
	// VirtualBox, WSL and Hyper-V adapters with private addresses no tablet
	// can reach. Connecting a UDP socket sends nothing but makes the OS pick
	// the interface its default route uses, which is the real LAN address.
	if c, err := net.Dial("udp4", "192.0.2.1:9"); err == nil {
		if ua, ok := c.LocalAddr().(*net.UDPAddr); ok && ua.IP.IsPrivate() && !tailnet.Contains(ua.IP) {
			lanIP = ua.IP
		}
		c.Close()
	}
	if lanIP != nil {
		out = append(out, connectURL{"Home network", fmt.Sprintf("http://%s:%d", lanIP, port)})
	}
	return out
}

// tailscaleDNSName asks the local tailscale CLI for this machine's MagicDNS
// name. Empty if tailscale isn't installed, isn't running, or is slow.
func tailscaleDNSName(ctx context.Context) string {
	ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "tailscale", "status", "--json")
	hideWindow(cmd)
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	var st struct {
		BackendState string
		Self         struct{ DNSName string }
	}
	if json.Unmarshal(out, &st) != nil || st.BackendState != "Running" {
		return ""
	}
	return strings.TrimSuffix(st.Self.DNSName, ".")
}

func registerConnectRoutes(mux *http.ServeMux, port int) {
	mux.HandleFunc("GET /api/connect", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"urls": connectURLs(r.Context(), port)})
	})
	// The QR image for one of the listed URLs. Only listed URLs are encoded,
	// so this isn't a general-purpose QR service.
	mux.HandleFunc("GET /api/connect/qr", func(w http.ResponseWriter, r *http.Request) {
		want := r.URL.Query().Get("url")
		for _, u := range connectURLs(r.Context(), port) {
			if u.URL != want {
				continue
			}
			png, err := qrcode.Encode(u.URL, qrcode.Medium, 512)
			if err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			w.Header().Set("Content-Type", "image/png")
			w.Header().Set("Cache-Control", "private, max-age=300")
			_, _ = w.Write(png)
			return
		}
		http.Error(w, "unknown url", http.StatusNotFound)
	})
}
