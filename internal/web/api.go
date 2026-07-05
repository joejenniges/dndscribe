package web

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"
	"unicode"

	"github.com/joe/dndscribe-go/internal/db"
)

// --- Interfaces ---

// BotController is implemented by the bot layer. The web server depends only on
// this interface, so the bot package can be developed independently.
type BotController interface {
	GetVoiceChannels() []VoiceChannelInfo
	GetVoiceMembers() []VoiceMemberInfo
	JoinChannel(channelID string, campaignID int64) (string, error)
	LeaveChannel() (*string, error)
	SetCharacterName(campaignID int64, userID string, name *string)
	IsRecording() bool
	GetStatus() BotStatus
	GetSaveRecordings() bool
	SetSaveRecordings(enabled bool)
	TogglePause() bool
	GetIgnoredUsers(campaignID int64) ([]db.IgnoredUser, error)
	IgnoreUser(campaignID int64, userID, username string) (bool, error)
	UnignoreUser(campaignID int64, userID string) (bool, error)
	GetNicknamePresets(campaignID int64, userID *string) ([]db.NicknamePreset, error)
	GetActiveCampaignID() *int64
}

// TranscriptionWorker manages live transcription state. The web layer uses it
// to read/update/delete lines in the active session.
type TranscriptionWorker interface {
	GetLines(sessionID *int64) ([]TranscriptionLine, error)
	UpdateLine(id int64, text string) ([]string, error) // returns auto-added hotwords
	DeleteLine(id int64) (bool, error)
	GetSessionID() *int64
	GetCampaignID() *int64
	SetCampaignID(id *int64)
}

// --- Types ---

type BotStatus struct {
	Recording   bool    `json:"recording"`
	ChannelName *string `json:"channelName"`
	CampaignID  *int64  `json:"campaignId"`
}

type VoiceChannelInfo struct {
	ID      string   `json:"id"`
	Name    string   `json:"name"`
	Members []string `json:"members"`
}

type VoiceMemberInfo struct {
	ID            string  `json:"id"`
	Username      string  `json:"username"`
	CharacterName *string `json:"characterName"`
}

// TranscriptionLine matches the frontend's expected JSON format exactly.
// Field names must match what the SvelteKit frontend parses.
type TranscriptionLine struct {
	ID             int64    `json:"id"`
	Timestamp      string   `json:"timestamp"`
	DiscordUsername string   `json:"discordUsername"`
	Nickname       *string  `json:"nickname"`
	Text           string   `json:"text"`
	AudioFilenames []string `json:"audioFilenames"`
	RMS            *float32 `json:"rms,omitempty"`
	Confidence     *float32 `json:"confidence,omitempty"`
	DurationMs     *int     `json:"durationMs,omitempty"`
}

// --- Helpers ---

func jsonResponse(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

func jsonError(w http.ResponseWriter, status int, msg string) {
	jsonResponse(w, status, map[string]string{"error": msg})
}

func readJSON(r *http.Request, v interface{}) error {
	return json.NewDecoder(r.Body).Decode(v)
}

func parseID(s string) (int64, error) {
	return strconv.ParseInt(s, 10, 64)
}

// --- Route patterns ---

var (
	reCampaigns          = regexp.MustCompile(`^/api/campaigns$`)
	reCampaignByID       = regexp.MustCompile(`^/api/campaigns/(\d+)$`)
	reSessions           = regexp.MustCompile(`^/api/campaigns/(\d+)/sessions$`)
	reSessionByID        = regexp.MustCompile(`^/api/campaigns/(\d+)/sessions/(\d+)$`)
	reSessionExport      = regexp.MustCompile(`^/api/campaigns/(\d+)/sessions/(\d+)/export$`)
	reSessionPreview     = regexp.MustCompile(`^/api/campaigns/(\d+)/sessions/(\d+)/preview$`)
	reSessionsMerge      = regexp.MustCompile(`^/api/campaigns/(\d+)/sessions/merge$`)
	reLines              = regexp.MustCompile(`^/api/campaigns/(\d+)/lines$`)
	reLineByID           = regexp.MustCompile(`^/api/campaigns/(\d+)/lines/(\d+)$`)
	reSearch             = regexp.MustCompile(`^/api/campaigns/(\d+)/search$`)
	reHotwords           = regexp.MustCompile(`^/api/campaigns/(\d+)/hotwords$`)
	reMembers            = regexp.MustCompile(`^/api/campaigns/(\d+)/members$`)
	reCharacterName      = regexp.MustCompile(`^/api/campaigns/(\d+)/character-name$`)
	reBulkNickname       = regexp.MustCompile(`^/api/campaigns/(\d+)/bulk-nickname$`)
	rePresets            = regexp.MustCompile(`^/api/campaigns/(\d+)/presets$`)
	rePresetByID         = regexp.MustCompile(`^/api/campaigns/(\d+)/presets/(\d+)$`)
	rePresetsMove        = regexp.MustCompile(`^/api/campaigns/(\d+)/presets/move$`)
	reCategories         = regexp.MustCompile(`^/api/campaigns/(\d+)/categories$`)
	reCategoryByID       = regexp.MustCompile(`^/api/campaigns/(\d+)/categories/(\d+)$`)
	reIgnoredUsers       = regexp.MustCompile(`^/api/campaigns/(\d+)/ignored-users$`)
	reStatus             = regexp.MustCompile(`^/api/status$`)
	reChannels           = regexp.MustCompile(`^/api/channels$`)
	reJoin               = regexp.MustCompile(`^/api/join$`)
	reLeave              = regexp.MustCompile(`^/api/leave$`)
	rePause              = regexp.MustCompile(`^/api/pause$`)
	reSaveRecordings     = regexp.MustCompile(`^/api/save-recordings$`)
)

// --- API Handler ---

// APIHandler holds references to the bot controller, websocket hub, and
// optionally a transcription worker. The transcription worker is set later
// once the bot is fully initialized.
type APIHandler struct {
	ctrl         BotController
	hub          *Hub
	recordingDir string
	tw           TranscriptionWorker
}

// SetTranscriptionWorker sets the transcription worker after initialization.
func (a *APIHandler) SetTranscriptionWorker(tw TranscriptionWorker) {
	a.tw = tw
}

func (a *APIHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	path := r.URL.Path
	method := r.Method
	ctx := r.Context()

	// --- Global endpoints ---

	if reStatus.MatchString(path) && method == http.MethodGet {
		a.getStatus(w, r)
		return
	}
	if reChannels.MatchString(path) && method == http.MethodGet {
		a.getChannels(w, r)
		return
	}
	if reJoin.MatchString(path) && method == http.MethodPost {
		a.postJoin(w, r)
		return
	}
	if reLeave.MatchString(path) && method == http.MethodPost {
		a.postLeave(w, r)
		return
	}
	if rePause.MatchString(path) && method == http.MethodPost {
		a.postPause(w, r)
		return
	}
	if reSaveRecordings.MatchString(path) {
		switch method {
		case http.MethodGet:
			a.getSaveRecordings(w, r)
		case http.MethodPost:
			a.postSaveRecordings(w, r)
		default:
			w.WriteHeader(http.StatusMethodNotAllowed)
		}
		return
	}

	// --- Campaign CRUD ---

	if m := reCampaigns.FindStringSubmatch(path); m != nil {
		switch method {
		case http.MethodGet:
			a.listCampaigns(w, r, ctx)
		case http.MethodPost:
			a.createCampaign(w, r, ctx)
		default:
			w.WriteHeader(http.StatusMethodNotAllowed)
		}
		return
	}

	// Campaign-scoped routes: check for sub-resources first (longer paths match first)

	if m := reSessionExport.FindStringSubmatch(path); m != nil && method == http.MethodGet {
		sid, _ := parseID(m[2])
		a.exportSession(w, r, ctx, sid)
		return
	}

	if m := reSessionPreview.FindStringSubmatch(path); m != nil && method == http.MethodGet {
		cid, _ := parseID(m[1])
		sid, _ := parseID(m[2])
		_ = cid
		a.getSessionPreview(w, r, ctx, sid)
		return
	}

	if m := reSessionsMerge.FindStringSubmatch(path); m != nil && method == http.MethodPost {
		cid, _ := parseID(m[1])
		a.mergeSessions(w, r, ctx, cid)
		return
	}

	if m := reSessionByID.FindStringSubmatch(path); m != nil {
		cid, _ := parseID(m[1])
		sid, _ := parseID(m[2])
		switch method {
		case http.MethodPut:
			a.updateSession(w, r, ctx, sid)
		case http.MethodDelete:
			a.deleteSession(w, r, ctx, cid, sid)
		default:
			w.WriteHeader(http.StatusMethodNotAllowed)
		}
		return
	}

	if m := reSessions.FindStringSubmatch(path); m != nil && method == http.MethodGet {
		cid, _ := parseID(m[1])
		a.listSessions(w, r, ctx, cid)
		return
	}

	if m := reLineByID.FindStringSubmatch(path); m != nil {
		lid, _ := parseID(m[2])
		switch method {
		case http.MethodPost:
			a.updateLine(w, r, ctx, lid)
		case http.MethodDelete:
			a.deleteLine(w, r, ctx, lid)
		default:
			w.WriteHeader(http.StatusMethodNotAllowed)
		}
		return
	}

	if m := reLines.FindStringSubmatch(path); m != nil && method == http.MethodGet {
		a.getLines(w, r, ctx)
		return
	}

	if m := reSearch.FindStringSubmatch(path); m != nil && method == http.MethodGet {
		cid, _ := parseID(m[1])
		a.searchTranscriptions(w, r, ctx, cid)
		return
	}

	if m := reHotwords.FindStringSubmatch(path); m != nil {
		cid, _ := parseID(m[1])
		switch method {
		case http.MethodGet:
			a.getHotwords(w, r, ctx, cid)
		case http.MethodPost:
			a.addHotword(w, r, ctx, cid)
		case http.MethodDelete:
			a.removeHotword(w, r, ctx, cid)
		case http.MethodPut:
			a.updateHotword(w, r, ctx, cid)
		default:
			w.WriteHeader(http.StatusMethodNotAllowed)
		}
		return
	}

	if m := reMembers.FindStringSubmatch(path); m != nil && method == http.MethodGet {
		a.getMembers(w, r)
		return
	}

	if m := reCharacterName.FindStringSubmatch(path); m != nil && method == http.MethodPost {
		cid, _ := parseID(m[1])
		a.setCharacterName(w, r, cid)
		return
	}

	if m := reBulkNickname.FindStringSubmatch(path); m != nil && method == http.MethodPost {
		cid, _ := parseID(m[1])
		a.bulkNickname(w, r, ctx, cid)
		return
	}

	if m := rePresetsMove.FindStringSubmatch(path); m != nil && method == http.MethodPost {
		cid, _ := parseID(m[1])
		a.movePreset(w, r, ctx, cid)
		return
	}

	if m := rePresetByID.FindStringSubmatch(path); m != nil {
		cid, _ := parseID(m[1])
		pid, _ := parseID(m[2])
		switch method {
		case http.MethodPut:
			a.updatePreset(w, r, ctx, cid, pid)
		case http.MethodDelete:
			a.deletePreset(w, r, ctx, cid, pid)
		default:
			w.WriteHeader(http.StatusMethodNotAllowed)
		}
		return
	}

	if m := rePresets.FindStringSubmatch(path); m != nil {
		cid, _ := parseID(m[1])
		switch method {
		case http.MethodGet:
			a.getPresets(w, r, ctx, cid)
		case http.MethodPost:
			a.addPreset(w, r, ctx, cid)
		default:
			w.WriteHeader(http.StatusMethodNotAllowed)
		}
		return
	}

	if m := reCategoryByID.FindStringSubmatch(path); m != nil {
		cid, _ := parseID(m[1])
		catID, _ := parseID(m[2])
		switch method {
		case http.MethodPut:
			a.updateCategory(w, r, ctx, cid, catID)
		case http.MethodDelete:
			a.deleteCategory(w, r, ctx, cid, catID)
		default:
			w.WriteHeader(http.StatusMethodNotAllowed)
		}
		return
	}

	if m := reCategories.FindStringSubmatch(path); m != nil {
		cid, _ := parseID(m[1])
		switch method {
		case http.MethodGet:
			a.getCategories(w, r, ctx, cid)
		case http.MethodPost:
			a.addCategory(w, r, ctx, cid)
		default:
			w.WriteHeader(http.StatusMethodNotAllowed)
		}
		return
	}

	if m := reIgnoredUsers.FindStringSubmatch(path); m != nil {
		cid, _ := parseID(m[1])
		switch method {
		case http.MethodGet:
			a.getIgnoredUsers(w, r, cid)
		case http.MethodPost:
			a.ignoreUser(w, r, cid)
		case http.MethodDelete:
			a.unignoreUser(w, r, cid)
		default:
			w.WriteHeader(http.StatusMethodNotAllowed)
		}
		return
	}

	if m := reCampaignByID.FindStringSubmatch(path); m != nil {
		cid, _ := parseID(m[1])
		switch method {
		case http.MethodGet:
			a.getCampaign(w, r, ctx, cid)
		case http.MethodPut:
			a.updateCampaign(w, r, ctx, cid)
		case http.MethodDelete:
			a.deleteCampaign(w, r, ctx, cid)
		default:
			w.WriteHeader(http.StatusMethodNotAllowed)
		}
		return
	}

	http.NotFound(w, r)
}

// ==================== Global endpoints ====================

func (a *APIHandler) getStatus(w http.ResponseWriter, _ *http.Request) {
	jsonResponse(w, http.StatusOK, a.ctrl.GetStatus())
}

func (a *APIHandler) getChannels(w http.ResponseWriter, _ *http.Request) {
	channels := a.ctrl.GetVoiceChannels()
	if channels == nil {
		channels = []VoiceChannelInfo{}
	}
	jsonResponse(w, http.StatusOK, channels)
}

func (a *APIHandler) postJoin(w http.ResponseWriter, r *http.Request) {
	var body struct {
		ChannelID  string `json:"channelId"`
		CampaignID int64  `json:"campaignId"`
	}
	if err := readJSON(r, &body); err != nil {
		jsonError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	channelName, err := a.ctrl.JoinChannel(body.ChannelID, body.CampaignID)
	if err != nil {
		jsonError(w, http.StatusInternalServerError, err.Error())
		return
	}
	jsonResponse(w, http.StatusOK, map[string]string{"channelName": channelName})
}

func (a *APIHandler) postLeave(w http.ResponseWriter, r *http.Request) {
	channelName, err := a.ctrl.LeaveChannel()
	if err != nil {
		jsonError(w, http.StatusInternalServerError, err.Error())
		return
	}
	jsonResponse(w, http.StatusOK, map[string]interface{}{"channelName": channelName})
}

func (a *APIHandler) postPause(w http.ResponseWriter, _ *http.Request) {
	paused := a.ctrl.TogglePause()
	jsonResponse(w, http.StatusOK, map[string]bool{"paused": paused})
}

func (a *APIHandler) getSaveRecordings(w http.ResponseWriter, _ *http.Request) {
	enabled := a.ctrl.GetSaveRecordings()
	jsonResponse(w, http.StatusOK, map[string]bool{"enabled": enabled})
}

func (a *APIHandler) postSaveRecordings(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Enabled bool `json:"enabled"`
	}
	if err := readJSON(r, &body); err != nil {
		jsonError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	a.ctrl.SetSaveRecordings(body.Enabled)
	jsonResponse(w, http.StatusOK, map[string]bool{"enabled": body.Enabled})
}

// ==================== Campaign CRUD ====================

func (a *APIHandler) listCampaigns(w http.ResponseWriter, _ *http.Request, ctx context.Context) {
	campaigns, err := db.ListCampaigns(ctx)
	if err != nil {
		jsonError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if campaigns == nil {
		campaigns = []db.Campaign{}
	}
	jsonResponse(w, http.StatusOK, campaigns)
}

func (a *APIHandler) createCampaign(w http.ResponseWriter, r *http.Request, ctx context.Context) {
	var body struct {
		Name        string  `json:"name"`
		Description *string `json:"description"`
	}
	if err := readJSON(r, &body); err != nil {
		jsonError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	if body.Name == "" {
		jsonError(w, http.StatusBadRequest, "name is required")
		return
	}
	campaign, err := db.CreateCampaign(ctx, body.Name, body.Description)
	if err != nil {
		jsonError(w, http.StatusInternalServerError, err.Error())
		return
	}
	jsonResponse(w, http.StatusCreated, campaign)
}

func (a *APIHandler) getCampaign(w http.ResponseWriter, _ *http.Request, ctx context.Context, id int64) {
	campaign, err := db.GetCampaign(ctx, id)
	if err != nil {
		jsonError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if campaign == nil {
		jsonError(w, http.StatusNotFound, "campaign not found")
		return
	}
	jsonResponse(w, http.StatusOK, campaign)
}

func (a *APIHandler) updateCampaign(w http.ResponseWriter, r *http.Request, ctx context.Context, id int64) {
	var body struct {
		Name        string  `json:"name"`
		Description *string `json:"description"`
	}
	if err := readJSON(r, &body); err != nil {
		jsonError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	campaign, err := db.UpdateCampaign(ctx, id, body.Name, body.Description)
	if err != nil {
		jsonError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if campaign == nil {
		jsonError(w, http.StatusNotFound, "campaign not found")
		return
	}
	jsonResponse(w, http.StatusOK, campaign)
}

func (a *APIHandler) deleteCampaign(w http.ResponseWriter, _ *http.Request, ctx context.Context, id int64) {
	if err := db.DeleteCampaign(ctx, id); err != nil {
		jsonError(w, http.StatusBadRequest, err.Error())
		return
	}
	jsonResponse(w, http.StatusOK, map[string]bool{"deleted": true})
}

// ==================== Sessions ====================

func (a *APIHandler) listSessions(w http.ResponseWriter, _ *http.Request, ctx context.Context, campaignID int64) {
	sessions, err := db.ListSessions(ctx, campaignID)
	if err != nil {
		jsonError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if sessions == nil {
		sessions = []db.Session{}
	}
	jsonResponse(w, http.StatusOK, sessions)
}

func (a *APIHandler) updateSession(w http.ResponseWriter, r *http.Request, ctx context.Context, sessionID int64) {
	var body struct {
		Name string `json:"name"`
	}
	if err := readJSON(r, &body); err != nil {
		jsonError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	if err := db.UpdateSessionName(ctx, sessionID, body.Name); err != nil {
		jsonError(w, http.StatusInternalServerError, err.Error())
		return
	}
	jsonResponse(w, http.StatusOK, map[string]bool{"updated": true})
}

func (a *APIHandler) deleteSession(w http.ResponseWriter, _ *http.Request, ctx context.Context, campaignID int64, sessionID int64) {
	audioFiles, err := db.DeleteSessionFull(ctx, sessionID)
	if err != nil {
		jsonError(w, http.StatusInternalServerError, err.Error())
		return
	}

	// Delete audio files from disk
	for _, f := range audioFiles {
		path := filepath.Join(a.recordingDir, f)
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			log.Printf("Failed to delete audio file %s: %v", path, err)
		}
	}

	// Broadcast updated sessions list
	sessions, err := db.ListSessions(ctx, campaignID)
	if err == nil {
		if sessions == nil {
			sessions = []db.Session{}
		}
		a.hub.Broadcast(map[string]interface{}{
			"type":     "sessions_updated",
			"sessions": sessions,
		}, &campaignID)
	}

	jsonResponse(w, http.StatusOK, map[string]bool{"deleted": true})
}

func (a *APIHandler) getSessionPreview(w http.ResponseWriter, _ *http.Request, ctx context.Context, sessionID int64) {
	first, last, err := db.GetSessionPreview(ctx, sessionID, 5)
	if err != nil {
		jsonError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if first == nil {
		first = []db.TranscriptionRow{}
	}
	if last == nil {
		last = []db.TranscriptionRow{}
	}
	jsonResponse(w, http.StatusOK, map[string]interface{}{
		"first": first,
		"last":  last,
	})
}

func (a *APIHandler) mergeSessions(w http.ResponseWriter, r *http.Request, ctx context.Context, campaignID int64) {
	var body struct {
		SessionIDs []int64 `json:"sessionIds"`
		Name       string  `json:"name"`
	}
	if err := readJSON(r, &body); err != nil {
		jsonError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	newID, err := db.MergeSessions(ctx, campaignID, body.SessionIDs, body.Name)
	if err != nil {
		jsonError(w, http.StatusBadRequest, err.Error())
		return
	}

	// Broadcast updated sessions list
	sessions, err := db.ListSessions(ctx, campaignID)
	if err == nil {
		if sessions == nil {
			sessions = []db.Session{}
		}
		a.hub.Broadcast(map[string]interface{}{
			"type":     "sessions_updated",
			"sessions": sessions,
		}, &campaignID)
	}

	jsonResponse(w, http.StatusOK, map[string]interface{}{"id": newID})
}

func (a *APIHandler) exportSession(w http.ResponseWriter, _ *http.Request, ctx context.Context, sessionID int64) {
	session, err := db.GetSession(ctx, sessionID)
	if err != nil {
		jsonError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if session == nil {
		jsonError(w, http.StatusNotFound, "session not found")
		return
	}

	lines, err := db.GetTranscriptionsBySession(ctx, sessionID)
	if err != nil {
		jsonError(w, http.StatusInternalServerError, err.Error())
		return
	}

	// Merge consecutive same-speaker lines.
	type merged struct {
		timestamp time.Time
		username  string
		nickname  string
		text      string
	}
	var out []merged

	for _, line := range lines {
		nick := ""
		if line.Nickname != nil {
			nick = *line.Nickname
		}

		text := strings.TrimSpace(line.SpokenText)
		if text == "" {
			continue
		}

		// Check if we can merge with the previous entry.
		if len(out) > 0 {
			prev := &out[len(out)-1]
			if prev.username == line.DiscordUsername && prev.nickname == nick {
				// Strip trailing "..." from previous text before merging.
				prevText := strings.TrimRight(prev.text, ".")
				prev.text = prevText + " " + text
				continue
			}
		}

		out = append(out, merged{
			timestamp: line.CreatedAt,
			username:  line.DiscordUsername,
			nickname:  nick,
			text:      text,
		})
	}

	// Format output lines.
	var sb strings.Builder
	for _, m := range out {
		// Ensure sentence-ending punctuation.
		text := strings.TrimSpace(m.text)
		text = strings.TrimRight(text, ".")
		text = strings.TrimSpace(text)
		if text == "" {
			continue
		}
		lastRune := rune(text[len(text)-1])
		if !unicode.IsPunct(lastRune) {
			text += "."
		}

		ts := m.timestamp.Format("3:04:05 PM")
		speaker := m.username
		if m.nickname != "" {
			speaker += " (" + m.nickname + ")"
		}

		fmt.Fprintf(&sb, "%s %s: %s\n", ts, speaker, text)
	}

	// Build filename from session channel name or ID.
	filename := "session"
	if session.ChannelName != nil && *session.ChannelName != "" {
		filename = *session.ChannelName
	}
	filename = strings.Map(func(r rune) rune {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '-' || r == '_' || r == ' ' {
			return r
		}
		return '_'
	}, filename)
	filename += ".txt"

	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.Header().Set("Content-Disposition", fmt.Sprintf(`attachment; filename="%s"`, filename))
	w.WriteHeader(http.StatusOK)
	w.Write([]byte(sb.String()))
}

// ==================== Transcription lines ====================

func (a *APIHandler) getLines(w http.ResponseWriter, r *http.Request, ctx context.Context) {
	if a.tw == nil {
		jsonError(w, http.StatusServiceUnavailable, "transcription worker not available")
		return
	}

	var sessionID *int64
	if s := r.URL.Query().Get("session"); s != "" {
		id, err := parseID(s)
		if err != nil {
			jsonError(w, http.StatusBadRequest, "invalid session id")
			return
		}
		sessionID = &id
	}

	lines, err := a.tw.GetLines(sessionID)
	if err != nil {
		jsonError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if lines == nil {
		lines = []TranscriptionLine{}
	}
	jsonResponse(w, http.StatusOK, lines)
}

func (a *APIHandler) updateLine(w http.ResponseWriter, r *http.Request, ctx context.Context, lineID int64) {
	if a.tw == nil {
		jsonError(w, http.StatusServiceUnavailable, "transcription worker not available")
		return
	}

	var body struct {
		Text string `json:"text"`
	}
	if err := readJSON(r, &body); err != nil {
		jsonError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	hotwords, err := a.tw.UpdateLine(lineID, body.Text)
	if err != nil {
		jsonError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if hotwords == nil {
		hotwords = []string{}
	}
	jsonResponse(w, http.StatusOK, map[string]interface{}{
		"updated":       true,
		"addedHotwords": hotwords,
	})
}

func (a *APIHandler) deleteLine(w http.ResponseWriter, r *http.Request, ctx context.Context, lineID int64) {
	if a.tw == nil {
		jsonError(w, http.StatusServiceUnavailable, "transcription worker not available")
		return
	}

	deleted, err := a.tw.DeleteLine(lineID)
	if err != nil {
		jsonError(w, http.StatusInternalServerError, err.Error())
		return
	}
	jsonResponse(w, http.StatusOK, map[string]bool{"deleted": deleted})
}

// ==================== Search ====================

func (a *APIHandler) searchTranscriptions(w http.ResponseWriter, r *http.Request, ctx context.Context, campaignID int64) {
	q := r.URL.Query().Get("q")
	if q == "" {
		jsonResponse(w, http.StatusOK, []interface{}{})
		return
	}
	results, err := db.SearchTranscriptions(ctx, campaignID, q, 50)
	if err != nil {
		jsonError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if results == nil {
		results = []db.TranscriptionRow{}
	}
	jsonResponse(w, http.StatusOK, results)
}

// ==================== Hotwords ====================

func (a *APIHandler) getHotwords(w http.ResponseWriter, _ *http.Request, ctx context.Context, campaignID int64) {
	words, err := db.GetHotwords(ctx, campaignID)
	if err != nil {
		jsonError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if words == nil {
		words = []string{}
	}
	jsonResponse(w, http.StatusOK, words)
}

func (a *APIHandler) addHotword(w http.ResponseWriter, r *http.Request, ctx context.Context, campaignID int64) {
	var body struct {
		Word string `json:"word"`
	}
	if err := readJSON(r, &body); err != nil {
		jsonError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	added, err := db.AddHotword(ctx, campaignID, body.Word)
	if err != nil {
		jsonError(w, http.StatusBadRequest, err.Error())
		return
	}

	// Broadcast updated hotwords
	a.broadcastHotwords(ctx, campaignID)

	jsonResponse(w, http.StatusOK, map[string]bool{"added": added})
}

func (a *APIHandler) removeHotword(w http.ResponseWriter, r *http.Request, ctx context.Context, campaignID int64) {
	var body struct {
		Word string `json:"word"`
	}
	if err := readJSON(r, &body); err != nil {
		jsonError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	removed, err := db.RemoveHotword(ctx, campaignID, body.Word)
	if err != nil {
		jsonError(w, http.StatusInternalServerError, err.Error())
		return
	}

	a.broadcastHotwords(ctx, campaignID)

	jsonResponse(w, http.StatusOK, map[string]bool{"removed": removed})
}

func (a *APIHandler) updateHotword(w http.ResponseWriter, r *http.Request, ctx context.Context, campaignID int64) {
	var body struct {
		OldWord string `json:"oldWord"`
		NewWord string `json:"newWord"`
	}
	if err := readJSON(r, &body); err != nil {
		jsonError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	updated, err := db.UpdateHotword(ctx, campaignID, body.OldWord, body.NewWord)
	if err != nil {
		jsonError(w, http.StatusBadRequest, err.Error())
		return
	}

	a.broadcastHotwords(ctx, campaignID)

	jsonResponse(w, http.StatusOK, map[string]bool{"updated": updated})
}

func (a *APIHandler) broadcastHotwords(ctx context.Context, campaignID int64) {
	words, err := db.GetHotwords(ctx, campaignID)
	if err != nil {
		log.Printf("broadcastHotwords: %v", err)
		return
	}
	if words == nil {
		words = []string{}
	}
	a.hub.Broadcast(map[string]interface{}{
		"type":     "hotwords_updated",
		"hotwords": words,
	}, &campaignID)
}

// ==================== Members / Character names ====================

func (a *APIHandler) getMembers(w http.ResponseWriter, _ *http.Request) {
	members := a.ctrl.GetVoiceMembers()
	if members == nil {
		members = []VoiceMemberInfo{}
	}
	jsonResponse(w, http.StatusOK, members)
}

func (a *APIHandler) setCharacterName(w http.ResponseWriter, r *http.Request, campaignID int64) {
	var body struct {
		UserID string  `json:"userId"`
		Name   *string `json:"name"`
	}
	if err := readJSON(r, &body); err != nil {
		jsonError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	a.ctrl.SetCharacterName(campaignID, body.UserID, body.Name)
	jsonResponse(w, http.StatusOK, map[string]bool{"updated": true})
}

func (a *APIHandler) bulkNickname(w http.ResponseWriter, r *http.Request, ctx context.Context, campaignID int64) {
	var body struct {
		LineIDs     []int64            `json:"lineIds"`
		NicknameMap map[string]*string `json:"nicknameMap"`
	}
	if err := readJSON(r, &body); err != nil {
		jsonError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	if err := db.BulkUpdateNicknames(ctx, body.LineIDs, body.NicknameMap); err != nil {
		jsonError(w, http.StatusInternalServerError, err.Error())
		return
	}

	// Update character names on the bot controller for each user
	for userID, name := range body.NicknameMap {
		a.ctrl.SetCharacterName(campaignID, userID, name)
	}

	jsonResponse(w, http.StatusOK, map[string]bool{"updated": true})
}

// ==================== Presets ====================

func (a *APIHandler) getPresets(w http.ResponseWriter, r *http.Request, ctx context.Context, campaignID int64) {
	var userID *string
	if u := r.URL.Query().Get("userId"); u != "" {
		userID = &u
	}

	presets, err := db.GetNicknamePresets(ctx, campaignID, userID)
	if err != nil {
		jsonError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if presets == nil {
		presets = []db.NicknamePreset{}
	}
	jsonResponse(w, http.StatusOK, presets)
}

func (a *APIHandler) addPreset(w http.ResponseWriter, r *http.Request, ctx context.Context, campaignID int64) {
	var body struct {
		DiscordUserID string `json:"discordUserId"`
		Label         string `json:"label"`
		Position      int    `json:"position"`
	}
	if err := readJSON(r, &body); err != nil {
		jsonError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	id, err := db.AddNicknamePreset(ctx, campaignID, body.DiscordUserID, body.Label, body.Position)
	if err != nil {
		jsonError(w, http.StatusInternalServerError, err.Error())
		return
	}

	a.hub.BroadcastPresets(ctx, campaignID)

	jsonResponse(w, http.StatusCreated, map[string]int64{"id": id})
}

func (a *APIHandler) updatePreset(w http.ResponseWriter, r *http.Request, ctx context.Context, campaignID int64, presetID int64) {
	var body struct {
		Label    string `json:"label"`
		Position *int   `json:"position"`
	}
	if err := readJSON(r, &body); err != nil {
		jsonError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	if err := db.UpdateNicknamePreset(ctx, presetID, body.Label, body.Position); err != nil {
		jsonError(w, http.StatusInternalServerError, err.Error())
		return
	}

	a.hub.BroadcastPresets(ctx, campaignID)

	jsonResponse(w, http.StatusOK, map[string]bool{"updated": true})
}

func (a *APIHandler) deletePreset(w http.ResponseWriter, _ *http.Request, ctx context.Context, campaignID int64, presetID int64) {
	if err := db.DeleteNicknamePreset(ctx, presetID); err != nil {
		jsonError(w, http.StatusInternalServerError, err.Error())
		return
	}

	a.hub.BroadcastPresets(ctx, campaignID)

	jsonResponse(w, http.StatusOK, map[string]bool{"deleted": true})
}

func (a *APIHandler) movePreset(w http.ResponseWriter, r *http.Request, ctx context.Context, campaignID int64) {
	var body struct {
		PresetID   int64  `json:"presetId"`
		CategoryID *int64 `json:"categoryId"`
	}
	if err := readJSON(r, &body); err != nil {
		jsonError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	if err := db.MovePresetToCategory(ctx, body.PresetID, body.CategoryID); err != nil {
		jsonError(w, http.StatusInternalServerError, err.Error())
		return
	}

	a.hub.BroadcastPresets(ctx, campaignID)

	jsonResponse(w, http.StatusOK, map[string]bool{"moved": true})
}

// ==================== Categories ====================

func (a *APIHandler) getCategories(w http.ResponseWriter, r *http.Request, ctx context.Context, campaignID int64) {
	var userID *string
	if u := r.URL.Query().Get("userId"); u != "" {
		userID = &u
	}

	categories, err := db.GetNicknameCategories(ctx, campaignID, userID)
	if err != nil {
		jsonError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if categories == nil {
		categories = []db.NicknameCategory{}
	}
	jsonResponse(w, http.StatusOK, categories)
}

func (a *APIHandler) addCategory(w http.ResponseWriter, r *http.Request, ctx context.Context, campaignID int64) {
	var body struct {
		DiscordUserID string `json:"discordUserId"`
		Name          string `json:"name"`
		Position      int    `json:"position"`
	}
	if err := readJSON(r, &body); err != nil {
		jsonError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	id, err := db.AddNicknameCategory(ctx, campaignID, body.DiscordUserID, body.Name, body.Position)
	if err != nil {
		jsonError(w, http.StatusInternalServerError, err.Error())
		return
	}

	a.hub.BroadcastPresets(ctx, campaignID)

	jsonResponse(w, http.StatusCreated, map[string]int64{"id": id})
}

func (a *APIHandler) updateCategory(w http.ResponseWriter, r *http.Request, ctx context.Context, campaignID int64, catID int64) {
	var body struct {
		Name     string `json:"name"`
		Position *int   `json:"position"`
	}
	if err := readJSON(r, &body); err != nil {
		jsonError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	if err := db.UpdateNicknameCategory(ctx, catID, body.Name, body.Position); err != nil {
		jsonError(w, http.StatusInternalServerError, err.Error())
		return
	}

	a.hub.BroadcastPresets(ctx, campaignID)

	jsonResponse(w, http.StatusOK, map[string]bool{"updated": true})
}

func (a *APIHandler) deleteCategory(w http.ResponseWriter, _ *http.Request, ctx context.Context, campaignID int64, catID int64) {
	if err := db.DeleteNicknameCategory(ctx, catID); err != nil {
		jsonError(w, http.StatusInternalServerError, err.Error())
		return
	}

	a.hub.BroadcastPresets(ctx, campaignID)

	jsonResponse(w, http.StatusOK, map[string]bool{"deleted": true})
}

// ==================== Ignored users ====================

func (a *APIHandler) getIgnoredUsers(w http.ResponseWriter, _ *http.Request, campaignID int64) {
	users, err := a.ctrl.GetIgnoredUsers(campaignID)
	if err != nil {
		jsonError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if users == nil {
		users = []db.IgnoredUser{}
	}
	jsonResponse(w, http.StatusOK, users)
}

func (a *APIHandler) ignoreUser(w http.ResponseWriter, r *http.Request, campaignID int64) {
	var body struct {
		UserID   string `json:"userId"`
		Username string `json:"username"`
	}
	if err := readJSON(r, &body); err != nil {
		jsonError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	added, err := a.ctrl.IgnoreUser(campaignID, body.UserID, body.Username)
	if err != nil {
		jsonError(w, http.StatusInternalServerError, err.Error())
		return
	}
	jsonResponse(w, http.StatusOK, map[string]bool{"added": added})
}

func (a *APIHandler) unignoreUser(w http.ResponseWriter, r *http.Request, campaignID int64) {
	var body struct {
		UserID string `json:"userId"`
	}
	if err := readJSON(r, &body); err != nil {
		jsonError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	removed, err := a.ctrl.UnignoreUser(campaignID, body.UserID)
	if err != nil {
		jsonError(w, http.StatusInternalServerError, err.Error())
		return
	}
	jsonResponse(w, http.StatusOK, map[string]bool{"removed": removed})
}

// Ensure unused imports don't cause errors
var _ = fmt.Sprintf
