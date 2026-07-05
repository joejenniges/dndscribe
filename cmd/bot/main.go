package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/joe/dndscribe-go/internal/bot"
	"github.com/joe/dndscribe-go/internal/config"
	"github.com/joe/dndscribe-go/internal/db"
	"github.com/joe/dndscribe-go/internal/transcribe"
	"github.com/joe/dndscribe-go/internal/web"
)

func main() {
	// Load config.
	cfgPath := "config.yaml"
	if len(os.Args) > 1 {
		cfgPath = os.Args[1]
	}
	cfg, err := config.Load(cfgPath)
	if err != nil {
		log.Fatal("Config: ", err)
	}

	// Set up logging to file + stdout.
	if err := os.MkdirAll("logs", 0o755); err == nil {
		ts := time.Now().Format("2006-01-02T15-04-05")
		logPath := filepath.Join("logs", fmt.Sprintf("%s.log", ts))
		if f, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644); err == nil {
			log.SetOutput(io.MultiWriter(os.Stdout, f))
			log.Printf("Logging to %s", logPath)
		}
	}

	// Init database.
	if err := db.Init(cfg.Database.URL); err != nil {
		log.Fatal("DB: ", err)
	}
	defer db.Close()
	log.Println("Database ready")

	// Create WebSocket hub.
	hub := web.NewHub()

	// Create transcription worker.
	worker := transcribe.NewWorker(cfg.Recordings.Dir)

	// Create Discord bot.
	b, err := bot.New(cfg, worker)
	if err != nil {
		log.Fatal("Bot: ", err)
	}

	// Wire bot events to WebSocket broadcasts.
	b.OnStatus = func() {
		hub.Broadcast(map[string]interface{}{"type": "status", "data": b.GetStatus()}, nil)
	}
	b.OnMembers = func() {
		hub.Broadcast(map[string]interface{}{"type": "members", "data": b.GetVoiceMembers()}, nil)
	}
	b.OnSessionStarted = func(id int64, channelName string, campaignID int64) {
		cid := campaignID
		hub.Broadcast(map[string]interface{}{
			"type": "session_started",
			"data": map[string]interface{}{
				"id":          id,
				"channelName": channelName,
				"campaignId":  campaignID,
				"startedAt":   time.Now().Format(time.RFC3339),
				"endedAt":     nil,
			},
		}, &cid)
	}
	b.OnSaveRecordings = func(enabled bool) {
		hub.Broadcast(map[string]interface{}{"type": "save_recordings", "data": enabled}, nil)
	}
	b.OnIgnoredUsers = func(campaignID int64) {
		users, err := db.GetIgnoredUsers(context.Background(), campaignID)
		if err != nil {
			log.Printf("Failed to fetch ignored users: %v", err)
			return
		}
		cid := campaignID
		hub.Broadcast(map[string]interface{}{"type": "ignored_users", "data": users}, &cid)
	}

	// Set up WS init message -- sent to every new client on connect.
	hub.SetInitFunc(func() []byte {
		initMsg := map[string]interface{}{
			"type":           "init",
			"status":         b.GetStatus(),
			"members":        b.GetVoiceMembers(),
			"saveRecordings": b.GetSaveRecordings(),
		}
		data, _ := json.Marshal(initMsg)
		return data
	})

	// Wire transcription worker events to WebSocket broadcasts.
	worker.OnLine = func(line transcribe.TranscriptionLine) {
		cid := worker.GetCampaignID()
		hub.Broadcast(map[string]interface{}{"type": "line", "data": line}, cid)
	}
	worker.OnHotwords = func(hotwords []string) {
		cid := worker.GetCampaignID()
		hub.Broadcast(map[string]interface{}{"type": "hotwords", "data": hotwords}, cid)
	}
	worker.OnFinalized = func() {
		hub.Broadcast(map[string]interface{}{"type": "finalized"}, nil)
	}
	// Streaming engine partials: ephemeral live hypotheses, not persisted. The
	// frontend renders these per-speaker and supersedes them when the final
	// "line" arrives. No-op when running the batch (whisper) engine.
	worker.OnPartial = func(userID, username, nickname, text string) {
		cid := worker.GetCampaignID()
		hub.Broadcast(map[string]interface{}{
			"type": "partial",
			"data": map[string]interface{}{
				"userId":          userID,
				"discordUsername": username,
				"nickname":        nickname,
				"text":            text,
			},
		}, cid)
	}

	// Start transcription worker.
	if err := worker.Init(&cfg.Transcribe); err != nil {
		log.Fatal("Transcribe: ", err)
	}
	worker.Start()
	defer worker.Stop()

	// Start web server.
	srv := web.Start(cfg, b, hub, worker)
	log.Printf("Web UI listening on http://localhost:%d", cfg.Web.Port)

	// Start Discord bot.
	if err := b.Start(); err != nil {
		log.Fatal("Discord: ", err)
	}
	defer b.Stop()
	log.Println("Discord connected")

	// Wait for shutdown signal.
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Println("Shutting down...")

	// Hard deadline: if any subsystem hangs (Discord disconnect, whisper
	// in-flight, stuck WS goroutine), force-exit rather than sit there.
	// 8s is enough for a healthy shutdown (voice disconnect ≤3s, web server
	// close ≤3s, DB close ≤1s) with margin.
	time.AfterFunc(8*time.Second, func() {
		log.Println("Shutdown exceeded 8s deadline, forcing exit")
		os.Exit(1)
	})

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		log.Printf("HTTP server shutdown: %v", err)
	}
}
