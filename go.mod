module github.com/joe/dndscribe-go

go 1.23

require (
	github.com/bwmarrin/discordgo v0.29.0
	github.com/ggerganov/whisper.cpp/bindings/go v0.0.0-20260318204338-ef3463bb29ef
	github.com/gorilla/websocket v1.5.3
	github.com/jackc/pgx/v5 v5.7.4
	github.com/k2-fsa/sherpa-onnx-go v1.13.3
	gopkg.in/hraban/opus.v2 v2.0.0-20230925203106-0188a62cb302
	gopkg.in/yaml.v3 v3.0.1
)

require (
	github.com/cloudflare/circl v1.6.3 // indirect
	github.com/jackc/pgpassfile v1.0.0 // indirect
	github.com/jackc/pgservicefile v0.0.0-20240606120523-5a60cdf6a761 // indirect
	github.com/jackc/puddle/v2 v2.2.2 // indirect
	github.com/k2-fsa/sherpa-onnx-go-linux v1.13.3 // indirect
	github.com/k2-fsa/sherpa-onnx-go-macos v1.13.3 // indirect
	github.com/k2-fsa/sherpa-onnx-go-windows v1.13.3 // indirect
	github.com/kr/text v0.2.0 // indirect
	github.com/rogpeppe/go-internal v1.14.1 // indirect
	golang.org/x/crypto v0.32.0 // indirect
	golang.org/x/sync v0.10.0 // indirect
	golang.org/x/sys v0.29.0 // indirect
	golang.org/x/text v0.21.0 // indirect
)

replace github.com/bwmarrin/discordgo => ./_deps/discordgo-fork
