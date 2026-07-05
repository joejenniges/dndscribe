# Build & Run

dndscribe-go is a Go backend plus a SvelteKit frontend. In production the Go
binary serves everything (API, WebSocket, audio files, and the prebuilt
frontend assets) on a single HTTP port. In development the Vite dev server
runs separately and proxies API/WS traffic to the backend.

## Prerequisites

- Go 1.23+ (go.mod declares `go 1.23`)
- Node 18+ and npm
- A working CGo toolchain — the backend links against libopus via
  `gopkg.in/hraban/opus.v2`
  - Windows: MSYS2 UCRT64 with `mingw-w64-ucrt-x86_64-gcc` and
    `mingw-w64-ucrt-x86_64-opus`. **`C:\msys64\ucrt64\bin` must appear on
    `PATH` before any other `mingw64\bin` directory** (notably the one Git
    for Windows ships at `C:\Program Files\Git\mingw64\bin`). If a stale
    MinGW `bin` is resolved first, `cc1.exe` loads ABI-incompatible DLLs
    (libgcc_s_seh-1, libgmp-10, libwinpthread-1) and dies silently during
    compilation, surfacing as `cgo: exit status 2` with no stderr.
    Easiest: build from the MSYS2 UCRT64 shell (`C:\msys64\ucrt64.exe`).
  - Linux: `build-essential` and `libopus-dev`
  - macOS: Xcode command line tools and `brew install opus`
- The streaming engine (sherpa-onnx) needs **no extra C toolchain** — its native
  libs are prebuilt and shipped in the Go module — but its DLLs must be bundled
  next to the binary at runtime (see "sherpa-onnx DLL bundling").
- PostgreSQL reachable at the URL in `config.yaml` → `database.url`

## Frontend build

SvelteKit with `@sveltejs/adapter-static` emits to `web/build/`. The Go server
reads from that directory at runtime, so the frontend must be built before the
backend is run in production mode.

```bash
cd web
npm install
npm run build
```

Outputs static files to `web/build/` including `200.html` used as the SPA
fallback.

## Backend build

Use the build script — it compiles and bundles the sherpa-onnx DLLs (see below):

```powershell
# from repo root, in the MSYS2 UCRT64 shell (or with ucrt64\bin on PATH)
./build.ps1            # builds bin/dndscribe-go.exe + bundles DLLs
./build.ps1 -Smoke     # also builds bin/sherpa-smoke.exe
```

Or build directly:

```bash
go build -o bin/dndscribe-go.exe ./cmd/bot
```

There are no build tags. Both transcription engines are always compiled in and
selected at runtime via `transcribe.engine` in config.yaml:

- **whisper** (batch, default) — links `github.com/ggerganov/whisper.cpp/bindings/go`;
  needs the whisper.cpp library available to the linker (already the case in this
  repo's setup).
- **sherpa** (streaming, low-latency) — links
  `github.com/k2-fsa/sherpa-onnx-go`. The Windows module ships **prebuilt** native
  libraries, so no extra C build is needed, BUT the DLLs must sit next to the
  `.exe` at runtime (handled by `build.ps1`; see "sherpa-onnx DLL bundling").

### sherpa-onnx DLL bundling

`sherpa-onnx-go-windows` links the binary against `sherpa-onnx-c-api.dll` and
`onnxruntime.dll`, which live in the Go module cache — not on `PATH`. A freshly
built `.exe` will fail to start (Windows error `0xc0000135`,
STATUS_DLL_NOT_FOUND) unless `onnxruntime.dll`, `sherpa-onnx-c-api.dll`, and
`sherpa-onnx-cxx-api.dll` sit beside it. `build.ps1` copies them from the module
cache automatically, so `bin/dndscribe-go.exe` just works.

**Important — the onnxruntime.dll must be the module's, not Windows'.** Windows
ships an older `onnxruntime.dll` in `System32` (Windows ML). The bundled
sherpa-onnx-c-api requests a newer ORT API, so if the System32 one loads first
you get a hard crash: *"The requested API version [24] is not available ...
Current ORT Version is: 1.17.1"*. Because an exe's own directory is searched
before `System32`, the bundled binary in `bin/` is fine — but a bare `go test`
runs its temp binary from a directory without the DLL and falls through to
System32. So **don't** rely on a `PATH` prepend for tests (PATH loses to
System32). Compile the test binary into `bin/` (next to the good DLLs) and run it
there:

```powershell
go test -c -o bin/transcribe.test.exe ./internal/transcribe
$env:SHERPA_MODEL_DIR = (Resolve-Path "models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17").Path
cd bin; ./transcribe.test.exe "-test.run=Sherpa|StreamResampler" "-test.v=true"; cd ..
```

The sherpa engine test skips automatically if no model is present, so the
resampler tests run without a download.

Note: UCRT64 gcc links these `x86_64-pc-windows-gnu` (mingw/msvcrt) DLLs fine —
the boundary is a clean dynamic C API, so the CRTs don't clash.

### Downloading models (sherpa only)

```powershell
./download-model.ps1            # ASR (20M streaming Zipformer) + punctuation model
./download-model.ps1 -SkipPunct # ASR only
```

Point `transcribe.sherpa.model_dir` at the ASR folder and
`transcribe.sherpa.punctuation.model_dir` at the punctuation folder. See
config.example.yaml.

### How streaming lines are chunked

- A new line is committed per **spoken turn** — a run of speech bounded by ~1.5s
  of silence (the voice-layer flush). Long monologues break at the next sentence
  boundary once a turn passes `soft_cap_seconds` (default 12), with
  `rule3_min_utterance_length` (30s) as a hard backstop.
- Raw streaming ASR is ALL-CAPS with no punctuation. The **punctuation +
  truecasing** model (`sherpa-onnx-online-punct-en-*`) is applied to committed
  finals, so lines read like prose and match the whisper engine. Live partials
  are shown lowercased-raw (no punctuation) for speed and "snap" to
  cased/punctuated text when the line commits. Set
  `transcribe.sherpa.punctuation.enabled: false` to ship raw caps.

## Running

### Production (single process, single port)

Run from the repo root so the relative path `web/build` resolves correctly
(`internal/web/server.go:59`):

```bash
./dndscribe-go.exe
```

The server listens on `cfg.Web.Port` and handles:

- `GET /api/...` — REST API
- `/ws` (and any path with the `Upgrade: websocket` header) — WebSocket hub
- `GET /recordings/raw/...` — audio file serving with HTTP Range support
- Everything else — static assets from `web/build/`, with SPA fallback to
  `web/build/200.html`

### Development (hot-reload frontend)

Two terminals:

```bash
# terminal 1 — backend
./dndscribe-go.exe
```

```bash
# terminal 2 — frontend with HMR
cd web
npm run dev
```

`npm run dev` starts Vite on `http://localhost:5173` and proxies `/api`,
`/recordings`, and `/ws` to `http://localhost:3001` (`web/vite.config.ts`).
Open `http://localhost:5173` in the browser — the backend still needs to be
running on port 3001 for API/WS calls to succeed.

## Configuration

Copy the example file on first setup:

```bash
cp config.example.yaml config.yaml
```

Fields (`config.yaml`):

```yaml
discord:
  token: ""          # Discord bot token
  guild_id: ""       # Guild the bot operates in

database:
  url: "postgresql://dev:dev@localhost:5432/dndscribe"

web:
  port: 3001         # HTTP port for the Go server

transcribe:
  engine: "whisper"  # whisper (batch) | sherpa (streaming, low-latency)
  model: "base"      # whisper model name
  threads: 4         # whisper CPU threads
  sherpa:            # used only when engine: "sherpa" -- see config.example.yaml
    model_dir: "models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"
    variant: "int8"

recordings:
  dir: "recordings"  # directory for saved PCM / encoded audio
  save_raw: false    # persist raw opus packets per SSRC
```

### Changing the port

Only one port is used. Edit `web.port` in `config.yaml`:

```yaml
web:
  port: 8080
```

If you change the backend port, update the proxy targets in
`web/vite.config.ts` so `npm run dev` still reaches the backend:

```ts
server: {
  proxy: {
    '/api':         'http://localhost:8080',
    '/recordings':  'http://localhost:8080',
    '/ws':          { target: 'ws://localhost:8080', ws: true },
  },
}
```

To run Vite itself on a different port:

```bash
npm run dev -- --port 5174
```

## One-shot full build

```bash
cd web && npm install && npm run build && cd ..
go build -o dndscribe-go.exe ./cmd/bot
./dndscribe-go.exe
```
