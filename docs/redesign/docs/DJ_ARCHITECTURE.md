# Discord DJ Studio — technical design

## The critical constraint

**Outgoing music is played into Discord from Go**, while incoming individual user voice streams continue to be captured and transcribed. The browser is a controller. Do not use browser `<audio>` or Web Audio as the authoritative mixed stream; preview is a separate optional feature.

This is a system design proposal, not verified against the existing source tree. Start by inspecting the Go Discord bot implementation, voice receive pipeline, session lifecycle, auth, music storage, router, and Svelte store architecture.

## Logical data flow

```text
Svelte mixer / playlists / explorer
  | REST commands + authenticated WebSocket state updates
  v
Go DJ coordinator (authorization, per-guild session, command serialization)
  |-- Media library (server-managed root, metadata/index, playlists)
  |-- Deck A: decode -> 48 kHz stereo float PCM -> gain -> \
  |                                                    sum -> master gain
  |-- Deck B: decode -> 48 kHz stereo float PCM -> gain -> /     -> limiter/clip protection
  |                                                              -> PCM -> Opus
  |                                                              -> Discord voice sender
  +-- state snapshots, telemetry, queue events -----------------> Svelte

Existing Discord incoming per-speaker voice receive -> existing transcription pipeline
       (must remain intact and avoid feeding outgoing bot music to STT)
```

## Audio engine specifics

1. Use FFmpeg/libav or a vetted Go decoder/resampler to normalize supported input formats (MP3, OGG, WAV, FLAC as deployment permits) into *aligned*, 48 kHz, stereo, float32 PCM. Keep a prebuffer for the next deck. Avoid restarting a shell process on every fade tick. Limit decoder resource usage and track errors.
2. Internal mixer clock: 48,000 samples/sec. For a typical 20ms voice packet, render 960 frames/channel, mix, limit, encode to **Opus** and send using the selected maintained Discord voice transport. Verify frame sizing and current encryption/E2EE requirements from current Discord documentation.
3. Let `x ∈ [0,1]` be crossfader position from all A to all B. For equal-power fading, `fA=cos(pi*x/2)` and `fB=sin(pi*x/2)`. Let `gA` and `gB` be individual linear deck gains derived from dB; let `gM` be master gain. Each stereo PCM sample: `y = limiter(gM * (fA*gA*A + fB*gB*B))`. Include per-deck mute and a master mute. Equal-power is a selectable/default curve; summing correlated tracks can boost levels, so provide headroom and limiter.
4. Animate crossfader values by the **audio sample clock** or frame clock, not by network request intervals. Auto DJ calculates target fade start from the current track remaining duration (or user cue/loop points), uses e.g. an 8-second ramp, and swaps decks when complete.
5. Manual mixing: A and B can both play; a queued deck is loaded/prebuffered but not playing until started or automation triggers. When the crossfader is at the center, both decks contribute, even if 'queued' status wording in mockups suggests otherwise.
6. Auto mode: tracks selected in playlist order. Start first on one deck; prebuffer next on the other. At fade boundary start second deck at 0 and ramp crossfader; as transition ends unload/replace outgoing deck. Manual knob/fader intervention **pauses Auto DJ scheduling** until user resumes (recommended default). Provide explicit Auto DJ UI state.
7. Master and channel controls accept dB values with bounded safe ranges, clamp requests server-side, smooth volume/fader changes to avoid clicks, and do not permit huge amplification. Master gain is not just a replacement for the crossfader.
8. No track? End of playlist? Document stop vs repeat policy. Very short tracks whose duration is shorter than fade length need a reduced transition length. Decoder failure must skip with surfaced error rather than silently stopping the engine.
9. Report actual stream health: voice transport connected, decode buffer, Opus send errors, reconnect state; do not derive 'healthy' from browser UI being open.
10. Provide seek only where source and decoder permit it, and define whether seeking while auto-mixing recalculates transition timing.

## Discord connection and incoming transcription coexistence

- Inventory current voice library and bot identity. Existing bot apparently receives distinct per-user streams for transcript generation; do **not** create a competing voice connection blindly. A bot typically has one active voice connection per guild, so share/coordinate a single connection if compatible, or deliberately design separate bot identities when required by concrete limitations.
- For an existing per-guild receive connection, determine if the library supports concurrent outbound Opus **and** inbound speaker audio with current Discord voice protocol and DAVE/E2EE. If not, choose a maintained voice transport or isolate responsibilities while keeping speech capture intact.
- Prevent outgoing bot music from being ingested as a speaker transcript; ignore the bot ID and/or isolate outgoing sender stream in capture routing. Check Discord echo/reference behavior before assuming perfect separation.
- Joining/leaving channels and sessions needs coherent ownership. What happens if the transcription session ends while music is playing? Recommended: explicit policies by guild, with DJ able to outlive the recording when permissions allow, but without tearing down a shared live voice connection unexpectedly.
- Verify bot permissions (View Channel, Connect, Speak), user authorization, channel selection, connection failures, rate limits, reconnects, and shard/session races.
- Current Discord voice connections require careful treatment of encryption and DAVE/E2EE; verify protocol and Go library compatibility at implementation time, not from vintage code snippets.

## Server lifecycle / concurrency

- DJ session keyed by Discord guild and voice context. One authoritative mixer state per guild; no duplicate pipelines on multiple admin browsers.
- Serialize operations through a coordinator/event loop. Each command has request ID and optional `expectedRevision` to avoid stale overwrites; server broadcasts snapshots and revisioned events.
- Maintain a bounded ring buffer and apply backpressure; underruns can emit Opus silence and recovery telemetry. Use monotonic clocks for audio scheduling.
- Decoder workers cancel on stop/skip/disconnect. Ensure no subprocess leaks and close resources on shutdown.
- Server-side playlist edits persist independently of playing state. On restart default to stopped/disconnected unless explicitly configured to resume; never surprise-send audio into Discord.
- Apply permissions server-side: only authorized guild admins/DJs can manage library, switch channels or control audio. Browser inputs are never trusted for path selection.

## Media/file explorer security

- Configure `MUSIC_ROOT`, a dedicated approved server directory. Browse **logical relative paths only**, never arbitrary absolute host paths. Canonicalize and check resolved paths remain inside root, including symlink behavior.
- Validate upload size, file type by sniffing, filename, extension, quotas, disk limits; optionally transcode/index. Do not accept URLs fetched by server without SSRF protection, or skip remote URL ingestion in MVP.
- Index duration, title, artist, album, file hash/fingerprint as available. Filename is an acceptable fallback title. Store playlist ordering in database; do not couple to physical file paths more tightly than necessary.
- Require authentication for file, upload and playback commands. Return structured errors; no host paths or credentials in errors.
- Use copyright-cleared or otherwise authorized tracks for Discord broadcast.

## Architectural decisions to document first

1. Current Go Discord library and its inbound/outbound interoperability.
2. Auth and per-guild permission model; which campaign owns the DJ workspace.
3. Media root path and file acquisition/index policy.
4. FFmpeg binary deployment, native encoder dependencies, and Opus implementation.
5. Whether DJ survives end-of-session transcription; connection ownership.
6. Multiple guild sessions and resource budget.
7. External API approach; use existing app conventions for REST and events.

## Suggested incremental spike

Before building the full UI, demonstrate on a test Discord guild: existing inbound per-user capture continues and retains correct speaker association while Go sends a continuous short test Opus music stream on the same (or intentionally coordinated) connection. Record results and any DAVE compatibility findings. This is the highest-risk dependency.

## Useful sources for implementation-time verification

- https://github.com/discord/discord-api-docs/blob/main/developers/topics/voice-connections.mdx
- https://discord-go.github.io/discord.go/high-level/voice-guide
- https://discord-go.github.io/discord.go/low-level/voice/
- https://svelte.dev/docs
