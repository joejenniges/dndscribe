# Proposed API contract — adapt to existing app conventions

**Design proposal only.** Audit existing endpoints and WebSocket frames first; do not replace working APIs unnecessarily. The server is authoritative for DJ playback. All requests require existing authentication and per-guild authorization; examples omit auth headers.

## Library and playlists

| Method | Suggested route | Purpose |
|---|---|---|
| GET | `/api/dj/library?folder=relative/path` | List indexed server-root folders/tracks, paginated; never accept absolute paths |
| POST | `/api/dj/library/scan` | Authorized re-index request; return job id |
| POST | `/api/dj/library/upload` | Validated multipart upload into an approved directory (optional MVP) |
| GET | `/api/dj/tracks/:trackId` | Metadata and supported playback operations |
| GET | `/api/dj/playlists` | List playlists visible to scope |
| POST | `/api/dj/playlists` | Create playlist |
| GET | `/api/dj/playlists/:id` | Playlist metadata and ordered tracks |
| PATCH | `/api/dj/playlists/:id` | Rename/configure |
| PUT | `/api/dj/playlists/:id/items` | Replace/reorder tracks atomically with revision |
| DELETE | `/api/dj/playlists/:id` | Delete with permission and confirmation |

## Connection and mixer

| Method | Suggested route | Purpose |
|---|---|---|
| GET | `/api/dj/guilds/:guildId/state` | Full authoritative status + state revision |
| POST | `/api/dj/guilds/:guildId/connection` | Connect to validated voice channel; body `{ "channelId": "..." }` |
| DELETE | `/api/dj/guilds/:guildId/connection` | Disconnect with clear impact on capture policies |
| POST | `/api/dj/guilds/:guildId/decks/A/load` | Load track by server ID; same for B |
| POST | `/api/dj/guilds/:guildId/decks/A/play` | Play; similarly pause, stop, seek |
| PATCH | `/api/dj/guilds/:guildId/mixer` | Set crossfader, channel gains, master gain and mute safely |
| POST | `/api/dj/guilds/:guildId/auto` | Start auto DJ with playlist ID and fade policy |
| PATCH | `/api/dj/guilds/:guildId/auto` | Pause/resume and edit next transition |
| DELETE | `/api/dj/guilds/:guildId/auto` | Stop auto DJ (define whether current audio stops) |
| PUT | `/api/dj/guilds/:guildId/queue` | Replace/reorder current queue |

**Do not put raw file paths in load requests**. Use indexed track IDs and validate scope. Idempotent commands should accept `requestId`. Use optimistic revisions for stateful multi-controller updates.

## Example mixer command

```json
{
  "requestId": "f3d836e7-b15d-4e2c-9b76-3da7b23d0de7",
  "expectedRevision": 38,
  "crossfader": 0.35,
  "deckAGainDb": -1.0,
  "deckBGainDb": -4.0,
  "masterGainDb": -3.0
}
```

Use debounced/coalesced slider updates; on pointer-up commit immediately. The Go mixer smooths parameter transitions internally, regardless of network update cadence. Define the accepted dB bounds server-side and report effective clamped values.

## Server event stream

Existing WS endpoint if available, otherwise scoped authenticated `/api/dj/guilds/:guildId/events`. On subscribe return a full `dj.state` snapshot, then ordered revisioned changes. Include heartbeats. On reconnect request fresh snapshot, do not trust stale optimistic UI.

Illustrative event:

```json
{
  "type": "dj.state",
  "guildId": "discord-guild-id",
  "revision": 39,
  "connection": {"state": "connected", "voiceChannelId": "channel-id"},
  "auto": {"mode": "stopped", "playlistId": null, "fadeSeconds": 8},
  "decks": {
    "A": {"trackId": "track-1", "status": "playing", "positionMs": 134000, "durationMs": 342000},
    "B": {"trackId": "track-2", "status": "loaded", "positionMs": 0, "durationMs": 258000}
  },
  "mixer": {"crossfader": 0.35, "deckAGainDb": -1, "deckBGainDb": -4, "masterGainDb": -3},
  "telemetry": {"transport": "healthy", "underrunCount": 0, "peakDbfs": -8.2}
}
```

Stream position updates at a practical rate (~4–10Hz for UI), interpolate locally for smooth labels, and do not flood clients with 50 audio-frame events per second. Command responses contain request id, accepted revision, and structured typed errors. Surface connection/permission/decoder/unsupported-format/playlist-empty/buffering errors in UI.

## Storage suggestions

- `media_tracks(id, relative_path, content_hash, duration_ms, format, title, artist, indexed_at, ...)`
- `dj_playlists(id, scope_id, name, created_by, updated_at, revision, ...)`
- `dj_playlist_items(playlist_id, ordinal, track_id, ...)`
- `dj_audit_events(guild_id, user_id, action, timestamp, outcome, ...)` optional

Adapt to existing SQL migrations and campaign/guild model; do not hardcode database product.
