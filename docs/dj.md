# DJ Studio: engine and realtime protocol

How the DJ works as built. The design handoff in `dndscribe-redesign/docs/v3/` describes the target UI; where it and this file disagree on backend behaviour, this file is right.

## Audio path

```
Deck A --gain--\
                crossfader (equal power) --+
Deck B --gain--/                           +-- master -- soft limiter -- opus -- Discord
Soundboard effects -- effects gain --------+
```

- Everything is mixed in the Go process (`internal/dj`) on a 20ms clock and sent on the bot's one voice connection, the same one used for recording. Browsers never play audio. Music keeps going with every page closed.
- Effects are mixed after the crossfader, so a crossfade never fades them out.
- Discord does not send the bot its own audio, so music and effects never reach the per-user transcription pipeline.
- Nothing plays while the bot isn't in a voice channel. Decks can still be loaded. If the voice connection drops, playing music is held for 10 seconds, then unloaded along with the queue. Decks that were only cued stay loaded.

## Decks, crossfader, queue

| Thing | Behaviour |
|---|---|
| Deck A / B | One loaded track each. Play, pause (200ms ramp), seek (restarts ffmpeg at the position with a 40ms crossfade), eject, gain 0..1. A deck whose track ended starts it again on Play. |
| Crossfader | 0 = A, 1 = B, equal-power (both at -3dB in the centre). Server state. Moving it by hand cancels any automated move immediately. |
| On air | The playing deck, or with both playing, the one the crossfader favours. |
| Queue | Shared "up next" list. A scene or folder replaces it. Add (next or last), move, remove, clear. Shuffle reorders what's left; turning it off restores the original order. Loop refills it when it runs out (a reshuffle never repeats the last track first). |
| History | The last 30 tracks that played for more than a second. |
| Auto DJ | On by default. Preloads the next queue track onto the idle deck, then crossfades to it starting `fade` seconds before the on-air track ends (at most a third of either track's length). If a track ends before that (unknown duration, or a preload that wasn't ready), it cuts to the other deck in the same frame. Off: a deck stops at the end of its track. |
| Scene | Replaces the queue with the tracks in its folders (sub-folders included), applies its shuffle, loop and volume, puts the first track on the idle deck and moves the crossfader over the scene's fade (or the global one). From silence it starts on the deck the crossfader already points at, with no fader move. Tapping the scene that is already playing does nothing. A newer scene, Next or track command replaces a transition in progress. |
| Next | Crossfades (1s) to whatever is on the idle deck, else the head of the queue. |
| Stop all | Fades both decks and all effects over 0.25s, unloads the decks and clears the queue. |
| Test tone | 440Hz for 2s on the effects bus; never touches the decks. |

Auto DJ preloads are marked `auto` in the state. Starting music from silence drops a stale `auto` preload left on the other deck; a track loaded by hand stays and plays next.

## WebSocket

One socket, at `/` on the web port. JSON text messages.

### Server to client

| type | When | Payload |
|---|---|---|
| `init` | On connect | `status`, `members`, `saveRecordings`, `dj` (a full DJ state) |
| `dj_state` | Any DJ change (at most 25/s), and every 5s while playing | Full state, see below |
| `dj_meters` | 10/s while anything is audible, then once at zero | `{decks:[a,b], master}`, peak 0..1 since the last report |
| `dj.ack` | Reply to the sender's `dj.cmd` only | `{id, ok, error?, rev}` |
| `presence` | A client says hello or leaves | `{clients:[{device, page}]}` |
| `dj_library` | Tracks, folders, scenes or pads changed (via HTTP) | none; refetch what you show |

The DJ state is a whole snapshot, never a delta. A client that missed messages uses the next one. `rev` increases with every change; after an ack with `rev` R, any snapshot with `rev >= R` includes that command.

Positions are not streamed. Each deck has `positionMs` sampled at `at` (server unix ms); clients extrapolate while `playing`. During an automated fade `transition` gives `{kind, toDeck, from, to, startedAt, durationMs}` and clients animate the fader from it.

```jsonc
{
  "rev": 1234, "at": 1795369854960, "connected": true,
  "decks": [
    {"track": {"id": 28, "title": "Descent", "artist": "Austin Wintory", "durationMs": 161060, "path": "Cave Music/..."},
     "playing": true, "ended": false, "positionMs": 42100, "gain": 1, "loading": false, "auto": true},
    {"track": null, "playing": false, "ended": false, "positionMs": 0, "gain": 1, "loading": false, "auto": false}
  ],
  "onAir": 0, "crossfader": 0, "transition": null, "autoDJ": true,
  "volume": 0.5, "fadeSeconds": 6,
  "sceneId": 3, "folder": null, "sourceName": "Cave", "loop": true, "shuffle": false,
  "queue": [{"qid": 17, "id": 31, "title": "...", "artist": "...", "durationMs": 0}], "queueTotal": 28,
  "history": [{"id": 27, "title": "...", "playedAt": 1795369800000}],
  "sfx": [{"padId": 4, "instance": 12, "label": "Thunder", "positionMs": 300, "durationMs": 2100}], "sfxGain": 1,
  "error": ""
  // plus playing/paused/track/positionMs/upNext/programName for the pre-v3 page
}
```

### Client to server

`{"type":"hello","device":"desktop|tablet|phone","page":"dj"}`: presence; send on connect and on page change. Informational only; no device gets more control than another.

`{"type":"dj.cmd","id":"<unique>","cmd":"<name>","args":{...}}`: a live control. Use a new `id` for every user action. Resend the same `id` only for a retry: a repeated id is not run again and gets the original ack. Don't queue commands while disconnected; drop them and let the user act again.

| cmd | args |
|---|---|
| `deck.load` | `deck` (0/1), `trackId` |
| `deck.play` / `deck.pause` / `deck.eject` | `deck` |
| `deck.seek` | `deck`, `positionMs` |
| `deck.gain` | `deck`, `value` 0..1 |
| `mix.crossfader` | `value` 0..1 (cancels automation) |
| `mix.master` / `mix.sfxGain` | `value` 0..1 |
| `mix.autoDJ` | `on` |
| `mix.next` / `mix.stopAll` | none |
| `mix.fadeTo` | `deck`: glide the crossfader to that deck's side over the global fade. Fader only: never starts, stops or unloads a deck (the A/B buttons by the crossfader, and X). Glides at the same speed when nothing is playing or the bot is not in voice (the engine tick steps it). |
| `scene.play` | `sceneId` |
| `folder.play` | `folder` (music-dir relative, "" = all), optional `trackId` to start at |
| `track.play` | `trackId` (crossfade to it now; the queue carries on after) |
| `tone.play` | optional `seconds` |
| `queue.add` | `trackIds` (or `trackId`), `next` (true = front) |
| `queue.move` | `qid`, `to` (index) |
| `queue.remove` | `qid` |
| `queue.clear` | none |
| `queue.shuffle` / `queue.loop` | `on` |
| `sfx.trigger` | `padId` (restarts that pad from zero if it's playing) |
| `sfx.stop` | optional `padId` (none = all effects) |

Faders: send at most ~30 per second while dragging and always the final value on release. While a user drags a control, keep showing their pointer position and ignore that control in incoming snapshots until release. Last write wins between devices; there is no locking.

Errors come back in the ack (`ok:false`, `error` is a sentence for a toast), for example `the bot is not in a voice channel`, `deck B is empty`.

## HTTP

Soundboard uploads ("+ Add", "+ Folder", or a drop on the soundboard) put loose files in the open tab; a file inside a folder goes into a tab named after its own folder (case-insensitive match on an existing tab, otherwise a new one). Nested folders flatten to their last component, since tabs don't nest. This is client-side: the upload route still takes one file and a tab id.

The soundboard search (`/` focuses it) matches pad labels and tab names across every tab with `web/src/lib/dj/fuzzy.ts`: each query word must match as a substring, as letters in order (initials only for 2-3 letter words, e.g. `ds` for Door Slam), or within 1 edit (4+ letters) / 2 edits (7+), swaps counting as one. Results replace the grid, best first; Enter plays the top one, 1-0 trigger results, Esc clears. Reordering is off while searching.

Library folders are the playlists. They can be renamed or moved (`POST /api/dj/folders/rename {from,to}`; a move is a rename to another parent) and reordered among their siblings (`PUT /api/dj/folders/order {paths}`, stored in `dj_folder_order`; folders without a saved position sort alphabetically after the ordered ones). A rename keeps track ids, rewrites track paths and scene folder lists in one transaction, and re-points queued and loaded items (`Engine.RepathFolder`), since items capture the file path when built. Windows refuses to rename a folder while a deck has a file in it open. Folder order is display order only; playing a folder still goes by track path.

Re-uploads are checked before anything is sent. The music upload dialog posts every target path to `POST /api/dj/files/exists` (same `os.Stat` test the upload uses for its 409) and, if any exist, asks "N items will be overwritten, skip or overwrite?". Skip sends only the new files; Overwrite sends everything with `overwrite=1` (track ids are kept, since rows are upserted by path). The soundboard does the same check client-side: a file whose name matches a pad's original file name (case-insensitive) in the tab it would land in is a duplicate. Overwrite there uploads with `?replace=<pad id>`, which swaps the pad's sound file and keeps its label, colour, image, gain and position; the old file is deleted.

Editing and uploads stay on HTTP under `/api/dj/` (tracks, folders, files, scenes, soundboard tabs/pads/images, waveforms, settings). Scenes and pads can each have a background image (`PUT`/`DELETE /api/dj/scenes/{id}/image`, `/api/dj/sfx/pads/{id}/image`, raw body, PNG/JPEG/GIF/WebP up to 5 MB, validated by content). Both are stored under `<soundboard dir>/images/` with random keys and served from `/api/dj/sfx/images/{key}`; saving a scene or pad never changes its image, and deleting one removes its file. The older transport routes (`/api/dj/play`, `pause`, `resume`, `next`, `seek`, `stop`, `volume`, `tone`) still work and act on the on-air deck.

`GET /api/connect` lists the addresses a tablet can open the app on (Tailscale name and IP, home network); `GET /api/connect/qr?url=` returns a QR PNG for one of them.

## Security

There is no login: the app is meant for a home LAN or a Tailscale tailnet. Don't expose the port to the internet. The WebSocket only accepts pages served from the app itself (same host) or localhost, so another website open in the same browser can't drive the DJ or read transcripts.

## Keyboard shortcuts

`web/src/lib/dj/hotkeys.ts`, installed only on the DJ page; the list in that file is what the in-page help (`?`) shows. Tab switches the selected deck (so Tab no longer moves focus on this page), Space plays/pauses it, X crossfades to it (`mix.fadeTo`), plus pads on 1-0, scenes on Shift+1-9, crossfader and master on the arrows, Esc / Shift+Esc to stop effects / everything, `/` to search the soundboard. The selected deck is per device. Shortcuts can be turned off per device in the help dialog, which gives Tab back. They are ignored in text fields, with a dialog open, and with Ctrl/Alt/Cmd held.

## Test channel

Settings page toggle (`testChannel` in `/api/dj/settings`, saved as `dj.test_channel`). When on, "Test channel (no Discord)" (id `test-channel`) is listed last among the voice channels. Joining it never touches Discord: the bot's status reports it as a music-only channel and the engine's `SwitchOutput` sends to a `TestOutput`, which drains frames at the real 20 ms rate (so decks, fades, Auto DJ and effects advance in real time) and copies each frame to any browser listening. Recording in it is refused. Joining a real channel, starting a recording, or switching the setting off leaves it.

The audio plays in the browser. `GET /api/dj/listen` is a WebSocket that sends one binary message per 20 ms Opus frame (48 kHz stereo, what Discord would get); a listener that falls more than a second behind has frames dropped rather than slowing the mixer. `web/src/lib/dj/listen.ts` decodes them with `opus-decoder` (WASM; WebCodecs is only available on https/localhost, and tablets use plain http on the LAN) and schedules them in Web Audio with a 120 ms jitter buffer, dropping frames when more than 350 ms is queued so latency can't creep. Listening is per device ("Listen here" next to Leave, remembered in localStorage as `dj.listen`) so two devices in one room don't echo; joining the test channel from a device turns it on there. After a reload, a remembered "on" waits for the first tap or key press if the browser blocks autoplay ("Tap to hear").

## Deck play and the crossfader

Crossfades are manual. `deck.play` never moves the crossfader, so a deck started on the far side plays silently until faded in, as on a hardware mixer; the deck panel says "Faded out" and the status bar points it out. The client decides "On air" from the deck playing *and* its crossfader weight (`deckHeard` in `web/src/lib/dj/live.ts`); the server's `onAir` ignores the fader. Scene changes, Auto DJ, Next and "Play now" from the library are the automated transitions that move the fader themselves. A glide keeps moving in real time when everything is silent or the bot is not in voice (`idleGlideLocked`), so the fader never jumps and never hangs half-done.
