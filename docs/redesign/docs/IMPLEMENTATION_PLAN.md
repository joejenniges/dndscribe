# Implementation plan and acceptance criteria

## Priority 0: repo discovery / critical voice spike

- Audit Svelte routing, state/store pattern, CSS/component library, auth, current routes, tests, Go package layout, Discord bot library/voice transport, transcript receive path, database, deployment, assets.
- Draw the existing inbound Discord capture / per-speaker transcription chain and identify audio connection ownership.
- Prove outgoing music Opus frames can coexist with incoming per-user audio capture under **current** Discord voice encryption/DAVE requirements. Start with a test track before designing a full player.
- Define server-side DJ scope (guild and associated campaign); select voice channel, authorize user actions. Verify Discord permissions. Document findings and unknowns.

**Acceptance:** test guild receives music and continues to produce correctly speaker-attributed user transcripts; bot output is not transcribed; voice connection lifecycle and reconnection tested. If unsupported, stop and propose compatible transport before building the whole UI.

## Priority 1: shell/navigation and shared components

- Build persistent top-level shell with Campaigns, Library, DJ Studio, Tools, Settings.
- Add campaign-level tabs to campaign pages only; retain working deep links and add redirects where needed.
- Establish CSS tokens/theme, responsive containers, 14–15px text, input/button sizes, spacing, states, focus rings.
- Create reusable card, header, list row, overflow menu, dialog, drawer, density toggle, loading/empty/error states.

**Acceptance:** all current pages accessible with browser back/forward and refresh; no loss of existing routes; keyboard navigation and layouts at 390/768/1366/1600px.

## Priority 2: campaigns, sessions, transcript

- Campaign grid/dashboard; new/edit modal and delete confirmation.
- Sessions searchable/sortable list with metadata and clear status; ignored-user administration relocated into settings.
- Transcript viewer with roomy messages, working playback per segment, stable speaker identity, optional confidence display, right control drawer/resizable panel, comfortable/compact density, live-tail affordance, search, hotwords.
- No change to transcript data schema unless required and migrated safely.

**Acceptance:** side-by-side compare with screenshots; existing campaigns/sessions/audio and all transcript features still behave correctly; performance on long sessions is acceptable.

## Priority 3: DJ library and playlists

- Lock down server-root file explorer: folder navigation, searchable indexed tracks, metadata extraction, scan progress, optional validated upload.
- Implement playlist CRUD and ordered items. Drag and drop with keyboard reorder fallback. Persist to database.
- State boundaries: playlists are permanent; queue is ephemeral, per DJ session; loading a track to a deck does not alter the playlist unless explicit.

**Acceptance:** cannot browse outside music root or via symlinks/traversal; malformed formats surfaced; playlist order survives restart; permissions checked server-side.

## Priority 4: server mixer / manual decks

- Two independently decodable decks; optional pause/seek, play/stop/loop; per-deck gain, master gain, crossfader; server-side sample-level mixing; limiter; Opus encode/send; timeline telemetry.
- Crossfader x=0 yields A only, x=1 yields B only, x=.5 both contribute per equal-power curve; actual gain controls apply separately.
- Shared state/event stream; multiple browsers reflect same authoritative controls. Update controls smoothly, not via one giant UI state blob per audio frame.
- Proper channel join, disconnect, cancellation, source errors, and voice send diagnostics.

**Acceptance:** crossfader can live mix without pops; master adjusts total output; UI disconnect does not unintentionally stop transcript recording; server continues audio with browser closed; multi-tab control reconciliation works.

## Priority 5: Auto DJ

- Playlist select, queue display/reorder, fade duration, loop/end policy, start/pause/stop.
- Prebuffer next deck and automatically overlap tracks for user-configured fade duration. Use equal-power curve. Expose status and countdown to next transition.
- Manual action suspends schedule (configurable but predictable). Recovery on decoder error or missing file; reconnect behavior explicit.

**Acceptance:** playlist of 3+ tracks crossfades end-to-end with no gap or harsh transition; short tracks handled; stop/loop behavior correct; UI closed doesn't interrupt audio; pause/resume and manual overrides tested.

## Priority 6: production hardening

- Reconnection, Discord voice handshakes, memory/CPU and file descriptor limits, race tests, auth, audit log, speaker capture isolation, deployment/FFmpeg packaging, monitoring, transport failures.
- Unit tests for crossfade curve, dB conversion, frame count, limiter/clipping, state transitions, playlist order, root path canonicalization.
- Integration tests with fake Opus sink and fake voice transport; manually smoke test on Discord test guild for current encryption protocol.
- Verify audio stays in sync over long sessions; account for clock drift, buffer starvation and timeouts. Document resource budgets.

## Suggested PR breakdown

1. `feat/ui-shell-navigation` — no backend behavior changes
2. `feat/ui-campaign-session-transcript-refresh` — visual refresh preserving features
3. `feat/dj-library-playlists` — persistent music resources
4. `feat/dj-engine-prototype` — test Discord coexistence and output
5. `feat/dj-manual-mixer` — decks and backend APIs
6. `feat/dj-auto-crossfade` — automation and queues
7. `chore/dj-hardening` — tests/security/observability

These are conceptual PR labels. Prioritize the voice coexistence experiment before committing to all the other DJ milestones.

## Not in MVP

- Beat detection, sync/BPM matching, EQ, FX, cue monitoring to headphones, key matching, hardware MIDI controllers, real waveform analysis, audio uploads from external URLs, collaborative cross-guild live sets. Can be added after fundamentals work.
