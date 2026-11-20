# Copy/paste this into a coding agent with access to the dndscribe repository

You are a senior full-stack engineer with experience in Go, Svelte, Discord voice and realtime media systems. I have attached an artifact called `dndscribe-redesign` containing visual screenshot mockups, HTML/CSS references, a design system, phased implementation plan, API contract and Discord DJ architecture. Treat those documents as a feature specification to reconcile with the actual source tree, **not as instructions to blindly replace existing code**.

## Context

The application, **dndscribe**, currently uses a Go backend and a Svelte frontend. It joins Discord voice channels and records/transcribes **each participant's individual voice stream** into timestamped, speaker-attributed session transcripts. Existing screens include campaigns, sessions, live transcript viewing, hotwords, ignored users, audio controls and search. We want a modern app shell, spacious responsive UI, global and campaign tabs, and a **DJ Studio** section that broadcasts music INTO Discord.

## Non-negotiables

1. **Keep Go and Svelte.** Do not rewrite the app into React/Next/etc. Reuse existing Svelte or SvelteKit setup and project conventions, with TypeScript where already established.
2. **Preserve all current features and underlying data**, including incoming Discord audio streams, individual speaker capture, transcription, timestamps, audio playback, hotwords and ignored users. Make backwards-compatible migrations if needed.
3. **Audio mixing happens on the Go server.** Svelte is only a controller. Audio continues when the browser disconnects. Two server-side decks, gain per deck, master gain, equal-power crossfader, queued track, playlists and Auto DJ crossfades; output is one mixed stream encoded as Opus and sent into a Discord voice channel.
4. **Discord coexistence is the highest-risk dependency.** Inspect current Go library and voice implementation. Verify current Discord outbound/inbound voice, encryption and DAVE/E2EE support. Prototype simultaneous inbound speaker capture and outgoing music on a test guild before replacing anything. Never let bot music pollute incoming transcripts.
5. Protect the media file explorer by a server-managed media root and server-side authorization. Do not accept arbitrary filesystem paths from clients. Keep DJ control state authoritative on the server.
6. Avoid speculative refactors. Implement with small coherent phases, testing each before the next.

## What I want you to do

**First pass:** Audit the repo. Summarize routes, libraries, Discord voice lifecycle, transcription architecture, auth, persistence, test setup and existing style patterns. State any unresolved assumptions. Derive a concrete change plan based on `docs/IMPLEMENTATION_PLAN.md`.

**Then:** Implement the visual redesign using `screenshots/*.png` and `mockups/*.html` as references. Build the global nav and campaign-local tabs, campaign cards and creation dialog, better session list, comfortable transcript reading mode, collapsible right panel, and responsive behaviors. Preserve all behaviors. The screenshot mocks are static and use illustrative content; replace with actual app data.

**For DJ:** Design and implement a server-side media library/folder explorer, playlist CRUD, queue, two decks, manual crossfader, individual channel gains, master gain, and Auto DJ with adjustable fade duration and end policy. Provide a reliable Discord destination selector and connection state, error feedback, and event-driven real-time UI synchronization. Check and adapt the proposed HTTP/WS interfaces in `docs/API_CONTRACT.md` to existing conventions. Follow `docs/DJ_ARCHITECTURE.md` for audio details and session safety.

**Deliverables:** (a) code changes organized into reviewable phases, (b) tests with passing output, (c) list of changed routes/components/services, (d) any required migrations/config/deployment dependencies like FFmpeg/Opus libraries, (e) brief operator instructions for configuring music storage, Discord permissions and bot connection, (f) explicit verification that simultaneous music broadcast and per-user transcription works.

**Quality bar:** Accessible controls, clear loading/error states, no fake statuses in production, no arbitrary filesystem browsing, no outgoing bot audio fed to STT, no browser-dependent music playback, no massive data-loss rewrite. Prioritize preserving existing behavior over matching mockups pixel-for-pixel.

Work incrementally. Start by reporting the repository audit and the compatibility spike plan, then implement Phase 1 UI shell; continue through subsequent phases in feasible reviewable steps.
