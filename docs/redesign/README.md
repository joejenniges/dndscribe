# dndscribe — UI modernization + Discord DJ Studio

**Handoff package** · November 19, 2026 · Concept/implementation specification, **not an implemented app**.

## What is included

| Path | Purpose |
|---|---|
| `screenshots/campaigns-desktop.png` | Campaign dashboard design screenshot |
| `screenshots/sessions-desktop.png` | Session browsing design screenshot |
| `screenshots/transcript-desktop.png` | Spacious transcript viewer design screenshot |
| `screenshots/dj-desktop.png` | Discord DJ two-deck mixer design screenshot |
| `mockups/*.html` and `mockups/styles.css` | Editable visual references (static HTML/CSS, **not functioning controls**) |
| `docs/IMPLEMENTATION_PLAN.md` | Sequenced tasks and acceptance criteria |
| `docs/DJ_ARCHITECTURE.md` | Server-side mixing and Discord voice integration design |
| `docs/API_CONTRACT.md` | Proposed HTTP/WebSocket endpoints and event shapes |
| `docs/DESIGN_SYSTEM.md` | UI tokens, layouts, navigation, accessibility |
| `docs/CODING_LLM_PROMPT.md` | Copy/paste prompt for an agent working in the actual repository |

## Reading order for a coding LLM

1. `docs/CODING_LLM_PROMPT.md`
2. `docs/IMPLEMENTATION_PLAN.md`
3. `docs/DJ_ARCHITECTURE.md`
4. `docs/API_CONTRACT.md`
5. `docs/DESIGN_SYSTEM.md`
6. Compare `screenshots/` with `mockups/` and the existing application

## Context and boundaries

- Existing app is a **Go backend and Svelte frontend**. **Keep both.** If already using SvelteKit, reuse it; otherwise do not migrate routers/frameworks without a compelling reason.
- App receives **individual participants' Discord voice streams and transcribes them**. Preserve that pipeline, speaker association, timestamps, session persistence, ignore rules, hotwords, and audio playback.
- The new DJ tab sends one **server-mixed music stream to a Discord voice channel**. The Svelte UI is a remote control; audio must not depend on the browser being open.
- The DJ implementation must coexist with incoming Discord voice capture rather than hijack it. Inspect existing bot/session/voice connection integration before adding any outbound audio.
- Everything in the mockups is **illustrative**: track names, playlist counts, statuses, channel names, music cover illustrations, and control positions are sample data only.
- A production implementation needs authorization, error handling, persistence, test coverage, observability, and Discord protocol compatibility.

## Frontend decision

**Stay with Svelte.** Svelte/TypeScript is well suited to responsive, stateful mixer controls and server-event updates. A framework migration would add risk without solving the hard part: reliable Go-side audio mixing and Discord voice transport. Prefer existing component patterns and package versions; optionally adopt Svelte 5 idioms incrementally if the repo already supports them.

## Preview

Open any HTML file in `mockups/` with a browser; all pages reference `styles.css`. The PNG files are captured at 1600px viewport width. The visual layouts are not a complete accessibility or mobile implementation; the agent should finish those details as described in the design guide.

## External technical references

- Discord voice protocol: https://github.com/discord/discord-api-docs/blob/main/developers/topics/voice-connections.mdx
- Official Svelte docs: https://svelte.dev/docs
- SvelteKit docs (if used): https://svelte.dev/docs/kit/introduction
- Go Discord voice library currently in use: **inspect the project's dependencies rather than assuming a library**. One option to investigate for recent voice/DAVE support: https://discord-go.github.io/discord.go/high-level/voice-guide

**Note:** Discord voice and encryption/E2EE requirements evolve. Verify the live protocol and the selected Go library's ability to transmit AND receive voice before implementation. Do not assume older DiscordGo examples remain compatible.
