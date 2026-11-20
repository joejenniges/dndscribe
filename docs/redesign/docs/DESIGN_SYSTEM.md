# UI / UX design system and navigation

## UX goals

- Existing screens mix narrow unused-center layouts (campaigns/sessions) with excessive transcript density. Give management screens readable centered containers and transcript/DJ screens full-width workspaces.
- Make essential actions discoverable. Move infrequently used actions into menus/drawers.
- Two-level navigation: **global** Campaigns | Library | DJ Studio | Tools | Settings; **campaign-local** Sessions | Characters | Notes | Settings. Do not display campaign-local navigation on every global page.
- Campaign switcher in persistent header (when in a campaign). Preserve browser history, deep links, and the currently selected campaign.
- Session viewer may enter Focus Mode: collapses sidebar and most global chrome.
- DJ Studio itself uses **Mixer | Music library | Playlists | History**. Separate the server-side file explorer/library from the active queue and deck controls.

## Reference layout

- Global bar: 64px desktop. Local nav: 54px.
- Campaigns and sessions: max content width 1240px; responsive cards/list.
- Transcript: flexible reading area + ~300px hideable tools panel. Do not introduce a 1240px constraint that wastes screen width on large monitors.
- DJ: 265px left music library / playlist rail and flexing mixer canvas; two decks flanking master mixer on wide screens. On smaller screens stack decks and provide library drawer.
- Persist panel widths, density mode, selected filters, and preferred DJ view when practical.

## Tokens

| Token | Suggested value |
|---|---|
| canvas | `#101113` |
| primary surface | `#1A1C20` |
| raised surface | `#25282E` |
| border | `#32353B` |
| accent | `#F43F62` (primary button may use deeper `#BD1C47`) |
| text | `#E8EAF0` |
| secondary text | `#9CA1AA` |
| success | `#74D7A9` |
| default body | 14–15px |
| headline | 24–30px |
| button height | 36–40px |
| form height | 38–42px |
| row height | 60–72px |
| card padding | 16–24px |
| common gap scale | 4, 8, 12, 16, 24, 32px |

Use semantic CSS variables, do not scatter literal hex values across production Svelte files. UI controls need accessible focus states and 4.5:1 text contrast where possible. Ensure contrast for muted text on dark surfaces.

## Screen-specific requirements

### Campaigns

- Grid/list toggle optional; initially use 2–3 cards at desktop widths, 1 column mobile.
- Campaign title, useful description (where present), last session date, session count, open action.
- New Campaign opens modal/drawer; edit and delete in overflow menu. Delete confirmation includes consequences and respects existing backend semantics.
- Empty state describes how to start.

### Sessions

- Search and sort. Click whole row to open transcript; prominent session title, date, status; robust pending, processing, failed, live, ended states based on real backend data.
- Manage ignored users in campaign settings; avoid tacking unrelated controls onto session history.
- Live channel action stays accessible without dominating every row.

### Transcript

- Responsive reading layout; message lines 14–15px with 1.5–1.65 line-height.
- Comfortable/compact density preference; retain semantics of any existing `Condensed` control if different.
- Align speaker, timestamp, segment playback, and confidence with predictable spacing.
- Keep speaker colors stable and accessible; add text names in addition to color.
- Right panel collapsible/resizable: voice members, hotwords, ignored users, save audio, confidence controls, search/filter preferences.
- Search supports results with surrounding context; preserve original transcript order, editing behavior, and audio alignment.
- Avoid autoscrolling a user who has scrolled away from live tail; offer `Jump to live`.
- Virtualize large transcripts if needed without breaking search/highlight or playback positioning.

### DJ Studio

- Always show voice connection state and destination channel; buttons disabled when connection/capability is missing.
- Two manual decks A and B, independent play/pause/stop/load/seek (if supported), track title, duration/progress, per-deck gain, clip status.
- **Crossfader** controls relative A/B mix, with clear L/R labels; **Master gain** controls overall output separately. Surface meters and safe output warnings.
- Auto DJ: choose playlist, choose crossfade length, choose end behavior (stop, loop, or next list), start/pause/stop, reorder queue. Specify whether manual intervention suspends automation.
- Music file explorer: folder tree + breadcrumb path + search + compatible file metadata, explicit upload/index workflow as backend supports it. Must never expose arbitrary host filesystem paths.
- Playlist CRUD and drag/reorder tracks. Queue behavior clearly distinguished from playlist ordering and from 'load into deck B'.
- DJ is globally reachable but output context is a particular Discord guild and voice channel, optionally associated with a campaign.

## Responsiveness & keyboard

- Desktop screenshot targets: 1600x1000 reference; also test 1920x1080, 1366x768, 1024px, 768px, and ~390px.
- On narrow screens a deck/panel can become a tab or drawer; do not require horizontal scrolling for basic playback actions.
- All buttons, toggles, sliders, dialogs, menus, and drag alternatives must work via keyboard with visible focus.
- Real sliders should expose accessible labels, current values, min/max/step, and optionally direct numeric entry.
- Destructive actions need confirmation; playback commands should report real server success/failure.

## Routes (adapt to existing app)

```text
/campaigns
/campaigns/:campaignId/sessions
/campaigns/:campaignId/sessions/:sessionId
/campaigns/:campaignId/characters
/campaigns/:campaignId/notes
/campaigns/:campaignId/settings
/library
/dj/mixer
/dj/library
/dj/playlists
/dj/history
/tools
/settings
```

Do not migrate existing URLs without redirects. Prefer the existing Svelte routing solution; if using SvelteKit, these become route folders. The screenshots are visual references, not exact pixel requirements.
