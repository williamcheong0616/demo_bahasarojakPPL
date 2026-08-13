# UI refresh, interface language selection, and conversation memory

Date: 2026-08-13

## Goal

Three changes to the Bahasa Rojak demo:

1. Redesign the frontend — distinctive visual direction, new functional affordances, works on phones.
2. Add an interface language selector for Malay, English, and Chinese.
3. Give the SLM conversation memory over the last 5 exchanges.

## Decisions

### Language selection is UI chrome only

The selector translates the interface — buttons, status text, start screen, message
labels. It does **not** change what language the AI replies in, does **not** change the
Whisper language hint (`language="ms"` stays), and does **not** touch `GREETING_TEXT`.

Rationale: the project's identity is Bahasa Rojak. The system prompt in all three
inference backends states "NEVER respond in pure Malay or pure English." The AI's own
voice is not chrome, so it stays fixed while the surrounding interface adapts to the
viewer.

Consequence: no language plumbing anywhere in the backend.

### Memory is 10 messages — 5 user, 5 assistant

The last 5 user→AI exchanges form the context chain for each prompt.

History is owned by the client: a JavaScript array, sliced to the last 10 messages
before each `/api/slm` call. The server re-caps at 10 as the authoritative limit — a
client is not trusted to bound the context. Clearing the conversation clears the array;
a page refresh starts fresh.

Rejected: server-side sessions keyed by cookie. They would survive a refresh but
introduce shared state, cleanup, and multi-tab interference during a live demo. Not
worth it for a single-user demo.

### Ollama context window must be raised

`num_ctx` defaults to 2048 on the Ollama backend. The Bahasa Rojak system prompt is
roughly 400 tokens, so 5 exchanges plus a response would silently truncate the oldest
turns and the memory feature would appear not to work. Set `num_ctx: 8192`.

MLX and CUDA have no equivalent per-request cap — they are bounded by the model's own
window, which comfortably exceeds 10 short messages. No change needed there.

### Frontend stays one self-contained file

`static/index.html` keeps its inline `<style>` and `<script>`, as documented in
`CLAUDE.md`. No build step, no new static assets.

### Visual direction: Kopitiam

Warm editorial. Cream and espresso base with a marigold accent. Serif display type for
the wordmark and headings, sans body text. Malaysian flavour comes from warmth and a
subtle songket-inspired hairline motif — not literal batik imagery. Light and dark
themes both supported via `prefers-color-scheme`.

Chosen over a neon dark treatment because it reads as deliberately designed rather than
as a default dark template, and stays legible on a projector.

## Backend changes

### `main.py`

- New `ChatMessage` model: `{role: "user" | "assistant", content: str}`.
- `SLMRequest` gains `history: list[ChatMessage] = []`.
- `/api/slm` truncates to `history[-10:]` and forwards it to `generate()`.
- New `GET /api/status` returning `{asr: bool, slm: bool, tts: bool}` so the UI can show
  which models loaded instead of discovering a failure via a 503 mid-pipeline.

### `inference_ollama.py`, `inference_mlx.py`, `inference.py`

Each `generate()` gains a keyword-only `history=None` parameter after `input_ctx`.
Messages become `system + history + user`. Keyword-only placement keeps the existing
positional callers — including `inference.py`'s CLI — working untouched.

`CLAUDE.md` marks `inference.py` as "do not modify". The change there is purely additive
and keeps the three backends in sync; it was approved as part of this design.

`inference_ollama.py` additionally sets `options.num_ctx = 8192`.

## Frontend changes

### Structure

- Header: wordmark, model status dot, language selector, clear-conversation button.
- Conversation: scrolling message list, unchanged bubble semantics (user right, AI left,
  replay button on anything with audio).
- Pipeline stage indicator: ASR → SLM → TTS, each lighting as it runs, replacing the
  single spinner status line.
- Composer: text input plus push-to-talk button.

### Interface language

An `I18N` object keyed `ms` / `en` / `zh`. Elements carry `data-i18n` attributes; an
`applyLang()` pass rewrites them. The choice persists in `localStorage` and also sets
`document.documentElement.lang`. Default is `ms`. The font stack includes a CJK face so
Chinese renders properly.

The wordmark "Bahasa Rojak AI" is a brand name and is not translated.

### Text input

Submitting the text field skips ASR entirely and runs SLM → TTS. This keeps the demo
usable when the microphone is blocked or ASR failed to load.

### Memory

A `history` array of `{role, content}`. A user turn is pushed on transcript or typed
submit; an assistant turn on SLM response. Sliced to the last 10 entries when sent. The
clear button empties it and the conversation view together.

### Responsive

Fluid max-width container, fluid spacing. Push-to-talk uses pointer events so it works
with touch, and no interaction depends on hover.

## Error handling

Unchanged in shape. Per-stage failures surface in the status line; a TTS failure still
leaves the AI text visible and skips playback. `/api/status` failing is non-fatal — the
status dot shows unknown and the pipeline behaves as it does today, surfacing 503s when
a stage is actually used.

## Risks

- The `/api/status` endpoint reads the module-level model globals. It reports load
  success, not ongoing health — a model that loaded and later errors still reads as
  available.
- Interface Chinese is machine-authored, not reviewed by a native speaker. It should be
  checked before the demo is shown to a Chinese-speaking audience.

## Out of scope

- Changing the AI's response language.
- Chinese or English ASR — the Whisper hint stays `ms`.
- Server-side session persistence.
