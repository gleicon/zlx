# Phase 18: Gemma 4 E4B - Context

**Gathered:** 2026-04-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Add Gemma 4 E4B (coding-tuned variant) as a supported model following the handler-per-model
pattern established in Phase 17. The inference layer is already stable (Phase 17 baseline).

Requirements in scope: MODEL-04, MODEL-05, MODEL-06, MODEL-07

**Target model:** Coding-assistant-tuned Gemma 4 E4B 4-bit (MLX quantized format, ~5GB VRAM).
Per the antoniosantos.io article (see canonical refs), this is the primary target — not the
base `gemma-4-e4b-it-4bit`. Chat template and tokenizer wiring must match the coding-tuned variant.

</domain>

<decisions>
## Implementation Decisions

### Backend Runtime
- **D-01:** Use llama.cpp as the backend for Gemma 4 E4B. llama.cpp already has Gemma 4 support
  compiled (`gemma4-iswa.cpp.o` in `src/llama.cpp/build/`). Follow the DeepSeek pattern exactly:
  - `chat_gemma4.zig` wraps `src/backends/llama_cpp.zig` (`LlamaBackend`)
  - `mlx_gemma4_backend.zig` is NOT created — the handler calls `llama_cpp.zig` directly,
    same as `chat_deepseek.zig`
  - Wire into `server.zig` with a `"gemma4"` prefix dispatch branch, alongside DeepSeek and GPT-OSS

### Handler Pattern (mirrors DeepSeek)
- **D-02:** Create `src/api/chat_gemma4.zig` matching the structure of `src/api/chat_deepseek.zig`:
  - Module-level `LlamaBackend` in `server.zig` (e.g., `g_gemma4_backend`)
  - Module-level handler global (e.g., `g_chat_gemma4_handler`)
  - `isGemma4Model(name)` predicate matching `"gemma4"` prefix
  - HTTP dispatch in `server.zig` routes `"gemma4"` prefix to `chat_gemma4.zig`
  - `backend.zig` `GenerationParams` reused — no new types

### Chat Template
- **D-03:** Implement the Gemma 4 chat template as static Zig string constants in `chat_gemma4.zig`.
  No Jinja parser, no file I/O at request time. Key tokens (verified from `src/llama.cpp/models/templates/gemma4.jinja`):
  - Turn start: `<|turn>{role}\n`
  - Turn end: `<turn|>\n`
  - BOS: model-specific (read from llama.cpp backend, not hardcoded)
  - Generation prompt suffix (no-think): `<|channel>thought\n<channel|>` (appended after `<|turn>model\n`)
  - System turn: `<|turn>system\n{content}<turn|>\n`
  - User turn: `<|turn>user\n{content}<turn|>\n`
  - Assistant turn: `<|turn>model\n{content}<turn|>\n`

### No-Think / Verbosity Mode
- **D-04:** Always-on for all Gemma 4 requests. The handler ALWAYS:
  1. Appends `<|channel>thought\n<channel|>` to the generation prompt (suppresses think channel)
  2. Sets `temperature = 0.3` if the request does not specify a temperature (don't override explicit user values)
  This matches the coding-assistant use case with no API schema changes. Log it at model load time:
  `[gemma4] verbosity reduction: always-on (no-think mode, temperature floor 0.3)`

### Model Directory Convention
- **D-05:** Local model directory: `./models/gemma4-e4b/` (consistent with existing `./models/<name>/` convention).
  Model name prefix for dispatch: `"gemma4"` (matches `--model gemma4-e4b` or any `gemma4-*` name).
  Researcher should confirm exact HuggingFace model ID for the coding-tuned variant from the
  antoniosantos.io article (see canonical refs) — this determines what users `git lfs pull` or
  `huggingface-cli download` to populate `./models/gemma4-e4b/`.

### Claude's Discretion
- Exact GGUF quantization format for Gemma 4 E4B — confirm whether the coding-tuned variant
  ships as GGUF (for llama.cpp) or MLX safetensors. If it ships only as MLX safetensors,
  researcher must flag this and evaluate whether llama.cpp can convert or if backend choice
  must be revisited.
- Whether `chat_deepseek.zig` already exists and is fully wired (Phase 17 should have created it)
  — if so, `chat_gemma4.zig` is a near-copy with only template tokens changed.
- BOS token injection — llama.cpp tokenizer may prepend BOS automatically; verify to avoid
  double-BOS in the rendered prompt.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase Requirements
- `.planning/REQUIREMENTS.md` §v2.0 — MODEL-04, MODEL-05, MODEL-06, MODEL-07
- `.planning/ROADMAP.md` §Phase 18 — Success criteria (4 items)

### Handler Pattern to Replicate
- `src/api/chat_deepseek.zig` — primary template for `chat_gemma4.zig` (llama.cpp backend path)
- `src/api/chat_gptoss.zig` — secondary reference (GPT-OSS handler structure)
- `src/api/server.zig` — HTTP dispatch block (add `"gemma4"` prefix branch alongside `"deepseek"`)

### Backend
- `src/backends/llama_cpp.zig` — `LlamaBackend` (real, working — use directly, same as DeepSeek)
- `src/backends/backend.zig` — `GenerationParams` shared types (reuse without changes)

### Chat Template Source of Truth
- `src/llama.cpp/models/templates/gemma4.jinja` — Gemma 4 chat template (control tokens, no-think logic)

### Target Model (coding-tuned variant)
- https://antoniosantos.io/posts/gemma-4-e4b-local-coding-assistant/ — researcher must read
  this to determine exact HuggingFace model ID, quantization format (GGUF vs MLX safetensors),
  and any model-specific tokenizer requirements

### Phase 17 Context (baseline)
- `.planning/phases/17-inference-gap-closure/17-CONTEXT.md` — D-01 establishes handler-per-model
  pattern this phase follows; D-04 lists surviving backend files

</canonical_refs>

<code_context>
## Existing Code Insights

### Pattern to Replicate (chat_gemma4.zig)
- `server.zig` dispatch block: add `"gemma4"` prefix branch after `"deepseek"` branch
- `server.zig` init block: add `g_gemma4_backend: LlamaBackend` + `g_chat_gemma4_handler` globals
- `chat_deepseek.zig`: near-copy target — only chat template tokens change

### llama.cpp Gemma 4 Build Status
- `src/llama.cpp/build/src/CMakeFiles/llama.dir/models/gemma4-iswa.cpp.o` — compiled
- `src/llama.cpp/build/bin/llama-gemma3-cli` — Gemma 3 CLI (Gemma 4 may need separate invocation)
- Verify whether `LlamaBackend.init()` in `llama_cpp.zig` auto-detects Gemma 4 from the GGUF
  metadata or requires an explicit `model_type` hint

### Tokenizer Compatibility
- The existing `tokenizer_mod.Tokenizer` from MLX.zig is BPE-based (`tokenizer.json` format)
- Gemma 4 uses a different tokenizer format — llama.cpp backend handles tokenization internally
  via `llama_tokenize()` C API, so no Zig tokenizer changes are needed (same as DeepSeek path)

### No-Think Token Clarification
- REQUIREMENTS say `<|think|>` block suppression, but the actual Gemma 4 template uses
  `<|channel>thought\n<channel|>` as the no-think generation suffix (verified from gemma4.jinja)
- REQUIREMENTS MODEL-06 description is approximate — implement per the actual jinja template

</code_context>

<specifics>
## Specific Implementation Notes

- Gemma 4 control tokens from jinja template: `<|turn>`, `<turn|>`, `<|channel>`, `<channel|>`,
  `<|think|>`, `<|tool>`, `<tool|>`, `<|image|>`, `<|audio|>`, `<|video|>`
  (image/audio/video out of scope for Phase 18 — text-only)
- No-think suffix for generation prompt: `<|channel>thought\n<channel|>` — appended after
  `<|turn>model\n` to suppress the thinking channel
- Temperature floor: 0.3 default if request doesn't set temperature explicitly
- Model name matching: `isGemma4Model` should return true for `"gemma4-e4b"`, `"gemma4-*"` variants
- Log at startup: `[gemma4] loaded {model_path}, verbosity reduction: always-on`

</specifics>

<deferred>
## Deferred Ideas

- Multimodal inputs (image, audio, video) — Gemma 4 architecture supports these but out of scope
  for Phase 18; note for a future phase if multimodal use cases arise
- Per-request no-think toggle (`no_think: bool` in request body) — always-on is sufficient for
  coding assistant use; add toggle only if needed after testing
- Base Gemma 4 E4B (non-coding-tuned) support — same architecture, different weights; trivially
  addable by registering a second model path if user wants both variants

</deferred>

---

*Phase: 18-gemma-4-e4b*
*Context gathered: 2026-04-07*
