# Phase 17: Inference Gap Closure - Context

**Gathered:** 2026-04-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Replace every `error.NotImplemented`, zero-stub, hardcoded mock, and silent noop in the
inference pipeline so that Qwen, DeepSeek, and GPT-OSS all produce real output from
`/v1/chat/completions`. Inference correctness is the goal — the build is already clean
(Phase 16 baseline).

Requirements in scope: GAP-01, GAP-02, GAP-03, GAP-04, GAP-05 (partial — see D-04),
GAP-06, GAP-07, MODEL-01, MODEL-02, MODEL-03

</domain>

<decisions>
## Implementation Decisions

### Architecture Principle
- **D-02:** MLX stays directly controlled — no generic backends abstraction layer between
  Zig and Metal. The project objective is MLX-native inference on Apple Silicon. llama.cpp
  is a valid tool for specific model families (MoE/GGUF) but must never become the primary
  routing path or obscure MLX control. The `backends/factory.zig` generic routing layer is
  deferred as a backlog item, not Phase 17 scope.

### Handler-Per-Model Pattern (MLX-Swift style)
- **D-01:** Each model family gets a dedicated handler + backend pair, owned directly by
  `server.zig`. This mirrors the GPT-OSS pattern already in production:
  - Qwen → `inference/mod.zig` → `MLX.zig Transformer` (unchanged, already works)
  - GPT-OSS → `chat_gptoss.zig` → `mlx_gptoss_backend.zig` (tokenizer works, fix `forward()`)
  - DeepSeek → **new** `chat_deepseek.zig` → `llama_cpp.zig` (direct, no factory)
  - Gemma 4 (Phase 18) → `chat_gemma4.zig` → `mlx_gemma4_backend.zig` (same pattern)

  HTTP dispatch in `server.zig` routes by model name prefix:
  `"gptoss"` → GPT-OSS handler, `"deepseek"` → DeepSeek handler, others → Qwen path.

### DeepSeek Wiring
- **D-01 (continued):** Create `chat_deepseek.zig` matching the structure of `chat_gptoss.zig`.
  Add a module-level `LlamaBackend` in `server.zig` alongside `g_gptoss_backend`. Wire
  `inference/mod.zig:261` and `inference/mod.zig:332` (`error.DeepSeekNotImplemented`) to
  route through `chat_deepseek.zig` or remove those branches if the dispatch happens
  at the HTTP layer before mod.zig is reached.

### backends/ Cleanup (GAP-05 partial)
- **D-04:** Delete orphaned stub files — they are never called by any real code path:
  - DELETE: `src/backends/factory.zig` (createMlxBackend returns a u8 stub)
  - DELETE: `src/backends/mlx_backend.zig` (all functions are stubs; Qwen uses mod.zig)
  - KEEP: `src/backends/llama_cpp.zig` (real, used by chat_deepseek.zig)
  - KEEP: `src/backends/mlx_gptoss_backend.zig` (real tokenizer, fix forward())
  - KEEP: `src/backends/backend.zig` (shared types: GenerationParams, BackendCapabilities —
    reused by Gemma 4 in Phase 18 and any future MLX backend)
  - UPDATE: `src/backends/mod.zig` to only re-export surviving files

### Speculative Decoding Removal (GAP-07)
- **D-03:** Remove all speculative decoding code — zero dead code policy. Remove:
  - CLI flags: `--draft-model`, `--no-speculation`, `--speculation-depth`
  - Config fields: `draft_model`, `no_speculation`, `speculation_depth` in config structs
  - Init block at `main.zig:559` (speculation subsystem init)
  - `draft_model: null` pass-through parameters in `mod.zig:280` and `mod.zig:405`
  - Any related metrics endpoint (`/v1/metrics/speculative`)
  Add a single comment at each removal site: `// speculative-decoding: removed — re-evaluate as dedicated phase after core inference is stable`
  Do NOT add to roadmap backlog — evaluate need from scratch when core is clean.

### Prompt Cache parseIndex (GAP-06)
- **D-05:** Fix `cache/prompt_cache.zig:433` — implement real `index.json` disk reads in
  `parseIndex()`. Read the index once at cache init and keep entries in memory. No repeated
  disk I/O per request. Speed is a first-class constraint.

### Hardcoded Values (GAP-02)
- **D-06:** All hardcoded `vocab_size=32000`, `eos_token=2`, `bos_token=1`, `disk=100GB`
  values must be replaced with values read from the model's `config.json`. The surviving
  backends (`llama_cpp.zig` already does this correctly via llama_c API — use as reference).
  `mlx_gptoss_backend.zig` and any new `chat_deepseek.zig` must read from `config.json`.

### Claude's Discretion
- GPT-OSS `forward()` implementation approach — whether to use MLX.zig Transformer
  directly (if architecture matches) or implement tensor ops layer-by-layer. Choose
  whichever produces real logits correctly; researcher should verify against `gptoss_mlx.zig`
  architecture definition.
- Exact `index.json` schema for prompt cache — read the existing file format, implement
  to match.
- Whether `error.DeepSeekNotImplemented` branches in `mod.zig` are reachable after HTTP
  dispatch rerouting — if not, remove the branches entirely rather than patching them.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase Requirements
- `.planning/REQUIREMENTS.md` §v2.0 — GAP-01 through GAP-07, MODEL-01 through MODEL-03
- `.planning/ROADMAP.md` §Phase 17 — Success criteria (6 items)
- `.planning/ROADMAP.md` §Phase 18 — Gemma 4 architecture (ensure backend.zig stays compatible)

### Active Inference Code (do not break)
- `src/inference/mod.zig` — Qwen inference path (working), DeepSeek stubs to remove
- `src/api/server.zig` — HTTP dispatch, GPT-OSS handler init pattern to replicate for DeepSeek
- `src/api/chat_gptoss.zig` — Template for new `chat_deepseek.zig`
- `src/backends/mlx_gptoss_backend.zig` — GPT-OSS backend (tokenize works, fix generate/forward)
- `src/backends/llama_cpp.zig` — DeepSeek backend (real, working — use directly)
- `src/backends/backend.zig` — Shared types (keep, reused by Phase 18)

### Stub Files to Delete
- `src/backends/factory.zig` — delete (stub routing layer)
- `src/backends/mlx_backend.zig` — delete (all stubs, Qwen doesn't use it)

### GPT-OSS Forward Pass
- `src/gptoss_mlx.zig` — current stub returning zeros (fix target)
- `src/gpt_oss.zig` — architecture definition (vocab_size=151936, layer config)
- `src/mlx.zig/src/mlx.zig` — MLX API surface available for tensor ops

### Prompt Cache
- `src/cache/prompt_cache.zig:433` — parseIndex stub (fix target)

### Reference Implementation (architecture style)
- `src/mlx.zig/src/` — MLX-Swift-style handler organization to follow

</canonical_refs>

<code_context>
## Existing Code Insights

### What Works Today (do not regress)
- `inference/mod.zig` → Qwen generates real tokens via `MLX.zig Transformer.init()/generate()`
- `mlx_gptoss_backend.zig:tokenize()` — real implementation using `tokenizer_mod.Tokenizer.init()`
  from `tokenizer.json` in the model directory (GAP-04 may already be resolved)
- `llama_cpp.zig` — `LlamaBackend.init()`, tokenize, generate, vocab_size, eos/bos all real

### Pattern to Replicate (chat_deepseek.zig)
- `src/api/server.zig:31-33` — module-level backend + handler globals for GPT-OSS
- `src/api/server.zig:59-65` — backend init pattern
- `src/api/server.zig:164-171` — HTTP dispatch by model name prefix
- `src/api/chat_gptoss.zig` — handler structure (init, handle, isGptOssModel predicate)

### Integration Points for New Handler
- `server.zig` HTTP dispatch block needs a new `"deepseek"` prefix branch
- `server.zig` init block needs `g_deepseek_backend` + `g_chat_deepseek_handler` globals
- `inference/mod.zig` DeepSeek branches become unreachable after HTTP-layer dispatch
  and should be deleted, not patched

</code_context>

<specifics>
## Specific Implementation Notes

- DeepSeek model names to match in dispatch: check existing registry for exact name patterns
  (`deepseek-coder-v2-lite`, `deepseek-*`, etc.)
- `mlx_gptoss_backend.zig:tokenize()` re-initializes tokenizer on every call — consider
  whether the backend should cache the tokenizer instance (speed optimization, D-05 spirit)
- `backend.zig` defines `GenerationParams` with temperature, top_p, max_tokens — reuse
  these in `chat_deepseek.zig` rather than creating new types
- Prompt cache `parseIndex()` fix: read `index.json` with `std.json`, populate the entries
  map; schema should match what the existing `save()` writes

</specifics>

<deferred>
## Deferred Ideas

- `backends/factory.zig` generic routing abstraction — noted as backlog, evaluate need
  after core inference is clean and library extraction is in progress
- Speculative decoding — removed entirely; re-evaluate from scratch as a dedicated phase
  if the need arises after core stability
- TurboQuant `error.NotImplemented` stubs — Phase 19 scope (not touched here)
- Tools API (`browser.zig`, `python.zig` Zig 0.15.2 breaks) — Phase 20 scope

</deferred>

---

*Phase: 17-inference-gap-closure*
*Context gathered: 2026-04-05*
