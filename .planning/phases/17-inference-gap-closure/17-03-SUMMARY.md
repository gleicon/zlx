---
phase: 17-inference-gap-closure
plan: 03
subsystem: api
tags: [deepseek, llama-cpp, gguf, inference, dispatch, handler-per-model]

requires:
  - phase: 17-inference-gap-closure/17-02
    provides: orphaned backend stubs removed, clean llama.cpp backend path
  - phase: 14
    provides: LlamaBackend (llama_cpp.zig) with tokenize/decode/sample/vocab accessors

provides:
  - ChatDeepSeekHandler wired into server.zig HTTP dispatch on deepseek prefix
  - llama.cpp vocab API updated to current (llama_model_get_vocab, llama_vocab_*)
  - ggml sub-libraries linked in build.zig — zig build now exits 0
  - DeepSeekNotImplemented removed from inference path

affects: [17-04, 17-05, phase-18]

tech-stack:
  added: []
  patterns:
    - "Handler-per-model (D-01): server.zig owns globals, dispatches by model name prefix"
    - "Lazy backend init: ChatDeepSeekHandler.loaded=false until first request with non-empty model_path"
    - "Vocab API migration: all llama.cpp vocab/token calls go through llama_model_get_vocab()"

key-files:
  created:
    - src/api/chat_deepseek.zig
  modified:
    - src/api/server.zig
    - src/backends/llama_c.zig
    - src/backends/llama_cpp.zig
    - build.zig

key-decisions:
  - "D-01 applied: no factory/registry — server.zig dispatches directly to ChatDeepSeekHandler via g_chat_deepseek_handler global"
  - "D-06 applied: vocab_size, eos_token, bos_token read from LlamaBackend accessors, never hardcoded"
  - "ggml sub-library linking added to build.zig — libllama.a requires libggml.a, libggml-base.a, libggml-cpu.a, libggml-blas.a, libggml-metal.a"
  - "llama.cpp API migration: seed moved out of context_params; llama_tokenize/llama_token_get_text take llama_vocab* not llama_model*; llama_sample_token replaced by llama_sampler_sample"

requirements-completed: [GAP-03, MODEL-03, MODEL-02]

duration: 35min
completed: 2026-04-06
---

# Phase 17 Plan 03: DeepSeek Handler Wiring Summary

**ChatDeepSeekHandler wired into server.zig dispatch on `deepseek` prefix using LlamaBackend (llama.cpp), with llama.cpp vocab API migrated to current and ggml sub-libraries linked — zig build exits 0**

## Performance

- **Duration:** ~35 min
- **Started:** 2026-04-06T00:30:00Z
- **Completed:** 2026-04-06T01:05:00Z
- **Tasks:** 2 (Task 1 pre-completed; Task 2 executed here)
- **Files modified:** 5

## Accomplishments

- Added `deepseek` dispatch branch in `handleChatCompletions()` — requests with model prefix `deepseek` now route to `ChatDeepSeekHandler` instead of falling through to the generic inference path
- Migrated `llama_cpp.zig` and `llama_c.zig` to the current llama.cpp vocab API (seed removed from context params, `llama_tokenize`/`llama_token_get_text` now take `llama_vocab*`, sampling uses `llama_sampler_sample`)
- Fixed `build.zig` to link all required ggml sub-libraries (`libggml.a`, `libggml-base.a`, `libggml-cpu.a`, `libggml-blas.a`, `libggml-metal.a`) — resolves 253 linker errors that were masked by a pre-existing tools.zig compilation error
- `zig build` now exits with code 0

## Task Commits

1. **Task 1: Create ChatDeepSeekHandler** - `0e199f6` (feat) — pre-completed
2. **Task 2: Wire handler into server.zig** - `57f8f27` (feat)

## Files Created/Modified

- `src/api/chat_deepseek.zig` — ChatDeepSeekHandler with LlamaBackend wiring, DeepSeek instruct prompt format, lazy init pattern
- `src/api/server.zig` — `deepseek` dispatch branch added to `handleChatCompletions()`; `g_deepseek_backend` and `g_chat_deepseek_handler` globals declared and initialized
- `src/backends/llama_c.zig` — Added `llama_model_get_vocab`, `llama_vocab_n_tokens`, `llama_vocab_bos`, `llama_vocab_eos`, `llama_sampler_sample`; removed `llama_sample_token`
- `src/backends/llama_cpp.zig` — Updated to use vocab API; added `vocab` field to `LlamaBackend`; fixed `sample()` to use `llama_sampler_sample`; fixed missing `.{}` on log call
- `build.zig` — Added ggml sub-library paths and link directives

## Decisions Made

- Used `std.mem.startsWith(u8, model, "deepseek")` as the dispatch predicate — matches all deepseek-* variants
- Stored `vocab: *const llama_c.llama_vocab` in `LlamaBackend` struct to avoid re-fetching on every tokenize/sample call; vocab pointer is owned by the model and lives as long as the model
- Kept `seed = 42` as a hardcoded default in the struct (not exposed at API level) since llama.cpp removed seed from context params; seed is passed via sampler chain if needed

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed res.write() called with argument**
- **Found during:** Task 2 (zig build)
- **Issue:** `chat_deepseek.zig:76` called `res.write("{...}")` but httpz `write()` takes no arguments in this version
- **Fix:** Changed to `res.writer().writeAll("{...}")` — matches pattern used elsewhere in chat_deepseek.zig
- **Files modified:** src/api/chat_deepseek.zig
- **Verification:** zig build compilation error cleared
- **Committed in:** 57f8f27 (Task 2 commit)

**2. [Rule 1 - Bug] Migrated llama.cpp vocab API (seed, n_vocab, eos, bos, tokenize, token_text)**
- **Found during:** Task 2 (zig build)
- **Issue:** `llama_cpp.zig` used removed/deprecated APIs: `ctx_params.seed` (removed), `llama_n_vocab(model)` / `llama_token_eos(model)` / `llama_token_bos(model)` (now take `llama_vocab*`), `llama_tokenize(model, ...)` (now takes `llama_vocab*`), `llama_token_get_text(model, ...)` (now takes `llama_vocab*`), `llama_sample_token` (removed; replaced by `llama_sampler_sample`)
- **Fix:** Added `llama_model_get_vocab()` call in `init()`; stored `vocab` field; updated all call sites; added new API exports to `llama_c.zig`; updated `sample()` to use `llama_sampler_sample(sampler, ctx, -1)`
- **Files modified:** src/backends/llama_cpp.zig, src/backends/llama_c.zig
- **Verification:** All compilation errors cleared
- **Committed in:** 57f8f27 (Task 2 commit)

**3. [Rule 3 - Blocking] Added ggml sub-library linking to build.zig**
- **Found during:** Task 2 (zig build linker stage)
- **Issue:** 253 `undefined symbol: _ggml_*` linker errors. `build.zig` only linked `libllama.a` from `build/src/` but not the ggml sub-libraries in `build/ggml/src/` that libllama.a depends on. This was masked by a pre-existing `tools.zig` compilation error; once tools.zig compiled, the linker ran and exposed the missing symbols.
- **Fix:** Added `addLibraryPath` and `linkSystemLibrary` for ggml, ggml-base, ggml-cpu, ggml-blas (for BLAS backend), and ggml-metal in build.zig
- **Files modified:** build.zig
- **Verification:** `zig build` exits 0; all 253 linker errors resolved
- **Committed in:** 57f8f27 (Task 2 commit)

---

**Total deviations:** 3 auto-fixed (1 bug, 1 bug [API migration], 1 blocking)
**Impact on plan:** All auto-fixes necessary for correctness. The llama.cpp API migration was caused by the version of llama.cpp in the submodule having moved past the API assumptions in the original code. The ggml linking gap was a pre-existing build.zig omission exposed by this plan reaching the linker stage for the first time.

## Issues Encountered

- `llama_sample_token_greedy` was also gone from the newer llama.cpp but had already been removed from actual call sites — only the export alias in `llama_c.zig` remained; removed the alias to avoid confusion

## Next Phase Readiness

- DeepSeek HTTP dispatch is wired and compiles; runtime behavior requires a real GGUF model loaded via `/v1/models/switch`
- `zig build` is now clean — Phase 17-04 and 17-05 can proceed without build blockers
- GAP-03 (DeepSeek dispatch), MODEL-02, MODEL-03 requirements closed

---
*Phase: 17-inference-gap-closure*
*Completed: 2026-04-06*
