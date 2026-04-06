---
phase: 17-inference-gap-closure
plan: 02
subsystem: backends
tags: [cleanup, dead-code-removal, stub-deletion, gap-closure]
dependency_graph:
  requires: [17-01]
  provides: [clean-backends-directory, no-mlx-stub-extern-fns]
  affects: [src/backends/, src/inference/backend_generator.zig, src/inference/generator.zig, src/test_backend_integration.zig]
tech_stack:
  added: []
  patterns: [stub-deletion, comment-at-removal-site]
key_files:
  deleted:
    - src/backends/factory.zig
    - src/backends/mlx_backend.zig
  modified:
    - src/backends/backend.zig
    - src/backends/mod.zig
    - src/inference/backend_generator.zig
    - src/inference/generator.zig
    - src/test_backend_integration.zig
decisions:
  - "D-04 executed: factory.zig and mlx_backend.zig deleted — confirmed pure stubs with no live callers"
  - "D-02 confirmed: MLX inference stays direct (inference/mod.zig + MLX.zig); Backend union is for llama_cpp and mlx_gptoss only"
  - "BackendGenerator stub kept (returns error.NotImplemented) to preserve type surface for Phase 18"
metrics:
  duration: "3 minutes"
  completed: "2026-04-06"
  tasks: 2
  files_modified: 5
  files_deleted: 2
---

# Phase 17 Plan 02: Delete Orphaned Backend Stubs — Summary

Deleted `factory.zig` and `mlx_backend.zig` (confirmed pure stubs), removed all `.mlx` arm references from the Backend union, and updated all callers. `zig build` now reaches the same error count as Phase 16 completion (1 pre-existing error in `chat_gptoss.zig` unrelated to this plan).

## Files Deleted

| File | Reason |
|------|--------|
| `src/backends/factory.zig` | Pure stub — factory abstraction never called by live inference path |
| `src/backends/mlx_backend.zig` | Pure stub — contained hardcoded vocab_size=32000, eos_token=2, bos_token=1 (GAP-02); never called |

## Files Modified

### src/backends/backend.zig
- Removed `BackendType.mlx` from enum
- Removed `BackendPreference.mlx` from enum
- Removed `Backend.mlx: *anyopaque` arm from union
- Removed all extern fn declarations for mlx: `mlxTokenize`, `mlxGenerate`, `mlxDeinit`, `mlxGetVocabSize`, `mlxEosToken`, `mlxBosToken`, `mlxGetKvCache`, `mlxApplyCompression`
- Updated `selectBackend()` — removed `.mlx` return arm
- Added comment at each removal: `// mlx backend: removed stub — Qwen uses inference/mod.zig + MLX.zig directly`

### src/backends/mod.zig
- Removed `pub const factory = @import("factory.zig")`
- Removed `pub const mlx_backend = @import("mlx_backend.zig")`
- Updated `getBackendDescription()` — removed `.mlx` arm
- Updated `supportsFeature()` — removed `.mlx` speculative_decoding path
- Added removal comments at all sites

### src/inference/backend_generator.zig
- Removed `const factory = @import("../backends/factory.zig")`
- Replaced `factory.createBackend(...)` calls with `return error.NotImplemented` stubs
- Changed `preference: factory.BackendPreference` to `preference: backends.BackendPreference`
- BackendGenerator struct preserved as stub type for Phase 18 surface

### src/inference/generator.zig
- Removed `const factory = @import("../backends/factory.zig")` import

### src/test_backend_integration.zig
- Removed `const factory = @import("backends/factory.zig")` import
- Removed all factory-dependent test blocks (detectBackendType, detectModelFormat, getCapabilities)
- Kept: registry tests, GenerationParams tests, CompressionParams tests, backend description/feature tests

## Build Status

`zig build` produces 1 error: `src/api/chat_gptoss.zig:115:20: error: member function expected 0 argument(s), found 1`

This is a **pre-existing error** (present before this plan, outside our scope) in the httpz `res.write()` API signature — tracked for Plan 17-03 or later.

Before this plan: 7 compilation errors (post-17-01 state).
After this plan: 1 compilation error (only the pre-existing chat_gptoss.zig error remains).

## Shared Types Confirmed Preserved in backend.zig

All Phase 18 (Gemma 4 E4B) required types are preserved:
- `GenerationParams` — temperature, top_p, top_k, max_tokens, seed, stop_sequences, penalties
- `StopReason` — eos, length, stop, timeout, error_status
- `TokenResult` — token, text, logprob, finish_reason
- `GenerationResult` — token_iterator, tokens_generated, finish_reason
- `TokenIterator` — next_fn, ctx, deinit_fn function pointer vtable
- `KvCacheHandle` — ptr, backend_type, getRaw(), getType()
- `CompressionParams` — enabled, bits, adaptive_layers, group_size
- `ModelLoadResult` — vocab_size, eos_token, bos_token, context_size, layers
- `BackendCapabilities` — supports_turboquant, supports_speculative_decoding, max_context_length, preferred_quantization
- `BackendStats` — tokens_generated, tokens_per_second, memory_used_mb, kv_cache_size_mb, gpu_layers

## GAP Requirements Closed

- **GAP-05 (D-04)**: factory.zig and mlx_backend.zig deleted — orphaned stub files removed
- **GAP-02**: Hardcoded vocab_size=32000, eos_token=2, bos_token=1 eliminated — these lived exclusively in `mlx_backend.zig` (now deleted)

## Deviations from Plan

### Auto-fixed Issues

None.

### Scope Notes

- The pre-existing `chat_gptoss.zig:115` build error was already present before this plan and is unrelated to factory/mlx_backend cleanup. Deferred to later plan.
- `test_backend_integration.zig` test for `BackendType.mlx` enum ordinal (`@intFromEnum(BackendType.mlx) == 0`) was removed since the `.mlx` variant no longer exists. The test was updated to use `.llama_cpp` and `.mlx_gptoss`.

## Self-Check: PASSED

- `src/backends/backend.zig` — EXISTS, no `.mlx` arm, no extern mlx* fns
- `src/backends/mod.zig` — EXISTS, no factory import, no mlx_backend import
- `src/backends/factory.zig` — DELETED (confirmed)
- `src/backends/mlx_backend.zig` — DELETED (confirmed)
- Commits: 4fa3bed (Task 1), 28f83ba (Task 2) — both confirmed in git log
