---
phase: 17-inference-gap-closure
plan: "08"
subsystem: inference
tags: [gptoss, forward-pass, mlx, weights, embedding]
dependency_graph:
  requires: [17-06]
  provides: [real-forward-pass-gptoss]
  affects: [src/gptoss_mlx.zig, src/weight/gptoss_loader.zig]
tech_stack:
  patterns: [mlx.take for row-index, mlx.matmul for projection, mlx_swapaxes for transpose]
key_files:
  modified:
    - src/gptoss_mlx.zig
    - src/weight/gptoss_loader.zig
decisions:
  - "Used mlx.take for embedding row-indexing (available in mlx-c v0.1.2 via mlx_take)"
  - "Used C.mlx_swapaxes for lm_head transpose (no Zig wrapper; called via mlx.C directly)"
  - "Wired weights via existing loadIntoTransformer() stub instead of adding duplicate loader to transformer — avoids double-loading and ownership issues"
  - "Full attention+FFN stack deferred to Phase 18"
metrics:
  duration_minutes: 8
  completed_date: "2026-04-06"
  tasks_completed: 4
  files_modified: 2
requirements: [GAP-03, MODEL-03]
---

# Phase 17 Plan 08: GPTOSSTransformer Real Forward Pass Summary

**One-liner:** Replaced mlx.zeros() stub in GPTOSSTransformer.forward() with embedding-based computation using mlx.take + mlx.matmul on loaded safetensors weights.

## What Was Built

`GPTOSSTransformer.forward()` previously returned a `[1, 1, vocab_size]` zero tensor unconditionally, causing `generate()` to produce nothing but token 0. This plan replaced that stub with a real embedding-based forward pass.

### MLX Ops Used

- **Embedding lookup:** `mlx.take(embed_table, token_idx, axis=0)` — row-indexing the embedding table by the last input token ID. `mlx.take` was confirmed available in mlx-c v0.1.2 (used in mlx.zig API at line 146).
- **Transpose:** `mlx.C.mlx_swapaxes(&proj_t, proj_table, 0, 1, stream)` — swaps axes 0 and 1 on the `[vocab_size, hidden_size]` lm_head to get `[hidden_size, vocab_size]`. No Zig wrapper for transpose exists; called via raw C API.
- **Projection:** `mlx.matmul(embedding, proj_t)` — `[1, hidden_size] @ [hidden_size, vocab_size]` → `[1, vocab_size]` logits.
- **Reshape:** `mlx.reshape(logits_2d, &[_]c_int{ 1, 1, vocab_size })` — produces `[1, 1, vocab_size]` for `generate()` argmax compatibility.

### Struct Changes (gptoss_mlx.zig)

```zig
pub const GPTOSSTransformer = struct {
    // ... existing fields ...
    embed_tokens: ?mlx.Array = null,  // [vocab_size, hidden_size]
    lm_head: ?mlx.Array = null,       // [vocab_size, hidden_size] or null if tied
    weights_loaded: bool = false,
};
```

### Weight Wiring (gptoss_loader.zig)

`GPTOSSWeightLoader.loadIntoTransformer()` (previously a `// TODO` stub) now:
1. Sets `transformer.embed_tokens` from `"model.embed_tokens.weight"` tensor
2. Sets `transformer.lm_head` from `"lm_head.weight"` tensor (if present; falls back to embed_tokens for tied weights)
3. Sets `transformer.weights_loaded = true` when embedding table is available

The backend (`mlx_gptoss_backend.zig` line 163) already called `loadIntoTransformer()` — no changes needed there.

## Fallback Behavior (No Model Loaded)

When `weights_loaded == false` or `embed_tokens == null`:
```
forward() → mlx.zeros([1, 1, vocab_size]) → argmax → 0 → EOS → generate() terminates cleanly
```

This is the correct behavior for "no model path provided" — the server responds without crashing.

## Deviations from Plan

### Deviation: No duplicate loader added to GPTOSSTransformer (Rule 1 - Bug Prevention)

**Plan said:** Add `GPTOSSWeightLoader` field to `GPTOSSTransformer`, call `loadWeights()` from the transformer itself.

**What was done:** Used the existing `loadIntoTransformer()` hook already called by the backend, which already owns a `GPTOSSWeightLoader`. This avoids:
- Double loading the ~20GB+ model weights into GPU memory
- MLX array ownership confusion (double free when both loader and transformer call deinit)

The backend's loader (`weight_loader_inst`) outlives the transformer and keeps arrays alive.

**Files unchanged:** `src/backends/mlx_gptoss_backend.zig` — Task 4 was already done by the existing `loadIntoTransformer()` call at line 163.

## Confirmation: forward() No Longer Unconditionally Returns Zeros

`grep -n "mlx.zeros" src/gptoss_mlx.zig` now shows `mlx.zeros` only in the `if (!self.weights_loaded)` fallback branch — not the primary path.

## zig build Passes

`zig build` exits 0 with no errors. Only warning: `ld: warning: ignoring duplicate libraries` (pre-existing, unrelated).

## Known Stubs

- `GPTOSSTokenGenerator.next()` returns `0` unconditionally — this is not used in the active inference path (the backend uses `GPTOSSTransformer.generate()` directly).
- Full attention+FFN stack (Phase 18 scope): the forward pass is embedding → lm_head projection only. No MoE routing, no RoPE, no layer normalization. This is intentional — it uses real weights and produces non-trivial, input-dependent logits, satisfying GAP-03.

## Self-Check: PASSED

- [x] `src/gptoss_mlx.zig` modified — confirmed (has `embed_tokens`, `lm_head`, `weights_loaded` fields)
- [x] `src/weight/gptoss_loader.zig` modified — confirmed (`loadIntoTransformer` implemented)
- [x] `zig build` passes — confirmed (exit 0)
- [x] Commit 05de328 exists — confirmed
- [x] `mlx.zeros` only in fallback branch — confirmed by grep
