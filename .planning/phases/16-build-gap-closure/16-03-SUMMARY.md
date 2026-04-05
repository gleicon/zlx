---
phase: 16-build-gap-closure
plan: "03"
subsystem: inference/loader
tags: [deepseek, mla, loader, fix, stub]
dependency_graph:
  requires: [16-01]
  provides: [GAP-10-resolved]
  affects: [src/inference/loader.zig, src/mlx.zig/src/mla.zig]
tech_stack:
  added: []
  patterns: [value-stub, struct-literal-init]
key_files:
  modified:
    - src/inference/loader.zig
decisions:
  - "Approach B (stub) chosen over Approach A (init+dereference) to prevent deinit() double-free on value type"
  - "head_dim computed as hidden_size / num_attention_heads since DeepSeekConfig has no head_dim field"
  - "mla_weights MLAWeights struct discarded with _ = ... since MultiHeadLatentAttention fields do not include a weights aggregate"
metrics:
  duration: "~5 minutes"
  completed: "2026-04-05"
  tasks_completed: 2
  files_modified: 1
---

# Phase 16 Plan 03: MLA Init Site Fix Summary

**One-liner:** Fixed `loader.zig:499` by replacing `.weights`/`.num_heads` struct-literal with correct field layout (`w_dq`, `w_dkv`, `w_up`, `w_kr`, `rope`, `base`, `config`) using stub approach to avoid deinit double-free.

## What Was Done

GAP-10: `loader.zig:499-506` used struct-literal syntax referencing a `.weights` field and `.num_heads` field that do not exist in `mla.MultiHeadLatentAttention`. This plan replaced the broken initialization with a correct stub value that matches the actual struct definition.

### Approach Used

**Approach B — Stub value** (as specified in plan, D-05 acceptable).

Rationale:
- `DeepSeekLayer.mla` is typed as `mla.MultiHeadLatentAttention` (value, not pointer)
- `mla.MultiHeadLatentAttention.init()` returns `!*Self` (heap pointer)
- `mla.deinit()` calls `self.base.allocator.destroy(self)` — designed for heap-allocated structs
- Dereferencing a heap pointer into a value and then calling `deinit()` on it would double-free memory
- Approach B constructs a zero-weight stub value directly, which compiles and is safe to deinit via value semantics

### Exact Lines Changed

**Removed (lines 489-506):**
```zig
// Create MLA weights
const mla_weights = deepseek.MLAWeights{
    .q_proj = q_proj,
    .kv_a_proj_with_mqa = kv_a_proj,
    .kv_b_proj = kv_b_proj,
    .o_proj = o_proj,
    .kv_a_layernorm = kv_a_layernorm,
};

// Initialize MLA (will be completed separately)
const mla_layer = mla.MultiHeadLatentAttention{
    .weights = mla_weights,    // ERROR: no field 'weights'
    .config = mla.MLAConfig{
        .hidden_size = config.hidden_size,
        .num_heads = config.num_attention_heads,  // ERROR: wrong field name
        .latent_dim = config.latent_dim,
    },
};
```

**Added (replacement):**
```zig
// Stub: MLAWeights captured but stored separately (not part of mla.MultiHeadLatentAttention)
_ = deepseek.MLAWeights{ ... };  // suppress unused variable

// Stub MLA value using actual struct field layout
const mla_head_dim = config.hidden_size / config.num_attention_heads;
const mla_config = mla.MLAConfig{
    .hidden_size = config.hidden_size,
    .num_attention_heads = config.num_attention_heads,
    .latent_dim = config.latent_dim,
    .head_dim = mla_head_dim,
};
const mla_layer = mla.MultiHeadLatentAttention{
    .base = mlx.Module.init(allocator, mlx.C.mlx_default_gpu_stream_new()),
    .config = mla_config,
    .w_dq = mlx.arrayNew(),
    .w_dkv = mlx.arrayNew(),
    .w_up = mlx.arrayNew(),
    .w_kr = null,
    .rope = null,
};
```

## Verification Results

### GAP-10 criteria satisfied:

1. `grep -n "\.weights" src/inference/loader.zig` — **empty** (no .weights field reference)
2. `zig build 2>&1 | grep "no field named 'weights'"` — **empty**
3. `zig build 2>&1 | grep "loader.zig:499"` — **empty**

### zig build output (tail -20):

```
src/inference/dequantize.zig:101:47: error: expected type '[]const i64', found '*const [*c]const c_int'
src/inference/loader.zig:542:40: error: missing struct field: num_shared_experts
error: the following command failed with 2 compilation errors:
Build Summary: 6/9 steps succeeded; 1 failed
```

**Error count:** 2

### zig build test output (tail -20):

```
test transitive failure
+- run test registry_test 7/7 passed, 2 leaked
+- run test models_test 10/10 passed, 4 leaked
+- run test backends_test transitive failure (pcre2.h not found)
+- run test backend_integration_test transitive failure
+- run test moe_test transitive failure
+- run test deepseek_test transitive failure
+- run test integration_test transitive failure
+- run test gptoss_test transitive failure
+- run test gptoss_manager_test transitive failure
48/48 tests passed (subset); 6 leaked
```

## Remaining Errors for Phase 17 Triage

| Error | File | Line | Description | Scope |
|-------|------|------|-------------|-------|
| Type mismatch | `src/inference/dequantize.zig` | 101 | `'[]const i64'` vs `'*const [*c]const c_int'` passed to `calculateNumElements` | Out of scope for 16-03 |
| Missing field | `src/inference/loader.zig` | 542 | `moe.MoEConfig` missing `num_shared_experts` in dense layer init | Out of scope for 16-03 (pre-existing) |
| pcre2.h not found | `src/mlx.zig/src/regex.zig` | 6 | C import fails in test builds — pcre2 include path issue | Out of scope |

## D-05 Success Criterion

**NOT met** — `zig build` still produces 2 errors (both pre-existing, outside Plan 03 scope).

Plan 03 scope (GAP-10: loader.zig MLA init site) is **complete** — the MLA type errors are eliminated.
Phase 16 completion requires Plans 01+02+03 combined; the 2 remaining errors are for Phase 17.

## Deviations from Plan

None — plan executed exactly as written (Approach B chosen as specified).

## Known Stubs

| File | Location | Description | Future Plan |
|------|----------|-------------|-------------|
| `src/inference/loader.zig` | ~499-520 | MLA stub uses empty arrays; real weights (q_proj, kv_a_proj, etc.) are discarded | Phase 17 (inference gap closure) will wire actual weight loading |

## Self-Check: PASSED

- [x] `src/inference/loader.zig` modified and committed (5dffaca)
- [x] `grep "\.weights" src/inference/loader.zig` returns empty
- [x] `zig build` no longer errors on `loader.zig:499` MLA init site
- [x] SUMMARY.md created at `.planning/phases/16-build-gap-closure/16-03-SUMMARY.md`
