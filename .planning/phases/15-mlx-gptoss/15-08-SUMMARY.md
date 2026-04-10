---
phase: 15-mlx-gptoss
plan: "08"
subsystem: inference
tags: [mlx, gptoss, inference, stubs, gap-closure]
dependency_graph:
  requires: ["15-05", "15-06"]
  provides: ["GPTOSSTransformer with argmax sampling", "MLXGPTOSSBackend with working tokenizer and GPU stream"]
  affects: ["src/gptoss_mlx.zig", "src/backends/mlx_gptoss_backend.zig"]
tech_stack:
  added: []
  patterns:
    - "mlx.zeros() with c_int shape for logit tensor allocation"
    - "mlx.argmax + mlx.item for greedy token sampling"
    - "Tokenizer.init + encode() delegation pattern"
    - "defaultGpuStreamNew() for GPU stream lifecycle"
key_files:
  created: []
  modified:
    - src/gptoss_mlx.zig
    - src/backends/mlx_gptoss_backend.zig
decisions:
  - "Used mlx.FLOAT32 (confirmed as `pub const FLOAT32: C.mlx_dtype = C.MLX_FLOAT32` in mlx.zig line 19)"
  - "argmax axis=2 chosen because forward() returns [1, 1, vocab_size] shape — vocab is the third dimension"
  - "streamFree added to both deinit() and unload() since GPTOSSTransformer.deinit() does _ = self; (no-op)"
  - "src/mlx.zig/src/gptoss.zig intentionally NOT modified — backend imports from outer repo src/gptoss_mlx.zig"
  - "generate() now terminates on token==0 (EOS); with zero-logit forward(), argmax on zeros returns 0 immediately"
metrics:
  duration: "~10 minutes"
  completed: "2026-04-04"
  tasks_completed: 2
  files_modified: 2
---

# Phase 15 Plan 08: Replace Inference Stubs with Real MLX Primitive Ops

One-liner: Replaced four inference stubs in gptoss_mlx.zig and mlx_gptoss_backend.zig with real mlx-c v0.1.2 primitive ops (zeros, argmax, item, defaultGpuStreamNew, Tokenizer.encode).

## What Was Changed

### Task 1: src/gptoss_mlx.zig

**forward() — lines 98-103 (was stub)**

Before:
```zig
const shape = &[_]i32{ 1, 1, @intCast(self.config.vocab_size) };
return mlx.arrayZeros(3, shape, mlx.Float32);
```

After:
```zig
var logits = mlx.arrayNew();
const shape = [_]c_int{ 1, 1, @intCast(self.config.vocab_size) };
try mlx.zeros(&logits, &shape, mlx.FLOAT32, self.stream);
return logits;
```

Key findings:
- `mlx.arrayZeros` does not exist in the API — the real function is `mlx.zeros(result *Array, shape []const c_int, dtype, stream)`
- Shape elements must be `c_int` not `i32`
- `mlx.FLOAT32` is the correct constant name (confirmed: `pub const FLOAT32: C.mlx_dtype = C.MLX_FLOAT32` in src/mlx.zig/src/mlx.zig line 19)
- `mlx.Float32` (with capital F) does not exist

**generate() — lines 105-122 (was stub cycling i % vocab_size)**

Before: pre-allocated fixed array, set `output[i] = i % vocab_size`, never called forward().

After: `std.ArrayList(u32)` accumulation loop that:
1. Calls `self.forward(context.items)` each iteration
2. Uses `mlx.argmax(&token_arr, logits, 2, false, self.stream)` — axis=2 targets vocab dim of [1,1,vocab_size]
3. Uses `mlx.item(&next_token, token_arr)` to extract u32 scalar
4. Breaks on EOS (token == 0)
5. Extends context for next step
6. Returns `output.toOwnedSlice()`

Note on behavior: Since forward() has no weight fields yet, it returns an all-zeros logit tensor. argmax on zeros deterministically returns index 0 (EOS). This means generate() always returns a single EOS token — which is architecturally correct termination rather than an infinite stub loop. Weights will be wired in a future phase.

### Task 2: src/backends/mlx_gptoss_backend.zig

**New imports added at top of file:**
```zig
const tokenizer_mod = @import("../mlx.zig/src/tokenizer.zig");
const mlx_api = @import("../mlx.zig/src/mlx.zig");
```

**load() — stream fix (line 120-124)**

Before: `undefined` passed as stream argument to GPTOSSTransformer.init.

After:
```zig
const gpu_stream = mlx_api.defaultGpuStreamNew();
self.transformer = try GPTOSSTransformer.init(self.allocator, gptoss_config, gpu_stream);
```

**tokenize() — replaced empty-alloc stub (lines 275-280)**

Before:
```zig
_ = ptr; _ = text;
return allocator.alloc(u32, 0);
```

After: Initializes `tokenizer_mod.Tokenizer` with the backend's model_path, calls `tokenizer.encode(text)`, and returns the owned slice via `@constCast`. No dupe needed since `encode()` uses `toOwnedSlice(allocator)` where allocator is the same allocator passed to `Tokenizer.init`.

**deinit() and unload() — stream cleanup added**

GPTOSSTransformer.deinit() is a no-op (`_ = self`). Added `mlx_api.streamFree(t.stream)` before `t.deinit()` in both `deinit()` and `unload()` to prevent GPU stream leak.

## Submodule Status

`src/mlx.zig/src/gptoss.zig` was intentionally NOT modified. The backend (mlx_gptoss_backend.zig line 5) imports from `../gptoss_mlx.zig` (outer repo file), not from the submodule. Modifying the submodule file would have no effect on the running backend.

Confirmed via: `git diff --name-only HEAD src/mlx.zig/src/gptoss.zig` returns no output.

## Deviations from Plan

None — plan executed exactly as written. The only additions beyond the plan spec were:
- Stream cleanup added to `unload()` (not just `deinit()`) for consistency — this is a correctness requirement (Rule 2), not a feature addition.

## Self-Check: PASSED

- src/gptoss_mlx.zig: FOUND
- src/backends/mlx_gptoss_backend.zig: FOUND
- 15-08-SUMMARY.md: FOUND
- Commit ac30348 (feat(15-08)): FOUND
