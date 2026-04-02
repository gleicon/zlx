---
phase: 07-turboquant-integration
plan: 01
phase_name: TurboQuant Integration
plan_name: TurboQuant Feasibility Spike
subsystem: compression
tags: [compression, turboquant, research, feasibility, metal, kv-cache]
dependencies:
  requires: [06-01]
  provides: []
  affects: []
tech_stack:
  added: []
  patterns: [stub-implementation, graceful-degradation]
key_files:
  created:
    - src/compression/mod.zig
    - src/compression/kv_compressor.zig
    - src/compression/turboquant_stub.zig
    - src/compression/metal_kernels.zig
    - src/compression/RESEARCH.md
  modified:
    - src/main.zig
key_decisions:
  - "DEFER TurboQuant integration to v1.2+ (40+ hours, high risk)"
  - "NO-GO decision for v1.1 - proceed to Phase 8 (Speculative Decoding)"
  - "Stub framework allows future implementation without breaking changes"
  - "Graceful fallback to NoOp compression when TurboQuant unavailable"
metrics:
  duration_minutes: 45
  tasks_completed: 3
  files_created: 5
  files_modified: 1
  tests_added: 20
  lines_of_documentation: 400
---

# Phase 07 Plan 01: TurboQuant Feasibility Spike - Summary

## One-Liner
Stub compression framework with TurboQuant placeholders and comprehensive porting analysis recommending deferral to v1.2+ due to 40+ hour effort and mlx-c upgrade risks.

## What Was Built

### 1. Compression Module Infrastructure (Task 1)
Created a pluggable compression framework that supports multiple backends:

- **`src/compression/mod.zig`** - Module exports for KvCompressor, CompressionType, CompressionConfig
- **`src/compression/kv_compressor.zig`** - Generic interface with:
  - `CompressionType` enum: NoOp, TurboQuant, FutureMethod
  - `CompressionConfig`: bits, adaptive_layers, enabled
  - `KvCompressor` struct with compress/decompress/isEnabled methods
  - NoOp backend fully implemented (default, no compression)
  - TurboQuant returns error.NotImplemented (stub)
  - Comprehensive unit tests (20 tests)

### 2. TurboQuant Stub Implementation (Task 2)
Documented the TurboQuant porting requirements while providing a stub interface:

- **`src/compression/turboquant_stub.zig`** - TurboQuantCompressor with:
  - Configuration validation (3-4 bits)
  - Adaptive layer logic (first/last N layers uncompressed)
  - Compression ratio calculations
  - All operations return error.NotImplemented with helpful messages
  - Extensive documentation of algorithm (PolarQuant, WHT, Lloyd-Max)

- **`src/compression/metal_kernels.zig`** - Metal kernel placeholders:
  - `WhtKernel`: Walsh-Hadamard Transform (serial and parallel)
  - `QuantizeKernel`: Lloyd-Max quantization/dequantization
  - `MetalKernelManager`: Metal dispatch management
  - Documented kernel signatures inferred from Python source

- **`src/compression/RESEARCH.md`** - Comprehensive porting analysis:
  - Source repository analysis (arozanov/turboquant-mlx)
  - Key files to port with effort estimates
  - 3 porting options evaluated (A: C++ custom ops, B: Direct Metal, C: MLX array ops)
  - **Recommendation: DEFER to v1.2+**
  - Rationale: 40-60 hours effort, HIGH risk, adequate alternatives exist

### 3. CLI Integration (Task 3)
Added graceful fallback for TurboQuant CLI flags:

- **`src/main.zig`** modifications:
  - `--turboquant` flag: Enable TurboQuant (EXPERIMENTAL)
  - `--turboquant-bits N`: Set quantization bits (3 or 4, default 4)
  - `--turboquant-adaptive N`: Set adaptive layers (default 4)
  - Input validation for bits (3-4 range)
  - Warning logged when TurboQuant requested but not available
  - Graceful fallback to NoOp compression
  - Updated help text with examples

## Key Decisions

### DECISION: NO-GO on TurboQuant for v1.1
**Status:** ✅ Confirmed after spike

**Rationale:**
1. **High effort:** 40-60 hours minimum for proper implementation
2. **High risk:** Requires mlx-c upgrade from v0.1.2 to v0.4.1+, may break MLX.zig compatibility
3. **Complex porting:** Metal kernels embedded as Python JIT strings need extraction and C++ porting
4. **Adequate alternatives:** Prompt caching (Phase 6) provides good performance gains
5. **Better ROI:** Speculative Decoding (Phase 8) offers clearer 1.5-2.5x speedup path

**Next Steps:**
- Skip to Phase 8 (Speculative Decoding)
- Keep stub framework for v1.2+ implementation if desired
- Update ROADMAP to mark Phase 7 complete (deferred)

## Technical Implementation

### Compression Interface Design
```zig
pub const CompressionType = enum {
    NoOp,         // No compression (default)
    TurboQuant,   // TurboQuant compression (stubbed)
    FutureMethod, // Placeholder for future methods
};

pub const KvCompressor = struct {
    pub fn compress(self: *Self, k: *mlx.Array, v: *mlx.Array) !CompressionResult;
    pub fn decompress(self: *Self, compressed: CompressionResult, k: *mlx.Array, v: *mlx.Array) !void;
    pub fn isEnabled(self: *Self) bool;
};
```

### Graceful Fallback Pattern
When `--turboquant` is specified:
1. CLI accepts the flag and logs: "TurboQuant requested (compression not yet implemented...)"
2. Server creates CompressionConfig with .TurboQuant type
3. On init, TurboQuant returns error.NotImplemented
4. Server catches error, logs warnings, falls back to NoOp
5. Server continues operating with uncompressed KV cache

## Verification

### Build Verification
```bash
$ zig build
# ✓ Clean build

$ ./zig-out/bin/zlx --help | grep turboquant
  --turboquant            Enable TurboQuant KV cache compression (EXPERIMENTAL)
  --turboquant-bits N     Quantization bits: 3 or 4 (default: 4)
  --turboquant-adaptive N Keep first/last N layers in FP16 (default: 4)

$ ./zig-out/bin/zlx --model ./models/nonexistent --turboquant 2>&1 | head -2
info: TurboQuant requested (compression not yet implemented, will use graceful fallback)
error: Model not found at: ./models/nonexistent
```

### Test Results
```bash
$ zig build test
# All 20 compression module tests pass
# NoOp compressor works correctly
# TurboQuant properly returns NotImplemented
```

## Deviations from Plan

### None
Plan executed exactly as written. All tasks completed with expected outputs.

## Known Stubs

These stubs are intentionally not implemented as per the NO-GO decision:

| File | Line | Stub | Reason |
|------|------|------|--------|
| turboquant_stub.zig:96 | init() | error.NotImplemented | Port postponed to v1.2+ |
| turboquant_stub.zig:153 | compress() | error.NotImplemented | Port postponed to v1.2+ |
| turboquant_stub.zig:183 | decompress() | error.NotImplemented | Port postponed to v1.2+ |
| metal_kernels.zig:55 | WhtKernel.applySerial() | error.NotImplemented | Port postponed to v1.2+ |
| metal_kernels.zig:78 | WhtKernel.applyParallel() | error.NotImplemented | Port postponed to v1.2+ |
| metal_kernels.zig:140 | QuantizeKernel.quantize() | error.NotImplemented | Port postponed to v1.2+ |
| metal_kernels.zig:164 | QuantizeKernel.dequantize() | error.NotImplemented | Port postponed to v1.2+ |

These stubs are **documented and tracked** for future implementation. The RESEARCH.md file contains the full porting guide for when this work is prioritized.

## Architecture Notes

### Integration Point with Prompt Cache
The compression module is designed to integrate with `src/cache/prompt_cache.zig`:
- KvCompressor can be instantiated per cached entry
- Compression can be applied during KV persistence
- NoOp default ensures no performance regression

### Future Implementation Path
If TurboQuant is prioritized in v1.2+:
1. Upgrade mlx-c to v0.4.1+ in MLX.zig fork
2. Verify MLX.zig compatibility
3. Implement C++ custom ops wrapper for TurboQuant kernels
4. Add Zig bindings through upgraded mlx-c
5. Replace stubs with actual Metal kernel dispatch

## Metrics

| Metric | Value |
|--------|-------|
| Execution Time | ~45 minutes |
| Tasks Completed | 3/3 |
| Files Created | 5 |
| Files Modified | 1 |
| Tests Added | 20 |
| Documentation Lines | 400+ |
| Go/No-Go Decision | **NO-GO** |

## Commits

- `7145a62`: feat(07-01): create KV compressor interface and compression module
- `6266e50`: feat(07-01): create TurboQuant stub with Metal kernel placeholders
- `31f2ae1`: feat(07-01): add --turboquant CLI flags with graceful fallback

## Next Steps

1. **Update ROADMAP** - Mark Phase 07 complete, proceed to Phase 08
2. **Skip to Phase 8** - Speculative Decoding (clearer implementation path)
3. **Document decision** - Record NO-GO in PROJECT.md or CHANGELOG

## Self-Check: PASSED

All files created and commits verified:
- src/compression/mod.zig ✓
- src/compression/kv_compressor.zig ✓
- src/compression/turboquant_stub.zig ✓
- src/compression/metal_kernels.zig ✓
- src/compression/RESEARCH.md ✓
- .planning/phases/07-turboquant-integration/07-01-SUMMARY.md ✓
- Commits 7145a62, 6266e50, 31f2ae1 ✓

- TurboQuant repository: https://github.com/arozanov/turboquant-mlx
- MLX.zig: https://github.com/jaco-bro/MLX.zig
- mlx-c C API: https://ml-explore.github.io/mlx-c/
- Full porting analysis: `src/compression/RESEARCH.md`
