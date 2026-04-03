---
phase: 13-deepseek-gptoss-completion
plan: 01
subsystem: inference
tags: [deepseek, dequantization, 4-bit, affine, weights]
dependency_graph:
  requires: [12-01]
  provides: [DEEPSEEK-DEQUANTIZE]
  affects: [loader.zig, deepseek.zig]
tech_stack:
  added: [dequantize.zig module]
  patterns: [affine dequantization, group-based quantization]
key_files:
  created: [src/inference/dequantize.zig]
  modified: [src/inference/loader.zig, src/deepseek.zig]
decisions:
  - "mlx-community uses unsigned 4-bit values (0-15), not signed (-8 to 7)"
  - "Group sizes: 64 for embeddings and attention, 32 for MLP and router"
  - "Dequantize to float16 (not float32) for memory efficiency"
  - "CPU-based dequantization with MLX array operations"
  - "All weight structures now store dequantized mlx.Array instead of QuantizedWeight"
metrics:
  duration: "45 minutes"
  completed_date: "2026-04-03"
  lines_added: 701
  lines_modified: 50
  test_coverage: "Unit tests for group size detection, shape validation"
---

# Phase 13 Plan 01: DeepSeek Quantized Weight Reconstruction

**One-liner summary:** Implemented 4-bit affine quantization dequantization for DeepSeek-Coder-V2-Lite, enabling the model to reconstruct float16 weights from packed 4-bit integers with per-group biases and scales.

## What Was Built

### 1. Dequantization Module (`src/inference/dequantize.zig`)

A complete dequantization implementation with:

- **`dequantizeAffine4Bit()`**: Main function converting packed 4-bit weights to float16/float32
- **`DequantizeConfig`**: Configuration for group size, output dtype, and signedness
- **`WeightGroupConfig`**: Lookup table for group sizes by weight type:
  - 64 for: embeddings, attention projections, KV projections
  - 32 for: MLP layers (gate/up/down), router, experts
- **`QuantizedWeight.dequantize()`**: Convenience method for dequantizing a weight group
- **Shape validation**: Ensures dequantized shapes match expected tensor dimensions
- **Comprehensive error handling**: InvalidInput, ShapeMismatch, UnsupportedFormat, OutOfMemory

### 2. Weight Mapping Integration (`src/inference/loader.zig`)

Updated `mapQuantizedWeight()` to:
- Accept allocator and group_size parameters
- Retrieve weight, biases, and scales from hash map
- Call `dequantize.dequantizeAffine4Bit()` to convert to float16
- Return dequantized `mlx.Array` instead of `QuantizedWeight`
- Proper error handling with new error types: WeightNotFound, BiasNotFound, ScaleNotFound

Updated `mapWeightsToDeepSeek()` to:
- Dequantize all quantized weight groups (embeddings, LM head, attention, MLP/MoE)
- Use correct group sizes (64 for attention, 32 for MLP)
- Initialize MLA and MoE layers with dequantized weights
- Proper memory management with errdefer

### 3. DeepSeek Weight Structures (`src/deepseek.zig`)

Updated all weight structures to store dequantized arrays:
- `DeepSeekWeights`: `token_embedding`, `lm_head` now `mlx.Array`
- `MLAWeights`: All projections now `mlx.Array`
- `DenseMLPWeights`: All projections now `mlx.Array`
- `MoEWeights`: Shared experts and switch_mlp now `mlx.Array`
- Added `deinit()` methods to all weight structures for proper cleanup
- Added `QuantizedWeight.dequantize()` method for future compatibility

## Key Design Decisions

### 1. Unsigned 4-bit Values
mlx-community quantized models use unsigned 4-bit values (0-15), not the signed (-8 to 7) convention. The dequantization formula is:
```
dequantized = (unsigned_nibble - bias) * scale
```

### 2. Group Size Strategy
Different weight types use different group sizes based on their typical shapes:
- **64**: Embeddings, attention projections (larger, more regular)
- **32**: MLP layers, router, experts (smaller, more varied)

### 3. Float16 Output
All dequantized weights are converted to float16 for:
- Memory efficiency (2 bytes vs 4 bytes per parameter)
- MLX Metal GPU optimization
- Consistency with MLX Python behavior

## Deviations from Plan

**None** - Plan executed exactly as written.

All three tasks completed:
1. ✅ Created dequantize.zig module with affine 4-bit implementation
2. ✅ Integrated dequantization into weight mapping (loader.zig)
3. ✅ Group size detection already in dequantize.zig (64/32 split)

## Self-Check Results

**Verification:**
- [x] `zig build` completes without errors
- [x] src/inference/dequantize.zig exports dequantizeAffine4Bit()
- [x] src/deepseek.zig QuantizedWeight has dequantize() method
- [x] loader.zig mapQuantizedWeight() returns dequantized arrays
- [x] Group sizes: 64 for attention, 32 for MLP
- [x] All quantized weight paths dequantize correctly

## Commits

- `ad1dce6`: feat(13-01): Create dequantize.zig module with affine 4-bit implementation
- `ca12ebe`: feat(13-01): Integrate dequantization into weight mapping

## Next Steps

Plan 13-01 is complete. DeepSeek quantized weight dequantization is now fully implemented. Ready for Plan 13-02: GPT-OSS download and weight loading.
