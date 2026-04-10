---
phase: 13-deepseek-gptoss-completion
plan: 02
subsystem: inference
tags: [gpt-oss, weight-loading, mxfp4, download]
dependency_graph:
  requires: [13-01]
  provides: [GPT-OSS-LOADING]
  affects: [loader.zig, registry.zig, gpt_oss.zig]
tech_stack:
  added: [gptoss_loader.zig module]
  patterns: [MXFP4 quantization, safetensors index parsing]
key_files:
  created: [src/inference/gptoss_loader.zig]
  modified: [src/inference/loader.zig, src/models/registry.zig, src/gpt_oss.zig, src/api/handlers.zig]
decisions:
  - "MLX handles MXFP4 dequantization natively (v0.18+)",
  - "GPT-OSS uses 24 layers (vs 27 for DeepSeek), 32 experts (vs 64)",
  - "Sliding window + Yarn RoPE are key differentiators for detection",
  - "Weight loading follows DeepSeek pattern: register keys -> load -> map",
  - "All weight structures have deinit() for proper memory cleanup"
metrics:
  duration: "60 minutes"
  completed_date: "2026-04-03"
  lines_added: 572
  lines_modified: 79
  test_coverage: "Unit tests for weight key registration, quantization detection"
---

# Phase 13 Plan 02: GPT-OSS Download and Weight Loading

**One-liner summary:** Implemented complete weight loading for GPT-OSS-20B model, including 11GB download support from HuggingFace, 24-layer architecture with 32 experts, and MXFP4 quantization handling.

## What Was Built

### 1. GPT-OSS Weight Loader (`src/inference/gptoss_loader.zig`)

A complete weight loading implementation:

- **`registerGptOssWeightKeys()`**: Registers all 2,400+ weight keys:
  - 3 embedding/output weights
  - 24 layers × (2 norms + 4 attention projections + 1 router + 32×3 experts)
  - Total: ~2,475 keys (including quantization components)
  
- **`loadGptOssWeights()`**: Loads and maps all weights:
  - Parses safetensors index for multi-shard models
  - Handles both quantized and unquantized weights
  - Dequantizes using appropriate method (affine or MXFP4)
  - Returns populated `GptOssWeights` struct
  
- **`QuantizationConfig.detectFromConfig()`**: Auto-detects format from config.json:
  - `.mxfp4`: Group size 32, handled natively by MLX
  - `.affine_4bit`: Group size 64, uses dequantize.zig
  - `.none`: No quantization

### 2. Integration with Main Loader (`src/inference/loader.zig`)

- Added `gptoss_loader` and `gpt_oss` imports
- Implemented `loadGptOssWeights()` following DeepSeek pattern:
  - Parse safetensors index
  - Detect quantization format from config
  - Delegate to gptoss_loader for actual loading
  - Return `gpt_oss.GptOssWeights`

### 3. Model Registry Updates (`src/models/registry.zig`)

Added GPT-OSS model metadata:
- Architecture: `gpt_oss` in `ModelArchitecture` enum
- Known model: `gpt-oss-20b` with 20B total / 5B active params
- Detection logic: `model_type.contains("gpt_oss")` or (MoE + sliding_window)
- Memory requirements: 11GB with TurboQuant
- Max context: 131,072 tokens

### 4. Weight Structure Updates (`src/gpt_oss.zig`)

Added memory management:
- `GptOssExpert`: Individual expert with gate/up/down projections
- `GptOssLayer`: Added `router` and `experts` fields, removed `moe`
- `GptOssWeights`: Added `deinit()` for cleanup
- `GptOssAttention`: Added `deinit()` for cleanup

### 5. API Handler Updates (`src/api/handlers.zig`)

- Added `gpt_oss` case to architecture switch statement
- Enables API to report correct model type for GPT-OSS models

## Key Design Decisions

### 1. MLX Native MXFP4 Support
GPT-OSS uses MXFP4 (Microscaling FP4) quantization. Modern MLX (v0.18+) handles this natively when loading safetensors. We:
- Detect MXFP4 from `quantization_config.quant_method` in config.json
- Let MLX handle dequantization automatically
- Fall back to custom implementation only if needed

### 2. Weight Key Registration Strategy
GPT-OSS has ~2,475 weight keys (vs ~800 for DeepSeek). We:
- Generate keys programmatically for all 24 layers × 32 experts
- Handle both quantized and unquantized variants
- Store in StringHashMap for O(1) lookup during loading

### 3. Architecture Detection
GPT-OSS is distinguished by:
- `model_type: "gpt_oss"` in config.json
- MoE with `sliding_window` attention (vs DeepSeek's full attention)
- Yarn RoPE with high theta (150,000)

## MXFP4 Quantization

MXFP4 format details:
- **Bits**: 4 bits per value
- **Format**: 3-bit mantissa + 1-bit exponent (microscaling)
- **Groups**: 32 values per group
- **Scaling**: Per-group micro-exponents

MLX handles this transparently when loading from safetensors, so our implementation focuses on:
1. Correct format detection from config
2. Proper group size configuration
3. Fallback handling if MLX doesn't support it

## Deviations from Plan

**None** - Plan executed exactly as written.

All four tasks completed:
1. ✅ Created gptoss_loader.zig with registerGptOssWeightKeys() and loadGptOssWeights()
2. ✅ Integrated GPT-OSS into loader.zig with loadGptOssWeights()
3. ✅ Added GPT_OSS_20B_CONFIG to model registry with hf_url
4. ✅ MXFP4 format detection and handling (via MLX native support)

## Self-Check Results

**Verification:**
- [x] `zig build` completes without errors
- [x] src/inference/gptoss_loader.zig exports registerGptOssWeightKeys() and loadGptOssWeights()
- [x] loader.zig has loadGptOssWeights() function
- [x] model_registry.zig has gpt_oss architecture and GPT-OSS model entry
- [x] handlers.zig handles gpt_oss in architecture switch
- [x] MXFP4 format detected from config.json
- [x] All GPT-OSS weight structures have deinit() methods

## Commits

- `5cf0dd7`: feat(13-02): Create GPT-OSS weight loader module
- `3ce8845`: feat(13-02): Integrate GPT-OSS into main loader
- `8a8ea7c`: feat(13-02): Add GPT-OSS to model registry with download URL

## Next Steps

Plan 13-02 is complete. GPT-OSS model download and weight loading is now fully implemented. Ready for Plan 13-03: Integration testing.
