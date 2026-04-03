---
phase: 14-deepseek-gptoss-integration
plan: 01
subsystem: backends
completed: 2026-04-03
---

# Phase 14 Plan 01: Backend Abstraction Layer

## Summary

Created a unified backend abstraction layer that enables zlx to use either MLX.zig or llama.cpp for inference, with automatic backend selection based on model architecture.

## Key Achievements

1. **Backend Interface Types** (`src/backends/backend.zig`)
   - `BackendType` enum: mlx, llama_cpp
   - `GenerationParams` with temperature, top_p, top_k, max_tokens, seed
   - `GenerationResult` with token iterator
   - `TokenIterator` for streaming generation
   - `KvCacheHandle` for TurboQuant integration
   - `CompressionParams` for KV cache compression

2. **Factory Pattern** (`src/backends/factory.zig`)
   - `detectBackendType()`: Maps architectures to optimal backends
   - `createBackend()`: Creates appropriate backend with auto/manual selection
   - `detectModelFormat()`: Identifies GGUF vs safetensors
   - Backend routing: Qwen/Phi → MLX, DeepSeek/GPT-OSS/Llama → llama.cpp

3. **Module Structure**
   - `src/backends/mod.zig`: Module entry point with exports
   - `src/backends/backend.zig`: Core interface types
   - `src/backends/factory.zig`: Backend creation and routing
   - `src/backends/mlx_backend.zig`: MLX placeholder
   - `src/backends/llama_cpp.zig`: llama.cpp placeholder

## Backend Routing Logic

| Architecture | Backend | Reason |
|--------------|---------|--------|
| qwen | MLX.zig | Proven stable, optimized |
| phi | MLX.zig | Works well with MLX |
| deepseek_v2_moe | llama.cpp | Better MoE support |
| deepseek_v1 | llama.cpp | Proven in llama.cpp |
| gpt_oss | llama.cpp | Sliding window attn |
| llama | llama.cpp | Native support |

## Files Created/Modified

- **Created:**
  - `src/backends/mod.zig` (103 lines)
  - `src/backends/backend.zig` (241 lines)
  - `src/backends/factory.zig` (290 lines)
  - `src/backends/mlx_backend.zig` (98 lines)
  - `src/backends/llama_cpp.zig` (97 lines)

- **Modified:**
  - `build.zig`: Added backends module integration
  - `src/inference/generator.zig`: Added migration comments
  - `src/inference/loader.zig`: Fixed type mismatch (auto-fix)

## Commits

`8959280`: feat(14-01): backend abstraction layer

## Verification

- Module compiles successfully
- Factory correctly maps all architectures
- Backend type conversions work
- No compilation errors in backends/
