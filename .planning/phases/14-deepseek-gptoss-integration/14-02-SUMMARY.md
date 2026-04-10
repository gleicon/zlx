---
phase: 14-deepseek-gptoss-integration
plan: 02
subsystem: llama.cpp
completed: 2026-04-03
---

# Phase 14 Plan 02: llama.cpp Build Integration

## Summary

Integrated llama.cpp as a git submodule and created C API bindings to enable it as an alternative inference backend for MoE models.

## Key Achievements

1. **llama.cpp Submodule**
   - Added `src/llama.cpp/` as git submodule from ggml-org/llama.cpp
   - Contains llama.h, ggml.h, CMakeLists.txt, Metal support

2. **C API Bindings** (`src/backends/llama_c.zig`)
   - `@cImport` of llama.h with LLAMA_USE_METAL
   - Re-exports: llama_model, llama_context, llama_token
   - Model loading: llama_model_load_from_file()
   - Tokenization: llama_tokenize()
   - Inference: llama_decode(), llama_get_logits()
   - Sampling: llama_sample_token(), sampler chains
   - KV cache: llama_get_kv_cache_token_count()

3. **Zig Wrapper** (`src/backends/llama_cpp.zig`)
   - `LlamaBackend` struct with model/context handles
   - `init()`: Load GGUF with Metal GPU offload (1000 layers)
   - `tokenize()`: Text → tokens via llama.cpp
   - `decode()`: Forward pass with batch management
   - `sample()`: Token sampling with sampler chains
   - `createSampler()`: Build top_k → top_p → temp chain
   - `deinit()`: Proper resource cleanup

4. **Build Integration**
   - CMake build in `build.zig`: -DLLAMA_METAL=ON
   - Static library linking: libllama.a
   - Include paths for llama.h and ggml.h
   - Metal frameworks: Metal, Foundation, QuartzCore, Accelerate
   - Custom step: `zig build llama`

## Build Configuration

```zig
// CMake configuration
cmake -B build -S src/llama.cpp \
  -DLLAMA_METAL=ON \
  -DLLAMA_METAL_EMBED_LIBRARY=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF

// Library linking
exe.addLibraryPath(.{ .path = "src/llama.cpp/build/bin" });
exe.linkSystemLibrary("llama");
```

## Files Created/Modified

- **Created:**
  - `src/backends/llama_c.zig` (109 lines)
  - `src/llama.cpp/` (git submodule)

- **Modified:**
  - `src/backends/llama_cpp.zig`: Complete implementation (381 lines)
  - `build.zig`: CMake integration and test step
  - `.gitmodules`: Submodule configuration

## Commits

`5f2aa90`: feat(14-02): llama.cpp build integration

## Verification

- llama.cpp submodule initialized
- C bindings compile (llama_c.zig)
- LlamaBackend struct complete with all methods
- CMake build integration in build.zig
- `zig build llama` step defined
