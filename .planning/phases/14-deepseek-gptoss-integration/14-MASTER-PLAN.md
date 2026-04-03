---
phase: 14-deepseek-gptoss-integration
plan: MASTER
type: overview
requirements: [MOE-11, MOE-12, MOE-13, MOE-14, MOE-15]
---

# Phase 14: DeepSeek & GPT-OSS Production Integration with llama.cpp

## Goal
Make DeepSeek-Coder-V2-Lite and GPT-OSS-20B fully functional with working inference, using llama.cpp as the primary backend for MoE models (better proven support for 4-bit quantization and MoE routing).

## Why This Phase

**Current State:**
- ✅ Phase 13: Weight loading, dequantization, and architecture implemented for DeepSeek/GPT-OSS via MLX.zig
- ⚠️ DeepSeek: MLX-based inference works but MoE routing still has edge cases with certain quantized formats
- ⚠️ GPT-OSS: Weight loading complete but MLX inference pipeline needs workarounds for sliding window attention
- ❌ TurboQuant: Cannot work effectively with llama.cpp models yet (different memory layout)
- ❌ Backend selection: No unified interface - hardcoded MLX.zig path for all models

**Problems to Solve:**
1. **Backend Abstraction**: Need unified interface that can use MLX.zig (Qwen) OR llama.cpp (DeepSeek/GPT-OSS)
2. **llama.cpp Integration**: Build and link llama.cpp library as alternative inference backend
3. **DeepSeek via llama.cpp**: Route DeepSeek to llama.cpp backend with proper GGUF support
4. **GPT-OSS via llama.cpp**: Download weights, convert to GGUF if needed, run through llama.cpp
5. **Unified Pipeline**: Automatic backend selection based on model type and availability

## Scope

**In Scope:**
- Unified Backend interface: `init()`, `generate()`, `tokenize()`, `deinit()`
- llama.cpp C API integration via `llama.h` bindings
- GGUF model format support for DeepSeek and GPT-OSS
- Backend selection logic (MLX.zig for Qwen, llama.cpp for DeepSeek/GPT-OSS)
- TurboQuant integration with llama.cpp KV cache
- Build system integration for llama.cpp compilation
- Automatic weight format detection and conversion helpers

**Out of Scope:**
- Replacing MLX.zig for Qwen (keep as-is, proven stable)
- Training or fine-tuning capabilities
- Cross-platform support beyond macOS aarch64
- GPU backends other than Metal (CUDA, ROCm)

## Requirements Mapping

| Requirement | Priority | Description |
|-------------|----------|-------------|
| MOE-11 | P0 | Unified backend abstraction layer |
| MOE-12 | P0 | llama.cpp build integration and C bindings |
| MOE-13 | P0 | DeepSeek-Coder-V2-Lite via llama.cpp backend |
| MOE-14 | P0 | GPT-OSS-20B via llama.cpp backend |
| MOE-15 | P1 | Automatic backend selection and unified generation pipeline |

## Success Criteria

1. `./test_models.sh` passes for all 4 models (Qwen, DeepSeek-MLX, DeepSeek-llama.cpp, GPT-OSS-llama.cpp)
2. DeepSeek generates tokens via llama.cpp without "MLX error" or MoE routing issues
3. GPT-OSS downloads, converts (if needed), and runs via llama.cpp
4. Backend selection is automatic: Qwen → MLX.zig, DeepSeek/GPT-OSS → llama.cpp
5. TurboQuant works with all models through unified KV cache interface
6. Single `zig build` command produces binary with both backends linked
7. No performance regression for Qwen (still uses MLX.zig optimally)

## Technical Architecture

### Backend Abstraction Layer

```zig
// src/backends/backend.zig
pub const Backend = union(enum) {
    mlx: MlxBackend,
    llama: LlamaBackend,
};

pub const BackendInterface = struct {
    init: *const fn (allocator: std.mem.Allocator, model_path: []const u8) anyerror!Backend,
    tokenize: *const fn (backend: Backend, text: []const u8) anyerror![]const u32,
    generate: *const fn (backend: Backend, tokens: []const u32, params: GenerationParams) anyerror!GenerationResult,
    deinit: *const fn (backend: Backend) void,
};
```

### llama.cpp Integration

**Build Strategy:**
- Git submodule: `src/llama.cpp/` pointing to ggml-org/llama.cpp stable release
- CMake build integration in `build.zig` (similar to MLX.zig pattern)
- C API bindings: `src/backends/llama_cpp.zig` with `@cImport(@cInclude("llama.h"))`

**Key Functions:**
- `llama_model_load_from_file()` - Load GGUF model
- `llama_tokenize()` - Text to tokens
- `llama_decode()` - Forward pass
- `llama_sample_*()` - Sampling strategies
- `llama_get_kv_cache()` - KV cache access for TurboQuant

### Model Routing

```zig
// src/backends/factory.zig
pub fn createBackend(allocator: std.mem.Allocator, model_info: ModelInfo) !Backend {
    return switch (model_info.architecture) {
        .qwen => Backend.initMlx(allocator, model_info.path), // Keep MLX for Qwen
        .deepseek_v2_moe => Backend.initLlama(allocator, model_info.path), // llama.cpp for DeepSeek
        .gpt_oss => Backend.initLlama(allocator, model_info.path), // llama.cpp for GPT-OSS
        else => error.UnsupportedArchitecture,
    };
}
```

## Plans Overview

| Plan | Objective | Wave | Dependencies |
|------|-----------|------|--------------|
| 14-01 | Backend abstraction layer | 1 | None |
| 14-02 | llama.cpp build integration | 1 | None (parallel with 14-01) |
| 14-03 | DeepSeek llama.cpp integration | 2 | 14-01, 14-02 |
| 14-04 | GPT-OSS llama.cpp integration | 2 | 14-01, 14-02 |
| 14-05 | Unified generation pipeline | 3 | 14-03, 14-04 |

## Dependencies

- **llama.cpp**: git submodule at `src/llama.cpp/` (stable release tag)
- **GGUF models**: DeepSeek-Coder-V2-Lite-GGUF, GPT-OSS-20B-GGUF from HuggingFace
- **TurboQuant**: Shared KV cache compression layer (already implemented in Phase 7)

## Risks and Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| llama.cpp Metal support gaps | High | Use CPU fallback initially, file issues upstream |
| GGUF conversion complexity | Medium | Document conversion workflow, automate if possible |
| Build time increase | Medium | Make llama.cpp optional feature flag initially |
| TurboQuant compatibility | Medium | Abstract KV cache interface to hide differences |

## Definition of Done

- [ ] Backend abstraction compiles and exports clean interface
- [ ] llama.cpp builds and links successfully
- [ ] DeepSeek runs inference via llama.cpp (passes test_models.sh)
- [ ] GPT-OSS runs inference via llama.cpp (passes test_models.sh)
- [ ] Qwen still works via MLX.zig (no regression)
- [ ] TurboQuant compression works with both backends
- [ ] test_models.sh --all passes for all 4 models
- [ ] Documentation updated with backend selection logic
