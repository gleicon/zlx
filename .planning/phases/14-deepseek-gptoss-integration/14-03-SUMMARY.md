---
phase: 14-deepseek-gptoss-integration
plan: 03
subsystem: deepseek
completed: 2026-04-03
---

# Phase 14 Plan 03: DeepSeek End-to-End Integration

## Summary

Completed llama.cpp backend implementation for DeepSeek-Coder-V2-Lite, enabling inference via GGUF format with automatic download support.

## Key Achievements

1. **Factory Integration** (`src/backends/factory.zig`)
   - `createLlamaBackend()`: Creates real LlamaBackend instances
   - DeepSeek automatically routed to llama.cpp
   - Proper error handling and cleanup

2. **Registry Updates** (`src/models/registry.zig`)
   - Added `BackendType` enum (mlx, llama_cpp)
   - Extended `KnownModelInfo` with:
     - `preferred_backend`: llama.cpp for DeepSeek
     - `download_urls`: HuggingFace GGUF links
     - `gguf_filename`: Model file name
   - Added `getDownloadInfo()`: URLs, sizes, checksums
   - Added `getPreferredBackend()`: Backend selection helper

3. **Download Infrastructure** (`src/download/deepseek.zig`)
   - `downloadDeepSeekModel()`: Download with resume support
   - `verifyDeepSeekModel()`: GGUF magic validation
   - Disk space check (~5GB required)
   - Progress callback support
   - Default quantization: Q4_K_M (~4.5GB)

4. **Testing** (`test_models.sh`)
   - `--deepseek-llama` flag for llama.cpp backend
   - `test_deepseek_llama()` function:
     - Downloads GGUF if missing
     - Starts zlx with --backend llama_cpp
     - Tests /v1/models endpoint
     - Tests chat completion
     - Verifies memory < 16GB with TurboQuant

## DeepSeek GGUF Source

- **Primary:** TheBloke/deepseek-coder-v2-lite-GGUF
- **File:** deepseek-coder-v2-lite.Q4_K_M.gguf
- **Size:** ~4.5GB
- **Quantization:** Q4_K_M (4-bit, K-quants, medium mix)

## Registry Entry

```zig
.{
    .id = "deepseek-coder-v2-lite",
    .architecture = .deepseek_v2_moe,
    .preferred_backend = .llama_cpp,
    .quantization = "Q4_K_M_GGUF",
    .download_urls = &.{
        "https://huggingface.co/TheBloke/deepseek-coder-v2-lite-GGUF/...",
    },
    .gguf_filename = "deepseek-coder-v2-lite.Q4_K_M.gguf",
}
```

## Files Created/Modified

- **Created:**
  - `src/download/deepseek.zig` (152 lines)

- **Modified:**
  - `src/backends/factory.zig`: Real LlamaBackend creation
  - `src/models/registry.zig`: DeepSeek GGUF info
  - `src/download/mod.zig`: Export deepseek module
  - `test_models.sh`: --deepseek-llama test path

## Commits

`9e4909c`: feat(14-03): DeepSeek end-to-end integration

## Verification

- Factory creates LlamaBackend for DeepSeek
- Backend union delegates correctly
- Download info available
- GGUF format detected
- test_models.sh --deepseek-llama path defined
