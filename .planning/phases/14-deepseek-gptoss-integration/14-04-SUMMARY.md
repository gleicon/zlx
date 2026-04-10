---
phase: 14-deepseek-gptoss-integration
plan: 04
subsystem: gpt-oss
completed: 2026-04-03
---

# Phase 14 Plan 04: GPT-OSS Download & Integration

## Summary

Enabled GPT-OSS-20B inference through llama.cpp backend with automatic download support for the 11GB GGUF model.

## Key Achievements

1. **GPT-OSS Download Module** (`src/download/gptoss.zig`)
   - `downloadGptOssModel()`: Download 11GB with resume
   - `verifyGptOssModel()`: GGUF magic + file size validation
   - `isGptOssAvailable()`: Check if model cached locally
   - `defaultProgressCallback()`: Log progress every 100MB
   - `formatBytes()`: Human-readable sizes
   - `calculateEta()`: Download time estimation
   - Disk space check: ~12GB required

2. **Registry Updates** (`src/models/registry.zig`)
   - Extended GPT-OSS entry:
     - `preferred_backend = .llama_cpp`
     - `quantization = "Q4_K_M_GGUF"`
     - `download_urls`: bartowski/unsloth mirrors
     - `gguf_filename = "GPT-OSS-20B-Q4_K_M.gguf"`
     - `memory_required_gb = 11.0`
   - `getDownloadInfo("gpt-oss-20b")`: Returns URL, size, filename

3. **Testing** (`test_models.sh`)
   - `--gptoss-llama` flag: Test via llama.cpp
   - `--gptoss-download` flag: Download + test
   - `test_gptoss_llama()` function:
     - 120s timeout for 11GB model load
     - Verifies file size > 10GB
     - Tests /v1/models endpoint
     - Tests chat completion
     - Memory check with TurboQuant

4. **Documentation** (`docs/GGUF_CONVERSION.md`)
   - Pre-converted GGUF download instructions
   - Manual conversion from OpenAI weights
   - Troubleshooting guide
   - Quantization selection table
   - Platform-specific notes

## GPT-OSS GGUF Sources

- **Primary:** bartowski/GPT-OSS-20B-GGUF
- **Mirror:** unsloth/GPT-OSS-20B-GGUF
- **File:** GPT-OSS-20B-Q4_K_M.gguf
- **Size:** ~11GB
- **Quantization:** Q4_K_M

## Download Features

- Resume support (HTTP Range)
- Progress reporting
- File validation (size + magic)
- Multiple mirror fallback
- Disk space pre-check

## Files Created/Modified

- **Created:**
  - `src/download/gptoss.zig` (191 lines)
  - `docs/GGUF_CONVERSION.md` (137 lines)

- **Modified:**
  - `src/download/mod.zig`: Export gptoss functions
  - `test_models.sh`: GPT-OSS llama test path

## Commits

`23207ed`: feat(14-04): GPT-OSS download and integration

## Verification

- Download module compiles
- 11GB file handling implemented
- GGUF validation checks file size + magic
- Registry has correct download info
- test_models.sh paths added
