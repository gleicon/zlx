---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: Completed 15-04-PLAN.md - GPT-OSS weight loading and HuggingFace downloader
last_updated: "2026-04-04T12:00:00.727Z"
progress:
  total_phases: 14
  completed_phases: 7
  total_plans: 37
  completed_plans: 35
  percent: 98
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-02)

**Core value:** A single `zig build` binary that lets OpenCode connect to local coding models without any Python or cloud dependency.
**Current focus:** Phase 15 — mlx-gptoss

## Version Update: v1.1.1 → v1.1.2

**v1.1.1 Released:** All core features complete with MLX-based MoE support
**v1.1.2 Goal:** Production-ready DeepSeek & GPT-OSS via llama.cpp backend

### Why v1.1.2?

- v1.1.1 has working MoE via MLX.zig but edge cases remain
- llama.cpp provides proven MoE implementation (DeepSeek V2 native support)
- Better GGUF ecosystem for 4-bit quantized models
- TurboQuant integration with llama.cpp KV cache

## Current Position

Milestone: v1.1.2 (MoE Models with llama.cpp)
Phase: 15 (mlx-gptoss) — EXECUTING
Plan: 5 of 6
Status: Ready to execute

Progress: [███████████] 98% → Phase 14 complete, ready for model testing

## Phase 14: DeepSeek & GPT-OSS Integration — ✅ COMPLETE

**Goal:** Make DeepSeek-Coder-V2-Lite and GPT-OSS-20B fully functional with llama.cpp backend

**All 5 Plans Completed:**

- ✅ 14-01: Backend abstraction layer (Backend union, factory pattern, routing)
- ✅ 14-02: llama.cpp build integration (submodule, C bindings, CMake)
- ✅ 14-03: DeepSeek integration (GGUF download, registry updates, tests)
- ✅ 14-04: GPT-OSS integration (11GB download, GGUF support, docs)
- ✅ 14-05: Unified generation pipeline (BackendGenerator, integration tests)

**Key Achievements:**

1. **Backend Abstraction**: Unified interface for MLX.zig and llama.cpp
2. **llama.cpp Integration**: C API bindings, CMake build, Metal GPU support
3. **DeepSeek**: Q4_K_M GGUF (~4.5GB), automatic download, llama.cpp backend
4. **GPT-OSS**: Q4_K_M GGUF (~11GB), resume download, 128K context support
5. **Testing**: 15 integration tests, comprehensive shell tests, --all-backends flag

**New Files Created:**

- `src/backends/` - Backend abstraction layer (5 files)
- `src/llama.cpp/` - llama.cpp submodule
- `src/download/deepseek.zig` - DeepSeek download support
- `src/download/gptoss.zig` - GPT-OSS download support
- `src/inference/backend_generator.zig` - Unified generator
- `src/test_backend_integration.zig` - 15 integration tests
- `docs/GGUF_CONVERSION.md` - Conversion guide

**Backend Routing:**

| Model | Default Backend | Override |
|-------|----------------|----------|
| Qwen | MLX.zig | --backend mlx |
| DeepSeek | llama.cpp | --backend llama_cpp |
| GPT-OSS | llama.cpp | --backend llama_cpp |

**Test Commands:**

```bash
./test_models.sh --deepseek-llama      # Test DeepSeek via llama.cpp
./test_models.sh --gptoss-llama        # Test GPT-OSS (requires download)
./test_models.sh --gptoss-download     # Download + test GPT-OSS
./test_models.sh --all-backends        # Test all configurations
```

**Commits:** 8959280, 5f2aa90, 9e4909c, 23207ed, 6bb8229, a3b5cb5

## Previous Phase: 13 — ✅ COMPLETE

Phase 13 completed with MLX-based MoE support:

- DeepSeek 4-bit dequantization
- GPT-OSS 11GB download infrastructure
- Integration testing framework

## Next Steps

1. **Model Testing** (2 hours)
   - Download DeepSeek GGUF and run inference
   - Download GPT-OSS and verify 11GB model loads
   - Run `./test_models.sh --all-backends`

2. **Performance Tuning** (2 hours)
   - Benchmark DeepSeek MLX vs llama.cpp
   - Tune TurboQuant for llama.cpp KV cache
   - Optimize batch sizes for each backend

3. **Release v1.1.2** (1 hour)
   - Tag release with llama.cpp support
   - Update CHANGELOG
   - Create release notes

## Decision Log

**2026-04-03:** Completed Phase 14 - DeepSeek & GPT-OSS with llama.cpp

- **Decision**: Implemented llama.cpp backend for MoE models alongside MLX.zig
- **Rationale**: llama.cpp has proven DeepSeek V2 support and better GGUF ecosystem
- **Outcome**: 
  - Both MLX.zig and llama.cpp backends available
  - Automatic backend selection by architecture
  - DeepSeek: 4.5GB GGUF via llama.cpp
  - GPT-OSS: 11GB GGUF via llama.cpp
  - Qwen: Still uses optimized MLX.zig path

## Session Continuity

Last session: 2026-04-04T12:00:00.725Z
Stopped at: Completed 15-04-PLAN.md - GPT-OSS weight loading and HuggingFace downloader
Resume: Model testing with actual GGUF downloads
