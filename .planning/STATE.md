---
gsd_state_version: 1.0
milestone: v2.0
milestone_name: "It Just Works"
status: executing
stopped_at: Completed 17-08-PLAN.md — real forward() in GPTOSSTransformer using embed_tokens+lm_head via mlx.take+matmul
last_updated: "2026-04-06T17:39:05.256Z"
last_activity: 2026-04-06
progress:
  total_phases: 20
  completed_phases: 9
  total_plans: 51
  completed_plans: 50
  percent: 96
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-05)

**Core value:** A single `zig build` binary that lets OpenCode connect to local coding models without any Python or cloud dependency.
**Current focus:** Phase 17 — inference-gap-closure

## Milestone: v2.0 "It Just Works"

**Goal:** Close every stub, mock, and orphaned route; ship verified inference for all claimed models; add Gemma 4 E4B; implement TurboQuant Metal kernels; wire tool APIs end-to-end; produce project documentation.

**Previous milestone:** v1.1.2 — Phase 15 (Native MLX GPT-OSS) — 8/9 plans complete

## Current Position

Milestone: v2.0 "It Just Works"
Phase: 17 (inference-gap-closure) — EXECUTING
Plan: 4 of 8
Status: Ready to execute
Last activity: 2026-04-06

Progress: [██████████] 46/48 plans (96%)

## v2.0 Phase Summary

| Phase | Goal | Requirements | Status |
|-------|------|--------------|--------|
| 16. Build & Gap Closure | Zig 0.15.2 clean build | GAP-08, GAP-09, GAP-10 | ✅ Complete (2026-04-05) |
| 17. Inference Gap Closure | Real forward pass, no stubs | GAP-01..07, MODEL-01..03 | Not started |
| 18. Gemma 4 E4B | New model end-to-end | MODEL-04..07 | Not started |
| 19. TurboQuant Metal | Metal kernels, 4x compression | TURBO-01..04 | Not started |
| 20. Tools API | Browser + Python HTTP endpoints | TOOLS-01..04 | Not started |
| 21. Project Documentation | History, guide, ecosystem map | DOCS-01..03 | Not started |

## Phase 15: Native MLX GPT-OSS — In Progress (8/9 plans)

**Goal:** High-performance GPT-OSS via native MLX with Metal kernels, Harmony format, and tools

**Plans Completed:**

- ✅ 15-01: MLX GPT-OSS transformer with Metal kernels
- ✅ 15-02: Harmony format parser and chat template
- ✅ 15-03: Browser and Python tool implementation
- ✅ 15-04: Weight loading and MXFP4 support
- ✅ 15-05: Integration with zlx server
- ✅ 15-06: Gap closure summary
- ✅ 15-07: ChatGPTOSSHandler and ToolsAPI wiring
- ✅ 15-08: Inference stub replacement

**Outstanding:** 15-MASTER-PLAN.md not written; UAT and verification docs exist

## Phase 14: DeepSeek & GPT-OSS Integration — ✅ COMPLETE

**All 5 Plans Completed:**

- ✅ 14-01: Backend abstraction layer (Backend union, factory pattern, routing)
- ✅ 14-02: llama.cpp build integration (submodule, C bindings, CMake)
- ✅ 14-03: DeepSeek integration (GGUF download, registry updates, tests)
- ✅ 14-04: GPT-OSS integration (11GB download, GGUF support, docs)
- ✅ 14-05: Unified generation pipeline (BackendGenerator, integration tests)

**Backend Routing:**

| Model | Default Backend | Override |
|-------|----------------|----------|
| Qwen | MLX.zig | --backend mlx |
| DeepSeek | llama.cpp | --backend llama_cpp |
| GPT-OSS | llama.cpp | --backend llama_cpp |

## Decision Log

**2026-04-06:** Phase 17 Plan 03 — DeepSeek handler wired, llama.cpp API migrated, zig build fixed (GAP-03, MODEL-02, MODEL-03)

- **Decision**: D-01 applied: `g_chat_deepseek_handler` global in `server.zig` dispatches by `deepseek` prefix — no factory/registry layer
- **Decision**: D-06 applied: `vocab_size`, `eos_token`, `bos_token` read from `LlamaBackend` accessors via `llama_model_get_vocab()` — never hardcoded
- **Decision**: ggml sub-libraries added to `build.zig` (`libggml.a`, `libggml-base.a`, `libggml-cpu.a`, `libggml-blas.a`, `libggml-metal.a`) — unblocks linker; `zig build` exits 0
- **Decision**: llama.cpp vocab API migrated: seed removed from context_params; all vocab/token functions now take `llama_vocab*`; `llama_sample_token` replaced by `llama_sampler_sample`

**2026-04-06:** Phase 17 Plan 02 — delete orphaned backend stubs (GAP-02, GAP-05)

- **Decision**: D-04 executed: `factory.zig` and `mlx_backend.zig` deleted — confirmed pure stubs never called by live inference path
- **Decision**: GAP-02 closed: hardcoded `vocab_size=32000`, `eos_token=2`, `bos_token=1` eliminated — lived exclusively in deleted `mlx_backend.zig`
- **Decision**: `BackendType.mlx` removed from union; Qwen inference uses `inference/mod.zig` + MLX.zig directly; Backend union now only has `.llama_cpp` and `.mlx_gptoss` arms
- **Decision**: `BackendGenerator` stub kept with `error.NotImplemented` — preserves type surface for Phase 18 without introducing false factory path

**2026-04-05:** Phase 16 complete — build gap closure

- **Decision**: MLA stub (Approach B) used for `MultiHeadLatentAttention` init — `init()` returns `!*Self` (heap pointer) but `DeepSeekLayer.mla` is a value type; dereferencing heap-allocated struct while `deinit()` calls `allocator.destroy(self)` would cause use-after-free
- **Decision**: `arrayIsEmpty` added to `src/mlx.zig/src/mlx.zig` wrapping `C.mlx_array_size(arr) == 0` — 20 call sites unblocked
- **Remaining**: 2 pre-existing errors (`dequantize.zig:101` type mismatch, `loader.zig:542` missing `num_shared_experts`) tracked for Phase 17

**2026-04-05:** v2.0 roadmap created

- **Decision**: Phase 16 must come before all other v2.0 phases — Zig 0.15.2 build errors block compilation
- **Decision**: Phase 17 bundles all inference stubs (GAP-01..07) with model verification (MODEL-01..03) — same code surface
- **Decision**: Phase 18 (Gemma 4 E4B) deferred until Phase 17 is complete — clean inference layer required
- **Decision**: Phase 19 (TurboQuant Metal) and Phase 20 (Tools) can run after Phase 16 in any order — both independent of Phase 17 and 18
- **Decision**: Phase 21 (Docs) is last — it documents what actually works, not what was planned

**2026-04-03:** Completed Phase 14 - DeepSeek & GPT-OSS with llama.cpp

- **Decision**: Implemented llama.cpp backend for MoE models alongside MLX.zig
- **Rationale**: llama.cpp has proven DeepSeek V2 support and better GGUF ecosystem
- **Outcome**: Both MLX.zig and llama.cpp backends available; Qwen still uses optimized MLX.zig path

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260405-bev | Fix Zig version references update to 0.15.2 and pin in CLAUDE.md | 2026-04-05 | d784f23 | [260405-bev-fix-zig-version-references-update-to-0-1](./quick/260405-bev-fix-zig-version-references-update-to-0-1/) |

## Session Continuity

Last activity: 2026-04-05 - Phase 16 complete (build gap closure)
Last session: 2026-04-06T17:39:05.252Z
Stopped at: Completed 17-08-PLAN.md — real forward() in GPTOSSTransformer using embed_tokens+lm_head via mlx.take+matmul
Resume: `/gsd:plan-phase 17`
