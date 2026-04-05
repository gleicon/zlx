---
gsd_state_version: 1.0
milestone: v2.0
milestone_name: "It Just Works"
status: verifying
stopped_at: Completed 16-03-PLAN.md
last_updated: "2026-04-05T18:52:59.456Z"
last_activity: 2026-04-05
progress:
  total_phases: 20
  completed_phases: 8
  total_plans: 43
  completed_plans: 42
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-05)

**Core value:** A single `zig build` binary that lets OpenCode connect to local coding models without any Python or cloud dependency.
**Current focus:** Phase 16 — build-gap-closure

## Milestone: v2.0 "It Just Works"

**Goal:** Close every stub, mock, and orphaned route; ship verified inference for all claimed models; add Gemma 4 E4B; implement TurboQuant Metal kernels; wire tool APIs end-to-end; produce project documentation.

**Previous milestone:** v1.1.2 — Phase 15 (Native MLX GPT-OSS) — 8/9 plans complete

## Current Position

Milestone: v2.0 "It Just Works"
Phase: 17
Plan: Not started
Status: Phase complete — ready for verification
Last activity: 2026-04-05

Progress: [░░░░░░░░░░░] 0% (0/6 phases complete)

## v2.0 Phase Summary

| Phase | Goal | Requirements | Status |
|-------|------|--------------|--------|
| 16. Build & Gap Closure | Zig 0.15.2 clean build | GAP-08, GAP-09, GAP-10 | Not started |
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

Last activity: 2026-04-05 - v2.0 roadmap defined (Phases 16-21)
Last session: 2026-04-05T18:49:00.331Z
Stopped at: Completed 16-03-PLAN.md
Resume: Begin Phase 16 with `/gsd:plan-phase 16`
