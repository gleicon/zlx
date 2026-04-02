---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: verifying
stopped_at: Completed 07-01 TurboQuant Feasibility Spike
last_updated: "2026-04-02T10:14:51.894Z"
progress:
  total_phases: 9
  completed_phases: 4
  total_plans: 10
  completed_plans: 12
  percent: 67
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-01)

**Core value:** A single `zig build` binary that lets OpenCode connect to local coding models without any Python or cloud dependency.
**Current focus:** Phase 07 — TurboQuant Integration (Feasibility Assessment)

## Current Position

Milestone: v1.1 (Production-Ready)
Phase: 07 (TurboQuant Integration) — PLANNED
Plan: 1 of 1 — Feasibility Spike Ready for Execution
Status: Phase complete — ready for verification

Progress: [██████▓░░░] 67% → Phase 7 planned, awaiting decision on TurboQuant porting

## Phase 07 Status

| Plan | Name | Status | Requirements |
|------|------|--------|--------------|
| 07-01 | TurboQuant Feasibility Spike | 📝 PLANNED | PERF-01 |

**Critical Finding:** TurboQuant is 100% Python with no C API. Porting requires extracting Metal kernels from Python strings and implementing custom ops — estimated 40+ hours.

## Phase 07 Plan Summary

**Plan 07-01: TurboQuant Integration - Feasibility Spike**

This is a research/spike plan, not a full implementation. Key findings:

1. **TurboQuant Reality Check:**
   - Repository: arozanov/turboquant-mlx
   - Language: 100% Python (no C/C++ API)
   - Metal kernels: Embedded as Python strings, JIT-compiled at runtime
   - Cannot directly bind from Zig

2. **Porting Options Identified:**
   - **Option A:** C++ Custom Ops via mlx-c upgrade (40-60 hours, HIGH risk)
   - **Option B:** Direct Metal in Zig (60-80 hours, MEDIUM risk)
   - **Option C:** MLX array ops approximation (20-30 hours, loses benefits)

3. **Plan Deliverables:**
   - Stub compression module (KvCompressor interface)
   - TurboQuant stub with NotImplemented markers
   - --turboquant CLI flag with graceful fallback
   - RESEARCH.md with porting analysis and recommendation

4. **Decision Required:**
   After executing 07-01, decide:

   - **GO:** Proceed with full TurboQuant port (40+ hours)
   - **NO-GO:** Defer TurboQuant, skip to Phase 8 (Speculative Decoding)

## Key Decisions Made

1. **Request ID Format**: Using `req-{timestamp}-{random}` for unique, debuggable IDs
2. **Timeout Default**: 60 seconds matches common API practices and OpenAI's default
3. **Partial Completion**: Return generated content even on timeout (better UX than empty response)
4. **Error Type Specificity**: JSON parse errors include specific error type mapping for easier debugging
5. **Sampling Pipeline Order**: logit_bias → penalties → top_k → min_p → temperature → softmax (matches OpenAI semantics)
6. **CPU-side Logit Modification**: Extract MLX array to CPU slice for sampling parameter application to work around MLX C API limitations
7. **Zero-overhead Design**: Only allocate/modify when sampling parameters are non-default
8. **Tokenizer Storage**: Typed pointer (?*mlx_tokenizer.Tokenizer) for proper decode access in generation
9. **Logprobs Capture**: Capture before sampling modifications for accurate probability reporting
10. **Memory Safety**: 20% margin added to memory requirements for model switching
11. **Generation Tracking**: Atomic counter with condition variable for graceful model transitions
12. **Health Checks**: 3-check system (model, memory, registry) for status determination
13. **Local over Cache**: Duplicate model names prefer ./models/ over ~/.cache/zlx/models/
14. **TurboQuant Reality**: Python-only library requires significant porting effort (40+ hours)

## Session Continuity

Last session: 2026-04-02T10:14:51.891Z
Stopped at: Completed 07-01 TurboQuant Feasibility Spike
Resume file: None

## Next Steps

### Phase 07 Decision Point

Phase 07 planning is complete. Before executing, decide:

**Question:** Should we proceed with TurboQuant porting or defer to Phase 8?

**Considerations:**

- TurboQuant porting: 40+ hours, high risk (mlx-c upgrade required)
- Current state: Prompt caching (Phase 6) provides adequate performance
- Alternative: Speculative Decoding (Phase 8) may offer better ROI
- No C API means custom implementation required

**Recommended approach:**

1. Execute 07-01 spike to get exact effort estimate
2. Review RESEARCH.md output
3. Make Go/No-Go decision
4. If No-Go: Update ROADMAP to skip Phase 7, proceed to Phase 8

## Research Artifacts

- STACK.md: Technology recommendations for v1.1
- FEATURES.md: Feature landscape and prioritization  
- ARCHITECTURE.md: Component architecture and patterns
- PITFALLS.md: Critical pitfalls and prevention strategies
- SUMMARY.md: Executive summary with roadmap implications
- Phase 07: src/compression/RESEARCH.md (to be created during execution)

## Phase 06 Artifacts

- 06-01-SUMMARY.md: Prompt caching with KV persistence, LRU eviction, metrics

## Phase 07 Artifacts (Planned)

- 07-01-PLAN.md: TurboQuant feasibility spike plan
- 07-01-SUMMARY.md: Portability assessment and Go/No-Go recommendation (after execution)
