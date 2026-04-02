---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: Completed 11-02 MLA Implementation
last_updated: "2026-04-02T21:11:05.365Z"
progress:
  total_phases: 10
  completed_phases: 6
  total_plans: 21
  completed_plans: 20
  percent: 90
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-02)

**Core value:** A single `zig build` binary that lets OpenCode connect to local coding models without any Python or cloud dependency.
**Current focus:** Phase 11 — DeepSeek MoE via mlx-c Upgrade

## Version Update: v1.1.0 → v1.1.1

**v1.1.0 Released:** All core features complete, production-ready
**v1.1.1 Goal:** Add DeepSeek-Coder-V2-Lite support via mlx-c upgrade

### Why v1.1.1?

- v1.1.0 is stable and complete for its intended scope
- DeepSeek MoE is a major feature requiring infrastructure changes
- Separate release allows focused testing of MoE functionality
- Users can stay on v1.1.0 if they don't need DeepSeek

## Current Position

Milestone: v1.1.1 (DeepSeek MoE Support)
Phase: 11 (DeepSeek MoE via mlx-c Upgrade) — EXECUTING
Plan: 4 of 6 - COMPLETE
Plans: 1 of 4 complete (11-03 MoE Routing)
Status: Ready to execute

Progress: [██████████░░] 90% → Plan 11-03 complete, advancing to 11-04

## Phase 11: DeepSeek MoE Support

### Strategy: Option 1 — mlx-c v0.4.x Upgrade (RECOMMENDED)

**Why Option 1?**

- Lower effort than custom implementation (8-12h vs 20-30h)
- Gets us latest MLX features, not just MoE
- Community support for v0.4.x
- Foundation for future model support

**Blockers to Address:**

1. MLX.zig currently pins mlx-c v0.1.2
2. API changes between v0.1.2 and v0.4.x
3. Testing required to ensure compatibility

### Phase 11 Plans

| Plan | Name | Status | Focus |
|------|------|--------|-------|
| 11-01 | mlx-c v0.4.x Integration | ✅ COMPLETE | Upgrade from v0.1.2 to v0.4.x |
| 11-02 | MLA Implementation | ⏳ QUEUED | Multi-head Latent Attention |
| 11-03 | MoE Routing Layer | ✅ COMPLETE | Mixture of Experts routing |
| 11-04 | DeepSeek Transformer | ⏳ QUEUED | Complete DeepSeek-V2 support |

### Technical Approach

**Step 1: mlx-c v0.4.x Upgrade**

- Update build.zig to fetch mlx-c v0.4.x
- Adapt MLX.zig to new API (breaking changes expected)
- Verify existing models still work (Qwen, Llama, Phi)
- Test build on macOS aarch64

**Step 2: MLA (Multi-head Latent Attention)**

- Implement DeepSeek's compressed attention
- 90% KV cache reduction vs standard MHA
- New file: `src/mlx.zig/src/mla.zig`

**Step 3: MoE Routing**

- Implement expert routing mechanism
- Shared + routed experts
- Top-k selection per token
- New file: `src/mlx.zig/src/moe.zig`

**Step 4: DeepSeek Transformer**

- Combine MLA + MoE in transformer
- Chat template for DeepSeek format
- Memory estimation for sparse params
- New file: `src/mlx.zig/src/deepseek.zig`

### Success Criteria

- [ ] DeepSeek model loads and runs inference
- [ ] Memory usage matches 2B active params (not 15.7B)
- [ ] 128k context works with TurboQuant
- [ ] Performance within 10% of MLX Python baseline
- [ ] Chat completions use correct format (User:/Assistant:)
- [ ] Existing models (Qwen, Llama) continue to work

### Estimated Effort

| Component | Hours | Risk |
|-----------|-------|------|
| mlx-c v0.4.x upgrade | 4-6 | Medium (API changes) |
| MLA implementation | 4-6 | High (new architecture) |
| MoE routing | 3-4 | High (complex routing) |
| DeepSeek transformer | 2-3 | Medium (integration) |
| Chat template | 1-2 | Low |
| Testing/optimization | 2-3 | Medium |
| **Total** | **16-24 hours** | |

### Research Required

**Before Implementation:**

1. Compare mlx-c v0.1.2 vs v0.4.x API differences
2. Study llama.cpp DeepSeek-V2 implementation
3. Review MLX Python MoE implementation
4. Check if custom ops API available in v0.4.x

### Risk Mitigation

**High Risk:** mlx-c v0.4.x breaks MLX.zig compatibility

- **Mitigation:** Create branch, test incrementally
- **Fallback:** Fork MLX.zig or custom MoE implementation

**High Risk:** MLA implementation incorrect

- **Mitigation:** Reference llama.cpp implementation
- **Mitigation:** Test against MLX Python for parity

**Medium Risk:** Performance regression

- **Mitigation:** Benchmark before/after upgrade
- **Mitigation:** Keep v1.1.0 available for rollback

---

## Decision Log

**2026-04-02:** Completed 11-03 MoE Routing Implementation

- **Decision**: Place MoE files in main repo (src/moe.zig) instead of mlx.zig submodule
- **Rationale**: MoE is zlx-specific extension, keeps submodule clean
- **Outcome**: 345-line moe.zig with Metal kernel integration

**2026-04-02:** Selected Option 1 (mlx-c upgrade) over Options 2/3

- Rationale: Lower effort, future-proofs codebase, community support
- Risk: API changes may require significant MLX.zig updates

**2026-04-02:** Moved to v1.1.1 for DeepSeek support

- Rationale: Major feature, separate release allows focused testing
- v1.1.0 remains stable production release

---

## Next Steps

1. **Research Phase (2 hours)**
   - Document mlx-c v0.1.2 → v0.4.x API changes
   - Identify breaking changes in MLX.zig
   - Study reference implementations

2. **Planning Phase (1 hour)**
   - Create detailed PLAN.md for each sub-plan
   - Identify integration points
   - Define test strategy

3. **Implementation Phase (16-24 hours)**
   - Execute plans 11-01 through 11-04
   - Parallel work possible after 11-01 complete

4. **Testing Phase (4 hours)**
   - Unit tests for MLA, MoE
   - Integration tests for DeepSeek
   - Performance benchmarks

5. **Release v1.1.1**
   - Tag release
   - Update documentation
   - Announce DeepSeek support

---

## Session Continuity

Last session: 2026-04-02T21:15:00.000Z
Stopped at: Completed 11-03 MoE Routing Layer
Resume: Ready for 11-04 DeepSeek Transformer Integration

---

*State updated after 11-03 execution*
