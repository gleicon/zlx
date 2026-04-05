---
phase: 16-build-gap-closure
plan: 01
subsystem: infra
tags: [zig, build-system, pathJoin, LazyPath, mlx-c, llama-cpp]

# Dependency graph
requires:
  - phase: 15-mlx-gptoss
    provides: GPT-OSS backend, harmony format, tools modules — all referenced in build.zig
provides:
  - "Verified clean build-script evaluation — no pathJoin/LazyPath errors"
  - "zig build reaches source compilation (llama.cpp cmake/make succeeds, 2 source errors remain)"
affects: [17-inference-gap-closure, 18-gemma4, 19-turbo-quant, 20-tools-api]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "LazyPath pattern: always use .{ .cwd_relative = b.pathJoin(...) } for addIncludePath/addObjectFile/addLibraryPath/setCwd/addFrameworkPath"
    - "addSystemCommand pattern: pass plain []const u8 directly — no LazyPath wrapper needed"

key-files:
  created: []
  modified:
    - build.zig
    - src/mlx.zig/build.zig (no changes needed — already correct)

key-decisions:
  - "build.zig .path field usage was already fixed to .cwd_relative before this plan ran — audit confirmed correctness"
  - "src/mlx.zig/build.zig all call sites already follow correct LazyPath pattern — no changes needed"

patterns-established:
  - "LazyPath audit: all b.pathJoin() results passed to LazyPath-expecting APIs must use .{ .cwd_relative = ... } wrapper"
  - "addSystemCommand audit: b.pathJoin() passed directly as []const u8 is correct — no wrapper"

requirements-completed: [GAP-08]

# Metrics
duration: 2min
completed: 2026-04-05
---

# Phase 16 Plan 01: Build Gap Closure — pathJoin Audit Summary

**zig build reaches source compilation with 0 build-script errors after confirming all pathJoin call sites use .cwd_relative LazyPath wrappers**

## Performance

- **Duration:** 2 min
- **Started:** 2026-04-05T18:42:08Z
- **Completed:** 2026-04-05T18:44:19Z
- **Tasks:** 2
- **Files modified:** 1 (build.zig — already had correct fixes staged from prior work)

## Accomplishments

- Audited all `b.pathJoin(...)` call sites in `src/mlx.zig/build.zig` — all 10 call sites already correct
- Audited all `b.pathJoin(...)` call sites in `build.zig` — all call sites already correct (`.path` was previously changed to `.cwd_relative` in 3 places: lines 113, 114, 117)
- Confirmed `zig build` passes build-script evaluation phase with 0 pathJoin/LazyPath errors
- Build reaches source compilation and fully builds llama.cpp (cmake + make, 100% target completion)
- Identified 2 remaining source errors (GAP-09/GAP-10) correctly deferred to plans 02 and 03

## Task Commits

Each task was committed atomically:

1. **Tasks 1+2: Audit src/mlx.zig/build.zig and build.zig for pathJoin call sites** - `4a3be74` (chore)

**Plan metadata:** (created below)

## Files Created/Modified

- `build.zig` — 3 changes from prior work: lines 113/114 `.path` → `.cwd_relative` for addIncludePath, line 117 `.path` → `.cwd_relative` for addLibraryPath; also corrected `llama_build_path/bin` → `llama_build_path/src` (library path)
- `src/mlx.zig/build.zig` — no changes; all pathJoin call sites already used correct patterns

## Decisions Made

- Both build files were already correct at time of execution — prior phase 15 work had applied the fixes; plan confirmed and committed the correct state
- No further pathJoin changes required for build-script evaluation phase to pass

## Deviations from Plan

None - plan executed exactly as written. Both files were already in the correct state. The audit verified each call site classification and confirmed no changes were needed to `src/mlx.zig/build.zig`.

## Issues Encountered

None. Build script evaluation passes cleanly. The 2 source compilation errors (`arrayIsEmpty` missing from mlx.zig API, `weights` field missing from `MultiHeadLatentAttention`) are expected and in scope for plans 02 and 03.

## `zig build` Output Summary (head -30, filtered)

```
[100%] Built target llama-cli    <- llama.cpp cmake/make: complete
install
+- install zlx
   +- compile exe zlx Debug aarch64-macos 2 errors
src/inference/dequantize.zig:92:12: error: root source file struct 'mlx.zig.src.mlx' has no member named 'arrayIsEmpty'
src/inference/loader.zig:500:14: error: no field named 'weights' in struct 'mlx.zig.src.mla.MultiHeadLatentAttention'
Build Summary: 6/9 steps succeeded; 1 failed
```

No `pathJoin`, `LazyPath`, or `use of undefined identifier` build-script errors. The build reaches source compilation as required.

## Next Phase Readiness

- Phase 16 Plan 02 can now fix `arrayIsEmpty` (API name change in mlx.zig — use correct function name)
- Phase 16 Plan 03 can fix the `weights` field issue in `MultiHeadLatentAttention`
- Build infrastructure is unblocked for all remaining v2.0 phases

---
*Phase: 16-build-gap-closure*
*Completed: 2026-04-05*
