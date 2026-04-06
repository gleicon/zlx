---
phase: 17-inference-gap-closure
plan: 06
subsystem: testing
tags: [zig, build, module-system, mlx, collision-fix]

requires:
  - phase: 17-05
    provides: prior inference gap work establishing module structure

provides:
  - shared_mlx_mod: single mlx module instance reused by all test sub-modules
  - module-collision-free zig build test: no "file exists in modules" errors remain
  - BackendType.mlx removed from test_models_gptoss.zig

affects:
  - 17-07: deepseek/integration test semantic errors now exposed for fixing
  - any future plan adding test targets: must use shared_mlx_mod pattern

tech-stack:
  added: []
  patterns:
    - "shared_mlx_mod pattern: create one b.createModule() for mlx before all test modules"
    - "named module extraction: extract file-relative imports to named modules to prevent split compilation units"
    - "dequantize_module: intermediate module to prevent deepseek.zig and loader.zig from each claiming dequantize.zig"

key-files:
  created: []
  modified:
    - build.zig
    - src/test_models_gptoss.zig

key-decisions:
  - "shared_mlx_mod: one b.createModule() for src/mlx.zig/src/mlx.zig created before all test module definitions"
  - "dequantize_module extracted: deepseek.zig imports @import(inference/dequantize.zig) and loader.zig imports dequantize.zig both as file-relative — extracting to named module prevents split compilation"
  - "inference_loader_mod has deepseek_mod and gpt_oss_mod wired under ../deepseek.zig and ../gpt_oss.zig — prevents loader.zig from creating file-relative copies"
  - "deepseek_test and integration_test now fail with semantic errors (import outside module path) not collision errors — acceptable per plan scope"

patterns-established:
  - "Pattern: shared_mlx_mod must be created before ALL test module definitions in build.zig"
  - "Pattern: any zig file that imports a file from a sibling directory (../X or relative/) risks collision if that file appears in multiple modules — extract to named module"

requirements-completed: [MODEL-01]

duration: 25min
completed: 2026-04-06
---

# Phase 17 Plan 06: Module Collision Fix Summary

**Single shared_mlx_mod instance wired to 7 test sub-modules via addImport, eliminating all "file exists in modules" collision errors from zig build test**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-04-06T17:07:00Z
- **Completed:** 2026-04-06T17:32:06Z
- **Tasks:** 4 (Tasks 1-3 executed together, Task 4 verified)
- **Files modified:** 2

## Accomplishments
- Eliminated ALL "file exists in modules" collision errors from `zig build test`
- Created `shared_mlx_mod` used by 17 references in build.zig
- Fixed `BackendType.mlx` → `BackendType.llama_cpp` in `test_models_gptoss.zig:255`
- `zig build` (main binary) still exits 0
- `registry_test 7/7` and `models_test 10/10` still pass

## Task Commits

1. **Tasks 1-4: shared_mlx_mod, module collision fix, BackendType.mlx fix, test verification** - `0dac417` (fix)

**Plan metadata:** (pending)

## Files Created/Modified
- `build.zig` - Added shared_mlx_mod; extracted named modules for moe, mla, deepseek, gpt_oss, dequantize, inference_loader, inference_mod; replaced 7+ inline mlx creates; added missing mlx import to mlx_gptoss_backend_mod
- `src/test_models_gptoss.zig` - BackendType.mlx → BackendType.llama_cpp at line 255

## Decisions Made
- `shared_mlx_mod` inserted before `// Test MoE module (PHASE-11-03)` block — earliest safe point before all test module definitions
- `dequantize_module` extracted as named shared module — prevents `deepseek.zig`'s file-relative `@import("inference/dequantize.zig")` and `loader.zig`'s `@import("dequantize.zig")` from each claiming the file
- `inference_loader_mod` registered under `"../deepseek.zig"` AND `"dequantize.zig"` to provide same module objects as `deepseek_mod` uses
- `inference_mod_module` does NOT re-register `"../deepseek.zig"` — avoids redundant registration that caused previous collision

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] dequantize.zig in two compilation units**
- **Found during:** Task 2 iteration — after initial shared_mlx_mod fix, deepseek.zig/loader.zig collision on dequantize.zig emerged
- **Issue:** `deepseek.zig` imports `"inference/dequantize.zig"` file-relatively. `loader.zig` imports `"dequantize.zig"` file-relatively. Same physical file in two modules.
- **Fix:** Extracted `dequantize_module` as a named shared module with `shared_mlx_mod` wired; registered under `"inference/dequantize.zig"` in `deepseek_mod` and `"dequantize.zig"` in `inference_loader_mod`
- **Files modified:** build.zig
- **Verification:** No "file exists in modules" errors for dequantize.zig
- **Committed in:** 0dac417

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug, iterative discovery)
**Impact on plan:** Required extra iteration beyond plan's described steps. All collision errors eliminated.

## zig build test Output (post-fix)

```
Build Summary: 18/33 steps succeeded; 8 failed; 48/48 tests passed; 6 leaked
+- run test registry_test 7/7 passed, 2 leaked
+- run test models_test 10/10 passed, 4 leaked
+- run test backend_integration_test transitive failure
+- run test moe_test transitive failure
+- run test deepseek_test transitive failure
+- run test integration_test transitive failure
+- run test gptoss_test transitive failure
+- run test gptoss_manager_test transitive failure
```

Remaining failures are semantic (import of file outside module path, C header not found) — NOT collision errors.

## Known Stubs

None introduced in this plan.

## Issues Encountered
- The collision fix required 4 iterations: (1) shared_mlx_mod for top-level creates, (2) extracting moe_mod/deepseek_mod/gpt_oss_mod, (3) creating inference_loader_mod/inference_mod_module, (4) extracting dequantize_module. Each iteration revealed a new transitive collision that was resolved by making more file-relative imports into named modules.
- `gptoss_manager_test` fails with `import of file outside module path` for `mlx_gptoss_backend.zig:10` (`tokenizer.zig`) — this is a missing named dep, not a collision. Out of scope for this plan.

## Next Phase Readiness
- No collision errors remain — tests now fail with semantic errors that can be fixed per-plan
- `zig build` main binary passes
- `shared_mlx_mod` pattern established — future test targets MUST use this single shared instance

---
*Phase: 17-inference-gap-closure*
*Completed: 2026-04-06*

## Self-Check: PASSED

- FOUND: 17-06-SUMMARY.md
- FOUND: build.zig
- FOUND: test_models_gptoss.zig
- FOUND: commit 0dac417
