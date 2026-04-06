---
phase: 17-inference-gap-closure
plan: "01"
subsystem: inference
tags: [cleanup, dead-code-removal, speculation, zero-dead-code]
dependency_graph:
  requires: []
  provides: [clean-inference-layer, no-speculation-symbols]
  affects: [src/main.zig, src/config.zig, src/inference/generator.zig, src/inference/mod.zig, src/api/streaming.zig]
tech_stack:
  added: []
  patterns: [zero-dead-code-policy]
key_files:
  created: []
  modified:
    - src/main.zig
    - src/config.zig
    - src/inference/generator.zig
    - src/inference/mod.zig
    - src/api/streaming.zig
    - src/config_test.zig
  deleted:
    - src/speculation/mod.zig
    - src/speculation/speculative_generator.zig
    - src/speculation/draft_selector.zig
    - src/speculation/integration_test.zig
    - src/speculation/speculative_generator_test.zig
decisions:
  - "D-03: Removed all speculative decoding code — zero dead code policy enforced"
  - "Removal comment added at every deletion site for future re-evaluation"
metrics:
  duration: "~15 minutes"
  completed: "2026-04-06T00:16:17Z"
  tasks_completed: 2
  files_modified: 6
  files_deleted: 5
  symbols_removed: 25+
requirements: [GAP-07]
---

# Phase 17 Plan 01: Remove All Speculative Decoding Code Summary

**One-liner:** Removed all 25+ speculation symbols from 6 files, deleted the 5-file `src/speculation/` directory, enforcing zero dead code policy D-03.

## What Was Done

The speculative decoding subsystem was scaffolded in Phase 14-15 but never produced real output. The `SpeculativeGenerator.init()` was never called with a real draft model — all call sites passed `null` and `0` for `draft_model_ref` and `speculation_depth`. Keeping this dead code raised complexity without benefit and made future audits harder. Per decision D-03, all speculation code was removed and `src/speculation/` deleted.

## Files Modified

| File | Changes |
|------|---------|
| `src/main.zig` | Removed `@import("speculation/mod.zig")`, 3 CLI flags (--draft-model, --speculation-depth, --no-speculation), 3 Config struct fields, USAGE entries, speculation init block, log block, convertFileConfig() assignments |
| `src/config.zig` | Removed draft_model/speculation_depth/no_speculation from Config and FileConfig structs, mergeConfig() assignments, 3 env var blocks, validation block, ConfigError.InvalidSpeculationDepth |
| `src/inference/generator.zig` | Removed speculation/draft_model imports, struct fields (speculative_generator, use_speculation, draft_model), init() params (draft_model_ref, speculation_depth), init block, next() speculation branch, deinit() speculation cleanup, generateAll() params |
| `src/inference/mod.zig` | Removed deepseek_v2_moe arm from generateWithTimeout() switch, null args from 2 GenerationState.init() call sites, unreachable DeepSeekNotImplemented return, unused buf variable |
| `src/api/streaming.zig` | Removed null draft_model/speculation_depth args from 2 GenerationState.init() call sites |
| `src/config_test.zig` | Removed test assertions for deleted config fields (Rule 1 auto-fix) |

## Files Deleted

- `src/speculation/mod.zig`
- `src/speculation/speculative_generator.zig`
- `src/speculation/draft_selector.zig`
- `src/speculation/integration_test.zig`
- `src/speculation/speculative_generator_test.zig`

## Symbols Removed

From `src/main.zig`:
- `const speculation = @import("speculation/mod.zig")`
- `Config.draft_model`, `Config.speculation_depth`, `Config.no_speculation` fields
- `--draft-model`, `--speculation-depth`, `--no-speculation` CLI parse blocks
- USAGE string lines for those 3 flags and example line
- Log block for speculation settings in `parseArgs()`
- 3 `convertFileConfig()` field assignments
- Entire speculation init block (~30 lines): `speculation.initSpeculation()`, `speculation.configure()`, `speculation.isInitialized()` calls

From `src/config.zig`:
- `ConfigError.InvalidSpeculationDepth`
- `Config.draft_model`, `Config.speculation_depth`, `Config.no_speculation` fields
- Same 3 fields from `FileConfig` struct
- 3 `mergeConfig()` assignments
- `ZLX_DRAFT_MODEL`, `ZLX_SPECULATION_DEPTH`, `ZLX_NO_SPECULATION` env var blocks
- `speculation_depth` validation block

From `src/inference/generator.zig`:
- `@import("../speculation/speculative_generator.zig")` and `@import("../models/draft_model.zig")`
- `GenerationState.speculative_generator`, `.use_speculation`, `.draft_model` struct fields
- `GenerationState.init()` params: `draft_model_ref`, `speculation_depth`
- Entire speculation initialization branch in `init()` (~35 lines)
- Speculation branch in `next()` (~45 lines)
- Speculation cleanup in `deinit()`
- `generateAll()` params: `draft_model_ref`, `speculation_depth`

From `src/inference/mod.zig`:
- `.deepseek_v2_moe` match arm in `generateWithTimeout()` (~18 lines)
- `null, // draft_model` and `0, // speculation_depth` from 2 `GenerationState.init()` calls
- `return error.DeepSeekNotImplemented` unreachable line
- Unused `buf: [512]u8` variable

From `src/api/streaming.zig`:
- `null, // draft_model` and `0, // speculation_depth` from 2 `GenerationState.init()` calls

## Build Verification

After all changes: `zig build` produces no speculation-related errors. The only errors are pre-existing:
- `src/api/chat_gptoss.zig:115` — `res.write()` member function arg count (pre-existing, unrelated to speculation removal; was previously hidden behind earlier errors in compilation order)

These pre-existing errors are tracked for Phase 17 subsequent plans.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed config_test.zig references to removed config fields**
- **Found during:** Task 2 verification
- **Issue:** `src/config_test.zig` contained test assertions for `cfg.speculation_depth`, `cfg.no_speculation`, and `cfg.draft_model` — fields deleted from `config.Config`. These would fail at test compilation.
- **Fix:** Removed the 4 test assertion lines referencing deleted fields from tests 1 and 7.
- **Files modified:** `src/config_test.zig`
- **Commit:** f371c87

### Out-of-scope discoveries (deferred)

The following files contain speculation-related code but are NOT in the inference path and were not breaking the build:
- `src/metrics/speculative_metrics.zig` — standalone metrics module, not imported by any other file
- `src/models/draft_model.zig` — imports deleted `../speculation/draft_selector.zig` but not imported by any other file

These orphaned files are logged to deferred-items for cleanup in a future plan.

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| Task 1 | 9fd61be | Remove speculation fields/imports from main.zig and config.zig |
| Task 2 | f371c87 | Remove speculation from generator.zig, mod.zig, streaming.zig; delete src/speculation/ |

## Known Stubs

None introduced by this plan. This was a pure deletion plan.

## Self-Check: PASSED

- FOUND: src/main.zig
- FOUND: src/config.zig
- FOUND: src/inference/generator.zig
- FOUND: src/inference/mod.zig
- FOUND: src/api/streaming.zig
- CONFIRMED: src/speculation/ does not exist
- FOUND: commit 9fd61be (Task 1)
- FOUND: commit f371c87 (Task 2)
- CONFIRMED: `grep -r "speculation" src/ --include="*.zig" | grep -v "removed"` returns only speculative_metrics.zig and draft_model.zig (orphaned, not in build path)
