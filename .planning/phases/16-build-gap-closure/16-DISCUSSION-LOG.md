# Phase 16: Build & Gap Closure - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-05
**Phase:** 16-build-gap-closure
**Areas discussed:** Submodule build.zig, arrayIsEmpty fix, MLA struct fix, Clean build definition

---

## Submodule build.zig

| Option | Description | Selected |
|--------|-------------|----------|
| Fix it directly | Edit src/mlx.zig/build.zig in place — already diverged from upstream, we are the only user | ✓ |
| Leave submodule alone | Only fix top-level build.zig, work around submodule | |
| Check if it's even broken | Verify first whether Zig 0.15.2 still accepts the old calls | |

**User's choice:** Fix it directly
**Notes:** Submodule is already modified from upstream; fixing in place is consistent with the project's approach.

---

## arrayIsEmpty fix

| Option | Description | Selected |
|--------|-------------|----------|
| Add a thin wrapper | Define mlx.arrayIsEmpty() wrapping mlx_array_size()==0 — keeps test assertions intact | ✓ |
| Replace calls inline | Replace each call with mlx.arraySize(x) == 0 directly in test file | |
| Remove the assertions | Delete assertions entirely — rewrite in Phase 17 | |

**User's choice:** Add a thin wrapper
**Notes:** Preserves test intent. Wrapper lives in src/mlx.zig/src/mlx.zig alongside existing API surface.

---

## MLA struct fix

| Option | Description | Selected |
|--------|-------------|----------|
| Fix loader.zig to use init() | Call MultiHeadLatentAttention.init() then loadWeights() — uses existing correct API | ✓ |
| Remove MLA from loader.zig | Delete broken init site, leave TODO for Phase 17 | |

**User's choice:** Fix loader.zig to use init()
**Notes:** MLA struct definition in mla.zig is correct. Bug is exclusively in loader.zig:499.

---

## Clean build definition

| Option | Description | Selected |
|--------|-------------|----------|
| zig build + zig build test | Both must pass. Stubs are fine if they compile correctly. | ✓ |
| zig build only | Binary compilation only — test failures are Phase 17 scope | |

**User's choice:** zig build + zig build test
**Notes:** Tests using stubs are acceptable (e.g., TurboQuant returning error.NotImplemented when the test expects that). Correctness is Phase 17.

---

## Claude's Discretion

- Exact mlx-c API for arrayIsEmpty implementation
- Whether additional pathJoin deprecations need fixing beyond identified ones
- Order of fixes within the phase

## Deferred Ideas

- TurboQuant error.NotImplemented stubs → Phase 19
- browser.zig / python.zig parseFree fixes → Phase 20
- Inference stubs (GPT-OSS forward, tokenizer) → Phase 17
