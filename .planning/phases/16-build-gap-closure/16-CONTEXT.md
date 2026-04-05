# Phase 16: Build & Gap Closure - Context

**Gathered:** 2026-04-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix all Zig 0.15.2 compile errors, undefined symbols, and struct mismatches so that `zig build` AND `zig build test` both succeed cleanly. Correctness of inference output is NOT in scope — stubs that compile are fine. The goal is a clean compilation baseline that unblocks Phases 17-20.

Requirements in scope: GAP-08, GAP-09, GAP-10

</domain>

<decisions>
## Implementation Decisions

### Submodule build.zig (src/mlx.zig/build.zig)
- **D-01:** Fix `src/mlx.zig/build.zig` directly in-place. It is already diverged from upstream and we are the only user. Track changes in the submodule — do not work around it from the top-level build.zig.

### mlx.arrayIsEmpty() fix
- **D-02:** Add a thin wrapper function `pub fn arrayIsEmpty(arr: Array) bool` to our MLX bindings layer, wrapping `mlx_array_size(arr) == 0` (or equivalent valid mlx-c API call). Do NOT remove the test assertions — they represent real intent and should survive Phase 16.

### MultiHeadLatentAttention struct mismatch
- **D-03:** Fix `loader.zig:499` to use the existing `MultiHeadLatentAttention.init(config, mlx_config)` constructor followed by `loadWeights(prefix, allocator)`. Do NOT use struct-literal initialization — the struct was designed for constructor use. The broken `{.weights = mla_weights}` site must be removed.
- **D-04:** The `mla.zig` struct definition is correct and must NOT be changed. It has fields: `w_dq`, `w_dkv`, `w_up`, `w_kr` (optional), `rope` (optional). The bug is exclusively in `loader.zig`.

### Success definition
- **D-05:** Phase 16 is complete when BOTH `zig build` and `zig build test` succeed without errors. Tests are allowed to use stubs (e.g., TurboQuant returning `error.NotImplemented`) as long as they compile and the test expects that error. Correctness of inference output is Phase 17.

### Claude's Discretion
- Exact mlx-c API call to use for array emptiness check (`mlx_array_size`, `mlx_array_nbytes`, or another) — verify against available mlx-c v0.4.x headers in the submodule.
- Whether any other `b.pathJoin` deprecation warnings need fixing in `build.zig` beyond the identified ones — fix any that fail compilation, ignore warnings-only if they don't block the build.
- Order of fixes within the phase — fix the submodule first (unblocks everything else), then symbol issues.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase Requirements
- `.planning/REQUIREMENTS.md` §v2.0 — GAP-08, GAP-09, GAP-10 definitions
- `.planning/ROADMAP.md` §Phase 16 — Success criteria (4 items)

### Affected Source Files
- `src/mlx.zig/build.zig` — submodule build script with old pathJoin API (fix directly)
- `build.zig` — top-level build script (check for any remaining deprecated calls)
- `src/test_integration.zig` — calls mlx.arrayIsEmpty() ~10 times (fix via wrapper)
- `src/inference/loader.zig:499` — wrong MLA struct initialization (fix to use init())
- `src/mlx.zig/src/mla.zig` — correct struct definition (reference only, do NOT modify)

### MLX.zig API Reference
- `src/mlx.zig/src/mlx.zig` — available public functions (arrayNew, arrayFree, zeros, arrayShape, etc.)
- `src/mlx.zig/src/mla.zig` — MultiHeadLatentAttention.init() and loadWeights() signatures

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/mlx.zig/src/mlx.zig` — existing pub fn pattern to follow when adding arrayIsEmpty wrapper (see arrayFree, arrayShape)
- `src/mlx.zig/src/mla.zig:137` — `init(config: MLAConfig, mlx_config: *mlx.MLXConfig) !*Self` — correct constructor signature
- `src/mlx.zig/src/mla.zig:173` — `loadWeights(prefix, allocator)` — loads w_dq/w_dkv/w_up/w_kr from weight set

### Established Patterns
- The submodule's `build.zig` already mixes old and new pathJoin styles — use `.cwd_relative = b.pathJoin(...)` pattern where LazyPath is required (already done correctly in some places)
- `mlx.arrayNew()` returns an uninitialized Array; `mlx.arrayFree()` frees it — wrapper should follow this pattern

### Integration Points
- `src/inference/loader.zig` is the bridge between safetensors weights and MLX model structs — MLA fix lives here
- `src/mlx.zig/src/mlx.zig` is where the arrayIsEmpty wrapper should be added (keeps it co-located with the rest of the MLX API surface)

</code_context>

<specifics>
## Specific Ideas

- The `mlx_array_size()` C function returns total element count; zero elements = empty array. This is the correct implementation for `arrayIsEmpty`.
- `loader.zig:499` should look like: `const mla_layer = try mla.MultiHeadLatentAttention.init(mla_config, mlx_config);` followed by `try mla_layer.loadWeights(layer_prefix, allocator);`

</specifics>

<deferred>
## Deferred Ideas

- `error.NotImplemented` in `src/compression/turboquant_stub.zig` and `src/compression/metal_kernels.zig` — Phase 19 (TurboQuant Metal) scope, not Phase 16.
- Fixing `src/tools/browser.zig` and `src/tools/python.zig` Zig 0.15.2 API breaks (`parseFree`) — Phase 20 (Tools API) scope.
- GPT-OSS forward() stub, tokenizer stubs, MLX backend factory stub — Phase 17 (Inference Gap Closure) scope.
- None — discussion stayed within phase scope.

</deferred>

---

*Phase: 16-build-gap-closure*
*Context gathered: 2026-04-05*
