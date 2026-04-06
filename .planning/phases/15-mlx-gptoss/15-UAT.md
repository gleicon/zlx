---
status: testing
phase: 15-mlx-gptoss
source: [15-06-SUMMARY.md, 15-07-SUMMARY.md, 15-08-SUMMARY.md]
started: 2026-04-04T00:00:00Z
updated: 2026-04-04T00:00:00Z
---

## Current Test

number: 1
name: Cold Start Smoke Test (zig build succeeds)
expected: |
  Run `zig build` from the repo root. Build should complete without any compile errors.
  The fixes in 15-06 removed parseFree, stringifyAlloc, and LazyPath.path usages that
  were breaking Zig 0.13.0 compilation. A clean build (no errors, binary produced) confirms
  the API compatibility fixes are in place.
awaiting: user response

## Tests

### 1. Cold Start Smoke Test (zig build succeeds)
expected: Run `zig build` from repo root. No compile errors. Binary produced.
result: issue
reported: "Build fails. Multiple compile errors. Root cause: Zig 0.15.2 installed, project requires 0.13.0. std.ArrayList and std.json.stringify APIs changed incompatibly. Additional pre-existing bugs: mlx.arrayIsEmpty() does not exist in MLX.zig API; loader.zig initializes MultiHeadLatentAttention with a .weights field that doesn't exist in the struct."
severity: blocker

### 2. GPT-OSS chat route dispatches to ChatGPTOSSHandler
expected: |
  POST /v1/chat/completions with model=gptoss-20b routes to ChatGPTOSSHandler (not 404).
result: blocked
blocked_by: server
reason: "Build fails (test 1). Cannot run server."

### 3. Non-GPT-OSS chat still routes correctly
expected: |
  POST /v1/chat/completions with non-GPT-OSS model routes to original handler.
result: blocked
blocked_by: server
reason: "Build fails (test 1). Cannot run server."

### 4. Browser tool route responds (not 404)
expected: POST /v1/tools/browser returns non-404 response.
result: blocked
blocked_by: server
reason: "Build fails (test 1). Cannot run server."

### 5. Python tool route responds (not 404)
expected: POST /v1/tools/python returns non-404 response.
result: blocked
blocked_by: server
reason: "Build fails (test 1). Cannot run server."

### 6. No deprecated API calls in source (grep verification)
expected: |
  grep -rn "parseFree|stringifyAlloc|\.path = b\.pathJoin" src/ build.zig → zero matches.
result: pass

## Summary

total: 6
passed: 1
issues: 1
pending: 0
skipped: 0
blocked: 4

## Gaps

- truth: "zig build completes with no compile errors and produces a binary"
  status: failed
  reason: "User reported: Build fails. Zig 0.15.2 installed but project requires 0.13.0. std.ArrayList API changed (unmanaged by default, no .init(allocator)). std.json.stringify removed. Pre-existing bugs: mlx.arrayIsEmpty() does not exist in MLX.zig API; MultiHeadLatentAttention has no .weights field in mla.zig but loader.zig sets it."
  severity: blocker
  test: 1
  root_cause: "1) Zig version mismatch (0.15.2 vs required 0.13.0) causing ArrayList and json API breaks across ~27 files. 2) mlx.arrayIsEmpty() called in dequantize.zig and test_integration.zig but never defined in MLX.zig API. 3) loader.zig:499 initializes MultiHeadLatentAttention{.weights=...} but struct has no .weights field — MLA struct changed between design and implementation."
  artifacts:
    - path: "src/inference/dequantize.zig:92"
      issue: "mlx.arrayIsEmpty() does not exist — use C.mlx_array_size(arr)==0"
    - path: "src/inference/loader.zig:499"
      issue: "MultiHeadLatentAttention has no .weights field — struct has w_dq, w_dkv, w_up, w_kr, rope"
    - path: "src/api/tools.zig"
      issue: "std.json.stringify removed in 0.15.2 — fixed to Stringify.valueAlloc but blocked by ArrayList shim complexity"
  missing:
    - "Install Zig 0.13.0 OR complete full 0.15.2 port (27 files, ArrayList+json+io.Writer API)"
    - "Add arrayIsEmpty wrapper to mlx.zig bindings"
    - "Fix loader.zig MLA initialization to match actual MultiHeadLatentAttention struct fields"
