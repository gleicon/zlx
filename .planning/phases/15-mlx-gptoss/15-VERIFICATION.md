---
phase: 15-mlx-gptoss
verified: 2026-04-04T12:18:25Z
status: gaps_found
score: 3/6 success criteria verified
re_verification: false
gaps:
  - truth: "GPT-OSS-20B runs at 30+ tokens/sec via native MLX (not llama.cpp)"
    status: failed
    reason: "Forward pass and MoE layer are confirmed stubs — gptoss.zig returns zero logits and token 1 repeatedly. Metal kernel dispatch is scaffolded but not wired (requires mlx-c v0.4.x which the project pins to v0.1.2). No actual inference is possible."
    artifacts:
      - path: "src/mlx.zig/src/gptoss.zig"
        issue: "forward() returns zero logits (line 359-362 stub), applyMoE() returns zeros (line 375-385 stub), generate() always returns token 1 (line 411 stub), initMetalKernels() is empty"
      - path: "src/backends/mlx_gptoss_backend.zig"
        issue: "tokenize() returns empty slice (line 276-279 stub), load() passes stream: undefined to GPTOSSTransformer.init()"
    missing:
      - "Real token embedding lookup in GPTOSSTransformer.forward()"
      - "Real MoE routing computation in applyMoE() — currently all zeros"
      - "Real sampling from logits in generate() — currently hardcoded token 1"
      - "Metal kernel dispatch via mlx-c v0.4.x OR a fallback MLX primitive path"
      - "GPT-OSS tokenizer (tiktoken/Qwen vocab) wired into tokenize()"
      - "Real MLX GPU stream in mlx_gptoss_backend.load()"

  - truth: "Browser and Python tools work end-to-end"
    status: failed
    reason: "Tools files use removed Zig 0.15.2 APIs (std.json.parseFree, std.json.stringifyAlloc) which will cause compile errors. ToolsAPI and ChatGPTOSSHandler are orphaned — neither is imported by main.zig, api/server.zig, or api/handlers.zig. No HTTP route at /v1/tools/* or /v1/chat/completions that dispatches to GPT-OSS exists in the running server."
    artifacts:
      - path: "src/tools/browser.zig"
        issue: "Uses std.json.parseFree (line 187) — removed in Zig 0.15.2"
      - path: "src/tools/python.zig"
        issue: "Uses std.json.parseFree (line 211) — removed in Zig 0.15.2"
      - path: "src/api/tools.zig"
        issue: "Uses std.json.parseFree (lines 57, 104) and std.json.stringifyAlloc (lines 60, 107) — both removed in Zig 0.15.2. File is orphaned: never imported by server routing."
      - path: "src/api/chat_gptoss.zig"
        issue: "ChatGPTOSSHandler is not imported or registered in main.zig, api/server.zig, or api/handlers.zig — no HTTP route dispatches to it"
    missing:
      - "Fix std.json.parseFree → parsed.deinit() in browser.zig, python.zig, api/tools.zig"
      - "Fix std.json.stringifyAlloc → std.json.fmt pattern in api/tools.zig"
      - "Wire ChatGPTOSSHandler.handle() into server routing for GPT-OSS model requests"
      - "Wire ToolsAPI into server routes at /v1/tools/browser and /v1/tools/python"

  - truth: "`./test_models.sh gptoss` passes"
    status: failed
    reason: "test_models.sh test_gptoss() function exists and would dispatch to the server, but the binary cannot serve GPT-OSS requests because: (a) the tokenizer stub returns empty slices, (b) generate() returns token 1 forever, and (c) build.zig has a pre-existing .path deprecation error on lines 113-118 that may block compilation on Zig 0.15.2. The test also requires the 11GB model to be present at ./models/gpt-oss-20b-MXFP4-Q4/."
    artifacts:
      - path: "build.zig"
        issue: "Lines 113-118 use .path = b.pathJoin(...) which was renamed to .cwd_relative in Zig 0.15.2 — blocks compilation"
    missing:
      - "Fix build.zig .path → .cwd_relative on lines 113, 114, 117, 118"
      - "Functional forward pass and generate loop (blocked by stub truth #1 above)"
      - "Tokenizer wired so prompt context is actually tokenized"

  - truth: "OpenCode integration works with native MLX backend"
    status: failed
    reason: "ChatGPTOSSHandler is not wired into any server route. main.zig only imports api/server.zig and api/handlers.zig; neither references chat_gptoss.zig or selectBackend(). GPT-OSS requests arriving at /v1/chat/completions would be handled by the generic handler, not the Harmony-aware GPT-OSS handler."
    artifacts:
      - path: "src/main.zig"
        issue: "Does not import chat_gptoss, gptoss_manager, or call selectBackend() — no routing to MLX GPT-OSS backend"
    missing:
      - "Import ChatGPTOSSHandler in main.zig or server.zig"
      - "Route GPT-OSS model requests to ChatGPTOSSHandler based on selectBackend() result"
      - "Register GPTOSSModelManager in server lifecycle"

human_verification:
  - test: "Performance benchmark: 30+ tokens/sec"
    expected: "GPT-OSS-20B inference achieves 30+ tokens/sec on Apple Silicon with real weights loaded"
    why_human: "Requires actual model weights (11GB), working MLX build, and timed measurement — cannot verify programmatically without running the server"
  - test: "Docker sandbox Python execution"
    expected: "PythonTool.execute() spawns Docker container with --network=none, runs code, returns stdout"
    why_human: "Requires Docker runtime at test time; cannot verify sandbox behavior statically"
---

# Phase 15: Native MLX GPT-OSS Verification Report

**Phase Goal:** Implement high-performance GPT-OSS support using native MLX (like openharmony-mlx), replacing the llama.cpp approach with a native Zig/MLX implementation that achieves 40 tokens/sec on Apple Silicon.
**Verified:** 2026-04-04T12:18:25Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (from ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | GPT-OSS-20B runs at 30+ tokens/sec via native MLX (not llama.cpp) | ✗ FAILED | forward() returns zero logits, generate() hardcodes token 1, Metal dispatch requires mlx-c v0.4.x which is not available |
| 2 | Harmony chat format is correctly parsed and formatted | ✓ VERIFIED | harmony.zig (207 lines), parser.zig (403 lines), template.zig (289 lines) all present; formatHarmonyChat() wired in chat_gptoss.zig:212; 23 tests claimed passing |
| 3 | Browser and Python tools work end-to-end | ✗ FAILED | browser.zig/python.zig/api/tools.zig use removed Zig 0.15.2 APIs (parseFree, stringifyAlloc); ToolsAPI orphaned — not wired to any server route |
| 4 | MXFP4 weights load and decompress correctly | ✓ VERIFIED | mxfp4.zig exports MXFP4Tensor, dequantizeMXFP4GPU, safetensorsToMLX; gptoss_loader.zig imports and calls mxfp4.safetensorsToMLX on MXFP4 tensors; safetensors.zig + gptoss_loader.zig fully wired |
| 5 | `./test_models.sh gptoss` passes | ✗ FAILED | Script exists with test_gptoss() function, but cannot pass: build.zig has .path deprecation error blocking compilation; stubs in generate() produce token 1 only |
| 6 | OpenCode integration works with native MLX backend | ✗ FAILED | ChatGPTOSSHandler not wired into server routing; main.zig does not import chat_gptoss.zig; selectBackend() exists but is never called from the request path |

**Score:** 2/6 success criteria fully verified (Harmony format, MXFP4 loading)

*Note: A third truth — "GPTOSSTransformer compiles with correct architecture and integrates into TransformerUnion" — is verified at the structural level. llm.zig imports GPTOSSTransformer and routes "gptoss" model type to it. However, the transformer's computation functions are stubs, so runtime inference is not functional.*

### Required Artifacts (from MASTER PLAN must_haves)

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/mlx.zig/src/gptoss.zig` | GPT-OSS transformer architecture | ✓ EXISTS / ✗ HOLLOW | 558 lines, has GPTOSSTransformer struct, MoERouter, GPTOSSTokenGenerator. forward() returns zero logits (stub, line 359-362), applyMoE() returns zeros (stub, line 375-385), generate() returns token 1 (stub, line 411). Compilation requires MLX build infrastructure. |
| `src/mlx.zig/src/gptoss_metal.metal` | Metal kernels for GPT-OSS | ✓ VERIFIED (source) / ✗ NOT DISPATCHED | 418 lines; has gptoss_moe_route (line 30), gptoss_moe_apply (line 115), gptoss_sw_attention (line 207). Metal source is complete but dispatch requires mlx-c v0.4.x; project is pinned to v0.1.2. |
| `src/harmony/` directory | Harmony format parser and chat template | ✓ VERIFIED | harmony.zig (207), parser.zig (403), template.zig (289), types.zig (17), harmony_test.zig (470). HarmonyEncoding and HarmonyConversation present. formatHarmonyChat() wired in chat_gptoss.zig. |
| `src/tools/browser.zig` | Browser tool implementation | ✗ STUB (compile broken) | 352 lines; BrowserTool.search/open/find present. Uses std.json.parseFree (line 187) — removed in Zig 0.15.2, will not compile. Orphaned from server routing. |
| `src/tools/python.zig` | Python tool implementation | ✗ STUB (compile broken) | 280 lines; PythonTool.execute present. Uses std.json.parseFree (line 211) — same Zig 0.15.2 issue. |
| `src/weight/gptoss_loader.zig` | GPT-OSS weight loading with MXFP4 | ✓ VERIFIED | 260 lines; GPTOSSWeightLoader present; loadMXFP4Tensor() calls mxfp4.safetensorsToMLX on f4_e2m1 dtype tensors. loadIntoTransformer() stub exists (ready for Phase 15-05 wiring). |
| `src/mxfp4.zig` | MXFP4 dequantization | ✓ VERIFIED | 217 lines; MXFP4Tensor, dequantizeMXFP4GPU (line 106), safetensorsToMLX (line 194). Note: MASTER plan expected function named `dequantizeMXFP4` — actual name is `dequantizeMXFP4GPU`; functionally equivalent. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `src/mlx.zig/src/gptoss.zig` | `src/mlx.zig/src/gptoss_metal.metal` | FastMetalKernel API (`mlx_fast_metal_kernel`) | ✗ NOT_WIRED | Metal source exists; initMetalKernels() is empty. Dispatch commented out — requires mlx-c v0.4.x which is not available (project pins v0.1.2). |
| `src/harmony/harmony.zig` | `src/api/chat.zig` | `formatHarmonyChat` / `HarmonyChat.format` | ✗ WRONG TARGET | `src/api/chat.zig` does not exist. Harmony IS wired to `src/api/chat_gptoss.zig` (line 212) — different file. Core wiring exists but at different path than plan specified. |
| `src/tools/browser.zig` | `src/api/tools.zig` | HTTP endpoint `/v1/tools/browser` via `browserToolHandler` | ⚠️ PARTIAL | browserToolHandler exists in tools.zig; browser.zig has asTool()/executeToolCall(). However, tools.zig itself is orphaned — no server registers the /v1/tools/browser route. |
| `src/mxfp4.zig` | `src/weight/gptoss_loader.zig` | `dequantizeMXFP4` function | ✓ WIRED | gptoss_loader.zig imports mxfp4 (line 6) and calls `mxfp4.safetensorsToMLX` (line 240) for MXFP4 tensors. Function name differs (dequantizeMXFP4GPU not dequantizeMXFP4) but the link is real and functional. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `src/mlx.zig/src/gptoss.zig` forward() | `logits` (mlx.Array) | Token embeddings → transformer layers → lm_head | No — `try mlx.zeros(&logits, ...)` always returns zero array | ✗ DISCONNECTED |
| `src/mlx.zig/src/gptoss.zig` generate() | output tokens | logits → sampling | No — `return &[_]u32{1}` hardcoded | ✗ DISCONNECTED |
| `src/backends/mlx_gptoss_backend.zig` tokenize() | prompt tokens | text input | No — `return allocator.alloc(u32, 0)` (empty) | ✗ DISCONNECTED |
| `src/api/chat_gptoss.zig` handle() | `harmony_prompt` | HarmonyTemplate.formatHarmonyChat | Yes — real Harmony formatting | ✓ FLOWING |
| `src/weight/gptoss_loader.zig` loadMXFP4Tensor() | MLX array from MXFP4 data | safetensors bytes → mxfp4.safetensorsToMLX | Yes — real dequantization logic | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| GPTOSSTransformer in TransformerUnion | `grep -q 'gptoss: GPTOSSTransformer' src/mlx.zig/src/llm.zig` | Pattern found at line 76 | ✓ PASS |
| selectBackend routes gpt-oss to mlx_gptoss | `grep -q 'gpt-oss.*mlx_gptoss' src/backends/backend.zig` | Pattern found at line 121 | ✓ PASS |
| binary compiles (check build blocker) | Check build.zig lines 113-118 for .path field | `.path = b.pathJoin(...)` found on lines 113, 114, 117 — invalid in Zig 0.15.2 | ✗ FAIL |
| ToolsAPI wired into server | `grep -rn 'ToolsAPI' src/main.zig src/api/server.zig src/api/handlers.zig` | No matches — ToolsAPI orphaned | ✗ FAIL |
| ChatGPTOSSHandler wired into server | `grep -rn 'chat_gptoss\|ChatGPTOSSHandler' src/main.zig src/api/` | Only self-reference in chat_gptoss.zig | ✗ FAIL |

Step 7b: Skipped for full server launch. Behavioral checks done via static analysis above.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| GPTOSS-01 | 15-01 | GPTOSSTransformer with MoE and sliding window attention compiles and integrates into TransformerUnion | ✓ PARTIAL | Structure present; TransformerUnion wired; Metal kernel dispatch and forward pass are stubs |
| GPTOSS-02 | 15-02 | Harmony format parser and chat template work correctly | ✓ SATISFIED | harmony.zig, parser.zig, template.zig all present and wired in chat_gptoss.zig |
| GPTOSS-03 | 15-03 | Browser and Python tools implemented with HTTP API endpoints | ✗ BLOCKED | Tools files use removed Zig 0.15.2 JSON APIs; ToolsAPI not wired into server routing |
| GPTOSS-04 | 15-04 | MXFP4 weight loading and dequantization | ✓ SATISFIED | mxfp4.zig and gptoss_loader.zig fully wired; safetensors parsing and MXFP4 dequant implemented |
| GPTOSS-05 | 15-05 | Integration with zlx server — end-to-end GPT-OSS inference via /v1/chat/completions | ✗ BLOCKED | ChatGPTOSSHandler orphaned from server routing; generate() is a stub; tokenizer not wired |

**Note on REQUIREMENTS.md:** The REQUIREMENTS.md file does not contain GPTOSS-01 through GPTOSS-05 entries — these requirements exist only in the phase plans and are not tracked in the central requirements registry. This is a traceability gap (requirements defined in phase plans but not registered in .planning/REQUIREMENTS.md), though it does not affect the verification outcome.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/mlx.zig/src/gptoss.zig` | 359-362 | `try mlx.zeros(&logits, ...)` — forward() always returns zero logits | BLOCKER | Token generation produces meaningless output |
| `src/mlx.zig/src/gptoss.zig` | 375-385 | `try mlx.zeros(output, ...)` — applyMoE() always returns zeros | BLOCKER | MoE computation is bypassed entirely |
| `src/mlx.zig/src/gptoss.zig` | 411 | `return &[_]u32{1}` — hardcoded token 1 in generate() | BLOCKER | Inference produces token 1 indefinitely |
| `src/backends/mlx_gptoss_backend.zig` | 276-279 | `return allocator.alloc(u32, 0)` — tokenize stub | BLOCKER | All prompts tokenized as empty; no context passed to model |
| `src/backends/mlx_gptoss_backend.zig` | ~load() | `stream: undefined` passed to transformer init | BLOCKER | Will crash at runtime when weights loaded |
| `src/tools/browser.zig` | 187 | `std.json.parseFree` — removed in Zig 0.15.2 | BLOCKER | Compile error when build.zig is fixed |
| `src/tools/python.zig` | 211 | `std.json.parseFree` — removed in Zig 0.15.2 | BLOCKER | Compile error when build.zig is fixed |
| `src/api/tools.zig` | 57, 104 | `std.json.parseFree` — removed in Zig 0.15.2 | BLOCKER | Compile error when build.zig is fixed |
| `src/api/tools.zig` | 60, 107 | `std.json.stringifyAlloc` — removed in Zig 0.15.2 | BLOCKER | Compile error when build.zig is fixed |
| `build.zig` | 113, 114, 117 | `.path = b.pathJoin(...)` — renamed to `.cwd_relative` in Zig 0.15.2 | BLOCKER | Prevents `zig build` from completing (pre-existing, from Phase 14) |
| `src/api/chat_gptoss.zig` | (whole file) | `ChatGPTOSSHandler` never imported by server routing | BLOCKER | GPT-OSS chat endpoint unreachable |
| `src/api/tools.zig` | (whole file) | `ToolsAPI` never imported by server routing | BLOCKER | /v1/tools/* endpoints do not exist |

### Human Verification Required

#### 1. Performance: 30+ tokens/sec on Apple Silicon

**Test:** After stubs are resolved and real weights loaded, run benchmark with `./test_models.sh --benchmark --gptoss`
**Expected:** GPT-OSS-20B generates 30+ tokens/sec measured via TPS column in benchmark output
**Why human:** Requires 11GB model weights, working MLX build, and timed measurement

#### 2. Docker sandbox isolation for Python tool

**Test:** Call POST /v1/tools/python with `{"code": "import os; print(os.getcwd())"}` after wiring is complete
**Expected:** Code executes in Docker container; output is the container workdir, not host filesystem
**Why human:** Requires Docker runtime; cannot verify container isolation statically

### Gaps Summary

Phase 15 achieved its structural goals: all 9 required files were created, TransformerUnion includes GPTOSSTransformer, the Harmony format system is complete and tested, and MXFP4 dequantization is fully implemented and wired to weight loading.

However, **4 of 6 success criteria are not met** due to three distinct gap categories:

**Gap 1 — Inference stubs (blocks truths 1, 5, 6):** The `gptoss.zig` forward pass, MoE application, and generation loop are intentional stubs that return zeros and hardcoded tokens. The Metal kernel dispatch was blocked by the mlx-c v0.1.2 pin. This was documented in the SUMMARY as intentional, but the phase goal requires actual inference — not just structural scaffolding. Without real computation, 30 tokens/sec is impossible and test_models.sh cannot pass.

**Gap 2 — Orphaned integration files (blocks truths 3, 6):** `ChatGPTOSSHandler` and `ToolsAPI` are fully implemented but never imported or registered with the HTTP server. Requests to `/v1/chat/completions` for GPT-OSS models and `/v1/tools/*` do not reach these handlers. This is a wiring gap in `main.zig` / `api/server.zig`.

**Gap 3 — Compile-breaking API calls (blocks truth 3):** Three tools files use `std.json.parseFree` and `std.json.stringifyAlloc` which were removed in Zig 0.15.2. This is in addition to the pre-existing `build.zig` `.path` deprecation from Phase 14. Until fixed, the tools module cannot compile.

The Harmony format (truth 2) and MXFP4 loading (truth 4) are genuinely complete and represent real implementation value from this phase.

---

_Verified: 2026-04-04T12:18:25Z_
_Verifier: Claude (gsd-verifier)_
