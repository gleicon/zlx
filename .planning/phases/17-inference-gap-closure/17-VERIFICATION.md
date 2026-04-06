---
phase: 17-inference-gap-closure
verified: 2026-04-05T21:00:00Z
status: gaps_found
score: 6/10 requirements verified
gaps:
  - truth: "GAP-06: loadIndex() reads index.json from disk and populates self.entries on cache init"
    status: failed
    reason: "loadIndex() still contains '// TODO: Parse index.json and populate entries'. Commit 757d3d0 (feat(17-05)) was made on branch worktree-agent-a7f5ba59 and never merged into Deepseek. The docs commit ccdb38d was cherry-picked but the implementation commit was not."
    artifacts:
      - path: "src/cache/prompt_cache.zig"
        issue: "Line 433 still reads: // TODO: Parse index.json and populate entries. openFileAbsolute and parseFromSlice are only present in saveIndex(), not loadIndex()."
    missing:
      - "Merge or cherry-pick commit 757d3d0 from worktree-agent-a7f5ba59 into Deepseek branch, or re-implement loadIndex() on this branch"

  - truth: "GAP-03 (REQUIREMENTS.md): GPT-OSS native MLX forward pass implemented with real tensor computation — not returning mlx.zeros()"
    status: failed
    reason: "gptoss_mlx.zig:109 still calls mlx.zeros() returning a zero-logit tensor. Plan 03 addressed DeepSeek dispatch (not named GAP-03 in REQUIREMENTS.md). Plan 04 only fixed EOS termination so the loop runs max_tokens instead of 0 — but forward() still returns zeros, so every generated token is token 0 (argmax of zero tensor), not real output."
    artifacts:
      - path: "src/gptoss_mlx.zig"
        issue: "Line 109: mlx.zeros(&logits, &shape, mlx.FLOAT32, self.stream) — no weight fields, no real attention/FFN computation. Produces max_tokens copies of token 0, not real output tokens."
    missing:
      - "Implement GPTOSSTransformer.forward() with real weight fields (embedding, attention layers, FFN) so it produces non-zero logits from actual model weights"
      - "This is explicitly documented as future work in the plan but is required by REQUIREMENTS.md GAP-03 and MODEL-03"

  - truth: "MODEL-01: zig build test produces passing integration tests (Qwen2.5-Coder end-to-end)"
    status: failed
    reason: "zig build test shows multiple test compilation failures: gptoss_test fails with 'file exists in modules mlx.zig/src/mlx.zig and ../backends/mlx_gptoss_backend.zig' (module collision in build.zig), integration_test fails similarly, deepseek_test fails, moe_test fails, gptoss_manager_test fails. Only registry_test (7/7) and models_test (10/10) pass, both with memory leaks."
    artifacts:
      - path: "src/test_models_gptoss.zig"
        issue: "Line 255: references BackendType.mlx which was removed in Plan 02. This will cause a semantic error once the module collision is fixed."
      - path: "build.zig"
        issue: "Multiple test targets have module aliasing collisions causing 'file exists in modules' compile errors. Tests that could verify MODEL-01 (Qwen E2E) cannot run."
    missing:
      - "Fix module aliasing collisions in build.zig for test targets (gptoss_test, deepseek_test, integration_test, moe_test, gptoss_manager_test)"
      - "Update test_models_gptoss.zig:255 to remove BackendType.mlx (replaced with llama_cpp per selectBackend fallback for qwen)"
      - "Run a Qwen2.5-Coder integration test that actually exercises the MLX.zig inference path end-to-end"

  - truth: "MODEL-03: GPT-OSS-20B generates real output tokens (not hardcoded token 1 or token 0)"
    status: partial
    reason: "Plan 04 fixed the EOS termination bug (was token 0 = immediate exit). Loop now runs for max_tokens iterations producing token 0 each time. Plan 04 summary explicitly documents: 'real weight fields wired in future phase — output is max_tokens copies of token 0'. REQUIREMENTS.md MODEL-03 requires real output tokens from native MLX, which requires forward() to use model weights. This is not satisfied."
    artifacts:
      - path: "src/gptoss_mlx.zig"
        issue: "forward() returns mlx.zeros() — no weight fields wired. Argmax of zeros is deterministically 0. Generate loop runs but emits token 0 repeated, not model output."
    missing:
      - "GPTOSSTransformer.forward() must load and use model weight tensors to produce non-trivial logits"
      - "This requires weight loading integration (safetensors loader already exists at src/weight/gptoss_loader.zig)"
human_verification:
  - test: "DeepSeek curl smoke test"
    expected: "curl -X POST http://localhost:8080/v1/chat/completions -d '{\"model\":\"deepseek-coder-v2-lite\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}' returns a JSON response (not 503 or error)"
    why_human: "Requires a real GGUF model loaded via /v1/models/switch — cannot verify without running server and having a model file available"
---

# Phase 17: Inference Gap Closure Verification Report

**Phase Goal:** Every claimed inference path produces real output — no error.NotImplemented, no hardcoded mocks, no empty token slices
**Verified:** 2026-04-05
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 1  | GAP-07: Speculation fully removed, src/speculation/ deleted | VERIFIED | `ls src/speculation/` = "No such file or directory". grep for live speculation imports returns nothing. Commits 9fd61be, f371c87 on Deepseek branch. |
| 2  | GAP-05: factory.zig and mlx_backend.zig deleted | VERIFIED | `src/backends/` contains: backend.zig, llama_cpp.zig, llama_c.zig, mlx_gptoss_backend.zig, mod.zig only. Commits 4fa3bed, 28f83ba. |
| 2b | GAP-02: Hardcoded vocab_size=32000, EOS=2, BOS=1 removed from live paths | VERIFIED | Values existed only in mlx_backend.zig (now deleted). No live code references 32000/2/1 as EOS/BOS/vocab. |
| 3  | GAP-04: GPT-OSS tokenizer stub replaced with real tokenize() call | VERIFIED | `src/api/chat_gptoss.zig:216` calls `mlx_gptoss_backend.tokenize(...)`. grep for `alloc(u32, 0)` shows only a comment (stub was replaced). |
| 4  | GAP-03 (plan scope): DeepSeek handler created and wired into server.zig dispatch | VERIFIED | `src/api/chat_deepseek.zig` exists and is substantive. `server.zig:183` dispatches "deepseek" prefix to ChatDeepSeekHandler. `error.DeepSeekNotImplemented` absent from mod.zig. |
| 5  | GAP-03 (REQUIREMENTS.md): GPT-OSS forward pass with real tensor computation | FAILED | `src/gptoss_mlx.zig:109` still calls `mlx.zeros()`. No weight fields in GPTOSSTransformer. Returns zero logits unconditionally. |
| 6  | GAP-06: loadIndex() reads index.json from disk, populates entries | FAILED | `src/cache/prompt_cache.zig:433` still reads: `// TODO: Parse index.json and populate entries`. Commit 757d3d0 exists but only on branch worktree-agent-a7f5ba59, not Deepseek. |
| 7  | MLX_gptoss token IDs read from config.json not hardcoded | VERIFIED | `mlx_gptoss_backend.zig:124-126` reads vocab_size/eos_token/bos_token from config.json at load() time. GPTOSSConfig.eos_token_id field present at `gptoss_mlx.zig:36`. |
| 8  | zig build passes with no errors | VERIFIED | `zig build` exits 0. Only CMake deprecation warnings (external to Zig). |
| 9  | MODEL-01: zig build test passes (Qwen integration test) | FAILED | Multiple test targets fail to compile with module collision errors. gptoss_test has latent semantic error (BackendType.mlx removed but still referenced in test_models_gptoss.zig:255). |
| 10 | MODEL-03: GPT-OSS generates real output tokens | FAILED | Generate loop runs for max_tokens (EOS fix verified), but produces token 0 repeated (argmax of zero-logit tensor). Not "real output tokens" as required. |

**Score:** 6/10 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/api/chat_deepseek.zig` | ChatDeepSeekHandler with init, handle, isDeepSeekModel | VERIFIED | Exists. Calls `llama_backend.tokenize()`, reads vocabSize/eosToken/bosToken from LlamaBackend accessors per D-06. |
| `src/api/server.zig` | g_deepseek_backend and g_chat_deepseek_handler globals, deepseek dispatch | VERIFIED | Lines 37, 39, 75, 183 confirmed. |
| `src/api/chat_gptoss.zig` | Real tokenize call replacing alloc(u32, 0) stub | VERIFIED | Line 216 calls `mlx_gptoss_backend.tokenize(...)`. |
| `src/backends/mlx_gptoss_backend.zig` | vocab_size, eos_token, bos_token read from config.json | VERIFIED | Lines 124-126 read from config.json at load(). Fallback defaults (151936/100257/100256) used when path is empty. |
| `src/gptoss_mlx.zig` | GPTOSSTransformer.generate() with correct EOS token not hardcoded 0 | PARTIAL | EOS fix verified (line 147 uses self.config.eos_token_id). But forward() still returns zeros — generate loop produces correct iteration count but all tokens are 0. |
| `src/cache/prompt_cache.zig` | loadIndex() with real std.json disk reads | FAILED | Line 433 still has `// TODO: Parse index.json and populate entries`. Only file access check + log is present. Implementation commit 757d3d0 exists on worktree-agent-a7f5ba59 branch only. |
| `src/backends/backend.zig` | BackendType without .mlx arm, no extern mlx* fns | VERIFIED | BackendType has only llama_cpp and mlx_gptoss. No extern mlx* declarations. |
| `src/backends/mod.zig` | No factory/mlx_backend imports | VERIFIED | Neither factory.zig nor mlx_backend.zig are imported. |
| `src/speculation/` | Deleted | VERIFIED | Directory does not exist. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `src/api/server.zig` | `src/api/chat_deepseek.zig` | startsWith "deepseek" dispatch at line 183 | WIRED | Pattern confirmed in code. |
| `src/api/chat_deepseek.zig` | `src/backends/llama_cpp.zig` | LlamaBackend.tokenize() call | WIRED | Line 169 calls `llama_backend.tokenize(prompt)`. |
| `src/api/chat_gptoss.zig` | `src/backends/mlx_gptoss_backend.zig` | tokenize() call | WIRED | Line 216 calls `mlx_gptoss_backend.tokenize(...)`. |
| `src/cache/prompt_cache.zig` | `src/cache/prompt_cache.zig` | loadIndex() called from init(), writes to self.entries | BROKEN | loadIndex() is called from init() but its body only checks file access and logs — it does not populate self.entries. The connection exists but the implementation is a stub. |
| `src/gptoss_mlx.zig` | weight tensors | GPTOSSTransformer.forward() → model weights | NOT_WIRED | No weight fields in GPTOSSTransformer struct. forward() does not load or reference any tensors except the zero array it creates. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `src/api/chat_gptoss.zig` | `prompt_tokens` | `mlx_gptoss_backend.tokenize()` → Tokenizer.encode() | YES — reads tokenizer.json and encodes text | FLOWING |
| `src/api/chat_deepseek.zig` | `llama_tokens` | `llama_backend.tokenize()` → llama_c tokenizer | YES — real llama.cpp tokenizer | FLOWING |
| `src/gptoss_mlx.zig` | `next_token` (generate output) | `self.forward()` → mlx.zeros → argmax | NO — always returns token 0 | DISCONNECTED |
| `src/cache/prompt_cache.zig` | `self.entries` at server start | `loadIndex()` | NO — TODO stub, entries never populated from disk | DISCONNECTED |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| zig build succeeds | `zig build` | Exits 0, only CMake deprecation warnings | PASS |
| speculation/ deleted | `ls src/speculation/` | No such file or directory | PASS |
| factory.zig deleted | `ls src/backends/` | Not present | PASS |
| loadIndex has real implementation | `grep "TODO" src/cache/prompt_cache.zig` | Line 433: `// TODO: Parse index.json and populate entries` | FAIL |
| gptoss forward() not zeros | `grep "mlx.zeros\|return.*zeros" src/gptoss_mlx.zig` | Line 109: `try mlx.zeros(...)` | FAIL |
| test suite passes | `zig build test` | gptoss_test, deepseek_test, integration_test, moe_test fail (module collision + BackendType.mlx); registry_test 7/7, models_test 10/10 (with leaks) | FAIL |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| GAP-07 | 17-01 | Speculative decoding explicitly removed with documentation | SATISFIED | src/speculation/ deleted. Removal comments at all sites. zig build clean. |
| GAP-05 | 17-02 | MLX backend factory deleted (factory.zig, mlx_backend.zig) | SATISFIED | Both files deleted. No live imports remain. |
| GAP-02 | 17-02 | Hardcoded mock values (32000/2/1) removed from live paths | SATISFIED | Values lived only in mlx_backend.zig, now deleted. mlx_gptoss_backend reads from config.json. |
| GAP-03 | 17-03 | GPT-OSS native MLX forward pass with real tensor computation | NOT SATISFIED | REQUIREMENTS.md definition requires "real tensor computation — not returning mlx.zeros()". gptoss_mlx.zig:109 still calls mlx.zeros(). Plan 03 was scoped to DeepSeek dispatch, not GPT-OSS forward pass. |
| GAP-04 | 17-04 | GPT-OSS tokenizer wired to real tokenizer | SATISFIED | chat_gptoss.zig:216 calls real tokenize(). No alloc(u32, 0) stub in live code. |
| GAP-06 | 17-05 | Prompt cache loadIndex() reads actual entries from disk | NOT SATISFIED | prompt_cache.zig:433 still has TODO stub. Commit 757d3d0 (implementation) never merged to Deepseek branch. |
| MODEL-01 | 17-03/04 | Qwen2.5-Coder integration test passing | NOT SATISFIED | zig build test has 6 test targets failing to compile (module collisions in build.zig). No runnable Qwen E2E test. |
| MODEL-02 | 17-03 | DeepSeek-Coder-V2-Lite wired via llama.cpp backend | PARTIAL | Dispatch wired, handler compiles, zig build passes. Runtime requires real GGUF model. Cannot verify without model file. |
| MODEL-03 | 17-04 | GPT-OSS-20B generates real output tokens (not token 1) | NOT SATISFIED | EOS termination fixed (not token 0 exit), but forward() returns zeros → generates token 0 repeated. Plan 04 summary acknowledges this is intentional stub behavior. |
| GAP-01 | 17-01/02 | No silent stubs in claimed production paths | PARTIAL | error.DeepSeekNotImplemented removed. backend_generator.zig has two NotImplemented returns but is only in test path (not in live HTTP handler chain). Compression stubs (turboquant_stub.zig, metal_kernels.zig) have NotImplemented but TurboQuant is explicitly deferred. Production HTTP paths are clean. |

**Orphaned Requirements (Phase 17 in REQUIREMENTS.md but not claimed by any plan):**

None — all 10 requirement IDs (GAP-01 through GAP-07, MODEL-01 through MODEL-03) appear in at least one plan's `requirements` field.

**Requirements with description mismatch:**

- **GAP-03**: REQUIREMENTS.md defines it as "GPT-OSS native MLX forward pass not returning mlx.zeros()". Plans 03 and 04 tagged it but focused on DeepSeek dispatch and EOS fix respectively. The core definition of GAP-03 (real forward pass computation) was not addressed.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/cache/prompt_cache.zig` | 433 | `// TODO: Parse index.json and populate entries` | Blocker | Cache never restores entries from disk on server restart. Self.entries remains empty after loadIndex() returns. |
| `src/gptoss_mlx.zig` | 109 | `mlx.zeros(...)` in forward() | Blocker | GPT-OSS generates token 0 repeated instead of model output. Acknowledged stub documented for future phase. |
| `src/test_models_gptoss.zig` | 255 | `BackendType.mlx` — enum variant removed in Plan 02 | Warning | Test will fail with enum-does-not-exist error once module collision is fixed. Currently masked by build.zig module aliasing issue. |
| `src/inference/backend_generator.zig` | 37, 52 | `return error.NotImplemented` | Info | BackendGenerator stubs preserved for Phase 18 type surface. Not in live HTTP path. Explicitly documented as backlog. |
| `src/models/draft_model.zig` | — | Imports deleted `../speculation/draft_selector.zig` | Info | Orphaned file. Not imported by any live file. Does not affect build since not in compilation unit. Logged in 17-01-SUMMARY deferred items. |

### Human Verification Required

#### 1. DeepSeek Runtime Smoke Test

**Test:** Start zlx server, issue `curl -X POST http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"deepseek-coder-v2-lite","messages":[{"role":"user","content":"hi"}]}'` with a real GGUF model loaded.
**Expected:** HTTP 200 JSON response with non-empty `choices[0].message.content`. Response uses tokens from LlamaBackend, not a hardcoded string.
**Why human:** Requires a real DeepSeek GGUF model file on disk and a running server. Cannot verify model path loading or actual llama.cpp inference without a model.

#### 2. GPT-OSS Tokenizer End-to-End

**Test:** Start server with a GPT-OSS model loaded, send `{"model":"gptoss","messages":[{"role":"user","content":"hello"}]}`.
**Expected:** Non-empty `choices[0].message.content` (even if all tokens are 0 due to forward() stub, a non-empty response body should be returned by the handler).
**Why human:** Requires model path and server running. The tokenizer call is now real but requires the model directory to actually contain a tokenizer.json.

### Gaps Summary

Four gaps block goal achievement:

1. **GAP-06 not merged**: The `loadIndex()` implementation (commit `757d3d0`) was committed to branch `worktree-agent-a7f5ba59` but never merged into `Deepseek`. The docs commit (`ccdb38d`) was cherry-picked but the code was not. The current `prompt_cache.zig` on Deepseek has a live TODO stub at line 433.

2. **GAP-03 scope mismatch**: REQUIREMENTS.md defines GAP-03 as "GPT-OSS forward pass with real tensor computation — not returning mlx.zeros()". The plans tagged GAP-03 addressed DeepSeek dispatch (Plan 03) and EOS termination (Plan 04). The forward() function in `gptoss_mlx.zig` was never addressed and still returns zeros. This makes MODEL-03 unachievable.

3. **MODEL-01 test failures**: `zig build test` has 6 failing test targets due to module aliasing collisions in `build.zig` (likely introduced when ggml sub-libraries were added). Additionally, `test_models_gptoss.zig:255` references the removed `BackendType.mlx` enum variant. No Qwen or DeepSeek integration test can run in the current state.

4. **MODEL-03 not achieved**: GPT-OSS generates token 0 repeated (argmax of zeros). The phase goal requires "real output tokens" which requires actual weight computation. Plan 04 explicitly acknowledges the remaining stub ("real weight loading tracked for a future phase") but REQUIREMENTS.md marks MODEL-03 as requiring actual real output.

The phase did achieve significant progress: speculation removed, dead backend stubs deleted, DeepSeek dispatch wired, tokenizer stub replaced, EOS termination fixed, config.json reads implemented for mlx_gptoss_backend. The zig build passes. But 4 of 10 requirements are not satisfied.

---

_Verified: 2026-04-05_
_Verifier: Claude (gsd-verifier)_
