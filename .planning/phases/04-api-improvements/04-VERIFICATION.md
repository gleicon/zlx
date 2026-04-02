---
phase: 04-api-improvements
verified: 2026-04-01T00:00:00Z
status: gaps_found
score: 4/6 must-haves verified
re_verification: false
gaps:
  - truth: "Stop sequences work — generation halts immediately when matched, with finish_reason: stop"
    status: failed
    reason: "Stop sequence detection infrastructure exists but is not wired to actual generation. Tokenizer integration required to decode tokens for text matching is stubbed out (lines 317-338 in generator.zig). The code has a TODO comment and placeholder that suppresses the unused field warning but does NOT actually check stop sequences."
    artifacts:
      - path: src/inference/generator.zig
        issue: "Lines 317-338 contain TODO and placeholder code. The stop sequence check logic is commented out and replaced with '_ = self.tokens_since_decode.items.len;' which does nothing. Actual implementation requires tokenizer.decode() integration."
    missing:
      - "Integrate tokenizer into GenerationState to decode tokens to text during generation"
      - "Implement actual stop sequence matching in the next() method's token generation loop"
      - "Return finish_reason: 'stop' when stop sequence is matched"
  - truth: "Logprobs returns top-5 log probabilities for each generated token"
    status: failed
    reason: "Logprobs infrastructure is complete (captureLogprobs(), LogprobEntry struct, top_logprobs buffer) BUT the captured data is not serialized into the API response. In streaming.zig line 387, logprobs are retrieved but immediately discarded with '_ = '. Response building code doesn't include logprobs JSON."
    artifacts:
      - path: src/api/streaming.zig
        issue: "Line 387: '_ = if (gen_options.logprobs_enabled) state.getLogprobs() else null;' — logprobs captured but discarded, not included in response"
      - path: src/api/handlers.zig
        issue: "buildChatCompletionResponse() doesn't include logprobs field in JSON output — only outputs id, object, created, model, choices, usage"
    missing:
      - "Add logprobs JSON serialization in streaming response building"
      - "Add logprobs JSON serialization in non-streaming response building (buildChatCompletionResponse)"
      - "Map token IDs to strings for the response (requires tokenizer.decode calls)"
  - truth: "All sampling parameters work: top_k, min_p, presence_penalty, frequency_penalty, repetition_penalty, logit_bias, seed"
    status: partial
    reason: "All parameter types, request parsing, and helper methods exist. The applyTopK(), applyMinP(), applyPenalties() functions are fully implemented. HOWEVER, these functions are NEVER called in the next() method's sampling pipeline. The current sampling code only applies temperature scaling and softmax — no top_k, min_p, penalties, or logit_bias are actually applied during token selection."
    artifacts:
      - path: src/inference/generator.zig
        issue: "Lines 239-288 (next() method sampling logic) do NOT call applyTopK(), applyMinP(), or applyPenalties(). The functions exist but are orphaned. Current sampling: temperature scaling → softmax → sample. Missing: top_k → min_p → penalties → logit_bias."
    missing:
      - "Call applyTopK() before softmax in next() method"
      - "Call applyMinP() before softmax in next() method"
      - "Call applyPenalties() before softmax in next() method"
      - "Apply logit_bias modifications before softmax"
      - "Wire up the complete sampling pipeline: logits → logit_bias → penalties → top_k → min_p → temperature → softmax → sample"
human_verification:
  - test: "Verify temperature=0 produces identical output for identical prompts"
    expected: "Running the same request twice with temperature=0 and same seed should produce identical token sequences"
    why_human: "Requires running actual model inference to verify determinism"
  - test: "Verify error responses include request_id field"
    expected: "All error responses (400, 408, 500, 404) should include request_id field in JSON"
    why_human: "Requires making actual HTTP requests to verify response format"
  - test: "Verify timeout returns partial completion"
    expected: "When a long generation exceeds timeout, response should contain text generated so far, not empty"
    why_human: "Requires running generation with very short timeout and verifying non-empty response"
---

# Phase 04: API Improvements Verification Report

**Phase Goal:** Full OpenAI API compatibility with complete sampling parameters, stop sequences, and logprobs
**Verified:** 2026-04-01
**Status:** gaps_found
**Score:** 4/6 must-haves verified (66%)

---

## Executive Summary

Phase 04 partially achieves its goal. The infrastructure for all features is in place, but **critical integration gaps** prevent full functionality:

| Must-Have | Status | Issue |
|-----------|--------|-------|
| Stop sequences (API-01) | ❌ FAILED | Infrastructure complete but NOT wired to generation loop |
| Logprobs (API-02) | ❌ FAILED | Captured but NOT serialized to response |
| Sampling parameters (API-03) | ⚠️ PARTIAL | Functions exist but NOT called in sampling pipeline |
| Temperature=0 (API-04) | ✅ VERIFIED | Fully implemented with argmax on raw logits |
| Seed parameter (API-05) | ✅ VERIFIED | PCG32 RNG with deterministic output |
| Error handling (INFRA-03) | ✅ VERIFIED | HTTP 400, 408, 500 with request IDs |
| Request timeout (INFRA-04) | ✅ VERIFIED | 60s default, partial completion on timeout |

---

## Observable Truths

### 1. Stop sequences work — generation halts when matched
**Status:** ❌ FAILED

**Expected:** Request with `stop: ["END", "\n\n"]` halts generation immediately when matched, returns `finish_reason: "stop"`

**Actual:** 
- `StopReason` enum exists (`.eos`, `.length`, `.stop`, `.timeout`)
- `checkStopSequence()` method correctly checks text suffix against stop sequences
- `truncateStopSequence()` removes matched sequences
- **BUT:** Lines 317-338 in `generator.zig:next()` contain only a TODO comment and placeholder code
- The actual decode-and-check logic is commented out because tokenizer integration is pending
- Generation NEVER actually checks stop sequences during token generation

**Evidence:**
```zig
// Lines 317-338 in generator.zig
if (should_check and self.options.stop_sequences.len > 0) {
    // TODO: Decode tokens_since_decode and append to decoded_text_buffer
    // This requires tokenizer integration to decode tokens to text
    // Once decoded and appended to decoded_text_buffer, call:
    // if (self.checkStopSequence()) { ... }
    
    // Placeholder: suppress unused field warning
    _ = self.tokens_since_decode.items.len;
}
```

---

### 2. Logprobs returns top-5 log probabilities
**Status:** ❌ FAILED

**Expected:** Request with `logprobs: true` returns top-5 alternative tokens with log probabilities for each generated token

**Actual:**
- `LogprobEntry` and `TopLogprob` structs fully implemented
- `captureLogprobs()` method correctly computes log_softmax and extracts top-5 tokens
- `logprobs_buffer` accumulates entries during generation
- **BUT:** In `streaming.zig:387`, logprobs are retrieved but immediately discarded:
```zig
_ = if (gen_options.logprobs_enabled) state.getLogprobs() else null;
```
- `buildChatCompletionResponse()` in handlers.zig does NOT include logprobs in JSON output
- Logprobs are captured but never serialized to the API response

---

### 3. All sampling parameters work
**Status:** ⚠️ PARTIAL

**Expected:** top_k, min_p, presence_penalty, frequency_penalty, repetition_penalty, logit_bias all affect token selection

**Actual:**
- All parameter types exist in `GenerationOptions`
- All request parsing helpers exist in `types.zig`
- `applyTopK()`, `applyMinP()`, `applyPenalties()` are fully implemented
- **BUT:** These functions are NEVER called in the sampling pipeline
- Current `next()` method sampling: temperature scaling → softmax → sample
- Missing the complete pipeline: logits → logit_bias → penalties → top_k → min_p → temperature → softmax

**Evidence:**
```zig
// Lines 239-288 in generator.zig:next()
// Only temperature scaling and softmax are applied
// NO calls to: applyTopK(), applyMinP(), applyPenalties(), or logit_bias application
```

---

### 4. Temperature=0 selects highest probability token
**Status:** ✅ VERIFIED

**Implementation:** Lines 239-244 in `generator.zig` correctly implement greedy selection:
```zig
if (self.options.temperature == 0) {
    // D-18: Greedy selection — argmax on raw logits (before temperature/softmax)
    var next_token_arr = mlx.arrayNew();
    defer mlx.arrayFree(next_token_arr);
    try mlx.argmax(&next_token_arr, last_logits, 1, false, transformer.mlx_config.stream);
    try mlx.item(&next_token, next_token_arr);
}
```

- Uses argmax directly on raw logits (before any scaling)
- No numerical instability from division by zero
- True greedy selection per OpenAI spec

---

### 5. Same seed produces identical output
**Status:** ✅ VERIFIED

**Implementation:** Lines 14-45 in `generator.zig` — PCG32 RNG:
```zig
pub const Pcg32Rng = struct {
    state: u64,
    inc: u64,
    const MULTIPLIER: u64 = 6364136223846793005;
    const DEFAULT_INC: u64 = 1442695040888963407;
    // ... init() and random() methods
};
```

- Same seed produces identical random sequence
- Used in sampling at line 273:
```zig
const random_value = if (self.rng) |*rng| rng.random() else std.crypto.random.float(f32);
```

---

### 6. API returns proper error codes
**Status:** ✅ VERIFIED

**Implementation:** All error handling functions implemented in `handlers.zig`:

| Function | Status Code | Evidence |
|----------|-------------|----------|
| `sendBadRequestError()` | 400 | Lines 305-317 |
| `sendTimeoutError()` | 408 | Lines 320-331 |
| `sendServerError()` | 500 | Lines 334-345 |
| Model not found | 404 | Line 124 |

All include `request_id` in response per D-28.

---

### 7. Request timeout cancels generation
**Status:** ✅ VERIFIED

**Implementation:** Lines 116-202 in `mod.zig`:
```zig
pub fn generateWithTimeout(self: *Self, prompt: []const u8, options: GenerationOptions, timeout_ms: u64) !TimeoutResult {
    const start_time = std.time.milliTimestamp();
    // ... tokenization and setup ...
    while (try state.next()) |token| {
        try output_tokens.append(self.allocator, token);
        // Check timeout every token (D-33: measured from request start)
        const elapsed = @as(u64, @intCast(std.time.milliTimestamp() - start_time));
        if (elapsed >= timeout_ms) {
            std.log.warn("Generation timed out after {d}ms, returning partial result", .{elapsed});
            timed_out = true;
            state.setStopReason(.timeout);
            break;
        }
    }
    // ...
}
```

- Timeout checked every token
- Partial completion returned if timeout occurs
- `--timeout` CLI flag with 60s default (main.zig:19, 37, 74-84)
- HTTP 408 returned on timeout (handlers.zig:198-202)

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/inference/generator.zig` | Stop sequences, RNG, sampling | ⚠️ PARTIAL | Infrastructure complete, integration gaps |
| `src/inference/mod.zig` | Timeout-aware generation | ✅ VERIFIED | generateWithTimeout fully implemented |
| `src/api/types.zig` | Request types with all params | ✅ VERIFIED | All parsing helpers present |
| `src/api/handlers.zig` | Error handling with request IDs | ✅ VERIFIED | All error functions implemented |
| `src/api/streaming.zig` | Logprobs in streaming | ❌ STUB | Logprobs captured but discarded |
| `src/main.zig` | CLI timeout flag | ✅ VERIFIED | --timeout with 60s default |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `generator.zig:next()` | `checkStopSequence()` | function call | ❌ NOT_WIRED | TODO comment, placeholder only |
| `generator.zig:next()` | `Pcg32Rng.random()` | conditional RNG | ✅ WIRED | Line 273: uses PCG32 when seed provided |
| `handlers.zig` | `generateWithTimeout()` | timeout wrapper | ✅ WIRED | Lines 183-187 in handleNonStreamingRequest |
| `handlers.zig` | `generateRequestId()` | error response | ✅ WIRED | Line 33, 390-394 in types.zig |
| `generator.zig:next()` | `applyTopK/MinP/Penalties()` | sampling pipeline | ❌ NOT_WIRED | Functions exist but never called |
| `streaming.zig` | `getLogprobs()` | response building | ❌ NOT_WIRED | Retrieved but discarded at line 387 |

---

## Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|-------------------|--------|
| `generator.zig:next()` | `next_token` | MLX argmax/sample | Yes | ✅ FLOWING |
| `generator.zig:next()` | `logprobs_buffer` | `captureLogprobs()` | Yes | ⚠️ CAPTURED but not output |
| `generator.zig:next()` | `stop_sequences` | Request options | No | ❌ DISCONNECTED — never checked |
| `mod.zig:generateWithTimeout()` | `timed_out` | Time elapsed check | Yes | ✅ FLOWING |
| `handlers.zig` | `request_id` | `generateRequestId()` | Yes | ✅ FLOWING |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| API-01 | 04-01 | Stop sequences | ⚠️ PARTIAL | Infrastructure complete, tokenizer integration pending |
| API-02 | 04-02 | Logprobs | ⚠️ PARTIAL | Capture works, JSON serialization stubbed |
| API-03 | 04-02 | Sampling parameters | ⚠️ PARTIAL | Functions exist, NOT wired to sampling pipeline |
| API-04 | 04-01 | Temperature=0 | ✅ COMPLETE | Argmax on raw logits (lines 239-244) |
| API-05 | 04-01 | Seed parameter | ✅ COMPLETE | PCG32 RNG, deterministic output |
| INFRA-03 | 04-03 | Error handling | ✅ COMPLETE | HTTP 400, 408, 500 with request IDs |
| INFRA-04 | 04-03 | Request timeout | ✅ COMPLETE | 60s default, partial completion, 408 response |

**Requirements Traceability:**
- All 7 requirement IDs from Phase 04 are accounted for in the plans
- Each ID appears in at least one PLAN frontmatter
- No orphaned requirements found

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `generator.zig` | 326 | `// TODO:` comment | ⚠️ Warning | Stop sequence integration pending |
| `generator.zig` | 337 | `_ = self.tokens_since_decode.items.len;` | 🛑 Blocker | Placeholder suppresses logic — stop sequences NEVER checked |
| `streaming.zig` | 387 | `_ = if (gen_options.logprobs_enabled) state.getLogprobs() else null;` | 🛑 Blocker | Logprobs captured but immediately discarded |
| `generator.zig` | 476-564 | `applyTopK/MinP/Penalties()` defined but unused | 🛑 Blocker | Sampling parameters don't affect generation |

---

## Gaps Summary

### Critical Gaps (Blocking Goal Achievement)

1. **Stop Sequence Integration (API-01)**
   - Root cause: Tokenizer not accessible in `GenerationState.next()`
   - Required: Pass tokenizer through init → decode tokens to text during generation
   - Work: ~2-3 hours — add tokenizer field, decode accumulated tokens, call checkStopSequence()

2. **Logprobs JSON Serialization (API-02)**
   - Root cause: Response building code doesn't include logprobs field
   - Required: Add logprobs to streaming.zig and handlers.zig response builders
   - Work: ~1-2 hours — add JSON serialization for LogprobEntry array

3. **Sampling Pipeline Wiring (API-03)**
   - Root cause: applyTopK/MinP/Penalties() functions exist but not called
   - Required: Restructure next() sampling to apply all modifications before softmax
   - Work: ~2-3 hours — integrate function calls into sampling pipeline

### Common Root Cause

All three gaps stem from the same issue: **infrastructure was built but integration was deferred**. The functions exist, types are correct, but the wiring between components is incomplete. This is a pattern of "horizontal" completion (all pieces exist) without "vertical" completion (pieces connected and working).

---

## Recommendations

1. **Create Phase 04b** — A follow-up plan specifically for:
   - Tokenizer integration in GenerationState
   - Logprobs JSON serialization
   - Sampling pipeline wiring

2. **Estimated effort to complete Phase 04 goal:** 6-8 hours

3. **Priority order:**
   1. Sampling pipeline (most impactful — affects all requests)
   2. Logprobs serialization (smaller, self-contained)
   3. Stop sequences (requires more design for tokenizer access)

---

_Verified: 2026-04-01_
_Verifier: GSD Phase Verifier_
