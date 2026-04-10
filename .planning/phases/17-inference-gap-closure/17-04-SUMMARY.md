---
phase: 17-inference-gap-closure
plan: "04"
subsystem: gptoss-inference
tags: [gptoss, tokenizer, config, eos-token, inference]
dependency_graph:
  requires: [17-01, 17-02]
  provides: [real-tokenizer-call, config-json-token-ids, eos-termination-fix]
  affects: [src/api/chat_gptoss.zig, src/backends/mlx_gptoss_backend.zig, src/gptoss_mlx.zig]
tech_stack:
  patterns:
    - config.json read at load() time — not init() — avoids empty model_path at construction
    - vtable-style free function tokenize() called via @ptrCast from handler
    - GPTOSSConfig struct fields carry eos/bos token IDs through generate() loop
key_files:
  modified:
    - src/api/chat_gptoss.zig
    - src/backends/mlx_gptoss_backend.zig
    - src/gptoss_mlx.zig
decisions:
  - "D-01 applied: handler calls mlx_gptoss_backend.tokenize() via @ptrCast — vtable free function pattern"
  - "D-06 applied: vocab_size/eos_token/bos_token read from config.json at load() time; GPT-OSS defaults (151936/100257/100256) used as fallback if file missing"
  - "GPTOSSConfig extended with eos_token_id and bos_token_id fields (defaulting to GPT-OSS values) — EOS check in generate() now uses self.config.eos_token_id instead of hardcoded 0"
metrics:
  duration_seconds: 141
  completed_date: "2026-04-05"
  tasks_completed: 3
  tasks_total: 3
  files_modified: 3
---

# Phase 17 Plan 04: GPT-OSS Tokenizer and EOS Fix Summary

Wire real tokenizer call, read token IDs from config.json, and fix EOS termination so GPT-OSS generate() loop produces non-empty output.

## Tasks Completed

| Task | Description | Commit | Files |
|------|-------------|--------|-------|
| 1 | Replace alloc(u32,0) stub with mlx_gptoss_backend.tokenize() call | cc8cd90 | src/api/chat_gptoss.zig |
| 2 | Add vocab_size/eos_token/bos_token struct fields; read from config.json in load() | d9803a3 | src/backends/mlx_gptoss_backend.zig |
| 3 | Add eos_token_id to GPTOSSConfig; fix generate() EOS check from ==0 to ==config.eos_token_id | 6b2e183 | src/gptoss_mlx.zig |

## Verification

**zig build:** PASSES with no errors

**Stub removed:**
```
grep -n "alloc(u32, 0)" src/api/chat_gptoss.zig → only in comment (line 215), no live stub
```

**Real tokenize call present:**
```
src/api/chat_gptoss.zig:216:        const prompt_tokens = try mlx_gptoss_backend.tokenize(
```

**config.json reads in place:**
```
src/backends/mlx_gptoss_backend.zig:124:  if (obj.get("vocab_size")) |v| self.vocab_size = @intCast(v.integer);
src/backends/mlx_gptoss_backend.zig:125:  if (obj.get("eos_token_id")) |v| self.eos_token = @intCast(v.integer);
src/backends/mlx_gptoss_backend.zig:126:  if (obj.get("bos_token_id")) |v| self.bos_token = @intCast(v.integer);
```

**EOS termination fixed:**
```
src/gptoss_mlx.zig:147:  if (next_token == self.config.eos_token_id) break;
```
(was: `if (next_token == 0) break` — caused immediate exit on zero-logit forward())

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

- `GPTOSSTransformer.forward()` still returns zero logits (no weight fields wired yet). This means generate() will produce `max_tokens` copies of token 0 (argmax of zeros) rather than real output text. This is intentional and documented — real weight loading is tracked for a future phase. The EOS fix in Task 3 ensures the loop runs for max_tokens rather than terminating on the first token.

## Self-Check: PASSED
