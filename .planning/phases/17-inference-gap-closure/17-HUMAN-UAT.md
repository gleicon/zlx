---
status: partial
phase: 17-inference-gap-closure
source: [17-VERIFICATION.md]
started: 2026-04-06T20:45:00Z
updated: 2026-04-06T20:45:00Z
---

## Current Test

[awaiting human testing]

## Tests

### 1. GPT-OSS forward pass with real model weights
expected: Load a GPT-OSS safetensors checkpoint (model.embed_tokens.weight + lm_head.weight), issue /v1/chat/completions. Tokens are input-dependent — different prompt produces different output (not always token 0).
result: [pending]

### 2. Prompt cache persistence across server restart
expected: Issue requests (populating the cache), restart the server, check log output for 'Loaded N cache entries' with N > 0.
result: [pending]

### 3. DeepSeek runtime smoke test
expected: curl -X POST http://localhost:8080/v1/chat/completions with deepseek model returns HTTP 200 with non-empty choices[0].message.content
result: [pending]

### 4. Qwen integration test with real model on disk
expected: Start server with a Qwen2.5-Coder GGUF or safetensors model, issue a /v1/chat/completions request, verify non-trivial response token output (satisfies ROADMAP.md Success Criteria #5).
result: [pending]

## Summary

total: 4
passed: 0
issues: 0
pending: 4
skipped: 0
blocked: 0

## Gaps
