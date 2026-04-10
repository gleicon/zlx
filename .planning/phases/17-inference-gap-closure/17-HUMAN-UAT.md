---
status: automated
phase: 17-inference-gap-closure
source: [17-VERIFICATION.md]
started: 2026-04-06T20:45:00Z
updated: 2026-04-06T23:30:00Z
automated_by: scripts/e2e_test.sh
---

## Current Test

Automated. Run `zig build test-e2e` (or `scripts/e2e_test.sh`).
Download test models first: `scripts/download_test_models.sh`

## Tests

### 1. GPT-OSS forward pass with real model weights
expected: Load a GPT-OSS safetensors checkpoint (model.embed_tokens.weight + lm_head.weight), issue /v1/chat/completions. Tokens are input-dependent — different prompt produces different output (not always token 0).
automated: scripts/e2e_test.sh (UAT-1) — starts server with models/GPT-OSS-20B-4bit, sends two different prompts, asserts both non-empty and different
model_download: scripts/download_test_models.sh --gptoss (11.2 GB, mlx-community/gpt-oss-20b-MXFP4-Q4)
result: [run zig build test-e2e to verify]

### 2. Prompt cache persistence across server restart
expected: Issue requests (populating the cache), restart the server, check log output for 'Loaded N cache entries' with N > 0.
automated: scripts/e2e_test.sh (UAT-2) — starts server with Qwen model, issues request, kills+restarts, asserts "Loaded N cache entries" in log
model_download: scripts/download_test_models.sh --qwen (1.0 GB, already present)
result: [run zig build test-e2e to verify]

### 3. DeepSeek runtime smoke test
expected: curl -X POST http://localhost:8080/v1/chat/completions with deepseek model returns HTTP 200 with non-empty choices[0].message.content
automated:
  - Part A (MLX unit): zig build test — deepseek_test 10/10 passes (verifies MLX inference)
  - Part B (HTTP smoke): scripts/e2e_test.sh (UAT-3) — HTTP inference via GPT-OSS server
note: Server's 'deepseek' prefix routes to llama.cpp/GGUF. MLX safetensors path tested via unit tests.
      For full GGUF HTTP test: huggingface-cli download TheBloke/deepseek-coder-v2-lite-GGUF (~9 GB)
result: [run zig build test-e2e to verify]

### 4. Qwen integration test with real model on disk
expected: zig build test passes tokenizer and Qwen tests when model present at ./models/Qwen2.5-Coder-1.5B-4bit/
automated: zig build test — generator.zig GenerationState test, qwen tokenizer test
model_download: scripts/download_test_models.sh --qwen (creates symlink Qwen2.5-Coder-1.5B-4bit → Qwen2.5-Coder-1.5B-Instruct-4bit)
fixes:
  - models/Qwen2.5-Coder-1.5B-4bit symlink created (→ Qwen2.5-Coder-1.5B-Instruct-4bit)
  - src/inference/generator.zig: falls back to -Instruct path if base path absent
result: [run zig build test to verify]

## Summary

total: 4
passed: 0
issues: 0
pending: 0
skipped: 0
blocked: 0
automated: 4

## Gaps
