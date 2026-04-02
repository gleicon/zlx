# Plan: Quick Model Support - GPT-OSS & DeepSeek Download

**Status:** Ready for Execution (Option A)  
**Effort:** 2-3 hours  
**DeepSeek MoE:** Deferred to backlog (Phase 11)

## Goal
Enable auto-download for both models. Test GPT-OSS with existing transformers. Document DeepSeek as pending MoE support.

## Tasks (Option A - Quick Wins)

### Task 1: Add Models to Download Registry (30 min)
**File:** `src/download/manager.zig`

Add to `KnownModel` enum and metadata:
- `gpt_oss_20b` - GPT-OSS 20B MXFP4 (11.2 GB)
- `deepseek_coder_v2_lite` - DeepSeek V2 Lite 4bit (8.84 GB)

### Task 2: Update CLI Help & Examples (15 min)
**File:** `src/main.zig`

Add new models to help text examples.

### Task 3: Test GPT-OSS Loading (30 min)
**Test:** Manual load with llama transformer
- Download model: `zlx --download-model gpt-oss-20b`
- Test loading: `zlx --model gpt-oss-20b --port 8081`
- Verify it runs or identify issues

### Task 4: Create DeepSeek Backlog Entry (15 min)
**File:** `.planning/BACKLOG.md`

Document MoE requirements and blockers.

### Task 5: Update Documentation (30 min)
**Files:** `README.md`, `docs/MODELS.md`

Document supported models and download instructions.

## Deferred to Phase 11 (Backlog)

### DeepSeek MoE Support
**Complexity:** High (20+ hours)  
**Blocked by:** MLX.zig MoE architecture support  
**Requirements:**
- Implement MLA (Multi-head Latent Attention)
- Implement MoE routing (shared + routed experts)
- Chat template for DeepSeek format
- Memory estimation for sparse parameters

### GPT-OSS Full Integration (if needed)
**Complexity:** Medium (4-6 hours)  
**Only if:** GPT-OSS doesn't work with llama transformer  
**Requirements:**
- Custom GPT-OSS transformer
- Tool use API support
- Complex chat template

## Success Criteria

- [ ] `zlx --download-model gpt-oss-20b` works
- [ ] `zlx --download-model deepseek-coder-v2-lite` works  
- [ ] GPT-OSS loads (or we document why it doesn't)
- [ ] README updated with new models
- [ ] BACKLOG.md created with DeepSeek MoE task

## Next Phase Trigger
When MLX.zig adds MoE support OR when we upgrade mlx-c to 0.4.x with custom ops API.
