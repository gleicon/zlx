# Plan: Add DeepSeek-Coder-V2-Lite and GPT-OSS Model Support

## Overview
Add support for two new state-of-the-art coding models:
1. **DeepSeek-Coder-V2-Lite-Instruct** (mlx-community) - MoE architecture, 2B active params
2. **GPT-OSS-20B** (mlx-community) - OpenAI's open source model, 20B params

## Research Summary

### Model 1: DeepSeek-Coder-V2-Lite-Instruct
- **Source:** `mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx`
- **Original:** `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`
- **Architecture:** DeepSeek-V2 (MoE - Mixture of Experts)
- **Parameters:** 15.7B total, 2B active per token
- **Context:** 128k tokens
- **Size:** ~8.84 GB (4-bit quantized)
- **Tags:** deepseek_v2, custom_code, moe
- **License:** deepseek-license
- **Features:** Code generation, instruction following

### Model 2: GPT-OSS-20B
- **Source:** `mlx-community/gpt-oss-20b-MXFP4-Q4`
- **Original:** `openai/gpt-oss-20b`
- **Architecture:** GPT-OSS (proprietary)
- **Parameters:** 20B
- **Context:** Standard (TBD)
- **Size:** ~11.2 GB (MXFP4 quantized)
- **Tags:** gpt_oss, conversational
- **License:** Apache-2.0
- **Features:** General chat, tool use support

## Critical Analysis: Implementation Requirements

### 1. Architecture Support Check
**DeepSeek-V2 MoE Architecture:**
- Uses MLA (Multi-head Latent Attention) - different from standard MHA
- MoE routing mechanism with shared experts + routed experts
- Not currently supported in MLX.zig
- **Status:** ⚠️ REQUIRES mlx-c upgrade or custom implementation

**GPT-OSS Architecture:**
- Likely based on GPT-4 architecture
- May work with existing transformer infrastructure
- **Status:** ⚠️ NEEDS VERIFICATION

### 2. Chat Template Support
Both models have custom chat templates that differ from Qwen:
- DeepSeek: Uses custom template with system prompts
- GPT-OSS: Complex template with tool support
- Need to extend tokenizer/chat template system

### 3. Quantization Format
- DeepSeek: Standard 4-bit MLX format
- GPT-OSS: MXFP4 (new MLX format)
- Need to verify MLX.zig supports MXFP4

## Implementation Plan

### Phase 1: Investigation & Feasibility (1-2 hours)
**Goal:** Determine if models can run with current MLX.zig

#### Task 1.1: Test Model Loading (High Priority)
- Attempt to load DeepSeek model with current qwen transformer
- Check if config.json parsing works
- Identify missing architecture fields

#### Task 1.2: Check mlx-c Version Compatibility
- Verify current mlx-c version (0.1.2)
- Research if MoE support exists in newer mlx-c versions
- Document upgrade path if needed

#### Task 1.3: Analyze Model Configs
- Download and inspect config.json for both models
- Identify architecture-specific fields
- Check tokenizer format compatibility

**Deliverable:** Feasibility report with specific gaps

### Phase 2: Architecture Implementation (8-16 hours)
**Goal:** Add MoE/DeepSeek architecture support

#### Task 2.1: DeepSeek Transformer Implementation
- Create `src/mlx.zig/src/deepseek.zig`
- Implement MLA (Multi-head Latent Attention)
- Implement MoE routing logic
- Integrate with existing transformer union

#### Task 2.2: Model Detection Enhancement
- Update `src/inference/loader.zig` to detect:
  - "deepseek" in model_type
  - "deepseek_v2" architecture
  - MoE-specific config fields

#### Task 2.3: GPT-OSS Transformer Support
- Test if GPT-OSS works with llama or qwen transformer
- If not, implement basic GPT-OSS transformer
- Add model type detection

**Deliverable:** Both models load and run inference

### Phase 3: Auto-Download Integration (4-6 hours)
**Goal:** Enable automatic model downloads

#### Task 3.1: Update Known Models Registry
Add to `src/download/manager.zig`:
```zig
.deepseek_coder_v2_lite = .{
    .id = "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx",
    .name = "DeepSeek-Coder-V2-Lite-Instruct",
    .size_mb = 9050, // ~8.84 GB
    .architecture = .deepseek_v2,
    .recommended = true,
},
.gpt_oss_20b = .{
    .id = "mlx-community/gpt-oss-20b-MXFP4-Q4",
    .name = "GPT-OSS-20B",
    .size_mb = 11470, // ~11.2 GB
    .architecture = .gpt_oss,
    .recommended = false, // Large model
},
```

#### Task 3.2: Add Model Download Command
- New CLI: `zlx --download-model deepseek-coder-v2-lite`
- Integration with existing download manager
- Progress tracking and resume support

#### Task 3.3: Memory Estimation Updates
- Add MoE-specific memory calculation (sparse params)
- Update memory requirements for GPT-OSS

**Deliverable:** Users can auto-download both models

### Phase 4: Chat Template & Tokenizer (4-6 hours)
**Goal:** Support different chat formats

#### Task 4.1: Chat Template System Enhancement
- Extract chat_template from tokenizer.json
- Create template parser for:
  - DeepSeek format (User: / Assistant:)
  - GPT-OSS format (complex with tools)

#### Task 4.2: Message Format Converter
- Convert OpenAI messages to model-specific format
- Handle system prompts correctly
- Support tool use for GPT-OSS

#### Task 4.3: Testing
- Test chat completion with each model
- Verify system prompts work
- Test streaming responses

**Deliverable:** Both models produce correct chat responses

### Phase 5: Integration & Testing (4-6 hours)
**Goal:** Full integration and validation

#### Task 5.1: Model Registry Integration
- Auto-detect downloaded models
- Show in `/v1/models` endpoint
- Update metadata correctly

#### Task 5.2: Performance Optimization
- Verify TurboQuant works with new models
- Test speculative decoding compatibility
- Benchmark performance vs Qwen

#### Task 5.3: End-to-End Testing
- Test with OpenCode integration
- Verify streaming works
- Test model switching

**Deliverable:** Production-ready support

## Implementation Order (Recommended)

### Wave 1: Quick Wins (2-3 hours)
1. Add models to download registry (Task 3.1)
2. Test basic loading with existing transformers (Task 1.1)
3. Add CLI download command (Task 3.2)

**Result:** Users can download, manual loading may work

### Wave 2: Architecture Support (8-16 hours)
1. Implement DeepSeek transformer (Task 2.1)
2. Add model detection (Task 2.2)
3. Test GPT-OSS compatibility (Task 2.3)

**Result:** Both models run inference

### Wave 3: Polish (4-8 hours)
1. Chat template support (Task 4.x)
2. Integration testing (Task 5.x)
3. Documentation updates

**Result:** Production-ready

## Risk Assessment

### HIGH RISK: MoE Architecture
- **Issue:** DeepSeek-V2 uses MoE, not supported in current mlx-c 0.1.2
- **Mitigation:** 
  - Option A: Upgrade mlx-c (may break MLX.zig compatibility)
  - Option B: Implement MoE routing in Zig (complex, 20+ hours)
  - Option C: Wait for MLX.zig MoE support

### MEDIUM RISK: Chat Templates
- **Issue:** Complex templates require parser updates
- **Mitigation:** Simplified template mapping as interim solution

### LOW RISK: GPT-OSS Support
- **Issue:** Unknown architecture compatibility
- **Mitigation:** Test early, fall back to llama transformer if compatible

## Success Criteria

- [ ] Both models load without errors
- [ ] `/v1/models` shows models with correct metadata
- [ ] Chat completions work with correct formatting
- [ ] Streaming responses work
- [ ] Auto-download works via CLI
- [ ] TurboQuant compression works
- [ ] Performance within 20% of Qwen baseline

## Resources Needed

### Files to Create/Modify
1. `src/mlx.zig/src/deepseek.zig` (new transformer)
2. `src/inference/loader.zig` (model detection)
3. `src/download/manager.zig` (known models)
4. `src/api/chat_template.zig` (new: template handling)
5. `src/models/architectures.zig` (architecture registry)

### External Dependencies
- May need mlx-c upgrade for MoE support
- Test with actual model files (~20 GB download)

## Conclusion

**Estimated Effort:** 18-35 hours depending on MoE complexity
**Priority:** High (state-of-the-art models)
**Complexity:** Medium-High (MoE architecture is the main challenge)

**Recommendation:** 
1. Start with Wave 1 to enable downloads
2. Test if models work with existing transformers (may be lucky!)
3. If MoE is required, evaluate mlx-c upgrade vs. custom implementation
4. Implement chat templates as Phase 4

**Alternative:** If MoE is too complex, implement GPT-OSS first (likely compatible with llama transformer) and defer DeepSeek until MLX.zig adds native MoE support.
