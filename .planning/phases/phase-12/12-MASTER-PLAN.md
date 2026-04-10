# Phase 12: MoE Models Production-Ready with TurboQuant

## Goal
Transform MoE model support from prototype to production-ready state, enabling both DeepSeek-Coder-V2-Lite and GPT-OSS-20B on small machines (8-16GB RAM) via TurboQuant KV compression and intelligent model sizing.

## Why This Phase

**Current State:**
- ✅ Qwen2.5-Coder-1.5B: Fully working (dense transformer)
- ⏳ DeepSeek-Coder-V2-Lite: Architecture implemented, weight loading stubbed
- ⏳ GPT-OSS-20B: Listed but not implemented (different MoE architecture)
- ✅ Generator segfault: Fixed (undefined array bug)
- ❌ TurboQuant: Listed as complete but not verified working
- ❌ Small machine support: No constraints or optimizations

**Problems to Solve:**
1. **Weight Loading**: DeepSeek stub returns empty arrays - needs real safetensors loading
2. **GPT-OSS Architecture**: Different MoE pattern (32 experts, sliding window attention)
3. **Memory Constraints**: 20B models won't fit on 8GB MacBooks without TurboQuant
4. **Model Selection**: Users need guidance on what fits their machine
5. **Testing**: No automated testing for MoE models

## Scope

**In Scope:**
- DeepSeek-Coder-V2-Lite: Full weight loading and inference
- GPT-OSS-20B: Architecture support + weight loading  
- TurboQuant KV cache compression (verify and fix if needed)
- Model size constraints for small machines (<16GB RAM)
- Automated testing shell script for all models

**Out of Scope:**
- Other MoE architectures (Mistral, Mixtral, etc.)
- Model quantization (assumes pre-quantized models from MLX community)
- Multi-GPU support
- Distributed inference

## Requirements Mapping

| Requirement | Priority | Models Affected |
|-------------|----------|-----------------|
| MOE-01: DeepSeek weight loading | P0 | DeepSeek-Coder-V2-Lite |
| MOE-02: GPT-OSS architecture | P0 | GPT-OSS-20B |
| MOE-03: TurboQuant integration | P0 | All MoE models |
| MOE-04: Memory-aware model selection | P1 | All models |
| MOE-05: Automated model testing | P1 | All models |

## Success Criteria

1. DeepSeek-Coder-V2-Lite loads weights and generates tokens without crash
2. GPT-OSS-20B loads weights and generates tokens without crash  
3. Both models work on 16GB MacBook with TurboQuant enabled
4. Server rejects model load if insufficient memory (with helpful message)
5. `./test_models.sh` passes for all three models (Qwen, DeepSeek, GPT-OSS)
6. 128K context works on 16GB machine with TurboQuant (4.6x compression)

## Model Specifications

| Model | Total Params | Active Params | MoE Config | Memory (Dense) | Memory (TurboQuant) |
|-------|-------------|---------------|------------|----------------|---------------------|
| Qwen2.5-Coder-1.5B | 1.5B | 1.5B | N/A (dense) | ~2GB | ~1.5GB |
| DeepSeek-Coder-V2-Lite | 15.7B | 2.0B | 64 experts, top-6 | ~8GB* | ~4GB* |
| GPT-OSS-20B | 20B | ~5B** | 32 experts, top-4 | ~11GB* | ~6GB* |

*Estimated with 4-bit quantization  
**Estimated based on expert ratio

## Technical Approach

### 1. DeepSeek Weight Loading (11-04 continuation)

**Root Cause:** `loadDeepSeekWeights()` in `src/inference/loader.zig` returns empty arrays.

**Solution:**
- Parse safetensors index file (`model.safetensors.index.json`)
- Load weight files using MLX's `loadSafetensors()`
- Map weight names to DeepSeek architecture:
  - `model.embed_tokens.weight` → token_embedding
  - `model.layers.{i}.input_layernorm.weight` → layer norms
  - `model.layers.{i}.self_attn.*` → MLA weights (kv_b_proj, q_proj, etc.)
  - `model.layers.{i}.mlp.*` → MoE weights (gate_proj, up_proj, down_proj)
  - `model.norm.weight` → final norm
  - `lm_head.weight` → output projection

**Key Challenge:** DeepSeek uses MLA (Multi-head Latent Attention) with compressed KV, different from standard attention.

### 2. GPT-OSS Architecture

**Architecture Differences from DeepSeek:**
- **Attention**: Sliding window (128 tokens) alternating with full attention every layer
- **MoE**: 32 local experts, 4 per token (vs DeepSeek 64 experts, 6 per token)
- **Context**: 128K with Yarn rope scaling
- **Special**: SwiGLU activation with limit, router aux loss

**Implementation Strategy:**
- Reuse MoE routing layer from DeepSeek
- Add sliding window attention mask
- Implement Yarn rope scaling
- Different weight naming convention

### 3. TurboQuant Integration

**Current Status:** Listed as complete in Phase 7, but not verified.

**Verification Steps:**
1. Check if TurboQuant kernels are compiled and linked
2. Verify KV cache compression is applied
3. Measure memory reduction (should be 4.6x)
4. Fix if not working

**Implementation if Missing:**
- Port TurboQuant Metal kernels from Python to Zig/C
- Add KV cache quantization hooks in MLA/MoE layers
- Expose compression level in config

### 4. Small Machine Support

**Memory Budget Calculator:**
```
available_ram_gb = system_ram - 4GB (OS overhead)
max_model_gb = available_ram_gb * 0.7 (safety margin)
required_kv_gb = (context_length * hidden_size * num_layers * 2) / (1024^3 * compression_ratio)
```

**Model Recommendations by RAM:**
- 8GB: Qwen2.5-Coder-1.5B only
- 16GB: All models with TurboQuant, 32K context max
- 32GB+: All models, full 128K context

**Implementation:**
- Add memory check before model load
- Provide `--max-context` flag to limit KV cache
- Auto-enable TurboQuant on <16GB systems

### 5. Testing Infrastructure

**Shell Script:** `test_models.sh`

Features:
- Discovers all models in `./models/`
- Starts server with each model
- Tests /v1/models endpoint
- Tests chat completions (streaming and non-streaming)
- Measures memory usage
- Reports pass/fail for each model
- Runs in CI/CD

## Plans

| Plan | Name | Focus | Est. Hours |
|------|------|-------|------------|
| 12-01 | DeepSeek Weight Loading | Safetensors parsing, weight mapping, MLA integration | 8 |
| 12-02 | GPT-OSS Architecture | Sliding window, Yarn rope, 32-expert MoE | 10 |
| 12-03 | TurboQuant Verification | Verify/fix KV compression, measure memory savings | 4 |
| 12-04 | Small Machine Constraints | Memory budgets, auto-TurboQuant, context limits | 3 |
| 12-05 | Testing Infrastructure | Automated model testing shell script | 3 |

Total: ~28 hours

## Dependencies

- MLX.zig with mlx-c v0.4.x (from Phase 11)
- MoE routing layer (from Phase 11-03)
- MLA attention (from Phase 11-02)
- Generator segfault fix (already applied)

## Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| TurboQuant Python→Zig port too complex | Medium | High | Use MLX built-in quantization instead |
| Weight naming conventions differ from expected | Medium | Medium | Log all missing keys, create mapping layer |
| GPT-OSS sliding window incompatible with current cache | Low | High | Use separate cache path for sliding window layers |
| Memory estimates wrong for MoE | Medium | Medium | Test on actual 16GB machine, adjust safety margins |

## Definition of Done

- [ ] `./test_models.sh` passes for Qwen2.5-Coder-1.5B
- [ ] `./test_models.sh` passes for DeepSeek-Coder-V2-Lite
- [ ] `./test_models.sh` passes for GPT-OSS-20B
- [ ] All three models work on 16GB MacBook
- [ ] TurboQuant achieves 4x+ KV cache compression
- [ ] Server rejects model load with helpful error if insufficient memory
- [ ] Documentation updated with model requirements
- [ ] OpenCode integration tested and documented
