# Phase 13: DeepSeek & GPT-OSS Completion

## Goal
Complete the MoE model support by implementing actual quantized weight dequantization for DeepSeek and full weight loading for GPT-OSS, enabling both models to pass test_models.sh and work with TurboQuant on small machines.

## Why This Phase

**Current State:**
- ✅ Phase 12: Infrastructure complete (architecture, detection, memory management)
- ⚠️ DeepSeek: Weights load but quantized components not dequantized (raw .weight/.biases/.scales)
- ⚠️ GPT-OSS: Architecture works but no weight loading implementation
- ❌ test_models.sh: Cannot pass for DeepSeek or GPT-OSS without working weight loading

**Problems to Solve:**
1. **DeepSeek Dequantization**: mlx-community format uses 4-bit quantized weights with `.weight`, `.biases`, `.scales` components that must be reconstructed to float16/float32
2. **GPT-OSS Weight Loading**: No `registerGptOssWeightKeys()` function exists in loader.zig
3. **MXFP4 Support**: GPT-OSS uses Microscaling FP4 quantization (different from DeepSeek's affine quantization)
4. **Integration Testing**: Both models must pass test_models.sh with actual weight files

## Scope

**In Scope:**
- DeepSeek 4-bit quantized weight dequantization (affine quantization)
- GPT-OSS weight loading with MXFP4 support
- Model download capability for GPT-OSS (11GB)
- Integration testing to verify both models work end-to-end
- TurboQuant verification with both MoE models

**Out of Scope:**
- New model architectures (this is completion, not expansion)
- Additional quantization formats (only 4-bit affine and MXFP4)
- Performance optimization beyond basic functionality
- Multi-GPU support

## Requirements Mapping

| Requirement | Priority | Description |
|-------------|----------|-------------|
| MOE-06 | P0 | DeepSeek 4-bit weight dequantization |
| MOE-07 | P0 | GPT-OSS weight loading |
| MOE-08 | P1 | GPT-OSS model download (11GB) |
| MOE-09 | P0 | test_models.sh passes for both models |
| MOE-10 | P1 | TurboQuant works with both models |

## Success Criteria

1. DeepSeek-Coder-V2-Lite loads weights, dequantizes correctly, and generates tokens
2. GPT-OSS-20B downloads weights (if needed), loads, and generates tokens
3. Both models pass `./test_models.sh` with all tests (load, inference, memory)
4. TurboQuant achieves 4.6x+ KV cache compression with both models
5. Both models work on 16GB MacBook with TurboQuant enabled
6. No "stub" or "not implemented" errors for either model

## Technical Approach

### 1. DeepSeek Quantized Weight Dequantization

**Problem:** Phase 12-01 FIX implemented weight mapping but left dequantization as stub:
```zig
// Current (stub): mapQuantizedWeight() returns raw components
const quantized = QuantizedWeight{
    .weight = weights_hash.get("xxx.weight"),     // 4-bit packed
    .biases = weights_hash.get("xxx.biases"),      // float32
    .scales = weights_hash.get("xxx.scales"),      // float32
};
// Missing: Reconstruction of float16/float32 weights
```

**Solution:**
Implement affine dequantization per MLX spec:
```
reconstructed_weight = (quantized_weight - biases) * scales
```

Where:
- `weight`: Packed 4-bit integers (2 values per byte)
- `biases`: Zero points per group
- `scales`: Scale factors per group
- Group size: typically 32 or 64

**Implementation:**
- Add `dequantizeAffine4Bit()` function in `src/inference/dequantize.zig`
- Integrate with existing `mapQuantizedWeight()` in loader.zig
- Support both float16 and float32 output
- Use MLX metal kernels for GPU acceleration

### 2. GPT-OSS Weight Loading

**Problem:** No weight loading implementation exists for GPT-OSS architecture.

**Solution:**
Follow DeepSeek pattern:
1. Add `registerGptOssWeightKeys()` in loader.zig
2. Implement `loadGptOssWeights()` function
3. Map weight names to GPT-OSS architecture

**Weight Naming (from Phase 12-02):**
- `model.embed_tokens.weight`
- `model.layers.{i}.input_layernorm.weight`
- `model.layers.{i}.post_attention_layernorm.weight`
- `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`
- `model.layers.{i}.mlp.router.weight`
- `model.layers.{i}.mlp.experts.{e}.{gate,up,down}_proj.weight`
- `model.norm.weight`
- `lm_head.weight`

**MXFP4 Consideration:**
GPT-OSS uses Microscaling FP4 (micro-exponent scaling):
- May require MLX custom dequantization
- Check if MLX handles MXFP4 natively or needs custom kernel
- Group sizes: 32 (main), 64 (embeddings/attention)

### 3. Model Download

**GPT-OSS Size:** ~11GB for 4-bit quantized version

**Implementation:**
- Reuse existing download infrastructure from Phase 9
- Add GPT-OSS to model registry with HuggingFace URL
- Support resume via HTTP Range requests
- Show progress during 11GB download

### 4. Integration Testing

**test_models.sh Requirements:**
- Load DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx
- Load GPT-OSS-20B-4bit (or downloaded equivalent)
- Test chat completions for both
- Verify memory usage stays under 16GB
- Verify TurboQuant compression

## Plans

| Plan | Name | Focus | Est. Hours |
|------|------|-------|------------|
| 13-01 | DeepSeek Quantized Weight Reconstruction | 4-bit affine dequantization, MLX integration | 6 |
| 13-02 | GPT-OSS Download and Weight Loading | Model download, MXFP4 support, weight mapping | 8 |
| 13-03 | Integration Testing and Verification | test_models.sh updates, end-to-end validation | 4 |

Total: ~18 hours

## Dependencies

- Phase 12 (all 5 plans + 3 gap closure fixes)
- Phase 11 (MoE infrastructure)
- MLX.zig with mlx-c v0.4.x
- TurboQuant from Phase 7

## Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| MLX doesn't support MXFP4 natively | Medium | High | Implement custom dequantization kernel or use CPU fallback |
| GPT-OSS 11GB download unreliable | Low | Medium | Retry logic, resume support, mirror URLs |
| Dequantization too slow on CPU | Medium | High | Use MLX metal kernels, benchmark before/after |
| Weight naming differs from expected | Medium | Medium | Log actual names, create mapping layer |

## Definition of Done

- [ ] `./test_models.sh --model DeepSeek-Coder-V2-Lite` passes all tests
- [ ] `./test_models.sh --model GPT-OSS-20B` passes all tests
- [ ] Both models generate coherent responses to prompts
- [ ] Memory usage under 16GB with TurboQuant enabled
- [ ] No stub functions or TODOs in weight loading code
- [ ] Documentation updated with model requirements
- [ ] v1.1.1 release ready with full MoE support
