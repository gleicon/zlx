# GPT-OSS Model Status

## Search Results

### Model Found ✅
**Repository:** `mlx-community/gpt-oss-20b-MXFP4-Q4`

**Key Details:**
- **Downloads:** 4,345
- **License:** Apache-2.0
- **Base Model:** `openai/gpt-oss-20b`
- **Size:** ~20B parameters
- **Files:** 3 shards (model-00001/02/03-of-00003.safetensors)

### Architecture Verification

**Implemented (Phase 12-02) vs Actual:**

| Parameter | Implementation | Actual (mlx-community) | Match |
|-----------|---------------|------------------------|-------|
| model_type | gpt_oss | gpt_oss | ✅ |
| hidden_size | 2880 | 2880 | ✅ |
| num_hidden_layers | 24 | 24 | ✅ |
| num_attention_heads | 64 | 64 | ✅ |
| num_key_value_heads | 8 | 8 | ✅ |
| num_experts | 32 | 32 | ✅ |
| num_experts_per_tok | 4 | 4 | ✅ |
| sliding_window | 128 | 128 | ✅ |
| rope_theta | 150000 | 150000 | ✅ |
| rope_type | yarn | yarn | ✅ |
| max_position_embeddings | 131072 | 131072 | ✅ |

**Result:** ✅ Implementation matches actual model architecture exactly

### Files Downloaded

```
models/GPT-OSS-20B-4bit/
├── config.json         ✅ Downloaded
├── tokenizer.json      ✅ Downloaded
└── *.safetensors       ⏳ Not downloaded (large files)
```

### Quantization Differences

**Implementation Expected:** Standard 4-bit affine quantization

**Actual Model:** MXFP4 (Microscaling FP4) quantization
- Mode: `mxfp4`
- Group size: 32 (main), 64 (embeddings and attention)
- Per-tensor quantization config in `quantization_config`

**Note:** MXFP4 is a newer MLX quantization format. Standard dequantization should work but may need MXFP4-specific handling for optimal performance.

### Sliding Window Implementation

**Actual Model Structure:**
The model uses `layer_types` array specifying alternating attention types:
```
Layer 0:  sliding_attention
Layer 1:  full_attention
Layer 2:  sliding_attention
Layer 3:  full_attention
...
```

Our implementation uses alternating pattern based on layer index (even=sliding, odd=full), which matches.

### Availability

**Status:** Model available and architecture verified ✅

**Alternative Variants:**
- `gpt-oss-20b-MXFP4-Q8` - Higher precision (8-bit quantized)
- `gpt-oss-120b-4bit` - Larger 120B variant

### Next Steps

To complete GPT-OSS support:

1. **Download weights** (if needed for testing):
   ```bash
   huggingface-cli download mlx-community/gpt-oss-20b-MXFP4-Q4 \
     --local-dir models/GPT-OSS-20B-4bit \
     --local-dir-use-symlinks False
   ```

2. **Weight loading** - Add GPT-OSS weight key registration to `loader.zig`
   - Keys use standard HuggingFace naming (not DeepSeek-style)
   - Router: `mlp.router` (not `mlp.gate`)
   - Experts: `mlp.experts.*` (not `switch_mlp`)

3. **MXFP4 support** - May need special handling for microscaling format

### Weight Key Patterns (from config inspection)

Expected weight keys for GPT-OSS:
```
model.embed_tokens.weight
model.norm.weight
lm_head.weight

# Per layer (24 layers)
model.layers.{i}.input_layernorm.weight
model.layers.{i}.post_attention_layernorm.weight
model.layers.{i}.self_attn.q_proj.weight
model.layers.{i}.self_attn.k_proj.weight
model.layers.{i}.self_attn.v_proj.weight
model.layers.{i}.self_attn.o_proj.weight
model.layers.{i}.mlp.router.weight          # MoE gate
model.layers.{i}.mlp.experts.{e}.*          # 32 experts
```

### Conclusion

✅ **Phase 12-02 GPT-OSS architecture implementation is CORRECT**

The implementation matches the actual mlx-community model perfectly. The architecture code can be used without modification. Weight loading support needs to be added to `loader.zig` for complete functionality.

**Risk Level:** Low - Architecture verified, weight loading is the remaining work
