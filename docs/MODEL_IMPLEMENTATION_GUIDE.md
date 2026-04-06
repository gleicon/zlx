# Model Implementation Guide

A practical guide for adding new LLM architectures to zlx.

## What Does "Implementing a Model" Mean?

Adding a model to zlx means teaching the system to:
1. **Parse the config** (`config.json`) — understand architecture, layers, hidden size, attention type
2. **Load the weights** (`.safetensors`) — map weight files to correct memory locations
3. **Run inference** — execute forward pass through the architecture
4. **Use the right attention** — dense, GQA, MQA, MLA, or sliding window
5. **Handle MoE if needed** — route tokens to the right experts (DeepSeek, GPT-OSS)
6. **Apply TurboQuant** — compress KV cache for small machines (optional)

---

## Architecture Components

### 1. Config Parser
Reads `config.json` and extracts:
- `model_type`: "qwen2", "deepseek", "gpt-oss"
- `vocab_size`, `hidden_size`, `num_hidden_layers`
- `num_attention_heads`, `num_key_value_heads` (for GQA/MQA)
- `rope_theta`, `max_position_embeddings` (context length)
- MoE configs: `n_routed_experts`, `num_experts_per_tok`, `n_shared_experts`
- Special: `use_mla`, `use_sliding_window`, `attention_dropout`

**File:** `src/models/model_config.zig` or similar

### 2. Weight Loader
Maps safetensors keys to internal arrays:
```
model.layers.0.self_attn.q_proj.weight → layer0.attention.q_weights
model.layers.0.mlp.gate_proj.weight → layer0.mlp.gate_weights
```

**Key challenges:**
- Naming conventions vary (HuggingFace vs mlx-community)
- Quantized weights have suffixes (`.scales`, `.biases` for 4-bit)
- MoE weights: `mlp.experts.{i}` vs `mlp.switch_mlp`
- MLA weights: `kv_a_proj`, `kv_b_proj` (compressed KV cache)

**File:** `src/inference/loader.zig`

### 3. Transformer Implementation
The actual forward pass logic:
- **Dense models** (Qwen): Standard self-attention + MLP
- **GQA/MQA**: Fewer KV heads than query heads (memory efficient)
- **MLA** (DeepSeek): Multi-head Latent Attention compresses KV by 90%
- **Sliding window** (GPT-OSS): Alternating full/local attention
- **MoE routing**: Choose top-k experts per token

**Files:**
- `src/qwen.zig` — Dense transformer
- `src/deepseek.zig` — MoE + MLA
- `src/gpt_oss.zig` — MoE + sliding window + Yarn RoPE

### 4. Tokenizer
Converts text ↔ tokens. Usually provided by MLX-community models as `tokenizer.json`.
**File:** Uses MLX.zig's tokenizer, may need chat template customization.

### 5. Chat Template
Formats messages for the model:
```
Qwen: <|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n
DeepSeek: User: {msg}\n\nAssistant:
```
**File:** `src/inference/chat_template.zig`

---

## Component Checklist

When adding a new model, you need:

| Component | Dense (Qwen) | MoE (DeepSeek) | MoE+SW (GPT-OSS) |
|-----------|--------------|----------------|------------------|
| Config parser | ✅ Basic | ✅ + MoE params | ✅ + sliding window |
| Weight loader | ✅ Standard | ✅ + quantized groups | ✅ + 32 experts |
| Attention | ✅ Dense | ✅ MLA | ✅ Sliding window |
| MoE router | ❌ | ✅ Top-6 of 64 | ✅ Top-4 of 32 |
| RoPE scaling | ✅ Standard | ✅ Standard | ✅ Yarn (128K ctx) |
| TurboQuant | ✅ Compatible | ✅ Compatible | ✅ Compatible |

---

## Quick Reference: Model Types

### Dense Models (Qwen2, Llama)
```
Every layer: Attention → MLP (dense)
Parameters = active parameters
Memory = weights + KV cache
```

### MoE Models (DeepSeek, GPT-OSS, Mixtral)
```
Each layer: Attention → Router → Selected Experts
Parameters = active × num_experts (sparse)
Memory = active params + small router overhead
KV cache = same as dense (but per-expert)
```

### Special Features
- **MLA (DeepSeek)**: KV cache is 90% smaller via latent compression
- **Sliding Window (GPT-OSS)**: Alternates full/128-token attention
- **Yarn RoPE**: Scales to 128K context without losing precision

---

## Implementation Steps

1. **Find a reference**: Get `config.json` and weight keys from HuggingFace
2. **Create config struct**: Map JSON fields to Zig struct
3. **Implement transformer**: Define forward pass (can reuse components)
4. **Map weight keys**: Match safetensors keys to internal arrays
5. **Add tests**: Verify against PyTorch reference outputs
6. **Document**: Memory requirements, compatible features

---

## Common Pitfalls

- **Weight naming**: mlx-community often differs from HuggingFace naming
- **Quantization**: 4-bit models need special handling (`.scales`, `.biases`)
- **Layer 0 special**: Some MoE models have dense Layer 0 (DeepSeek)
- **KV cache size**: MoE models need TurboQuant more than dense models
- **Chat templates**: Models fail silently if template doesn't match

---

## Example: Adding a Model

```zig
// 1. Config struct
pub const MyModelConfig = struct {
    vocab_size: usize,
    hidden_size: usize,
    num_layers: usize,
    use_moe: bool,
    num_experts: usize,
    experts_per_token: usize,
};

// 2. Parse config.json
pub fn parseConfig(allocator: Allocator, json_content: []const u8) !MyModelConfig {
    // Use std.json.parse
}

// 3. Transformer (reuse components)
pub const MyTransformer = struct {
    layers: []Layer,
    
    pub fn forward(self: *Self, tokens: []const u32, cache: *KVCache) ![]f32 {
        // Apply attention + MoE layers
    }
};

// 4. Weight mapping
pub fn registerWeights(loader: *WeightLoader) void {
    loader.map("model.layers.{i}.self_attn.q_proj.weight", &.{"layers", "{i}", "attn", "q"});
    loader.map("model.layers.{i}.mlp.experts.{e}.gate_proj.weight", &.{"layers", "{i}", "mlp", "experts", "{e}", "gate"});
}
```

---

## Files to Touch

| Task | File(s) |
|------|---------|
| Add model config | `src/models/model_config.zig` |
| Add transformer | `src/{model_name}.zig` |
| Add weight loading | `src/inference/loader.zig` |
| Add chat template | `src/inference/chat_template.zig` |
| Register model | `src/models/registry.zig` |
| Add tests | `tests/integration/{model}_test.zig` |
| Document | `docs/models/{model}.md` |

---

## Testing Checklist

- [ ] Config parses without errors
- [ ] All weight keys found and loaded
- [ ] Model generates tokens (no crash)
- [ ] Output quality reasonable (not gibberish)
- [ ] Memory usage matches estimate
- [ ] TurboQuant works (if supported)
- [ ] Chat completions work end-to-end

---

**Key Principle:** Reuse as much as possible. Most models are 80% similar.

- Attention patterns: reuse GQA, MQA, MLA modules
- MoE routing: reuse top-k selection
- Weight loading: reuse quantization handling
- Chat templates: reuse formatting with variations
