# Native MLX Gemma 4 Implementation Plan

## Core Insight

MLX.zig (in `src/mlx.zig/src/mlx.zig`) provides a **generic Transformer** pattern:

```zig
pub fn Transformer(comptime ModelType: type, comptime ConfigType: type) type {
    return struct {
        // ... initialization and generation logic
    };
}
```

Each model exports:
```zig
// qwen.zig
pub const ModelConfig = mlx.ModelConfig;  // or custom config
pub const Model = mlx.Model(TransformerBlock, ModelConfig);
pub const Transformer = mlx.Transformer(Model, ModelConfig);
```

## What We Need for Gemma 4

1. **Gemma4Config** - Model configuration struct
2. **Gemma4TransformerBlock** - The actual layer implementation
3. **Gemma4Model** - Model assembly
4. **Gemma4Transformer** - The full transformer type alias
5. **Generic GenerationState** - Works with any transformer type

## Architecture

```
MLX.zig Generic Transformer System:
┌─────────────────────────────────────────────┐
│           mlx.Transformer()                  │
│              (generic)                       │
├─────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Model      │  │  TransformerBlock   │  │
│  │  (layers)   │  │  (attention + mlp)  │  │
│  └─────────────┘  └─────────────────────┘  │
│         │                    │              │
│         └────────────────────┘              │
│                   │                         │
│         ┌─────────▼──────────┐             │
│         │   Config (params)   │             │
│         └─────────────────────┘             │
└─────────────────────────────────────────────┘
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
┌────────┐    ┌────────┐    ┌──────────┐
│  Qwen  │    │  Llama │    │ Gemma 4  │
│Transformer│   │Transformer│   │ Transformer│
└────────┘    └────────┘    └──────────┘
```

## Implementation Steps

### Step 1: Create Gemma4Config

Gemma 4 specific parameters from config.json:
- hidden_size: 2560
- num_hidden_layers: 42
- num_attention_heads: 8
- num_key_value_heads: 2 (GQA)
- intermediate_size: 10240
- rms_norm_eps: 1e-6
- rope_theta: 10000.0
- sliding_window: 512
- vocab_size: 262144

### Step 2: Create Gemma4TransformerBlock

Components:
1. **SlidingWindowAttention** - Standard attention with window=512
2. **GatedMLP** - Standard SwiGLU MLP
3. **RMSNorm** - Pre-norm architecture

Key differences from Qwen:
- Sliding window attention (512 tokens)
- Fewer KV heads (2 vs 8)
- Pre-norm (norm before attention/MLP)

### Step 3: Update GenerationState to be Generic

Change from:
```zig
pub const GenerationState = struct {
    transformer: ?*qwen.Transformer = null,
    // ...
};
```

To:
```zig
pub fn GenerationState(comptime TransformerType: type) type {
    return struct {
        transformer: ?*TransformerType = null,
        // ...
    };
}
```

### Step 4: Update Inference Module

Add branch in model type switch:
```zig
switch (self.model_type) {
    .gemma4 => {
        var transformer = try gemma4.Transformer.init(...);
        var state = try GenerationState(gemma4.Transformer).init(...);
    },
    // ...
}
```

## Why This Works

1. **MLX Backend**: Uses MLX C++ framework (Metal GPU accelerated)
2. **Native**: No Python, no llama.cpp - pure MLX
3. **Fast**: Compiled Zig + MLX C++ = excellent performance
4. **Compatible**: Works with existing safetensors model

## Testing

1. Config loading from JSON
2. Transformer initialization
3. Token generation
4. Chat formatting
5. Integration with server

## Files to Create/Modify

**New:**
- `src/mlx.zig/src/gemma4.zig` - Main implementation

**Modify:**
- `src/inference/generator.zig` - Make GenerationState generic
- `src/inference/mod.zig` - Add Gemma4 branch, fix types
- `src/api/streaming.zig` - Update GenerationState usage
- `src/api/chat_gemma4.zig` - Use native MLX instead of llama.cpp

## Success Criteria

1. ✅ `zig build` succeeds
2. ✅ `./zig-out/bin/zlx --model gemma4-e4b` starts
3. ✅ Chat completion generates tokens
4. ✅ Streaming works
5. ✅ OpenCode connects and works

## Timeline

- Step 1-2: 1-2 hours (Gemma4 implementation)
- Step 3-4: 1-2 hours (GenerationState generic)
- Testing: 1 hour
- Total: 3-5 hours

## Comparison with llama.cpp Approach

| Aspect | Native MLX | llama.cpp |
|--------|-----------|-----------|
| Speed | Faster (no conversion) | Slower (GGUF loading) |
| Purity | ✅ Pure MLX stack | ❌ Mixed backends |
| Complexity | Medium (needs transformer) | Low (uses existing) |
| Maintenance | One stack | Two stacks |
| Goal Alignment | ✅ Perfect | ❌ Compromise |

## Decision

**Use Native MLX** - Aligns with project goal of being a native Mac MLX app.

The complexity is worth it for:
- Performance (no conversion overhead)
- Purity (single MLX stack)
- Future-proofing (easier to add TurboQuant later)
- User experience (works with original model)
