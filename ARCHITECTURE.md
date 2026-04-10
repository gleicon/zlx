## Architecture

### System Overview

zlx is a minimal Zig HTTP server exposing OpenAI-compatible `/v1/chat/completions` for local coding models on Apple Silicon.

```
┌─────────────────────────────────────────────────────────────┐
│                        zlx Architecture                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │ HTTP Handler │───▶│ Model Router │───▶│ MLX Backend  │ │
│  │   (httpz)    │    │  (Registry)  │    │   (MLX.zig)  │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│         │                   │                   │         │
│         ▼                   ▼                   ▼         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │   OpenAI     │    │   Model      │    │   MLX C      │ │
│  │  API Format  │    │   Loading    │───▶│   Bindings   │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│                                                   │         │
│                                                   ▼         │
│                                            ┌──────────────┐ │
│                                            │  MLX C++     │ │
│                                            │   Metal      │ │
│                                            └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

1. **HTTP Layer** (`src/api/`)
   - httpz for HTTP/1.1 server
   - OpenAI-compatible JSON request/response format
   - SSE streaming support

2. **Model Registry** (`src/models/`)
   - Alias resolution (e.g., "qwen" → "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit")
   - Local model directory scanning
   - Memory estimation and safety checks

3. **MLX Backend** (`src/mlx.zig/`)
   - Transformer generic with attention, MLP, normalization
   - Model-specific implementations (Qwen, Llama, Gemma, etc.)
   - C bindings to mlx-c

4. **Inference** (`src/inference/`)
   - Generation state machine
   - KV cache management
   - Token sampling

### Model Architecture Patterns

#### Standard Transformer (Qwen, Llama, GPT-OSS)
```
embed_tokens → [layer × N] → norm → lm_head

where layer = attention + FFN + residuals
```

#### Per-Layer Embeddings (Gemma 4 E4B)
```
embed_tokens ─┬─▶ [layer_0 + PLE_0] ─┐
              ├─▶ [layer_1 + PLE_1] ┤
              │           ...       ├──▶ norm → lm_head
              └─▶ [layer_N + PLE_N] ┘

where PLE_n = per_layer_input_gate(embed) * embed_tokens_per_layer[token, n]
              → per_layer_projection → per_layer_norm
```

**Critical Difference:** Gemma 4 requires PLE lookup and injection at every layer.

### Known Limitations

1. **MLX C Bindings:** Manual and brittle - every new MLX feature needs binding updates
2. **Weight Loading:** Regex-based key prefix stripping is fragile
3. **No PLE:** Gemma 4 E4B not yet supported (architecture mismatch)
4. **DeepSeek MoE:** Router logic incomplete

### Future Architecture Directions

**Option A: Continue with Zig**
- Implement PLE architecture properly
- Better abstractions for weight loading
- Maintain single-binary simplicity

**Option B: Swift MLX**
- Native `import MLX` (no bindings)
- Access to mlx-swift ecosystem
- Full rewrite required

See ARCHITECTURE_ANALYSIS.md for detailed comparison.

### Conventions

- Model configs loaded from `text_config` (not root) for Gemma 4 compatibility
- Weight keys registered with full path, stripped during load
- All MLX arrays use stream-based operations for GPU
- Memory allocator passed explicitly, no globals
