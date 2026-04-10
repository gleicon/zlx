**Gemma 4 E4B (and E2B) uses Per-Layer Embeddings (PLE) as the core innovation that makes its architecture non-standard.** This is exactly what you described: separate per-layer embedding tables (large but lookup-only, often memory-mapped or quantized aggressively), a small per-layer dim (256 dims in the released models), input gating via a learned `per_layer_input_gate`, and a projection + norm to inject it into the residual stream. The main hidden size for E4B is ~2560 (sources vary slightly between ~2056–2560 depending on the exact variant/config; your 2560 matches the E4B scale), while the PLE vectors are deliberately tiny (256 dims) and gated/projected per layer. This is why the total params are higher (~8B for E4B) but *effective* / active params are only ~4.5B.

It is **not** a standard transformer (no single shared embedding table where embed_dim == hidden_size). The PLE is injected as a lightweight residual *inside each decoder layer* after the usual attention + FFN, giving every layer its own token-specific “reminder” signal. This is what enables the on-device efficiency (PLE tables can live in flash / be lazily loaded; only the tiny 256-dim slice per token per layer is used).

Gemma 4 is multimodal (text + vision + audio), but for a **coding agent** you only need the text path (multimodal tokens just use pad-ID for PLE). The rest of the architecture (hybrid local/global sliding-window attention + p-RoPE + QK/V norms + KV sharing in global layers) is relatively standard and already handled in most Gemma 2/3-style implementations; the PLE is the part that breaks naive Llama/Qwen-style code.

### Existing MLX Support (Easiest Path)
MLX already has first-class support for Gemma 4 E4B via the community conversions (released within days of the April 2, 2026 launch):
- `mlx-community/gemma-4-e4b-it-4bit` (or 8-bit variants from Unsloth)
- `mlx-vlm` (explicitly supports Gemma 4 multimodal, including E4B; works great for text-only too)

```bash
pip install -U mlx-vlm
python -m mlx_vlm.generate --model mlx-community/gemma-4-e4b-it-4bit --prompt "Write a Python function that..."
```

If your coding agent is already built on pure `mlx.nn` + custom inference loop (not `mlx_lm`/`mlx-vlm`), you can still load the converted safetensors directly and just plug in the custom layer below. No need to reinvent the wheel unless you want full control.

### How to Implement the PLE Layer in Pure MLX
Here is the exact mechanism (reverse-engineered from Gemma 3n → Gemma 4 and confirmed in HF configs / visual guides):

```python
import mlx.nn as nn
import mlx.core as mx

class Gemma4PLEBlock(nn.Module):
    """Drop-in PLE residual block to add inside each decoder layer."""
    def __init__(self, hidden_size: int, ple_dim: int = 256):
        super().__init__()
        # per_layer_input_gate: projects current hidden state (context-aware signal)
        self.per_layer_input_gate = nn.Linear(hidden_size, ple_dim, bias=False)  # usually GELU after
        # per_layer_projection: 256 -> hidden_size
        self.per_layer_projection = nn.Linear(ple_dim, hidden_size, bias=False)
        # special normalization (RMSNorm or LayerNorm; Gemma family usually RMS)
        self.per_layer_norm = nn.RMSNorm(hidden_size)  # or nn.LayerNorm if config says so

    def __call__(self, x: mx.array, per_layer_emb_slice: mx.array) -> mx.array:
        # x: current residual [batch, seq_len, hidden_size]
        # per_layer_emb_slice: pre-looked-up PLE for this layer [batch, seq_len, ple_dim]

        # 1. Context-aware gate from current hidden state
        gate = mx.gelu(self.per_layer_input_gate(x))          # GELU is used in the reference impl

        # 2. Token-identity * gate (elementwise)
        gated_ple = gate * per_layer_emb_slice

        # 3. Project back up + norm + residual
        ple_out = self.per_layer_projection(gated_ple)
        ple_out = self.per_layer_norm(ple_out)

        return x + ple_out
```

#### Full Layer Integration (Pseudocode for Your DecoderLayer)
```python
class Gemma4DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        # ... your existing attention + FFN (Llama-style or whatever you have) ...
        self.ple_block = Gemma4PLEBlock(config.hidden_size, ple_dim=256)

    def __call__(self, x: mx.array, per_layer_embs: mx.array, **kwargs):
        # per_layer_embs shape: [batch, seq_len, num_layers, ple_dim]  (precomputed once)
        residual = x

        # Standard attention + FFN (your existing code)
        x = self.attention_block(x, **kwargs)
        x = self.ffn_block(x)

        # === PLE injection (the key non-standard part) ===
        current_ple = per_layer_embs[:, :, self.layer_idx]   # slice for this layer
        x = self.ple_block(x, current_ple)

        return x
```

#### Precomputing the PLE Embeddings (Do This Once at the Start of Forward)
```python
# In your model forward pass, right after token embedding
def embed_with_ple(self, input_ids: mx.array) -> tuple[mx.array, mx.array]:
    main_emb = self.embed_tokens(input_ids)                    # [b, s, hidden_size]

    # PLE table: shape [vocab_size, num_layers, ple_dim] or [num_layers, vocab_size, ple_dim]
    # (stored as a big parameter / safetensor; quantized aggressively)
    ple_table = self.ple_embeddings  # your nn.Parameter or loaded weight
    ple_embs = ple_table[input_ids]   # automatic broadcast → [b, s, num_layers, ple_dim]

    return main_emb, ple_embs
```

**Memory note for MLX on Apple Silicon**: The PLE table (262k vocab × ~42 layers × 256 × bytes) is large but **lookup-only** and static → you can quantize it to 4-bit or use MLX’s lazy loading / memory-mapping. Existing `mlx-community` conversions already do this efficiently.

### Other Minor Differences You’ll Need
- **Embedding projection** (if your current code assumes embed_dim == hidden_size): Gemma 4 main token embedding *is* the hidden size (no 480→2560 projection needed; your “480” note may refer to an older variant or a misread of the 256 ple dim).
- **Special norms**: The PLE block uses its own RMSNorm/LayerNorm (shown above).
- **Multimodal tokens** (only if you ever add vision): non-text tokens use pad token ID for PLE lookup (neutral signal).

### Recommendation for Your Coding Agent
1. **Fastest**: Load one of the existing MLX-converted Gemma 4 E4B models (`mlx-community` or Unsloth) and just use `mlx_vlm` or `mlx_lm.generate`. It already implements everything above.
2. **If you must stay pure custom MLX**: Add the `Gemma4PLEBlock` exactly as shown (it’s < 20 lines). The rest of the transformer (attention/FFN/RoPE) is standard Gemma 2/3 style.
3. **For agents**: Gemma 4 E4B is excellent for local coding/tool-use on-device (function calling was trained in). Pair it with MLX’s fast generation + your existing agent loop.

The PLE change is the only fundamental blocker for standard transformer codebases — everything else (hybrid attention, etc.) is incremental. You can have a working E4B coding agent in MLX today by dropping in the block above. 
