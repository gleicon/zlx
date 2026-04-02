# Speculative Decoding in zlx

## Overview

Speculative decoding is a performance optimization that uses a small, fast "draft" model to predict tokens ahead of time, then verifies those predictions with the full target model. This technique achieves **1.5-2.8x speedup** with no degradation in output quality.

### How It Works

1. **Draft Generation**: The small draft model generates K tokens autoregressively (default K=4)
2. **Parallel Verification**: The target model runs once on all K draft tokens in parallel
3. **Accept/Reject**: Each draft token is accepted with probability `min(1, p_target / p_draft)`
4. **Continue**: If all tokens accepted, continue speculation; if rejected, restart from rejection point

## Algorithm Details

### Acceptance Probability

For each token position, the acceptance probability is:

```
P(accept) = min(1.0, p_target(token) / p_draft(token))
```

Where:
- `p_target(token)` is the target model's probability for the draft token
- `p_draft(token)` is the draft model's probability for the draft token

### Speedup Formula

Expected speedup depends on speculation depth (K) and acceptance rate (α):

```
speedup ≈ K / (1 + (1-α) * K)
```

With K=4 and 75% acceptance: **2.0x speedup**
With K=4 and 100% acceptance: **4.0x speedup**

## Usage

### Automatic Draft Selection

zlx automatically selects compatible draft models from your model registry:

```bash
zlx --model Qwen2.5-Coder-7B-4bit
# Automatically selects: Qwen2.5-Coder-1.5B-4bit (if available)
```

Selection criteria:
- Same architecture family (Qwen → Qwen, Llama → Llama)
- Same tokenizer (matching vocab size)
- Size ratio between 1:4 and 1:8
- Same quantization type

### Manual Draft Selection

Override automatic selection with `--draft-model`:

```bash
zlx --model Qwen2.5-Coder-7B-4bit \
    --draft-model Qwen2.5-Coder-0.5B-4bit \
    --speculation-depth 6
```

### Disabling Speculation

```bash
zlx --model Qwen2.5-Coder-7B-4bit --no-speculation
```

## Configuration Options

### CLI Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--draft-model <NAME>` | Draft model to use (null = auto) | auto |
| `--speculation-depth N` | Tokens to speculate ahead (1-8) | 4 |
| `--no-speculation` | Disable speculative decoding | false |

### Recommended Settings

For most use cases, the defaults work well:

- **Default depth (4)**: Optimal balance of speedup and memory
- **Auto-selection**: Finds best draft from available models

## Compatible Draft/Target Pairs

| Target Model | Recommended Draft | Expected Speedup |
|--------------|-------------------|------------------|
| Qwen 7B | Qwen 1.5B | 2.0-2.8x |
| Qwen 7B | Qwen 0.5B | 1.8-2.5x |
| Llama 8B | Llama 1B | 1.5-2.2x |

### Architecture Requirements

Draft and target models must have:
- **Same tokenizer** (identical vocab size)
- **Compatible architecture** (e.g., both Qwen2.5, both Llama3)
- **Proper size ratio** (draft should be 1/4 to 1/8 the size)

## Monitoring

### Metrics Endpoint

Access speculative decoding metrics:

```bash
curl http://localhost:8080/v1/metrics/speculative
```

Example response:

```json
{
  "speculative_decoding": {
    "enabled": true,
    "draft_model": "Qwen2.5-Coder-1.5B-4bit",
    "speculation_depth": 4,
    "stats": {
      "total_speculations": 1523,
      "tokens_accepted": 4874,
      "tokens_rejected": 1218,
      "acceptance_rate": 0.80,
      "avg_tokens_per_step": 3.2,
      "estimated_speedup": 2.35
    }
  }
}
```

### Key Metrics

| Metric | Description | Target Value |
|--------|-------------|--------------|
| `acceptance_rate` | Fraction of draft tokens accepted | > 0.6 |
| `avg_tokens_per_step` | Average tokens per speculation | > 2.0 |
| `estimated_speedup` | Calculated speedup based on acceptance | 1.5-2.8x |

## Troubleshooting

### Low Acceptance Rate (< 0.4)

**Symptoms**: Speedup below 1.5x, many rejections

**Solutions**:
1. Use a larger draft model (closer size ratio)
2. Reduce speculation depth: `--speculation-depth 3`
3. Verify draft and target use same tokenizer
4. Check that models are from same architecture family

### Out of Memory

**Symptoms**: System runs out of GPU memory

**Solutions**:
1. Use `--no-speculation` flag
2. Use a smaller draft model
3. Reduce speculation depth to 2-3

### Draft Model Not Found

**Symptoms**: "No compatible draft model found" message

**Check**:
1. Draft model is in `./models/` directory
2. Draft model has same architecture as target
3. Models are properly downloaded with all required files

## Implementation Details

### Memory Usage

Speculative decoding requires:
- **Draft model weights**: ~3GB for 1.5B params (4-bit quantized)
- **Draft KV cache**: ~200MB for 4K context
- **Target KV cache**: Standard memory (unaffected by speculation)

**Total overhead**: ~3-4GB additional GPU memory

### Thread Safety

All speculative decoding operations are thread-safe:
- Metrics use atomic counters
- Draft model manager uses mutex for state changes
- Multiple concurrent generations are supported

### Fallback Behavior

If speculation fails:
1. Draft model unavailable → Falls back to standard generation
2. Draft model load fails → Logs warning, uses standard generation
3. Memory budget exceeded → Skips draft loading
4. Acceptance rate drops below threshold → Can auto-disable (configurable)

## Technical Reference

### Algorithm Source

Based on: "Fast Inference from Transformers via Speculative Decoding" (arXiv:2211.17192)

### Configuration Precedence

1. CLI flags (highest priority)
2. Environment variables
3. Config file settings
4. Default values (lowest priority)

### Files

- `src/speculation/speculative_generator.zig` - Core algorithm
- `src/speculation/draft_selector.zig` - Auto-selection logic
- `src/models/draft_model.zig` - Draft model management
- `src/speculation/mod.zig` - Public API
- `src/metrics/speculative_metrics.zig` - Metrics collection
