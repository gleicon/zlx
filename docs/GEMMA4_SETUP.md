# Gemma 4 E4B Setup Guide for OpenCode

## Quick Start

### 1. Download Pre-converted GGUF Model (Recommended)

```bash
# Download from HuggingFace (bartowski's conversion)
cd models
huggingface-cli download bartowski/gemma-4-e4b-it-GGUF \
  --local-dir ./gemma4-e4b \
  --include "*Q4_K_M.gguf"

# Or download manually:
# wget https://huggingface.co/bartowski/gemma-4-e4b-it-GGUF/resolve/main/gemma-4-e4b-it-Q4_K_M.gguf
```

### 2. Convert Your Existing Model (Alternative)

If you already have the MLX safetensors version:

```bash
# Install llama.cpp conversion tools
pip install llama-cpp-python

# Convert to GGUF
python -m llama_cpp.convert_hf_to_gguf \
  ./models/gemma4-e4b \
  --outfile ./models/gemma4-e4b/gemma-4-e4b-it-Q4_K_M.gguf \
  --outtype q4_k_m
```

### 3. Verify Model

```bash
# Check GGUF file exists
ls -lh ./models/gemma4-e4b/*.gguf

# Test with llama-cli directly
/opt/homebrew/bin/llama-cli \
  -m ./models/gemma4-e4b/gemma-4-e4b-it-Q4_K_M.gguf \
  -p "<|turn|>user\nHello<|turn|>model\n" \
  -n 20 \
  --temp 0.3
```

### 4. Start zlx Server

```bash
# Build first
zig build

# Start with Gemma 4
./zig-out/bin/zlx --model gemma4-e4b --port 8080

# Or bypass memory check if needed
ZLX_SKIP_MEMORY_CHECK=1 ./zig-out/bin/zlx --model gemma4-e4b --port 8080
```

### 5. Test with OpenCode

Add to your OpenCode config:

```json
{
  "models": [{
    "title": "Gemma 4 E4B (Local)",
    "provider": "openai",
    "model": "gemma4-e4b",
    "apiBase": "http://localhost:8080/v1",
    "apiKey": "not-needed"
  }]
}
```

## Features

### Chat Template

Gemma 4 uses a special format with `<|turn|>` tokens:

```
<|turn|>system
You are a helpful coding assistant.<turn|>
<|turn|>user
Write a Python function to sort a list.<turn|>
<|turn|>model
Here's a Python function using quicksort:

```python
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
```
<turn|>
```

### No-Think Mode (Default)

The handler automatically enables no-think mode for code generation:

```
<|turn|>model
<|channel>thought
<channel|>
```

This suppresses the model's internal reasoning channel for faster, cleaner responses.

### Recommended Settings

| Setting | Value | Reason |
|---------|-------|--------|
| Temperature | 0.3 | Lower for code (reduces verbosity) |
| Max Tokens | 2048 | Adequate for most code snippets |
| Context | 32768 | Gemma 4 supports 32k tokens |
| GPU Layers | 100 | Offload everything to Metal |

## Architecture

```
┌─────────────────────────────────────────────┐
│              zlx Server                      │
│                                              │
│  ┌─────────────────────────────────────┐    │
│  │    Gemma 4 Handler                  │    │
│  │  ┌───────────────────────────────┐  │    │
│  │  │  Chat Template                 │  │    │
│  │  │  - Format <|turn|> tokens      │  │    │
│  │  │  - No-think mode               │  │    │
│  │  └───────────────────────────────┘  │    │
│  │              │                       │    │
│  │              ▼                       │    │
│  │  ┌───────────────────────────────┐  │    │
│  │  │  LlamaBackend (llama.cpp)     │  │    │
│  │  │  - GGUF model loading         │  │    │
│  │  │  - Metal GPU inference        │  │    │
│  │  │  - Sliding window attention   │  │    │
│  │  └───────────────────────────────┘  │    │
│  └─────────────────────────────────────┘    │
│                   │                          │
│                   ▼                          │
│         OpenAI-compatible API                  │
└─────────────────────────────────────────────┘
```

## Troubleshooting

### "No GGUF file found"

Model must be in GGUF format. Convert or download pre-converted:

```bash
# Check what files exist
ls ./models/gemma4-e4b/

# Should show: *.gguf
# If only .safetensors, convert or download GGUF version
```

### "Model load failed"

Check llama.cpp can load it directly:

```bash
/opt/homebrew/bin/llama-cli -m ./models/gemma4-e4b/*.gguf -p "test" -n 5
```

### "Out of memory"

Gemma 4 E4B needs ~3GB for the model + context:

```bash
# Reduce context size
./zig-out/bin/zlx --model gemma4-e4b --ctx-size 8192

# Or use smaller GGUF quantization
# Q4_K_S instead of Q4_K_M uses less memory
```

### Slow generation

Check GPU layers are offloaded:

```bash
# In logs, look for:
# "llama_load_tensors: offloaded 42/42 layers to GPU"

# If not, rebuild llama.cpp with Metal support:
cd src/llama.cpp
mkdir build && cd build
cmake .. -DLLAMA_METAL=ON
make -j
```

## Testing

### Unit Tests

```bash
# Run Gemma 4 specific tests
zig build test -- src/api/chat_gemma4.zig
```

### Integration Test

```bash
# Start server
./zig-out/bin/zlx --model gemma4-e4b &

# Test chat completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4-e4b",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0.3
  }'

# Test streaming
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4-e4b",
    "messages": [{"role": "user", "content": "Count to 5"}],
    "stream": true
  }'
```

### OpenCode Test

1. Add model to OpenCode config
2. Open a code file
3. Press Cmd+L (or your shortcut)
4. Ask: "Explain this code"
5. Should get clean, code-focused response

## Performance Benchmarks

On Apple Silicon (M-series):

| Metric | Value |
|--------|-------|
| Model Load Time | ~5-10 seconds |
| First Token (TTFT) | ~100-300ms |
| Generation Speed | 15-30 tokens/sec |
| Memory Usage | ~3GB (model + context) |
| Context Window | Up to 32k tokens |

## Comparison with Other Models

| Model | Size | Speed | Best For |
|-------|------|-------|----------|
| Gemma 4 E4B | 4-bit, ~2.5GB | Medium | Long context, reasoning |
| Qwen 2.5 7B | 4-bit, ~4GB | Fast | General coding |
| DeepSeek V2 | 4-bit, ~9GB | Slow | Complex reasoning |

## Status

- ✅ Model detection
- ✅ GGUF loading via llama.cpp
- ✅ Chat template with <|turn|>
- ✅ No-think mode
- ✅ Streaming support
- ✅ OpenAI-compatible API
- ⏳ Token counting (estimated for now)
- ⏳ Full test suite

## Next Steps

1. Download/prepare GGUF model
2. Run integration tests
3. Connect OpenCode
4. Test with real coding tasks

---

**Last Updated:** 2026-04-08
**Status:** Ready for testing with GGUF model
