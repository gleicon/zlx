# zlx

Local inference server with OpenAI-compatible API for MLX.zig models on Apple Silicon.

Runs Qwen, Llama, Phi, and other models via MLX (Metal GPU acceleration). Exposes a `/v1/chat/completions` endpoint that works with OpenCode, Claude Desktop, and other OpenAI-compatible clients.

## Quick Start

### 1. Clone and Build

```bash
git clone https://github.com/gleicon/zlx
cd zlx
zig build -Doptimize=ReleaseFast
```

### 2. Download a Model

```bash
mkdir -p models
cd models

# Download Qwen2.5-Coder-1.5B-Instruct (recommended)
wget https://huggingface.co/mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit/resolve/main/model.safetensors
wget https://huggingface.co/mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit/resolve/main/config.json
wget https://huggingface.co/mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit/resolve/main/tokenizer.json

cd ..
```

### 3. Run

```bash
./zig-out/bin/zlx --model ./models/Qwen2.5-Coder-1.5B-Instruct-4bit --port 8081
```

### 4. Configure OpenCode

In OpenCode settings:

```
Provider: OpenAI Compatible
API URL: http://127.0.0.1:8081/v1
API Key: (leave empty)
Model: Qwen2.5-Coder-1.5B-Instruct-4bit
```

## Architecture

### Core Components

**HTTP Server (`src/api/server.zig`)**
- httpz-based HTTP/1.1 server with keep-alive
- OpenAI-compatible endpoints: `/v1/chat/completions`, `/v1/models`
- SSE streaming support for real-time responses
- 10MB request body limit to handle large conversation histories

**Request Handling (`src/api/handlers.zig`)**
- JSON parsing with custom field types for OpenAI compatibility
- Streaming and non-streaming response paths
- Request validation and error handling
- CORS support for browser clients

**Types (`src/api/types.zig`)**
- MessageContent: Handles both string and array content (for multimodal)
- JsonFloat: Accepts integers or floats for temperature/top_p
- ChatCompletionRequest: Full OpenAI API compatibility

**Streaming (`src/api/streaming.zig`)**
- SSE chunk generation with proper JSON structure
- UTF-8 safe token-by-token streaming
- Special token stripping (endoftext, im_end, etc.)

**Metrics (`src/api/metrics.zig`)**
- 10-second interval stats logging
- Tracks: requests/sec, tokens/sec, TTFT, memory
- Thread-safe counters

**Inference Core (`src/inference/mod.zig`)**
- MLX.zig integration for model loading
- Token generation with KV cache
- Context length limiting (8K tokens max)
- Metal GPU backend

### Data Flow

1. HTTP request arrives at `/v1/chat/completions`
2. JSON parsed into ChatCompletionRequest with custom field handlers
3. Messages converted to prompt using chat template
4. Prompt tokenized via MLX tokenizer
5. Transformer.generate() produces tokens
6. Tokens decoded to text
7. Special tokens stripped
8. Response formatted as OpenAI JSON or SSE chunks

## Dependencies

### Build Dependencies
- Zig 0.15.2+
- CMake 3.20+ (for mlx-c build)
- Xcode Command Line Tools (Metal SDK)
- curl (for downloading mlx-c)

### Runtime Dependencies
- macOS 14+ on Apple Silicon (M1/M2/M3)
- Metal-capable GPU
- No Python runtime required

### Libraries (vendored/built)
- MLX-C 0.1.2 (C bindings for MLX)
- MLX (C++ ML framework with Metal backend)
- httpz 0.0.0 (Zig HTTP server)
- pcre2 10.45 (regex for tokenizer)

## Build Instructions

### Prerequisites

```bash
# Install Zig
brew install zig

# Install CMake
brew install cmake

# Verify Xcode tools
xcode-select --install
```

### Build

```bash
# Clone
git clone https://github.com/gleicon/zlx
cd zlx

# Build (downloads and builds MLX-C automatically)
zig build -Doptimize=ReleaseFast

# Binary at: ./zig-out/bin/zlx
```

### First Build Notes
- Initial build downloads MLX-C 0.1.2 and builds it via CMake
- This takes 5-10 minutes on M1/M2 Macs
- Subsequent builds use cached mlx-c library
- Build artifacts in `.zig-cache/`

## Setup

### 1. Model Directory Structure

```
models/
└── Qwen2.5-Coder-1.5B-Instruct-4bit/
    ├── config.json          # Model configuration
    ├── tokenizer.json       # Tokenizer definition
    └── model.safetensors   # Weights (4-bit quantized)
```

### 2. Supported Models

Download from [mlx-community](https://huggingface.co/mlx-community) on HuggingFace:

**Recommended:**
- `mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit` - Best for coding, fast
- `mlx-community/Qwen2.5-7B-Instruct-4bit` - Larger, slower but better quality

**Other options:**
- `mlx-community/Llama-3.2-1B-Instruct-4bit`
- `mlx-community/Phi-3-mini-4k-instruct-4bit`

### 3. Configuration

Environment variables:
```bash
export ZLX_MODEL_PATH=./models/Qwen2.5-Coder-1.5B-Instruct-4bit
export ZLX_PORT=8081
export ZLX_HOST=127.0.0.1
```

Command line:
```bash
./zig-out/bin/zlx --model ./models/Qwen2.5-Coder-1.5B-Instruct-4bit --port 8081
```

### 4. Verification

```bash
# Check server is running
curl http://localhost:8081/v1/models

# Test generation
curl http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen2.5-Coder-1.5B-Instruct-4bit","messages":[{"role":"user","content":"Say hello"}]}'
```

## Client Configuration

### OpenCode
Settings -> Models -> Custom Model:
- Provider: OpenAI Compatible
- API URL: `http://127.0.0.1:8081/v1`
- API Key: (leave empty, no auth required)
- Model: `Qwen2.5-Coder-1.5B-Instruct-4bit`
- Max tokens: 4096 (server caps at this anyway)
- Temperature: 0.7

### Claude Desktop (claude_desktop_config.json)
```json
{
  "llm": {
    "provider": "openai-compatible",
    "baseUrl": "http://127.0.0.1:8081",
    "apiKey": "unused"
  }
}
```

### curl / Direct API
```bash
curl -X POST http://127.0.0.1:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-Coder-1.5B-Instruct-4bit",
    "messages": [
      {"role": "system", "content": "You are a helpful coding assistant."},
      {"role": "user", "content": "Write a hello world in Python"}
    ],
    "max_tokens": 500,
    "temperature": 0.7,
    "stream": false
  }'
```

## Monitoring

### Logs
```bash
# Real-time logs
tail -f /tmp/zlx.log

# Stats every 10 seconds
tail -f /tmp/zlx.log | grep STATS
# Output: [STATS] Requests: 42 (+3) | Tokens: 1250 (+89) | Avg TPS: 45.2 | TTFT: 1234.5ms

# GPU usage
sudo powermetrics -s gpu_power -n 1 2>/dev/null | grep "GPU active residency"
```

### Performance Expectations
- **Qwen 1.5B 4-bit**: ~40-60 tokens/sec on M1 Pro
- **Qwen 7B 4-bit**: ~15-25 tokens/sec on M1 Pro
- **With Speculative Decoding**: 1.5-2.8x speedup on compatible models (7B + 1.5B draft)
- **TTFT**: 500-2000ms depending on prompt length
- **Memory**: ~2GB for 1.5B model, ~5GB for 7B model, +3-4GB with draft model

### Speculative Decoding (NEW)

zlx supports speculative decoding for 1.5-2.8x speedup on compatible models:

```bash
# Automatic draft model selection (recommended)
zlx --model Qwen2.5-Coder-7B-4bit

# Manual draft model selection
zlx --model Qwen2.5-Coder-7B-4bit --draft-model Qwen2.5-Coder-1.5B-4bit

# View speculation metrics
curl http://localhost:8080/v1/metrics/speculative
```

Compatible pairs:
- Qwen 7B + Qwen 1.5B → 2.0-2.8x speedup
- Qwen 7B + Qwen 0.5B → 1.8-2.5x speedup

See [docs/SPECULATIVE_DECODING.md](docs/SPECULATIVE_DECODING.md) for full documentation.

## Troubleshooting

### "Cannot get gpu stream without gpu backend"
MLX was built without Metal support. Rebuild:
```bash
rm -rf .zig-cache/mlx-c
zig build -Doptimize=ReleaseFast
```

### "Invalid JSON in request body"
Check logs for field type mismatches. Common issues:
- `top_p: 1` instead of `top_p: 1.0` (fixed in current version)
- Array content `[]` instead of string `""` (fixed)
- Large strings with escaped chars (fixed)

### Slow generation
- Check GPU is being used: Activity Monitor -> GPU history
- Reduce context length: conversation history is truncated to 8K tokens
- Use smaller model (1.5B vs 7B)
- Check thermal throttling: `sudo thermal levels`

### Out of memory
- Context is auto-truncated to fit in 8K tokens
- Server caps max_tokens at 4096
- Use 4-bit quantized models (built-in)

## API Endpoints

### POST /v1/chat/completions
OpenAI-compatible chat completion endpoint. Accepts:
- `model`: Model identifier (must match loaded model)
- `messages`: Array of {role, content} objects
- `max_tokens`: Maximum tokens to generate (default: 256, max: 4096)
- `temperature`: 0.0-2.0 (default: 0.7)
- `top_p`: 0.0-1.0 (default: 0.9)
- `stream`: Boolean for SSE streaming
- `stop`: Stop sequences

### GET /v1/models
Returns available model (currently only the loaded model).

### GET /v1/health
Health check endpoint. Returns 200 OK.

### GET /v1/metrics/speculative
Speculative decoding metrics endpoint. Returns:
- `acceptance_rate`: Fraction of draft tokens accepted (0.0-1.0)
- `avg_tokens_per_step`: Average tokens per speculation round
- `estimated_speedup`: Calculated speedup based on acceptance rate

## Technical Details

### Model Format
Uses MLX safetensors format with quantization:
- 4-bit quantization for weights
- BF16 for activations
- Group size 64
- Symmetric quantization

### Tokenizer
BPE-based tokenizer from HuggingFace `tokenizer.json`:
- Vocabulary: ~150K tokens
- Special tokens: im_start, im_end, endoftext, fim_*, etc.
- Supports Chinese and code

### Memory Management
- Context length: 8K tokens max (input + output)
- Input truncation: Keeps most recent context when limit exceeded
- KV cache: Per-layer key-value caching for efficiency
- Metal memory pools: Automatic GPU memory management

## Development

### Project Structure
```
src/
├── main.zig              # CLI entry point
├── api/
│   ├── server.zig        # HTTP server setup
│   ├── handlers.zig      # Request handlers
│   ├── types.zig         # API types
│   ├── streaming.zig      # SSE implementation
│   └── metrics.zig       # Performance tracking
├── inference/
│   └── mod.zig           # MLX inference wrapper
└── mlx.zig/              # MLX.zig submodule
    ├── src/
    │   ├── mlx.zig       # C bindings
    │   ├── qwen.zig      # Qwen model implementation
    │   └── tokenizer.zig # Tokenization
    └── build.zig         # MLX.zig build config
```

### Testing
```bash
# Run tests
zig build test

# Test specific module
zig test src/api/types.zig
```

### Debugging
```bash
# Verbose logging
./zig-out/bin/zlx --model ./models/MODEL --port 8081 2>&1 | tee zlx.log

# Debug build (slower, more checks)
zig build
./zig-out/bin/zlx --model ./models/MODEL
```

## License

MIT License - See LICENSE file

MLX is Copyright (c) 2023-2024 Apple Inc. and the MLX project authors.
