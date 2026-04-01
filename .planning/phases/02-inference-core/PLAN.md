# Phase 2: Inference Core - Plan

## Goal

A CLI harness loads a quantized model from `./models/` and streams tokens to stdout one at a time via Metal GPU.

## Success Criteria

1. Running `./zig-out/bin/zlx --model qwen2.5-coder-7b` loads model weights from `./models/qwen2.5-coder-7b/` without crash
2. Token generation runs on Metal GPU — Activity Monitor shows GPU activity during generation
3. `GenerationState.next()` yields one token per call — observable as incremental stdout output in the CLI harness
4. Sending two back-to-back generation requests does not corrupt output — mutex serialization is effective
5. `--model`, `--port`, and `--max-kv-size` flags are accepted and applied

## Requirements Mapping

- INFER-01: Model loading from disk (config.json, weights.safetensors, tokenizer.json)
- INFER-02: Tokenizer integration (encode prompts, decode tokens)
- INFER-03: GPU-based token generation loop via MLX.zig Transformer
- INFER-04: Generation state management (KV cache, position tracking)
- INFER-05: Thread-safe inference for concurrent requests

## Architecture Overview

```
src/
├── main.zig          # CLI entry, flag parsing, model path resolution
├── server.zig        # HTTP server setup (Phase 3 preparation)
├── inference/
│   ├── mod.zig       # Public API exports
│   ├── loader.zig    # Model weight loading from safetensors
│   ├── tokenizer.zig # Tokenizer wrapper (BPE/tiktoken)
│   └── generator.zig # GenerationState struct with .next() iterator
└── models/
    └── mod.zig       # Model configuration types
```

## Key Design Decisions

1. **MLX.zig integration**: Use existing `Transformer.init()` and `generate()` from MLX.zig, wrapping in a stateful iterator pattern
2. **Model format**: Support Safetensors weights (MLX native), HuggingFace format with config.json
3. **Tokenizer**: Leverage MLX.zig's tokenizer bindings or implement BPE in Zig
4. **Concurrency**: Single global model instance with mutex-guarded `generate()` calls
5. **CLI flags**: `--model` (required), `--port` (8080 default), `--max-kv-size` (context length)

## Tasks

1. **Research MLX.zig Transformer API** (30 min)
   - Read src/mlx.zig/src/transformer.zig or similar
   - Document: init(), generate() signatures, required parameters
   - Verify GPU stream setup

2. **Create model loading infrastructure** (45 min)
   - Implement loader.zig: loadModelFromPath(path) → Model struct
   - Parse config.json for architecture params
   - Load weights from .safetensors files
   - Error handling for missing files

3. **Implement tokenizer wrapper** (30 min)
   - Create tokenizer.zig with encode() and decode() methods
   - Support Qwen/Llama tokenizer formats
   - Handle special tokens (bos, eos, pad)

4. **Build GenerationState iterator** (60 min)
   - Create generator.zig with GenerationState struct
   - Implement next() → ?Token that calls MLX.zig generate step
   - Manage KV cache state between calls
   - Support max_tokens limit and stop sequences

5. **CLI harness and flag parsing** (30 min)
   - Add clap or std.process.arg support
   - Parse --model, --port, --max-kv-size
   - Validate model path exists
   - Print help/usage

6. **Test GPU inference** (30 min)
   - Run with sample prompt: "Write a hello world in Python"
   - Verify token-by-token output
   - Check Activity Monitor for GPU usage

7. **Implement thread safety** (30 min)
   - Add std.Thread.Mutex around model.generate()
   - Test concurrent requests

## Risks

- **Risk**: MLX.zig generate() API may not support incremental token streaming
  - *Mitigation*: May need to wrap full generate() and manually stream output
  
- **Risk**: Tokenizer format complexity (BPE merge files)
  - *Mitigation*: Start with MLX.zig's built-in tokenizer if available

- **Risk**: Large model files (7B+ params) loading slowly
  - *Mitigation*: Memory-map weights, load on first request

## Success Verification

Run the following and confirm output:

```bash
# Build
zig build

# Test model loading
./zig-out/bin/zlx --model qwen2.5-coder-7b --help

# Test generation
./zig-out/bin/zlx --model qwen2.5-coder-7b
# (interactive mode, type prompt and see tokens stream)
```

Check GPU usage in Activity Monitor → Window → GPU History during generation.

## Time Estimate

- Research: 30 min
- Implementation: 3 hours
- Testing: 30 min
- **Total: ~4 hours**
