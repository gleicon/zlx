Technical Product Specification: Zig-MLX Inference Server ("mlx-zig-serve")Version: 1.0 (Draft for coding agent execution)
Target: MacBook with Apple Silicon (M4/M5 series) + MLX framework
Goal: A minimal, zero-dependency Zig HTTP server that exposes an OpenAI-compatible /v1/chat/completions endpoint for local coding models (e.g., Qwen2.5-Coder, Llama-3.2-Instruct, Phi-4, DeepSeek-Coder variants). It uses native MLX acceleration, KV cache, and TurboQuant KV-cache compression for maximum speed + context length on your Mac.  Why Zig + MLX?  Zig provides clean C/C++ interop and a single-command build (zig build).  
MLX.zig (existing binding) already ships a working LLM runtime for exactly the models you need — no Bazel, no CMake hell, no Python runtime in the final binary.  
Fully Mac-native (Metal GPU + unified memory).  
Easy for OpenCode / Continue.dev / any OpenAI client to point at http://localhost:8080/v1.

Non-goals (keeps it simple)  No distributed inference, no fine-tuning, no vision/multimodal.  
No full web framework — only a minimal HTTP + JSON handler.  
No complex dependency graph beyond one lightweight HTTP lib.

1. Core FeaturesOpenAI-compatible API (chat completions + streaming SSE).  
Model auto-download (from mlx-community HF repos — pre-converted for MLX).  
KV Cache (standard MLX transformer cache, enabled by default).  
TurboQuant KV Cache (Google’s near-lossless 3–4 bit KV compression with custom Metal kernels — 4–6× memory reduction, near-FP16 speed).  
Quantized weights (4-bit / 8-bit via MLX.zig model configs).  
Simple CLI: zig build run-server -- --model qwen2.5-coder-7b-4bit --port 8080 --turboquant.  
Zero-config for OpenCode: Just set base URL to http://127.0.0.1:8080/v1 and model name.

2. Architecture (high-level, coding-agent friendly)

mlx-zig-serve/
├── build.zig + build.zig.zon          (Zig package manager)
├── src/
│   ├── main.zig                       (server entry + CLI)
│   ├── server.zig                     (HTTP listener + OpenAI JSON/SSE)
│   ├── inference.zig                  (wraps MLX.zig llm + KV + TurboQuant)
│   ├── model_downloader.zig           (simple HF fetcher)
│   ├── turboquant_kv.zig              (optional Metal kernel binding)
│   └── mlx.zig/                       (git submodule or vendored fork of jaco-bro/MLX.zig)
├── models/                            (runtime dir for downloaded/converted weights)
└── zig-out/bin/mlx-zig-serve          (single static binary)

All inference logic reuses MLX.zig’s existing llm implementation (Llama/Phi/Qwen configs, tokenizer, generation loop).3. Dependencies (minimal & explicit)Zig 0.13.0+ (required).  
MLX.zig (https://github.com/jaco-bro/MLX.zig) — already has full LLM support for your coding models.  
HTTP library: Use httpz (https://github.com/karlseguin/http.zig) — single-file, zero-deps, perfect for this. Add via build.zig.zon.  
CMake (only for initial MLX build inside MLX.zig — already handled).  
No Python, no Python MLX runtime, no Bazel.

4. API Specification (OpenAI compatible — copy-paste ready for OpenCode)POST /v1/chat/completions
Body (JSON):json

{
  "model": "qwen2.5-coder-7b-4bit",
  "messages": [{"role": "user", "content": "Write a Zig function to..."}],
  "max_tokens": 2048,
  "temperature": 0.7,
  "stream": true
}

Response: Standard OpenAI streaming SSE (data: {"choices": [...]}) or non-streaming.
Supported models: Any MLX.zig config (Qwen2.5-Coder, Llama-3.2, etc.).5. Performance RequirementsKV Cache: Always enabled (MLX default).  
TurboQuant: Optional flag. Uses the Metal kernels from https://github.com/arozanov/turboquant-mlx (or flovflo’s HF variants). 4–6× smaller KV cache, near-zero perplexity loss, ~0.98× FP16 decode speed on M-series.  
Expected: 80–120 tokens/s decode on M4/M5 for 7B–14B 4-bit coding models with 32k+ context.

6. Clear Instructions for Coding Agent (execute step-by-step)Step 1: Bootstrap the project (5 minutes)bash

mkdir mlx-zig-serve && cd mlx-zig-serve
zig init
# Replace contents of build.zig and build.zig.zon with the templates below
git submodule add https://github.com/jaco-bro/MLX.zig.git src/mlx.zig
git submodule update --init --recursive

Step 2: Update build.zig.zon (dependencies)zig

.{
    .name = "mlx-zig-serve",
    .version = "0.1.0",
    .dependencies = .{
        .httpz = .{ .url = "https://github.com/karlseguin/http.zig/archive/refs/heads/master.tar.gz", .hash = "..." }, // get latest hash with zig fetch
        .mlx = .{ .path = "src/mlx.zig" },
    },
}

Step 3: Update build.zig (copy-paste this exact block)zig

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{ .default_target = .{ .cpu_arch = .aarch64, .os_tag = .macos } });
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "mlx-zig-serve",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Link MLX.zig (it already pulls MLX + Metal)
    exe.root_module.addImport("mlx", b.dependency("mlx", .{}).module("mlx"));

    // HTTP server
    const httpz_dep = b.dependency("httpz", .{});
    exe.root_module.addImport("httpz", httpz_dep.module("httpz"));

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    const run_step = b.step("run-server", "Run the inference server");
    run_step.dependOn(&run_cmd.step);
}

Step 4: Implement src/main.zig + inference.zig (core)main.zig: Parse CLI flags (--model, --port, --turboquant, --max-kv-size), call server.start().
inference.zig: Thin wrapper around MLX.zig’s llm module.  Load model with chosen config (qwen, llama, etc.).  
Use existing generation loop with KV cache.  
For TurboQuant: Import the custom KV cache from turboquant_kv.zig (see Step 5).

Provide a template inference.zig that re-exports the exact same generate call used in MLX.zig’s llm binary.

Step 5: TurboQuant KV Cache (critical for your request)Clone https://github.com/arozanov/turboquant-mlx.  
Port the Metal kernels (quantize_key, dequantize, residual sketch) into MLX.zig’s C++ layer (or as custom ops via mlx::core::custom — already supported in MLX C++ API).  
In turboquant_kv.zig: Create a drop-in replacement for the standard KVCache that calls the TurboQuant kernels on every append / attention step.  
Add compile-time flag -Dturboquant that swaps the cache implementation.
(Reference: PR #1067 in mlx-lm and the fused Metal kernels in arozanov/turboquant-mlx — they are deliberately simple and already benchmarked on M4 Pro.)

Step 6: Model downloading (easy-to-use)
In model_downloader.zig:Use std.http.Client to fetch https://huggingface.co/mlx-community/Qwen2.5-Coder-7B-Instruct-4bit/resolve/main/... (weights, config, tokenizer.model).  
Store in ./models/{model-name}/.  
On first run, if folder missing → auto-download + verify SHA.  
Fallback: Print mlx_lm.convert command if user prefers manual conversion.

Step 7: Server implementation (src/server.zig)Use httpz.Server with one route: POST /v1/chat/completions.  
Parse JSON with std.json.  
Call inference with streaming flag → send SSE chunks (data: {json}\n\n).  
50 lines max — keep it dead simple.

Step 8: Build & Runbash

zig build
./zig-out/bin/mlx-zig-serve --model qwen2.5-coder-7b-4bit --turboquant --port 8080

Test:bash

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-coder-7b-4bit","messages":[{"role":"user","content":"Write Zig code for a simple HTTP server"}],"stream":true}'

Step 9: Integrate with OpenCode
In your OpenCode / Continue.dev / Cursor settings:json

{
  "models": {
    "default": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "model": "qwen2.5-coder-7b-4bit"
    }
  }
}

Next Steps for Coding AgentExecute Steps 1–3 → you should have a compilable skeleton.  
Implement inference.zig by copying the generation logic from MLX.zig’s llm example.  
Add TurboQuant kernel binding (the Metal shaders are <200 LOC).  
Add the downloader.  
Test with a small 3B/7B coding model.

This spec gives you a single static binary, full MLX performance, TurboQuant KV cache, and OpenAI compatibility — exactly what you asked for, with zero extra complexity.  Copy this spec into a SPEC.md and start the coding agent. Let me know when you want the exact code templates for server.zig, inference.zig, or the TurboQuant kernel wrapper!


