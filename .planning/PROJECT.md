# zlx

## What This Is

`zlx` is a minimal Zig HTTP server that exposes an OpenAI-compatible `/v1/chat/completions` endpoint for local coding models on Apple Silicon Macs. It runs inference via MLX.zig (native Metal GPU acceleration) with TurboQuant KV-cache compression for extended context at high throughput. Built for personal use — point OpenCode/Continue.dev at `http://localhost:8080/v1` and work locally with no Python runtime.

## Core Value

A single `zig build` binary that lets OpenCode connect to local coding models without any Python or cloud dependency.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] POST /v1/chat/completions — streaming SSE and non-streaming JSON responses
- [ ] MLX.zig integration — load and run quantized models (Qwen2.5-Coder, Llama-3.2, Phi-4) via Metal GPU
- [ ] TurboQuant KV cache — bind arozanov/turboquant-mlx C++ Metal kernels via Zig C interop; activated with `--turboquant` flag
- [ ] CLI interface — `--model <name>`, `--port <n>`, `--turboquant`, `--max-kv-size <n>`
- [ ] Manual model loading — read weights from `./models/<model-name>/`; no auto-download required
- [ ] httpz HTTP server — single POST route, JSON parsing via `std.json`, SSE chunk streaming
- [ ] OpenCode integration — works when `baseUrl` is set to `http://127.0.0.1:8080/v1` in OpenCode settings

### Out of Scope

- Model auto-download from HuggingFace — adds complexity, manual model placement is sufficient for personal use
- Distributed inference — single-machine only
- Fine-tuning — inference only
- Vision / multimodal — text completions only
- Full web framework — single route, minimal HTTP layer

## Context

- **Existing skeleton**: `build.zig` and `src/main.zig` are bootstrapped; MLX.zig is added as a git submodule at `src/mlx.zig`
- **MLX.zig** (github.com/jaco-bro/MLX.zig) already provides a working LLM runtime with Llama/Phi/Qwen configs, tokenizer, and generation loop — inference.zig wraps this
- **TurboQuant** (github.com/arozanov/turboquant-mlx) provides `quantize_key`, `dequantize`, and `residual sketch` Metal kernels (<200 LOC) — bound via Zig `extern` / `@cImport`; 4–6× smaller KV cache at ~0.98× decode speed
- **httpz** (github.com/karlseguin/http.zig) is the HTTP layer — single-file, zero-deps, added via build.zig.zon
- **Target hardware**: MacBook with Apple Silicon (M4/M5), 32k+ context on 7B–14B 4-bit models
- **Primary test model**: Qwen2.5-Coder-7B-Instruct-4bit (mlx-community HF pre-converted format)

## Constraints

- **Tech stack**: Latest stable Zig, MLX.zig, httpz — no Python, no Bazel, no CMake in final binary
- **Platform**: macOS aarch64 only — Metal GPU required, no cross-platform target
- **Dependencies**: TurboQuant C++ kernels must be bound (not rewritten) — scope limited to interop, not porting
- **Distribution**: Personal use only — no installer, no multi-user auth, no TLS required
- **Build**: Single `zig build` command must produce a working binary

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Binary name: `zlx` | Matches repo name and existing build.zig | — Pending |
| TurboQuant in MVP | User requires it from day one; context length matters for coding use cases | — Pending |
| Bind C++ kernels (not port) | Saves weeks; arozanov's kernels are already benchmarked on M4 | — Pending |
| Manual model placement | Removes HF auth/network complexity; models are large and slow to re-download | — Pending |
| httpz for HTTP | Single-file, zero-dep; matches PRD's "50 lines max" server goal | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-03-30 after initialization*
