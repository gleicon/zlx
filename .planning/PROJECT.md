# zlx

## What This Is

`zlx` is a minimal Zig HTTP server that exposes an OpenAI-compatible `/v1/chat/completions` endpoint for local coding models on Apple Silicon Macs. It runs inference via MLX.zig (native Metal GPU acceleration) with optional TurboQuant KV-cache compression for extended context at high throughput. Built for personal use — point OpenCode/Continue.dev at `http://localhost:8080/v1` and work locally with no Python runtime.

## Core Value

A single `zig build` binary that lets OpenCode connect to local coding models without any Python or cloud dependency.

## Requirements

### Validated

**Milestone v1.0 (COMPLETE)**
- [x] **BUILD-01** — `zig build` produces working binary with MLX.zig and httpz integrated
- [x] **BUILD-02** — MLX-C library linked with Metal GPU backend
- [x] **BUILD-03** — httpz server configured with routing and CORS
- [x] **INFER-01** — Model loading from `./models/<name>/` with safetensors weights
- [x] **INFER-02** — Token generation via Metal GPU with KV cache
- [x] **INFER-03** — Token-by-token generation iterator
- [x] **INFER-04** — Thread-safe inference with mutex serialization
- [x] **HTTP-01** — POST /v1/chat/completions endpoint
- [x] **HTTP-02** — SSE streaming for token-by-token responses
- [x] **HTTP-03** — Non-streaming JSON responses
- [x] **HTTP-04** — OpenAI-compatible response format
- [x] **HTTP-05** — CORS support for browser clients
- [x] **HTTP-06** — GET /v1/models endpoint
- [x] **INT-01** — OpenCode compatibility (field types, arrays, stop tokens)
- [x] **INT-02** — Metrics logging (tokens/sec, TTFT, active generations)

### Current Milestone: v1.1 Production-Ready

**Goal:** Transform zlx into a production-ready local LLM server with advanced performance, multi-model support, and complete API compatibility.

**Priority 1 — API Completeness (Table Stakes)**
- [ ] **API-01** — Stop sequences (string matching, not just EOS token)
- [ ] **API-02** — Logprobs tracking (top-k logits during sampling)
- [ ] **API-03** — Complete sampling parameters: top_k, min_p, presence_penalty, frequency_penalty, repetition_penalty, logit_bias
- [ ] **API-04** — Temperature sampling refinements (handles edge cases)
- [ ] **API-05** — Seed parameter for reproducible generation

**Priority 2 — Performance & Scale (Differentiators)**
- [ ] **PERF-01** — TurboQuant KV-cache compression (4.6x memory reduction via Metal kernels)
- [ ] **PERF-02** — Prompt caching with KV persistence (sub-second TTFT for repeated contexts)
- [ ] **PERF-03** — Multi-model hot-swap (switch models without restart)
- [ ] **PERF-04** — Speculative decoding (1.5-2.8x speedup with draft models)
- [ ] **PERF-05** — Memory budget management (prevent OOM, graceful degradation)

**Priority 3 — Convenience & UX**
- [ ] **UX-01** — Model auto-download from HuggingFace Hub
- [ ] **UX-02** — Open WebUI integration support (CORS, external UI)
- [ ] **UX-03** — Configuration file support (JSON/YAML settings)
- [ ] **UX-04** — Background model loading (non-blocking download)
- [ ] **UX-05** — Model registry and discovery

**Priority 4 — Infrastructure**
- [ ] **INFRA-01** — Enhanced metrics (memory usage, cache hit rates, model switch times)
- [ ] **INFRA-02** — Health check endpoint improvements
- [ ] **INFRA-03** — Better error handling and recovery
- [ ] **INFRA-04** — Request timeout handling

### Active (Deferred from v1.0)

- [ ] TurboQuant KV cache — originally planned for v1.0, moved to v1.1 due to complexity

### Out of Scope

- Vision / multimodal — text completions only (still valid)
- Function calling / tool execution — out of scope for v1.1 (requires external integrations)
- Distributed inference — single-machine only (still valid)
- Fine-tuning — inference only (still valid)
- Custom Web UI — integrate Open WebUI instead (research finding)
- Continuous batching — not beneficial for single-user server (research finding)
- PagedAttention — vLLM optimization, not applicable to MLX unified memory (research finding)

## Context

- **Current state (v1.0)**: Single-model HTTP server operational with OpenCode compatibility
- **MLX.zig**: Provides working LLM runtime with Llama/Phi/Qwen configs, tokenizer, generation loop
- **Research findings (v1.1)**:
  - TurboQuant: Extract Metal kernel source from Python, compile via `xcrun metal`
  - Web UI: Integrate Open WebUI (46K stars), don't build custom
  - Architecture: 7 new components following llama.cpp/mlx-lm patterns
- **Target hardware**: MacBook Apple Silicon (M1/M2/M3/M4), 32k+ context on 7B-14B 4-bit models
- **Primary test models**: Qwen2.5-Coder-1.5B-4bit, Qwen2.5-Coder-7B-4bit

## Constraints

- **Tech stack**: Zig 0.15.2+, MLX.zig, httpz — no Python runtime in final binary
- **Platform**: macOS 14+ on Apple Silicon — Metal GPU required
- **TurboQuant**: Metal kernels must be extracted and compiled, not Python bindings
- **Distribution**: Personal/small-team use — no enterprise features
- **Build**: Single `zig build` command produces working binary
- **Memory**: Must handle 7B models in 8GB RAM, 14B in 16GB RAM

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Binary name: `zlx` | Matches repo name | ✅ v1.0 — Adopted |
| Manual model placement v1.0 | Removes HF auth complexity | ✅ v1.0 — Works well |
| httpz for HTTP | Zero deps, matches minimal goal | ✅ v1.0 — Working |
| MessageContent custom parser | Handles string + array content | ✅ v1.0 — Fixes OpenCode compatibility |
| JsonFloat for numeric fields | Accepts int or float (OpenCode sends `1` not `1.0`) | ✅ v1.0 — Fixes parse errors |
| TurboQuant deferred to v1.1 | Metal kernel extraction needs research phase | 🔄 v1.1 — In progress |
| Integrate Open WebUI | 46K stars, already OpenAI-compatible | 🔄 v1.1 — Planned |
| Skip custom Web UI | Saves 1000+ lines frontend code | 🔄 v1.1 — Planned |
| MLX `mlx_save`/`mlx_load` for caching | Native KV persistence, no external deps | 🔄 v1.1 — Planned |
| Speculative decoding with draft model | 1.5-2.8x speedup, compatible pairs exist | 🔄 v1.1 — Planned |

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
*Last updated: 2026-04-01 — Milestone v1.1 initialized*
