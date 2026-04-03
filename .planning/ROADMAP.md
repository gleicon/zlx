# Roadmap: zlx

## Overview

zlx goes from a broken skeleton to a working single-binary OpenAI-compatible inference server in three phases. Phase 1 unblocks the build — nothing compiles today. Phase 2 produces a working token step-iterator over Metal GPU. Phase 3 wires it to HTTP and proves OpenCode can use it.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): v1.0 — COMPLETE
- Integer phases (4, 5, 6, 7, 8, 9): v1.1 — COMPLETE
- Phase 11-13: v1.1.1 — DeepSeek MoE Support (IN PROGRESS)
- Phase 14: v1.1.2 — llama.cpp Backend Integration (PLANNED)

**v1.0 (COMPLETE):**
- [x] **Phase 1: Foundation & Build** - Resolve build blockers so `zig build` succeeds on macOS aarch64
- [x] **Phase 2: Inference Core** - Load a model and generate tokens one at a time via Metal GPU
- [x] **Phase 3: HTTP API & Integration** - Expose completions over HTTP and verify OpenCode works end-to-end

**v1.1 (COMPLETE):**
- [x] **Phase 4: API Improvements** - Full OpenAI API compatibility: stop sequences, logprobs, sampling parameters, error handling, timeouts
- [x] **Phase 5: Multi-Model Support** - Model registry, hot-swap, memory budget management
- [x] **Phase 6: Prompt Caching** - KV cache persistence with sub-second TTFT for repeated contexts (completed 2026-04-02)
- [x] **Phase 7: TurboQuant Integration** - Metal kernel KV compression for 4.6x memory reduction
- [x] **Phase 8: Speculative Decoding** - Draft model speculation for 1.5-2.8x throughput increase
- [x] **Phase 9: Model Management** - Auto-download, configuration, and Open WebUI integration (COMPLETE)

**v1.1.1 (COMPLETE):**
- [x] **Phase 11: DeepSeek MoE Infrastructure** - MLA attention, MoE routing, mlx-c v0.4.x upgrade (COMPLETED)
- [x] **Phase 12: MoE Models Production-Ready** - Infrastructure: safetensors loading, GPT-OSS architecture, memory constraints, testing (COMPLETED)
- [x] **Phase 13: DeepSeek & GPT-OSS Completion** - Quantized weight dequantization, model download, integration testing (COMPLETED)

**v1.1.2 (PLANNED):**
- [ ] **Phase 14: DeepSeek & GPT-OSS llama.cpp Integration** - Backend abstraction, llama.cpp build integration, unified multi-backend generation pipeline (PLANNED)
- [ ] **Phase 15: Native MLX GPT-OSS** - High-performance native MLX implementation with Metal kernels, Harmony format, and tools (PLANNED)

## Phase Details

### Phase 1: Foundation & Build
**Goal**: `zig build` produces a working binary with MLX.zig and httpz correctly integrated
**Depends on**: Nothing (first phase)
**Requirements**: BUILD-01, BUILD-02, BUILD-03, BUILD-04
**Success Criteria** (what must be TRUE):
  1. `zig build` completes without errors on macOS aarch64 with latest stable Zig
  2. `zig-out/bin/zlx` binary exists and exits cleanly when invoked
  3. No `b.dependency("mlx").module("mlx")` panic at build time — MLX.zig linked via direct `.a` path
  4. httpz pinned to a tagged release compatible with the installed Zig version, with a valid SHA256 hash in build.zig.zon
  5. All C interop flows through a single `src/c.zig` — no duplicate @cImport type errors
**Plans**: 1 (PLAN.md in .planning/phases/01-foundation-build/)

### Phase 2: Inference Core
**Goal**: A CLI harness loads a quantized model from `./models/` and streams tokens to stdout one at a time via Metal GPU
**Depends on**: Phase 1
**Requirements**: INFER-01, INFER-02, INFER-03, INFER-04, INFER-05
**Success Criteria** (what must be TRUE):
  1. Running `./zig-out/bin/zlx --model qwen2.5-coder-7b` loads model weights from `./models/qwen2.5-coder-7b/` without crash
  2. Token generation runs on Metal GPU — Activity Monitor shows GPU activity during generation
  3. `GenerationState.next()` yields one token per call — observable as incremental stdout output in the CLI harness
  4. Sending two back-to-back generation requests does not corrupt output — mutex serialization is effective
  5. `--model`, `--port`, and `--max-kv-size` flags are accepted and applied
**Plans**: 1 (PLAN.md in .planning/phases/02-inference-core/)

### Phase 3: HTTP API & Integration
**Goal**: OpenCode connects to `http://127.0.0.1:8080/v1` and receives correct streaming completions from a locally loaded model
**Depends on**: Phase 2
**Requirements**: HTTP-01, HTTP-02, HTTP-03, HTTP-04, HTTP-05, HTTP-06, INT-01, INT-02
**Success Criteria** (what must be TRUE):
  1. `curl -X POST http://localhost:8080/v1/chat/completions -d '{"model":"...","messages":[...]}'` returns valid OpenAI JSON with `id`, `object`, `choices[].message`, `finish_reason`, `usage`
  2. The same request with `"stream": true` delivers visible token-by-token SSE output in the terminal, terminated with `data: [DONE]`
  3. `GET /v1/models` returns a JSON list containing the loaded model name
  4. OpenCode configured with `baseUrl: "http://127.0.0.1:8080/v1"` receives completions and displays them correctly in the editor
  5. No garbled characters appear in streamed output for code containing non-ASCII content — UTF-8 boundaries are respected
**Plans**: 1 (PLAN.md in .planning/phases/03-http-api/)

---

### Phase 4: API Improvements
**Goal**: Full OpenAI API compatibility with complete sampling parameters, stop sequences, and logprobs
**Depends on**: Phase 3
**Requirements**: API-01, API-02, API-03, API-04, API-05, INFRA-03, INFRA-04
**Success Criteria** (what must be TRUE):
  1. User can specify 1-4 stop sequences per request; generation halts immediately when matched, with `finish_reason: "stop"`
  2. Request with `logprobs: true` returns top-5 log probabilities for each generated token
  3. All sampling parameters work: `top_k`, `min_p`, `presence_penalty`, `frequency_penalty`, `repetition_penalty`, `logit_bias`, `seed`
  4. Same seed with same prompt produces identical output; different seeds produce different outputs
  5. API returns proper error codes: 400 for bad JSON, 408 for timeouts, 500 with request ID for generation errors
  6. Request timeout (default 60s) cancels generation and returns clean error response
**Plans:** 5 plans (3 complete + 2 gap closure)

Plans:
- [x] 04-01-PLAN.md — Stop sequences, seed-based determinism, temperature=0 greedy
- [x] 04-02-PLAN.md — Logprobs tracking, sampling parameters (top_k, min_p, penalties, logit_bias)
- [x] 04-03-PLAN.md — Error handling with request IDs, request timeouts
- [ ] 04-04-PLAN.md — Gap closure: Wire sampling parameters into generation pipeline (API-03)
- [ ] 04-05-PLAN.md — Gap closure: Stop sequences + logprobs serialization (API-01, API-02)

### Phase 5: Multi-Model Support
**Goal**: Multiple models can be loaded, switched, and managed without server restart
**Depends on**: Phase 4
**Requirements**: PERF-03, PERF-05, INFRA-01, INFRA-02
**Success Criteria** (what must be TRUE):
  1. `GET /v1/models` returns all available models with metadata (size, status, memory required)
  2. Switching models completes in <2 seconds for 1.5B models, <5 seconds for 7B models
  3. Memory is properly released on switch — no OOM after 10+ model switches
  4. Server rejects model load if insufficient memory available (with helpful error message)
  5. `GET /v1/health` returns detailed status including model loaded, GPU available, memory OK
  6. Metrics endpoint shows memory breakdown: weights, KV cache, temporaries, peak usage
**Plans**: 3 plans (Wave 1: Registry, Wave 2: Manager/Hot-swap, Wave 3: Memory/Health/Metrics)

Plans:
- [x] 05-01-PLAN.md — Model registry and discovery (UX-05, INFRA-01)
- [x] 05-02-PLAN.md — Multi-model manager and hot-swap (PERF-03, PERF-05)
- [x] 05-03-PLAN.md — Memory management, health checks, and metrics (PERF-05, INFRA-01, INFRA-02)

### Phase 6: Prompt Caching
**Goal**: Repeated prompts achieve sub-second TTFT via persistent KV cache
**Depends on**: Phase 5
**Requirements**: PERF-02, INFRA-01
**Success Criteria** (what must be TRUE):
  1. First request with new prompt: normal TTFT (500-2000ms depending on length)
  2. Second identical request: <500ms TTFT (cache hit)
  3. Cache persists across server restarts when configured
  4. Cache key includes model hash — different models don't share cache entries
  5. LRU eviction maintains cache under size limit (default 10GB configurable)
  6. Metrics show cache hit rate, size, and evictions in real-time
**Plans**: 1 plan (Wave 1: Complete caching infrastructure)

Plans:
- [x] 06-01-PLAN.md — Prompt caching with KV persistence, LRU eviction, and metrics (PERF-02, INFRA-01)

### Phase 7: TurboQuant Integration
**Goal**: KV cache compression achieves 5-6x memory reduction via botirk38/turboquant Zig library
**Depends on**: Phase 6
**Requirements**: PERF-01, PERF-02
**Success Criteria** (what must be TRUE):
  1. `--turboquant` flag activates real KV cache compression (not stub)
  2. Memory reduction of 5-6x validated on Qwen 7B (verified via metrics)
  3. Speed degradation <5% compared to standard FP16 cache
  4. Layer-adaptive mode: first and last N layers remain FP16 for quality
  5. Compression ratio visible in metrics endpoint
  6. CLI flags `--turboquant-bits` and `--turboquant-adaptive` function correctly
**Plans**: 2 plans (1 complete, 1 planned)

Plans:
- [x] 07-01-PLAN.md — TurboQuant feasibility spike: Metal kernel analysis, stub framework with graceful fallback (decision: INTEGRATE)
- [ ] 07-02-PLAN.md — TurboQuant library integration: botirk38/turboquant dependency, MLX bridge, real compression engine

### Phase 8: Speculative Decoding
**Goal**: Draft model speculation achieves 1.5-2.8x throughput increase
**Depends on**: Phase 7
**Requirements**: PERF-04
**Success Criteria** (what must be TRUE):
  1. Speculative decoding achieves 1.5-2.8x speedup on compatible target/draft pairs
  2. User can specify `--draft-model <name>` or auto-select compatible draft
  3. Falls back to standard generation if draft model unavailable or incompatible
  4. Speculative depth configurable (default 4 tokens ahead)
  5. Metrics show draft acceptance rate and tokens accepted per step
  6. No quality degradation compared to standard generation
**Plans**: 3 plans (Wave 1: Config, Wave 2: Downloader, Wave 3: Background Load + WebUI)

Plans:
- [x] 09-01-PLAN.md — Configuration file system with priority chain (CLI > Env > Config > Defaults)
- [x] 09-02-PLAN.md — Model auto-download from HuggingFace with resume and progress
- [x] 09-03-PLAN.md — Background model loading endpoints and Open WebUI CORS integration

### Phase 9: Model Management
**Goal**: Convenient model discovery, download, and configuration
**Depends on**: Phase 8
**Requirements**: UX-01, UX-02, UX-03, UX-04, UX-05
**Success Criteria** (what must be TRUE):
  1. `zlx --model mlx-community/Qwen2.5-Coder-7B` downloads from HuggingFace automatically
  2. Interrupted downloads resume via HTTP Range requests
  3. Download progress visible; background download doesn't block inference
  4. Settings loaded from `~/.config/zlx/config.json` with CLI flags overriding
  5. Open WebUI connects successfully to zlx at `http://localhost:8081` with CORS enabled
  6. `GET /v1/models` shows all models: available, loading, loaded with memory requirements

### Phase 11: DeepSeek MoE Infrastructure
**Goal**: Build foundation for MoE models via mlx-c v0.4.x with MLA attention and MoE routing
**Depends on**: Phase 9
**Requirements**: New capability — DeepSeek MoE architecture components
**Success Criteria** (what must be TRUE):
  1. mlx-c v0.4.x integrated alongside v0.1.2 for Fast Custom Ops API
  2. MLA (Multi-head Latent Attention) compresses KV cache by 90%
  3. MoE routing layer with sparse expert activation (top-k selection)
  4. DeepSeek transformer architecture defined with MLA + MoE layers
  5. Chat template for DeepSeek format (User:/Assistant:)
**Plans**: 6 plans complete (11-01 through 11-05 + MASTER)

Plans:
- [x] 11-01-PLAN.md — mlx-c v0.4.x integration for Fast Custom Ops API
- [x] 11-02-PLAN.md — Multi-head Latent Attention (MLA) with 90% KV compression
- [x] 11-03-PLAN.md — Mixture of Experts (MoE) routing with sparse expert activation
- [x] 11-04-PLAN.md — DeepSeek transformer architecture (integration layer)
- [x] 11-05-PLAN.md — Chat template, registry, memory estimation, end-to-end testing

### Phase 12: MoE Models Production-Ready
**Goal**: Make MoE models (DeepSeek + GPT-OSS) work on small machines (8-16GB RAM) with TurboQuant
**Depends on**: Phase 11 (infrastructure complete)
**Requirements**: MOE-01, MOE-02, MOE-03, MOE-04, MOE-05
**Success Criteria** (what must be TRUE):
  1. DeepSeek-Coder-V2-Lite loads weights and generates tokens without crash
  2. GPT-OSS-20B loads weights and generates tokens without crash
  3. Both models work on 16GB MacBook with TurboQuant enabled (4.6x compression)
  4. Server auto-enables TurboQuant on <16GB systems and limits context if needed
  5. `./test_models.sh` passes for all three models (Qwen, DeepSeek, GPT-OSS)
  6. CI/CD runs automated tests on every PR with regression detection
**Plans**: 5 plans (12-01 through 12-05) + 3 gap closure fixes

Plans:
- [x] 12-01-PLAN.md — DeepSeek weight loading from safetensors files
- [x] 12-02-PLAN.md — GPT-OSS architecture (sliding window, Yarn RoPE, 32-expert MoE)
- [x] 12-03-PLAN.md — TurboQuant verification/fix (4.6x KV cache compression)
- [x] 12-04-PLAN.md — Small machine constraints (memory budgets, auto-TurboQuant)
- [x] 12-05-PLAN.md — Testing infrastructure (automated testing shell script, CI/CD)

### Phase 13: DeepSeek & GPT-OSS Completion
**Goal**: Complete MoE model support with working quantized weight dequantization (DeepSeek) and full weight loading (GPT-OSS)
**Depends on**: Phase 12 (infrastructure complete)
**Requirements**: MOE-06, MOE-07, MOE-08, MOE-09, MOE-10
**Success Criteria** (what must be TRUE):
  1. DeepSeek-Coder-V2-Lite dequantizes 4-bit weights correctly and generates tokens
  2. GPT-OSS-20B downloads weights (11GB) and loads without errors
  3. Both models pass `./test_models.sh` with all tests passing
  4. TurboQuant achieves 4.6x+ KV cache compression with both models
  5. No stub functions or TODOs remain in weight loading code
  6. CI/CD integration testing works for all three models
**Plans**: 3 plans (13-01 through 13-03)

Plans:
- [ ] 13-01-PLAN.md — DeepSeek quantized weight reconstruction (4-bit affine dequantization)
- [ ] 13-02-PLAN.md — GPT-OSS download and weight loading (MXFP4 support)
- [ ] 13-03-PLAN.md — Integration testing and verification (test_models.sh, CI/CD)

### Phase 14: DeepSeek & GPT-OSS llama.cpp Integration
**Goal**: Make DeepSeek-Coder-V2-Lite and GPT-OSS-20B fully functional using llama.cpp as the primary backend for MoE models
**Depends on**: Phase 13 (MLX-based implementation complete)
**Requirements**: MOE-11, MOE-12, MOE-13, MOE-14, MOE-15
**Success Criteria** (what must be TRUE):
  1. Unified Backend interface abstracts both MLX.zig and llama.cpp implementations
  2. llama.cpp builds and links as static library via CMake integration
  3. DeepSeek-Coder-V2-Lite loads and generates via llama.cpp using GGUF format
  4. GPT-OSS-20B downloads and runs via llama.cpp backend
  5. Backend selection is automatic: Qwen→MLX.zig, DeepSeek/GPT-OSS→llama.cpp
  6. TurboQuant works with both backends through unified KV cache interface
  7. `./test_models.sh --all-backends` passes for all 4 model configurations
  8. No regression in Qwen MLX.zig performance
**Plans**: 5 plans (14-01 through 14-05) + MASTER

Plans:
- [ ] 14-MASTER-PLAN.md — Overview and architecture (MOE-11 through MOE-15)
- [ ] 14-01-PLAN.md — Backend abstraction layer (MOE-11)
- [ ] 14-02-PLAN.md — llama.cpp build integration and C bindings (MOE-12)
- [ ] 14-03-PLAN.md — DeepSeek end-to-end via llama.cpp (MOE-13)
- [ ] 14-04-PLAN.md — GPT-OSS download and llama.cpp integration (MOE-14)
- [ ] 14-05-PLAN.md — Unified generation pipeline (MOE-15)

### Phase 15: Native MLX GPT-OSS
**Goal**: Implement high-performance GPT-OSS support using native MLX (like openharmony-mlx), replacing the llama.cpp approach with a native Zig/MLX implementation that achieves 40 tokens/sec on Apple Silicon
**Depends on**: Phase 11 (mlx-c v0.4.x), Phase 14 architecture decisions
**Requirements**: GPTOSS-01, GPTOSS-02, GPTOSS-03, GPTOSS-04, GPTOSS-05
**Success Criteria** (what must be TRUE):
  1. GPT-OSS-20B runs at 30+ tokens/sec via native MLX (not llama.cpp)
  2. Harmony chat format is correctly parsed and formatted
  3. Browser and Python tools work end-to-end
  4. MXFP4 weights load and decompress correctly
  5. `./test_models.sh gptoss` passes
  6. OpenCode integration works with native MLX backend
**Plans**: 5 plans (15-01 through 15-05) + MASTER

Plans:
- [ ] 15-MASTER-PLAN.md — Overview and architecture (GPTOSS-01 through GPTOSS-05)
- [x] 15-01-PLAN.md — MLX GPT-OSS transformer with Metal kernels (GPTOSS-01)
- [x] 15-02-PLAN.md — Harmony format parser and chat template (GPTOSS-02)
- [x] 15-03-PLAN.md — Browser and Python tool implementation (GPTOSS-03)
- [ ] 15-04-PLAN.md — Weight loading and MXFP4 support (GPTOSS-04)
- [ ] 15-05-PLAN.md — Integration with zlx server (GPTOSS-05)

## Progress

**Execution Order:**
v1.0: 1 → 2 → 3 (COMPLETE)
v1.1: 4 → 5 → 6 → 7 → 8 → 9 (IN PROGRESS - Phase 4 Complete)

**Phase 4 Status: ✅ COMPLETE** — All 5 plans complete:
- 04-01: Stop sequences, seed-based determinism, temperature=0 greedy ✅
- 04-02: Logprobs tracking, complete sampling parameters ✅
- 04-03: Error handling with request IDs, configurable timeouts ✅
- 04-04: Wire sampling parameters into generation pipeline ✅
- 04-05: Gap closure - tokenizer, stop sequences, logprobs ✅

**Phase 5 Status: ✅ COMPLETE** — All 3 plans complete, ready for Phase 6

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation & Build | 1/1 | ✅ Complete | 2026-03-31 |
| 2. Inference Core | 1/1 | ✅ Complete | 2026-03-31 |
| 3. HTTP API & Integration | 1/1 | ✅ Complete | 2026-03-31 |
| 4. API Improvements | 5/5 | ✅ Complete | 2026-04-01 |
| 5. Multi-Model Support | 3/3 | ✅ Complete | 2026-04-02 |
| 6. Prompt Caching | 1/1 | ✅ Complete | 2026-04-02 |
| 7. TurboQuant Integration | 2/2 | ✅ Complete | 2026-04-02 |
| 8. Speculative Decoding | 1/1 | ✅ Complete | 2026-04-02 |
| 9. Model Management | 3/3 | ✅ Complete | 2026-04-02 |
| 11. DeepSeek MoE Infrastructure | 6/6 | ✅ Complete | 2026-04-03 |
| 12. MoE Production-Ready | 8/8 | ✅ Complete | 2026-04-03 |
| 13. DeepSeek & GPT-OSS Completion | 0/3 | 🔄 Planned | In Progress |
| 14. llama.cpp Backend Integration | 0/5 | 📋 Planned | Not Started |
| 15. Native MLX GPT-OSS | 3/6 | In Progress|  |

**v1.1.0 Progress:** 9/9 phases complete | All phases COMPLETE | v1.1.0 Released

**v1.1.1 Target:** MoE Models Production-Ready | Phase 13 Planning Complete | DeepSeek + GPT-OSS Weight Loading

**v1.1.2 Target:** llama.cpp Backend Support | Phase 14 Planning Complete | Multi-Backend Architecture
