# Roadmap: zlx

## Overview

zlx goes from a broken skeleton to a working single-binary OpenAI-compatible inference server in three phases. Phase 1 unblocks the build — nothing compiles today. Phase 2 produces a working token step-iterator over Metal GPU. Phase 3 wires it to HTTP and proves OpenCode can use it.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): v1.0 — COMPLETE
- Integer phases (4, 5, 6, 7, 8, 9): v1.1 — Production-Ready Inference Server
- Decimal phases (X.1, X.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

**v1.0 (COMPLETE):**
- [x] **Phase 1: Foundation & Build** - Resolve build blockers so `zig build` succeeds on macOS aarch64
- [x] **Phase 2: Inference Core** - Load a model and generate tokens one at a time via Metal GPU
- [x] **Phase 3: HTTP API & Integration** - Expose completions over HTTP and verify OpenCode works end-to-end

**v1.1 (Production-Ready):**
- [x] **Phase 4: API Improvements** - Full OpenAI API compatibility: stop sequences, logprobs, sampling parameters, error handling, timeouts
- [ ] **Phase 5: Multi-Model Support** - Model registry, hot-swap, memory budget management
- [x] **Phase 6: Prompt Caching** - KV cache persistence with sub-second TTFT for repeated contexts (completed 2026-04-02)
- [ ] **Phase 7: TurboQuant Integration** - Metal kernel KV compression for 4.6x memory reduction
- [ ] **Phase 8: Speculative Decoding** - Draft model speculation for 1.5-2.8x throughput increase
- [ ] **Phase 9: Model Management** - Auto-download, configuration, and Open WebUI integration

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
**Plans**: TBD

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
| 7. TurboQuant Integration | 1/2 | 📝 Planned | — |
| 8. Speculative Decoding | 0/1 | Not started | — |
| 9. Model Management | 0/1 | Not started | — |

**v1.1 Progress:** 5/6 phases | Phase 7: 2 plans ready (07-01 complete, 07-02 ready to execute)
