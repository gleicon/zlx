# Requirements: zlx

**Defined:** 2026-03-30
**Core Value:** A single `zig build` binary that lets OpenCode connect to local coding models without any Python or cloud dependency.

## v1.0 Requirements (COMPLETE)

### Build

- [x] **BUILD-01**: `zig build` completes without errors on macOS aarch64 with latest stable Zig
- [x] **BUILD-02**: MLX.zig linkage is correctly integrated (no module-name panic; uses `configureExecutable` C-interop path)
- [x] **BUILD-03**: httpz pinned to a tagged release compatible with installed Zig version, with correct hash in `build.zig.zon`
- [x] **BUILD-04**: Single `@cImport` boundary (`src/c.zig`) for all C interop (mlx-c headers)

### Inference

- [x] **INFER-01**: Server loads a quantized model from `./models/<model-name>/` on startup
- [x] **INFER-02**: Inference runs on Metal GPU via MLX.zig (not CPU fallback)
- [x] **INFER-03**: `GenerationState.next()` step-iterator yields one token per call (required for SSE streaming)
- [x] **INFER-04**: Inference calls are mutex-serialized (no concurrent GPU access)
- [x] **INFER-05**: CLI flags `--model <name>`, `--port <n>`, `--max-kv-size <n>` are parsed and applied

### HTTP API

- [x] **HTTP-01**: `POST /v1/chat/completions` returns non-streaming JSON response with correct OpenAI wire format (`id`, `object`, `created`, `model`, `choices[].message`, `choices[].finish_reason`, `usage`)
- [x] **HTTP-02**: `POST /v1/chat/completions` with `"stream": true` delivers streaming SSE chunks (`data: {...}\n\n`) with `role: "assistant"` on first delta
- [x] **HTTP-03**: Streaming response terminates with `data: [DONE]\n\n`
- [x] **HTTP-04**: `GET /v1/models` returns a list of available models (scanned from `./models/`)
- [x] **HTTP-05**: Stop sequences in request body halt generation when matched (handled via EOS + custom content parsing)
- [x] **HTTP-06**: SSE chunks are flushed only at valid UTF-8 boundaries (no split multi-byte characters)

### Integration

- [x] **INT-01**: OpenCode can be configured with `baseUrl: "http://127.0.0.1:8081/v1"` and receive completions from a loaded local model
- [x] **INT-02**: curl smoke test against `/v1/chat/completions` with `"stream": true` produces visible token output

### v1.0 Deferred to v1.1

- [ ] **HTTP-05-ext**: Full stop sequence string matching (not just EOS token)
- [ ] **PARAM-01**: `temperature`, `top_p`, `presence_penalty`, `frequency_penalty` forwarded to MLX inference (temperature/top_p done, penalties pending)

---

## v1.1 Requirements — Production-Ready Inference Server

**Status:** Active (defining)  
**Goal:** Transform zlx into a production-ready local LLM server with advanced performance, multi-model support, and complete API compatibility.

### Priority 1 — API Completeness (Table Stakes)

These requirements ensure full OpenAI API compatibility for standard use cases.

#### API-01: Stop Sequences
- [ ] User can specify 1-4 stop sequences per request
- [ ] Generation stops immediately when any stop sequence is encountered
- [ ] Stop sequence is not included in output
- [ ] `finish_reason` is set to "stop" (not "length")
- [ ] Works with both streaming and non-streaming responses
- [ ] Stop sequences can be multi-character strings

#### API-02: Logprobs Tracking
- [ ] When `logprobs: true`, response includes logprobs for each token
- [ ] Top-k alternative tokens included (k=5 default, configurable)
- [ ] Logprobs are natural log probabilities (negative numbers)
- [ ] Works with streaming (token-level) and non-streaming (full sequence)
- [ ] Can specify `top_logprobs: n` to get alternatives

#### API-03: Complete Sampling Parameters
- [ ] `top_k`: Only sample from top k tokens (integer, default 0 = disabled)
- [ ] `min_p`: Minimum probability for nucleus sampling (0.0-1.0, default 0.0)
- [ ] `presence_penalty`: Penalize tokens already in text (-2.0 to 2.0)
- [ ] `frequency_penalty`: Penalize by frequency in text (-2.0 to 2.0)
- [ ] `repetition_penalty`: Penalize repeated sequences (1.0-2.0, default 1.0)
- [ ] `logit_bias`: Map token IDs to bias values (-100 to 100)

#### API-04: Temperature Sampling Refinements
- [ ] `temperature: 0` selects highest probability token (greedy)
- [ ] Very low temperatures (0.0-0.1) work correctly without numerical issues
- [ ] High temperatures (>1.0) produce expected randomness
- [ ] Handles edge case: temperature + min_p conflicts gracefully

#### API-05: Seed Parameter
- [ ] `seed: integer` produces deterministic output for same input
- [ ] Same seed with same parameters produces identical output
- [ ] Different seeds produce different outputs
- [ ] Seed applies to both token sampling and dropout (if applicable)

---

### Priority 2 — Performance & Scale (Differentiators)

These requirements provide competitive advantages through performance optimization.

#### PERF-01: TurboQuant KV-Cache Compression
- [ ] `--turboquant` flag activates KV cache compression
- [ ] 4.6x memory reduction achieved (validated on Qwen 7B)
- [ ] Speed degradation <5% at 4-bit compression
- [ ] Works with all supported model architectures
- [ ] Automatic fallback to standard cache if TurboQuant fails
- [ ] Memory usage visible in logs/metrics
- **Note:** Requires Metal kernel extraction from arozanov/turboquant-mlx Python source

#### PERF-02: Prompt Caching
- [ ] First request: normal speed, cache saved
- [ ] Second identical request: <500ms TTFT (time to first token)
- [ ] Cache persists across server restarts (optional)
- [ ] Cache key includes: model hash + prompt hash + sampling params
- [ ] LRU eviction when cache size exceeds limit (default 10GB)
- [ ] Cache hit rate visible in metrics

#### PERF-03: Multi-Model Hot-Swap
- [ ] `POST /v1/models/switch` endpoint or automatic by model name
- [ ] Switch time <2 seconds for 1.5B model, <5s for 7B
- [ ] Memory properly released (no OOM after multiple switches)
- [ ] Active generation completes before switch
- [ ] Graceful degradation if target model fails to load
- [ ] Previous model remains available if switch fails

#### PERF-04: Speculative Decoding
- [ ] 1.5-2.8x throughput increase (validated on Qwen 7B)
- [ ] Automatic draft model selection for supported target/draft pairs
- [ ] User can specify `--draft-model <name>`
- [ ] Falls back to standard generation if draft unavailable
- [ ] Speculative depth configurable (default 4 tokens ahead)
- [ ] Verification rate visible in metrics

#### PERF-05: Memory Budget Management
- [ ] Detect available system memory on startup
- [ ] Reject model load if insufficient memory (with helpful message)
- [ ] Gracefully handle OOM by clearing KV cache
- [ ] Memory usage visible in real-time metrics
- [ ] Warning when memory >80% capacity
- [ ] Automatic context truncation when approaching limit

---

### Priority 3 — Convenience & UX

These requirements improve the user experience for setup and daily use.

#### UX-01: Model Auto-Download
- [ ] `zlx --model mlx-community/Qwen2.5-Coder-7B` downloads automatically
- [ ] Resume interrupted downloads (HTTP Range requests)
- [ ] Progress indicator during download
- [ ] Background download (doesn't block inference of other models)
- [ ] Verify checksums after download
- [ ] Cache downloaded files in `~/.cache/zlx/models/`

#### UX-02: Open WebUI Integration
- [ ] zlx CORS headers allow Open WebUI connection
- [ ] Documented setup: point Open WebUI at `http://localhost:8081`
- [ ] All Open WebUI features work (chat, system prompts, parameters)
- [ ] No custom UI code in zlx repository

#### UX-03: Configuration File
- [ ] Load settings from `~/.config/zlx/config.json` or `./zlx.json`
- [ ] All CLI flags available as config options
- [ ] Environment variables override config file
- [ ] Command-line flags override environment
- [ ] Validate config on startup with helpful errors

#### UX-04: Background Model Loading
- [ ] `POST /v1/models/load` starts background load
- [ ] Current model remains usable during load
- [ ] Progress endpoint: `GET /v1/models/load-status`
- [ ] Auto-switch to new model when ready (optional)
- [ ] Cancel ongoing load if requested

#### UX-05: Model Registry
- [ ] `GET /v1/models` returns all available models (not just loaded)
- [ ] Shows: name, size, status (available/loading/loaded)
- [ ] Shows: memory required, approximate speed
- [ ] Scan `~/.cache/zlx/models/` and `./models/` directories

---

### Priority 4 — Infrastructure

These requirements improve reliability, monitoring, and operations.

#### INFRA-01: Enhanced Metrics
- [ ] Memory usage: current, peak, by component (weights, KV cache, temporaries)
- [ ] Cache metrics: hit rate, size, evictions
- [ ] Model metrics: load time, switch time, active time
- [ ] GPU utilization percentage
- [ ] Request latency percentiles (p50, p95, p99)
- [ ] Export to Prometheus format (optional)

#### INFRA-02: Health Check Improvements
- [ ] `GET /v1/health` returns detailed status
- [ ] Checks: model loaded, GPU available, memory OK
- [ ] Returns 503 if any check fails with reason
- [ ] Response includes: status, version, uptime, model name

#### INFRA-03: Error Handling & Recovery
- [ ] JSON parse errors: return 400 with details
- [ ] Model load failures: return 503 with helpful message
- [ ] Generation errors: return 500 with request ID for debugging
- [ ] OOM errors: clear cache and retry once
- [ ] All errors logged with context

#### INFRA-04: Request Timeout Handling
- [ ] Configurable timeout (default 60s)
- [ ] Return 408 Request Timeout if exceeded
- [ ] Cancel ongoing generation
- [ ] Clean up resources on timeout

---

## Out of Scope (v1.1)

| Feature | Reason |
|---------|--------|
| Vision / multimodal | Text completions only (still valid from v1.0) |
| Function calling / tool execution | Requires external integrations, defer to v2.0 |
| Custom Web UI | Integrate Open WebUI instead (research finding) |
| Continuous batching | Not beneficial for single-user server (research finding) |
| PagedAttention | vLLM optimization, not applicable to MLX unified memory |
| Auth / API keys | Personal use only; single-user, local network |
| Rate limiting | Inference is already mutex-serialized |
| TLS / HTTPS | OpenCode connects to localhost only |
| Embeddings endpoint | Out of scope for chat completions focus |
| Fine-tuning | Inference only |
| Distributed inference | Single-machine Metal GPU only |

---

## Traceability

### v1.0 Complete

| Requirement | Phase | Status |
|-------------|-------|--------|
| BUILD-01 | Phase 1 | ✅ Complete |
| BUILD-02 | Phase 1 | ✅ Complete |
| BUILD-03 | Phase 1 | ✅ Complete |
| BUILD-04 | Phase 1 | ✅ Complete |
| INFER-01 | Phase 2 | ✅ Complete |
| INFER-02 | Phase 2 | ✅ Complete |
| INFER-03 | Phase 2 | ✅ Complete |
| INFER-04 | Phase 2 | ✅ Complete |
| INFER-05 | Phase 2 | ✅ Complete |
| HTTP-01 | Phase 3 | ✅ Complete |
| HTTP-02 | Phase 3 | ✅ Complete |
| HTTP-03 | Phase 3 | ✅ Complete |
| HTTP-04 | Phase 3 | ✅ Complete |
| HTTP-05 | Phase 3 | ✅ Complete |
| HTTP-06 | Phase 3 | ✅ Complete |
| INT-01 | Phase 3 | ✅ Complete |
| INT-02 | Phase 3 | ✅ Complete |

### v1.1 Active

| Requirement | Phase | Status |
|-------------|-------|--------|
| API-01 | Phase 4 | Not started |
| API-02 | Phase 4 | Not started |
| API-03 | Phase 4 | Not started |
| API-04 | Phase 4 | Not started |
| API-05 | Phase 4 | Not started |
| INFRA-03 | Phase 4 | Not started |
| INFRA-04 | Phase 4 | Not started |
| PERF-03 | Phase 5 | Not started |
| PERF-05 | Phase 5 | Not started |
| INFRA-01 | Phase 5 | Not started |
| INFRA-02 | Phase 5 | Not started |
| PERF-02 | Phase 6 | Not started |
| INFRA-01 | Phase 6 | Not started |
| PERF-01 | Phase 7 | Not started |
| PERF-04 | Phase 8 | Not started |
| UX-01 | Phase 9 | Not started |
| UX-02 | Phase 9 | Not started |
| UX-03 | Phase 9 | Not started |
| UX-04 | Phase 9 | Not started |
| UX-05 | Phase 5, 9 | Not started |

**Coverage:**
- v1.1 requirements: 19 total
- Mapped to phases: 19 ✓
- Unmapped: 0

---
*Requirements defined: 2026-03-30 (v1.0)*  
*Updated: 2026-04-01 (v1.1 requirements added)*
