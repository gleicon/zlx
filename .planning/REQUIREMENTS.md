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

---

## v2.0 Requirements — "It Just Works"

**Status:** Active
**Goal:** Close every stub and mock, ship verified inference for all claimed models, add Gemma 4 E4B, implement TurboQuant Metal kernels, wire tool APIs end-to-end, and produce a project history document.

### Gap Closure & Honest Inventory

- [ ] **GAP-01**: Every `error.NotImplemented` return in claimed production paths is either fully implemented or replaced by explicit removal with documentation — no silent stubs in the inference pipeline
- [ ] **GAP-02**: All hardcoded mock values (vocab_size=32000, EOS=2, BOS=1, disk space=100GB) replaced with values read from actual model config or system APIs
- [ ] **GAP-03**: GPT-OSS native MLX forward pass implemented with real tensor computation (layer embeddings, attention, FFN) — not returning `mlx.zeros()`
- [ ] **GAP-04**: GPT-OSS tokenizer wired to real tokenizer (tiktoken or Qwen-compatible) — `tokenize()` does not return empty slice
- [ ] **GAP-05**: MLX backend factory (`createMlxBackend`) creates a real backend object, not a stub `u8` pointer
- [ ] **GAP-06**: Prompt cache `parseIndex()` reads actual cache index entries from disk — not returning empty
- [ ] **GAP-07**: Speculative decoding wired to main generation path (or explicitly removed with clear documentation if deferred)
- [ ] **GAP-08**: Build system compiles without deprecation errors on Zig 0.15.2 (`b.pathJoin` → correct API, all `@compileError` and deprecated paths resolved)
- [ ] **GAP-09**: `mlx.arrayIsEmpty()` calls removed or replaced with a valid MLX.zig API call — no undefined symbol at link time
- [ ] **GAP-10**: `MultiHeadLatentAttention` struct field mismatch in `loader.zig` resolved — struct definition and initialization agree

### Model Verification & Gemma 4 E4B

- [ ] **MODEL-01**: User can run Qwen2.5-Coder end-to-end with a verified passing integration test (`zig build test`)
- [ ] **MODEL-02**: User can run DeepSeek-Coder-V2-Lite end-to-end via llama.cpp backend with a verified passing integration test
- [ ] **MODEL-03**: User can run GPT-OSS-20B end-to-end via native MLX — generates real output tokens (not hardcoded token 1)
- [ ] **MODEL-04**: Gemma 4 E4B architecture implemented in Zig/MLX — 42 layers, hybrid sliding-window (512-token) + global attention with Proportional RoPE, 128K context, 262K-token vocabulary, Per-Layer Embeddings loading
- [ ] **MODEL-05**: Gemma 4 E4B chat template implemented per Google spec — `<|turn>` / `<turn|>` control tokens, system/user/model roles, no-think variant supported
- [ ] **MODEL-06**: Gemma 4 E4B verbosity-reduction config available — `temperature=0.3`, `<|think|>` block suppression in system prompt achieves ~83% token reduction vs default
- [ ] **MODEL-07**: User can load Gemma 4 E4B 4-bit (MLX quantized format, ~5GB VRAM) and receive completions via `/v1/chat/completions`

### TurboQuant Metal Compression

- [ ] **TURBO-01**: Walsh-Hadamard Transform Metal kernel implemented in Zig/Metal C — logic ported from Python source, no Python runtime dependency
- [ ] **TURBO-02**: KV cache compression and decompression produces correct output — Lloyd-Max codebook quantization verified against reference values
- [ ] **TURBO-03**: TurboQuant active on Qwen with measured memory reduction ≥4x vs uncompressed KV cache
- [ ] **TURBO-04**: `--turboquant` flag enables compression; memory savings visible in logs; automatic fallback if compression fails

### Tool API

- [ ] **TOOLS-01**: `browser.zig` and `python.zig` compile cleanly against Zig 0.15.2 — `parseFree` and other removed APIs replaced with `Parsed(T).deinit()` and equivalent
- [ ] **TOOLS-02**: Tool routes registered in main HTTP server — browser search and Python execution are reachable HTTP endpoints
- [ ] **TOOLS-03**: User can POST to browser tool and receive real search results from at least one provider (DuckDuckGo)
- [ ] **TOOLS-04**: User can POST to Python tool and have code executed in a subprocess with captured stdout/stderr output

### Project Documentation

- [ ] **DOCS-01**: Human-readable project history document — milestone timeline, model inventory (what works, what was stubbed, what was deferred), key design decisions and why
- [ ] **DOCS-02**: Zig ecosystem extraction candidates documented — MLX C bindings layer, safetensors parser, BPE tokenizer, httpz SSE helper — each with scope estimate and contribution path
- [ ] **DOCS-03**: Working model setup guide — download commands, directory layout, test invocations for each supported model (Qwen, DeepSeek, GPT-OSS, Gemma 4 E4B)

---

## Out of Scope (v2.0)

| Feature | Reason |
|---------|--------|
| Vision / multimodal | Gemma 4 E4B has vision encoder but text-only scope for v2.0; audio similarly deferred |
| Custom Web UI | Use Open WebUI instead |
| Continuous batching | Not beneficial for single-user server |
| PagedAttention | vLLM optimization, not applicable to MLX unified memory |
| Auth / API keys | Personal use only |
| TLS / HTTPS | Localhost only |
| Embeddings endpoint | Out of scope for chat completions focus |
| Fine-tuning | Inference only |
| Distributed inference | Single-machine Metal GPU only |

## Out of Scope (v1.1, historical)

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
| PERF-02 | Phase 6 | ✅ Complete |
| INFRA-01 | Phase 6 | ✅ Complete |
| PERF-01 | Phase 7 | ✅ Complete |
| PERF-04 | Phase 8 | ✅ Complete |
| UX-01 | Phase 9 | ✅ Complete |
| UX-02 | Phase 9 | ✅ Complete |
| UX-03 | Phase 9 | ✅ Complete |
| UX-04 | Phase 9 | ✅ Complete |
| UX-05 | Phase 5, 9 | ✅ Complete |

**Coverage:**
- v1.1 requirements: 19 total
- Mapped to phases: 19 ✓
- Unmapped: 0

### v2.0 Traceability (filled by roadmapper)

| Requirement | Phase | Status |
|-------------|-------|--------|
| GAP-01 through GAP-10 | TBD | Not started |
| MODEL-01 through MODEL-07 | TBD | Not started |
| TURBO-01 through TURBO-04 | TBD | Not started |
| TOOLS-01 through TOOLS-04 | TBD | Not started |
| DOCS-01 through DOCS-03 | TBD | Not started |

---
*Requirements defined: 2026-03-30 (v1.0)*
*Updated: 2026-04-01 (v1.1 requirements added)*
*Updated: 2026-04-05 (v2.0 requirements added — 24 requirements across 5 categories)*
