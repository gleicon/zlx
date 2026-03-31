# Requirements: zlx

**Defined:** 2026-03-30
**Core Value:** A single `zig build` binary that lets OpenCode connect to local coding models without any Python or cloud dependency.

## v1 Requirements

### Build

- [ ] **BUILD-01**: `zig build` completes without errors on macOS aarch64 with Zig 0.13.0
- [ ] **BUILD-02**: MLX.zig linkage is correctly integrated (no module-name panic; uses `configureExecutable` C-interop path)
- [ ] **BUILD-03**: httpz pinned to zig-0.13 branch with correct hash in `build.zig.zon`
- [ ] **BUILD-04**: Single `@cImport` boundary (`src/c.zig`) for all C interop (mlx-c headers)

### Inference

- [ ] **INFER-01**: Server loads a quantized model from `./models/<model-name>/` on startup
- [ ] **INFER-02**: Inference runs on Metal GPU via MLX.zig (not CPU fallback)
- [ ] **INFER-03**: `GenerationState.next()` step-iterator yields one token per call (required for SSE streaming)
- [ ] **INFER-04**: Inference calls are mutex-serialized (no concurrent GPU access)
- [ ] **INFER-05**: CLI flags `--model <name>`, `--port <n>`, `--max-kv-size <n>` are parsed and applied

### HTTP API

- [ ] **HTTP-01**: `POST /v1/chat/completions` returns non-streaming JSON response with correct OpenAI wire format (`id`, `object`, `created`, `model`, `choices[].message`, `choices[].finish_reason`, `usage`)
- [ ] **HTTP-02**: `POST /v1/chat/completions` with `"stream": true` delivers streaming SSE chunks (`data: {...}\n\n`) with `role: "assistant"` on first delta
- [ ] **HTTP-03**: Streaming response terminates with `data: [DONE]\n\n`
- [ ] **HTTP-04**: `GET /v1/models` returns a list of available models (scanned from `./models/`)
- [ ] **HTTP-05**: `stop` sequences in request body halt generation when matched
- [ ] **HTTP-06**: SSE chunks are flushed only at valid UTF-8 boundaries (no split multi-byte characters)

### Integration

- [ ] **INT-01**: OpenCode can be configured with `baseUrl: "http://127.0.0.1:8080/v1"` and receive completions from a loaded local model
- [ ] **INT-02**: curl smoke test against `/v1/chat/completions` with `"stream": true` produces visible token output

## v2 Requirements

### TurboQuant KV Cache

- **TURBO-01**: Research: identify whether TurboQuant Metal kernel MSL source can be extracted from Python and compiled standalone via MLX-C custom kernel API
- **TURBO-02**: Implement `turboquant_kv.zig` as a drop-in replacement for `mlx.KVCache` activated by `--turboquant` CLI flag
- **TURBO-03**: Validate 4–6× KV cache size reduction with near-parity decode speed vs standard KV cache

### Tool Calling

- **TOOL-01**: `POST /v1/chat/completions` with `tools` array returns tool call deltas in streaming response
- **TOOL-02**: OpenCode agentic mode (file read/write/execute tools) works end-to-end via zlx

### Extended Parameters

- **PARAM-01**: `temperature`, `top_p`, `presence_penalty`, `frequency_penalty` are forwarded to MLX inference
- **PARAM-02**: `system` role messages are prepended to context correctly

## Out of Scope

| Feature | Reason |
|---------|--------|
| Model auto-download from HuggingFace | Network/auth complexity; manual placement sufficient for personal use |
| TurboQuant C++ kernel binding | No C/extern API exists — pure Python JIT; defer until research phase clarifies MSL extraction path |
| Auth / API keys | Personal use only; single-user, local network |
| Rate limiting | Inference is already mutex-serialized; no multi-user concern |
| TLS / HTTPS | OpenCode connects to localhost only |
| Embeddings endpoint | Out of scope for chat completions focus |
| Legacy `/v1/completions` (non-chat) | Not required by OpenCode; adds format complexity |
| Web UI / dashboard | Binary only; no frontend |
| Distributed inference | Single-machine Metal GPU only |
| Fine-tuning | Inference only |
| Vision / multimodal | Text completions only |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| BUILD-01 | Phase 1 | Pending |
| BUILD-02 | Phase 1 | Pending |
| BUILD-03 | Phase 1 | Pending |
| BUILD-04 | Phase 1 | Pending |
| INFER-01 | Phase 2 | Pending |
| INFER-02 | Phase 2 | Pending |
| INFER-03 | Phase 2 | Pending |
| INFER-04 | Phase 2 | Pending |
| INFER-05 | Phase 2 | Pending |
| HTTP-01 | Phase 3 | Pending |
| HTTP-02 | Phase 3 | Pending |
| HTTP-03 | Phase 3 | Pending |
| HTTP-04 | Phase 3 | Pending |
| HTTP-05 | Phase 3 | Pending |
| HTTP-06 | Phase 3 | Pending |
| INT-01 | Phase 3 | Pending |
| INT-02 | Phase 3 | Pending |

**Coverage:**
- v1 requirements: 17 total
- Mapped to phases: 17
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-30*
*Last updated: 2026-03-30 after initial definition*
