# Roadmap: zlx

## Overview

zlx goes from a broken skeleton to a working single-binary OpenAI-compatible inference server in three phases. Phase 1 unblocks the build — nothing compiles today. Phase 2 produces a working token step-iterator over Metal GPU. Phase 3 wires it to HTTP and proves OpenCode can use it.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Foundation & Build** - Resolve build blockers so `zig build` succeeds on macOS aarch64
- [x] **Phase 2: Inference Core** - Load a model and generate tokens one at a time via Metal GPU
- [x] **Phase 3: HTTP API & Integration** - Expose completions over HTTP and verify OpenCode works end-to-end

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

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation & Build | 1/1 | ✅ Complete | 2026-03-31 |
| 2. Inference Core | 1/1 | ✅ Complete | 2026-03-31 |
| 3. HTTP API & Integration | 1/1 | ✅ Complete | 2026-03-31 |
