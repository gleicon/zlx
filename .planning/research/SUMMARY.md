# Project Research Summary

**Project:** zlx — Zig LLM Inference Server
**Domain:** OpenAI-compatible local LLM inference server (Zig + MLX + Apple Silicon)
**Researched:** 2026-03-30
**Confidence:** MEDIUM (stack and architecture HIGH; TurboQuant integration LOW — PRD assumption is wrong)

## Executive Summary

zlx is a single-binary, Python-free LLM inference server targeting Apple Silicon, intended to serve OpenAI-compatible API endpoints (primarily `/v1/chat/completions`) to local coding tools such as OpenCode. The recommended approach is a thin 4-file Zig project (`main.zig`, `server.zig`, `inference.zig`, `turboquant_kv.zig`) wrapping the MLX.zig submodule (already vendored) with the httpz HTTP library (zig-0.13 branch). The stack is tightly version-locked: Zig 0.13.0, MLX.zig pinned to the vendored submodule commit, mlx-c v0.1.2, and httpz's `zig-0.13` branch. Deviating from any single version breaks the build.

The critical finding that overrides the PRD is that **TurboQuant (arozanov/turboquant-mlx) is a Python-only library with no C/C++ API**. The PRD describes binding "C++ Metal kernels via Zig `@cImport`" — this path does not exist. All three research files (STACK, PITFALLS, and ARCHITECTURE) independently confirm this. TurboQuant must be re-scoped from a Phase 1 binding task into a future research-then-implement milestone that reimplements the quantization logic natively using MLX-C primitives. The `--turboquant` CLI flag should be accepted in MVP but print "not yet implemented." The standard MLX KV cache provides a working baseline; TurboQuant is an enhancement, not a prerequisite.

The most technically demanding part of the project is not the HTTP layer — it is adapting MLX.zig's batch-only `Transformer.generate()` into a token-step iterator (`GenerationState.next()`) that enables SSE streaming. This restructuring must happen before any HTTP code is written. The build also has two known blockers that must be resolved first: MLX.zig does not export a Zig module (the `b.dependency("mlx").module("mlx")` call in build.zig will panic), and the httpz dependency hash in build.zig.zon is a placeholder `"..."` that must be replaced by running `zig fetch`.

## Key Findings

### Recommended Stack

The project is built entirely in Zig 0.13.0 with no Python in the final binary. The MLX.zig submodule (already at `src/mlx.zig/`) provides the transformer, tokenizer, and MLX-C bridge. It links Metal, Foundation, QuartzCore, Accelerate, and C++ frameworks at build time via CMake-built `libmlxc.a` and `libmlx.a`. httpz (zig-0.13 branch) provides a zero-native-dep HTTP/1.1 server with SSE support capable of ~140K req/s on M2 hardware, which is appropriate for a personal-use server where the bottleneck is always inference time.

**Core technologies:**
- **Zig 0.13.0**: Language and build system — MLX.zig explicitly targets this version; do not upgrade to 0.14+
- **MLX.zig (submodule)**: Transformer generic, tokenizer, MLX-C bindings — already vendored; `Transformer.init/generate` and `Tokenizer.encodeChat/decode` are verified
- **mlx-c v0.1.2**: C API bridge to MLX C++ framework — fetched by MLX.zig's build.zig at build time via curl; do not upgrade independently
- **httpz (zig-0.13 branch)**: HTTP/1.1 server with SSE — `res.startEventStream()` + `res.chunk()` is the streaming path; the `master` branch targets Zig 0.15.1 and is incompatible
- **CMake + Xcode CLI tools**: Build-time only prerequisites for compiling `libmlxc.a`; must be on PATH for `zig build` to succeed

**Open build blockers (must fix in Phase 1):**
- MLX.zig does not export a named Zig module — build.zig must be rewritten to directly link its `.a` files and import its `.zig` source files
- httpz dependency hash in build.zig.zon is `"..."` — run `zig fetch <zig-0.13 URL>` to generate the real SHA256 hash
- A canonical `src/c.zig` module must be the single `@cImport` boundary for all MLX-C types — multiple `@cImport` calls produce incompatible types

### Expected Features

The MVP must implement the exact OpenAI wire format. Any deviation in field names, JSON structure, or SSE event format causes the client (OpenCode, Continue.dev) to break silently.

**Must have (table stakes — P1):**
- `POST /v1/chat/completions` non-streaming — correct `choices[0].message`, `finish_reason`, `usage` fields
- `POST /v1/chat/completions` SSE streaming — `delta` (not `message`) per chunk, `finish_reason: null` until last chunk, `data: [DONE]\n\n` terminator
- `GET /v1/models` — one entry for the loaded model; OpenCode probes this on connection
- `stop` sequences — pass through to MLX generation loop
- `GET /health` — liveness check
- Correct HTTP error responses — 400 for malformed requests, 500 for internal errors, OpenAI error JSON body

**Should have (differentiators — P2):**
- Tool/function calling — enables OpenCode agentic mode; requires streaming deltas already working
- Extended sampling parameters (`top_k`, `min_p`, `repetition_penalty`) — low complexity, meaningful for model output quality

**Defer (v2+):**
- TurboQuant KV cache — re-scoped as a research-then-implement milestone; `--turboquant` flag accepted but inert in MVP
- LoRA adapter loading at startup
- `--served-model-name` CLI override

**Anti-features (deliberately not built):**
Model auto-download, API key auth, rate limiting, legacy `/v1/completions`, embeddings, multi-model hot-swap, vision input, Web UI, TLS, concurrent multi-session inference.

### Architecture Approach

The architecture is a 4-component pipeline with a clear layer separation. The HTTP layer (`server.zig`) is thin by design — it handles JSON parsing and SSE serialization only. All inference logic is isolated in `inference.zig`, which is the only file that imports MLX.zig. A single `std.Thread.Mutex` serializes access to the inference layer because MLX's Metal stream and KVCache are not concurrency-safe. The critical structural adaptation is extracting the body of MLX.zig's batch `generate()` loop into a `GenerationState` struct with a `next()` method that advances one token per call — this is the prerequisite for SSE streaming.

**Major components:**
1. **`main.zig`** — CLI argument parsing (`--model`, `--port`, `--turboquant`, `--max-kv-size`), allocator setup, server start
2. **`server.zig`** — httpz HTTP listener, route registration, JSON deserialization, SSE chunking, mutex-guarded inference calls
3. **`inference.zig`** — model lifecycle, tokenizer encode/decode, `GenerationState` step iterator, KV cache lifecycle
4. **`turboquant_kv.zig`** — future drop-in `KVCache` replacement; interface matches `mlx.KVCache` (`update/get/set/deinit`)
5. **`src/c.zig`** — single canonical `@cImport` boundary for all MLX-C types

**Note on ARCHITECTURE.md TurboQuant description:** Lines 61 and 154 of ARCHITECTURE.md still describe TurboQuant as a C++ Metal kernel binding via `@cImport`. This is invalidated by STACK.md and PITFALLS.md findings. The `turboquant_kv.zig` component exists as a placeholder; its implementation strategy must be redesigned during the TurboQuant phase.

### Critical Pitfalls

1. **TurboQuant is Python-only — the PRD binding strategy does not exist** — Re-scope before writing any binding code. The `--turboquant` flag is accepted in MVP but inert. A future phase reimplements the quantization logic using MLX-C primitives, not a Python library binding.

2. **`b.dependency("mlx").module("mlx")` panics at build time** — MLX.zig only creates executables, not a library module. Rewrite build.zig to directly link `libmlxc.a`/`libmlx.a` and import MLX.zig source files. Fix this first; nothing else compiles until it is resolved.

3. **SSE requires `res.startEventStream()` + `res.chunk()`, not `res.body`** — Using the normal response path delivers the entire response in one TCP segment after generation completes, defeating streaming. The chunk format must also be exact: `delta` (not `message`), `finish_reason: null` until last chunk, `data: [DONE]\n\n` terminator with double newline.

4. **KV cache GPU memory is MLX-owned — never pass to Zig allocator** — MLX tensor pointers are managed by MLX's reference-counting system. Wrapping them in a struct with a `deinit()` that calls `mlx_array_free()` is the correct pattern. Also: use a per-request `ArenaAllocator` reset via `arena.reset(.retain_capacity)` to prevent unbounded RSS growth.

5. **UTF-8 partial character buffering in SSE** — Single token decode can produce incomplete UTF-8 multi-byte sequences. Buffer decoded bytes and flush only when `std.unicode.utf8ValidateSlice()` confirms a valid boundary. Missing this causes JSON parse errors in OpenCode for any non-ASCII output.

## Implications for Roadmap

Based on the dependency chain established in ARCHITECTURE.md and the constraint that TurboQuant must be re-scoped, the following phase structure is recommended:

### Phase 1: Foundation and Build
**Rationale:** Two known build blockers prevent any code from compiling. The `@cImport` single-import rule and MLX-C build prerequisites must be established before any feature work begins.
**Delivers:** A building project (`zig build` succeeds), `src/c.zig` canonical C import boundary, verified mlx.metallib presence in `zig-out/bin/`, resolved httpz dependency hash
**Addresses:** All 10 pitfalls have prevention steps that must be designed here (allocator strategy, c.zig, TurboQuant re-scoping decision)
**Avoids:** Build panics from module export failure, type incompatibility from multiple `@cImport` calls, CMake/mlx-c build failures

### Phase 2: Inference Core
**Rationale:** The `GenerationState` step iterator is the hardest architectural adaptation. It must work and be independently testable before HTTP is introduced, so inference failures are not confused with HTTP failures.
**Delivers:** `inference.zig` with working model load, batch generation, and `GenerationState.next()` token iterator; testable via a CLI harness (not HTTP)
**Uses:** MLX.zig `Transformer.init`, `Tokenizer.encodeChat/decode`, `mlx.KVCache` lifecycle
**Implements:** Token-step iterator (Pattern 1 from ARCHITECTURE.md)

### Phase 3: HTTP API and SSE Streaming
**Rationale:** With inference working, the HTTP layer is straightforward. Non-streaming response must be confirmed correct before SSE is added. OpenAI wire format precision (FEATURES.md wire format section) is the most error-prone part.
**Delivers:** `server.zig` with `POST /v1/chat/completions` (non-streaming then streaming SSE), `GET /v1/models`, `GET /health`, correct error responses
**Uses:** httpz zig-0.13 branch — `startEventStream()`, `res.chunk()`, mutex-serialized inference calls
**Implements:** Mutex-serialized inference access (Pattern 2 from ARCHITECTURE.md), exact OpenAI SSE chunk format

### Phase 4: TurboQuant KV Cache (Re-scoped)
**Rationale:** TurboQuant cannot be a C++ binding task as described in the PRD. This phase requires dedicated feasibility research before implementation: inspect TurboQuant's Metal shader source strings to determine if they can be extracted and compiled as standalone `.metal` files, or whether the equivalent quantization must be reimplemented from scratch using MLX-C array primitives.
**Delivers:** `turboquant_kv.zig` implementing the `KVCache` interface using Zig-native quantization; activated via `--turboquant` CLI flag
**Research flag:** This phase requires `/gsd:research-phase` before planning. The implementation strategy is not determined by current research.
**Avoids:** TurboQuant sketch reset pitfall (Pitfall 9), Metal shader compilation pipeline pitfall (Pitfall 4)

### Phase 5: Tool Calling and Extended Parameters
**Rationale:** Tool/function calling enables agentic OpenCode workflows but requires streaming already working (tool call arguments arrive as streaming deltas). Deferring until Phase 3 is confirmed stable is correct.
**Delivers:** `tools` array and `tool_choice` parsing, `tool_calls` in response shape, streaming `function.arguments` deltas; extended sampling params (`top_k`, `min_p`, `repetition_penalty`)

### Phase Ordering Rationale

- Phase 1 before everything: two known build blockers exist today; nothing compiles until they are resolved
- Phase 2 before Phase 3: the step iterator is the highest-risk adaptation; isolating it from HTTP reduces debugging surface
- Phase 3 non-streaming before SSE: SSE chunk format is identical to non-streaming response format split into deltas; get the JSON shape right first
- Phase 4 after Phase 3: TurboQuant is orthogonal to the HTTP layer; implementing it requires a stable generation loop as a baseline for correctness testing
- Phase 5 last: tool calling is a complexity spike that requires streaming deltas already working

### Research Flags

Phases requiring `/gsd:research-phase` during planning:
- **Phase 4 (TurboQuant):** Implementation strategy is unresolved. Requires inspecting TurboQuant's Python source to extract Metal shader strings, and determining whether MLX-C v0.1.2 (the pinned version) exposes sufficient primitives for native reimplementation. High uncertainty.

Phases with standard patterns (skip research-phase):
- **Phase 1 (Foundation):** Build system issues are well-understood from STACK.md source inspection; fixes are deterministic
- **Phase 2 (Inference):** MLX.zig API shapes are verified from source; the step iterator pattern is documented in ARCHITECTURE.md
- **Phase 3 (HTTP/SSE):** httpz SSE API is confirmed; OpenAI wire format is documented in FEATURES.md with byte-level precision
- **Phase 5 (Tool calling):** Tool call JSON structure is documented by multiple sources; complexity is high but patterns are known

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | MEDIUM | Zig/MLX.zig/httpz versions verified from source; TurboQuant integration strategy is LOW confidence (PRD assumption wrong, replacement approach undetermined) |
| Features | HIGH | OpenAI wire format verified against official docs and multiple client implementations; client requirements verified against OpenCode and ai-sdk sources |
| Architecture | HIGH | Component design and data flow verified from MLX.zig source inspection; GenerationState pattern is well-reasoned; TurboQuant architecture section is invalidated and must be redesigned |
| Pitfalls | HIGH | All critical pitfalls sourced from primary code inspection, not secondhand reports |

**Overall confidence:** MEDIUM — the core product (OpenAI-compatible inference server with streaming) is HIGH confidence. The TurboQuant component is LOW confidence and requires a dedicated research phase before it can be planned.

### Gaps to Address

- **TurboQuant implementation strategy**: The exact path from "Python Metal shader strings in TurboQuant repo" to "Zig-callable Metal kernels using MLX-C v0.1.2" is not established. This is the largest open question in the project. Must be resolved in Phase 4 research before any implementation begins.

- **httpz SSE exact API signatures**: ARCHITECTURE.md notes (line 223) that the exact `res.startEventStream()` / `chunk()` call signatures should be confirmed by reading `src/response.zig` in the httpz dependency. PITFALLS.md confirms the function names from source inspection, but argument types and flush semantics need verification during Phase 3 implementation.

- **TurboQuant performance benchmarks**: The PRD claims 4-6x KV size reduction at ~0.98x FP16 decode speed. PITFALLS.md (Pitfall 1) cites actual benchmarks showing ~60% of baseline speed on Qwen2.5-7B. The correct performance expectation on the target hardware (M4/M5) is unknown. Benchmark during Phase 4 rather than relying on the PRD numbers.

- **mlx-c v0.1.2 custom ops API availability**: If TurboQuant Phase 4 determines that custom Metal ops are needed, mlx-c v0.1.2 predates the custom ops API (available in v0.4+). Upgrading mlx-c would break MLX.zig's pinned version. This version coupling must be resolved during Phase 4 research.

## Sources

### Primary (HIGH confidence — source code inspected directly)
- `/Users/gleicon/code/zig/zlx/src/mlx.zig/src/mlx.zig` — Transformer generic, C binding pattern, KVCache struct (lines 980-1038), generate loop (lines 908-931)
- `/Users/gleicon/code/zig/zlx/src/mlx.zig/src/llm.zig` — TransformerUnion dispatch, ChatConfig shape
- `/Users/gleicon/code/zig/zlx/src/mlx.zig/src/qwen.zig` — Transformer.init/generate API usage pattern
- `/Users/gleicon/code/zig/zlx/src/mlx.zig/build.zig` — mlx-c v0.1.2 pin, configureExecutable Metal/Foundation linkage
- `github.com/arozanov/turboquant-mlx` — confirmed Python-only, no C/C++ headers (invalidates PRD binding strategy)
- `github.com/karlseguin/http.zig response.zig` — confirmed `startEventStream()`, `startEventStreamSync()`, `chunk()` API

### Secondary (MEDIUM confidence — rendered docs, community sources)
- `github.com/karlseguin/http.zig` (deepwiki + examples) — httpz Server init, routing, SSE confirmation
- `docs.openvino.ai/2025/model-server/ovms_docs_rest_api_chat.html` — OpenAI chat completions wire format reference
- `raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/SERVER.md` — mlx-lm server reference for feature comparison
- `docs.vllm.ai/en/stable/serving/openai_compatible_server/` — vLLM OpenAI-compatible server endpoint list
- `ml-explore.github.io/mlx-c/build/html/overview.html` — mlx-c C API overview

### Tertiary (LOW confidence — needs validation during implementation)
- TurboQuant 4-6x / 0.98x performance claims (PRD origin) — contradicted by actual repo benchmarks showing ~60% baseline speed; validate on target hardware
- httpz thread pool configuration for long-running inference requests — documented behavior but not load-tested for this workload

---
*Research completed: 2026-03-30*
*Ready for roadmap: yes*
