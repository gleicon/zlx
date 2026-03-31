# Pitfalls Research

**Domain:** Zig LLM inference server — MLX Metal GPU + TurboQuant KV cache compression
**Researched:** 2026-03-30
**Confidence:** HIGH (primary sources: actual repo inspection, MLX.zig build.zig analysis, TurboQuant repo scan, official Zig docs, httpz source)

---

## Critical Pitfalls

### Pitfall 1: TurboQuant Has No C/C++ API — It Is Python-Only

**What goes wrong:**
The PRD and PROJECT.md describe binding "TurboQuant C++ Metal kernels via Zig extern / @cImport." TurboQuant-MLX (github.com/arozanov/turboquant-mlx) is a **Python library**. There are no `.h` headers, no `.cpp` files, no `.so`/`.a` libraries, and no C ABI surface to import. The Metal kernels are embedded as Python strings and compiled at runtime through the MLX Python bindings. Attempting `@cImport` against TurboQuant will produce an immediate build failure with no path forward.

**Why it happens:**
The description "C++ Metal kernels" conflates the Metal Shading Language (MSL) source (which is in `.metal` or embedded strings) with a C++ library. The actual project is a pip-installable Python package that wraps MLX's Python API.

**How to avoid:**
Accept this constraint before writing any binding code. The approach must change: either (a) reimplement TurboQuant's quantization logic in Zig using MLX-C primitives (the real MLX C API already used by MLX.zig), or (b) pre-compute TurboQuant codebooks offline and call MLX-C's quantized matmul primitives from Zig directly. The Metal shader source in TurboQuant may be extractable (it appears to be embedded in Python strings at runtime), but this was not confirmed — the kernel source file was not accessible during research. This extraction path needs manual inspection of the TurboQuant repo before it can be relied upon as a recovery option.

**Warning signs:**
- Any reference in build.zig to TurboQuant headers or `turboquant-mlx/include/`
- Attempt to `@cImport("turboquant.h")` — file will not exist

**Phase to address:**
Phase 1 (foundation / dependency audit). This must be resolved before any inference code is written. The `--turboquant` flag scope must be explicitly re-defined as "Zig-native KV quantization using MLX-C primitives" rather than a binding to an external C++ library.

**Performance expectation mismatch (related):**
The PRD claims TurboQuant achieves "4–6× smaller KV cache at ~0.98× FP16 decode speed." The actual TurboQuant repo benchmarks on Qwen2.5-7B show TQ3 adaptive running at 30.7–33.0 tokens/second vs. 52.1 tokens/second FP16 baseline — approximately 60% of baseline speed, not 98%. The "98% FP16 speed" claim appears to come from earlier or different benchmark conditions. Do not use the PRD performance numbers as a correctness baseline. Run fresh benchmarks on the target hardware (M4/M5) once the Zig implementation is functional.

---

### Pitfall 2: `@cImport` Cannot Import C++ — Requires `extern "C"` Shims

**What goes wrong:**
Zig's `@cImport` translates C headers, not C++. If any dependency exposes only a C++ API (mangled symbols, classes, templates), `@cImport` will fail or produce unusable types. MLX.zig solves this correctly by using `mlx-c` — the official C wrapper for MLX — and never importing MLX's C++ headers directly. Any custom C++ code (e.g., extracted TurboQuant Metal kernel loaders written as a C++ shim) must expose `extern "C"` functions to be callable from Zig.

**Why it happens:**
C++ uses name mangling for all symbols. Zig's linker can link against C++ object files if the symbols are `extern "C"`, but `@cImport` cannot parse C++ class syntax, templates, or `std::` types at all.

**How to avoid:**
For any C++ component: create a `shim.h` + `shim.cpp` pair that wraps every function the Zig side needs with `extern "C"` linkage. Compile the shim as a C source via `exe.addCSourceFile(...)` in build.zig. Pass only C-compatible types (pointers, scalars, opaque handles) across the boundary.

**Warning signs:**
- Build error: `error: expected ';' after declaration` or `error: unknown type name 'class'` inside a `@cImport` block
- Linker error: `undefined symbol: _ZN...` (mangled C++ name)

**Phase to address:**
Phase 1 — before any FFI code is written. Design the shim API as the first deliverable of any C++ integration task.

---

### Pitfall 3: MLX.zig Requires CMake + `mlx-c` Build on First Run

**What goes wrong:**
MLX.zig's `build.zig` clones `mlx-c` v0.1.2 via `curl`, runs `cmake` + `make`, and produces `libmlxc.a` + `libmlx.a` in the Zig cache directory. If `cmake` is not installed, if the network is unavailable, or if the cache directory is stale with a corrupted partial build, the error surface is confusing — Zig reports missing object files, not a CMake failure.

**Why it happens:**
MLX.zig uses `b.addSystemCommand` to shell out to `sh -c "curl | tar ..."` and then `cmake`. These are opaque to Zig's build graph; failures appear as "file not found" for `libmlxc.a` rather than "cmake failed."

**How to avoid:**
- Verify `cmake`, `make`, and Xcode Command Line Tools are installed before attempting `zig build`
- Run the `install-mlx-c` step explicitly first: `zig build install-mlx-c`
- Check that `mlx.metallib` is copied to the binary's working directory — it is required at runtime for Metal dispatch

**Warning signs:**
- `error: FileNotFound: libmlxc.a` on first build
- `mlx.metallib not found` at server startup despite successful build
- Segfault on first inference call (metallib not loaded)

**Phase to address:**
Phase 1. Document the exact prerequisite commands in the project README before attempting any Zig code.

---

### Pitfall 4: Metal Shader Compilation Is Outside the Zig Build Graph

**What goes wrong:**
`.metal` files must be compiled to `.air` (intermediate) and then `.metallib` (GPU library) via `xcrun metal` and `xcrun metallib`. The Zig build system has no native Metal shader step. If TurboQuant's quantization kernels are reimplemented as standalone Metal shaders, the build pipeline must shell out to `xcrun` — and the resulting `.metallib` must be co-located with the binary at runtime. Forgetting this means the binary builds successfully but crashes on first inference.

**Why it happens:**
Zig's build system is designed for Zig/C/C++ compilation. Apple's Metal toolchain is entirely separate and requires `xcrun`. There is no `addMetalShader()` in `std.Build`.

**How to avoid:**
Add explicit `b.addSystemCommand` steps to build.zig for Metal compilation:
```
xcrun -sdk macosx metal kernel.metal -o kernel.air
xcrun -sdk macosx metallib kernel.air -o kernel.metallib
```
Then use `b.addInstallFile(...)` to copy the `.metallib` into `zig-out/bin/` alongside the executable. At runtime, load via MLX-C's custom kernel API with a path relative to the executable.

**Warning signs:**
- Build succeeds but inference crashes with Metal shader load error
- `.metallib` file absent from `zig-out/bin/`
- Tests pass in CI (no GPU) but fail on actual Mac hardware

**Phase to address:**
Phase 1 / Phase 2 when TurboQuant integration begins. Prototype the Metal compilation pipeline as a standalone step before wiring it into inference.

---

### Pitfall 5: httpz SSE Requires `startEventStream()` — Not the Normal Response Path

**What goes wrong:**
Using `res.body = ...` or `res.json(...)` for streaming closes the connection after sending the full body. To stream SSE, httpz provides `res.startEventStream()` and `res.startEventStreamSync()` functions, plus `res.chunk()` for chunked transfer. Sending the OpenAI `data: {...}\n\n` format through the normal response path will deliver the entire generated response as a single JSON object, breaking streaming behavior expected by OpenCode/Continue.dev.

**Why it happens:**
httpz has distinct code paths for normal responses (Content-Length header, connection close) vs. streaming (Transfer-Encoding: chunked, connection keep-alive). Developers unfamiliar with the library default to the normal path.

**How to avoid:**
For SSE responses:
1. Call `res.startEventStream()` before writing any chunks
2. Use `res.chunk(data)` for each SSE event: format as `"data: {json}\n\n"`
3. Send `"data: [DONE]\n\n"` as the final chunk
4. Handle `error.BrokenPipe` (client disconnected mid-stream) gracefully — do not panic

**Warning signs:**
- OpenCode receives complete response only after generation finishes (not token-by-token)
- Response has `Content-Length` header instead of `Transfer-Encoding: chunked`
- First token latency equals total generation time

**Phase to address:**
Phase 2 (HTTP server + SSE implementation). Write a standalone SSE smoke test before connecting it to inference.

---

### Pitfall 6: OpenAI SSE Format — Wrong JSON Structure Breaks Client Parsers

**What goes wrong:**
The OpenAI streaming format uses `delta` (not `message`) in each chunk's choice object. The `content` field is present only when there is new text; finish chunks have `delta: {}` and `finish_reason: "stop"`. Missing the `[DONE]` terminator causes clients to hang indefinitely. Sending `finish_reason` as `null` in all chunks except the last is required — some clients break if it appears early.

Common specific mistakes:
- Using `"message": {"content": "..."}` instead of `"delta": {"content": "..."}`
- Sending `data: [DONE]` without the trailing `\n\n` — the double newline is required by the SSE spec
- Inconsistent `id` field across chunks within one response (clients may treat this as separate responses)
- Content-Type header as `text/event-stream; charset=utf-8` — some clients reject the charset suffix; use `text/event-stream` only

**Why it happens:**
The OpenAI streaming format is described across multiple API docs versions. The `delta` structure is non-obvious and differs from the non-streaming response format. The `[DONE]` terminator is non-JSON and breaks naive JSON parsers in some SDKs.

**How to avoid:**
Test against the actual OpenCode client from day one. Capture a real OpenAI streaming response and compare structure byte-for-byte. Ensure the `[DONE]` check is done before JSON parsing, not inside it. Use a single `id` generated per-request, passed through all chunks.

**Warning signs:**
- OpenCode shows empty response or spinner that never resolves
- Client logs show JSON parse errors on the final chunk
- Tokens arrive but are not displayed until full generation completes

**Phase to address:**
Phase 2. Wire OpenCode against a stub SSE server returning hardcoded chunks before connecting real inference.

---

### Pitfall 7: KV Cache GPU Memory Is Not Zig-Managed — Do Not `allocator.free()` It

**What goes wrong:**
MLX tensors (including the KV cache) are allocated in Metal's unified memory managed by MLX's own reference-counting system, not Zig's allocator. Calling `allocator.free()` on a pointer to an MLX tensor is undefined behavior. Conversely, failing to release MLX tensors via MLX-C's `mlx_free()` / `mlx_array_free()` functions causes GPU memory growth across requests.

**Why it happens:**
Zig's allocator discipline is strong — developers rightfully free everything they allocate. MLX-C returns raw pointers, and it is easy to mistake them for Zig-allocator-owned memory.

**How to avoid:**
- Never pass MLX tensor pointers to Zig allocators
- Create a wrapper struct `MlxTensor` that holds the C pointer and implements a `deinit()` that calls `mlx_array_free()`
- Use `defer tensor.deinit()` consistently
- Log GPU memory usage (Metal Instruments) during a multi-turn conversation test to detect leaks

**Warning signs:**
- Increasing RSS after each request even after generation completes
- Metal memory pressure warnings in macOS Console
- Performance degrading after N requests (metal allocator starts evicting)

**Phase to address:**
Phase 2 (inference integration). Establish the MLX tensor ownership pattern as a design constraint before writing any generation code.

---

### Pitfall 8: Per-Request ArenaAllocator Reset — Do Not Reuse Across Requests

**What goes wrong:**
Using a single long-lived `ArenaAllocator` for all requests without calling `arena.reset()` between them causes unbounded memory growth. Token buffers, logits scratch space, JSON parse output, and SSE chunk buffers all accumulate. With 32k-context generation, a single request can allocate tens of MB of scratch data.

**Why it happens:**
Arena allocators are convenient and fast. Developers often initialize one at server start and use it globally rather than scoping it per-request.

**How to avoid:**
- Allocator hierarchy: `std.heap.page_allocator` for server-lifetime data (model weights, HTTP listener state)
- Per-request `ArenaAllocator` reset via `arena.reset(.retain_capacity)` after each response completes
- Use `std.heap.GeneralPurposeAllocator` (debug mode) or `std.heap.DebugAllocator` during development to catch leaks
- Switch to `std.heap.c_allocator` for release builds after leak-free validation

**Warning signs:**
- `top` shows RSS growing monotonically across requests
- Zig GPA reporting leaks in test mode that don't reproduce with arena
- OOM kill after N requests in stress testing

**Phase to address:**
Phase 2. Establish the allocator architecture in `server.zig` before writing any per-request handler logic.

---

### Pitfall 9: TurboQuant Residual Sketch Must Reset Between Unrelated Requests

**What goes wrong:**
TurboQuant's residual sketch accumulates quantization error correction across tokens. If the sketch is not reset between separate inference requests (separate conversations), the error from request A contaminates the KV cache of request B. This causes silent quality degradation — output becomes progressively worse with no error or warning.

**Why it happens:**
The sketch is a stateful buffer associated with the KV cache. It is natural to initialize it once and reuse it for performance. The bug is invisible without perplexity tracking.

**How to avoid:**
- Tie sketch initialization to request lifecycle, not server lifecycle
- Reset sketch state in the same cleanup path as the KV cache reset
- Add an integration test that runs two unrelated prompts sequentially and checks output coherence

**Warning signs:**
- Model outputs that degrade in quality after the first request
- Second request produces garbled or off-topic completions
- Quality differences between `--turboquant` and non-turboquant responses grow over time

**Phase to address:**
Phase 3 (TurboQuant integration). Design the TurboQuant state wrapper to make sketch lifetime co-owned with request lifetime from the first line of code.

---

### Pitfall 10: Single `@cImport` Rule — Multiple Imports Cause Type Incompatibility

**What goes wrong:**
If multiple Zig files each call `@cImport` importing the same or overlapping C headers, the types generated in each import are distinct from each other even if structurally identical. Passing a value of `cImport_a.SomeType` to a function expecting `cImport_b.SomeType` is a compile error. This affects `mlx-c` header usage if MLX.zig's bindings are consumed across multiple files.

**Why it happens:**
Zig's `@cImport` is a comptime operation that generates a new namespace each time it is called. There is no global deduplication.

**How to avoid:**
Define all C imports in a single `c.zig` module and re-export only what other files need. All files import from `@import("c.zig")` — never call `@cImport` directly from feature modules.

**Warning signs:**
- Compile error: `expected type 'cImport(...)..SomeStruct', found 'cImport(...)..SomeStruct'`
- Two `@cImport` blocks in two different `.zig` files importing the same header

**Phase to address:**
Phase 1 (project structure). Create `src/c.zig` as the canonical C import boundary before any feature code references C types.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Global ArenaAllocator for all requests | Simpler code | Unbounded memory growth; OOM under load | Never — even single-user personal tool will leak across long sessions |
| Hardcode model path in inference.zig | Faster prototype | Makes CLI `--model` flag nonfunctional | Never — CLI is core requirement from day one |
| Skip `[DONE]` SSE terminator | Slightly simpler server code | OpenCode hangs indefinitely after generation | Never |
| Use `res.body` instead of `res.chunk()` for streaming | One fewer API call | Client receives full response only after generation ends | Never — defeats the entire point of streaming |
| Single @cImport per file | Locally convenient | Type mismatch errors when files share C types | Never — establish `c.zig` immediately |
| Assume TurboQuant is a C++ library (per PRD) | Skip research | Phase 2–3 blocked with no path forward | Never — verify dependency APIs before planning binding strategy |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| MLX.zig as git submodule | Forgetting `--recursive` in submodule init | `git submodule update --init --recursive` — MLX.zig has its own deps |
| MLX-C library path | Hardcoding absolute cache path | Use `b.cache_root.path` as MLX.zig does — path is machine-specific |
| `mlx.metallib` at runtime | Binary built but metallib not copied | Use `b.addInstallFile` to put metallib next to binary in `zig-out/bin/` |
| httpz dependency hash | Pinning to `master` branch URL with `hash = "..."` placeholder | Run `zig fetch` to get real SHA256 hash; stale hash causes build failure |
| httpz Zig version mismatch | Pinning to httpz `master` while using Zig 0.13.0 | httpz master targets Zig 0.15+ as of 2025; build fails on 0.13. Pin to a tagged release compatible with Zig 0.13, or upgrade Zig. Verify with `zig build` before writing any HTTP code. |
| OpenCode base URL | Using `localhost` instead of `127.0.0.1` | Some macOS configurations route `localhost` to IPv6 `::1`; httpz may bind only IPv4. Use `127.0.0.1` explicitly in docs and config |
| SSE Content-Type | `text/event-stream; charset=utf-8` | Use `text/event-stream` only — charset suffix breaks some SSE parsers |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Synchronous JSON serialization of full response before first SSE chunk | First-token latency = total generation time | Stream tokens as they are generated; serialize per-token | Any generation over ~50 tokens |
| Eager KV dequantization (TurboQuant) | Memory savings disappear, speed degrades | Dequantize only at attention computation time, not when inserting into cache | Every request if done eagerly |
| Not flushing SSE chunks immediately | Tokens buffered in OS socket buffer; client sees bursts | Call `res.flush()` or rely on httpz's chunk API which flushes per call | Noticeable on fast M4/M5 where tokens arrive in rapid succession |
| Loading model weights on every request | >10s cold start per request | Load weights once at server startup, keep in memory | First production test request |
| Allocating large scratch buffers in per-token loop | GC pressure (Zig doesn't GC, but allocator fragmentation) | Pre-allocate token scratch buffers sized to `max_tokens` at request start | High-throughput usage with many small allocations per token |

---

## "Looks Done But Isn't" Checklist

- [ ] **SSE streaming:** Server returns 200 and streams chunks — verify first token arrives before generation ends, not after. Use `curl -N` to check.
- [ ] **`[DONE]` terminator:** Generation appears complete — verify `data: [DONE]\n\n` is the last SSE event and the connection closes cleanly. Missing terminator causes OpenCode to hang.
- [ ] **`mlx.metallib` runtime presence:** Binary runs locally — verify `mlx.metallib` is in `zig-out/bin/` alongside the executable. Build succeeds without it but inference crashes.
- [ ] **TurboQuant sketch reset:** First request produces correct output — verify second and third requests also produce correct output. Sketch contamination only manifests after the first request.
- [ ] **Memory stability:** Ten requests complete without error — run 100 requests and check RSS remains bounded. Per-request arena must be reset, not accumulated.
- [ ] **OpenCode integration:** Server responds to curl — verify OpenCode's streaming parser receives tokens incrementally. The `delta` vs `message` structure difference only surfaces in real client testing.
- [ ] **Model path handling:** Works with hardcoded model — test `--model` flag with a different model name. Many server prototypes bake in the model path.

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| TurboQuant Python-only discovery | HIGH | Re-scope TurboQuant integration: design Zig-native quantization using MLX-C primitives; treat as a new research phase |
| C++ binding type mismatch (multiple @cImport) | LOW | Create `src/c.zig`, move all @cImport there, update imports in affected files |
| mlx.metallib missing at runtime | LOW | Add `b.addInstallFile(...)` step to build.zig; `zig build install` copies it |
| SSE using wrong response path | MEDIUM | Refactor server.zig handler to use `startEventStream()` + `chunk()`; existing JSON serialization logic is reusable |
| Arena memory growth | MEDIUM | Identify the allocation site via GPA leak report; add `arena.reset()` to request cleanup path |
| Residual sketch contamination | MEDIUM | Add sketch reset to request lifecycle; regression-test with two consecutive unrelated prompts |
| cmake/mlx-c build failure | LOW-MEDIUM | Install Xcode CLT (`xcode-select --install`), clear MLX.zig cache dir, re-run `zig build install-mlx-c` |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| TurboQuant is Python-only | Phase 1 (dependency audit) | Research TurboQuant repo structure; re-scope before any binding code |
| C++ requires extern "C" shim | Phase 1 (project structure) | Compile a minimal C shim against any C++ dep before integration |
| MLX.zig CMake prerequisites | Phase 1 | `zig build install-mlx-c` completes successfully; `mlx.metallib` present |
| Metal shader outside Zig build graph | Phase 1 / Phase 3 | `zig build` produces `.metallib` in `zig-out/bin/` automatically |
| Single @cImport rule | Phase 1 | Create `src/c.zig`; grep confirms no other file calls `@cImport` |
| httpz SSE path (startEventStream) | Phase 2 | `curl -N` receives chunked response; first token arrives early |
| OpenAI SSE format (delta, [DONE]) | Phase 2 | OpenCode receives streaming tokens without hang |
| KV cache GPU memory ownership | Phase 2 | No RSS growth after 20 requests; GPA reports no leaks |
| Per-request ArenaAllocator reset | Phase 2 | RSS stable across 100 requests in stress test |
| TurboQuant sketch reset | Phase 3 | Two consecutive unrelated prompts both produce coherent output |

---

## Sources

- MLX.zig build.zig (local submodule): `/Users/gleicon/code/zig/zlx/src/mlx.zig/build.zig` — confirmed CMake + mlx-c pattern, Metal framework linking
- TurboQuant-MLX repository (github.com/arozanov/turboquant-mlx): confirmed Python-only, pip-installable, no C/C++ headers
- httpz response.zig (github.com/karlseguin/http.zig): confirmed `startEventStream()`, `startEventStreamSync()`, `chunk()` API
- httpz example 11_html_streaming.zig: confirmed `res.chunk()` is the streaming primitive
- Zig/C++ interop (tuple.app/blog/zig-cpp-interop): opaque type sizing, pointer-only cross-boundary patterns
- Zig @cImport single-import rule (github.com/ziglang/zig/issues/16607): confirmed type incompatibility across multiple @cImport calls
- OpenAI SSE format issues (github.com/llamastack/llama-stack/issues/4744, github.com/openai/openai-python/issues/2722): confirmed [DONE] parsing and delta structure requirements
- Apple Metal command-line compilation (developer.apple.com/documentation/metal/building-a-shader-library-by-precompiling-source-files): xcrun metal + xcrun metallib pipeline
- Zig allocator guide (zig.guide/standard-library/allocators): ArenaAllocator.reset() patterns

---
*Pitfalls research for: Zig LLM inference server (zlx) — MLX Metal GPU + TurboQuant KV cache*
*Researched: 2026-03-30*
