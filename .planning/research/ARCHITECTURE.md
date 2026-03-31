# Architecture Research

**Domain:** Local LLM inference server (Zig + MLX + OpenAI-compatible HTTP)
**Researched:** 2026-03-30
**Confidence:** HIGH (based on direct source inspection of MLX.zig, httpz docs, and TurboQuant design)

## Standard Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                       HTTP Layer (httpz)                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  server.zig — POST /v1/chat/completions                    │ │
│  │  JSON parse → ChatRequest → JSON/SSE response              │ │
│  └────────────────────────┬───────────────────────────────────┘ │
└───────────────────────────│─────────────────────────────────────┘
                            │ mutex-guarded call
┌───────────────────────────▼─────────────────────────────────────┐
│                   Inference Layer                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  inference.zig — thin wrapper around MLX.zig            │    │
│  │  • load model (TransformerUnion: llama/phi/qwen)        │    │
│  │  • encode messages → token IDs (Tokenizer)              │    │
│  │  • generateStep() — one token per call (iterator)       │    │
│  │  • decode token IDs → UTF-8 text (Tokenizer)            │    │
│  └──────────────────────────┬──────────────────────────────┘    │
└─────────────────────────────│───────────────────────────────────┘
                              │ direct call (same process)
┌─────────────────────────────▼───────────────────────────────────┐
│                   MLX.zig Runtime                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  mlx.zig — C bindings to Apple MLX C API               │    │
│  │  llama.zig / qwen.zig / phi.zig — Transformer models   │    │
│  │  tokenizer.zig — SentencePiece tokenizer               │    │
│  └──────────────────────────┬──────────────────────────────┘    │
└─────────────────────────────│───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                   KV Cache Layer                                  │
│  ┌─────────────────┐         ┌─────────────────────────────┐    │
│  │  mlx.KVCache    │  OR     │  turboquant_kv.zig           │    │
│  │  (fp16/bf16)    │ (flag)  │  quantize_key + dequantize  │    │
│  │  standard cache │         │  ~4x smaller, 0.98x speed   │    │
│  └─────────────────┘         └─────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│               Metal GPU / Unified Memory (Apple Silicon)         │
└─────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Communicates With |
|-----------|----------------|-------------------|
| `main.zig` | CLI argument parsing (`--model`, `--port`, `--turboquant`, `--max-kv-size`), server start, allocator setup | `server.zig`, `inference.zig` |
| `server.zig` | httpz HTTP listener, route registration (`POST /v1/chat/completions`), JSON deserialization, SSE chunking, JSON non-streaming response | `inference.zig` (via mutex-guarded function call) |
| `inference.zig` | Model lifecycle (load/deinit), tokenizer encode/decode, generation step loop, cache lifecycle | `mlx.zig` TransformerUnion, `tokenizer.zig`, KV cache |
| `turboquant_kv.zig` | Drop-in KVCache replacement using Metal kernels; quantizes keys on `update()`, dequantizes on `get()` | `inference.zig` (swapped in by flag), TurboQuant C++ Metal kernels via `@cImport` |
| `mlx.zig/src/` (submodule) | LLM transformer forward passes, KVCache state, tokenization, MLX C API bindings | Metal GPU via MLX C library |

## Recommended Project Structure

```
src/
├── main.zig              # entry point: CLI flags, init model, start server
├── server.zig            # httpz listener, route handler, JSON/SSE serialization
├── inference.zig         # wrapper: Transformer lifecycle, generateStep(), tokenize
├── turboquant_kv.zig     # optional KVCache replacement using TurboQuant Metal kernels
└── mlx.zig/              # git submodule: MLX.zig (llama/phi/qwen + tokenizer + C bindings)
    └── src/
        ├── mlx.zig       # MLX C bindings + KVCache + Cache structs
        ├── llama.zig     # Llama-3.2 transformer
        ├── qwen.zig      # Qwen2.5 transformer
        ├── phi.zig       # Phi-4 transformer
        ├── tokenizer.zig # SentencePiece tokenizer
        └── llm.zig       # TransformerUnion (dispatch table across model types)

models/                   # runtime: manually-placed model weight directories
└── <model-name>/
    ├── *.safetensors
    ├── config.json
    └── tokenizer.model

zig-out/bin/zlx           # single static binary output
```

### Structure Rationale

- **server.zig is thin by design:** All heavy logic lives in `inference.zig`. The server handles exactly one route. Target is ~50 lines.
- **inference.zig isolates MLX.zig coupling:** Only this file imports the MLX.zig submodule. This makes TurboQuant swapping and future changes to the runtime containable.
- **turboquant_kv.zig is conditionally compiled:** Activated via `-Dturboquant` build flag and a runtime `--turboquant` CLI arg. The standard `mlx.KVCache` and the TurboQuant version share the same functional interface (`update`, `get`, `set`, `deinit`), so `inference.zig` selects one at runtime.

## Architectural Patterns

### Pattern 1: Token-Step Iterator for Streaming

**What:** Instead of calling MLX.zig's `generate()` which collects all tokens before returning, `inference.zig` exposes a `generateStep()` function that advances the generation loop exactly one iteration and returns the produced token ID. `server.zig` calls this in a loop, flushing one SSE chunk per step.

**When to use:** Required for SSE streaming. The existing `Transformer.generate()` in mlx.zig is a batch loop (lines 908–931 of mlx.zig): it accumulates tokens into a `[]u32` slice and returns the whole array. That API cannot stream.

**Trade-offs:** Requires restructuring the inner loop into a resumable state machine (or a struct that holds loop state between calls). Worth doing — it also enables early stopping on client disconnect and max-token limits per-request without duplicating logic.

**Shape (Zig pseudocode):**
```zig
// inference.zig
pub const GenerationState = struct {
    cache: mlx.Cache,
    toks: mlx.Array,
    logits: mlx.Array,
    offset: usize,
    // ...

    pub fn next(self: *GenerationState) !?u32 {
        // one forward pass, argmax, return token id or null on EOS
    }

    pub fn deinit(self: *GenerationState) void { ... }
};

pub fn startGeneration(allocator: std.mem.Allocator, input_ids: []const u32) !GenerationState {
    // init cache, set toks, return state
}
```

### Pattern 2: Mutex-Serialized Inference Access

**What:** A single `std.Thread.Mutex` wraps all calls into the `inference.zig` layer. httpz uses a thread pool for HTTP handling; without serialization, two concurrent requests can corrupt shared KVCache state and MLX array buffers.

**When to use:** Always, for this project. Metal on Apple Silicon processes MLX streams serially. Concurrent forward passes on the same model are not safe — they share the same `mlx.Stream`, KVCache arrays, and logit buffers.

**Trade-offs:** Any second request blocks until the first completes. Acceptable for personal use; a 7B-4bit model at 80–120 tok/s finishes a typical coding completion in 10–30 seconds. If two requests arrive simultaneously, the second waits. Queue depth is httpz's connection queue. Do not attempt parallel inference.

**Shape:**
```zig
// server.zig or inference.zig
var inference_mutex: std.Thread.Mutex = .{};

pub fn handleChatCompletion(req: *httpz.Request, res: *httpz.Response) !void {
    inference_mutex.lock();
    defer inference_mutex.unlock();
    // ... call inference ...
}
```

### Pattern 3: TurboQuant as Drop-In KVCache

**What:** `turboquant_kv.zig` implements the same `update/get/set/deinit` interface as `mlx.KVCache`, but intercepts `update()` to call the TurboQuant Metal kernels (`quantize_key`, `dequantize`, `residual sketch`) before concatenation. The transformer model code in MLX.zig calls `cache.update()` identically regardless of which implementation is active.

**When to use:** Activated by `--turboquant` CLI flag. The KVCache selection happens in `inference.zig` during model/cache initialization, not in the transformer layer.

**Trade-offs:** Adds ~200 LOC of C interop. The Metal kernels are already benchmarked at ~0.98x FP16 decode speed (per TurboQuant paper) with 4–6x smaller KV footprint. The interop must bind `quantize_key`, `dequantize`, and optionally `residual_sketch` from `arozanov/turboquant-mlx` via `@cImport` / `extern` declarations in `turboquant_kv.zig`.

## Data Flow

### Non-Streaming Request Flow

```
Client POST /v1/chat/completions
    ↓
httpz thread pool (accept + parse)
    ↓
server.zig: std.json.parseFromSlice(ChatRequest)
    ↓
inference_mutex.lock()
    ↓
inference.zig: tokenizer.encodeChat(messages) → []u32
    ↓
inference.zig: startGeneration(input_ids)
    ↓
[loop] GenerationState.next() → token_id  (one MLX forward pass per call)
    |→ tokenizer.decodeIncremental(token_id) → text fragment
    |→ accumulate into output buffer
    ↓ (EOS or max_tokens reached)
inference_mutex.unlock()
    ↓
server.zig: serialize OpenAI ChatCompletion JSON
    ↓
res.json(response)
    ↓
Client receives complete response
```

### SSE Streaming Request Flow

```
Client POST /v1/chat/completions  {stream: true}
    ↓
httpz thread pool
    ↓
server.zig: parse JSON, set Content-Type: text/event-stream
    ↓
inference_mutex.lock()
    ↓
inference.zig: tokenizer.encodeChat(messages) → []u32
    ↓
inference.zig: startGeneration(input_ids)
    ↓
[loop]
  GenerationState.next() → token_id
    ↓
  tokenizer.decodeIncremental(token_id) → text fragment (see UTF-8 note below)
    ↓
  server.zig: write SSE chunk  "data: {\"choices\":[{\"delta\":{\"content\":\"...\"},\"finish_reason\":null}]}\n\n"
    ↓
  res.flush()  (push bytes to client immediately)
    ↓ (loop until EOS or max_tokens)
write "data: [DONE]\n\n"
inference_mutex.unlock()
    ↓
Client receives tokens as they are generated
```

### Key Data Flow Concerns

1. **Token batch vs. single-step:** MLX.zig's `generate()` collects all output tokens before returning. For streaming, `inference.zig` must implement a step-by-step loop, as described in Pattern 1. This is the primary architectural adaptation needed.

2. **UTF-8 partial character buffering:** The tokenizer's `decode()` operates on full `[]u32` slices. When calling it with a single token ID, the resulting bytes may form an incomplete UTF-8 multi-byte sequence (e.g., a Chinese character spans multiple bytes that may be split across token boundaries). `server.zig` or `inference.zig` must buffer decoded bytes until a complete UTF-8 code point is available before flushing an SSE chunk. Sending truncated UTF-8 to the client causes JSON parse errors.

3. **httpz SSE API (MEDIUM confidence):** httpz's README confirms SSE support exists. The exact function call signatures (e.g., `res.startEventStream()`, chunk write method, flush call) must be confirmed against httpz source during Phase 1 implementation. Do not assume API shape — inspect `src/response.zig` in the httpz dependency.

4. **Cache state per request:** A `mlx.Cache` (or TurboQuant equivalent) is created fresh at the start of each request and destroyed when the request completes. Cache state is not shared across requests. This is the correct model for a single-user server.

## Build Order (Dependency Chain)

Build these in sequence — each phase depends on the previous:

```
1. inference.zig (standalone)
   ↓ prerequisite
2. GenerationState step iterator (token-by-token)
   ↓ prerequisite
3. server.zig (httpz HTTP + JSON, non-streaming first)
   ↓ depends on 2
4. SSE streaming (connect step iterator to SSE chunks)
   ↓ independent of 3
5. turboquant_kv.zig (swap KVCache, compile-time flag)
```

**Rationale:**
- `inference.zig` must be independently testable before HTTP is introduced. Validate model loading and batch generation without any HTTP.
- The step iterator refactor is the hardest adaptation. Getting it right before adding HTTP reduces the debugging surface.
- `server.zig` non-streaming (`stream: false`) should work first; add SSE after non-streaming is confirmed.
- TurboQuant is architecturally independent of HTTP — it only touches the KVCache interface inside `inference.zig`. Build it last once the generation loop is stable.

## Concurrency Model

**Model:** Single inference worker, multi-threaded HTTP layer.

httpz runs a configurable thread pool for accepting and dispatching HTTP requests. This is appropriate for I/O handling and JSON parsing. However:

- **Inference is serialized.** A mutex in `server.zig` or `inference.zig` ensures only one request runs through the model at a time. This is required because the `mlx.Stream`, `KVCache` arrays, and logit buffers are not thread-safe across concurrent model invocations.
- **Metal is single-stream for this workload.** Apple's MLX dispatches ops to the default GPU stream. Concurrent submission from multiple threads to the same stream produces undefined behavior.
- **httpz thread pool size for this use case:** Because requests block in inference (10–30 seconds), set `httpz.Config.thread_pool` workers to a small number (1–4). Extra workers only help if requests queue up — they cannot speed up inference.
- **Future consideration:** A dedicated inference worker thread with a bounded channel/queue is a cleaner design than a mutex for production use. For personal use, a mutex is sufficient and simpler to implement.

## Anti-Patterns

### Anti-Pattern 1: Calling generate() for Streaming

**What people do:** Call MLX.zig's existing `Transformer.generate(input, num_tokens)` and attempt to stream the result after it returns.

**Why it's wrong:** The function returns all tokens at once as `[]u32`. Streaming a batch response after the fact defeats SSE — the client receives the entire response in one TCP segment after model completion, not incrementally.

**Do this instead:** Implement `GenerationState` with a `next()` method (Pattern 1). Extract the body of the `while (i < num_tokens)` loop in `mlx.zig:908–931` into a resumable struct.

### Anti-Pattern 2: Parallel Inference

**What people do:** Allow multiple httpz thread pool workers to each call inference concurrently (no mutex).

**Why it's wrong:** The `KVCache` struct (k/v MLX arrays) is mutated in-place on every `update()` call. Two concurrent requests will corrupt each other's KV state. MLX array operations via the C API are not thread-safe for concurrent writes to the same array.

**Do this instead:** Use `inference_mutex.lock()/unlock()` around the full inference path. One request runs at a time; the second queues in httpz's connection pool.

### Anti-Pattern 3: Re-encoding the Full Message List Per Step

**What people do:** Re-run `tokenizer.encodeChat(messages)` on every generation step, or reconstruct the prompt context from scratch per SSE chunk.

**Why it's wrong:** Tokenization is not free; for long coding contexts (32k tokens), re-encoding wastes CPU on every token generated.

**Do this instead:** Encode once before the generation loop. Pass the resulting `[]u32` into `startGeneration()` and hold the `GenerationState` for the duration of the request.

### Anti-Pattern 4: Sending Partial UTF-8 in SSE Chunks

**What people do:** Decode each token ID individually and immediately write the raw byte string to the SSE response without checking UTF-8 completeness.

**Why it's wrong:** LLM tokenizers (SentencePiece) can produce tokens whose decoded byte sequences begin or end mid-character. Multi-byte UTF-8 characters (e.g., CJK characters, emoji) will appear broken in the client, causing JSON parse failures in OpenCode/Continue.dev.

**Do this instead:** Buffer decoded bytes. Only flush an SSE chunk once the accumulated byte buffer ends on a valid UTF-8 boundary. `std.unicode.utf8ValidateSlice()` is the check to use.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| MLX C library | `@cImport("mlx/c/mlx.h")` via mlx.zig | Already handled by MLX.zig submodule build system |
| TurboQuant Metal kernels | `extern` declarations in `turboquant_kv.zig` + link C++ `.a` from arozanov/turboquant-mlx | Must compile TurboQuant to static lib; add to `build.zig` as `exe.addObjectFile(...)` |
| OpenCode / Continue.dev | Standard HTTP client pointing at `http://127.0.0.1:8080/v1` | No integration code needed; OpenAI-compatible JSON/SSE format is sufficient |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `server.zig` ↔ `inference.zig` | Direct function calls (same process, same thread for inference) | Always mutex-guarded; never call inference from multiple threads concurrently |
| `inference.zig` ↔ `mlx.zig` (submodule) | Import via `@import("mlx")` build dependency | Only `inference.zig` imports mlx; server.zig does not |
| `inference.zig` ↔ `turboquant_kv.zig` | Comptime flag selects KVCache implementation | `turboquant_kv.zig` implements same interface as `mlx.KVCache`; switchable without changing generation loop |
| `server.zig` ↔ SSE client | HTTP/1.1 chunked transfer, `text/event-stream` MIME type | httpz SSE API: confirm exact call signatures from httpz source before implementing |

## Scaling Considerations

This server is scoped to personal use, single machine, single user. Scaling is not a goal. For completeness:

| Scale | Architecture Adjustments |
|-------|--------------------------|
| 1 user (personal) | Current design: mutex-serialized inference, httpz thread pool for HTTP |
| 2–5 concurrent users | Mutex serialization queues requests; add bounded request queue with 503 on overflow to avoid unbounded memory use |
| Production multi-user | Replace mutex with a dedicated inference worker thread + async channel; or run multiple server processes on different ports behind a load balancer |

## Sources

- MLX.zig source (inspected directly): `/Users/gleicon/code/zig/zlx/src/mlx.zig/src/mlx.zig` — KVCache struct (lines 980–1038), generate loop (lines 908–931)
- MLX.zig source: `llama.zig`, `qwen.zig`, `phi.zig` — Attention.forward signature with `cache: ?*mlx.KVCache` parameter (confirms drop-in replacement pattern)
- MLX.zig source: `llm.zig` — TransformerUnion dispatch, ChatConfig shape, generate call
- httpz (HIGH confidence): [github.com/karlseguin/http.zig](https://github.com/karlseguin/http.zig) — thread pool model (kqueue/epoll for I/O, thread pool for request processing), SSE support confirmed; exact SSE API signatures require source inspection
- TurboQuant: Referenced in prd.md and PROJECT.md — `quantize_key`, `dequantize`, `residual sketch` Metal kernels at [arozanov/turboquant-mlx](https://github.com/arozanov/turboquant-mlx); <200 LOC, benchmarked on M4 Pro (MEDIUM confidence — not directly inspected)
- Project context: `/Users/gleicon/code/zig/zlx/.planning/PROJECT.md`

---
*Architecture research for: zlx — Zig LLM inference server (MLX + OpenAI-compatible HTTP)*
*Researched: 2026-03-30*
