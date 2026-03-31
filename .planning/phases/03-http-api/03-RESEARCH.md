# Phase 3: HTTP API & Integration - Research

**Researched:** 2026-03-31
**Domain:** OpenAI-compatible HTTP API, SSE streaming, httpz integration
**Confidence:** HIGH

## Summary

This research documents the requirements and implementation approach for Phase 3: exposing zlx inference capabilities through an OpenAI-compatible HTTP API. The phase involves three core deliverables:

1. **HTTP Server with httpz**: Using the zig-0.13 branch of httpz to expose endpoints on port 8080
2. **OpenAI API Compatibility**: Implementing `/v1/chat/completions` and `/v1/models` endpoints with proper request/response JSON schemas
3. **SSE Streaming**: Supporting streaming responses via Server-Sent Events for real-time token delivery

**Primary recommendation:** Use httpz's `res.writer()` for SSE streaming, implement OpenAI-compatible JSON schemas using Zig's std.json, and integrate with the existing `GenerationState` iterator for streaming token generation.

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| HTTP-01 | POST /v1/chat/completions returns OpenAI-compatible JSON | Documented request/response schemas; httpz POST handler pattern |
| HTTP-02 | POST /v1/chat/completions with "stream": true delivers SSE chunks | SSE format specification; httpz streaming writer pattern |
| HTTP-03 | Streaming terminates with `data: [DONE]\n\n` | OpenAI SSE termination protocol |
| HTTP-04 | GET /v1/models returns available models list | Static JSON response pattern |
| HTTP-05 | Stop sequences halt generation when matched | Integration with existing `stop_on_eos` in `GenerationOptions` |
| HTTP-06 | SSE chunks respect UTF-8 boundaries | SSE spec requires UTF-8; token-level streaming ensures boundaries |
| INT-01 | OpenCode can connect and receive completions | OpenAI client libraries accept any base URL with `/v1` path |
| INT-02 | curl smoke test works | Standard HTTP POST with JSON body |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| httpz (karlseguin/http.zig) | zig-0.13 branch | HTTP/1.1 server with routing | Zero native deps, ~140K req/s, SSE support via `res.writer()`, already integrated in build.zig |
| std.json | Zig 0.13.0 stdlib | JSON parsing/serialization | No external deps, sufficient for OpenAI schemas |
| std.ArrayList | Zig stdlib | Dynamic buffer for SSE chunks | Standard pattern for building strings |

### Supporting
| Component | Purpose | When to Use |
|-----------|---------|-------------|
| GenerationState iterator | Streaming token generation | SSE streaming responses |
| Tokenizer.encode/decode | Text ↔ token conversion | Request/response processing |
| InferenceContext | Model loading & thread-safe generation | Handler state management |

### Dependencies Already Present
From previous research (STACK.md):
- httpz is already configured in `build.zig` and `build.zig.zon`
- MLX.zig inference infrastructure is complete from Phase 2
- Tokenizer (stub) is in place at `src/inference/tokenizer.zig`

## Architecture Patterns

### Recommended Project Structure
```
src/
├── main.zig                    # Entry point, server initialization
├── inference/
│   ├── mod.zig                # InferenceContext (thread-safe)
│   ├── generator.zig          # GenerationState iterator (existing)
│   └── tokenizer.zig          # Tokenizer interface (existing)
├── api/
│   ├── server.zig             # httpz server setup, router
│   ├── handlers.zig           # Request handlers (/v1/chat/completions, /v1/models)
│   ├── types.zig              # OpenAI-compatible JSON types
│   └── streaming.zig          # SSE streaming implementation
└── mlx.zig/                   # MLX.zig submodule (existing)
```

### Pattern 1: httpz Handler with Request Parsing
**What:** Parse JSON request body, route to appropriate handler
**When to use:** All POST endpoints with JSON bodies
**Example:**
```zig
const std = @import("std");
const httpz = @import("httpz");

fn chatCompletions(req: *httpz.Request, res: *httpz.Response) !void {
    // Parse JSON body
    const body = req.body() orelse return error.MissingBody;
    
    var parsed = try std.json.parseFromSlice(ChatCompletionRequest, res.arena, body, .{});
    defer parsed.deinit();
    
    // Route to streaming or non-streaming
    if (parsed.value.stream) {
        try handleStreaming(res, parsed.value);
    } else {
        try handleNonStreaming(res, parsed.value);
    }
}
```

### Pattern 2: SSE Streaming with res.writer()
**What:** Use httpz response writer for chunked SSE delivery
**When to use:** `stream: true` requests
**Example:**
```zig
fn handleStreaming(res: *httpz.Response, request: ChatCompletionRequest) !void {
    // Set SSE headers
    res.status = 200;
    res.content_type = "text/event-stream";
    try res.header("Cache-Control", "no-cache");
    try res.header("Connection", "keep-alive");
    
    const writer = res.writer();
    
    // Stream tokens
    var state = try initGeneration(request);
    defer state.deinit();
    
    var chunk_id: u32 = 0;
    while (try state.next()) |token| {
        const text = try decodeToken(token);
        
        // Write SSE format: data: {...}\n\n
        try writer.print("data: ", .{});
        try writeChunkJson(writer, .{
            .id = chunk_id,
            .choices = .[{ .delta = .{ .content = text } }],
        });
        try writer.print("\n\n", .{});
        try res.flush(); // Ensure immediate delivery
        
        chunk_id += 1;
    }
    
    // Termination signal
    try writer.print("data: [DONE]\n\n", .{});
}
```

### Pattern 3: Thread-Safe Model Access
**What:** Use `InferenceContext.generation_mutex` for concurrent request handling
**When to use:** All generation requests (models are not thread-safe for parallel generation)
**Example:**
```zig
// In handlers.zig
var g_inference_ctx: ?inference.InferenceContext = null;

pub fn initInferenceContext(allocator: std.mem.Allocator, model_path: []const u8) !void {
    g_inference_ctx = try inference.InferenceContext.init(allocator, model_path);
}

fn handleGeneration(request: ChatCompletionRequest) ![]const u8 {
    const ctx = g_inference_ctx.?;
    
    // InferenceContext.generate() already holds mutex internally
    return try ctx.generate(buildPrompt(request.messages), .{
        .max_tokens = request.max_tokens,
        .temperature = request.temperature,
        .stop_on_eos = true,
    });
}
```

### Pattern 4: OpenAI Types (JSON Schema)
**What:** Define structs matching OpenAI API exactly
**When to use:** Request parsing and response serialization
**Example:**
```zig
// types.zig
pub const ChatCompletionRequest = struct {
    model: []const u8,
    messages: []Message,
    stream: bool = false,
    max_tokens: ?u32 = null,
    temperature: ?f32 = null,
    stop: ?StopSequence = null,
    
    pub const Message = struct {
        role: []const u8,  // "system", "user", "assistant"
        content: []const u8,
    };
    
    pub const StopSequence = union(enum) {
        string: []const u8,
        array: []const []const u8,
    };
};

pub const ChatCompletionResponse = struct {
    id: []const u8,
    object: []const u8 = "chat.completion",
    created: i64,
    model: []const u8,
    choices: []Choice,
    usage: Usage,
    
    pub const Choice = struct {
        index: u32 = 0,
        message: Message,
        finish_reason: ?[]const u8 = null,
    };
    
    pub const Usage = struct {
        prompt_tokens: u32,
        completion_tokens: u32,
        total_tokens: u32,
    };
};
```

### Anti-Patterns to Avoid
- **Using res.body for streaming:** SSE requires `res.writer()` for incremental output; `res.body` buffers entire response
- **Ignoring UTF-8 boundaries:** Token-level streaming naturally respects boundaries; don't split multi-byte sequences
- **Thread-unsafe model access:** Always hold mutex during generation; MLX models are not thread-safe
- **Hardcoding model IDs:** Read from config or scan `./models/` directory

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| JSON parsing | Manual byte parsing | std.json.parseFromSlice | Complex nested structures, escape handling |
| HTTP routing | Manual path matching | httpz router | Path parameters, method routing, middleware |
| SSE formatting | String concatenation | Structured writer.print | Field ordering, newline handling |
| Date/timestamp | Manual time calculation | std.time.timestamp | Unix epoch for OpenAI `created` field |
| UUID generation | Random strings | std.crypto.random + fmt | OpenAI uses `chatcmpl-` prefix + random ID |
| Base URL routing | Manual string split | httpz router with `/v1` prefix | Clean separation, testability |

## Common Pitfalls

### Pitfall 1: SSE Format Errors
**What goes wrong:** Missing `data:` prefix, incorrect newlines, or forgetting the final blank line
**Why it happens:** OpenAI clients expect exact format: `data: {...}\n\n`
**How to avoid:**
- Always prefix with `data: `
- End each chunk with `\n\n` (two newlines)
- Terminate stream with `data: [DONE]\n\n`
**Warning signs:** Client receives no events, or events are batched unexpectedly

### Pitfall 2: JSON Schema Mismatches
**What goes wrong:** OpenAI clients expect specific field names and types
**Why it happens:** Using Zig naming conventions instead of OpenAI's (e.g., `finish_reason` vs `finishReason`)
**How to avoid:**
- Use `std.json` with explicit field names via `@"field-name"`
- Test against actual OpenAI client libraries
- Reference llama.cpp's `oai.h` for field names
**Warning signs:** Client parses response but fields are empty/missing

### Pitfall 3: UTF-8 Multi-byte Token Splitting
**What goes wrong:** Multi-byte UTF-8 characters split across SSE chunks
**Why it happens:** Tokenizer may emit partial characters at chunk boundaries
**How to avoid:**
- Decode tokens to strings before chunking
- Buffer incomplete UTF-8 sequences
**Warning signs:** Mojibake (garbled characters) in client output

### Pitfall 4: Missing CORS Headers
**What goes wrong:** Browser-based clients (like OpenCode web) can't connect
**Why it happens:** Missing `Access-Control-Allow-Origin` headers
**How to avoid:** Add CORS middleware:
```zig
res.header("Access-Control-Allow-Origin", "*") catch {};
res.header("Access-Control-Allow-Methods", "POST, GET, OPTIONS") catch {};
res.header("Access-Control-Allow-Headers", "Content-Type, Authorization") catch {};
```
**Warning signs:** Browser console shows CORS errors

### Pitfall 5: Model Path Resolution
**What goes wrong:** Model not found when using relative paths from server context
**Why it happens:** Different working directory when running as server vs CLI
**How to avoid:**
- Resolve model paths relative to executable at startup
- Cache resolved path in global state
**Warning signs:** Model loads in CLI mode but not server mode

## Code Examples

### OpenAI-Compatible Request Handler
```zig
// src/api/handlers.zig
const std = @import("std");
const httpz = @import("httpz");
const types = @import("types.zig");
const inference = @import("../inference/mod.zig");

var g_model_path: []const u8 = undefined;
var g_inference_ctx: inference.InferenceContext = undefined;

pub fn init(allocator: std.mem.Allocator, model_path: []const u8) !void {
    g_model_path = try allocator.dupe(u8, model_path);
    g_inference_ctx = try inference.InferenceContext.init(allocator, g_model_path);
}

pub fn chatCompletions(req: *httpz.Request, res: *httpz.Response) !void {
    const body = req.body() orelse {
        res.status = 400;
        try res.json(.{ .error = "Missing request body" }, .{});
        return;
    };
    
    const parsed = std.json.parseFromSlice(
        types.ChatCompletionRequest,
        res.arena,
        body,
        .{ .ignore_unknown_fields = true }
    ) catch |err| {
        res.status = 400;
        try res.json(.{ .error = "Invalid JSON", .details = @errorName(err) }, .{});
        return;
    };
    defer parsed.deinit();
    
    if (parsed.value.stream) {
        try handleStreaming(res, parsed.value);
    } else {
        try handleNonStreaming(res, parsed.value);
    }
}

fn handleNonStreaming(res: *httpz.Response, request: types.ChatCompletionRequest) !void {
    const prompt = try buildChatPrompt(res.arena, request.messages);
    
    const response_text = try g_inference_ctx.generate(prompt, .{
        .max_tokens = request.max_tokens orelse 256,
        .temperature = request.temperature orelse 0.7,
        .stop_on_eos = true,
    });
    
    const response = types.ChatCompletionResponse{
        .id = try generateId(res.arena, "chatcmpl"),
        .created = std.time.timestamp(),
        .model = request.model,
        .choices = &[_]types.ChatCompletionResponse.Choice{.{
            .message = .{
                .role = "assistant",
                .content = response_text,
            },
            .finish_reason = "stop",
        }},
        .usage = .{
            .prompt_tokens = @intCast(request.messages.len), // Estimate
            .completion_tokens = @intCast(response_text.len / 4), // Rough estimate
            .total_tokens = 0, // Calculate
        },
    };
    
    try res.json(response, .{});
}
```

### SSE Streaming Implementation
```zig
// src/api/streaming.zig
const std = @import("std");
const httpz = @import("httpz");
const types = @import("types.zig");

pub fn streamResponse(
    res: *httpz.Response,
    request: types.ChatCompletionRequest,
    ctx: anytype,
) !void {
    // SSE headers
    res.status = 200;
    res.content_type = "text/event-stream";
    try res.header("Cache-Control", "no-cache");
    try res.header("Connection", "keep-alive");
    try res.header("Access-Control-Allow-Origin", "*");
    
    const writer = res.writer();
    const id = try generateId(res.arena, "chatcmpl");
    const created = std.time.timestamp();
    
    // Initialize generation
    var gen_state = try ctx.initGeneration(request);
    defer gen_state.deinit();
    
    var chunk_index: u32 = 0;
    
    // Stream each token
    while (try gen_state.next()) |token| {
        const text = try gen_state.decodeToken(token);
        if (text.len == 0) continue;
        
        const chunk = types.ChatCompletionChunk{
            .id = id,
            .object = "chat.completion.chunk",
            .created = created,
            .model = request.model,
            .choices = &[_]types.ChunkChoice{.{
                .index = 0,
                .delta = .{ .content = text },
                .finish_reason = null,
            }},
        };
        
        // Write SSE format
        try writer.writeAll("data: ");
        try std.json.stringify(chunk, .{}, writer);
        try writer.writeAll("\n\n");
        
        // Flush to ensure immediate delivery
        try res.flush();
        
        chunk_index += 1;
    }
    
    // Final chunk with finish_reason
    const final_chunk = types.ChatCompletionChunk{
        .id = id,
        .object = "chat.completion.chunk",
        .created = created,
        .model = request.model,
        .choices = &[_]types.ChunkChoice{.{
            .index = 0,
            .delta = .{},
            .finish_reason = "stop",
        }},
    };
    
    try writer.writeAll("data: ");
    try std.json.stringify(final_chunk, .{}, writer);
    try writer.writeAll("\n\n");
    
    // Termination signal
    try writer.writeAll("data: [DONE]\n\n");
    try res.flush();
}
```

### Models Endpoint
```zig
// GET /v1/models
pub fn listModels(req: *httpz.Request, res: *httpz.Response) !void {
    _ = req;
    
    const models = types.ModelsResponse{
        .object = "list",
        .data = &[_]types.ModelInfo{
            .{
                .id = "qwen2.5-coder-1.5b",
                .object = "model",
                .created = 1700000000,
                .owned_by = "mlx-community",
            },
            // Add more models as needed
        },
    };
    
    try res.json(models, .{});
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Custom HTTP server | httpz | 2024 | Production-ready, tested, SSE support |
| Manual JSON building | std.json with structs | Zig 0.13 | Type safety, validation |
| WebSocket streaming | SSE | 2011+ standard | Simpler, auto-reconnect, firewall-friendly |
| Python inference | MLX.zig native | 2025 | Zero Python deps, Metal GPU |

**Deprecated/outdated:**
- Python-based inference servers: Not applicable, zlx is Zig native
- HTTP/2 or WebSocket streaming: SSE is sufficient for one-way streaming

## Integration with Existing Code

### Leveraging Phase 2 Infrastructure

The existing `GenerationState` iterator in `src/inference/generator.zig` is the key integration point:

```zig
// Existing from Phase 2 - usable directly
pub const GenerationState = struct {
    pub fn next(self: *Self) !?Token;  // Yields one token at a time
    pub fn deinit(self: *Self) void;
};
```

### Changes Required to Existing Files

| File | Changes | Purpose |
|------|---------|---------|
| `src/main.zig` | Add server mode alongside CLI mode | Route to server if `--port` flag present |
| `src/inference/mod.zig` | None - use existing `InferenceContext` | Already thread-safe with mutex |
| `src/inference/generator.zig` | Add `decodeToken()` method | Decode single token for SSE chunks |
| `build.zig` | None - httpz already integrated | Already configured |

### New Files to Create

| File | Purpose |
|------|---------|
| `src/api/server.zig` | httpz server initialization, router setup |
| `src/api/handlers.zig` | `/v1/chat/completions`, `/v1/models` handlers |
| `src/api/types.zig` | OpenAI-compatible JSON request/response types |
| `src/api/streaming.zig` | SSE streaming implementation |

## Open Questions

1. **Stop sequences beyond EOS tokens**
   - What we know: `GenerationOptions` has `stop_on_eos`
   - What's unclear: How to implement arbitrary string stop sequences
   - Recommendation: Start with EOS-only, add string matching in future phase

2. **Token counting for usage metrics**
   - What we know: OpenAI reports prompt/completion token counts
   - What's unclear: Count tokens before generation (requires full prompt tokenization)
   - Recommendation: Estimate or defer; OpenCode doesn't strictly require accurate counts

3. **Chat format template application**
   - What we know: MLX.zig has `encodeChat()` in llm.zig
   - What's unclear: How to map OpenAI message format to MLX chat template
   - Recommendation: Implement simple concatenation for MVP, enhance later

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| httpz | HTTP server | ✓ | zig-0.13 branch | — |
| std.json | JSON parsing | ✓ | Zig 0.13.0 | — |
| MLX.zig | Inference | ✓ | Submodule | — |

**Missing dependencies with no fallback:** None

**Missing dependencies with fallback:** None

## Validation Architecture

### Test Approach
Since Phase 3 is primarily about HTTP API compatibility, validation is integration-focused:

| Test Type | Command | Purpose |
|-----------|---------|---------|
| curl smoke | `curl -X POST http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"test","messages":[{"role":"user","content":"hi"}]}'` | Non-streaming response |
| curl streaming | `curl -N ... -d '{...,"stream":true}'` | SSE chunks delivered |
| OpenCode config | Point OpenCode to `http://localhost:8080` | Full integration |

**No unit test gaps:** Phase 3 primarily wires existing Phase 2 logic; integration tests cover the HTTP layer.

## Sources

### Primary (HIGH confidence)
- httpz zig-0.13 example: `https://raw.githubusercontent.com/karlseguin/http.zig/refs/heads/zig-0.13/example/simple.zig`
- SSE specification: `https://html.spec.whatwg.org/multipage/server-sent-events.html`
- OpenAI Chat Completions API: `https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create/`
- OpenAI Streaming guide: `https://developers.openai.com/api/docs/guides/streaming-responses/`
- Existing zlx codebase: `src/inference/generator.zig`, `src/inference/mod.zig`

### Secondary (MEDIUM confidence)
- llama.cpp server implementation patterns (OpenAI compatibility reference)
- WebSearch results for OpenAI API schema details

### Tertiary (LOW confidence)
- None - all critical information verified from primary sources

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - httpz already integrated, working build
- Architecture: HIGH - Clear patterns from source code and httpz examples
- Pitfalls: MEDIUM-HIGH - Based on SSE spec and common API implementation issues

**Research date:** 2026-03-31
**Valid until:** 2026-06-30 (stable API specs, httpz zig-0.13 branch stable)

---

## Integration Notes for Planner

### Key Implementation Details

1. **httpz Response Writer for SSE**
   - Use `res.writer()` not `res.body`
   - Call `try res.flush()` after each chunk for real-time delivery
   - Set `res.content_type = "text/event-stream"` before writing

2. **JSON Struct Naming**
   - OpenAI uses snake_case: `finish_reason`, `max_tokens`
   - Use Zig's `@"field-name"` syntax: `finish_reason: []const u8`

3. **Thread Safety**
   - `InferenceContext.generate()` already holds mutex
   - Safe to call from multiple httpz handler threads
   - Each request gets sequential access to model

4. **CORS for Browser Clients**
   - Add CORS headers to all responses
   - Required for OpenCode web interface

5. **Model Discovery**
   - `/v1/models` can be static list initially
   - Future: scan `./models/` directory at startup

### Dependencies from Previous Phases
- Phase 1 (Build): httpz integration complete
- Phase 2 (Inference): `GenerationState`, `Tokenizer`, `InferenceContext` ready
- Phase 3 can start immediately - no blockers
