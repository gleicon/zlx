# Phase 3: HTTP API & Integration - Plan

## Goal

OpenCode connects to `http://127.0.0.1:8080/v1` and receives correct streaming completions from a locally loaded model.

## Success Criteria

1. `curl -X POST http://localhost:8080/v1/chat/completions` returns OpenAI-compatible JSON with completion text
2. `curl -N -X POST ... -d '{..."stream":true...}'` receives SSE chunks with tokens streaming in real-time
3. Streaming terminates with `data: [DONE]\n\n` as per OpenAI protocol
4. `curl http://localhost:8080/v1/models` returns list of available models
5. Stop sequences halt generation when matched (start with EOS token, extend to string matching)
6. SSE chunks respect UTF-8 boundaries — no mojibake in multi-byte character output
7. OpenCode configured with base URL `http://localhost:8080` successfully completes chat requests
8. Both streaming and non-streaming smoke tests pass with curl

## Requirements Mapping

| ID | Requirement | Implementation Target |
|----|-------------|---------------------|
| HTTP-01 | POST /v1/chat/completions returns OpenAI JSON | `src/api/handlers.zig` - `handleNonStreaming()` |
| HTTP-02 | POST /v1/chat/completions with stream: true delivers SSE | `src/api/streaming.zig` - `streamResponse()` |
| HTTP-03 | Streaming terminates with `data: [DONE]\n\n` | `src/api/streaming.zig` - final SSE write |
| HTTP-04 | GET /v1/models returns models list | `src/api/handlers.zig` - `listModels()` |
| HTTP-05 | Stop sequences halt generation | `src/inference/generator.zig` - check stop strings |
| HTTP-06 | UTF-8 boundaries respected in SSE | `src/api/streaming.zig` - token-level chunking |
| INT-01 | OpenCode configuration works end-to-end | Config validation, smoke test |
| INT-02 | curl smoke test works | Both streaming and non-streaming |

## Architecture Overview

```
src/
├── main.zig              # Entry point, routes to server or CLI mode
├── inference/
│   ├── mod.zig          # InferenceContext (thread-safe from Phase 2)
│   ├── generator.zig    # GenerationState iterator (from Phase 2, add decodeToken())
│   └── tokenizer.zig    # Tokenizer interface (from Phase 2)
├── api/
│   ├── server.zig       # httpz server setup, router, CORS middleware
│   ├── handlers.zig     # /v1/chat/completions, /v1/models handlers
│   ├── types.zig        # OpenAI-compatible JSON structs
│   └── streaming.zig    # SSE streaming implementation
└── mlx.zig/             # MLX.zig submodule (existing)
```

## Key Design Decisions

1. **httpz integration**: Use zig-0.13 branch (already in build.zig.zon). Router with `/v1` prefix, handlers with request/response types.
2. **OpenAI type mapping**: Use `std.json` with explicit field names via `@{"field-name"}` for snake_case (e.g., `finish_reason`).
3. **Streaming approach**: GenerationState.next() yields tokens one at a time — perfect for SSE. Write `data: {...}\n\n` format with flush after each chunk.
4. **Thread safety**: Reuse InferenceContext from Phase 2 — mutex already guards generation calls.
5. **CORS headers**: Required for OpenCode web client — add middleware to all responses: `Access-Control-Allow-Origin: *`, methods, headers.
6. **Model listing**: Static list initially (qwen2.5-coder-7b, qwen2.5-coder-1.5b). Future: scan `./models/` directory.
7. **Entry point**: Replace CLI with HTTP server mode — `--model` still required, server starts on `--port` (8080 default).

## Tasks

1. **Create src/api/types.zig with OpenAI JSON structs** (30 min)
   - ChatCompletionRequest with messages, model, stream, max_tokens, temperature, stop
   - ChatCompletionResponse with id, object, created, model, choices, usage
   - ChatCompletionChunk for streaming responses
   - ModelsResponse and ModelInfo for /v1/models
   - Use `@{"snake_case"}` syntax for JSON field names

2. **Create src/api/streaming.zig with SSE implementation** (45 min)
   - `streamResponse()` function taking res, request, and InferenceContext
   - Set SSE headers: content-type, cache-control, connection, CORS
   - Loop through GenerationState.next(), write each token as SSE chunk
   - Format: `data: {json}\n\n` with flush after each
   - Write final chunk with finish_reason, then `data: [DONE]\n\n`

3. **Create src/api/handlers.zig - chat completions** (60 min)
   - `chatCompletions()` handler: parse JSON body, route to streaming/non-streaming
   - `handleNonStreaming()`: collect all tokens, return complete JSON response
   - Build chat prompt from messages array (simple concatenation for MVP)
   - Error handling: 400 for invalid JSON, 404 for unknown model
   - Token counting estimates for usage field

4. **Create src/api/handlers.zig - streaming & models** (30 min)
   - Integrate streaming.zig for stream: true requests
   - `listModels()` handler: return static model list
   - Add CORS headers to all responses

5. **Create src/api/server.zig** (30 min)
   - Initialize httpz server on port 8080
   - Router setup: POST /v1/chat/completions, GET /v1/models
   - Global InferenceContext initialization at startup
   - CORS middleware wrapper

6. **Integrate into src/main.zig** (30 min)
   - Replace CLI mode with HTTP server as default
   - Keep --model flag (required), --port flag (8080 default)
   - Initialize InferenceContext, then start httpz server
   - Graceful shutdown handling

7. **Add stop sequence support** (30 min)
   - Extend GenerationState to check for stop strings beyond EOS
   - Pass stop sequences from request to generator options
   - Update finish_reason to "stop" when halted by sequence

8. **Test non-streaming with curl** (15 min)
   - Start server: `./zig-out/bin/zlx --model qwen2.5-coder-7b`
   - Test: `curl -X POST http://localhost:8080/v1/chat/completions ...`
   - Verify JSON structure, choices[0].message.content has text

9. **Test streaming SSE with curl** (15 min)
   - Test: `curl -N -X POST ... -d '{..."stream":true}'`
   - Verify chunks arrive incrementally, not batched
   - Verify final `data: [DONE]\n\n` present

10. **Test OpenCode integration** (30 min)
    - Configure OpenCode with base URL `http://localhost:8080`
    - Send test message, verify completion received
    - Test both streaming and non-streaming modes
    - Document any required configuration tweaks

## Risks

- **Risk**: SSE format errors causing client to not recognize chunks
  - *Mitigation*: Verify exact format with curl, test against actual OpenAI client libraries
  
- **Risk**: JSON schema mismatches — OpenAI clients expect specific field names
  - *Mitigation*: Use explicit `@{"field-name"}` syntax, reference llama.cpp's oai.h for field names

- **Risk**: UTF-8 multi-byte characters split across SSE chunks
  - *Mitigation*: Decode tokens to strings before chunking; MLX tokenizer respects boundaries

- **Risk**: Missing CORS headers prevent browser-based OpenCode client from connecting
  - *Mitigation*: Add CORS middleware to all responses, test with browser dev tools

- **Risk**: Model path resolution differs between CLI and server modes
  - *Mitigation*: Resolve paths relative to executable at startup, cache resolved path

- **Risk**: Stop sequence implementation slows token generation
  - *Mitigation*: Check stop strings only every N tokens or at natural boundaries

## Success Verification

Run the following and confirm output:

```bash
# Build
zig build

# Start server
./zig-out/bin/zlx --model qwen2.5-coder-7b &
SERVER_PID=$!
sleep 2  # Wait for startup

# Test non-streaming
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-coder-7b",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 10
  }' | jq .

# Test streaming
curl -N -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-coder-7b",
    "messages": [{"role": "user", "content": "Count to 3"}],
    "stream": true,
    "max_tokens": 20
  }'

# Verify streaming terminates with [DONE]

# Test models endpoint
curl http://localhost:8080/v1/models | jq .

# Stop server
kill $SERVER_PID
```

Then configure OpenCode:
1. Open OpenCode settings
2. Set base URL to `http://localhost:8080`
3. Select model from list
4. Send test message
5. Verify streaming response

## Time Estimate

- Types and structs: 30 min
- Streaming implementation: 45 min
- Handlers (non-streaming): 60 min
- Handlers (streaming/models): 30 min
- Server setup: 30 min
- Main integration: 30 min
- Stop sequences: 30 min
- Testing (curl): 30 min
- Testing (OpenCode): 30 min
- **Total: ~5.5 hours**
