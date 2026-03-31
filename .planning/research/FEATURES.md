# Feature Research

**Domain:** OpenAI-compatible local LLM inference server (Apple Silicon / personal use)
**Researched:** 2026-03-30
**Confidence:** HIGH (OpenAI wire format verified against official docs and multiple client implementations; client requirements verified against OpenCode, ai-sdk source)

---

## Feature Landscape

### Table Stakes (Users Expect These)

Features that must be correct or the client (OpenCode, Continue.dev, curl) immediately breaks.
Missing or malformed = the server is not usable.

| Feature | Why Expected | Complexity (Zig) | Notes |
|---------|--------------|------------------|-------|
| `POST /v1/chat/completions` — non-streaming | Core OpenAI wire format. Every client sends this shape first. | LOW | `messages` array, `model`, `max_tokens`, `temperature`, `top_p`, `stream: false`. Response must include `id`, `object: "chat.completion"`, `created`, `model`, `choices[0].message.{role,content}`, `choices[0].finish_reason`, `usage.{prompt_tokens,completion_tokens,total_tokens}` |
| `POST /v1/chat/completions` — streaming SSE | OpenCode defaults `stream: true`. Without it the UI freezes until generation completes. | MEDIUM | Each chunk: `data: {"id":…,"object":"chat.completion.chunk","choices":[{"delta":{"content":"…"},"finish_reason":null}]}\n\n`. Final chunk sets `finish_reason: "stop"` or `"length"`. Terminated with `data: [DONE]\n\n`. httpz supports chunked transfer; the chunk format must be exact. |
| Correct `finish_reason` values | ai-sdk/openai-compatible parses this to decide if generation is complete or was truncated. | LOW | Must emit `"stop"` (EOS hit), `"length"` (max_tokens hit). Missing or `null` on the final non-streaming response breaks loop detection in clients. |
| `usage` token counts in response | OpenCode and cost-tracking tooling read this field. | LOW | Must be a real count, not zeros. MLX.zig generation loop produces token counts; surface them. |
| `GET /v1/models` | OpenCode probes this on connection to enumerate available models. Fails to connect if endpoint returns 404. | LOW | Response: `{"object":"list","data":[{"id":"<model-name>","object":"model","created":<unix-ts>,"owned_by":"local"}]}`. One entry for the loaded model is sufficient. |
| `stop` sequences | Coding assistants (OpenCode, Continue.dev) inject stop sequences to terminate at code block boundaries or chat turns. | LOW | Accept `stop` field as string or array of strings. Pass through to MLX generation loop's stop criteria. Zig `std.json` can decode to `union(enum)` or check for both types. |
| Correct HTTP response codes | Clients interpret 4xx vs 5xx differently. Bad status codes mask errors as successes. | LOW | 200 for success, 400 for malformed request (bad JSON, missing `messages`), 500 for internal errors. Include `{"error":{"message":"…","type":"…"}}` body on errors — OpenCode surfaces this to the user. |
| `Content-Type: text/event-stream` header for SSE | Browser-based clients and ai-sdk SSE parsers refuse to parse chunks without this header. | LOW | httpz header set before first chunk write. Also set `Cache-Control: no-cache` and `Connection: keep-alive`. |
| `model` field echo in response | Clients match request model to response model for routing. OpenCode validates this. | LOW | Echo back the model name from the request, or the served model name if different. |

### Differentiators (Competitive Advantage)

Features that distinguish this server for its specific use case. None are required for basic compatibility, but some significantly raise the ceiling.

| Feature | Value Proposition | Complexity (Zig) | Notes |
|---------|-------------------|------------------|-------|
| TurboQuant KV cache | 4-6x smaller KV footprint enables 32k+ context on 7B/14B 4-bit models on M-series. No other single-binary Zig server offers this. Core reason for the project's existence. | HIGH | C++ Metal kernel binding via `@cImport`. Zig `extern` for `quantize_key`, `dequantize`, `residual_sketch`. Drop-in replacement for standard MLX KVCache on the inference path. Activated via `--turboquant` CLI flag. |
| Single static binary, no Python | Python-free deployment. `zig build` produces one binary that OpenCode can point at. Zero virtualenv, zero pip install, zero runtime dependency beyond macOS. | MEDIUM | Already the architectural choice. The differentiator is maintaining it — avoid any build step that shells out to Python. MLX.zig handles the C++ bridge. |
| Metal GPU acceleration via MLX | Native Apple Silicon unified memory usage. No CPU fallback penalty. Faster TTFT and decode vs. CPU-bound servers like some llama.cpp builds. | MEDIUM | MLX.zig already handles this. The server must not introduce CPU bottlenecks in the HTTP path (httpz is non-blocking; keep inference on the Metal thread). |
| Tool/function calling support | Enables agentic OpenCode workflows: multi-step code edits, file reads, shell commands. Without this, OpenCode operates in "chat only" mode and cannot execute agent tasks. | HIGH | Requires: accepting `tools` array (JSON Schema) and `tool_choice` in request; emitting `choices[0].message.tool_calls[].{id,type,function.{name,arguments}}` in response; streaming deltas for tool call chunks. Manual nested JSON construction in Zig — no serde equivalent. This is the biggest complexity spike on the path to full agentic use. |
| `--served-model-name` override | Allows pointing any OpenAI client at the server without knowing the exact filesystem model path. `model: "qwen-coder"` in config maps to whatever is loaded. | LOW | CLI flag that sets the string returned in `/v1/models` and echoed in responses. |
| `GET /health` endpoint | Enables liveness probes and simple `curl http://localhost:8080/health` checks during development. | LOW | Not in OpenAI spec but universally expected by devs. Returns `{"status":"ok","model":"<name>"}`. |
| Generation parameters pass-through | Power users tune `top_k`, `min_p`, `repetition_penalty` for specific models. Standard OpenAI params (`temperature`, `top_p`) are table stakes; extended params are a differentiator. | LOW | MLX.zig generation loop accepts these. Accept them in request JSON and forward. Ignore unknown params silently rather than 400ing. |

### Anti-Features (Things to Deliberately NOT Build)

Features that seem natural but are scope traps for a personal-use, single-binary server.

| Feature | Why Requested | Why Problematic for This Project | Alternative |
|---------|---------------|----------------------------------|-------------|
| Model auto-download from HuggingFace | Convenience — users want `--model qwen2.5-coder` to just work. | Requires HF auth tokens, SHA verification, chunked HTTP download, partial download recovery. Adds 200+ lines and a network failure mode on startup. PROJECT.md explicitly scoped this out. | Manual placement in `./models/<name>/`. Document the `huggingface-cli download` one-liner in README. |
| API key authentication | Every OpenAI client sends `Authorization: Bearer <key>`. Looks like a feature request. | This is a localhost personal server. Auth adds config burden with zero security benefit (anyone on the machine can hit localhost already). | Accept and ignore the `Authorization` header. Never validate it. |
| Rate limiting | Production servers need this. Looks like good practice. | Single user, single model. There is nothing to rate-limit. Adds middleware complexity. | — |
| `/v1/completions` (legacy text completions) | Old OpenAI endpoint still used by some tools. | Chat completions cover every modern coding client. Legacy completions require a different prompt format and add a second inference path to maintain. | Document that only `/v1/chat/completions` is supported. Clients using old API should be configured to use the chat endpoint. |
| `/v1/embeddings` | Needed for RAG, semantic search, some context tools. | Requires a separate embedding model loaded alongside the generation model. Doubles memory consumption. No embedding use case in the PROJECT.md scope. | Use a dedicated embedding server (e.g., Ollama with a separate embedding model) if needed. |
| Multi-model hot-swap / model registry | Users might want to switch models without restarting. | Requires a model management layer, lazy loading, and unload logic. Apple Silicon unified memory makes this complex (MLX doesn't guarantee clean unload). Single model per process is the correct scope. | Restart the server with a different `--model` flag. |
| Vision / multimodal input | Some coding models accept images. | PROJECT.md explicitly out of scope. `image_url` content parts add message parsing complexity and a second inference path. | Text-only. Return 400 with clear message if `image_url` content parts are detected. |
| Web UI / chat interface | llama.cpp server and LM Studio have browser UIs. | Not needed for CLI/IDE use. Adds a static file server and frontend asset pipeline. `curl` and OpenCode are the interfaces. | — |
| TLS / HTTPS | Secure the local endpoint. | Localhost only. TLS adds certificate management complexity with zero threat model benefit. | If tunneling is ever needed, use `ngrok` or `cloudflared` in front of the plain HTTP server. |
| Concurrent multi-session inference | Multiple clients submitting simultaneous requests. | MLX inference is single-threaded on the Metal command queue. Concurrent requests would queue or corrupt KV cache state. Personal use = one session at a time. | Queue requests sequentially. Return 503 with `retry-after` if a request arrives while one is in-flight (optional; busy-wait is also acceptable for personal use). |
| Fine-tuning or LoRA hot-load | Some servers support LoRA adapters. | MLX.zig supports LoRA loading at startup but runtime adapter swapping is not in scope. Adds significant state management complexity. | Load with adapter path at startup via CLI flag if needed (future). |

---

## Feature Dependencies

```
/v1/models (GET)
    └── independent of inference path

POST /v1/chat/completions (non-streaming)
    └──requires──> MLX.zig generation loop
    └──requires──> std.json request parsing (messages, model, params)
    └──requires──> correct response shape (id, object, choices, usage)

POST /v1/chat/completions (streaming SSE)
    └──requires──> non-streaming response shape (same JSON, chunked)
    └──requires──> httpz chunked write support
    └──enhances──> user experience (progressive output)

stop sequences
    └──requires──> non-streaming or streaming chat completions
    └──enhances──> coding assistant accuracy at turn boundaries

TurboQuant KV cache
    └──requires──> working MLX.zig inference path
    └──requires──> C++ kernel binding (turboquant_kv.zig)
    └──enhances──> effective context length (32k+)
    └──independent of HTTP/API layer

tool/function calling
    └──requires──> correct non-streaming chat completions (message shapes)
    └──requires──> streaming delta support for tool_calls chunks
    └──enhances──> agentic OpenCode workflows
    └──conflicts with── keeping response serialization simple (HIGH complexity spike)

/health (GET)
    └── independent of inference path
```

### Dependency Notes

- **Streaming requires non-streaming first:** The chunk format is the same JSON as non-streaming responses, just split across SSE events with `delta` instead of `message`. Implement non-streaming correctly first, then split it.
- **Tool calling requires streaming deltas:** OpenCode's ai-sdk sends tool call requests with `stream: true`. Tool call arguments arrive as streaming `function.arguments` deltas. This means tool calling cannot be implemented without streaming already working.
- **TurboQuant is orthogonal to the API layer:** The KV compression happens inside inference.zig. The HTTP server does not need to know about it. Implement and test the API layer with standard KV cache first, then activate TurboQuant via the `--turboquant` flag without changing any HTTP code.

---

## MVP Definition

### Launch With (v1) — Minimum for OpenCode to work in chat mode

- [x] `POST /v1/chat/completions` non-streaming — correct response shape with all required fields
- [x] `POST /v1/chat/completions` streaming SSE — exact chunk format, `[DONE]` terminator
- [x] `GET /v1/models` — returns one entry for the loaded model
- [x] `stop` sequences — pass through to MLX generation
- [x] `GET /health` — simple liveness check
- [x] Correct HTTP error responses (400/500 with JSON body)
- [x] TurboQuant KV cache — activated via `--turboquant` (core value proposition, required from day one per PROJECT.md)

### Add After Validation (v1.x) — Once basic chat is confirmed working

- [ ] Tool/function calling — trigger: OpenCode agentic mode needed for multi-step workflows
- [ ] Extended generation params (`top_k`, `min_p`, `repetition_penalty`) — trigger: model output quality tuning needed

### Future Consideration (v2+) — Defer until core is stable

- [ ] LoRA adapter loading at startup — trigger: model fine-tuning workflow needed
- [ ] `--served-model-name` override — trigger: multiple model configs in OpenCode settings

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Non-streaming chat completions | HIGH | LOW | P1 |
| Streaming SSE chat completions | HIGH | MEDIUM | P1 |
| `/v1/models` endpoint | HIGH | LOW | P1 |
| `stop` sequences | HIGH | LOW | P1 |
| TurboQuant KV cache | HIGH | HIGH | P1 (core value) |
| Correct error responses | MEDIUM | LOW | P1 |
| `/health` endpoint | MEDIUM | LOW | P1 |
| `--served-model-name` | LOW | LOW | P2 |
| Extended sampling params | MEDIUM | LOW | P2 |
| Tool/function calling | HIGH | HIGH | P2 (enables agentic use) |
| Embeddings endpoint | LOW | HIGH | P3 |
| Legacy `/v1/completions` | LOW | MEDIUM | P3 |

**Priority key:**
- P1: Must have for launch
- P2: Should have, add when possible
- P3: Nice to have, future consideration

---

## Competitor Feature Analysis

| Feature | llama.cpp server | mlx-lm server (Python) | Ollama | zlx target |
|---------|-----------------|------------------------|--------|------------|
| Chat completions | Yes | Yes | Yes | Yes (P1) |
| SSE streaming | Yes | Yes | Yes | Yes (P1) |
| `/v1/models` | Yes | Yes | Yes | Yes (P1) |
| Tool calling | Yes | Yes | Yes | v1.x (P2) |
| Embeddings | Yes | No | Yes | No (anti-feature) |
| Legacy completions | Yes | No | Yes | No (anti-feature) |
| Health endpoint | Yes (`/health`) | No | Yes | Yes (P1) |
| Web UI | Yes | No | Yes | No (anti-feature) |
| Model auto-download | No (manual) | Yes (HF) | Yes | No (anti-feature) |
| Multi-model | Yes | Yes (lazy) | Yes | No (anti-feature) |
| Auth/API key | Yes (optional) | No | No | No (anti-feature) |
| TurboQuant KV | No | No | No | Yes (differentiator) |
| Metal native (no Python) | No | No | No | Yes (differentiator) |
| Single static binary | No | No | Yes | Yes (differentiator) |
| Zero Python runtime | No | No | No | Yes (differentiator) |

---

## Wire Format Precision Notes

These fields must be exactly correct. OpenCode's ai-sdk parses them structurally.

### Non-streaming response (required shape)

```json
{
  "id": "chatcmpl-<unique-string>",
  "object": "chat.completion",
  "created": 1711000000,
  "model": "<model-name-from-request>",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "<generated text>"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 42,
    "completion_tokens": 128,
    "total_tokens": 170
  }
}
```

### Streaming chunk (required shape per chunk)

```
data: {"id":"chatcmpl-<id>","object":"chat.completion.chunk","created":1711000000,"model":"<model>","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-<id>","object":"chat.completion.chunk","created":1711000000,"model":"<model>","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}

data: {"id":"chatcmpl-<id>","object":"chat.completion.chunk","created":1711000000,"model":"<model>","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

Notes:
- First chunk includes `"role": "assistant"` in delta; subsequent chunks omit it
- `finish_reason` is `null` on all chunks except the final one
- `data: [DONE]` is a literal string, not JSON; it terminates the SSE stream
- Required HTTP headers: `Content-Type: text/event-stream`, `Cache-Control: no-cache`, `Connection: keep-alive`

### Error response (required shape)

```json
{
  "error": {
    "message": "messages field is required",
    "type": "invalid_request_error",
    "param": null,
    "code": null
  }
}
```

---

## Sources

- [OpenAI Chat Completions API — OpenVINO docs (wire format reference)](https://docs.openvino.ai/2025/model-server/ovms_docs_rest_api_chat.html)
- [mlx-lm SERVER.md (official MLX server reference)](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/SERVER.md)
- [vLLM OpenAI-compatible server — supported endpoints](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/)
- [LM Studio REST API v0 — endpoint list](https://lmstudio.ai/docs/developer/rest/endpoints)
- [llama.cpp server README — endpoints and params](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [OpenCode providers docs — baseURL configuration](https://opencode.ai/docs/providers/)
- [Local LLM Hosting Complete 2025 Guide — ecosystem overview](https://medium.com/@rosgluk/local-llm-hosting-complete-2025-guide-ollama-vllm-localai-jan-lm-studio-more-f98136ce7e4a)
- [Experiments with Local LLMs for agentic coding](https://www.jethrocarr.com/2025/08/17/experiments-with-local-llms-for-agentic-coding/)
- [mlx-lm tool use guide — tool calling in MLX server](https://medium.com/@levchevajoana/a-job-postings-tool-a-guide-to-mlx-lm-server-and-tool-use-with-the-openai-client-edb9a5d75b4c)

---

*Feature research for: OpenAI-compatible local LLM inference server (personal use, Apple Silicon)*
*Researched: 2026-03-30*
