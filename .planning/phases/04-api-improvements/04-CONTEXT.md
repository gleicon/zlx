# Phase 4: API Improvements - Context

**Gathered:** 2026-04-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Full OpenAI API compatibility for the inference server — adding stop sequences, logprobs, complete sampling parameters (top_k, min_p, penalties, logit_bias), seed-based determinism, and proper error handling with request timeouts. This phase extends the existing HTTP API and inference pipeline without changing the single-model architecture.

</domain>

<decisions>
## Implementation Decisions

### Stop Sequence Detection
- **D-01:** Check for stop sequences after each generated token (greedy left-to-right matching)
- **D-02:** Support 1-4 stop sequences per request as specified in OpenAI API
- **D-03:** Multi-character string matching — check if any stop sequence matches the end of generated text
- **D-04:** Stop sequence is NOT included in the output (truncate at match point)
- **D-05:** Set `finish_reason: "stop"` when halted by stop sequence (not "length")
- **D-06:** Works with both streaming (immediate termination) and non-streaming responses

### Logprobs Implementation
- **D-07:** Track top-5 log probabilities for each generated token during sampling
- **D-08:** Return natural log probabilities (negative floating point values)
- **D-09:** Include alternative tokens and their logprobs in `top_logprobs` field
- **D-10:** Works with streaming (per-token) and non-streaming (full sequence)
- **D-11:** Optional field — only returned when `logprobs: true` in request
- **D-12:** Store logits during generation to avoid re-computation (memory tradeoff acceptable for single-user server)

### Sampling Parameter Architecture
- **D-13:** API layer validates all parameter ranges (clamp invalid values, don't error)
- **D-14:** Pass validated parameters through `GenerationOptions` struct to generator
- **D-15:** Implement parameters in this priority order: temperature → top_p → top_k → min_p → penalties
- **D-16:** Default values: top_k=0 (disabled), min_p=0.0, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0
- **D-17:** Logit bias applied as final step before softmax (add bias to specific token IDs)
- **D-18:** Temperature=0 selects highest probability token (greedy, no randomness)

### Seed-Based Determinism
- **D-19:** Use PCG32 RNG (standard deterministic PRNG) when seed is provided
- **D-20:** Same seed + same prompt + same parameters = identical output
- **D-21:** Different seeds produce different outputs
- **D-22:** Seed applies to token sampling only (not dropout, which MLX handles internally)
- **D-23:** No seed specified → use `std.crypto.random` (current behavior, non-deterministic)
- **D-24:** Seed stored in `GenerationOptions`, validated as i32

### Error Handling Strategy
- **D-25:** Return HTTP 400 for JSON parse errors and invalid request structure
- **D-26:** Return HTTP 408 (Request Timeout) when generation exceeds timeout limit
- **D-27:** Return HTTP 500 for generation errors with request ID for debugging
- **D-28:** Request ID format: `req-{timestamp}-{random}` (same pattern as completion IDs)
- **D-29:** Error response includes: `error.message`, `error.type`, `error.param` (if applicable), `error.code` (if applicable)
- **D-30:** Log full error details server-side for troubleshooting
- **D-31:** No stack traces in production error responses (security)

### Request Timeouts
- **D-32:** Default timeout: 60 seconds per request (configurable)
- **D-33:** Timeout measured from request start to completion (not per-token)
- **D-34:** Cancel ongoing generation on timeout (release GPU, clean up KV cache)
- **D-35:** Return partial completion if timeout occurs mid-generation
- **D-36:** `finish_reason: "timeout"` or keep existing reason if generation completed
- **D-37:** Timeout configurable via CLI flag `--timeout <seconds>` and config file

### the agent's Discretion
- Loading/initializing RNG implementations
- Exact algorithm for frequency/presence penalty calculation
- Logit bias internal storage format
- Error message phrasing (keep clear and helpful)
- Timeout cancellation mechanism details

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### API Specification
- `src/api/types.zig` — Existing request/response types, JsonFloat pattern, StopSequence union
- `src/api/handlers.zig` — HTTP endpoint handlers, error response pattern, CORS headers
- `src/api/streaming.zig` — SSE streaming implementation (needs logprobs integration)

### Inference Layer
- `src/inference/generator.zig` — GenerationState, temperature/top_p sampling, next() iterator
- `src/inference/mod.zig` — InferenceContext, GenerationOptions, mutex serialization
- `src/mlx.zig/src/mlx.zig` — MLX array operations, softmax, argmax, random functions

### OpenAI API Reference
- Requirements: API-01 through API-05, INFRA-03, INFRA-04
- OpenAI Chat Completions API spec (stop sequences, logprobs, sampling parameters)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `GenerationOptions` struct — extend with new sampling parameters (currently has max_tokens, temperature, top_p, stop_on_eos)
- `GenerationState` — iterator pattern, `next()` yields one token at a time, ideal for stop sequence checks
- `ChatCompletionRequest` — already has `stop`, `logprobs`, `seed`, `presence_penalty`, `frequency_penalty` fields
- `LogProbs` type — defined in types.zig, not yet populated
- `JsonFloat` pattern — use for all floating-point parameters to accept int or float from clients

### Established Patterns
- API types use `jsonParse` custom parsers for flexibility (MessageContent, JsonFloat)
- Sampling happens in `generator.zig:next()` — apply new parameters there
- Error responses use `sendError()` helper in handlers.zig — extend for different status codes
- Thread-safe inference via mutex in `InferenceContext.generate()`

### Integration Points
- Stop sequence checking hooks into `generator.zig:next()` after each token generation
- Logprobs tracked during sampling in `generator.zig` (store logits before softmax)
- Error handling in `handlers.zig:handleChatCompletions()` — extend try/catch blocks
- Timeout handling needs to wrap the generation call in `InferenceContext.generate()`

### Implementation Notes
- Current sampling: temperature scaling → softmax → random sample (or greedy if temp=0)
- Need to add: top_k filtering → top_p (nucleus) filtering → min_p filtering → penalties → logit_bias → softmax
- Logprobs: Store top-5 token IDs and their log probabilities before sampling
- Stop sequences: Buffer generated text, check suffix against stop sequences after each token
- Seed: Replace `std.crypto.random.float(f32)` with seeded PRNG when seed is specified

</code_context>

<specifics>
## Specific Ideas

- Match OpenAI's exact error response format for maximum client compatibility
- Keep sampling parameter validation at API layer — inference layer trusts inputs
- For stop sequences, consider using a simple string search (std.mem.endsWith) rather than complex trie for 1-4 sequences
- Logprobs: MLX provides logits as f32 array — use that directly rather than re-computing
- Timeout implementation may need signal handling or async cancellation (research MLX interrupt capability)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within Phase 4 scope. All v1.1 requirements beyond Phase 4 (multi-model, prompt caching, TurboQuant, speculative decoding, model management) are properly scoped to later phases.

</deferred>

---

*Phase: 04-api-improvements*
*Context gathered: 2026-04-01*
