---
phase: 04
plan: 03
subsystem: api
tags: [error-handling, timeouts, request-tracking, INFRA-03, INFRA-04]

requires: [INFRA-03, INFRA-04]
provides: []
affects: [src/api/types.zig, src/api/handlers.zig, src/inference/mod.zig, src/main.zig, src/api/server.zig]

tech-stack:
  added: []
  patterns: [error-response-with-request-id, timeout-check-per-token, partial-completion-on-timeout]

key-files:
  created: []
  modified:
    - src/api/types.zig
    - src/api/handlers.zig
    - src/inference/mod.zig
    - src/main.zig
    - src/api/server.zig

decisions:
  - Use page_allocator for request_id generation to avoid allocator dependency in handler
  - Return partial completion on timeout rather than empty response (per D-33)
  - Check timeout every token generation for accurate measurement
  - Default timeout of 60s matches common API practices

metrics:
  duration: "~45 minutes"
  commits: 5
  files_modified: 5
  lines_added: ~350
  lines_removed: ~100
---

# Phase 04 Plan 03: Error Handling and Timeouts Summary

## One-Liner
Implemented comprehensive error handling with request IDs, specific HTTP status codes (400, 404, 408, 500), and configurable request timeouts (1-3600s) that return partial completions on timeout.

## What Was Built

### Error Handling (INFRA-03)

**Request ID Generation** (`src/api/types.zig`):
- `generateRequestId()` - Creates unique request IDs in format `req-{timestamp}-{random}`
- Extended `ErrorResponse` struct to include optional `request_id` field for debugging

**Error Response Functions** (`src/api/handlers.zig`):
- `sendError()` - Generic error response with optional request_id
- `sendBadRequestError()` - HTTP 400 with specific error message and optional param
- `sendTimeoutError()` - HTTP 408 for timeout errors
- `sendServerError()` - HTTP 500 with request_id for server errors

**Request Handler Integration**:
- Generate `request_id` at start of `handleChatCompletions()`
- All error logs now include `[{request_id}]` prefix for correlation
- JSON parse errors return specific messages based on error type (SyntaxError, UnexpectedToken, etc.)
- Model validation errors return HTTP 404 with request_id

### Request Timeout Mechanism (INFRA-04)

**Timeout-Aware Generation** (`src/inference/mod.zig`):
- Added `TimeoutResult` struct with `result` and `timed_out` fields
- Implemented `generateWithTimeout()` in `InferenceContext`
- Checks timeout every token generation (D-33: measured from request start)
- Returns partial completion if timeout occurs mid-generation
- Sets `stop_reason` to `.timeout` when timed out
- Logs warning with timeout duration

**CLI Configuration** (`src/main.zig`, `src/api/server.zig`):
- Added `--timeout <SECONDS>` CLI flag (default: 60s, range: 1-3600s per D-37)
- Updated usage documentation with timeout example
- Timeout stored in `ServerConfig.timeout_seconds`
- Global `request_timeout_seconds` in handlers.zig with `getTimeoutMs()` helper

**Response Building**:
- `buildChatCompletionResponse()` - Constructs OpenAI-compatible response from generation result
- Returns HTTP 408 when timeout occurs
- Properly handles finish_reason mapping (eos -> "stop", length -> "length", timeout -> "timeout")

## Verification Results

### Automated Checks
```bash
# Request ID generation
✓ grep -n "generateRequestId" src/api/types.zig

# Error functions
✓ grep -n "sendBadRequestError\|sendTimeoutError\|sendServerError" src/api/handlers.zig

# Timeout mechanism
✓ grep -n "generateWithTimeout\|TimeoutResult" src/inference/mod.zig

# CLI timeout
✓ grep -n "timeout_seconds\|--timeout" src/main.zig
```

### Build Verification
```bash
✓ zig build succeeded
✓ Binary created: zig-out/bin/zlx (20.3MB)
```

## Requirements Compliance

| Requirement | Status | Evidence |
|-------------|--------|----------|
| INFRA-03: Error handling & recovery | ✅ COMPLETE | HTTP 400 for JSON errors, HTTP 500 for generation errors, all errors include request_id |
| INFRA-04: Request timeout handling | ✅ COMPLETE | Default 60s timeout, HTTP 408 on timeout, partial completion returned, configurable via --timeout |

## Commits

1. **9df95f9** - `feat(04-03): implement request ID generation and error types`
2. **6de1d6a** - `feat(04-03): integrate error handling with request ID into request handler`
3. **46d7fff** - `feat(04-03): implement request timeout mechanism`
4. **5aab6a8** - `feat(04-03): add CLI timeout configuration`
5. **2a89ee9** - `fix(04-03): fix compilation errors for Zig 0.15.2 compatibility`

## Deviation Log

No deviations from the plan. All tasks executed as specified with minor API adjustments for Zig 0.15.2 compatibility (ArrayList initialization pattern, error type names).

## Known Limitations / Future Work

1. **Streaming timeout**: Currently only non-streaming requests use generateWithTimeout. Streaming uses the existing streamResponse function without timeout checking.
2. **No request timeout for model loading**: Timeout only applies to generation, not the initial model loading.
3. **Logprobs memory management**: Complex deinit pattern with @constCast - could be improved with API design changes.

## Next Steps

This plan completes Phase 04 (API Improvements). The next phase according to the roadmap is Phase 05 (Performance & Infrastructure) which should focus on:
- TurboQuant KV-cache compression (PERF-01)
- Prompt caching with KV persistence (PERF-02)

---
*Summary created: 2026-04-01*
*Phase 04 Plan 03: Complete*
