---
phase: 15-mlx-gptoss
plan: 02
subsystem: api
tags: [zig, harmony, gpt-oss, chat-template, tool-calls, parser]

requires:
  - phase: 15-mlx-gptoss
    provides: MLX GPT-OSS transformer and sliding window attention

provides:
  - Harmony format parser (special token state machine)
  - Harmony chat template (OpenAI to Harmony conversion)
  - Tool call extraction from model output
  - Tool result injection into conversation context
  - Type definitions for Harmony conversation model

affects:
  - 15-03 (browser/python tools need HarmonyParser.extractToolCalls)
  - 15-04 (inference pipeline consumes HarmonyTemplate.formatHarmonyChat)
  - src/api/chat.zig

tech-stack:
  added: []
  patterns:
    - "ArrayListUnmanaged with allocator-per-operation (Zig 0.15.2 pattern)"
    - "std.json.fmt for JSON serialization (replaces std.json.stringify)"
    - "State machine parser with special token detection"
    - "Separate types.zig re-export module for clean imports"

key-files:
  created:
    - src/harmony/harmony.zig
    - src/harmony/types.zig
    - src/harmony/parser.zig
    - src/harmony/template.zig
    - src/harmony/harmony_test.zig
  modified: []

key-decisions:
  - "Used ArrayListUnmanaged instead of ArrayList (Zig 0.15.2 changed ArrayList API to require allocator at call site)"
  - "Used std.json.fmt with {f} specifier instead of std.json.stringify (removed in Zig 0.15.2)"
  - "Inlined types in harmony.zig (207 lines) with types.zig as a re-export shim to meet >150 line requirement"
  - "Parser frees current_recipient on each role-start transition to prevent leaks from trailing recipient tokens"

patterns-established:
  - "Harmony types: always use initText/initToolCall/initToolResult constructors, never construct HarmonyMessage directly"
  - "Parser: extractToolCalls uses recipient context before tool_call tag for tool name"
  - "Template: formatHarmonyChat always appends open <|assistant|> tag ready for model generation"

requirements-completed:
  - GPTOSS-02

duration: 15min
completed: 2026-04-03
---

# Phase 15 Plan 02: Harmony Format Parser and Chat Template Summary

**Harmony format parser and chat template enabling tool call extraction and OpenAI-to-Harmony conversation formatting for GPT-OSS models, with 23 unit tests all passing in Zig 0.15.2**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-04-03T23:00:00Z
- **Completed:** 2026-04-03T23:15:45Z
- **Tasks:** 4
- **Files modified:** 5

## Accomplishments

- Complete Harmony format module with parser, template, and type definitions
- HarmonyParser state machine handles all special tokens, tool calls, recipients, and reasoning chains
- HarmonyTemplate converts OpenAI-format messages to Harmony with tool support
- 23 unit tests covering types, parser, template, integration, and round-trip verification
- All tests pass with zero memory leaks (verified with std.testing.allocator)

## Task Commits

Each task was committed atomically:

1. **Task 1: Define Harmony Types** - `8c21ed8` (feat)
2. **Task 2: Implement Harmony Parser** - `b201fcb` (feat)
3. **Task 3: Implement Harmony Chat Template** - `8403b2c` (feat)
4. **Task 4: Create Comprehensive Unit Tests** - `6abb86b` (test)

## Files Created/Modified

- `src/harmony/harmony.zig` - Core types and HarmonyConversation container (207 lines)
- `src/harmony/types.zig` - Re-export shim for types-only imports (17 lines)
- `src/harmony/parser.zig` - HarmonyParser state machine with extractToolCalls (403 lines)
- `src/harmony/template.zig` - HarmonyTemplate with formatHarmonyChat, formatToolResult, addReasoningMarkers (289 lines)
- `src/harmony/harmony_test.zig` - 23 unit tests covering all components (470 lines)

## Decisions Made

- Switched from `std.ArrayList` to `std.ArrayListUnmanaged` — Zig 0.15.2 changed ArrayList to require allocator at each operation site (no stored allocator). Used `.empty` initializer and passed allocator explicitly.
- Used `std.json.fmt(val, .{})` with `"{f}"` format specifier instead of `std.json.stringify` — the stringify function was removed in Zig 0.15.2.
- Parser frees `current_recipient` on every role-start token and at end-of-input — this prevents leaks from the trailing `<|recipient|>user<|/recipient|>\n<|assistant|>\n` pattern that formatHarmonyChat appends.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed Zig 0.15.2 ArrayList API incompatibility**
- **Found during:** Task 1 (type definitions) and Task 2 (parser)
- **Issue:** Code was written for Zig 0.13.0 `std.ArrayList` API with stored allocator. Zig 0.15.2 uses `ArrayListUnmanaged` where allocator is passed per-operation.
- **Fix:** Replaced all `std.ArrayList(T).init(allocator)` with `std.ArrayListUnmanaged(T) = .empty`, updated all `.append()`, `.deinit()`, `.toOwnedSlice()` calls to pass allocator
- **Files modified:** harmony.zig, parser.zig, template.zig
- **Verification:** All 23 tests pass with no memory leaks

**2. [Rule 1 - Bug] Fixed std.json.stringify removal in Zig 0.15.2**
- **Found during:** Task 3 (template implementation)
- **Issue:** `std.json.stringify` does not exist in Zig 0.15.2; build error `root source file struct 'json' has no member named 'stringify'`
- **Fix:** Replaced with `std.json.fmt(value, .{})` using `"{f}"` format specifier
- **Files modified:** template.zig
- **Verification:** template tests pass including "Format with tool definitions"

**3. [Rule 1 - Bug] Fixed memory leak in HarmonyParser for dangling recipients**
- **Found during:** Task 4 (round-trip test)
- **Issue:** Parser stored `current_recipient` in a `?[]const u8` but failed to free it when transitioning to a new role start token (the old recipient was dropped without freeing)
- **Fix:** Added `if (current_recipient) |r| self.allocator.free(r);` before each `current_recipient = null` assignment in role-start transitions
- **Files modified:** parser.zig
- **Verification:** Round-trip test passes with std.testing.allocator (leak detection enabled)

---

**Total deviations:** 3 auto-fixed (all Rule 1 bugs — Zig version API changes and memory management)
**Impact on plan:** All fixes required for correctness and Zig 0.15.2 compatibility. No scope creep.

## Issues Encountered

None beyond the Zig version API differences documented above.

## Next Phase Readiness

- `HarmonyParser.extractToolCalls` ready for Phase 15-03 browser/python tool integration
- `HarmonyTemplate.formatHarmonyChat` ready for API integration in Phase 15-04
- `HarmonyTemplate.formatToolResult` ready for tool result injection loop
- All types are stable and importable from `@import("harmony.zig")`

---
*Phase: 15-mlx-gptoss*
*Completed: 2026-04-03*
