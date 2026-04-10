---
phase: 15-mlx-gptoss
plan: "07"
subsystem: api
tags: [server, routing, gpt-oss, tools-api, gap-closure]
dependency_graph:
  requires: [15-05, 15-06]
  provides: [live-gptoss-chat-route, live-tools-routes]
  affects: [src/api/server.zig]
tech_stack:
  added: []
  patterns: [module-level-optional-vars, httpz-free-function-wrappers, optional-capture-instance-dispatch]
key_files:
  created: []
  modified:
    - src/api/server.zig
decisions:
  - "Used module-level optional vars (g_gptoss_backend, g_chat_gptoss_handler, g_tools_api, g_tool_executor) following the g_server pattern already established in server.zig"
  - "ChatGPTOSSHandler.handle called via optional capture (if (g_chat_gptoss_handler) |*handler|) to preserve instance method semantics"
  - "MLXGPTOSSBackend initialized with empty model_path for lazy loading — actual model set later via /v1/models/switch"
  - "Model dispatch peeks JSON body using req.arena for zero-leak temporary allocation"
metrics:
  duration: 12m
  completed: "2026-04-04"
  tasks: 2
  files: 1
---

# Phase 15 Plan 07: Wire ChatGPTOSSHandler and ToolsAPI into HTTP Server Summary

**One-liner:** Wired orphaned ChatGPTOSSHandler and ToolsAPI into server.zig routing so GPT-OSS chat requests and /v1/tools/* calls reach their handlers instead of falling to 404.

## What Was Done

### Task 1 — Module-level state, imports, and ToolsAPI routes

Added four imports to `src/api/server.zig`:

```zig
const chat_gptoss = @import("chat_gptoss.zig");
const tools_api_mod = @import("tools.zig");
const tool_executor_mod = @import("../tools/tool_executor.zig");
const mlx_gptoss_backend_mod = @import("../backends/mlx_gptoss_backend.zig");
```

Added four module-level optional vars following the `g_server` pattern:

```zig
var g_gptoss_backend: ?mlx_gptoss_backend_mod.MLXGPTOSSBackend = null;
var g_chat_gptoss_handler: ?chat_gptoss.ChatGPTOSSHandler = null;
var g_tools_api: ?tools_api_mod.ToolsAPI = null;
var g_tool_executor: ?tool_executor_mod.ToolExecutor = null;
```

Initialization in `Server.init()`:
- `MLXGPTOSSBackend.init(allocator, "", .{})` — empty model_path for lazy loading
- `ChatGPTOSSHandler.init(allocator, &g_gptoss_backend.?)` — pointer to the backend optional
- `ToolExecutor.init(allocator)`
- `ToolsAPI.init(allocator, &g_tool_executor.?)`

Cleanup in `Server.stop()`:
- `b.deinit()` on `g_gptoss_backend`
- `te.deinit()` on `g_tool_executor`
- All four vars set to null after stop

New routes registered:
- `POST /v1/tools/browser` → `handleBrowserTool`
- `OPTIONS /v1/tools/browser` → `handleOptions`
- `POST /v1/tools/python` → `handlePythonTool`
- `OPTIONS /v1/tools/python` → `handleOptions`

New module-level wrapper functions dispatch via optional capture to avoid pointer arithmetic on null:

```zig
fn handleBrowserTool(req: *httpz.Request, res: *httpz.Response) !void {
    if (g_tools_api) |*api| {
        try api.browserToolHandler(req, res);
    } else {
        res.status = 503;
        try res.json(.{ .@"error" = "Tools not initialized" }, .{});
    }
}
```

### Task 2 — ChatGPTOSSHandler dispatch in handleChatCompletions

The existing `handleChatCompletions` wrapper was replaced with a version that peeks the `model` field using `req.arena` for temporary allocation:

```zig
fn handleChatCompletions(req: *httpz.Request, res: *httpz.Response) !void {
    const body = req.body() orelse {
        try handlers.handleChatCompletions(req, res);
        return;
    };
    const ModelPeek = struct { model: []const u8 = "" };
    const peek = std.json.parseFromSlice(ModelPeek, req.arena, body, .{ .ignore_unknown_fields = true }) catch {
        try handlers.handleChatCompletions(req, res);
        return;
    };
    defer peek.deinit();

    if (std.mem.startsWith(u8, peek.value.model, "gpt-oss") or
        std.mem.startsWith(u8, peek.value.model, "gptoss"))
    {
        if (g_chat_gptoss_handler) |*handler| {
            try handler.handle(req, res);   // INSTANCE method call
        } else {
            res.status = 503;
            try res.json(.{ .@"error" = "GPT-OSS handler not initialized" }, .{});
        }
    } else {
        try handlers.handleChatCompletions(req, res);
    }
}
```

Key detail: `handler.handle(req, res)` is called on `*ChatGPTOSSHandler` obtained via the optional capture `|*handler|`. This is required because `ChatGPTOSSHandler.handle` is an instance method that needs `self: *ChatGPTOSSHandler`.

## Deviations from Plan

None — plan executed exactly as written. All interface signatures matched the plan's documented contracts.

## Known Stubs

None introduced in this plan. The empty `model_path = ""` in `MLXGPTOSSBackend.init` is intentional and documented — the model is loaded lazily via `/v1/models/switch`.

## Self-Check: PASSED

- `src/api/server.zig` modified: confirmed present
- Commit `2084385` present in git log
- `grep -c "v1/tools/browser"` returns 6 (route + options + wrapper + comment = non-zero)
- `grep -c "g_chat_gptoss_handler"` returns 5 (declaration, init, cleanup, dispatch, null-set)
- `grep -n "handler.handle"` finds line 168 confirming instance method dispatch
