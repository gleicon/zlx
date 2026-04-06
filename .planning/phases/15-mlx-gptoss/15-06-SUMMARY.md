---
phase: 15-mlx-gptoss
plan: "06"
status: completed
completed: 2026-04-04
---

# Plan 15-06: Fix Compile-Breaking API Calls — Summary

## Changes Made

### src/tools/browser.zig (line 186-187)
**Before:**
```zig
const args = try std.json.parseFromSlice(Args, allocator, request.arguments, .{});
defer std.json.parseFree(Args, allocator, args);
```
**After:**
```zig
const args_parsed = try std.json.parseFromSlice(Args, allocator, request.arguments, .{});
defer args_parsed.deinit();
const args = args_parsed.value;
```

### src/tools/python.zig (line 210-211)
Same `parseFree` → `Parsed(T).deinit()` fix as browser.zig.

### src/api/tools.zig (lines 52-65 and 99-112)
Two fixes applied at both `browserToolHandler` and `pythonToolHandler`:

**parseFree fix:**
```zig
// Before:
const parsed = std.json.parseFromSlice(BrowserRequest, ...);
defer std.json.parseFree(BrowserRequest, self.allocator, parsed);
// After:
const parsed_result = std.json.parseFromSlice(BrowserRequest, ...);
defer parsed_result.deinit();
const parsed = parsed_result.value;
```

**stringifyAlloc fix:**
```zig
// Before:
const args = try std.json.stringifyAlloc(self.allocator, parsed, .{});
defer self.allocator.free(args);
// After:
var args_buf = std.ArrayList(u8).init(self.allocator);
defer args_buf.deinit();
try std.json.stringify(parsed, .{}, args_buf.writer());
const args = args_buf.items;
```

### build.zig (lines 113, 114, 117)
**Before:**
```zig
exe.addIncludePath(.{ .path = b.pathJoin(&.{ llama_cpp_path, "include" }) });
exe.addIncludePath(.{ .path = b.pathJoin(&.{ llama_cpp_path, "ggml", "include" }) });
exe.addLibraryPath(.{ .path = b.pathJoin(&.{ llama_build_path, "bin" }) });
```
**After:**
```zig
exe.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ llama_cpp_path, "include" }) });
exe.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ llama_cpp_path, "ggml", "include" }) });
exe.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ llama_build_path, "bin" }) });
```

### src/api/types_test.zig (bonus fix, not in original plan scope)
Also fixed `stringifyAlloc` in the test file to prevent compile errors during `zig build test`.

## Verification

- `grep -rn "parseFree|stringifyAlloc" src/` → no matches
- `grep -n "\.path = b\.pathJoin" build.zig` → no matches

## Notes

- No new functionality added — purely API compatibility fixes
- The `parseFree` API was replaced by `Parsed(T).deinit()` in Zig 0.13.0+
- The `stringifyAlloc` removal requires explicit ArrayList + stringify pattern
- The `LazyPath.path` field was renamed to `LazyPath.cwd_relative` in Zig 0.15.2
