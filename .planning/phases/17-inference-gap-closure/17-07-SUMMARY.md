---
phase: 17-inference-gap-closure
plan: "07"
subsystem: cache
tags: [cache, json, disk-io, prompt-cache]
dependency_graph:
  requires: []
  provides: [GAP-06]
  affects: [src/cache/prompt_cache.zig]
tech_stack:
  added: []
  patterns: [std.json.parseFromSlice, std.fs.openFileAbsolute, atomic-value-init]
key_files:
  modified:
    - src/cache/prompt_cache.zig
decisions:
  - "file_path defaulted to empty string — saveIndex() does not write file_path to index.json, so no restore is possible; entries are metadata-only after restart"
  - "access_count is u64 (not u32 as plan template suggested) — matched to actual CacheEntry field type"
  - "addToLru() called for each loaded entry so LRU eviction tracking is consistent from startup"
  - "ensureTotalCapacity uses @intCast(len) — usize to u32 cast required by StringHashMap API"
metrics:
  duration: "< 5 minutes"
  completed: "2026-04-06"
  tasks_completed: 2
  files_modified: 1
---

# Phase 17 Plan 07: loadIndex() Implementation Summary

**One-liner:** Real `std.json` disk reads in `loadIndex()` replace the TODO stub, populating `self.entries` at init from `index.json`.

## What Was Done

Replaced the TODO stub in `PromptCache.loadIndex()` with a full implementation that:

1. Opens `index.json` with `std.fs.openFileAbsolute`
2. Reads content with `file.readToEndAlloc(self.allocator, 1024 * 1024)`
3. Parses with `std.json.parseFromSlice(std.json.Value, ...)`
4. Iterates `entries` array, extracting `key`, `size_bytes`, `created_at`, `access_count`
5. Dupes all string values before `parsed.deinit()` frees them
6. Constructs `CacheEntry` with loaded values; `file_path` defaults to `""` (not in JSON)
7. Calls `self.entries.put(key_copy, entry)` and `self.addToLru(key_copy)` per entry
8. Logs: `"Loaded {d} cache entries from {s}"`

The existing "no index.json" early-return path (logging `"starting fresh"`) was preserved unchanged.

## Verification

```
grep -n "TODO" src/cache/prompt_cache.zig
(no output — TODO stub gone)

grep -n "parseFromSlice" src/cache/prompt_cache.zig
440:        const parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, content, .{});

grep -n "openFileAbsolute" src/cache/prompt_cache.zig
434:        const file = try std.fs.openFileAbsolute(index_path, .{});

grep -n "Loaded.*cache entries\|starting fresh" src/cache/prompt_cache.zig
430:            std.log.info("No existing cache index found, starting fresh", .{});
480:        std.log.info("Loaded {d} cache entries from {s}", .{ entries_arr.items.len, index_path });

zig build: PASS (exit 0)
```

## CacheEntry Fields: Loaded vs Defaulted

| Field | Source | Notes |
|-------|--------|-------|
| `key` | set to `undefined` | Map key (the string) is the canonical key; `CacheEntry.key` is a `CacheKey` struct not present in JSON |
| `file_path` | `""` (default) | Not written by `saveIndex()`; KV state files not tracked in index |
| `size_bytes` | loaded from JSON | `i64` → `u64` via `@intCast` |
| `created_at` | loaded from JSON | `i64` direct |
| `last_accessed` | `std.time.timestamp()` | Not stored; reset to now on load |
| `access_count` | loaded from JSON | `i64` → `u64` via `@intCast` |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `ensureTotalCapacity` type mismatch**
- **Found during:** Task 2 (zig build)
- **Issue:** `entries_arr.items.len` is `usize`; `StringHashMap.ensureTotalCapacity` expects `u32`
- **Fix:** `@intCast(entries_arr.items.len)`
- **Files modified:** src/cache/prompt_cache.zig
- **Commit:** affbf13

**2. [Plan Template Correction] `access_count` is `u64` not `u32`**
- **Found during:** Task 1 (reading CacheEntry)
- **Issue:** Plan interface template used `u32` for `access_count`; actual field is `std.atomic.Value(u64)`
- **Fix:** Used `u64` type throughout
- **No extra commit** — corrected in main implementation

## Known Stubs

None introduced. `CacheEntry.key` is set to `undefined` for loaded entries because the key string is the HashMap key (not the `CacheKey` struct). This is consistent — `lookup()` uses the map key string, not `entry.key`.

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| Task 1+2 | affbf13 | feat(17-07): implement loadIndex() with real std.json disk reads |

## Self-Check: PASSED

- File exists: `src/cache/prompt_cache.zig` — FOUND
- Commit affbf13 exists — FOUND
- No TODO in loadIndex area — CONFIRMED
- `zig build` passes — CONFIRMED
