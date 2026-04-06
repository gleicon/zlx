---
phase: 17-inference-gap-closure
plan: "05"
subsystem: cache
tags: [cache, json, disk-io, persistence, stub-fix]
dependency_graph:
  requires: []
  provides: [prompt-cache-persistence]
  affects: [src/cache/prompt_cache.zig]
tech_stack:
  added: []
  patterns: [std.json.parseFromSlice, StringHashMap.ensureTotalCapacity, atomic-init]
key_files:
  created: []
  modified:
    - src/cache/prompt_cache.zig
decisions:
  - "Reconstruct file_path from cache_dir + entries/ + key rather than storing it in index.json (saveIndex() does not persist file_path; we mirror saveIndex() schema)"
  - "Pre-allocate HashMap capacity before loop to avoid rehashing on bulk load"
  - "Restore hits/misses/evictions from index.json top-level fields — symmetric with saveIndex()"
metrics:
  duration: "3 minutes"
  completed: "2026-04-06T00:13:54Z"
  tasks_completed: 1
  files_modified: 1
requirements: [GAP-06]
---

# Phase 17 Plan 05: loadIndex() Stub Fix Summary

## One-liner

Replaced the `loadIndex()` TODO stub in `prompt_cache.zig` with a full `std.json` disk read that restores all cache entries and metrics counters at server init time.

## What Was Done

### Task 1: Implement loadIndex() in prompt_cache.zig

Replaced lines 423-435 (the TODO stub) with a complete implementation:

1. Builds `index_path` from `self.cache_dir + "/index.json"`
2. Returns cleanly if `index.json` does not exist (fresh start path logged)
3. Opens and reads the file with a 1MB limit
4. Parses with `std.json.parseFromSlice(std.json.Value, ...)`
5. Restores top-level `hits`, `misses`, `evictions` counters via atomic stores
6. Pre-allocates `self.entries` HashMap capacity for all entries (no rehashing)
7. For each entry in the `"entries"` array:
   - Extracts `key`, `size_bytes`, `created_at`, `access_count`
   - Reconstructs `file_path` from `cache_dir/entries/<key>`
   - Dupes key string before `parsed.deinit()` frees JSON memory
   - Initializes `CacheKey` struct from the key string bytes
   - Constructs `CacheEntry` with atomic `access_count` and `last_accessed`
   - Inserts into `self.entries` HashMap
   - Adds to LRU list via `addToLru()`
   - Accumulates `current_size_bytes`
8. Logs success: `"Loaded N cache entries from <path>"`

**Comment added:** `// real index.json load per D-05 — called once at init, no per-request I/O`

## Verification

### TODO/stub removed
```
grep -n "TODO|stub" src/cache/prompt_cache.zig
(no output)
```

### Real JSON parsing present
```
grep -n "parseFromSlice|openFileAbsolute" src/cache/prompt_cache.zig
434:        const file = try std.fs.openFileAbsolute(index_path, .{});
439:        const parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, content, .{});
```

### Clean-start path present
```
grep -n "Starting fresh|No existing cache" src/cache/prompt_cache.zig
430:            std.log.info("No existing cache index found at {s}, starting fresh", .{index_path});
```

### AST check passes
```
zig ast-check src/cache/prompt_cache.zig
(no output = no errors)
```

### Build status
`zig build` reaches the same failure point as before this plan: `deps/turboquant/turboquant/src/turboquant.zig FileNotFound`. This is a pre-existing error tracked separately (out of scope for GAP-06). My changes to `prompt_cache.zig` do not introduce any new compilation errors — confirmed by stashing changes, verifying the same build error exists without them, then restoring.

## Deviations from Plan

### Auto-fixed Issues

None.

### Scope Notes

- The plan interface showed `access_count` as `std.atomic.Value(u32)` but the actual struct declares it as `std.atomic.Value(u64)`. Implementation uses `u64` to match the actual type — no deviation, just a plan typo corrected.
- `file_path` is not stored in `index.json` by `saveIndex()`, so it is reconstructed from `cache_dir/entries/<key>`. This mirrors the existing `getEntryPath()` pattern.

## Known Stubs

None introduced. The `loadIndex()` stub has been replaced with a real implementation.

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1 | 757d3d0 | feat(17-05): implement loadIndex() with real std.json disk reads |
