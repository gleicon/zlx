---
phase: 05-multi-model-support
plan: 02
subsystem: models
tags: [model-manager, hot-swap, memory-checking]
dependency_graph:
  requires: [05-01]
  provides: [05-03]
  affects: [src/api/handlers.zig, src/main.zig, src/api/server.zig]
tech_stack:
  added: []
  patterns: [global-manager-pattern, atomic-counters, condition-variables]
key_files:
  created: [src/models/manager.zig]
  modified: [src/api/handlers.zig, src/api/server.zig, src/api/types.zig, src/main.zig]
decisions:
  - "Memory safety: 20% margin added to memory requirements"
  - "Generation tracking: Atomic counter with condition variable for graceful shutdown"
  - "Model switching: Blocked until active generations complete"
  - "API errors: Return specific error codes (404 model_not_found, 503 insufficient_memory)"
  - "Switch endpoint: Explicit POST /v1/models/switch for programmatic control"
  - "Auto-switch: Request with different model triggers automatic switch"
metrics:
  duration_minutes: 55
  commits: 3
  files_created: 1
  files_modified: 4
  tests_added: 0
---

# Phase 05 Plan 02: Model Manager Summary

## One-Liner
Implemented model manager with hot-swap capability, memory budget validation, and explicit/ automatic model switching with graceful generation shutdown.

## What Was Built

### Model Manager (`src/models/manager.zig`)
- **ModelManager struct**: Manages model lifecycle with thread-safe operations
- **Memory checking**: `canLoadModel()` validates available memory with 20% safety margin
- **Hot-swap**: `switchModel()` loads new model, waits for active generations, unloads old
- **Generation tracking**: Atomic counter with condition variable for graceful transitions
- **Global instance**: `initGlobalManager`, `deinitGlobalManager`, `getGlobalManager`

### Memory Detection (macOS)
```zig
// Uses sysctl to get total system memory
const CTL_HW = 6;
const HW_MEMSIZE = 24;
// Returns 70% of total as available (conservative estimate)
```

### Automatic Model Switching in handlers.zig
**Before:**
- Request with different model → 404 error

**After:**
- Request with different model → automatic switch
- Check registry for model availability
- Validate memory with 20% margin
- Switch with timing logging
- Continue with new model

**Error handling:**
- 404: Model not in registry
- 503: Insufficient memory
- 500: Switch failed

### Explicit Switch Endpoint
**POST /v1/models/switch**

Request:
```json
{
  "model": "qwen2.5-coder-7b"
}
```

Response (success):
```json
{
  "status": "success",
  "model": "qwen2.5-coder-7b",
  "previous_model": "qwen2.5-coder-1.5b",
  "duration_ms": 1500
}
```

Response (error):
```json
{
  "error": {
    "message": "Insufficient memory to load model",
    "type": "insufficient_memory"
  }
}
```

### Generation Tracking Integration
- `startGeneration()` called when generation begins
- `endGeneration()` called when generation ends (via defer)
- Switch waits for counter to reach 0
- Condition variable signals completion

### Initialization Order in main.zig
```zig
1. Parse CLI args
2. Init model registry (scans directories)
3. Init model manager (references registry)
4. Load initial model via manager.switchModel()
5. Init metrics
6. Start server
```

## Thread Safety Strategy

### Active Generation Tracking
```zig
active_generations: std.atomic.Value(u32),
generation_condition: std.Thread.Condition,
switch_mutex: std.Thread.Mutex,
```

**Pattern:**
1. `startGeneration()`: `fetchAdd(1)` - atomic increment
2. `endGeneration()`: `fetchSub(1)` + `broadcast()` if reaches 0
3. `switchModel()`: Wait on condition variable while > 0

### Switch Mutex
- Prevents concurrent switches
- Held during entire switch operation
- Other requests block or get 503

## API Changes

### New Endpoint
- `POST /v1/models/switch` - Explicit model switching

### Modified Behavior
- `POST /v1/chat/completions` - Auto-switches if model differs
- Both streaming and non-streaming track generations

## Deviations from Plan

### Minor Adjustments
1. **Memory API**: Used raw sysctl constants instead of std.c (compatibility)
2. **Context Update**: Used global_context directly instead of passing around context pointer
3. **Testing**: Manager tests compile as part of main build due to cross-dependencies

## Verification

- ✅ `zig build` compiles without errors
- ✅ `zig build test` passes
- ✅ Model switching logic implemented
- ✅ Memory checking with margin
- ✅ Generation tracking in handlers
- ✅ POST /v1/models/switch endpoint added
- ✅ Manager integrated into main initialization

## Known Limitations

1. **Memory Detection**: Uses conservative 70% of total RAM estimate
2. **Architecture Detection**: Based on parameter count, not config.json
3. **No Streaming Switch**: Stream requests don't support mid-stream switching
4. **Single-threaded Switch**: Other requests block during model switch

## Next Steps

Plan 05-03 will add:
- Memory tracking with component breakdown
- Enhanced health checks with detailed status
- Prometheus-compatible metrics export
- Memory budget enforcement

## Commits

1. `a61f001` - Create model manager
2. `e5c2466` - Integrate manager and automatic switching
3. `3b85cc4` - Add switch endpoint and wire into main
