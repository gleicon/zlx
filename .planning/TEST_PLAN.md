# Test Implementation Plan

## Current Status
- 5 test files exist
- 44 tests passing
- 6 memory leaks in registry/manager tests

## Missing Test Coverage

### Critical (High Priority)
1. **API Handlers Tests** (`src/api/handlers_test.zig`)
   - Request parsing validation
   - Response formatting
   - Error handling paths
   - Chat completion logic

2. **Types Tests** (`src/api/types_test.zig`)
   - JSON serialization/deserialization
   - OpenAI compatibility
   - Edge cases (null fields, invalid data)

3. **Cache Tests** (`src/cache/prompt_cache_test.zig`)
   - LRU eviction
   - Cache hit/miss
   - Size limits
   - Persistence

### Medium Priority
4. **Streaming Tests** (`src/api/streaming_test.zig`)
   - SSE chunk formatting
   - Token streaming
   - UTF-8 boundary handling

5. **Metrics Tests** (`src/api/metrics_test.zig`)
   - Counter increments
   - Stats calculation
   - Thread safety

6. **Download Tests** (`src/download/*_test.zig`)
   - HuggingFace ID parsing
   - URL building
   - Progress tracking

### Fix Existing Issues
7. **Fix Memory Leaks**
   - Registry test: Free model metadata on cleanup
   - Manager test: Proper cleanup of resources

## Implementation Order

1. Fix existing memory leaks (30 min)
2. Create API types tests (1 hour) - Foundation for handler tests
3. Create Cache tests (1 hour) - Core feature
4. Create Handlers tests (1.5 hours) - Complex
5. Create Streaming tests (1 hour)
6. Create Metrics tests (45 min)
7. Create Download tests (1 hour)
8. Run full test suite and verify (30 min)

Total: ~6 hours
