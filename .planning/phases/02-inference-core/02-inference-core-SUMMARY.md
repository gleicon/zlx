# Phase 2 Plan: Inference Core - Summary

**Phase:** 02-inference-core
**Plan:** PLAN.md
**Subsystem:** inference
**Tags:** inference, mlx, gpu, tokenizer, generator

**Dependency Graph:**
- Requires: Phase 1 (Foundation & Build)
- Provides: Phase 3 (HTTP Server API)
- Affects: src/inference/, src/main.zig, src/mlx.zig/

**Tech Stack Added:**
- MLX.zig Transformer API integration
- GenerationState iterator pattern
- Tokenizer wrapper (BPE/tiktoken via MLX.zig)
- Model loading infrastructure
- CLI harness with flag parsing

**Key Files:**
- **Created:**
  - `src/inference/mod.zig` - Public API exports
  - `src/inference/loader.zig` - Model loading from disk
  - `src/inference/tokenizer.zig` - Tokenizer wrapper (stub)
  - `src/inference/generator.zig` - GenerationState with next() iterator
- **Modified:**
  - `src/main.zig` - CLI harness with flag parsing
  - `src/mlx.zig/src/mlx.zig` - Zig 0.15 compatibility fixes
  - `src/mlx.zig/src/utils.zig` - Zig 0.15 compatibility fixes

**Decisions Made:**

1. **MLX.zig Compatibility Strategy**: MLX.zig was written for Zig 0.13. Required extensive porting for Zig 0.15:
   - Type info field naming changed (.Struct → .@"struct", .Pointer → .pointer, etc.)
   - ArrayList API completely changed (init → .empty, append needs allocator)
   - std.mem tokenization functions renamed
   - std.io.getStdOut() moved to std.fs.File.stdout()

2. **Tokenizer Approach**: Created minimal stub tokenizer since full MLX.zig tokenizer needs BPE implementation porting. The stub uses byte-level encoding for testing.

3. **GenerationState Pattern**: Implemented true iterator pattern with `next() → ?Token` that yields one token at a time, managing MLX arrays and KV cache internally.

4. **CLI Design**: Supports both interactive mode (read from stdin) and single-shot mode (prompt as argument).

**Deviations from Plan:**

### Auto-fixed Issues (Rule 3 - Blocking)

**1. [Rule 3 - Blocking] MLX.zig Zig 0.15 Compatibility**
- **Found during:** Build attempts
- **Issue:** MLX.zig written for Zig 0.13, incompatible with 0.15
- **Fix:** Systematic port of type info access, ArrayList API, mem functions
- **Files modified:** src/mlx.zig/src/mlx.zig, src/mlx.zig/src/utils.zig
- **Commit:** 16c43d8

**2. [Rule 2 - Missing Critical] Tokenizer Implementation**
- **Found during:** Build attempts
- **Issue:** Full MLX.zig tokenizer has complex BPE that needs porting
- **Fix:** Created minimal stub tokenizer for testing
- **Files modified:** src/inference/tokenizer.zig
- **Commit:** 27aeda0

**3. [Rule 1 - Bug] stdout API Changes**
- **Found during:** Build attempts
- **Issue:** std.io.getStdOut() removed in Zig 0.15
- **Fix:** Use std.fs.File.stdout() or std.debug.print
- **Files modified:** src/main.zig
- **Commit:** 27aeda0

### Known Stubs

**1. Tokenizer is Byte-Level Stub**
- **File:** src/inference/tokenizer.zig
- **Line:** 38-58 (encode/decode methods)
- **Reason:** Full BPE tokenizer in MLX.zig needs porting from 0.13 to 0.15
- **Resolution:** Port tokenizer.zig BPE implementation in future milestone

### Incomplete Items

1. **GPU Verification**: Cannot verify GPU usage without actual model weights and full tokenizer
2. **Concurrent Request Serialization**: Mutex implemented but not tested (HTTP server in Phase 3)
3. **Full Tokenizer**: BPE implementation stubbed; needs MLX.zig tokenizer port completion

**Performance Metrics:**
- **Duration:** 1 session (significant MLX.zig porting required)
- **Tasks completed:** 7/7 attempted, with deviations noted above
- **Files created:** 4
- **Files modified:** 3 (plus MLX.zig submodule)
- **Commits:** 2

**Success Criteria Assessment:**

1. ✅ `./zig-out/bin/zlx --model qwen2.5-coder-1.5b --help` works - shows help
2. ⚠️ Token generation on Metal GPU - requires full tokenizer and model weights
3. ✅ `GenerationState.next()` yields one token per call - implemented
4. ⚠️ Concurrent request mutex - implemented but not tested (Phase 3)
5. ✅ CLI flags --model, --port, --max-kv-size, --max-tokens, --temperature work

**Self-Check:**
- ✅ Created files exist: src/inference/{mod,loader,tokenizer,generator}.zig
- ✅ Binary builds: zig-out/bin/zlx (17MB)
- ✅ Help works: --help shows usage
- ⚠️ Full inference test: needs model weights and tokenizer completion

**Next Steps for Phase 3:**
1. Complete tokenizer BPE porting (or use pre-tokenized input)
2. Download test model weights to verify GPU generation
3. Implement HTTP server endpoints using httpz
4. Test concurrent request handling with mutex
