# Deferred Items — Phase 15 mlx-gptoss

## Pre-existing Build Issues (Out of Scope for Phase 15-05)

### build.zig: `.path` field deprecated in Zig 0.15.2
- **File:** `build.zig` lines 113-118 (llama.cpp include/library paths)
- **Error:** `no field named 'path' in union 'Build.LazyPath'`
- **Cause:** Phase 14 used `.path = b.pathJoin(...)` which was valid in Zig 0.13.0 but renamed to `.cwd_relative` in Zig 0.15.2
- **Fix:** Replace `.path =` with `.cwd_relative =` for all `addIncludePath` and `addLibraryPath` calls that use `b.pathJoin`
- **Affected lines:** 113, 114, 117, 118 in build.zig
- **Status:** Pre-existing — NOT introduced by Phase 15-05

## Known Stubs (Phase 15-05)

### Tokenizer not wired
- **File:** `src/backends/mlx_gptoss_backend.zig` `tokenize()` function
- **Issue:** Returns empty slice — GPT-OSS tokenizer (tiktoken/Qwen tokenizer) not integrated
- **Impact:** `prompt_tokens` is always empty in `chat_gptoss.zig`, so context is not tokenized
- **Future:** Wire tiktoken or reuse MLX.zig's tokenizer for GPT-OSS vocabulary

### Transformer stream undefined
- **File:** `src/backends/mlx_gptoss_backend.zig` `load()` function
- **Issue:** `stream: undefined` passed to GPTOSSTransformer.init() — needs real MLX stream
- **Impact:** Will cause issues when real generation is triggered with weights
- **Future:** Initialize proper MLX GPU stream in load()

### GPT-OSS KV cache not integrated with TurboQuant
- **File:** `src/backends/backend.zig` `getKvCache()` for `.mlx_gptoss`
- **Issue:** Returns `null` — TurboQuant compression not yet wired for GPT-OSS
- **Future:** Phase 16 — integrate TurboQuant with GPT-OSS SlidingKVCache
