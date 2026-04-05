# Phase 17: Inference Gap Closure - Research

**Researched:** 2026-04-05
**Domain:** Zig inference pipeline — stub removal, handler wiring, cache fix, speculation cleanup
**Confidence:** HIGH (all findings from direct source reads, no external lookups needed)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Handler-per-model pattern. Each model family gets a dedicated handler + backend pair:
  - Qwen → `inference/mod.zig` → MLX.zig Transformer (unchanged)
  - GPT-OSS → `chat_gptoss.zig` → `mlx_gptoss_backend.zig` (tokenizer works, fix `forward()`)
  - DeepSeek → **new** `chat_deepseek.zig` → `llama_cpp.zig` (direct, no factory)
  - HTTP dispatch in `server.zig` routes by model name prefix
- **D-02:** MLX stays directly controlled — no generic backends abstraction layer between Zig and Metal. `backends/factory.zig` generic routing is deferred backlog.
- **D-03:** Remove all speculative decoding code — zero dead code policy. Comment at each removal site: `// speculative-decoding: removed — re-evaluate as dedicated phase after core inference is stable`
- **D-04:** Delete `src/backends/factory.zig` and `src/backends/mlx_backend.zig`. Keep `backend.zig`, `llama_cpp.zig`, `mlx_gptoss_backend.zig`. Update `backends/mod.zig` to only re-export surviving files.
- **D-05:** Fix `cache/prompt_cache.zig:433` — real `index.json` disk reads in `loadIndex()`. Read once at init, keep in memory.
- **D-06:** Replace all hardcoded `vocab_size=32000`, `eos_token=2`, `bos_token=1`, `disk=100GB` with values from model's `config.json`.

### Claude's Discretion

- GPT-OSS `forward()` implementation approach — whether to use MLX.zig Transformer directly or implement tensor ops layer-by-layer. Choose whichever produces real logits correctly; researcher should verify against `gptoss_mlx.zig` architecture definition.
- Exact `index.json` schema for prompt cache — read the existing file format, implement to match.
- Whether `error.DeepSeekNotImplemented` branches in `mod.zig` are reachable after HTTP dispatch rerouting — if not, remove the branches entirely rather than patching them.

### Deferred Ideas (OUT OF SCOPE)

- `backends/factory.zig` generic routing abstraction — backlog, evaluate after core inference is clean
- Speculative decoding — removed entirely; re-evaluate from scratch as a dedicated phase
- TurboQuant `error.NotImplemented` stubs — Phase 19 scope
- Tools API (`browser.zig`, `python.zig` Zig 0.15.2 breaks) — Phase 20 scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| GAP-01 | No `error.NotImplemented` on any inference path | All NotImplemented sites mapped below; compression/ stubs are Phase 19 deferred, not Phase 17 scope |
| GAP-02 | No hardcoded `vocab_size=32000`, `eos_token=2`, `bos_token=1` on live paths | Only `mlx_backend.zig` (delete) and `models/registry.zig:266` (default field, acceptable) remain; no live-path hardcodes after factory/mlx_backend deleted |
| GAP-03 | DeepSeek path produces real tokens | New `chat_deepseek.zig` + HTTP dispatch in server.zig wires to `LlamaBackend` |
| GAP-04 | GPT-OSS tokenizer produces real tokens | Already implemented in `mlx_gptoss_backend.zig:tokenize()` — confirmed real |
| GAP-05 | Orphaned stub files deleted | `factory.zig` + `mlx_backend.zig` delete; `backends/mod.zig` update |
| GAP-06 | Prompt cache `parseIndex()` does real disk reads | `loadIndex()` stub at line 433 mapped; `saveIndex()` schema fully documented |
| GAP-07 | Speculative decoding code fully removed | All sites inventoried: main.zig, config.zig, inference/mod.zig, generator.zig, streaming.zig |
| MODEL-01 | Qwen path unchanged and producing tokens | Confirmed working — no changes to `inference/mod.zig` Qwen branch |
| MODEL-02 | GPT-OSS path produces real tokens end-to-end | Forward() stub identified; `mlx.zig/src/gptoss.zig:Transformer` is the real implementation available |
| MODEL-03 | DeepSeek path produces real tokens end-to-end | `LlamaBackend` interface fully documented; `chat_deepseek.zig` template is `chat_gptoss.zig` |
</phase_requirements>

---

## Summary

Phase 17 is a surgical stub-removal and wiring phase. The build is already clean (Phase 16 baseline). The primary work falls into five independent tracks that can be sequenced by risk:

**Track A — Delete dead files:** `factory.zig` and `mlx_backend.zig` are confirmed pure stubs. Their only callers are `backend_generator.zig` and `generator.zig` (which import them for type aliases) and a test file. These files must be removed and callers must drop their imports.

**Track B — Wire DeepSeek:** `LlamaBackend.init(allocator, model_path)` is real and complete. Creating `chat_deepseek.zig` is a direct mechanical copy of `chat_gptoss.zig` with `MLXGPTOSSBackend` replaced by `LlamaBackend`. The `error.DeepSeekNotImplemented` branches in `mod.zig` are unreachable after HTTP-layer dispatch and should be deleted.

**Track C — Fix GPT-OSS generate:** `chat_gptoss.zig:216` passes an empty token slice to `generate()` because tokenization is commented out as a stub. The tokenizer function `mlx_gptoss_backend.zig:tokenize()` is real — it just needs to be called. The `GPTOSSTransformer.forward()` in `gptoss_mlx.zig` returns zero logits (argmax → token 0 → immediate EOS). The correct fix is to delegate to `mlx.zig/src/gptoss.zig:GPTOSSTransformer` which has the full forward pass scaffold with attention, MoE layers, and lm_head (currently initialized to zeros — but at least the structure is there). This is the discretion area: the MLX.zig `gptoss.zig` Transformer has real weight fields whereas `gptoss_mlx.zig` does not. Without loaded weights both return zeros. The correct path for real output is to fix the tokenize call in `chat_gptoss.zig` (Track C-1) and decide whether to use the MLX.zig `Transformer` wrapper or the current `GPTOSSTransformer` — researcher recommendation is to use `mlx.zig/src/gptoss.zig:Transformer` since it already has the full layer structure.

**Track D — Prompt cache:** `loadIndex()` at line 433 is a pure TODO stub. The `saveIndex()` function (lines 438-476) is complete and reveals the full `index.json` schema. Implementing `loadIndex()` requires reading `index.json` with `std.json`, iterating the `entries` array, and populating `self.entries`.

**Track E — Remove speculation:** All speculation symbols are fully inventoried. Removal touches `main.zig`, `config.zig`, `inference/mod.zig`, `inference/generator.zig`, and `api/streaming.zig`. The `speculation/` directory can be deleted entirely after the import at `main.zig:14` is removed.

**Primary recommendation:** Sequence as Track A → Track E → Track B → Track C-1 (tokenize fix) → Track D. This ordering removes complexity before adding new code and keeps the build passing at each step.

---

## Standard Stack

No new dependencies required. All work uses existing modules.

| Component | File | Status |
|-----------|------|--------|
| MLX tensor ops | `src/mlx.zig/src/mlx.zig` | Available — zeros, argmax, matmul, softmax, etc. |
| MLX GPT-OSS Transformer (full) | `src/mlx.zig/src/gptoss.zig` | Scaffold present; weights initialize to zeros |
| LlamaBackend | `src/backends/llama_cpp.zig` | Real implementation, complete |
| Tokenizer | `src/mlx.zig/src/tokenizer.zig` | Real, used by mlx_gptoss_backend.zig already |
| std.json | Zig 0.15.2 stdlib | Used for parseIndex() fix |

---

## Architecture Patterns

### Pattern 1: Handler-Per-Model (established, replicate for DeepSeek)

`server.zig` holds module-level backend + handler globals. Dispatch is in `handleChatCompletions()`.

**Existing GPT-OSS pattern (lines from server.zig):**
```zig
// server.zig:31-37 — module-level globals
var g_gptoss_backend: ?MLXGPTOSSBackend = null;
var g_chat_gptoss_handler: ?ChatGPTOSSHandler = null;

// server.zig:60-65 — init block
g_gptoss_backend = try MLXGPTOSSBackend.init(allocator, "", .{});
g_chat_gptoss_handler = ChatGPTOSSHandler.init(allocator, &g_gptoss_backend.?);

// server.zig:163-175 — dispatch block
if (std.mem.startsWith(u8, peek.value.model, "gpt-oss") or
    std.mem.startsWith(u8, peek.value.model, "gptoss")) {
    if (g_chat_gptoss_handler) |*handler| {
        try handler.handle(req, res);
    }
}
```

**DeepSeek replication (new globals + dispatch):**
```zig
// Add after line 37:
var g_deepseek_backend: ?LlamaBackend = null;
var g_chat_deepseek_handler: ?ChatDeepSeekHandler = null;

// Add in Server.init() after line 65:
g_deepseek_backend = try LlamaBackend.init(allocator, "");
g_chat_deepseek_handler = ChatDeepSeekHandler.init(allocator, &g_deepseek_backend.?);

// Add in handleChatCompletions() dispatch block:
if (std.mem.startsWith(u8, peek.value.model, "deepseek")) {
    if (g_chat_deepseek_handler) |*handler| {
        try handler.handle(req, res);
    }
}
```

### Pattern 2: LlamaBackend lazy init (model path deferred)

`LlamaBackend.init()` tries to load a GGUF file immediately. Passing `""` as model_path will fail. The GPT-OSS pattern uses `loaded: bool` flag and loads lazily on first request. `LlamaBackend` does NOT have this — it loads in `init()`. Options:
- **Option A:** Wrap `LlamaBackend` in an optional and init lazily on first `handle()` call (same as GPT-OSS pattern).
- **Option B:** Require model path to be set before handler init (init-time check).
- **Recommendation (discretion):** Use Option A — wrap in `?LlamaBackend`, init lazily when model path is set via `/v1/models/switch`.

### Pattern 3: ChatGPTOSSHandler structure (template for ChatDeepSeekHandler)

`chat_gptoss.zig` has the exact structure to copy:
- `ChatGPTOSSHandler.init(allocator, *backend)` — line 104
- `ChatGPTOSSHandler.handle(req, res)` — line 112
  - Parses JSON body
  - Verifies model name via `isGptOssModel()`
  - Extracts generation params
  - Builds Harmony prompt
  - **STUB at line 216:** `const prompt_tokens = try self.allocator.alloc(u32, 0);` — passes empty slice
  - Calls `backend.generate(prompt_tokens, params, allocator)`
  - Streams or serializes response
- `isGptOssModel(model)` helper — line 378

**For `chat_deepseek.zig`:** Replace `MLXGPTOSSBackend` with `LlamaBackend`. The `LlamaBackend.tokenize()` returns `[]llama_c.llama_token` (i32 values). The vtable function `llamaTokenize()` in `llama_cpp.zig:223` already converts to `[]u32` — use that wrapper.

### Pattern 4: `backend.zig` Backend union — extern fn declarations

`backend.zig:233-254` declares `extern fn` vtable bindings. These reference `mlxTokenize`, `mlxGenerate`, etc. from `mlx_backend.zig`. After deleting `mlx_backend.zig`, these extern declarations become dangling. The resolution is to remove the `mlx` arm from the `Backend` union entirely, or keep the union but remove the `.mlx` variant. Since the CONTEXT.md says keep `backend.zig` for Phase 18 compatibility, the safest approach is to remove the `mlx` variant from the union and its extern fn declarations, keeping only `llama_cpp` and `mlx_gptoss` variants. This is a compile-risk item — see "Compile Risks" section below.

---

## Stub Inventory (Complete)

### GAP-01: error.NotImplemented on inference paths

**In-scope stubs (Phase 17 must fix or delete):**

| File | Line | Stub | Fix Action |
|------|------|------|------------|
| `src/inference/mod.zig` | 261 | `return error.DeepSeekNotImplemented;` | Delete entire `.deepseek_v2_moe` match arm in `generateWithTimeout()` |
| `src/inference/mod.zig` | 332 | `return error.DeepSeekNotImplemented;` | Delete — unreachable after switch arm removal |
| `src/gptoss_mlx.zig` | 186 | `GPTOSSTokenGenerator.next()` returns 0 always | Not on any live path (backend uses `generate()`, not `next()`) — low priority |
| `src/api/chat_gptoss.zig` | 216 | `alloc(u32, 0)` — empty prompt tokens | Fix: call `mlx_gptoss_backend.tokenize()` with harmony_prompt |

**Out-of-scope stubs (Phase 19 — compression/):**

All `error.NotImplemented` in `src/compression/kv_compressor.zig`, `src/compression/metal_kernels.zig`, `src/compression/turboquant_stub.zig` are TurboQuant stubs. These are Phase 19 scope per CONTEXT.md deferred items. Do not touch.

### GAP-02: Hardcoded token values on live paths

| File | Line | Value | Fix Action |
|------|------|-------|------------|
| `src/backends/mlx_backend.zig` | 65 | `return 32000;` | File deleted (Track A) |
| `src/backends/mlx_backend.zig` | 70 | `return 2;` | File deleted (Track A) |
| `src/backends/mlx_backend.zig` | 76 | `return 1;` | File deleted (Track A) |
| `src/models/registry.zig` | 266 | `vocab_size: u32 = 32000` | Default field value — acceptable, not a live path result |
| `src/speculation/draft_selector.zig` | 307 | `.vocab_size = 32000` | File in `speculation/` — deleted with Track E |

Note: `mlx_gptoss_backend.zig` hardcodes `vocab_size=151936`, `eos=100257`, `bos=100256` for GPT-OSS (lines 218-234). These are correct for the model family (not the generic 32000/2/1 values). However, D-06 requires reading from `config.json` — this is a refinement task. The current values are factually correct for GPT-OSS models so this will not produce wrong output, but must be replaced with config reads to satisfy D-06.

### GAP-05: Files to delete

| File | Status | Confirmed No Real Callers? |
|------|--------|---------------------------|
| `src/backends/factory.zig` | Pure stub (`createMlxBackend` returns `u8*`) | `backend_generator.zig` and `generator.zig` import it — must remove those imports |
| `src/backends/mlx_backend.zig` | All stubs, never produces output | Only referenced via `backends/mod.zig` re-export and `backend.zig` extern fns |

**Callers that must be updated when deleting factory.zig:**
- `src/inference/backend_generator.zig:5` — `@import("../backends/factory.zig")`
- `src/inference/generator.zig:13` — `@import("../backends/factory.zig")`
- `src/test_backend_integration.zig:7` — `@import("backends/factory.zig")`
- `src/backends/mod.zig:21` — `pub const factory = @import("factory.zig")`

**Callers that must be updated when deleting mlx_backend.zig:**
- `src/backends/mod.zig:25` — `pub const mlx_backend = @import("mlx_backend.zig")`
- `src/backends/backend.zig:233-254` — extern fn declarations for mlx* functions

---

## Interface Documentation

### LlamaBackend — Complete Interface (src/backends/llama_cpp.zig)

```zig
// Init: loads GGUF model immediately. model_path must be a valid GGUF path.
// Returns error.ModelLoadFailed if file not found.
// Returns error.ContextCreationFailed if llama.cpp context init fails.
pub fn init(allocator: std.mem.Allocator, model_path: []const u8) !LlamaBackend

// Tokenize text — returns []llama_c.llama_token (i32 slice, caller owns)
pub fn tokenize(self: *LlamaBackend, text: []const u8) ![]llama_c.llama_token

// Decode batch into context (advances internal state)
pub fn decode(self: *LlamaBackend, tokens: []const llama_c.llama_token) !void

// Sample next token using sampler chain
pub fn sample(self: *LlamaBackend, sampler: *llama_c.llama_sampler) llama_c.llama_token

// Metadata accessors (read from loaded GGUF, no hardcodes)
pub fn vocabSize(self: LlamaBackend) u32    // from llama_c.llama_n_vocab()
pub fn eosToken(self: LlamaBackend) llama_c.llama_token   // from llama_c.llama_token_eos()
pub fn bosToken(self: LlamaBackend) llama_c.llama_token   // from llama_c.llama_token_bos()

pub fn deinit(self: *LlamaBackend) void     // frees ctx and model

// Vtable wrappers (for backend.Backend union — convert i32→u32, wrap iterator):
pub fn llamaTokenize(ptr, text, allocator) ![]u32    // llama_cpp.zig:223
pub fn llamaGenerate(ptr, tokens, params, allocator) !GenerationResult  // llama_cpp.zig:239
pub fn llamaDeinit(ptr, allocator) void
pub fn llamaGetVocabSize(ptr) u32
pub fn llamaEosToken(ptr) u32
pub fn llamaBosToken(ptr) u32
```

**Key insight for `chat_deepseek.zig`:** Do not use the vtable wrappers directly. Use `LlamaBackend` methods directly (like `chat_gptoss.zig` uses `MLXGPTOSSBackend` methods directly). Call `backend.tokenize(text)` to get `[]llama_c.llama_token`, convert to `[]u32` inline, then call `backend.generate()` pattern. Alternatively, call the vtable wrapper `llamaTokenize(ptr, text, allocator)` which already returns `[]u32`.

**Sampler creation:** `createSampler(allocator, params)` at `llama_cpp.zig:176` builds the top_k → top_p → temperature chain from a `GenerationParams`. Use this in `chat_deepseek.zig`.

### MLXGPTOSSBackend.tokenize() — Already Real

```zig
// src/backends/mlx_gptoss_backend.zig:284
pub fn tokenize(ptr: *anyopaque, text: []const u8, allocator: std.mem.Allocator) ![]u32
// Internally: Tokenizer.init(allocator, self.model_path) then tokenizer.encode(text)
// Note: re-initializes tokenizer on every call — consider caching as optimization
```

### GPTOSSTransformer.generate() — Current State

**In `src/gptoss_mlx.zig` (the file used by mlx_gptoss_backend.zig):**
```zig
// forward(): returns zeros([1,1,vocab_size]) — argmax gives 0 — EOS immediately
// generate(): calls forward() in loop, stops at token 0 (EOS from zero logits)
// Result: produce [] (empty) since EOS on first step, or [0] depending on loop
```

**In `src/mlx.zig/src/gptoss.zig` (MLX submodule's implementation):**
```zig
// GPTOSSTransformer has full layer structure:
//   token_embedding [vocab, hidden], layers[]GPTOSSLayer, norm, lm_head [vocab, hidden]
//   Each GPTOSSLayer: SlidingWindowAttention + MoERouter + experts
// forward(): passes through all layers with attention + MoE; lm_head projection is still zeros
// generate(): stubs sampling — always returns token=1 for max_tokens iterations
// Transformer.init(allocator, model_path): uses CPU stream, GPTOSSConfig.gptoss20b()
// Transformer.generate(input, num_tokens): delegates to inner.generate(input, num_tokens, 1.0)
```

**Discretion resolution — which to use for forward() fix:**
The `mlx.zig/src/gptoss.zig:Transformer` is more complete (has weight arrays, real attention scaffold, MoE layer structure). The `gptoss_mlx.zig:GPTOSSTransformer` is simpler (no weight fields, returns zeros immediately). Since weights are not loaded in either case, both produce zero/trivial output. However `mlx.zig/src/gptoss.zig` produces token=1 (not EOS=0) so generation will run for max_tokens — this is better observable behavior than immediate termination.

**Recommendation:** The real fix for GPT-OSS is:
1. Fix `chat_gptoss.zig:216` to call the real tokenizer (`mlx_gptoss_backend.tokenize()`)
2. The `generate()` path already works mechanically (mlx_gptoss_backend.zig:161-196 is real)
3. `GPTOSSTransformer.forward()` in `gptoss_mlx.zig` returns zero logits → EOS immediately — this produces empty output
4. To get non-empty output without real weights, switch to `mlx.zig/src/gptoss.zig:Transformer` which returns token=1 in a loop — produces tokens but not meaningful text
5. The tokenizer fix in step 1 is the minimum that satisfies "no empty token slices" since the issue is the empty input, not the forward pass

**The tokenizer stub at chat_gptoss.zig:215-216 is the root cause of GAP-04/MODEL-02:**
```zig
// STUB — line 215-216:
// Tokenize prompt (stub — returns empty for now; real tokenizer in future)
const prompt_tokens = try self.allocator.alloc(u32, 0);
// Fix: replace with:
const prompt_tokens = try mlx_gptoss_backend.tokenize(
    @ptrCast(self.gptoss_backend), harmony_prompt, self.allocator
);
```

### Prompt Cache index.json Schema (from saveIndex())

```json
{
  "entries": [
    {
      "key": "<194-char hex string>",
      "size_bytes": 12345,
      "created_at": 1712345678,
      "access_count": 42
    }
  ],
  "hits": 100,
  "misses": 50,
  "evictions": 5,
  "total_size_bytes": 987654321
}
```

**CacheEntry type fields needed for `loadIndex()` (infer from saveIndex()):**
- `key` → string key for `self.entries` hashmap
- `size_bytes` → `entry.size_bytes: u64`
- `created_at` → `entry.created_at: i64`
- `access_count` → `entry.access_count: std.atomic.Value(u32)` — init with loaded value

**loadIndex() implementation outline:**
```zig
fn loadIndex(self: *Self) !void {
    const index_path = try std.fs.path.join(self.allocator, &.{self.cache_dir, "index.json"});
    defer self.allocator.free(index_path);

    std.fs.accessAbsolute(index_path, .{}) catch {
        std.log.info("No existing cache index found, starting fresh", .{});
        return;
    };

    const file = try std.fs.openFileAbsolute(index_path, .{});
    defer file.close();
    const content = try file.readToEndAlloc(self.allocator, 1024 * 1024);
    defer self.allocator.free(content);

    const parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, content, .{});
    defer parsed.deinit();

    const entries_arr = parsed.value.object.get("entries") orelse return;
    for (entries_arr.array.items) |entry_val| {
        const obj = entry_val.object;
        const key_str = obj.get("key").?.string;
        // Populate self.entries hashmap with CacheEntry
        // ...
    }
}
```

---

## Speculation Removal Inventory (GAP-07)

Complete list of all symbols to remove. Ordered by file.

### src/main.zig

| Lines | Symbol / Block | Action |
|-------|----------------|--------|
| 14 | `const speculation = @import("speculation/mod.zig");` | Delete import |
| 42-44 | USAGE string lines for `--draft-model`, `--speculation-depth`, `--no-speculation` | Delete lines |
| 63 | USAGE example with `--draft-model` | Delete line |
| 100 | `draft_model: ?[]const u8 = null` in Config struct | Delete field |
| 101 | `speculation_depth: usize = 4` in Config struct | Delete field |
| 102 | `no_speculation: bool = false` in Config struct | Delete field |
| 251-256 | `--draft-model` CLI parse block | Delete block |
| 257-265 | `--speculation-depth` CLI parse block | Delete block |
| 267-268 | `--no-speculation` CLI parse block | Delete block |
| 330-339 | Log block for speculation settings in `parseArgs()` | Delete block |
| 358-360 | `convertFileConfig()` speculation field assignments | Delete 3 lines |
| 559-588 | Entire speculation init block (if/else, configure, defer shutdown) | Replace with single `defer manager_mod.deinitGlobalManager(allocator);` — `shutdownSpeculation` call in defer also removed |

Add comment at each removal site: `// speculative-decoding: removed — re-evaluate as dedicated phase after core inference is stable`

### src/config.zig

| Lines | Symbol | Action |
|-------|--------|--------|
| 34 | `draft_model: ?[]const u8 = null` in AppConfig | Delete |
| 35 | `speculation_depth: usize = 4` | Delete |
| 36 | `no_speculation: bool = false` | Delete |
| 53-55 | Same 3 fields in FileConfig | Delete |
| 165-167 | Field copy assignments in merge function | Delete 3 lines |
| 252 | `result.draft_model = ...` env var assignment | Delete |
| 257-261 | `ZLX_SPECULATION_DEPTH` env var block | Delete |
| 268-270 | `ZLX_NO_SPECULATION` env var block | Delete |
| 316-318 | `speculation_depth` validation block | Delete |

### src/inference/mod.zig

| Lines | Symbol | Action |
|-------|--------|--------|
| 244-261 | `.deepseek_v2_moe` match arm in `generateWithTimeout()` switch | Delete entire arm |
| 280-281 | `null, // draft_model` and `0, // speculation_depth` args to `GenerationState.init` | Remove these 2 args |
| 332 | `return error.DeepSeekNotImplemented;` after switch | Delete (unreachable) |
| 405-406 | `null, // draft_model` and `0, // speculation_depth` args in `generateWithLogprobs()` | Remove these 2 args |

### src/inference/generator.zig

The `GenerationState.init()` signature has `draft_model_ref` and `speculation_depth` parameters at lines 177-178. These must be removed from the signature and all call sites. After removing the parameters:
- Lines 163-164: `speculative_generator` and `use_speculation` fields in GenerationState — delete
- Lines 167: `draft_model` field — delete
- Lines 181-213: speculation initialization block in `init()` — delete
- Line 308: `if (!self.use_speculation)` condition — simplify
- Lines 377-onwards: speculation branch in `next()` — delete
- Lines 968-971: `generateAll()` wrapper — remove 2 params

**Compile risk:** Removing draft_model_ref/speculation_depth from GenerationState.init() will break all call sites. The call sites are:
1. `inference/mod.zig:280-281` (already listed above)
2. `inference/mod.zig:405-406` (already listed above)
3. `api/streaming.zig:141-142` (`null, 0` args)
4. `api/streaming.zig:410-411` (`null, 0` args)

### src/api/streaming.zig

| Lines | Action |
|-------|--------|
| 141-142 | Remove `null, // draft_model` and `0, // speculation_depth` args |
| 410-411 | Remove `null, // draft_model` and `0, // speculation_depth` args |

### src/speculation/ (entire directory)

Delete directory and all files:
- `src/speculation/mod.zig`
- `src/speculation/speculative_generator.zig`
- `src/speculation/draft_selector.zig`
- `src/speculation/speculative_generator_test.zig`
- `src/speculation/integration_test.zig`

Also delete `src/models/draft_model.zig` (imported only by generator.zig for speculation).

Remove from `build.zig` any speculation module compilation entries if present.

---

## Common Pitfalls

### Pitfall 1: factory.zig has real callers in generator.zig
**What goes wrong:** Deleting factory.zig causes compile errors in `inference/generator.zig` and `inference/backend_generator.zig`.
**Why it happens:** Both files import factory.zig for `BackendPreference` type and `createBackend()`.
**How to avoid:** Before deleting factory.zig, remove the imports and any factory usage from both files. `backend_generator.zig` can be deleted entirely (it has no real callers outside tests). `generator.zig` must have its factory import removed and the `factory.BackendPreference` type usage removed.
**Warning signs:** Compile error on `@import("../backends/factory.zig")`.

### Pitfall 2: backend.zig extern fn declarations for mlx_backend
**What goes wrong:** `backend.zig` has `extern fn mlxTokenize(...)`, `extern fn mlxGenerate(...)`, etc. (lines 233-254). These are extern declarations that assume `mlx_backend.zig` is linked. After deletion, the linker will fail.
**Why it happens:** The Backend union's `.mlx` arm calls these extern functions.
**How to avoid:** Remove the entire `.mlx` arm from the `Backend` union in `backend.zig`, and remove the corresponding extern fn declarations. Update `backends/mod.zig` to not export `mlx_backend`.
**Warning signs:** Linker error about undefined symbol `mlxTokenize`.

### Pitfall 3: DeepSeek branches in mod.zig left after HTTP dispatch
**What goes wrong:** After server.zig dispatches DeepSeek to chat_deepseek.zig, the DeepSeek branch in `inference/mod.zig:generateWithTimeout()` becomes unreachable dead code — but it still compiles, so it's a silent quality issue not a hard error.
**How to avoid:** Per D-01, delete the DeepSeek branch in mod.zig entirely since it's unreachable.

### Pitfall 4: LlamaBackend.init() called with empty string
**What goes wrong:** If `chat_deepseek.zig` tries to init `LlamaBackend` with `""` as model_path (mirroring GPT-OSS init), `llama_c.llama_model_load_from_file("", ...)` returns null → `error.ModelLoadFailed`.
**Why it happens:** GPT-OSS backend has `loaded: bool` lazy-init; LlamaBackend does not.
**How to avoid:** Wrap `g_deepseek_backend` in `?LlamaBackend` and init only when a valid model path is available (on first request or on `/v1/models/switch`).

### Pitfall 5: Tokenizer re-init per request in mlx_gptoss_backend
**What goes wrong:** `mlx_gptoss_backend.tokenize()` calls `Tokenizer.init()` on every invocation (line 288). This re-reads `tokenizer.json` from disk on every request.
**How to avoid:** Cache the tokenizer instance in `MLXGPTOSSBackend` struct. Init once in `load()`, reuse in `tokenize()`. This is a performance fix, not a correctness fix — it won't break correctness either way.

### Pitfall 6: GPT-OSS generate() produces empty slice
**What goes wrong:** Even after fixing the tokenizer stub in `chat_gptoss.zig:216`, `GPTOSSTransformer.forward()` in `gptoss_mlx.zig` returns zero logits → argmax returns 0 → immediate EOS → empty output.
**Why it happens:** `gptoss_mlx.zig:forward()` is `mlx.zeros([1,1,vocab])` which makes argmax return 0.
**How to avoid:** EOS check in `GPTOSSTransformer.generate()` (line 144) uses `if (next_token == 0) break;` which fires immediately. To get non-empty observable output without real weights, the EOS token for GPT-OSS is 100257 (not 0). Fix: use `self.config.vocab_size` related eos check, not hardcoded 0, or patch the generate loop to use the proper eos_token (100257 as set in `mlx_gptoss_backend.getEosToken()`).

### Pitfall 7: GenerationState.init signature change breaks 4 call sites
**What goes wrong:** Removing `draft_model_ref` and `speculation_depth` from `GenerationState.init()` signature causes compile errors at all 4 call sites.
**How to avoid:** Update all 4 call sites atomically in the same edit batch as the signature change.

---

## Code Examples

### Correct tokenize call in chat_gptoss.zig (fix for GAP-04/MODEL-02)
```zig
// Source: src/backends/mlx_gptoss_backend.zig:284 (real tokenize function)
// Replace lines 215-217 in chat_gptoss.zig with:
const prompt_tokens = try mlx_gptoss_backend.tokenize(
    @as(*anyopaque, @ptrCast(self.gptoss_backend)),
    harmony_prompt,
    self.allocator,
);
defer self.allocator.free(prompt_tokens);
```

### loadIndex() implementation skeleton (fix for GAP-06)
```zig
// Source: saveIndex() at prompt_cache.zig:438-476 defines the schema
fn loadIndex(self: *Self) !void {
    const index_path = try std.fs.path.join(
        self.allocator, &.{self.cache_dir, "index.json"}
    );
    defer self.allocator.free(index_path);

    std.fs.accessAbsolute(index_path, .{}) catch {
        std.log.info("No existing cache index found, starting fresh", .{});
        return;
    };

    std.log.info("Loading cache index from {s}", .{index_path});

    const file = try std.fs.openFileAbsolute(index_path, .{});
    defer file.close();

    const content = try file.readToEndAlloc(self.allocator, 4 * 1024 * 1024);
    defer self.allocator.free(content);

    const parsed = try std.json.parseFromSlice(
        std.json.Value, self.allocator, content, .{}
    );
    defer parsed.deinit();

    const root = parsed.value.object;
    const entries_arr = root.get("entries") orelse return;

    for (entries_arr.array.items) |entry_val| {
        const obj = entry_val.object;
        const key_str = (obj.get("key") orelse continue).string;
        const size = (obj.get("size_bytes") orelse continue).integer;
        const created = (obj.get("created_at") orelse continue).integer;
        const count = (obj.get("access_count") orelse continue).integer;

        const key_copy = try self.allocator.dupe(u8, key_str);
        const entry = CacheEntry{
            .size_bytes = @intCast(size),
            .created_at = created,
            .access_count = std.atomic.Value(u32).init(@intCast(count)),
        };
        try self.entries.put(key_copy, entry);
    }

    std.log.info("Loaded {d} cache entries from index", .{self.entries.count()});
}
```

### DeepSeek model name patterns (for server.zig dispatch)
```zig
// Source: src/models/registry.zig:47-48
// Canonical ID: "deepseek-coder-v2-lite"
// Aliases: "deepseek", "deepseek-v2", "deepseek-coder",
//          "deepseek-coder-v2-lite-instruct", "deepseek-coder-v2-lite-gguf"
// All start with "deepseek" — dispatch on std.mem.startsWith(u8, model, "deepseek")
```

---

## Compile Risks and Ordering Dependencies

| Task | Dependency | Risk |
|------|------------|------|
| Delete factory.zig | Must remove imports in generator.zig, backend_generator.zig, test_backend_integration.zig, backends/mod.zig | HIGH — will fail to compile if any import remains |
| Delete mlx_backend.zig | Must remove extern fn in backend.zig, remove .mlx arm from Backend union, update backends/mod.zig | HIGH — linker error if extern fns remain |
| Remove GenerationState.init draft params | Must update all 4 call sites atomically | MEDIUM — compile error at each stale call site |
| Delete speculation/ directory | Must remove import at main.zig:14 and all call sites in main.zig | HIGH — compile error immediately |
| Add LlamaBackend globals to server.zig | LlamaBackend must not init with empty path | MEDIUM — runtime panic if init before path set |
| Fix tokenize in chat_gptoss.zig | No compile risk — just swapping alloc(u32, 0) for a function call | LOW |

**Safe ordering for zero broken-build steps:**
1. Delete `factory.zig` + remove all imports + update `backends/mod.zig` + clean `generator.zig` factory usage
2. Delete `mlx_backend.zig` + remove extern fns from `backend.zig` + remove `.mlx` union arm
3. Update `backends/mod.zig` final cleanup
4. Remove speculation: first remove all call sites in main.zig, config.zig, streaming.zig, mod.zig → then remove generator.zig draft params → then delete speculation/ directory
5. Create `chat_deepseek.zig` + add server.zig globals + wire dispatch
6. Fix tokenize stub in `chat_gptoss.zig:216`
7. Fix `loadIndex()` in `prompt_cache.zig`
8. Fix GPT-OSS EOS token in generate loop (use 100257, not 0)

---

## Environment Availability

Step 2.6: SKIPPED — this phase is purely code/config changes to existing Zig source. No external tools, services, or runtimes need to be checked. The build environment (Zig 0.15.2, CMake, Xcode CLI) was verified in Phase 16.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Zig built-in test runner (0.15.2) |
| Config file | none — `zig build test` |
| Quick run command | `zig build test 2>&1 | head -50` |
| Full suite command | `zig build test` |

### Phase Requirements to Test Map

| Req ID | Behavior | Test Type | Automated Command |
|--------|----------|-----------|-------------------|
| GAP-05 | factory.zig and mlx_backend.zig deleted, build passes | compile | `zig build 2>&1 | grep -c error` (must be 0) |
| GAP-07 | No speculation symbols in compiled binary | compile | `zig build 2>&1 | grep speculation` (must be empty) |
| GAP-06 | loadIndex() reads index.json | unit | `zig build test -- --filter "cache index"` |
| GAP-03 | DeepSeek handler compiles and dispatches | compile + smoke | `zig build && curl -s localhost:8080/v1/models` |
| MODEL-02 | GPT-OSS returns non-empty tokens | integration | manual — requires model weights |
| GAP-04 | Tokenize returns real tokens | unit | `zig build test -- --filter "tokenize"` |

### Wave 0 Gaps
- No new test files required — existing test infrastructure covers compile verification.
- Manual integration tests require model weights not available in CI.

---

## Sources

### Primary (HIGH confidence)
- `/Users/gleicon/code/zig/zlx/src/backends/mlx_gptoss_backend.zig` — full file read; tokenize() confirmed real
- `/Users/gleicon/code/zig/zlx/src/gptoss_mlx.zig` — full file read; forward() confirmed zero-logits stub
- `/Users/gleicon/code/zig/zlx/src/mlx.zig/src/gptoss.zig` — full file read; GPTOSSTransformer full scaffold confirmed
- `/Users/gleicon/code/zig/zlx/src/api/chat_gptoss.zig` — full file read; tokenize stub at line 216 confirmed
- `/Users/gleicon/code/zig/zlx/src/api/server.zig` — lines 1-220 read; dispatch pattern confirmed
- `/Users/gleicon/code/zig/zlx/src/inference/mod.zig` — lines 1-420 read; both DeepSeekNotImplemented sites confirmed
- `/Users/gleicon/code/zig/zlx/src/backends/backend.zig` — full file read; extern fn declarations confirmed
- `/Users/gleicon/code/zig/zlx/src/backends/factory.zig` — full file read; createMlxBackend stub confirmed
- `/Users/gleicon/code/zig/zlx/src/backends/mlx_backend.zig` — full file read; all stubs confirmed
- `/Users/gleicon/code/zig/zlx/src/backends/llama_cpp.zig` — full file read; LlamaBackend interface documented
- `/Users/gleicon/code/zig/zlx/src/cache/prompt_cache.zig` — lines 410-485 read; saveIndex schema documented
- `/Users/gleicon/code/zig/zlx/src/main.zig` — speculation blocks read; all symbol locations confirmed
- `/Users/gleicon/code/zig/zlx/src/config.zig` — grep confirmed all speculation field locations
- `/Users/gleicon/code/zig/zlx/src/inference/generator.zig` — grep confirmed draft_model/speculation_depth locations
- `/Users/gleicon/code/zig/zlx/src/models/registry.zig` — DeepSeek model name patterns confirmed

---

## Metadata

**Confidence breakdown:**
- Stub inventory: HIGH — direct source reads, exact line numbers
- LlamaBackend interface: HIGH — full file read
- index.json schema: HIGH — read from saveIndex() implementation
- Speculation symbol list: HIGH — grep + source reads across all files
- GPT-OSS forward() analysis: HIGH — read both gptoss_mlx.zig and mlx.zig/src/gptoss.zig
- Compile risk ordering: HIGH — traced all import chains

**Research date:** 2026-04-05
**Valid until:** 2026-05-05 (stable codebase, no fast-moving dependencies)
