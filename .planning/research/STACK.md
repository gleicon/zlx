# Stack Research

**Domain:** Zig LLM inference server wrapping MLX on Apple Silicon
**Researched:** 2026-03-30
**Confidence:** MEDIUM overall — MLX.zig HIGH (source read), httpz MEDIUM, TurboQuant LOW (PRD assumption is wrong), build integration LOW

---

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Zig | 0.13.0 | Language / build system | PRD-mandated; MLX.zig explicitly targets 0.13.0. Do NOT use Zig master (0.14+) — MLX.zig and httpz zig-0.13 branch are not compatible with it yet. |
| MLX.zig (jaco-bro/MLX.zig) | 0.0.0 (pre-release, pinned to submodule commit) | LLM inference runtime on Metal GPU | Provides Transformer.init/generate loop, tokenizer, Llama/Phi/Qwen model configs, and the mlx-c linkage. Already vendored as git submodule at `src/mlx.zig/`. |
| mlx-c | v0.1.2 (pinned inside MLX.zig build.zig) | C API bridge between Zig's @cImport and the MLX C++ framework | MLX.zig's `build.zig` fetches `mlx-c-0.1.2.tar.gz` at build time via `curl`. Zig consumes `mlx/c/mlx.h` via `@cImport`. Upstream mlx-c is at v0.4.1 but MLX.zig pins 0.1.2 — do not upgrade independently. |
| httpz (karlseguin/http.zig) | zig-0.13 branch (master targets Zig 0.15.1) | HTTP/1.1 server | Single dependency, zero native deps, ~140K req/s on M2, SSE support via `res.writer()`. Must use the `zig-0.13` branch archive URL, not master. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pcre2 | 10.45 (via MLX.zig's build.zig.zon) | Regex for tokenizer | Pulled automatically as part of MLX.zig; no action required |
| std.json | stdlib (Zig 0.13.0) | JSON parsing for request body, model config loading | Use for POST body deserialization and `config.json` / `tokenizer.json` loading |
| std.http.Client | stdlib (Zig 0.13.0) | Model weight download (optional) | Only if implementing the model downloader; out of scope per PRD |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| CMake | Required for the initial `mlx-c` build inside MLX.zig | MLX.zig's build.zig shells out to `cmake` and `make`; must be on PATH for `zig build` to succeed |
| `zig fetch` | Generate correct content hash for build.zig.zon | Run `zig fetch <url>` to get the `hash` value; the current build.zig.zon has `"..."` placeholder |
| Xcode command-line tools | Metal shader compilation | Required at build time; MLX links Metal.framework, Foundation.framework, QuartzCore.framework, Accelerate.framework |

---

## Critical API Shapes (Verified from Source)

### MLX.zig — Transformer API

The `Transformer` is a comptime generic defined in `src/mlx.zig/src/mlx.zig`. Each model (Llama, Phi, Qwen) exports its own `Transformer` type alias:

```zig
// qwen.zig exports:
pub const Transformer = mlx.Transformer(Model, ModelConfig);

// Usage pattern (from qwen.zig tests and llm.zig):
var transformer = try Transformer.init(allocator, "Qwen2.5-Coder-7B-Instruct-4bit");
defer transformer.deinit();
// input_ids = token IDs produced by Tokenizer.encodeChat()
const output_ids = try transformer.generate(input_ids, max_tokens);
defer allocator.free(output_ids);
```

**CRITICAL CONSTRAINT — generate() is batch-only, not streaming.**
`Transformer.generate()` runs the full decode loop and returns `[]u32` when done. It prints token IDs to stderr as a side effect but provides no callback, iterator, or yield mechanism. SSE streaming requires either:
1. Forking the generate loop into a custom streaming version that calls a per-token callback, or
2. Running inference in a background thread and feeding tokens to an SSE channel.

This is not a minor integration detail — it is a core architectural constraint for the server.

### MLX.zig — Tokenizer API

```zig
// Reads tokenizer.json from model_name directory
var tokenizer = try Tokenizer.init(allocator, model_name); // model_name is local path
defer tokenizer.deinit();
const input_ids = try tokenizer.encodeChat(chat_format, messages_slice);
defer allocator.free(input_ids);
const text = try tokenizer.decode(token_ids);
defer allocator.free(text);
```

`Tokenizer.init` takes a directory path and reads `{path}/tokenizer.json`. It does NOT auto-download from HuggingFace. Models must be placed manually in `./models/<model-name>/`.

### MLX.zig — C Interop Pattern

```zig
// mlx.zig wraps mlx-c headers:
pub const C = @cImport({
    @cInclude("mlx/c/mlx.h");
    @cInclude("stdio.h");
});

// Opaque types from mlx-c:
pub const Array  = C.mlx_array;
pub const Stream = C.mlx_stream;

// Link requirements (from configureExecutable in MLX.zig build.zig):
exe.addIncludePath(.{ .cwd_relative = mlx_c_path });
exe.addObjectFile(.{ .cwd_relative = "libmlxc.a" });
exe.addObjectFile(.{ .cwd_relative = "libmlx.a" });
exe.linkFramework("Metal");
exe.linkFramework("Foundation");
exe.linkFramework("QuartzCore");
exe.linkFramework("Accelerate");
exe.linkLibCpp();
```

### MLX.zig — Module Export Problem

**CRITICAL: MLX.zig does NOT export a Zig module named `"mlx"`.**
The current `build.zig.zon` uses `.mlx = .{ .path = "src/mlx.zig" }` (a directory), and `build.zig` calls `b.dependency("mlx", .{}).module("mlx")`. This will fail because MLX.zig's `build.zig` only creates executables, not a library module. The zlx `build.zig` needs to inline `configureExecutable`'s logic (the `addObjectFile` / `linkFramework` calls) and import `.zig` source files directly rather than using the module API.

### httpz — Handler Pattern (zig-0.13 branch)

```zig
const httpz = @import("httpz");

// Server init:
var server = try httpz.Server(void).init(allocator, .{ .port = 8080 }, {});
defer server.deinit();

// Routing:
var router = server.router(.{});
router.post("/v1/chat/completions", chatCompletions, .{});

// Handler signature:
fn chatCompletions(req: *httpz.Request, res: *httpz.Response) !void {
    // Parse JSON body:
    const body = try req.json(RequestBody);

    // Non-streaming response:
    res.status = 200;
    res.content_type = .JSON;
    res.body = json_string;

    // Streaming (SSE): write chunks via res.writer()
    // SSE format: "data: {json}\n\n"
    const writer = res.writer();
    try writer.writeAll("data: {\"choices\":[...]}\n\n");
}

// Start server (blocks):
try server.listen();
```

**Note:** The `master` branch of http.zig targets Zig 0.15.1. For Zig 0.13.0 use the `zig-0.13` branch. The build.zig.zon dependency URL must be:
```
https://github.com/karlseguin/http.zig/archive/refs/heads/zig-0.13.tar.gz
```
Run `zig fetch <url>` to generate the correct content hash.

---

## TurboQuant — Critical Finding: PRD Assumption Is Wrong

**The PRD states: "bind arozanov/turboquant-mlx C++ Metal kernels via Zig C interop."**
**This is incorrect. arozanov/turboquant-mlx is 100% Python.**

Confirmed findings:
- Repository language: Python only (no C++, no C headers, no `.metal` files to link)
- Metal kernel source strings are embedded in Python via `mx.fast.metal_kernel()` (MLX's JIT kernel API)
- The `quantize_key`, `dequantize`, and Hadamard rotation kernels are Python functions that JIT-compile Metal shader source strings at runtime through MLX's Python API
- There is no C header, no shared library, and no extern ABI to call from Zig

**Implication:** The planned TurboQuant integration path does not exist as described. The feasible options are:

| Option | Effort | Confidence | Notes |
|--------|--------|------------|-------|
| Port Metal kernel source strings to C++ and expose via mlx-c custom ops API | High | LOW | mlx-c v0.1.2 predates the custom ops API; would require upgrading mlx-c and implementing the C++ primitive, which breaks MLX.zig's pinned version |
| Write Zig wrappers around equivalent mlx-c array ops (Hadamard-ish, quantized matmul) | Medium | LOW | Approximation; loses kernel fusion benefit; not the same algorithm |
| Defer TurboQuant entirely from MVP | Low | HIGH | Correct scope given complexity |
| Use a Python sidecar process for TurboQuant compression, communicate via IPC | Very High | MEDIUM | Defeats the "no Python" requirement |

**Recommendation: Defer TurboQuant from MVP.** The `--turboquant` flag can be accepted by the CLI and print a "not yet implemented" message. The standard MLX KV cache (always enabled in `Transformer.generate()`) already provides reasonable context. TurboQuant is a separate milestone requiring dedicated feasibility work.

**Confidence: LOW** — TurboQuant integration as specified in the PRD is not feasible without significant out-of-scope work.

---

## Installation / Build Setup

```bash
# Zig 0.13.0 required -- verify:
zig version   # must be 0.13.0

# CMake required for mlx-c build:
brew install cmake

# Xcode CLI tools for Metal:
xcode-select --install

# Init submodule (already done per PROJECT.md):
git submodule update --init --recursive

# Fix build.zig.zon: replace "..." hash for httpz
zig fetch https://github.com/karlseguin/http.zig/archive/refs/heads/zig-0.13.tar.gz
# Copy printed hash into build.zig.zon

# Build (first run fetches mlx-c via curl + cmake):
zig build
```

---

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| httpz zig-0.13 branch | httpz master | Only when project upgrades to Zig 0.14+ |
| httpz zig-0.13 branch | std.http.Server (Zig stdlib) | Never for this project — 10-100x slower, no routing, no SSE helpers |
| httpz zig-0.13 branch | zap (facil.io wrapper) | If needing HTTP/2 or TLS termination — overkill for local single-user server |
| MLX.zig submodule | mlx-c directly | Only if MLX.zig's generate loop needs to be replaced (streaming) |
| mlx-c v0.1.2 (via MLX.zig) | mlx-c v0.4.1 | Only if custom ops API is needed for TurboQuant — breaks MLX.zig pin |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| httpz `master` branch | Targets Zig 0.15.1; incompatible with Zig 0.13.0 | httpz `zig-0.13` branch |
| Python MLX runtime | PRD explicitly excludes Python in the final binary | MLX.zig + mlx-c C bindings |
| CMake in final binary | Build artifact only; CMake is only needed to compile libmlxc.a | CMake runs at `zig build` time, not at runtime |
| arozanov/turboquant-mlx directly | 100% Python; no C/C++ API to bind | Defer TurboQuant, or write C++ wrapper as a future milestone |
| Zig 0.14+ or Zig master | MLX.zig targets 0.13.0; breaking changes in build API | Stay on Zig 0.13.0 until MLX.zig updates |
| `b.dependency("mlx").module("mlx")` | MLX.zig does not export a named module — call will panic at build time | Directly include MLX.zig source files and replicate `configureExecutable` linkage |

---

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| Zig 0.13.0 | MLX.zig 0.0.0, httpz zig-0.13 branch, pcre2-10.45 | Lock this combination; do not upgrade any piece without testing the others |
| mlx-c v0.1.2 | MLX.zig 0.0.0 | MLX.zig fetches this exact version at build time via curl |
| mlx-c v0.4.1 (upstream current) | NOT compatible with MLX.zig 0.0.0 as-is | Header and API changes between 0.1.2 and 0.4.1 |
| httpz master | Zig 0.15.1 only | Do not use with Zig 0.13.0 |

---

## Open Build Issues (Must Resolve)

1. **MLX.zig module export**: `b.dependency("mlx", .{}).module("mlx")` will fail. The build.zig must be rewritten to directly link MLX.zig's compiled objects and expose its `.zig` source files as imports rather than a package module.

2. **httpz hash placeholder**: `build.zig.zon` contains `.hash = "..."` which is not a valid hash. Run `zig fetch` to generate it.

3. **Streaming generate()**: `Transformer.generate()` is synchronous and batch-only. SSE streaming requires a custom token loop (fork from mlx.zig's Transformer.generate body) with a per-token writer callback.

4. **Model path convention**: `Tokenizer.init(allocator, path)` reads `{path}/tokenizer.json`. The model path must be a directory, not a model name. Document the expected layout: `./models/Qwen2.5-Coder-7B-Instruct-4bit/config.json`, `tokenizer.json`, `*.safetensors`.

---

## Sources

- `/Users/gleicon/code/zig/zlx/src/mlx.zig/src/mlx.zig` — MLX.zig source (Transformer generic, C binding pattern, mlx-c types) — HIGH confidence
- `/Users/gleicon/code/zig/zlx/src/mlx.zig/src/llm.zig` — TransformerUnion dispatch, ChatConfig, generate loop invocation — HIGH confidence
- `/Users/gleicon/code/zig/zlx/src/mlx.zig/src/qwen.zig` — QwenTransformer test showing Transformer.init/generate API — HIGH confidence
- `/Users/gleicon/code/zig/zlx/src/mlx.zig/build.zig` — mlx-c v0.1.2 pin, configureExecutable linkage (Metal/Foundation/etc) — HIGH confidence
- `/Users/gleicon/code/zig/zlx/src/mlx.zig/build.zig.zon` — pcre2 dependency, no mlx-c in zon (fetched via curl in build.zig) — HIGH confidence
- `https://github.com/karlseguin/http.zig/blob/zig-0.13/example/simple.zig` — httpz zig-0.13 routing/handler pattern — MEDIUM confidence (rendered, not raw)
- `https://deepwiki.com/karlseguin/http.zig/1.1-getting-started` — httpz Server(H).init, res.writer(), listen() — MEDIUM confidence
- `https://github.com/arozanov/turboquant-mlx` — "Language: 100% Python", no C headers — HIGH confidence (confirms PRD assumption is wrong)
- `https://ml-explore.github.io/mlx-c/build/html/overview.html` — mlx-c C API overview (mlx_array, mlx_stream, constructor/free pattern) — MEDIUM confidence

---

*Stack research for: Zig LLM inference server (zlx) wrapping MLX on Apple Silicon*
*Researched: 2026-03-30*
