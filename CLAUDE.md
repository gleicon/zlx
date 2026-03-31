<!-- GSD:project-start source:PROJECT.md -->
## Project

**zlx**

`zlx` is a minimal Zig HTTP server that exposes an OpenAI-compatible `/v1/chat/completions` endpoint for local coding models on Apple Silicon Macs. It runs inference via MLX.zig (native Metal GPU acceleration) with TurboQuant KV-cache compression for extended context at high throughput. Built for personal use — point OpenCode/Continue.dev at `http://localhost:8080/v1` and work locally with no Python runtime.

**Core Value:** A single `zig build` binary that lets OpenCode connect to local coding models without any Python or cloud dependency.

### Constraints

- **Tech stack**: Latest stable Zig, MLX.zig, httpz — no Python, no Bazel, no CMake in final binary
- **Platform**: macOS aarch64 only — Metal GPU required, no cross-platform target
- **Dependencies**: TurboQuant C++ kernels must be bound (not rewritten) — scope limited to interop, not porting
- **Distribution**: Personal use only — no installer, no multi-user auth, no TLS required
- **Build**: Single `zig build` command must produce a working binary
<!-- GSD:project-end -->

<!-- GSD:stack-start source:research/STACK.md -->
## Technology Stack

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
## Critical API Shapes (Verified from Source)
### MLX.zig — Transformer API
### MLX.zig — Tokenizer API
### MLX.zig — C Interop Pattern
### MLX.zig — Module Export Problem
### httpz — Handler Pattern (zig-0.13 branch)
## TurboQuant — Critical Finding: PRD Assumption Is Wrong
- Repository language: Python only (no C++, no C headers, no `.metal` files to link)
- Metal kernel source strings are embedded in Python via `mx.fast.metal_kernel()` (MLX's JIT kernel API)
- The `quantize_key`, `dequantize`, and Hadamard rotation kernels are Python functions that JIT-compile Metal shader source strings at runtime through MLX's Python API
- There is no C header, no shared library, and no extern ABI to call from Zig
| Option | Effort | Confidence | Notes |
|--------|--------|------------|-------|
| Port Metal kernel source strings to C++ and expose via mlx-c custom ops API | High | LOW | mlx-c v0.1.2 predates the custom ops API; would require upgrading mlx-c and implementing the C++ primitive, which breaks MLX.zig's pinned version |
| Write Zig wrappers around equivalent mlx-c array ops (Hadamard-ish, quantized matmul) | Medium | LOW | Approximation; loses kernel fusion benefit; not the same algorithm |
| Defer TurboQuant entirely from MVP | Low | HIGH | Correct scope given complexity |
| Use a Python sidecar process for TurboQuant compression, communicate via IPC | Very High | MEDIUM | Defeats the "no Python" requirement |
## Installation / Build Setup
# Zig 0.13.0 required -- verify:
# CMake required for mlx-c build:
# Xcode CLI tools for Metal:
# Init submodule (already done per PROJECT.md):
# Fix build.zig.zon: replace "..." hash for httpz
# Copy printed hash into build.zig.zon
# Build (first run fetches mlx-c via curl + cmake):
## Alternatives Considered
| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| httpz zig-0.13 branch | httpz master | Only when project upgrades to Zig 0.14+ |
| httpz zig-0.13 branch | std.http.Server (Zig stdlib) | Never for this project — 10-100x slower, no routing, no SSE helpers |
| httpz zig-0.13 branch | zap (facil.io wrapper) | If needing HTTP/2 or TLS termination — overkill for local single-user server |
| MLX.zig submodule | mlx-c directly | Only if MLX.zig's generate loop needs to be replaced (streaming) |
| mlx-c v0.1.2 (via MLX.zig) | mlx-c v0.4.1 | Only if custom ops API is needed for TurboQuant — breaks MLX.zig pin |
## What NOT to Use
| Avoid | Why | Use Instead |
|-------|-----|-------------|
| httpz `master` branch | Targets Zig 0.15.1; incompatible with Zig 0.13.0 | httpz `zig-0.13` branch |
| Python MLX runtime | PRD explicitly excludes Python in the final binary | MLX.zig + mlx-c C bindings |
| CMake in final binary | Build artifact only; CMake is only needed to compile libmlxc.a | CMake runs at `zig build` time, not at runtime |
| arozanov/turboquant-mlx directly | 100% Python; no C/C++ API to bind | Defer TurboQuant, or write C++ wrapper as a future milestone |
| Zig 0.14+ or Zig master | MLX.zig targets 0.13.0; breaking changes in build API | Stay on Zig 0.13.0 until MLX.zig updates |
| `b.dependency("mlx").module("mlx")` | MLX.zig does not export a named module — call will panic at build time | Directly include MLX.zig source files and replicate `configureExecutable` linkage |
## Version Compatibility
| Package | Compatible With | Notes |
|---------|-----------------|-------|
| Zig 0.13.0 | MLX.zig 0.0.0, httpz zig-0.13 branch, pcre2-10.45 | Lock this combination; do not upgrade any piece without testing the others |
| mlx-c v0.1.2 | MLX.zig 0.0.0 | MLX.zig fetches this exact version at build time via curl |
| mlx-c v0.4.1 (upstream current) | NOT compatible with MLX.zig 0.0.0 as-is | Header and API changes between 0.1.2 and 0.4.1 |
| httpz master | Zig 0.15.1 only | Do not use with Zig 0.13.0 |
## Open Build Issues (Must Resolve)
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
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd:quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd:debug` for investigation and bug fixing
- `/gsd:execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd:profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
