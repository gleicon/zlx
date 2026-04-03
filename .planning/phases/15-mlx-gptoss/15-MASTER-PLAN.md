---
phase: 15-mlx-gptoss
plan: MASTER
type: execute
wave: 1
depends_on: []
files_modified: []
autonomous: true
requirements:
  - GPTOSS-01
  - GPTOSS-02
  - GPTOSS-03
  - GPTOSS-04
  - GPTOSS-05
must_haves:
  truths:
    - "GPT-OSS-20B generates at 30+ tokens/sec via native MLX (not llama.cpp)"
    - "Harmony chat format is correctly parsed and formatted"
    - "Browser tool works for web search/open/find operations"
    - "Python tool executes code in sandboxed environment"
    - "MXFP4 weights load and decompress correctly"
    - "All models pass test_models.sh"
  artifacts:
    - path: "src/mlx.zig/src/gptoss.zig"
      provides: "GPT-OSS transformer architecture"
      min_lines: 200
    - path: "src/mlx.zig/src/gptoss_metal.metal"
      provides: "Metal kernels for GPT-OSS"
      contains: "kernel void gptoss_moe"
    - path: "src/harmony/"
      provides: "Harmony format parser and chat template"
      contains: "HarmonyChat, HarmonyEncoding"
    - path: "src/tools/browser.zig"
      provides: "Browser tool implementation"
      contains: "BrowserTool.search/open/find"
    - path: "src/tools/python.zig"
      provides: "Python tool implementation"
      contains: "PythonTool.execute"
    - path: "src/weight/gptoss_loader.zig"
      provides: "GPT-OSS weight loading with MXFP4"
      contains: "loadMXFP4Weights"
    - path: "src/mxfp4.zig"
      provides: "MXFP4 dequantization"
      contains: "dequantizeMXFP4"
  key_links:
    - from: "src/mlx.zig/src/gptoss.zig"
      to: "src/mlx.zig/src/gptoss_metal.metal"
      via: "FastMetalKernel API"
      pattern: "mlx_fast_metal_kernel"
    - from: "src/harmony/harmony.zig"
      to: "src/api/chat.zig"
      via: "formatHarmonyChat"
      pattern: "HarmonyChat.format"
    - from: "src/tools/browser.zig"
      to: "src/api/tools.zig"
      via: "HTTP endpoint /v1/tools/browser"
      pattern: "browserToolHandler"
    - from: "src/mxfp4.zig"
      to: "src/weight/gptoss_loader.zig"
      via: "dequantizeMXFP4 function"
      pattern: "dequantizeMXFP4"
---

<objective>
Implement high-performance GPT-OSS support using native MLX (like openharmony-mlx), replacing the llama.cpp approach with a native Zig/MLX implementation that achieves 40 tokens/sec on Apple Silicon.

Purpose: Provide a fast, native MLX-based inference path for GPT-OSS models that outperforms the llama.cpp backend while supporting the full feature set (Harmony format, tools, MXFP4).

Output: Complete GPT-OSS inference pipeline with native Metal kernels, Harmony chat support, browser/python tools, and MXFP4 weight loading.
</objective>

<execution_context>
@$HOME/.config/opencode/get-shit-done/workflows/execute-plan.md
@$HOME/.config/opencode/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/STACK.md

## Key Insights from openharmony-MLX

Reference: https://github.com/arthurcolle/openharmony-mlx

**Architecture:**
- Native MLX implementation (not llama.cpp wrapper)
- Metal kernels for MoE routing and attention
- MXFP4 quantization native to GPT-OSS MoE layers
- Harmony chat format for tool support
- Browser + Python tools as first-class citizens

**Performance:**
- 40 tokens/sec on Apple Silicon for GPT-OSS-20B
- Native Metal GPU acceleration
- Sliding window attention for long context
- MoE with 117B total / 5.1B active parameters

**Files to Study:**
- `gpt_oss/mlx_gpt_oss/` - MLX implementation
- `gpt_oss/metal/` - Metal kernels reference
- `gpt_oss/tools/` - Browser and Python tools

## Current State

- Phase 13: GPT-OSS weight loading for llama.cpp (in progress)
- Phase 14: llama.cpp backend integration (planned)
- This Phase 15: Native MLX alternative to llama.cpp

## Architecture Decision

Dual-backend approach:
1. **llama.cpp backend** (Phase 14): GGUF-based, compatible with many models
2. **Native MLX backend** (Phase 15): Maximum performance for GPT-OSS

User can select backend via config or auto-detect based on model.
</context>

## Implementation Plans

| Plan | Focus | Wave | Est. Hours | Dependencies |
|------|-------|------|------------|--------------|
| 15-01 | MLX GPT-OSS Transformer + Metal Kernels | 1 | 10-12 | 11-01 (mlx-c v0.4.x), 12-02 (GPT-OSS arch) |
| 15-02 | Harmony Format Parser | 1 | 4-6 | None |
| 15-03 | Browser + Python Tools | 2 | 6-8 | 15-02 |
| 15-04 | Weight Loading + MXFP4 Support | 2 | 6-8 | 15-01 |
| 15-05 | Integration with zlx Server | 3 | 4-6 | 15-01, 15-02, 15-03, 15-04 |

## Technical Architecture

### 1. GPT-OSS Transformer (15-01)

```
src/mlx.zig/src/gptoss.zig
├── GPTOSSTransformer
│   ├── token_embedding
│   ├── layers: []GPTOSSTransformerLayer
│   │   ├── attention: SlidingWindowAttention
│   │   ├── moe: MixtureOfExperts (native MLX)
│   │   └── norms
│   ├── norm
│   └── lm_head
└── GPTOSSTokenGenerator
    ├── generate()
    └── sample()

src/mlx.zig/src/gptoss_metal.metal
├── gptoss_moe_route() - Expert selection
├── gptoss_moe_apply() - Sparse expert computation
└── gptoss_sw_attention() - Sliding window attention
```

### 2. Harmony Format (15-02)

```
src/harmony/
├── harmony.zig
│   ├── HarmonyEncoding
│   ├── HarmonyMessage
│   └── HarmonyConversation
├── parser.zig
│   ├── parseHarmonyEncoding()
│   └── parseMessagesFromTokens()
└── template.zig
    ├── formatHarmonyChat()
    └── renderConversation()
```

**Key insight:** Harmony uses special tokens for tool calls:
- `<|tool_call|>` / `<|/tool_call|>`
- `<|tool_result|>` / `<|/tool_result|>`
- `<|recipient|>` / `<|/recipient|>`

### 3. Tools (15-03)

```
src/tools/
├── browser.zig
│   ├── BrowserTool
│   │   ├── search(query) -> SearchResult[]
│   │   ├── open(url) -> PageContent
│   │   └── find(query, page_id) -> TextLocation[]
│   └── browser_mcp.zig (optional MCP integration)
└── python.zig
    ├── PythonTool
    │   ├── execute(code) -> ExecutionResult
    │   └── sandbox: Docker-based
    └── python_docker.zig
```

### 4. Weight Loading (15-04)

```
src/weight/gptoss_loader.zig
├── GPTOSSWeightLoader
│   ├── loadFromSafetensors() - Load original weights
│   ├── loadFromMLX() - Load pre-converted MLX format
│   └── convertToMLX() - Convert safetensors → MLX
└── MXFP4 support
    ├── dequantizeMXFP4()
    ├── loadMXFP4Blocks()
    └── applyBlockScales()

src/mxfp4.zig
├── MXFP4Tensor
├── dequantize() -> mlx.Array (BF16)
└── GPU-accelerated via Metal kernel
```

### 5. Integration (15-15)

```
src/backends/
├── backend.zig (existing Backend trait)
├── llama_backend.zig (Phase 14)
└── mlx_gptoss_backend.zig (this phase)
    ├── MLXGPTOSSBackend
    │   ├── loadModel()
    │   ├── generate()
    │   └── supportsTools()
    └── tool orchestration
```

## Dependencies Between Plans

```
Wave 1 (Independent):
├── 15-01: MLX GPT-OSS Transformer
│   └── Needs: mlx-c v0.4.x (11-01), GPT-OSS arch knowledge (12-02)
└── 15-02: Harmony Format Parser
    └── Can run in parallel

Wave 2:
├── 15-03: Tools (depends on Harmony parsing)
└── 15-04: Weight Loading (depends on Transformer architecture)

Wave 3:
└── 15-05: Integration (depends on all above)
```

## Success Criteria

| Criteria | Target | How to Verify |
|----------|--------|---------------|
| **Performance** | 30+ tokens/sec GPT-OSS-20B | Benchmark script |
| **Harmony Format** | Parse/format correctly | Unit tests |
| **Browser Tool** | Search/open/find work | Integration test |
| **Python Tool** | Execute code safely | Integration test |
| **MXFP4 Loading** | Load 20B model | `./test_models.sh gptoss` |
| **End-to-end** | OpenCode integration | Manual test |

## Risk Assessment

| Risk | Level | Mitigation |
|------|-------|------------|
| Metal kernel performance | High | Profile with Metal System Trace; fallback to CPU kernels |
| MXFP4 dequantization | Medium | Use reference implementation; validate numerics |
| Harmony format complexity | Low | Use openai-harmony reference; comprehensive tests |
| Tool security | Medium | Docker sandbox for Python; URL whitelist for browser |

## References

- openharmony-mlx: https://github.com/arthurcolle/openharmony-mlx
- openai-harmony: https://github.com/openai/harmony
- GPT-OSS models: https://huggingface.co/openai/gpt-oss-20b
- MXFP4 spec: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
- MLX custom kernels: https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html

---

**Next: Execute plan 15-01 for MLX GPT-OSS transformer implementation**
