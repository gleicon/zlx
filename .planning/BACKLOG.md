# Project Backlog

## Phase 11: DeepSeek MoE Support (PENDING)

**Priority:** High  
**Complexity:** Very High (20+ hours)  
**Status:** Blocked - waiting for MLX.zig MoE support or mlx-c upgrade

### Description
Full support for DeepSeek-Coder-V2-Lite-Instruct and other DeepSeek-V2 architecture models with MoE (Mixture of Experts).

### Blockers
1. **MLX.zig MoE Support:** Current MLX.zig (v0.0.0) only supports dense transformers (Llama, Qwen, Phi)
2. **mlx-c Version:** Locked to v0.1.2, MoE ops may require v0.4.x
3. **Architecture Complexity:** MLA (Multi-head Latent Attention) differs significantly from standard MHA

### Technical Requirements

#### 1. MLA (Multi-head Latent Attention) Implementation
```zig
// New file: src/mlx.zig/src/mla.zig
pub const MultiHeadLatentAttention = struct {
    // DeepSeek's compressed attention mechanism
    // Reduces KV cache by ~90% vs standard MHA
};
```

#### 2. MoE Routing Layer
```zig
// src/mlx.zig/src/moe.zig
pub const MixtureOfExperts = struct {
    num_experts: u32,           // Total experts (e.g., 64)
    num_shared_experts: u32,  // Always active (e.g., 2)
    top_k: u32,               // Routed experts per token (e.g., 6)
    
    pub fn route(self: *Self, hidden_states: mlx.Array) !ExpertSelection;
};
```

#### 3. Memory Estimation Updates
Current estimation assumes dense parameters:
```
// DeepSeek-V2: 15.7B total, 2B active per token
// Memory = embedding + active_params + kv_cache
// NOT = total_params (would overestimate by 8x!)
```

#### 4. Chat Template Support
DeepSeek uses unique format:
```
User: {message}
Assistant: {response}
```
Not standard OpenAI format.

### Research Notes

**DeepSeek-V2 Paper Highlights:**
- MLA reduces KV cache from 140MB to 4MB per sequence
- MoE enables 15.7B model with inference cost of 2B dense
- 128k context length (vs 8k for Qwen 1.5B)
- State-of-the-art code generation benchmarks

**Alternative Implementations:**
- llama.cpp: Added DeepSeek-V2 support in v0.3.0 (reference implementation)
- vLLM: Supports via custom attention backend
- MLX Python: Full support via `mlx-lm`

### Success Criteria
- [ ] DeepSeek model loads and runs inference
- [ ] Memory usage matches 2B active params (not 15.7B)
- [ ] 128k context works with TurboQuant
- [ ] Performance within 10% of MLX Python baseline
- [ ] Chat completions use correct format

### Estimated Effort
- **Core MoE/MLA:** 15-20 hours
- **Chat template:** 2-3 hours  
- **Testing/optimization:** 3-5 hours
- **Total:** 20-28 hours

### Unblocking Strategy

**Option 1: Upgrade mlx-c (Recommended)**
- Upgrade to mlx-c v0.4.x (latest)
- Implement MoE as custom primitive
- Effort: 8-12 hours + upgrade work

**Option 2: Wait for MLX.zig**
- Monitor jaco-bro/MLX.zig for MoE PRs
- Contribute if possible
- Effort: 0 hours (passive)

**Option 3: Custom Implementation**
- Implement MLA + MoE from scratch in Zig
- Reference llama.cpp implementation
- Effort: 20-30 hours

### Priority Justification
DeepSeek-Coder-V2-Lite is currently:
- #1 on multiple code generation benchmarks
- 5x better than Qwen 1.5B on HumanEval
- Free license for research/commercial use
- Significant competitive advantage for zlx

---

## Other Deferred Items

### Large Model Support (60B+)
**Models:** Llama 3.1 405B, GPT-OSS 120B  
**Blocker:** Memory management, paging, quantization  
**Effort:** Medium (8-12 hours)

### speculative decoding cross-architecture
**Current:** Same architecture only (Qwen→Qwen)  
**Goal:** Qwen target + Llama draft  
**Blocker:** Logit space alignment  
**Effort:** Medium (4-6 hours)

### Multi-GPU Support
**Blocker:** MLX multi-device API  
**Effort:** Low (2-4 hours once MLX supports)

### WebUI Features
**Features:** Chat history, model comparison, parameters UI  
**Blocker:** Out of scope (zlx is API-only)  
**Note:** Use Open WebUI, Continue.dev, etc.
