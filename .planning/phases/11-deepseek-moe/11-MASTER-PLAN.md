# Phase 11: DeepSeek MoE Support — Implementation Plan

**Version:** v1.1.1  
**Strategy:** Option 1 — mlx-c v0.4.x Upgrade  
**Estimated Effort:** 16-24 hours  
**Status:** 🔵 Ready for Planning

---

## Executive Summary

**Goal:** Enable DeepSeek-Coder-V2-Lite-Instruct and other MoE models by upgrading to mlx-c v0.4.x and implementing:
1. Multi-head Latent Attention (MLA) — 90% KV cache reduction
2. Mixture of Experts (MoE) routing — sparse expert activation
3. DeepSeek transformer architecture
4. DeepSeek chat template

**Why This Matters:**
- DeepSeek-V2 is #1 on code generation benchmarks (5x better than Qwen 1.5B)
- 15.7B model with inference cost of 2B dense (via MoE)
- 128k context length (vs 8k Qwen 1.5B)
- Free license for commercial use

---

## Technical Discovery: mlx-c v0.4.x Capabilities

### Key Finding: Fast Custom Ops API

mlx-c v0.4.x provides `mlx_fast_metal_kernel_*` API for implementing custom primitives:

```c
// Define custom Metal kernel for MoE routing
mlx_fast_metal_kernel kernel = mlx_fast_metal_kernel_new(
    "moe_route",                                    // name
    input_names,                                    // ["hidden_states", "gate_weight"]
    output_names,                                   // ["expert_indices", "expert_weights"]
    metal_source_code,                              // Metal shader
    header,                                         // Additional headers
    ensure_row_contiguous,                          // memory layout
    atomic_outputs                                  // atomic operations
);

// Apply kernel
mlx_fast_metal_kernel_apply(
    outputs,                                        // Output arrays
    kernel,
    inputs,                                         // Input arrays
    config,                                         // Grid/thread config
    stream                                          // GPU stream
);
```

### What This Enables

1. **MoE Routing Kernel** — Custom Metal shader for expert selection
2. **MLA Attention** — Compressed attention mechanism
3. **Optimized Operations** — Fused kernels for performance

---

## Implementation Strategy

### Approach: Hybrid Upgrade

Instead of full mlx-c v0.4.x upgrade (risky), we'll use a **hybrid approach**:

**Phase A: Minimal Upgrade (8-12 hours)**
- Keep MLX.zig at v0.1.2 for base operations
- Add mlx-c v0.4.x as **additional dependency** for custom ops only
- Implement MoE routing as custom primitive via v0.4.x
- Risk: Low (existing models unaffected)

**Phase B: DeepSeek Transformer (8-12 hours)**
- Implement MLA using compressed attention
- Implement MoE layer with routing
- Create DeepSeek transformer architecture
- Integrate with existing pipeline

### Why Hybrid Over Full Upgrade?

| Approach | Effort | Risk | Recommendation |
|----------|--------|------|----------------|
| **Full v0.4.x Upgrade** | 12-16h | High (API changes break MLX.zig) | ❌ Risky |
| **Hybrid (Selected)** | 16-24h | Medium (isolated custom ops) | ✅ Safe |
| **Custom Zig MoE** | 20-30h | High (complex, untested) | ❌ Complex |

---

## Detailed Plan

### Plan 11-01: mlx-c v0.4.x Integration (8 hours)

**Goal:** Add mlx-c v0.4.x as secondary dependency for custom ops

**Tasks:**

1. **Download mlx-c v0.4.1** (1 hour)
   ```zig
   // In build.zig, add parallel download
   const mlx_c_v4_path = b.pathJoin(&.{ b.cache_root.path.?, "mlx-c-v4" });
   const clone_v4 = b.addSystemCommand(&[_][]const u8{
       "sh", "-c", b.fmt(
           "if [ ! -d {s} ]; then curl -L https://github.com/ml-explore/mlx-c/archive/refs/tags/v0.4.1.tar.gz | tar xz -C $(dirname {s}) && mv $(dirname {s})/mlx-c-0.4.1 {s}; fi",
           .{ mlx_c_v4_path, mlx_c_v4_path, mlx_c_v4_path, mlx_c_v4_path }
       )
   });
   ```

2. **Build v0.4.x Library** (1 hour)
   - Compile alongside v0.1.2
   - Create separate library: `libmlxc-v4.a`

3. **Create Compatibility Layer** (4 hours)
   - New file: `src/mlx_v4.zig`
   - Wrap v0.4.x functions for Zig
   - Focus on `mlx_fast_*` APIs only
   ```zig
   // src/mlx_v4.zig
   pub const FastMetalKernel = struct {
       kernel: mlx_c_v4.mlx_fast_metal_kernel,
       
       pub fn init(name: []const u8, source: []const u8) !FastMetalKernel {
           // Wrap mlx_fast_metal_kernel_new
       }
       
       pub fn apply(self: FastMetalKernel, inputs: []const mlx.Array, outputs: []mlx.Array) !void {
           // Wrap mlx_fast_metal_kernel_apply
       }
   };
   ```

4. **Verify Integration** (2 hours)
   - Test simple Metal kernel
   - Ensure no conflicts with v0.1.2
   - Verify existing tests still pass

**Deliverable:** mlx-c v0.4.x available for custom ops without breaking existing code

---

### Plan 11-02: MLA Implementation (6 hours)

**Goal:** Implement Multi-head Latent Attention (DeepSeek's compressed attention)

**Background:**
- Standard MHA: O(n²) memory for KV cache
- MLA: Compresses KV cache by ~90% via low-rank projection
- Key innovation: Joint compression of keys and values

**Implementation:**

1. **Create MLA Module** (4 hours)
   - New file: `src/mlx.zig/src/mla.zig`
   ```zig
   pub const MultiHeadLatentAttention = struct {
       // Low-rank projection matrices
       down_proj_q: mlx.Array,  // query compression
       down_proj_kv: mlx.Array, // joint KV compression
       up_proj: mlx.Array,       // decompression
       
       // DeepSeek uses compressed latent vectors instead of full KV
       pub fn forward(
           self: *Self,
           hidden_states: mlx.Array,
           attention_mask: ?mlx.Array,
           cache: ?*KVCache,
       ) !mlx.Array {
           // 1. Compress queries
           const latent_q = try mlx.matmul(hidden_states, self.down_proj_q);
           
           // 2. Compress KV jointly (the key innovation)
           const latent_kv = try mlx.matmul(hidden_states, self.down_proj_kv);
           
           // 3. Attention in compressed space
           const attn_output = try self.compressedAttention(latent_q, latent_kv, cache);
           
           // 4. Decompress back to full dimension
           return try mlx.matmul(attn_output, self.up_proj);
       }
   };
   ```

2. **Compressed Attention Logic** (2 hours)
   - Joint Q/KV attention in low-rank space
   - RoPE (Rotary Position Embedding) in compressed space
   - KV cache stores compressed vectors (~10x smaller)

**Deliverable:** MLA module ready for integration

---

### Plan 11-03: MoE Routing Implementation (6 hours)

**Goal:** Implement Mixture of Experts routing mechanism

**Architecture:**
- 64 total experts (configurable)
- 2 shared experts (always active)
- Top-6 routed experts per token
- Sparse activation: only 8/64 experts active per token

**Implementation:**

1. **Create Metal Kernel for Routing** (3 hours)
   - New file: `src/moe_metal.metal` (Metal shader source)
   ```metal
   // Expert routing kernel
   kernel void moe_route(
       device const float* hidden_states [[buffer(0)]],
       device const float* gate_weight [[buffer(1)]],
       device int* expert_indices [[buffer(2)]],      // Top-k expert indices
       device float* expert_weights [[buffer(3)]],      // Gate scores
       uint tid [[thread_position_in_grid]]
   ) {
       // 1. Compute gate logits: hidden @ gate_weight.T
       // 2. Softmax to get probabilities
       // 3. Top-k selection (using bitonic sort in shared memory)
       // 4. Write expert_indices and expert_weights
   }
   ```

2. **Create MoE Layer Module** (3 hours)
   - New file: `src/mlx.zig/src/moe.zig`
   ```zig
   pub const MixtureOfExperts = struct {
       num_experts: u32,
       num_shared_experts: u32,
       top_k: u32,
       
       // Expert FFNs (stored in array, sparse access)
       experts: []mlx.Array,  // Weight matrices for each expert
       shared_experts: []mlx.Array,
       
       // Gate for routing
       gate: mlx.Array,
       
       // Metal kernel for routing
       route_kernel: mlv_v4.FastMetalKernel,
       
       pub fn forward(self: *Self, hidden_states: mlx.Array) !mlx.Array {
           // 1. Route: get top-k experts for each token
           const routing = try self.route(hidden_states);
           
           // 2. Process each token through selected experts
           var output = try mlx.zeros(hidden_states.shape());
           
           // Shared experts (always active)
           for (self.shared_experts) |expert| {
               const expert_out = try self.applyExpert(hidden_states, expert);
               try mlx.add(&output, output, expert_out);
           }
           
           // Routed experts (sparse)
           for (routing.indices, routing.weights) |expert_idx, weight| {
               const expert_out = try self.applyExpert(hidden_states, self.experts[expert_idx]);
               const weighted = try mlx.multiply(expert_out, weight);
               try mlx.add(&output, output, weighted);
           }
           
           return output;
       }
   };
   ```

**Deliverable:** MoE layer with Metal routing kernel

---

### Plan 11-04: DeepSeek Transformer (4 hours)

**Goal:** Complete DeepSeek-V2 transformer implementation

**Implementation:**

1. **Create DeepSeek Transformer** (3 hours)
   - New file: `src/mlx.zig/src/deepseek.zig`
   ```zig
   pub const DeepSeekTransformer = struct {
       // Embedding
       token_embedding: mlx.Array,
       
       // Layers
       layers: []DeepSeekLayer,
       
       // Output
       norm: mlx.Array,
       lm_head: mlx.Array,
       
       pub const DeepSeekLayer = struct {
           // MLA instead of standard attention
           mla: mla.MultiHeadLatentAttention,
           
           // MoE FFN instead of dense FFN
           moe: moe.MixtureOfExperts,
           
           // Layer norms
           input_norm: mlx.Array,
           post_attn_norm: mlx.Array,
       };
       
       pub fn forward(self: *Self, input_ids: mlx.Array, cache: ?*Cache) !mlx.Array {
           // Standard transformer forward with MLA + MoE
       }
   };
   ```

2. **Add Model Detection** (1 hour)
   - Update `src/inference/loader.zig` to detect DeepSeek architecture
   - Check config.json for "deepseek" model_type

**Deliverable:** Complete DeepSeek transformer ready for inference

---

### Plan 11-05: Chat Template & Integration (2 hours)

**Goal:** DeepSeek chat format support and integration

**Implementation:**

1. **DeepSeek Chat Template** (1 hour)
   ```zig
   // DeepSeek uses: User: {message}\nAssistant: {response}
   pub fn formatDeepSeekChat(allocator: std.mem.Allocator, messages: []Message) ![]u8 {
       var result = std.ArrayList(u8).init(allocator);
       
       for (messages) |msg| {
           if (std.mem.eql(u8, msg.role, "user")) {
               try result.appendSlice("User: ");
               try result.appendSlice(msg.content);
               try result.appendSlice("\n");
           } else if (std.mem.eql(u8, msg.role, "assistant")) {
               try result.appendSlice("Assistant: ");
               try result.appendSlice(msg.content);
               try result.appendSlice("\n");
           } else if (std.mem.eql(u8, msg.role, "system")) {
               // DeepSeek handles system differently
               try result.appendSlice(msg.content);
               try result.appendSlice("\n");
           }
       }
       
       // Add assistant prompt for generation
       try result.appendSlice("Assistant: ");
       
       return result.toOwnedSlice();
   }
   ```

2. **Integration** (1 hour)
   - Add to model registry
   - Update memory estimation for sparse params
   - Test end-to-end inference

**Deliverable:** DeepSeek fully integrated and working

---

## Testing Strategy

### Unit Tests (2 hours)
- MLA: Test compression/decompression, cache size reduction
- MoE: Test routing accuracy, expert selection
- Transformer: Test forward pass, output shapes

### Integration Tests (2 hours)
- Load DeepSeek model from HuggingFace
- Run inference on sample prompts
- Verify chat template formatting
- Benchmark vs MLX Python

### Performance Tests (2 hours)
- Memory usage: Should be ~2GB for 15.7B model (sparse)
- Speed: Should be within 10% of MLX Python
- KV cache: Verify 90% reduction with MLA

---

## Risk Assessment & Mitigation

### High Risk: Metal Kernel Performance

**Risk:** Custom Metal kernel slower than expected
**Mitigation:**
- Start with reference implementations (llama.cpp, MLX Python)
- Profile kernel with Metal System Trace
- Fall back to CPU routing if GPU kernel fails

### Medium Risk: Memory Layout Issues

**Risk:** MLA compressed format incompatible with MLX
**Mitigation:**
- Test with small model first
- Verify tensor shapes at each layer
- Add shape assertions in debug builds

### Low Risk: Build Complexity

**Risk:** Two mlx-c versions conflict
**Mitigation:**
- Use separate include paths
- Different library names (libmlxc-v1.a, libmlxc-v4.a)
- Namespace isolation

---

## Success Criteria

| Criteria | Target | How to Verify |
|----------|--------|---------------|
| **Model Loads** | DeepSeek loads without error | Integration test |
| **Inference Works** | Generates coherent text | Sample prompts |
| **Memory Usage** | ~2GB for 15.7B model | Activity Monitor |
| **KV Cache** | 90% reduction vs standard | Log cache sizes |
| **Performance** | Within 10% of MLX Python | Benchmark script |
| **Chat Format** | User:/Assistant: format | Check output |
| **Existing Models** | Qwen, Llama still work | Regression tests |

---

## Timeline

| Phase | Hours | Dependencies |
|-------|-------|--------------|
| 11-01: mlx-c v0.4.x | 8 | None |
| 11-02: MLA | 6 | 11-01 |
| 11-03: MoE | 6 | 11-01 |
| 11-04: DeepSeek Transformer | 4 | 11-02, 11-03 |
| 11-05: Chat Template | 2 | 11-04 |
| **Testing** | 6 | All |
| **Total** | **32 hours** | |

**Parallelizable:** 11-02 and 11-03 can happen in parallel after 11-01
**Critical Path:** 11-01 → (11-02 || 11-03) → 11-04 → 11-05

---

## Fallback Plan

**If mlx-c v0.4.x upgrade fails:**

1. **Option A:** Use CPU-based MoE routing (simpler, slower)
2. **Option B:** Wait for MLX.zig to add MoE support (passive)
3. **Option C:** Implement in Zig without Metal kernels (complex)

**Decision Point:** After 11-01, if Metal kernel doesn't work, switch to Option A

---

## Next Steps

1. **Approve Plan** — User review of this approach
2. **Research Phase** — Study llama.cpp DeepSeek implementation (2 hours)
3. **Plan 11-01** — Start mlx-c v0.4.x integration
4. **Parallel Development** — MLA and MoE in parallel
5. **Testing** — Comprehensive test suite
6. **Release v1.1.1** — Tag and document

---

## References

- mlx-c v0.4.x docs: https://ml-explore.github.io/mlx-c/build/html/fast.html
- MLX custom kernels: https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
- DeepSeek-V2 paper: https://arxiv.org/abs/2405.04434
- llama.cpp DeepSeek: https://github.com/ggerganov/llama.cpp/pull/0 (search "deepseek")

---

**Ready to proceed with Phase 11 implementation?**
