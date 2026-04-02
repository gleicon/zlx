# TurboQuant Porting Analysis

## Executive Summary

**Recommendation: DEFER TurboQuant integration to v1.2 or later.**

TurboQuant (arozanov/turboquant-mlx) is a Python-only library with Metal kernels embedded as JIT-compiled Python strings. Direct binding from Zig is impossible without significant porting effort (40+ hours minimum).

## Source Repository Analysis

### Repository Details
- **Repository:** https://github.com/arozanov/turboquant-mlx
- **Language:** 100% Python (confirmed via GitHub linguist)
- **License:** Apache 2.0
- **Dependencies:** MLX (Python), NumPy

### Key Files to Port

#### 1. metal_kernels.py (Serial WHT - v1)
- **Contents:** Metal shader source strings for Walsh-Hadamard Transform
- **Implementation:** Python functions wrapping `mx.fast.metal_kernel()` JIT calls
- **Complexity:** ~200-300 lines of Python containing Metal shader code
- **Algorithm:** Serial butterfly pattern, O(d log d) per vector
- **Performance:** Baseline (simpler, slower)

#### 2. metal_kernels_v2.py (Parallel WHT - v2)
- **Contents:** Parallel WHT with threadgroup barriers
- **Implementation:** Similar JIT pattern but with threadgroup memory
- **Complexity:** ~300-400 lines (more complex synchronization)
- **Algorithm:** Parallel butterfly with threadgroup barriers
- **Performance:** 1.3-2.3x speedup over v1
- **Challenge:** Threadgroup barrier semantics differ between Metal versions

#### 3. quantizer.py (PolarQuant Algorithm)
- **Contents:** Lloyd-Max quantization/dequantization
- **Implementation:** MLX array operations + Metal kernels
- **Complexity:** ~150 lines
- **Algorithm:**
  1. Random sign flipping (for Gaussianization)
  2. Walsh-Hadamard Transform
  3. Lloyd-Max quantization with precomputed codebook
  4. Bit packing (2 values/byte for 4-bit, complex packing for 3-bit)

#### 4. rotation.py (WHT Logic)
- **Contents:** Walsh-Hadamard Transform mathematical operations
- **Implementation:** MLX array ops for CPU-side WHT reference
- **Complexity:** ~100 lines

#### 5. cache.py (TurboQuantKVCache)
- **Contents:** Drop-in KVCache replacement for mlx-lm
- **Implementation:** Layer-adaptive compression logic
- **Complexity:** ~200 lines
- **Key Feature:** First/last N layers kept in FP16 for quality

## Porting Options Evaluation

### Option A: C++ Custom Ops via mlx-c Upgrade (Recommended if Proceeding)

**Approach:**
1. Upgrade mlx-c from v0.1.2 to v0.4.1+ (custom ops API added in v0.2.0+)
2. Implement TurboQuant as C++ primitives using MLX C++ API
3. Expose via mlx-c C API
4. Bind from Zig through `@cImport`

**Effort:** 40-60 hours

**Breakdown:**
- mlx-c upgrade & compatibility testing: 8-12 hours
- Metal kernel extraction & C++ porting: 16-20 hours
- C wrapper API design: 4-6 hours
- Zig bindings & integration: 8-12 hours
- Testing & optimization: 8-10 hours

**Risk:** HIGH
- mlx-c v0.1.2 is pinned in MLX.zig for compatibility
- Upgrade to v0.4.1+ may break MLX.zig bindings
- MLX.zig might need updates to work with newer mlx-c
- Potential cascade of breaking changes

**Confidence:** LOW - Major dependency upgrade is risky

---

### Option B: Direct Metal in Zig

**Approach:**
1. Extract Metal kernel source strings from Python files
2. Implement Metal kernel compilation via MLX's C++ `metal_kernel()` API
3. Write Zig wrappers for Metal buffer management and dispatch
4. Implement PolarQuant algorithm in Zig

**Effort:** 60-80 hours

**Breakdown:**
- Kernel source extraction: 4-6 hours
- Metal/Zig interop research: 8-12 hours
- Kernel manager implementation: 12-16 hours
- WHT algorithm port: 16-20 hours
- Quantization/dequantization: 12-16 hours
- Integration & testing: 12-16 hours

**Risk:** MEDIUM
- MLX C++ Metal dispatch API not well-documented
- Metal threadgroup barriers are subtle and device-dependent
- May require deep MLX internals knowledge

**Confidence:** MEDIUM - Doable but complex

---

### Option C: MLX Array Operations (Approximation)

**Approach:**
1. Use existing mlx-c array operations (no custom Metal kernels)
2. Implement "Hadamard-ish" transform using MLX matrix ops
3. Use MLX quantization primitives
4. Accept lower performance and compression ratio

**Effort:** 20-30 hours

**Breakdown:**
- MLX quantization op research: 4-6 hours
- WHT approximation: 8-12 hours
- Quantization integration: 4-6 hours
- Testing: 4-6 hours

**Risk:** LOW
- Uses existing, well-tested MLX operations
- No custom Metal kernel complexity

**Trade-offs:**
- Loses kernel fusion benefits (separate ops vs fused kernels)
- Lower compression ratio (~3x vs 4.6x target)
- Slower than TurboQuant (~50-70% speed)
- **Not really TurboQuant** - just generic quantization

**Confidence:** HIGH - But loses key benefits

---

### Option D: Python Sidecar Process

**Approach:**
1. Keep TurboQuant as Python subprocess
2. IPC between Zig and Python via shared memory or sockets
3. Zig handles HTTP/API, Python handles compression

**Effort:** 40-50 hours

**Risk:** MEDIUM
- Complex IPC layer
- Defeats "no Python in final binary" PRD requirement
- Synchronization overhead

**Verdict:** NOT RECOMMENDED - Violates core PRD constraint

## Algorithm Deep Dive

### Why TurboQuant Works

1. **Hadamard Rotation:** WHT rotates values to approximate Gaussian distribution
2. **Gaussian Benefits:** Lloyd-Max quantization is optimal for Gaussian inputs
3. **Layer Adaptivity:** First/last layers more sensitive to quantization errors
4. **Kernel Fusion:** Metal kernels fuse WHT + quantization for speed

### Key Metal Kernels Needed

```metal
// WHT Serial Kernel (simplified)
kernel void wht_serial(
    device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    // Butterfly pattern: stride halves each iteration
    // O(d log d) operations
}

// WHT Parallel Kernel (simplified)
kernel void wht_parallel(
    device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    // Threadgroup barriers for butterfly sync
    // threadgroup memory for intermediate results
}

// Quantization Kernel (simplified)
kernel void quantize(
    device const float* input [[buffer(0)]],
    device uint8_t* output [[buffer(1)]],
    device const float* codebook [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    // Find nearest codebook entry
    // Pack into bytes
}
```

### Lloyd-Max Codebook

Precomputed for Gaussian distribution with mean=0, std=1:

```
4-bit levels (16 values): [-2.15, -1.57, -1.21, -0.94, -0.71, -0.51, -0.33, -0.16,
                             0.16,  0.33,  0.51,  0.71,  0.94,  1.21,  1.57,  2.15]

3-bit levels (8 values):  [-1.75, -1.05, -0.63, -0.31, 0.31, 0.63, 1.05, 1.75]
```

Can be precomputed offline and embedded as constant arrays.

## Recommendation

### Primary Recommendation: DEFER

**Rationale:**

1. **High effort (40+ hours minimum)** - Significant engineering investment
2. **High risk (mlx-c upgrade)** - May destabilize existing MLX.zig bindings
3. **Complex porting** - Metal kernels in Python strings require careful extraction
4. **Adequate alternatives** - Prompt caching (Phase 6) provides good performance
5. **Better ROI elsewhere** - Speculative Decoding (Phase 8) offers clearer path to speedup

### Alternative Path: v1.2+ Implementation

If TurboQuant is desired in future:

1. **Phase 1:** Upgrade mlx-c to v0.4.1+ in fork of MLX.zig
2. **Phase 2:** Verify MLX.zig compatibility (may require fixes)
3. **Phase 3:** Implement C++ custom ops wrapper for TurboQuant kernels
4. **Phase 4:** Add Zig bindings through upgraded mlx-c
5. **Phase 5:** Integrate with KvCompressor interface

Estimated timeline: 2-3 weeks of focused work

### Immediate Next Steps

1. ✅ Execute 07-01 plan to create stub framework
2. ⏭️ **Skip to Phase 8:** Speculative Decoding (clearer implementation path)
3. 📋 **Update ROADMAP:** Mark Phase 7 as deferred, prioritize Phase 8
4. 📝 **Document decision:** Record in PROJECT.md why TurboQuant was deferred

## Metrics Comparison

| Feature | Current (Phase 6) | TurboQuant (If Implemented) | Speculative Decoding (Phase 8) |
|---------|---------------------|----------------------------|------------------------------|
| Speedup | Baseline (prompt cache) | 98% of FP16 speed | 1.5-2.5x tokens/sec |
| Memory | No reduction | 4.6x reduction | Slight increase |
| Effort | Completed | 40-60 hours | 20-30 hours |
| Risk | Low | High | Low-Medium |
| Confidence | High | Low-Medium | High |

## Conclusion

TurboQuant is a sophisticated optimization that would benefit zlx, but the porting effort and risks outweigh the benefits for v1.1. The stub framework created in 07-01 preserves the option to implement it later while allowing the project to move forward with higher-ROI features like Speculative Decoding.

**Decision: NO-GO on TurboQuant for v1.1. Proceed to Phase 8.**
