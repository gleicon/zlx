# ZLX Post-Gemma4 Analysis & Strategic Decision

**Date:** 2025-01-10  
**Status:** Phase 18 Analysis Complete - Decision Required  
**Document:** ARCHITECTURE_ANALYSIS.md created with full technical comparison

---

## EXECUTIVE SUMMARY

We've hit a fundamental architectural limitation: **Gemma 4's PLE (Per-Layer Embeddings) architecture breaks our standard transformer assumptions.** 

After 3+ days in the implementation loop, we've learned:
1. PLE requires per-layer lookup tables + per-layer processing (not just per-layer parameters)
2. Our MLX C bindings approach is too slow for implementing novel architectures
3. Swift MLX has working Gemma 4 support today via `mlx-swift-models`

**We need to decide:** Continue with Zig and implement PLE properly, OR pivot to Swift for ecosystem access.

---

## WHAT WE ACCOMPLISHED

### ✅ Phase 18 Analysis Complete
- **PLE Architecture Understood**: Per-layer embeddings (256 dims × 42 layers = 10,752 total)
- **PLE-safe Weights Downloaded**: `FakeRockert543/gemma-4-e4b-it-MLX-4bit` (9.6GB)
- **Model Alias Updated**: `gemma4-e4b` now points to PLE-safe weights
- **Simplified Gemma4.zig**: Standard transformer blocks ready (but PLE processing needed)

### ⚠️ Current Blockers
- **Shape Mismatch**: 2016 vs 10752 tensor dimensions
- **RMSNorm Error**: Embedding dimensions don't match hidden size (PLE artifact)
- **Missing PLE Pipeline**: Need per-layer: lookup → gate → project → norm → inject

---

## THE DECISION

### Option A: Continue with Zig (Finish What We Started)

**What we'd do:**
1. Implement full PLE pipeline in Zig
2. Per-layer embedding lookup from `embed_tokens_per_layer`
3. Per-layer gate (`per_layer_input_gate`), projection, norm
4. Integration with existing transformer blocks
5. Test with PLE-safe weights

**Time:** 3-5 days  
**Pros:**
- Keep existing working code (Qwen, GPT-OSS)
- Single binary, simple build
- Fine-grained memory control
- No rewrite needed

**Cons:**
- Manual MLX C bindings forever
- Future models may have similar issues
- 3-5 days of PLE work only fixes Gemma 4

---

### Option B: Swift MLX (Working Code Available)

**What we'd do:**
1. Create Swift project with provided Package.swift
2. Copy PLE implementation (15 lines provided)
3. Add Hummingbird HTTP server (shown in example)
4. Integrate model loading from `MLXLMCommon`
5. Test Gemma 4 generation

**Time:** 3-5 days (working code provided!)  
**Pros:**
- ✅ Native `import MLX` (no bindings!)
- ✅ **PLE code provided** - Gemma 4 works immediately
- ✅ Single binary output (like Zig)
- ✅ HTTP server (Hummingbird) similar to httpz
- ✅ Can extend KVCache for TurboQuant

**Cons:**
- 1-2 weeks for full feature parity (all models, caches, etc.)
- Rewrite all existing Zig code
- Lose TurboQuant temporarily (need Swift port)
- Lose prompt cache temporarily (need Swift port)

---

## MY RECOMMENDATION (Updated with Swift Code)

**Swift is now the clear winner for Gemma 4.**

The provided Swift code shows:
- PLE implementation in **15 lines** (vs days in Zig)
- Working HTTP server with Hummingbird
- Single binary output
- Native MLX access (no bindings)

**Recommended Path:**

1. **Immediate**: Start Swift rewrite
   - Use provided code as foundation
   - 3-5 days to working Gemma 4
   - 1-2 weeks for full feature parity

2. **Preserve Zig code** in branch:
   - Qwen 2.5 works perfectly
   - Can reference implementations
   - Fallback if needed

3. **Swift advantages:**
   - Gemma 4 works today (code provided)
   - Ecosystem access (mlx-swift-lm)
   - Better debugging
   - No C bindings maintenance
   - Still single binary, no Python

**Trade-offs accepted:**
- 1-2 week rewrite time
- Temporarily lose TurboQuant (reimplement in Swift)
- Temporarily lose prompt cache (reimplement in Swift)
- macOS-only (acceptable per constraints)

---

## KEY LEARNINGS (Now in CLAUDE.md)

1. **Model Architecture Assumptions Are Dangerous** - Gemma 4 PLE breaks standard patterns
2. **MLX C Bindings Are The Bottleneck** - Every new feature needs hand-written bindings
3. **Weight Loading Complexity** - Key prefixes, quantization scales, per-family conventions
4. **Quantization + Special Architectures = Problems** - PLE layers can't be quantized
5. **Swift vs Zig Decision Looms** - Bindings vs ecosystem tradeoff
6. **Better Abstractions Needed** - Current HTTP→Registry→MLX.zig→C→Metal is too coupled

---

## NEXT STEPS

**Your decision needed on:**

1. **Should we evaluate Swift MLX?** (2-3 day prototype)
2. **Should we finish PLE in Zig instead?** (3-5 days)
3. **Or stick with Qwen for now?** (works perfectly today)

**Documents updated:**
- ✅ ARCHITECTURE_ANALYSIS.md - Full technical comparison
- ✅ CLAUDE.md - Key learnings added
- ✅ ARCHITECTURE.md - System overview with PLE notes
- ✅ ROADMAP.md - Phase 18 marked with decision point

**Ready for:**
- GSD discuss-phase (detailed planning)
- Or immediate implementation of chosen path

---

## QUESTIONS FOR YOU

1. **Is Gemma 4 a must-have, or is Qwen sufficient for your coding agent?**
2. **How important is build simplicity (`zig build` vs Xcode)?**
3. **Would you accept a 1-2 week rewrite for better long-term maintainability?**
4. **Do you need cross-platform (Linux), or is macOS-only acceptable?**
5. **Is TurboQuant critical, or is standard KV cache acceptable?**

Your answers will determine which path we take.

---

**See:** ARCHITECTURE_ANALYSIS.md for the full technical deep-dive
