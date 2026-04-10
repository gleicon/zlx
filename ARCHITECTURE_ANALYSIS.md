# ZLX Architecture Analysis & Strategic Decision Document

**Date:** 2025-01-10  
**Status:** POST-GEMMA4-LOOP - Time to step back and architect  
**Purpose:** Document learnings, analyze Swift vs Zig paths, plan next phase

---

## 1. CURRENT STATE: What We Built

### ✅ Working Features
- **HTTP API Server**: OpenAI-compatible `/v1/chat/completions` endpoint
- **MLX.zig Integration**: Native Zig bindings to MLX C framework
- **Multi-Model Support**: Qwen 2.5 (✓ working), GPT-OSS, DeepSeek (via llama.cpp backend)
- **Generation Pipeline**: Streaming and non-streaming responses
- **Memory Management**: Custom tracking, model switching, safety margins
- **TurboQuant**: KV-cache compression (integrated but needs verification)
- **Prompt Cache**: Disk-based caching for repeated prompts

### ⚠️ Blocked/Incomplete
- **Gemma 4 E4B**: PLE (Per-Layer Embeddings) architecture - fundamental mismatch
  - Shape errors: 2016 vs 10752 expected
  - RMSNorm dimension mismatches
  - PLE requires per-layer processing we haven't implemented
- **DeepSeek MoE**: Router logic not fully implemented
- **Speculative Decoding**: Draft model integration incomplete

### 🔧 Technical Debt
1. **MLX C Bindings Layer**: Manual and fragile
   - Every new MLX feature requires hand-written C bindings
   - Error handling is C-style (error codes), not Zig-style (error unions)
   - Type conversions between C and Zig are error-prone
2. **Weight Loading**: Hardcoded key prefix stripping (`language_model.`)
3. **Model Detection**: Regex-based architecture detection is brittle
4. **No PLE Architecture**: Can't support Gemma 4, future models with similar patterns

---

## 2. KEY LEARNINGS FROM THE LOOP

### Lesson 1: Architecture Mismatch is Expensive
- **Gemma 4 PLE** seemed like "just another model"
- Reality: Requires fundamentally different tensor flow
- **Cost:** 3+ days, multiple iterations, still not working
- **Root cause:** Assuming all models fit standard Transformer pattern

### Lesson 2: C Bindings are a Bottleneck
- Every MLX feature (RoPE variant, attention mask, new op) requires:
  1. C header update
  2. Zig binding declaration
  3. Manual memory management code
  4. Testing and debugging
- **Swift alternative:** Direct `import MLX` - no bindings needed

### Lesson 3: Weight Format Complexity
- Safetensors → MLX arrays with quantization
- Key prefix stripping (`language_model.`)
- Per-tensor quantization scales/biases
- Different conventions per model family

### Lesson 4: We Need Better Abstractions
Current flow is too coupled:
```
HTTP Handler → Model Registry → MLX.zig → C bindings → MLX C++ → Metal
```

Need cleaner layers:
```
HTTP Handler → Model Interface → Tensor Backend (MLX/Swift/Other)
```

---

## 3. SWIFT vs ZIG: COMPARATIVE ANALYSIS

### Option A: Continue with Zig + MLX C

**Pros:**
- ✅ Already built, working for Qwen/GPT-OSS
- ✅ Zig's comptime for zero-cost abstractions
- ✅ Single binary, no runtime dependencies
- ✅ Fine-grained memory control (GPA, arena allocators)
- ✅ Build system is simple (`zig build`)

**Cons:**
- ❌ **MLX C bindings are manual and brittle** (blocking issue)
- ❌ **New MLX features require C API exposure** (slow, depends on MLX team)
- ❌ **PLE architecture needs significant work** (2-3 days minimum)
- ❌ **Error handling** between C→Zig is messy
- ❌ **No MLX ecosystem** - can't leverage mlx-lm, mlx-vlm
- ❌ **Smaller community** - harder to find help

**Path to Gemma 4:**
1. Implement full PLE pipeline in Zig (per-layer lookups, gating, projection)
2. Debug shape mismatches with quantized weights
3. Test with multiple input sizes
4. **Estimated time:** 3-5 days

---

### Option B: Swift + MLX Swift (Updated: Working Code Available)

**UPDATE:** Concrete Swift implementation provided showing:
- ✅ **PLE Block**: 15 lines of Swift implements full PLE pipeline
- ✅ **HTTP Server**: Hummingbird library (similar to httpz)
- ✅ **Model Loading**: `MLXLMCommon` handles weight loading
- ✅ **Single Binary**: `swift build -c release` produces `.build/release/MyCustomMLXServer`
- ✅ **No Python**: Pure Swift + MLX, no subprocesses

**Pros:**
- ✅ **Native MLX**: `import MLX` - no bindings needed
- ✅ **PLE Ready**: Working code provided for Gemma 4 PLE
- ✅ **Ecosystem**: mlx-swift-lm has models, tokenizers, cache management
- ✅ **Type Safety**: Swift catches shape errors at compile time
- ✅ **Debugging**: LLDB, Xcode instruments
- ✅ **Single Binary**: Just like Zig

**Cons:**
- ❌ **Full rewrite**: All Zig code → Swift (1-2 weeks)
- ❌ **TurboQuant**: Need to reimplement (Swift has quantized KV helpers)
- ❌ **Prompt Cache**: Need to reimplement
- ❌ **Build**: SPM instead of `zig build`
- ❌ **Linux**: macOS-only for now

**Path to Gemma 4:**
1. Create Swift project (Package.swift provided)
2. Copy PLE block code (15 lines, shown above)
3. Integrate with Hummingbird HTTP server
4. **Estimated time:** 3-5 days (working code provided!)

---

## 4. FEATURE PRESERVATION MATRIX

| Feature | Zig + MLX C | Swift + MLX | Notes |
|---------|-------------|-------------|-------|
| **Qwen 2.5** | ✅ Working | ✅ Via mlx-swift | Native in Swift |
| **GPT-OSS** | ✅ Working | ✅ Via mlx-swift | Native in Swift |
| **DeepSeek** | ⚠️ llama.cpp | ✅ Native MLX | Better in Swift |
| **Gemma 4** | ❌ Blocked | ✅ Working | **Swift wins** |
| **TurboQuant** | ✅ Integrated | ❌ Reimplement | **Zig wins** |
| **Speculative** | ⚠️ Partial | ✅ Native | Swift wins |
| **Prompt Cache** | ✅ Working | ❌ Reimplement | **Zig wins** |
| **Multi-model** | ✅ Working | ❌ Reimplement | **Zig wins** |
| **Memory tracking** | ✅ Working | ❌ Reimplement | **Zig wins** |
| **Build simplicity** | ✅ `zig build` | ❌ Xcode/SPM | **Zig wins** |

---

## 5. PERFORMANCE CONSIDERATIONS

### Token Speed
- **Zig:** Direct C calls, minimal overhead
- **Swift:** Objective-C runtime overhead for MLX bridge, but MLX Swift is optimized
- **Winner:** Likely comparable; Swift's compiler is very good

### Memory
- **Zig:** Explicit control, custom allocators
- **Swift:** ARC, less control but very optimized
- **Winner:** Zig for control, Swift for ease

### Startup Time
- **Zig:** ~100ms (single binary)
- **Swift:** ~200-500ms (runtime init)
- **Winner:** Zig

---

## 6. DECISION FRAMEWORK

### If we continue with Zig:
**Pros:** Keep existing work, build simplicity, memory control  
**Cons:** Manual bindings forever, PLE takes 3-5 days, future models may have similar issues  
**Best for:** When we want maximum control and can afford implementation time

### If we rewrite in Swift:
**Pros:** Ecosystem access, Gemma 4 works today, better debugging  
**Cons:** 1-2 week rewrite, lose TurboQuant/prompt cache features (temporarily)  
**Best for:** When we want ecosystem leverage and faster feature velocity

---

## 7. MY RECOMMENDATION

**Hybrid Approach (Best of Both):**

1. **Short term:** Fix Zig implementation for PLE
   - Finish Gemma 4 properly in Zig
   - Document the architecture for future models
   - Time: 3-5 days

2. **Medium term:** Evaluate Swift MLX bindings
   - Create small Swift prototype with Gemma 4
   - Compare token/s, memory, developer experience
   - Time: 2-3 days

3. **Decision point:** After evaluation
   - If Swift is significantly better: Plan migration
   - If comparable: Stay with Zig, improve abstractions
   - Time: 1 day decision

**Why this approach:**
- We don't abandon working code
- We make an informed decision with data
- We preserve optionality

---

## 8. QUESTIONS TO RESOLVE

1. **Is there a middle path?** Can we use Swift MLX from Zig via C interop?
2. **What's the real cost of losing TurboQuant?** Is it critical for your use case?
3. **How important is build simplicity?** Is `zig build` vs Xcode a dealbreaker?
4. **Do we need cross-platform?** Linux support important?
5. **What's the next model after Gemma 4?** Will Swift ecosystem cover it faster?

---

## 9. DOCUMENTS TO UPDATE

- [ ] ARCHITECTURE.md - Add PLE architecture notes
- [ ] CONVENTIONS.md - Document model integration patterns
- [ ] CLAUDE.md - Add this analysis as decision record
- [ ] docs/MODEL_IMPLEMENTATION_GUIDE.md - Add Gemma 4 learnings
- [ ] .planning/ROADMAP.md - Update with decision outcome

---

**Next Step:** User decision on path forward (GSD discuss-phase)
