# Migration Guide: Zig → Swift

**From:** zlx (Zig + MLX C bindings)  
**To:** ZLXServer (Swift + native MLX)  
**Reason:** Gemma 4 PLE architecture support  
**Timeline:** 1-2 weeks for feature parity

---

## 🎯 Why We're Migrating

### The Problem
Gemma 4 E4B uses Per-Layer Embeddings (PLE) - a fundamentally different architecture:
- Per-layer embedding lookups
- Context-aware gating
- Special normalization paths

In **Zig**: Would require 3-5 days implementing manual PLE pipeline with MLX C bindings.

In **Swift**: 15 lines of code using native MLX.

### The Solution
Swift provides:
- `import MLX` - no bindings needed
- Working PLE implementation (provided)
- Access to mlx-swift-lm ecosystem
- Better debugging (Xcode, LLDB)
- Still single binary, no Python

---

## 📊 Feature Comparison

| Feature | Zig (Old) | Swift (New) | Notes |
|---------|-----------|-------------|-------|
| **Gemma 4** | ❌ Blocked | ✅ Working | Main driver for migration |
| **Qwen 2.5** | ✅ Native | ✅ Via MLXLMCommon | Ecosystem wins |
| **GPT-OSS** | ✅ Native | ✅ Via MLXLMCommon | Ecosystem wins |
| **TurboQuant** | ✅ Custom | ⏳ Port needed | 1-2 days work |
| **Prompt Cache** | ✅ Custom | ⏳ Port needed | 1-2 days work |
| **Memory Track** | ✅ Custom | ⏳ Port needed | 1-2 days work |
| **HTTP Server** | ✅ httpz | ✅ Hummingbird | Comparable |
| **Build** | ✅ `zig build` | ✅ `swift build` | Comparable |
| **Debug** | ⚠️ Printf | ✅ LLDB/Xcode | Swift wins |
| **Bindings** | ❌ Manual | ✅ Native | Swift wins |

---

## 🗺️ Migration Path

### Phase 1: Foundation (Days 1-2)
**Goal:** HTTP server responding, model loading stub

**Files to create:**
1. `Package.swift` - Dependencies
2. `Sources/ZLXServer/main.swift` - HTTP server, CLI
3. `Sources/ZLXServer/Gemma4PLEBlock.swift` - PLE implementation
4. `Sources/ZLXServer/OpenAITypes.swift` - API types

**Verification:**
```bash
curl http://localhost:8080/health
# Should return {"status": "ok"}
```

---

### Phase 2: Model Integration (Days 3-5)
**Goal:** Load Gemma 4 and generate tokens

**Key classes:**
```swift
// Custom model integrating PLE
class Gemma4Model: Module {
    let embedTokens: Embedding
    let layers: [Gemma4DecoderLayer]  // Uses PLEBlock
    let norm: RMSNorm
    
    func generate(tokens: [Int], cache: KVCache?) async -> String
}
```

**Port from Zig:**
- `src/mlx.zig/src/gemma4.zig` → `Gemma4Model.swift`
- `src/inference/generator.zig` → Generation loop in Swift
- `src/chat/templates.zig` → Chat template handling

**Verification:**
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -d '{"model": "gemma-4-e4b", "messages": [{"role": "user", "content": "Hi"}]}'
# Should return valid OpenAI response
```

---

### Phase 3: Feature Parity (Days 6-10)
**Goal:** All features from Zig version working

**Port list:**
1. **TurboQuant** (2-3 days)
   - Port Walsh-Hadamard kernels
   - Integrate with KVCache
   - Add quantization parameters

2. **Prompt Cache** (1-2 days)
   - Disk-based KV cache storage
   - Index management
   - Cache lookup on request

3. **Memory Tracking** (1 day)
   - Track allocations
   - Enforce budgets
   - Auto-cleanup

4. **Multi-Model** (1 day)
   - Model registry
   - Hot-swapping
   - Memory estimation

**Verification:**
```bash
# All models working
./test_models.sh  # Should pass for all models
```

---

### Phase 4: Polish (Days 11-14)
**Goal:** Production-ready

**Tasks:**
- Streaming SSE responses
- Error handling & logging
- Configuration files
- Documentation
- Performance tuning

---

## 🔄 Porting Specific Components

### 1. HTTP Handlers

**Zig (httpz):**
```zig
pub fn chatCompletions(ctx: *HandlerContext, req: Request, res: Response) !void {
    const body = try req.json(ChatCompletionRequest);
    const result = try generate(ctx, body);
    try res.json(result);
}
```

**Swift (Hummingbird):**
```swift
app.router.post("/v1/chat/completions") { request async throws -> HBResponse in
    let body = try await request.decode(as: OpenAIChatRequest.self)
    let result = try await generate(body)
    return HBResponse(json: result)
}
```

**Key differences:**
- Swift uses async/await (cleaner)
- Hummingbird has similar routing to httpz
- JSON encoding/decoding is similar

---

### 2. Model Loading

**Zig (Manual):**
```zig
const model = try Gemma4Transformer.init(allocator, model_path);
try loadSafetensors(weights_hash, model_path);
```

**Swift (MLXLMCommon):**
```swift
let modelContainer = try await loadGemma4Model(path: modelPath)
// MLXLMCommon handles weight loading
```

**Key differences:**
- Swift uses ecosystem (less code)
- Weight loading is automatic
- But: need custom loader for PLE

---

### 3. Generation Loop

**Zig:**
```zig
while (tokens_generated < max_tokens) {
    try transformer.generate(&next_token, input, cache);
    if (isEos(next_token)) break;
    output.append(next_token);
}
```

**Swift:**
```swift
for _ in 0..<maxTokens {
    let logits = model(input, cache: kvCache)
    let nextToken = sample(logits, temperature: temp)
    if eosTokens.contains(nextToken) { break }
    output.append(nextToken)
}
```

**Key differences:**
- Very similar logic
- Swift has better type safety
- MLX operations look the same

---

### 4. KV Cache

**Zig (Custom):**
```zig
pub const KVCache = struct {
    k: Array,
    v: Array,
    // TurboQuant compression here
};
```

**Swift (Extend MLXLMCommon):**
```swift
class TurboQuantCache: KVCache {
    // Add Walsh-Hadamard compression
    // Port from Zig TurboQuant
}
```

**Key differences:**
- Need to conform to existing protocol
- Port compression kernels
- More work but doable

---

## 🧪 Testing Strategy

### Unit Tests (Swift)
```bash
swift test
```

### Integration Tests
```bash
# Start server
.build/release/ZLXServer --model gemma-4-e4b &

# Test endpoints
curl http://localhost:8080/health
curl http://localhost:8080/v1/models
curl -X POST http://localhost:8080/v1/chat/completions -d '{...}'

# Compare output with expected
```

### Performance Tests
```bash
# Token/s measurement
# Memory usage
# Compare with Zig baseline
```

---

## 📚 Learning Resources

### Swift
- [Swift Tour](https://docs.swift.org/swift-book/documentation/the-swift-programming-language/guidedtour/)
- [Concurrency](https://docs.swift.org/swift-book/documentation/the-swift-programming-language/concurrency/)

### MLX Swift
- [mlx-swift examples](https://github.com/ml-explore/mlx-swift/tree/main/Examples)
- [mlx-swift-lm models](https://github.com/ml-explore/mlx-swift-lm/tree/main/Libraries/MLXLLM/Models)

### Hummingbird
- [Hummingbird docs](https://hummingbird-project.github.io/hummingbird-docs/)

---

## 🎯 Success Criteria

**Before merge to main:**
- [ ] All models working (Qwen, GPT-OSS, Gemma 4)
- [ ] HTTP API compatible with OpenAI spec
- [ ] Token/s comparable to Zig version
- [ ] Single binary output
- [ ] No Python dependencies
- [ ] Tests passing

**Performance targets:**
- Qwen 2.5: >40 tok/s (match Zig)
- Gemma 4: >30 tok/s (new!)
- Memory: <12GB for 4-bit models

---

## 🚨 Risk Mitigation

**Risk:** Swift rewrite takes longer than expected  
**Mitigation:** Preserve Zig branch, can switch back

**Risk:** Performance worse than Zig  
**Mitigation:** Profile and optimize, MLX Swift is well-tuned

**Risk:** Missing features in Swift  
**Mitigation:** Port incrementally, feature flags

**Risk:** Swift learning curve  
**Mitigation:** Start with provided code, iterate

---

## 📞 Questions?

- **Zig branch:** `git checkout zig-mlx-gemma4-analysis`
- **Architecture analysis:** See `ARCHITECTURE_ANALYSIS.md`
- **Decision record:** See `DECISION_SUMMARY.md`

---

**Status:** Foundation code in place. Ready to implement model loading.
