# ZLX Server - Swift MLX Rewrite

**Branch:** `swift-mlx-rewrite`  
**Status:** Foundation code in place - Gemma 4 PLE support incoming  
**Previous:** See `zig-mlx-gemma4-analysis` branch for working Zig implementation

---

## 🎯 Goal

Pure Swift implementation of OpenAI-compatible inference server with:
- ✅ Gemma 4 E4B support (including PLE architecture)
- ✅ Native MLX on Apple Silicon (no Python)
- ✅ Single binary output
- ✅ TurboQuant-style KV cache compression (planned)
- ✅ All features from Zig version

---

## 📁 Structure

```
.
├── Package.swift              # Swift Package Manager manifest
├── Sources/
│   └── ZLXServer/
│       ├── main.swift         # HTTP server & CLI entry point
│       ├── Gemma4PLEBlock.swift    # PLE implementation (15 lines!)
│       └── Gemma4Model.swift       # Custom model with PLE
└── README.md                  # This file
```

---

## 🚀 Quick Start

### 1. Build
```bash
swift build -c release
```

### 2. Run
```bash
.build/release/ZLXServer --model mlx-community/gemma-4-e4b-it-4bit --port 8080
```

### 3. Test
```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-e4b",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

---

## 🏗️ Architecture

### PLE (Per-Layer Embeddings)

The key innovation in Gemma 4 - implemented cleanly in Swift:

```swift
// Gemma4PLEBlock.swift - Just 15 lines!
public func callAsFunction(_ x: MLXArray, pleSlice: MLXArray) -> MLXArray {
    let gate = gelu(inputGate(x))        // Context-aware gating
    let gated = gate * pleSlice           // Combine with token PLE
    let pleOut = norm(projection(gated)) // Project and normalize
    return pleOut
}
```

Compare to Zig: Would need C bindings, manual tensor operations, error handling.

### HTTP Server

Using Hummingbird (similar to httpz):
- OpenAI-compatible endpoints
- Streaming SSE support (planned)
- JSON request/response handling

### Model Loading

Via `MLXLMCommon`:
- Safetensors weight loading
- Tokenizer integration
- Generation parameters

---

## 📋 Implementation Checklist

### Phase 1: Foundation ✅
- [x] Package.swift with dependencies
- [x] Basic HTTP server structure
- [x] OpenAI API types
- [x] PLE block implementation

### Phase 2: Model Integration
- [ ] Custom Gemma4Model with PLE layers
- [ ] Model loading from HuggingFace/local
- [ ] Tokenizer integration
- [ ] Generation loop

### Phase 3: Features
- [ ] Streaming responses
- [ ] Chat template handling
- [ ] Tool calling API
- [ ] Multi-model support

### Phase 4: Performance
- [ ] TurboQuant KV cache port
- [ ] Prompt cache implementation
- [ ] Memory tracking
- [ ] Performance metrics

### Phase 5: Polish
- [ ] Configuration file support
- [ ] Logging
- [ ] Error handling
- [ ] Documentation

---

## 🔄 Migration from Zig

| Feature | Zig Status | Swift Status |
|---------|-----------|--------------|
| Qwen 2.5 | ✅ Working | ⏳ Via MLXLMCommon |
| GPT-OSS | ✅ Working | ⏳ Via MLXLMCommon |
| Gemma 4 | ❌ Blocked | ✅ Foundation ready |
| TurboQuant | ✅ Integrated | ⏳ Need port |
| Prompt Cache | ✅ Working | ⏳ Need port |
| HTTP API | ✅ httpz | ✅ Hummingbird |
| Build | ✅ `zig build` | ✅ `swift build` |

---

## 🧠 Key Decisions

**Why Swift over continuing with Zig?**

1. **Gemma 4 PLE**: 15 lines in Swift vs days in Zig
2. **No bindings**: Native `import MLX` vs manual C interop
3. **Ecosystem**: mlx-swift-lm has models we don't
4. **Debugging**: Xcode, LLDB vs printf debugging
5. **Still single binary**: Meets core constraint

**Trade-offs accepted:**
- 1-2 weeks rewrite time
- Temporarily lose TurboQuant/prompt cache
- macOS-only (was already constraint)

---

## 📚 References

- **Previous branch:** `zig-mlx-gemma4-analysis`
- **Architecture analysis:** `ARCHITECTURE_ANALYSIS.md`
- **Decision summary:** `DECISION_SUMMARY.md`
- **PLE documentation:** `a_swift_rewrite.md`
- **MLX Swift:** https://github.com/ml-explore/mlx-swift
- **MLX Swift LM:** https://github.com/ml-explore/mlx-swift-lm

---

## 🛠️ Development

### Add Dependency
Edit `Package.swift`, then:
```bash
swift package resolve
```

### Run Tests
```bash
swift test
```

### Debug Build
```bash
swift build
.build/debug/ZLXServer --verbose
```

---

## 📄 License

Same as original zlx project.

---

**Next:** Implement model loading and generation loop. See `Sources/ZLXServer/main.swift` TODOs.
