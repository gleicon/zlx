# Testing Strategy - ZLX Swift Implementation

**Branch:** swift-mlx-rewrite  
**Status:** Build ✅, MLX Resources ⚠️  
**Date:** 2025-01-10

---

## 🎯 Testing Philosophy

Given our constraints:
- ✅ Swift code compiles and builds
- ⚠️ MLX Metal libraries need setup for real inference
- ✅ HTTP server structure complete
- ✅ All models registered

**Testing Approach:**
1. **Stub Mode** - Test HTTP/API without MLX (immediate)
2. **Unit Tests** - Test logic with mocked MLX (fast)
3. **Integration with Zig** - Compare outputs (validation)
4. **Real MLX** - Full e2e after metallib setup (final)

---

## 📊 Test Levels

### Level 1: Stub Mode (No MLX Required) ✅ Ready Now

**Purpose:** Verify HTTP server, routing, request/response handling without actual inference

**What to test:**
```bash
# 1. Build and start in stub mode
.build/debug/ZLXServer --model qwen2.5-coder-1.5b --port 8080 &

# 2. Health check
curl http://localhost:8080/health
# Expected: {"status": "ok"}

# 3. List models
curl http://localhost:8080/v1/models
# Expected: JSON with all registered models

# 4. Chat completion (stub response)
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-coder-1.5b",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 10
  }'
# Expected: Valid OpenAI JSON with stub text

# 5. Streaming test
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-coder-1.5b",
    "messages": [{"role": "user", "content": "Hi"}],
    "stream": true
  }'
# Expected: SSE stream with chunks
```

**Success Criteria:**
- [ ] Server starts without crashing
- [ ] All endpoints respond with valid JSON
- [ ] Response format matches OpenAI spec
- [ ] No memory leaks in stub mode

---

### Level 2: Unit Tests (Swift Testing) ⏳ Add Next

**Purpose:** Test individual components with mocked dependencies

**Create:** `Tests/ZLXServerTests/`

```swift
// ModelRegistryTests.swift
import XCTest
@testable import ZLXServer

class ModelRegistryTests: XCTestCase {
    func testRegisterAndRetrieve() async {
        let registry = ModelRegistry.shared
        
        let model = ModelInfo(
            id: "test-model",
            name: "Test Model",
            architecture: .qwen2_5,
            description: "Test",
            sizeGB: 1.0,
            path: "/tmp/test"
        )
        
        await registry.register(model)
        let retrieved = await registry.getModel(id: "test-model")
        
        XCTAssertNotNil(retrieved)
        XCTAssertEqual(retrieved?.id, "test-model")
    }
    
    func testAliasResolution() async {
        let registry = ModelRegistry.shared
        let resolved = await registry.resolveAlias("qwen")
        XCTAssertEqual(resolved, "qwen2.5-coder-1.5b")
    }
}

// ChatTemplateTests.swift
class ChatTemplateTests: XCTestCase {
    func testQwenTemplate() {
        let messages = [
            ChatMessage(role: "system", content: "You are helpful"),
            ChatMessage(role: "user", content: "Hello")
        ]
        
        let result = QwenChatTemplate.apply(messages: messages)
        
        XCTAssertTrue(result.contains("<|im_start|>system"))
        XCTAssertTrue(result.contains("You are helpful"))
        XCTAssertTrue(result.contains("<|im_start|>user"))
        XCTAssertTrue(result.contains("Hello"))
    }
    
    func testGemma4Template() {
        let messages = [
            ChatMessage(role: "user", content: "Hi")
        ]
        
        let result = Gemma4ChatTemplate.apply(messages: messages)
        
        XCTAssertTrue(result.contains("<|turn|>user"))
        XCTAssertTrue(result.contains("Hi"))
    }
}

// PLEBlockTests.swift (with mocked MLX)
class PLEBlockTests: XCTestCase {
    func testPLEForward() {
        // Mock MLX arrays
        let hiddenSize = 2560
        let pleDim = 256
        
        // Create PLE block
        let pleBlock = Gemma4PLEBlock(hiddenSize: hiddenSize, pleDim: pleDim)
        
        // Mock input (would use real MLX in integration)
        // let x = MLXArray.zeros([1, 10, hiddenSize])
        // let pleSlice = MLXArray.zeros([1, 10, pleDim])
        // let result = pleBlock(x, pleSlice: pleSlice)
        
        // Assert shape and values
        // XCTAssertEqual(result.shape, [1, 10, hiddenSize])
    }
}
```

**Run:**
```bash
swift test
```

---

### Level 3: Integration with Working Zig (Validation) ✅ Ready Now

**Purpose:** Compare Swift stub responses with real Zig outputs

**Setup:**
```bash
# Terminal 1: Start Zig server (known working)
git checkout main
zig build
./zig-out/bin/zlx --model Qwen2.5-Coder-1.5B-Instruct-4bit --port 8080 &

# Terminal 2: Start Swift server
git checkout swift-mlx-rewrite
.build/debug/ZLXServer --model qwen2.5-coder-1.5b --port 8081 &
```

**Comparison Tests:**
```bash
# 1. Compare /v1/models responses
curl -s http://localhost:8080/v1/models > /tmp/zig_models.json
curl -s http://localhost:8081/v1/models > /tmp/swift_models.json

# Should have same structure (Swift has more models registered)
diff <(cat /tmp/zig_models.json | jq '.data[].id') \
     <(cat /tmp/swift_models.json | jq '.data[].id' | grep -E "qwen|Qwen")

# 2. Compare response format
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen2.5-Coder-1.5B-Instruct-4bit", "messages": [{"role": "user", "content": "Say hi"}], "max_tokens": 5}' \
  | jq '{id, object, model, choices: [.choices[0] | {index, message: {role}, finish_reason}]}' > /tmp/zig_response.json

curl -s -X POST http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5-coder-1.5b", "messages": [{"role": "user", "content": "Say hi"}], "max_tokens": 5}' \
  | jq '{id, object, model, choices: [.choices[0] | {index, message: {role}, finish_reason}]}' > /tmp/swift_response.json

# Compare structure (not content, since Swift is stub)
diff <(cat /tmp/zig_response.json | jq 'keys') \
     <(cat /tmp/swift_response.json | jq 'keys')

# 3. Compare streaming format
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen2.5-Coder-1.5B-Instruct-4bit", "messages": [{"role": "user", "content": "Hi"}], "stream": true, "max_tokens": 3}' > /tmp/zig_stream.txt

curl -s -X POST http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5-coder-1.5b", "messages": [{"role": "user", "content": "Hi"}], "stream": true, "max_tokens": 3}' > /tmp/swift_stream.txt

# Check SSE format
echo "Zig stream lines: $(wc -l < /tmp/zig_stream.txt)"
echo "Swift stream lines: $(wc -l < /tmp/swift_stream.txt)"
grep -c "^data:" /tmp/zig_stream.txt
grep -c "^data:" /tmp/swift_stream.txt
```

**Validation:**
- [ ] Response structure matches
- [ ] JSON fields identical
- [ ] HTTP headers correct
- [ ] Streaming format compatible

---

### Level 4: Real MLX (After Metallib Setup) ⏳ Final Validation

**Purpose:** Full end-to-end with actual model inference

**Prerequisites:**
```bash
# Follow CLI_ONLY_DEVELOPMENT.md to setup MLX resources
./setup-mlx-resources.sh
# Or download pre-built metallib
```

**Tests:**
```bash
# 1. Build with resources
swift build -c release

# 2. Test with real Qwen model
.build/release/ZLXServer --model qwen2.5-coder-1.5b --port 8080 &

# 3. Generation test
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-coder-1.5b",
    "messages": [{"role": "user", "content": "Write a Python hello world"}],
    "max_tokens": 50
  }' | jq '.choices[0].message.content'

# 4. Token speed benchmark
# (Would need timing script)

# 5. Memory test
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-coder-1.5b",
    "messages": [{"role": "user", "content": "Long prompt..."}],
    "max_tokens": 100
  }'
# Monitor memory with: vm_stat or Activity Monitor

# 6. Gemma 4 specific test
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-e4b",
    "messages": [{"role": "user", "content": "Explain PLE architecture"}],
    "max_tokens": 100
  }'

# 7. Concurrent requests test
for i in {1..5}; do
  curl -s -X POST http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"qwen2.5-coder-1.5b\", \"messages\": [{\"role\": \"user\", \"content\": \"Request $i\"}], \"max_tokens\": 10}" &
done
wait
```

**Success Criteria:**
- [ ] Generates coherent text
- [ ] Token speed >30 tok/s (comparable to Zig)
- [ ] Memory usage <12GB for 4-bit models
- [ ] No crashes on concurrent requests
- [ ] Gemma 4 produces correct output (validates PLE)

---

## 🛠️ Automated Test Script

Create `test.sh`:

```bash
#!/bin/bash
set -e

echo "🧪 ZLX Swift Test Suite"
echo "======================"

# Test 1: Build
echo "[1/5] Testing build..."
swift build -c release
echo "✅ Build successful"

# Test 2: CLI help
echo "[2/5] Testing CLI..."
.build/release/ZLXServer --help > /dev/null
echo "✅ CLI works"

# Test 3: Server start (stub mode)
echo "[3/5] Testing server startup..."
.build/release/ZLXServer --model qwen2.5-coder-1.5b --port 9999 &
SERVER_PID=$!
sleep 2

# Test 4: HTTP endpoints
echo "[4/5] Testing HTTP endpoints..."
curl -sf http://localhost:9999/v1/models > /dev/null
echo "✅ /v1/models responds"

# Test 5: Chat completion (structure only)
echo "[5/5] Testing chat completion..."
RESPONSE=$(curl -sf -X POST http://localhost:9999/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5-coder-1.5b", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 5}')

if echo "$RESPONSE" | jq -e '.choices[0].message.content' > /dev/null; then
  echo "✅ Chat completion structure valid"
else
  echo "❌ Chat completion response invalid"
  exit 1
fi

# Cleanup
kill $SERVER_PID 2>/dev/null || true

echo ""
echo "✅ All tests passed!"
echo ""
echo "Next steps for full validation:"
echo "1. Setup MLX resources: ./setup-mlx-resources.sh"
echo "2. Rebuild with resources"
echo "3. Test with real model generation"
```

---

## 📈 Performance Benchmarks

Compare Swift vs Zig:

```bash
# Create benchmark script test_performance.sh
#!/bin/bash

PROMPT="Write a Python function to calculate fibonacci"
MAX_TOKENS=100

echo "Benchmarking token generation speed..."
echo "Prompt: $PROMPT"
echo "Max tokens: $MAX_TOKENS"
echo ""

# Zig version
echo "Zig version:"
start=$(date +%s.%N)
 curl -s -X POST http://localhost:8080/v1/chat/completions \
   -H "Content-Type: application/json" \
   -d "{\"model\": \"Qwen2.5-Coder-1.5B-Instruct-4bit\", \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}], \"max_tokens\": $MAX_TOKENS}"
end=$(date +%s.%N)
zig_time=$(echo "$end - $start" | bc)
echo "Time: ${zig_time}s"

# Swift version (after MLX setup)
echo "Swift version:"
start=$(date +%s.%N)
curl -s -X POST http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"qwen2.5-coder-1.5b\", \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}], \"max_tokens\": $MAX_TOKENS}"
end=$(date +%s.%N)
swift_time=$(echo "$end - $start" | bc)
echo "Time: ${swift_time}s"

echo ""
echo "Swift/Zig ratio: $(echo "scale=2; $swift_time / $zig_time" | bc)x"
```

---

## 🎯 Testing Roadmap

### Phase 1: Now (Pre-Merge) ✅
- [x] Build test
- [x] CLI test  
- [x] Server startup test
- [x] HTTP endpoint structure test
- [ ] Add unit test target
- [ ] Create automated test.sh

### Phase 2: Post-Merge (Week 1)
- [ ] Setup MLX resources
- [ ] Real generation test
- [ ] Performance benchmark
- [ ] Concurrent load test
- [ ] Memory leak test

### Phase 3: Optimization (Week 2)
- [ ] Profile with Instruments
- [ ] Compare with Zig baseline
- [ ] Optimize hot paths
- [ ] Add TurboQuant test

---

## 📝 Test Documentation

Each test should output:
```
Test: <name>
Status: ✅ PASS / ❌ FAIL
Duration: <time>ms
Details: <relevant info>
```

---

## 🚀 Quick Test Now

```bash
# Immediate validation (no MLX required)
cd /Users/gleicon/code/zig/zlx

# 1. Build
swift build -c release

# 2. Start server
.build/release/ZLXServer --model qwen2.5-coder-1.5b --port 8080 &

# 3. Test endpoints
curl http://localhost:8080/v1/models | jq '.data[].id'
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5-coder-1.5b", "messages": [{"role": "user", "content": "Hello"}]}'

# 4. Kill server
pkill ZLXServer
```

---

**Summary:**
- **Immediate:** Stub mode tests (working now)
- **Short term:** Unit tests + comparison with Zig
- **Full validation:** Real MLX after metallib setup
- **Automated:** test.sh script for CI

**Next step:** Create test.sh and run immediate validation?
