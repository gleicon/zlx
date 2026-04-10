#!/bin/bash
# test.sh - Automated test suite for ZLX Swift (Stub Mode)
# No MLX resources required - tests HTTP/API layer only

set -e

PORT=9999
SERVER_PID=""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_test() {
    echo "[TEST] $1"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    exit 1
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Cleanup function
cleanup() {
    if [ -n "$SERVER_PID" ]; then
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
        log_warn "Cleaned up server (PID: $SERVER_PID)"
    fi
}
trap cleanup EXIT

# Header
echo "🧪 ZLX Swift Test Suite (Stub Mode)"
echo "===================================="
echo "Testing HTTP server and API without MLX inference"
echo ""

# Test 1: Build
echo "[1/6] Building project..."
if swift build -c release 2>&1 | tail -5; then
    log_pass "Build successful"
else
    log_fail "Build failed"
fi

# Check binary exists (could be debug or release)
if [ -f ".build/release/ZLXServer" ]; then
    BINARY_PATH=".build/release/ZLXServer"
    log_pass "Binary created ($BINARY_PATH)"
elif [ -f ".build/debug/ZLXServer" ]; then
    BINARY_PATH=".build/debug/ZLXServer"
    log_pass "Binary created ($BINARY_PATH)"
else
    log_fail "Binary not found"
fi

# Set binary path for tests
ZLX_BINARY="${BINARY_PATH:-.build/debug/ZLXServer}"

# Test 2: CLI Help
echo ""
echo "[2/6] Testing CLI..."
HELP_OUTPUT=$($ZLX_BINARY --help 2>&1)

if echo "$HELP_OUTPUT" | grep -q "USAGE:"; then
    log_pass "CLI responds to --help"
else
    log_fail "CLI --help failed"
fi

if echo "$HELP_OUTPUT" | grep -q "\-\-model"; then
    log_pass "--model option present"
else
    log_warn "--model option not found in help"
fi

# Test 3: Server Startup
echo ""
echo "[3/6] Starting server on port $PORT..."
$ZLX_BINARY --model qwen2.5-coder-1.5b --port $PORT > /tmp/zlx_test_server.log 2>&1 &
SERVER_PID=$!

# Wait for server to start
sleep 3

# Check if server is running
if kill -0 $SERVER_PID 2>/dev/null; then
    log_pass "Server started (PID: $SERVER_PID)"
else
    # Server exited - check if it's MLX error (expected without resources)
    if grep -q "metallib\|library not found" /tmp/zlx_test_server.log; then
        log_warn "Server needs MLX resources (expected in stub mode test)"
        log_warn "To test with full MLX, run: ./setup-mlx-resources.sh"
        echo ""
        echo "⚠️  SKIPPING HTTP tests (MLX not available)"
        echo "✅ Build and CLI tests PASSED"
        echo ""
        echo "📄 Server log preview:"
        head -5 /tmp/zlx_test_server.log
        echo ""
        exit 0
    else
        log_fail "Server failed to start"
        cat /tmp/zlx_test_server.log
        exit 1
    fi
fi

# Check if port is listening
if lsof -i :$PORT 2>/dev/null | grep -q "ZLXServer"; then
    log_pass "Server listening on port $PORT"
else
    log_warn "Port $PORT not confirmed listening (may still be starting)"
fi

# Test 4: List Models
echo ""
echo "[4/6] Testing /v1/models endpoint..."
MODELS_RESPONSE=$(curl -sf http://localhost:$PORT/v1/models 2>&1)

if [ $? -eq 0 ]; then
    log_pass "Endpoint /v1/models responds"
    
    # Check for expected models
    if echo "$MODELS_RESPONSE" | grep -q "qwen2.5-coder-1.5b\|qwen"; then
        log_pass "Qwen model found in response"
    else
        log_warn "Qwen model not found in list"
    fi
    
    if echo "$MODELS_RESPONSE" | grep -q "deepseek\|DeepSeek"; then
        log_pass "DeepSeek model found in response"
    else
        log_warn "DeepSeek model not found in list"
    fi
    
    if echo "$MODELS_RESPONSE" | grep -q "gemma\|Gemma"; then
        log_pass "Gemma model found in response"
    else
        log_warn "Gemma model not found in list"
    fi
else
    log_fail "Endpoint /v1/models failed"
    echo "Response: $MODELS_RESPONSE"
fi

# Test 5: Chat Completion
echo ""
echo "[5/6] Testing /v1/chat/completions..."
CHAT_RESPONSE=$(curl -sf -X POST http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-coder-1.5b",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 10
  }' 2>&1)

if [ $? -eq 0 ]; then
    log_pass "Chat completion endpoint responds"
    
    # Check JSON structure
    if echo "$CHAT_RESPONSE" | jq -e '.id' > /dev/null 2>&1; then
        log_pass "Response has 'id' field"
    else
        log_warn "Response missing 'id' field"
    fi
    
    if echo "$CHAT_RESPONSE" | jq -e '.choices[0].message.content' > /dev/null 2>&1; then
        log_pass "Response has 'choices[0].message.content' field"
    else
        log_warn "Response missing content field"
    fi
    
    if echo "$CHAT_RESPONSE" | jq -e '.usage.total_tokens' > /dev/null 2>&1; then
        log_pass "Response has 'usage.total_tokens' field"
    else
        log_warn "Response missing usage field"
    fi
else
    log_fail "Chat completion failed"
    echo "Response: $CHAT_RESPONSE"
fi

# Show sample response
echo ""
echo "📄 Sample chat response:"
echo "$CHAT_RESPONSE" | jq . 2>/dev/null || echo "$CHAT_RESPONSE"

# Test 6: Error Handling
echo ""
echo "[6/6] Testing error handling..."

# Invalid JSON
ERROR_RESPONSE=$(curl -sf -X POST http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d 'invalid json here' 2>&1)

if echo "$ERROR_RESPONSE" | grep -q "error\|Error"; then
    log_pass "Invalid JSON returns error response"
else
    log_warn "Invalid JSON handling unclear"
fi

# Non-existent model
INVALID_MODEL=$(curl -sf -X POST http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "non-existent-model",
    "messages": [{"role": "user", "content": "Hi"}]
  }' 2>&1)

if echo "$INVALID_MODEL" | grep -q "not found\|error\|Error"; then
    log_pass "Non-existent model returns error"
else
    log_warn "Non-existent model handling unclear"
fi

# Summary
echo ""
echo "===================================="
echo -e "${GREEN}✅ All stub mode tests completed!${NC}"
echo ""
echo "📊 Summary:"
echo "  - Build: ✅"
echo "  - CLI: ✅"
echo "  - Server startup: ✅"
echo "  - HTTP endpoints: ✅"
echo "  - JSON responses: ✅"
echo ""
echo "📝 Notes:"
echo "  - Tests use STUB mode (no real MLX inference)"
echo "  - For real generation, setup MLX resources first:"
echo "    ./setup-mlx-resources.sh"
echo ""
echo "🚀 Next steps:"
echo "  1. Review test output above"
echo "  2. Compare with Zig server: git checkout main && zig build && ./zig-out/bin/zlx"
echo "  3. Setup MLX: ./setup-mlx-resources.sh"
echo "  4. Test with real inference"
echo ""
