#!/bin/bash
# test_gemma4_setup.sh - Verify Gemma 4 MLX implementation

echo "=========================================="
echo "Gemma 4 Native MLX Implementation Test"
echo "=========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track results
PASSED=0
FAILED=0

test_step() {
    echo -n "Testing: $1... "
}

pass() {
    echo -e "${GREEN}✓ PASS${NC}"
    ((PASSED++))
}

fail() {
    echo -e "${RED}✗ FAIL${NC}"
    echo "  Error: $1"
    ((FAILED++))
}

warn() {
    echo -e "${YELLOW}⚠ WARN${NC}"
    echo "  $1"
}

echo "1. Model Detection"
echo "------------------"

test_step "Gemma 4 alias in registry"
if grep -q "gemma4-e4b" src/download/mod.zig; then
    pass
else
    fail "Alias not found"
fi

test_step "Gemma4 architecture enum"
if grep -q "gemma4" src/models/registry.zig && grep -q "gemma4" src/inference/loader.zig; then
    pass
else
    fail "Architecture not registered"
fi

test_step "Detection order (before GPT-OSS)"
if grep -B5 "gpt_oss" src/inference/loader.zig | grep -q "gemma4"; then
    pass
else
    fail "Detection order may be wrong"
fi

echo ""
echo "2. MLX Implementation"
echo "---------------------"

test_step "gemma4.zig exists"
if [ -f "src/mlx.zig/src/gemma4.zig" ]; then
    pass
else
    fail "File not found"
fi

test_step "Gemma4Config defined"
if grep -q "Gemma4Config\|ModelConfig" src/mlx.zig/src/gemma4.zig; then
    pass
else
    fail "Config not found"
fi

test_step "SlidingWindowAttention defined"
if grep -q "SlidingWindowAttention" src/mlx.zig/src/gemma4.zig; then
    pass
else
    fail "Attention not found"
fi

test_step "TransformerBlock defined"
if grep -q "TransformerBlock" src/mlx.zig/src/gemma4.zig; then
    pass
else
    fail "TransformerBlock not found"
fi

test_step "Transformer type alias"
if grep -q "Transformer.*=" src/mlx.zig/src/gemma4.zig; then
    pass
else
    fail "Transformer alias not found"
fi

echo ""
echo "3. GenerationState Generic"
echo "--------------------------"

test_step "GenerationState is generic function"
if grep -q "pub fn GenerationState.*comptime.*type.*type" src/inference/generator.zig; then
    pass
else
    warn "GenerationState may not be fully generic"
fi

test_step "Takes TransformerType parameter"
if grep -q "TransformerType.*type" src/inference/generator.zig; then
    pass
else
    fail "No TransformerType parameter"
fi

echo ""
echo "4. Model Availability"
echo "---------------------"

test_step "MLX model exists (safetensors)"
if ls models/gemma4-e4b/*.safetensors 2>/dev/null | grep -q .; then
    pass
    echo "  Found: $(ls models/gemma4-e4b/*.safetensors 2>/dev/null | wc -l) files"
else
    fail "No safetensors found"
fi

test_step "Config JSON exists"
if [ -f "models/gemma4-e4b/config.json" ]; then
    pass
else
    fail "config.json not found"
fi

test_step "GGUF model (optional)"
if ls models/gemma4-e4b/*.gguf 2>/dev/null | grep -q .; then
    GGUF_SIZE=$(ls -lh models/gemma4-e4b/*.gguf 2>/dev/null | awk '{print $5}')
    if [ "$GGUF_SIZE" != "0B" ]; then
        pass
        echo "  Size: $GGUF_SIZE"
    else
        warn "GGUF file is empty (download incomplete)"
    fi
else
    warn "No GGUF found (need to convert or download)"
fi

echo ""
echo "5. Build Status"
echo "---------------"

test_step "Project builds"
if zig build 2>&1 | grep -q "error"; then
    fail "Build errors present"
    zig build 2>&1 | grep "error:" | head -5
else
    pass
fi

echo ""
echo "6. Documentation"
echo "--------------"

test_step "Setup guide exists"
if [ -f "docs/GEMMA4_SETUP.md" ]; then
    pass
else
    fail "Setup guide not found"
fi

test_step "Status document exists"
if [ -f "docs/GEMMA4_STATUS.md" ]; then
    pass
else
    fail "Status document not found"
fi

echo ""
echo "=========================================="
echo "Results"
echo "=========================================="
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Fix GenerationState.init signature to use TransformerType"
    echo "  2. Update all GenerationState usages with explicit type"
    echo "  3. Test with: ./zig-out/bin/zlx --model gemma4-e4b"
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    echo ""
    echo "Please review the failures above."
    exit 1
fi
