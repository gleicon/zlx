#!/bin/bash
# setup-mlx-resources.sh - Setup MLX Metal resources for CLI builds

set -e

echo "🔧 Setting up MLX Metal resources for CLI-only development..."

# Create directories
mkdir -p Resources/metallib
mkdir -p .mlx-resources

echo "📦 Checking mlx-swift structure..."

# Find the mlx-c source
MLX_C_PATH=".build/checkouts/mlx-swift/Source/Cmlx"

if [ ! -d "$MLX_C_PATH" ]; then
    echo "⚠️  mlx-swift not found. Running package resolve..."
    swift package resolve
fi

# Check for metallib generation
# MLX generates metallib during cmake build
MLX_BUILD_PATH=".build/checkouts/mlx-swift/.build"

if [ -d "$MLX_BUILD_PATH" ]; then
    echo "🔍 Looking for generated metallib..."
    find "$MLX_BUILD_PATH" -name "*.metallib" 2>/dev/null | head -5
fi

echo ""
echo "📝 CLI-Only Development Guide:"
echo ""
echo "Option 1: Use Pre-built Resources (Recommended for CLI)"
echo "--------------------------------------------------------"
echo "1. Download mlx-swift example app resources:"
echo "   git clone https://github.com/ml-explore/mlx-swift-examples"
echo "   cd mlx-swift-examples && swift build"
echo ""
echo "2. Copy generated metallib to Resources/:"
echo "   cp -R path/to/metallib ./Resources/"
echo ""
echo "Option 2: Environment Variable (Quick Test)"
echo "--------------------------------------------"
echo "Set the metallib path manually:"
echo "   export MLX_METAL_PATH=/path/to/metallib"
echo "   .build/release/ZLXServer --model qwen2.5-coder-1.5b"
echo ""
echo "Option 3: Runtime Download (Planned)"
echo "-------------------------------------"
echo "Implement metallib download on first run:"
echo "   - Check ~/.cache/zlx/metallib/"
echo "   - Download from mlx-swift releases if missing"
echo "   - Set environment before MLX initialization"
echo ""
echo "Current Status:"
echo "--------------"
echo "✅ Swift code compiles and builds"
echo "✅ HTTP server starts and responds"
echo "✅ Model registry loads all models"
echo "⚠️  Metal library needs separate setup for CLI"
echo ""
echo "Next Steps:"
echo "1. Run swift package resolve"
echo "2. Build mlx-swift-examples to get metallib"
echo "3. Copy metallib to Resources/"
echo "4. Rebuild ZLXServer"
