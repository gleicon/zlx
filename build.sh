#!/bin/bash
# build.sh - Build ZLXServer with Metal resources for CLI-only development

set -e

echo "🔨 Building ZLXServer (CLI-only, no Xcode required)..."

# Find mlx-swift checkout
MLX_SWIFT_PATH=".build/checkouts/mlx-swift"
METAL_LIB_PATH="$MLX_SWIFT_PATH/Source/Cmlx/mlx-c/mlx/c/metallib"

echo "📦 Checking for Metal libraries..."

# Check if metallib exists
if [ ! -d "$METAL_LIB_PATH" ]; then
    echo "⚠️  Metal library not found at $METAL_LIB_PATH"
    echo "   This is expected on first build. Will resolve packages first."
fi

# Build with resources
echo "🚀 Running swift build..."
swift build -c release "$@"

# Create Resources directory in build output
BUILD_RESOURCES=".build/release/ZLXServer.bundle/Resources"
mkdir -p "$BUILD_RESOURCES"

# Copy Metal libraries if they exist
if [ -d "$METAL_LIB_PATH" ]; then
    echo "📋 Copying Metal libraries..."
    cp -R "$METAL_LIB_PATH" "$BUILD_RESOURCES/"
    echo "✅ Metal libraries copied"
else
    echo "⚠️  Warning: Metal libraries not found"
    echo "   MLX operations may fail. Run 'swift package resolve' first."
fi

echo ""
echo "✅ Build complete!"
echo "   Binary: .build/release/ZLXServer"
echo "   Resources: $BUILD_RESOURCES"
echo ""
echo "🚀 To run:"
echo "   ./build.sh          # Build with resources"
echo "   .build/release/ZLXServer --model qwen2.5-coder-1.5b --port 8080"
