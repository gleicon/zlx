#!/bin/bash
# run.sh - Run ZLXServer with correct MLX metallib paths

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Find the binary
if [ -f "$SCRIPT_DIR/.build/debug/ZLXServer" ]; then
    BINARY="$SCRIPT_DIR/.build/debug/ZLXServer"
elif [ -f "$SCRIPT_DIR/.build/release/ZLXServer" ]; then
    BINARY="$SCRIPT_DIR/.build/release/ZLXServer"
else
    echo "❌ ZLXServer binary not found. Build first with: swift build"
    exit 1
fi

# Check for metallib
METALLIB_DIR="$SCRIPT_DIR/Sources/ZLXServer/Resources"
if [ ! -f "$METALLIB_DIR/default.metallib" ]; then
    echo "⚠️  default.metallib not found at $METALLIB_DIR"
    echo "   Run setup first: ./setup-mlx-resources.sh"
    exit 1
fi

# Option 1: Create symlink in expected location (relative to binary)
BINARY_DIR="$(dirname "$BINARY")"
if [ ! -d "$BINARY_DIR/Resources" ]; then
    echo "📁 Creating Resources symlink..."
    ln -sf "$METALLIB_DIR" "$BINARY_DIR/Resources"
fi

# Option 2: Set environment variable (backup method)
export MLX_METAL_PATH="$METALLIB_DIR/default.metallib"

# Option 3: Copy metallib to binary directory (most reliable)
if [ ! -f "$BINARY_DIR/default.metallib" ]; then
    echo "📋 Copying metallib to binary directory..."
    cp "$METALLIB_DIR"/*.metallib "$BINARY_DIR/" 2>/dev/null || true
fi

echo "🚀 Starting ZLXServer with MLX..."
echo "   Binary: $BINARY"
echo "   Metallib: $METALLIB_DIR"
echo ""

# Run with all arguments passed through
exec "$BINARY" "$@"
