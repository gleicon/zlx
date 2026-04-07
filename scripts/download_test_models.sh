#!/usr/bin/env bash
# download_test_models.sh — Download models needed for Phase 17 E2E tests
#
# Usage:
#   scripts/download_test_models.sh          # download all test models
#   scripts/download_test_models.sh --qwen   # only Qwen (1.0 GB, fastest)
#   scripts/download_test_models.sh --gptoss # only GPT-OSS (11.2 GB)
#
# Models:
#   Qwen2.5-Coder-1.5B-Instruct-4bit  ~1.0 GB  mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit
#   GPT-OSS-20B (MXFP4-Q4)           ~11.2 GB  mlx-community/gpt-oss-20b-MXFP4-Q4
#   DeepSeek (already present, no download)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"

DOWNLOAD_QWEN=true
DOWNLOAD_GPTOSS=true
for arg in "$@"; do
    case "$arg" in
        --qwen)   DOWNLOAD_GPTOSS=false ;;
        --gptoss) DOWNLOAD_QWEN=false ;;
    esac
done

mkdir -p "$MODELS_DIR"
cd "$PROJECT_DIR"

# ── helpers ───────────────────────────────────────────────────────────────────

hf_download() {
    local repo="$1"
    local dest="$2"
    echo "Downloading $repo → $dest"
    if command -v huggingface-cli &>/dev/null; then
        huggingface-cli download "$repo" --local-dir "$dest" --local-dir-use-symlinks False
    elif command -v python3 &>/dev/null && python3 -c "import huggingface_hub" 2>/dev/null; then
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$repo', local_dir='$dest', local_dir_use_symlinks=False)
"
    else
        echo ""
        echo "ERROR: huggingface-cli not found."
        echo "Install with:  pip install huggingface-hub"
        echo "Then re-run this script."
        exit 1
    fi
}

has_weights() {
    local dir="$1"
    ls "$dir"/model*.safetensors 2>/dev/null | head -1 | grep -q . || \
    ls "$dir"/model-*.safetensors 2>/dev/null | head -1 | grep -q .
}

# ── 1. Qwen2.5-Coder-1.5B — fix path and download if needed ──────────────────

if $DOWNLOAD_QWEN; then
    QWEN_INSTRUCT="$MODELS_DIR/Qwen2.5-Coder-1.5B-Instruct-4bit"
    QWEN_EXPECTED="$MODELS_DIR/Qwen2.5-Coder-1.5B-4bit"

    if has_weights "$QWEN_INSTRUCT"; then
        echo "✓ Qwen model already present: $QWEN_INSTRUCT"
    else
        echo "Downloading Qwen2.5-Coder-1.5B-Instruct-4bit (~1.0 GB)..."
        hf_download "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit" "$QWEN_INSTRUCT"
        echo "✓ Qwen download complete"
    fi

    # Create the path-name symlink that generator.zig and zig build test expect
    if [ -L "$QWEN_EXPECTED" ] || [ -d "$QWEN_EXPECTED" ]; then
        echo "✓ Qwen expected path exists: $QWEN_EXPECTED"
    else
        ln -sfn "Qwen2.5-Coder-1.5B-Instruct-4bit" "$QWEN_EXPECTED"
        echo "✓ Created symlink: Qwen2.5-Coder-1.5B-4bit → Qwen2.5-Coder-1.5B-Instruct-4bit"
    fi
fi

# ── 2. DeepSeek — verify already present (no download needed) ─────────────────

DEEPSEEK_PATH="$MODELS_DIR/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx"
if has_weights "$DEEPSEEK_PATH"; then
    echo "✓ DeepSeek MLX model present: $DEEPSEEK_PATH"
else
    echo "⚠ DeepSeek MLX model not found at $DEEPSEEK_PATH"
    echo "  To download: git clone https://huggingface.co/mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx $DEEPSEEK_PATH"
    echo "  (8.84 GB — deepseek_test will skip without this model)"
fi

# ── 3. GPT-OSS 20B — download weights ────────────────────────────────────────

if $DOWNLOAD_GPTOSS; then
    GPTOSS_PATH="$MODELS_DIR/GPT-OSS-20B-4bit"
    if has_weights "$GPTOSS_PATH"; then
        echo "✓ GPT-OSS model weights present: $GPTOSS_PATH"
    else
        echo "Downloading GPT-OSS-20B MXFP4-Q4 (~11.2 GB)..."
        echo "  This will take a while on first run."
        mkdir -p "$GPTOSS_PATH"
        hf_download "mlx-community/gpt-oss-20b-MXFP4-Q4" "$GPTOSS_PATH"
        echo "✓ GPT-OSS download complete"
    fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo "Model status:"
has_weights "$MODELS_DIR/Qwen2.5-Coder-1.5B-Instruct-4bit" 2>/dev/null && echo "  ✓ Qwen2.5-Coder-1.5B"  || echo "  ✗ Qwen2.5-Coder-1.5B  (missing)"
has_weights "$DEEPSEEK_PATH" 2>/dev/null                                 && echo "  ✓ DeepSeek MLX"        || echo "  ✗ DeepSeek MLX         (missing)"
has_weights "$MODELS_DIR/GPT-OSS-20B-4bit" 2>/dev/null                  && echo "  ✓ GPT-OSS-20B"         || echo "  ✗ GPT-OSS-20B          (missing — run without --qwen)"
echo ""
echo "Run E2E tests: zig build test-e2e"
