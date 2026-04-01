#!/bin/bash

# download-and-run.sh - Download a test model and run zlx

set -e

MODEL_NAME="${1:-Qwen2.5-Coder-1.5B-4bit}"
MODELS_DIR="${2:-./models}"
PORT="${3:-8080}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}zlx model downloader and test runner${NC}"
echo "===================================="
echo "Model: $MODEL_NAME"
echo "Directory: $MODELS_DIR"
echo "Port: $PORT"
echo ""

# Check if zlx is built
if [ ! -f "./zig-out/bin/zlx" ]; then
    echo -e "${YELLOW}Building zlx...${NC}"
    zig build
    echo -e "${GREEN}Build complete${NC}"
    echo ""
fi

# Create models directory
mkdir -p "$MODELS_DIR"
cd "$MODELS_DIR"

# Check if model already exists
if [ -d "$MODEL_NAME" ] && [ -f "$MODEL_NAME/config.json" ] && [ -f "$MODEL_NAME/tokenizer.json" ]; then
    echo -e "${GREEN}Model $MODEL_NAME already exists, skipping download${NC}"
else
    echo -e "${YELLOW}Downloading model files...${NC}"
    
    # Download config and tokenizer first
    BASE_URL="https://huggingface.co/mlx-community/$MODEL_NAME/resolve/main"
    
    mkdir -p "$MODEL_NAME"
    
    echo "Downloading config.json..."
    curl -L --progress-bar "$BASE_URL/config.json" -o "$MODEL_NAME/config.json" || {
        echo -e "${RED}Failed to download config.json${NC}"
        exit 1
    }
    
    echo "Downloading tokenizer.json..."
    curl -L --progress-bar "$BASE_URL/tokenizer.json" -o "$MODEL_NAME/tokenizer.json" || {
        echo -e "${RED}Failed to download tokenizer.json${NC}"
        exit 1
    }
    
    # Check for model.safetensors.index.json to see if model is sharded
    echo "Checking for sharded weights..."
    if curl -sI "$BASE_URL/model.safetensors.index.json" | grep -q "200 OK"; then
        echo "Found sharded weights, downloading index..."
        curl -L --progress-bar "$BASE_URL/model.safetensors.index.json" -o "$MODEL_NAME/model.safetensors.index.json"
        
        # Parse index.json and download all shard files
        echo "Downloading weight shards (this may take a while)..."
        # Extract unique filenames from weight_map using Python or jq if available
        if command -v python3 &> /dev/null; then
            python3 << 'EOF' - "$MODEL_NAME" "$BASE_URL"
import json
import sys
import subprocess
import os

model_name = sys.argv[1]
base_url = sys.argv[2]

with open(f"{model_name}/model.safetensors.index.json", "r") as f:
    index = json.load(f)

# Get unique weight files
files = set(index["weight_map"].values())
print(f"Found {len(files)} weight shard(s) to download")

for filename in sorted(files):
    url = f"{base_url}/{filename}"
    output = f"{model_name}/{filename}"
    if os.path.exists(output):
        print(f"  {filename} already exists, skipping")
        continue
    print(f"  Downloading {filename}...")
    subprocess.run(["curl", "-L", "--progress-bar", "-o", output, url], check=True)

print("All weight shards downloaded")
EOF
        else
            echo -e "${YELLOW}Python3 not available, downloading single model.safetensors${NC}"
            curl -L --progress-bar "$BASE_URL/model.safetensors" -o "$MODEL_NAME/model.safetensors" || {
                echo -e "${RED}Failed to download model.safetensors${NC}"
                exit 1
            }
        fi
    else
        echo "Downloading model.safetensors (this may take a while)..."
        curl -L --progress-bar "$BASE_URL/model.safetensors" -o "$MODEL_NAME/model.safetensors" || {
            echo -e "${RED}Failed to download model.safetensors${NC}"
            exit 1
        }
    fi
    
    echo -e "${GREEN}Model download complete!${NC}"
fi

cd - > /dev/null

echo ""
echo -e "${GREEN}Starting zlx server...${NC}"
echo "===================================="
echo "Server will be available at: http://127.0.0.1:$PORT"
echo ""
echo "Test commands:"
echo "  # List models"
echo "  curl http://127.0.0.1:$PORT/v1/models"
echo ""
echo "  # Non-streaming completion"
echo "  curl -X POST http://127.0.0.1:$PORT/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"$MODEL_NAME\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hello\"}]}'"
echo ""
echo "  # Streaming completion"
echo "  curl -N -X POST http://127.0.0.1:$PORT/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"$MODEL_NAME\", \"messages\": [{\"role\": \"user\", \"content\": \"Count to 3\"}], \"stream\": true}'"
echo ""
echo "Press Ctrl+C to stop the server"
echo "===================================="
echo ""

# Run zlx
./zig-out/bin/zlx --model "$MODEL_NAME" --port "$PORT"
