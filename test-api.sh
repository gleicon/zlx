#!/bin/bash

# test-api.sh - Quick test script for zlx API endpoints

set -e

MODEL_NAME="${1:-Qwen2.5-Coder-1.5B-4bit}"
PORT="${2:-8080}"
BASE_URL="http://127.0.0.1:$PORT"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}zlx API Test Script${NC}"
echo "===================="
echo "Model: $MODEL_NAME"
echo "Port: $PORT"
echo ""

# Check if server is running
if ! curl -s "$BASE_URL/v1/health" > /dev/null 2>&1; then
    echo -e "${YELLOW}Server not running. Starting...${NC}"
    
    # Build if needed
    if [ ! -f "./zig-out/bin/zlx" ]; then
        echo "Building zlx..."
        zig build
    fi
    
    # Start server in background
    ./zig-out/bin/zlx --model "$MODEL_NAME" --port "$PORT" &
    SERVER_PID=$!
    
    # Wait for server to start
    echo "Waiting for server to start..."
    for i in {1..30}; do
        if curl -s "$BASE_URL/v1/health" > /dev/null 2>&1; then
            echo -e "${GREEN}Server ready!${NC}"
            break
        fi
        sleep 1
    done
    
    if ! curl -s "$BASE_URL/v1/health" > /dev/null 2>&1; then
        echo "Server failed to start"
        exit 1
    fi
    
    echo ""
fi

# Test 1: List models
echo -e "${BLUE}Test 1: GET /v1/models${NC}"
curl -s "$BASE_URL/v1/models" | python3 -m json.tool 2>/dev/null || curl -s "$BASE_URL/v1/models"
echo ""
echo ""

# Test 2: Non-streaming completion
echo -e "${BLUE}Test 2: POST /v1/chat/completions (non-streaming)${NC}"
curl -s -X POST "$BASE_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$MODEL_NAME\", \"messages\": [{\"role\": \"user\", \"content\": \"Say 'Hello from zlx!'\"}], \"max_tokens\": 50}" | \
    python3 -m json.tool 2>/dev/null || \
    curl -s -X POST "$BASE_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"$MODEL_NAME\", \"messages\": [{\"role\": \"user\", \"content\": \"Say 'Hello from zlx!'\"}], \"max_tokens\": 50}"
echo ""
echo ""

# Test 3: Streaming completion
echo -e "${BLUE}Test 3: POST /v1/chat/completions (streaming)${NC}"
echo "Streaming response:"
curl -N -X POST "$BASE_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$MODEL_NAME\", \"messages\": [{\"role\": \"user\", \"content\": \"Count: 1,2,3\"}], \"stream\": true, \"max_tokens\": 30}"
echo ""
echo ""

echo -e "${GREEN}All tests complete!${NC}"
echo ""
echo "Server running on $BASE_URL"
echo "Press Ctrl+C to stop"

# Keep script running if we started the server
if [ -n "$SERVER_PID" ]; then
    wait $SERVER_PID
fi
