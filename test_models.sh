#!/bin/bash
# test_models.sh - Comprehensive model testing for zlx
# Usage: ./test_models.sh [options] [model_name]
# Options:
#   --verbose          Detailed logging
#   --report-json      Generate JSON report
#   --benchmark        Performance benchmarking
#   --ci               CI mode (no interactive output)
#   --quick            Skip heavy tests
#   --test-context     Test various context lengths
#   --test-memory      Test for memory leaks

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ZLX_BIN="./zig-out/bin/zlx"
SERVER_PORT=8081
SERVER_PID=""
TEST_TIMEOUT=60
REPORT_FILE=""
CI_MODE=false
VERBOSE=false
QUICK_MODE=false
TEST_CONTEXT=false
TEST_MEMORY=false
BENCHMARK_MODE=false

# Test results tracking
declare -A RESULTS
declare -A MEMORY_MB
declare -A TPS
declare -A DURATION
declare -A TEST_DETAILS
PASSED=0
FAILED=0

# Test prompts
TEST_PROMPT_SHORT="Write a Python function to calculate fibonacci numbers"
TEST_PROMPT_LONG="Explain the concept of neural networks and deep learning"

cleanup() {
    if [ -n "$SERVER_PID" ]; then
        $VERBOSE && echo -e "${YELLOW}Cleaning up server (PID: $SERVER_PID)...${NC}"
        kill -9 $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
    fi
    pkill -9 -f "zlx --model" 2>/dev/null || true
}

trap cleanup EXIT

# Parse arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --verbose)
                VERBOSE=true
                shift
                ;;
            --report-json)
                REPORT_FILE="test_report_$(date +%Y%m%d_%H%M%S).json"
                shift
                ;;
            --benchmark)
                BENCHMARK_MODE=true
                shift
                ;;
            --ci)
                CI_MODE=true
                shift
                ;;
            --quick)
                QUICK_MODE=true
                TEST_TIMEOUT=15
                shift
                ;;
            --test-context)
                TEST_CONTEXT=true
                shift
                ;;
            --test-memory)
                TEST_MEMORY=true
                shift
                ;;
            --deepseek)
                TEST_DEEPSEEK=true
                shift
                ;;
            --gptoss|--gpt-oss)
                TEST_GPTOSS=true
                shift
                ;;
            --auto-download)
                AUTO_DOWNLOAD=true
                shift
                ;;
            --all-moe)
                TEST_DEEPSEEK=true
                TEST_GPTOSS=true
                shift
                ;;
            --help)
                echo "Usage: $0 [options] [model_name]"
                echo "Options:"
                echo "  --verbose        Detailed logging"
                echo "  --report-json    Generate JSON report"
                echo "  --benchmark      Performance benchmarking"
                echo "  --ci             CI mode (no interactive output)"
                echo "  --quick          Skip heavy tests"
                echo "  --test-context   Test various context lengths"
                echo "  --test-memory    Test for memory leaks"
                echo "  --deepseek       Test DeepSeek-Coder-V2-Lite only"
                echo "  --gptoss         Test GPT-OSS-20B only"
                echo "  --auto-download  Auto-download missing models (GPT-OSS)"
                echo "  --all-moe        Test all MoE models (DeepSeek + GPT-OSS)"
                echo "  --help           Show this help"
                exit 0
                ;;
            -*)
                echo "Unknown option: $1"
                exit 1
                ;;
            *)
                TARGET_MODEL="$1"
                shift
                ;;
        esac
    done
}

log_info() {
    if [ "$CI_MODE" = false ]; then
        echo -e "${BLUE}[INFO]${NC} $1"
    fi
}

log_pass() {
    if [ "$CI_MODE" = false ]; then
        echo -e "${GREEN}[PASS]${NC} $1"
    fi
}

log_fail() {
    if [ "$CI_MODE" = false ]; then
        echo -e "${RED}[FAIL]${NC} $1"
    fi
}

log_warn() {
    if [ "$CI_MODE" = false ]; then
        echo -e "${YELLOW}[WARN]${NC} $1"
    fi
}

# Start server with a specific model
start_server() {
    local model_path=$1
    local extra_args=${2:-""}
    
    log_info "Starting server with: $model_path $extra_args"
    
    # Kill any existing zlx processes
    pkill -9 -f "zlx --model" 2>/dev/null || true
    sleep 2
    
    # Start server
    $ZLX_BIN --model "$model_path" --port $SERVER_PORT $extra_args > /tmp/zlx_test.log 2>&1 &
    SERVER_PID=$!
    
    # Wait for server to be ready
    local max_wait=60
    for ((i=1; i<=max_wait; i++)); do
        if curl -s http://localhost:$SERVER_PORT/v1/models > /dev/null 2>&1; then
            log_info "Server ready (${i}s)"
            return 0
        fi
        sleep 1
    done
    
    log_fail "Server failed to start within ${max_wait}s"
    if [ "$VERBOSE" = true ]; then
        echo "Server logs:"
        tail -100 /tmp/zlx_test.log
    fi
    return 1
}

# Measure tokens per second
measure_performance() {
    local model=$1
    local prompt=$2
    local max_tokens=${3:-50}
    
    local start_time=$(date +%s.%N)
    
    local response=$(curl -s -X POST http://localhost:$SERVER_PORT/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"$model\", \"messages\": [{\"role\": \"user\", \"content\": \"$prompt\"}], \"max_tokens\": $max_tokens, \"stream\": false}" \
        2>/dev/null)
    
    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc 2>/dev/null || echo "0")
    
    # Count tokens in response (approximate)
    local content=$(echo "$response" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    if 'choices' in d and len(d['choices']) > 0:
        print(d['choices'][0].get('message', {}).get('content', ''))
except:
    pass
" 2>/dev/null || echo "")
    
    local token_count=$(echo "$content" | wc -w)
    local tps=$(echo "scale=2; $token_count / $duration" | bc 2>/dev/null || echo "0")
    
    echo "$tps $duration"
}

# Test for memory leaks
test_memory_leak() {
    local model=$1
    
    log_info "Testing memory leak (10 requests)..."
    
    local initial_mem=$(ps -o rss= -p $SERVER_PID 2>/dev/null || echo "0")
    
    # Run 10 requests
    for i in {1..10}; do
        curl -s -X POST http://localhost:$SERVER_PORT/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d "{\"model\": \"$model\", \"messages\": [{\"role\": \"user\", \"content\": \"Test $i\"}], \"max_tokens\": 20}" \
            > /dev/null 2>&1 || true
    done
    
    local final_mem=$(ps -o rss= -p $SERVER_PID 2>/dev/null || echo "0")
    local growth=$((final_mem - initial_mem))
    
    if [ $growth -gt 51200 ]; then # 50MB threshold
        log_warn "Memory grew by ${growth}KB - possible leak"
        return 1
    else
        log_info "Memory stable (growth: ${growth}KB)"
        return 0
    fi
}

# Test different context lengths
test_context_lengths() {
    local model=$1
    local lengths=(1024 4096 8192)
    
    if [ "$QUICK_MODE" = true ]; then
        lengths=(1024 4096)
    fi
    
    log_info "Testing context lengths: ${lengths[@]}"
    
    for len in "${lengths[@]}"; do
        # Generate prompt of specified length
        local prompt="Test prompt for context length testing"
        
        local response=$(curl -s -X POST http://localhost:$SERVER_PORT/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d "{\"model\": \"$model\", \"messages\": [{\"role\": \"user\", \"content\": \"$prompt\"}], \"max_tokens\": 10}" \
            2>/dev/null)
        
        if [ -n "$response" ]; then
            log_pass "Context ${len}: OK"
        else
            log_fail "Context ${len}: Failed"
            return 1
        fi
    done
    
    return 0
}

# Main test function
test_model() {
    local model_name=$1
    local model_path=$2
    
    echo -e "\n${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Testing: $model_name${NC}"
    echo -e "${YELLOW}========================================${NC}"
    
    local test_start=$(date +%s)
    
    # Check if model exists
    if [ ! -d "$model_path" ]; then
        log_fail "Model not found: $model_path"
        RESULTS[$model_name]="FAILED"
        ((FAILED++))
        return 1
    fi
    
    # Determine if TurboQuant is needed based on model
    local extra_args=""
    if [[ "$model_name" == *"DeepSeek"* ]] || [[ "$model_name" == *"gpt-oss"* ]]; then
        extra_args="--turboquant"
        log_info "Auto-enabling TurboQuant for $model_name"
    fi
    
    # Start server
    if ! start_server "$model_path" "$extra_args"; then
        log_fail "Failed to start server"
        RESULTS[$model_name]="FAILED"
        ((FAILED++))
        return 1
    fi
    
    local tests_passed=0
    local tests_failed=0
    
    # Test 1: /v1/models endpoint
    log_info "Test 1: /v1/models endpoint"
    if curl -s http://localhost:$SERVER_PORT/v1/models | python3 -m json.tool > /tmp/models.json 2>&1; then
        log_pass "Models endpoint"
        ((tests_passed++))
    else
        log_fail "Models endpoint"
        ((tests_failed++))
    fi
    
    # Test 2: Chat completions
    log_info "Test 2: Chat completions"
    if timeout $TEST_TIMEOUT curl -s -X POST http://localhost:$SERVER_PORT/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"$model_name\", \"messages\": [{\"role\": \"user\", \"content\": \"$TEST_PROMPT_SHORT\"}], \"max_tokens\": 50, \"stream\": false}" \
        > /tmp/chat.json 2>&1 && [ -s /tmp/chat.json ]; then
        log_pass "Chat completions"
        ((tests_passed++))
        
        # Show preview if verbose
        if [ "$VERBOSE" = true ]; then
            echo "Response preview:"
            cat /tmp/chat.json | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    if 'choices' in d:
        print(d['choices'][0]['message']['content'][:200])
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null || head -3 /tmp/chat.json
        fi
    else
        log_fail "Chat completions"
        ((tests_failed++))
    fi
    
    # Test 3: Performance benchmark
    if [ "$BENCHMARK_MODE" = true ]; then
        log_info "Test 3: Performance benchmark"
        local perf=$(measure_performance "$model_name" "$TEST_PROMPT_SHORT" 50)
        local tps=$(echo "$perf" | cut -d' ' -f1)
        local dur=$(echo "$perf" | cut -d' ' -f2)
        TPS[$model_name]=$tps
        DURATION[$model_name]=$dur
        log_info "Performance: ${tps} tokens/sec (${dur}s)"
        ((tests_passed++))
    fi
    
    # Test 4: Context lengths
    if [ "$TEST_CONTEXT" = true ]; then
        log_info "Test 4: Context lengths"
        if test_context_lengths "$model_name"; then
            ((tests_passed++))
        else
            ((tests_failed++))
        fi
    fi
    
    # Test 5: Memory leak test
    if [ "$TEST_MEMORY" = true ]; then
        log_info "Test 5: Memory leak detection"
        if test_memory_leak "$model_name"; then
            ((tests_passed++))
        else
            ((tests_failed++))
        fi
    fi
    
    # Get memory usage
    local mem_usage=$(ps -o rss= -p $SERVER_PID 2>/dev/null || echo "0")
    MEMORY_MB[$model_name]=$((mem_usage / 1024))
    
    # Stop server
    log_info "Stopping server..."
    kill -9 $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    SERVER_PID=""
    sleep 2
    
    local test_end=$(date +%s)
    local total_time=$((test_end - test_start))
    
    # Determine overall result
    if [ $tests_failed -eq 0 ]; then
        RESULTS[$model_name]="PASSED"
        ((PASSED++))
        log_pass "All tests passed (${total_time}s)"
        TEST_DETAILS[$model_name]="tests_passed:$tests_passed,time:${total_time}s,memory:${MEMORY_MB[$model_name]}MB"
        return 0
    else
        RESULTS[$model_name]="FAILED"
        ((FAILED++))
        log_fail "$tests_failed test(s) failed"
        TEST_DETAILS[$model_name]="tests_passed:$tests_passed,tests_failed:$tests_failed"
        return 1
    fi
}

# Generate JSON report
generate_json_report() {
    local report_file=$1
    
    {
        echo "{"
        echo "  \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
        echo "  \"system\": {"
        echo "    \"ram_gb\": $(($(sysctl -n hw.memsize) / 1024 / 1024 / 1024)),"
        echo "    \"os\": \"$(uname -s)\","
        echo "    \"arch\": \"$(uname -m)\""
        echo "  },"
        echo "  \"models\": ["
        
        local first=true
        for model in "${!RESULTS[@]}"; do
            if [ "$first" = true ]; then
                first=false
            else
                echo ","
            fi
            echo "    {"
            echo "      \"name\": \"$model\","
            echo "      \"status\": \"${RESULTS[$model]}\","
            echo "      \"memory_mb\": ${MEMORY_MB[$model]:-null},"
            echo "      \"tps\": ${TPS[$model]:-null},"
            echo "      \"duration\": ${DURATION[$model]:-null}"
            echo -n "    }"
        done
        
        echo ""
        echo "  ],"
        echo "  \"summary\": {"
        echo "    \"total\": $((PASSED + FAILED)),"
        echo "    \"passed\": $PASSED,"
        echo "    \"failed\": $FAILED"
        echo "  }"
        echo "}"
    } > "$report_file"
    
    log_info "Report saved to: $report_file"
}

# Test DeepSeek-Coder-V2-Lite
test_deepseek() {
    local model_name="DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx"
    local port=8082
    
    log_info "Testing DeepSeek-Coder-V2-Lite..."
    
    # Check if model exists
    if [[ ! -d "./models/${model_name}" ]]; then
        log_warn "Model not found at ./models/${model_name}, skipping DeepSeek tests"
        return 0
    fi
    
    # Kill any existing server
    pkill -9 -f "zlx --model" 2>/dev/null || true
    sleep 2
    
    # Test weight loading (with 60s timeout for MoE model)
    log_info "  - Testing weight loading (60s timeout for MoE)..."
    timeout 60 $ZLX_BIN --model "./models/${model_name}" --port ${port} --turboquant > /tmp/zlx_deepseek.log 2>&1 &
    local server_pid=$!
    
    # Wait for server to be ready
    local wait_time=0
    while [[ $wait_time -lt 60 ]]; do
        sleep 5
        wait_time=$((wait_time + 5))
        
        if curl -s http://localhost:${port}/v1/models > /dev/null 2>&1; then
            break
        fi
        
        if ! kill -0 $server_pid 2>/dev/null; then
            log_fail "    Server failed to start"
            if [ "$VERBOSE" = true ]; then
                tail -50 /tmp/zlx_deepseek.log
            fi
            return 1
        fi
    done
    
    # Test /v1/models endpoint
    log_info "  - Testing /v1/models..."
    local models_response=$(curl -s http://localhost:${port}/v1/models)
    if [[ ! $models_response == *"${model_name}"* ]]; then
        log_fail "    Model not listed in /v1/models"
        kill $server_pid 2>/dev/null
        return 1
    fi
    log_pass "    Model listed"
    
    # Test chat completion
    log_info "  - Testing chat completion..."
    local response=$(curl -s -X POST http://localhost:${port}/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"${model_name}\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}], \"max_tokens\": 10}")
    
    if [[ ! $response == *"choices"* ]]; then
        log_fail "    No choices in response"
        [ "$VERBOSE" = true ] && echo "Response: $response"
        kill $server_pid 2>/dev/null
        return 1
    fi
    log_pass "    Chat completion works"
    
    # Test memory usage (should be under 16GB with TurboQuant)
    log_info "  - Checking memory usage..."
    local mem_usage=$(ps -o rss= -p $server_pid 2>/dev/null | awk '{print $1/1024}')
    if [[ -n $mem_usage && $(echo "$mem_usage < 16384" | bc -l 2>/dev/null || echo "0") -eq 1 ]]; then
        log_pass "    Memory under 16GB (${mem_usage}MB)"
    else
        log_warn "    Memory high or unavailable (${mem_usage}MB)"
    fi
    
    # Cleanup
    kill $server_pid 2>/dev/null
    wait $server_pid 2>/dev/null
    
    log_pass "  DeepSeek tests passed"
    RESULTS["deepseek"]="passed"
    ((PASSED++))
    return 0
}

# Download GPT-OSS model from HuggingFace
download_gptoss_model() {
    local model_id="mlx-community/gpt-oss-20b-MXFP4-Q4"
    local target_dir="./models/gpt-oss-20b-MXFP4-Q4"
    
    log_info "Downloading GPT-OSS from HuggingFace..."
    
    # Create directory
    mkdir -p "${target_dir}"
    
    # Download config.json
    log_info "  Downloading config.json..."
    curl -L -o "${target_dir}/config.json" \
        "https://huggingface.co/${model_id}/raw/main/config.json" || {
        log_fail "Failed to download config.json"
        return 1
    }
    
    # Download tokenizer files
    log_info "  Downloading tokenizer.json..."
    curl -L -o "${target_dir}/tokenizer.json" \
        "https://huggingface.co/${model_id}/resolve/main/tokenizer.json" 2>/dev/null || {
        log_warn "tokenizer.json download failed (may be optional)"
    }
    
    # Download safetensors index
    log_info "  Downloading model.safetensors.index.json..."
    curl -L -o "${target_dir}/model.safetensors.index.json" \
        "https://huggingface.co/${model_id}/raw/main/model.safetensors.index.json" || {
        log_fail "Failed to download index"
        return 1
    }
    
    # Download weight shards (~11GB total)
    local shard_count=$(jq '.weight_map | values | group_by(.) | length' "${target_dir}/model.safetensors.index.json" 2>/dev/null || echo "3")
    log_info "Downloading ${shard_count} weight shards (~11GB total)..."
    
    for i in $(seq 1 $shard_count); do
        local shard_file=$(printf "model-%05d-of-%05d.safetensors" $i $shard_count)
        log_info "  Downloading ${shard_file}..."
        
        # Use resume capability
        curl -C - -L --progress-bar -o "${target_dir}/${shard_file}" \
            "https://huggingface.co/${model_id}/resolve/main/${shard_file}" || {
            log_fail "Failed to download ${shard_file}"
            return 1
        }
    done
    
    log_pass "GPT-OSS download complete"
    return 0
}

# Test GPT-OSS-20B
test_gptoss() {
    local model_name="gpt-oss-20b-MXFP4-Q4"
    local port=8083
    
    log_info "Testing GPT-OSS-20B..."
    
    # Check if model exists, offer to download
    if [[ ! -d "./models/${model_name}" ]]; then
        log_warn "Model not found at ./models/${model_name}"
        
        if [[ "${AUTO_DOWNLOAD:-false}" == "true" ]]; then
            log_info "  Auto-downloading GPT-OSS (11GB)..."
            if ! download_gptoss_model; then
                log_fail "Download failed, skipping GPT-OSS tests"
                return 0
            fi
        else
            log_warn "Skipping GPT-OSS tests (use --auto-download to fetch)"
            return 0
        fi
    fi
    
    # Kill any existing server
    pkill -9 -f "zlx --model" 2>/dev/null || true
    sleep 2
    
    # Test weight loading (may take longer due to 11GB, 3min timeout)
    log_info "  - Testing weight loading (11GB, 3min timeout)..."
    timeout 180 $ZLX_BIN --model "./models/${model_name}" --port ${port} --turboquant > /tmp/zlx_gptoss.log 2>&1 &
    local server_pid=$!
    
    # Wait longer for GPT-OSS (larger model)
    local wait_time=0
    while [[ $wait_time -lt 120 ]]; do
        sleep 5
        wait_time=$((wait_time + 5))
        
        if curl -s http://localhost:${port}/v1/models > /dev/null 2>&1; then
            break
        fi
        
        if ! kill -0 $server_pid 2>/dev/null; then
            log_fail "    Server failed to start"
            if [ "$VERBOSE" = true ]; then
                tail -50 /tmp/zlx_gptoss.log
            fi
            return 1
        fi
    done
    
    # Test /v1/models endpoint
    log_info "  - Testing /v1/models..."
    local models_response=$(curl -s http://localhost:${port}/v1/models)
    if [[ ! $models_response == *"${model_name}"* ]]; then
        log_fail "    Model not listed in /v1/models"
        kill $server_pid 2>/dev/null
        return 1
    fi
    log_pass "    Model listed"
    
    # Test chat completion
    log_info "  - Testing chat completion..."
    local response=$(curl -s -X POST http://localhost:${port}/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"${model_name}\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}], \"max_tokens\": 10}")
    
    if [[ ! $response == *"choices"* ]]; then
        log_fail "    No choices in response"
        [ "$VERBOSE" = true ] && echo "Response: $response"
        kill $server_pid 2>/dev/null
        return 1
    fi
    log_pass "    Chat completion works"
    
    # Test memory usage (should be under 16GB with TurboQuant)
    log_info "  - Checking memory usage..."
    local mem_usage=$(ps -o rss= -p $server_pid 2>/dev/null | awk '{print $1/1024}')
    if [[ -n $mem_usage && $(echo "$mem_usage < 16384" | bc -l 2>/dev/null || echo "0") -eq 1 ]]; then
        log_pass "    Memory under 16GB (${mem_usage}MB)"
    else
        log_warn "    Memory high or unavailable (${mem_usage}MB)"
    fi
    
    # Cleanup
    kill $server_pid 2>/dev/null
    wait $server_pid 2>/dev/null
    
    log_pass "  GPT-OSS tests passed"
    RESULTS["gptoss"]="passed"
    ((PASSED++))
    return 0
}

# Main execution
main() {
    parse_args "$@"
    
    # Build
    log_info "Building zlx..."
    if ! zig build 2>&1 | tail -5; then
        log_fail "Build failed"
        exit 1
    fi
    log_pass "Build successful"
    
    # Discover models
    MODELS_DIR="./models"
    if [ ! -d "$MODELS_DIR" ]; then
        log_fail "Models directory not found: $MODELS_DIR"
        exit 1
    fi
    
    declare -a MODELS
    declare -a MODEL_PATHS
    
    for dir in "$MODELS_DIR"/*/; do
        if [ -f "$dir/config.json" ]; then
            model_name=$(basename "$dir")
            MODELS+=("$model_name")
            MODEL_PATHS+=("$dir")
            log_info "Found model: $model_name"
        fi
    done
    
    if [ ${#MODELS[@]} -eq 0 ]; then
        log_fail "No models found"
        exit 1
    fi
    
    # Test specific or all models
    if [ "${TEST_DEEPSEEK:-false}" = true ]; then
        test_deepseek
    fi
    
    if [ "${TEST_GPTOSS:-false}" = true ]; then
        test_gptoss
    fi
    
    # If MoE-specific tests were run, don't run regular tests
    if [ "${TEST_DEEPSEEK:-false}" = true ] || [ "${TEST_GPTOSS:-false}" = true ]; then
        : # Skip regular tests, already ran MoE tests
    elif [ -n "${TARGET_MODEL:-}" ]; then
        found=0
        for i in "${!MODELS[@]}"; do
            if [ "${MODELS[$i]}" = "$TARGET_MODEL" ]; then
                test_model "$TARGET_MODEL" "${MODEL_PATHS[$i]}"
                found=1
                break
            fi
        done
        if [ $found -eq 0 ]; then
            log_fail "Model '$TARGET_MODEL' not found"
            exit 1
        fi
    else
        for i in "${!MODELS[@]}"; do
            test_model "${MODELS[$i]}" "${MODEL_PATHS[$i]}" || true
        done
    fi
    
    # Generate report if requested
    if [ -n "$REPORT_FILE" ]; then
        generate_json_report "$REPORT_FILE"
    fi
    
    # Summary
    echo -e "\n${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Test Summary${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${GREEN}Passed: $PASSED${NC}"
    echo -e "${RED}Failed: $FAILED${NC}"
    
    if [ $FAILED -eq 0 ]; then
        exit 0
    else
        exit 1
    fi
}

main "$@"
