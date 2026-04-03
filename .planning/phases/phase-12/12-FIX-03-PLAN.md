---
phase: phase-12
plan: FIX-03
type: execute
wave: 1
depends_on: []
files_modified: [src/inference/loader.zig, src/gpt_oss.zig (if exists)]
autonomous: true
gap_closure: true
requirements: [MOE-02]
must_haves:
  truths:
    - "GPT-OSS-20B model is available from mlx-community or documented alternative"
    - "Weight architecture matches implemented GPT-OSS transformer code"
    - "Model can load without 'model not found' errors"
  artifacts:
    - path: "models/GPT-OSS-20B*/"
      provides: "GPT-OSS model files (weights + config)"
      min_files: ["config.json", "*.safetensors"]
    - path: "src/inference/loader.zig"
      provides: "Updated GPT-OSS detection and loading"
      changes: ["Verify architecture compatibility"]
  key_links:
    - from: "HuggingFace mlx-community"
      to: "Local models directory"
      pattern: "Download or verify GPT-OSS availability"
---

## Objective
Verify GPT-OSS-20B availability from mlx-community, download if available, and ensure architecture compatibility with implemented code.

**Purpose:** Complete GPT-OSS support that was implemented in Phase 12-02 but lacks actual model weights.

**Output:** Working GPT-OSS model or documented reason why it's unavailable.

## Problem Statement

**Current State:**
- GPT-OSS architecture was implemented in Phase 12-02 (sliding window attention, MoE routing, Yarn RoPE)
- No model files exist in `./models/` directory for GPT-OSS
- Cannot test or verify the implementation works

**Error (when trying to load):**
```
ModelNotFound: ./models/GPT-OSS-20B*
```

## Technical Context

### GPT-OSS-20B Architecture (Implemented)
From Phase 12-02 implementation:

| Parameter | Value |
|-----------|-------|
| Total Params | 20B |
| Hidden Size | 4608 |
| Num Layers | 24 |
| Attention Heads | 32 |
| Num Experts | 32 |
| Experts per Token | 4 |
| Context Length | 128K |
| Sliding Window | 128 tokens (every other layer) |
| RoPE Type | Yarn (32x scaling) |

### Expected Model Files
Standard MLX-community layout:
```
models/GPT-OSS-20B-4bit/
├── config.json
├── model.safetensors (or model-00001-of-NN.safetensors)
├── tokenizer.json
└── tokenizer_config.json
```

## Solution Approach

### Step 1: Search for GPT-OSS on HuggingFace
Check multiple possible locations:
1. `mlx-community/GPT-OSS-20B-4bit` (most likely)
2. `mlx-community/gpt-oss-20b` (lowercase variant)
3. `openai/gpt-oss-20b` (official, if available)
4. Any MLX-community repos containing "gpt" and "oss"

### Step 2: Download if Available
Use appropriate method:
- Direct `curl` / `wget` for individual files
- `huggingface-cli download` for full repo
- Python `mlx-lm` library with cache

### Step 3: Verify Architecture Compatibility
Once downloaded:
1. Parse config.json
2. Compare with implemented architecture:
   - Confirm `sliding_window` parameter exists
   - Verify `num_experts` matches (32)
   - Check `rope_scaling.type` is "yarn"
3. Test weight loading
4. Document any discrepancies

### Step 4: Document Alternative if Unavailable
If GPT-OSS is not available on HuggingFace mlx-community:
1. Document the reason (license, gated, not converted)
2. Identify similar models that could test the implementation:
      - Other sliding window + MoE models
   - Alternative: Test with mock weights
3. Update Phase 12-02 status to "architecture only, no weights"

## Tasks

<task type="auto">
  <name>Task 1: Search for GPT-OSS on HuggingFace</name>
  <files>N/A</files>
  <action>
    Search HuggingFace for GPT-OSS model availability:
    
    **Search queries to try:**
    1. `huggingface-cli search "gpt-oss" --organization mlx-community`
    2. Web search: `site:huggingface.co mlx-community GPT OSS`
    3. Web search: `site:huggingface.co "gpt-oss" mlx`
    4. Check openai organization: `openai/gpt-oss-*`
    
    **Document findings:**
    - Exact repository name if found
    - Quantization options (4bit, 8bit, fp16)
    - File sizes and requirements
    - Any access restrictions (gated, license)
    
    **If not found:**
    - Document search attempts
    - Note why unavailable (OpenAI hasn't released MLX version?)
  </action>
  <verify>
    <automated>curl -s "https://huggingface.co/api/models?search=gpt-oss&author=mlx-community" | jq -r '.[].modelId' 2>/dev/null || echo "Search API unavailable or no results"</automated>
  </verify>
  <done>Documented whether GPT-OSS is available and where</done>
</task>

<task type="auto">
  <name>Task 2: Download Model if Available</name>
  <files>models/GPT-OSS-20B-*/</files>
  <action>
    If GPT-OSS found, download to models/:
    
    **Method 1: HuggingFace CLI (preferred)**
    ```bash
    huggingface-cli download mlx-community/GPT-OSS-20B-4bit \
      --local-dir models/GPT-OSS-20B-4bit \
      --local-dir-use-symlinks False
    ```
    
    **Method 2: Direct download (if CLI unavailable)**
    ```bash
    mkdir -p models/GPT-OSS-20B-4bit
    cd models/GPT-OSS-20B-4bit
    
    # Download config and tokenizer
    curl -LO "https://huggingface.co/mlx-community/GPT-OSS-20B-4bit/raw/main/config.json"
    curl -LO "https://huggingface.co/mlx-community/GPT-OSS-20B-4bit/raw/main/tokenizer.json"
    curl -LO "https://huggingface.co/mlx-community/GPT-OSS-20B-4bit/raw/main/tokenizer_config.json"
    
    # Download weights (may be large, use huggingface-cli for resume support)
    huggingface-cli download mlx-community/GPT-OSS-20B-4bit \
      --include "*.safetensors" \
      --local-dir . \
      --local-dir-use-symlinks False
    ```
    
    **Validation:**
    - config.json exists and is valid JSON
    - At least one .safetensors file present
    - Total size reasonable (~10-20GB for 20B 4-bit)
  </action>
  <verify>
    <automated>ls models/GPT-OSS-20B-* 2>/dev/null | head -10 || echo "No GPT-OSS model directory found"</automated>
  </verify>
  <done>GPT-OSS model files downloaded or documented why unavailable</done>
</task>

<task type="auto">
  <name>Task 3: Verify Architecture Compatibility</name>
  <files>models/GPT-OSS-20B-*/config.json, src/inference/loader.zig</files>
  <action>
    Verify downloaded model matches implemented architecture:
    
    **Check config.json fields:**
    ```json
    {
      "model_type": "gpt_oss",  // Should be detected
      "hidden_size": 4608,      // Implementation expects this
      "num_hidden_layers": 24,  // Implementation has 24 layers
      "num_attention_heads": 32,
      "sliding_window": 128,    // Critical for GPT-OSS
      "num_experts": 32,        // MoE parameter
      "num_experts_per_tok": 4,
      "rope_scaling": {
        "type": "yarn",
        "factor": 32.0
      }
    }
    ```
    
    **Update loader.zig if needed:**
    - Ensure GPT-OSS detection handles actual config format
    - Check if weight key naming matches expectations
    - Document any architecture differences
    
    **Test loading:**
    ```bash
    ./zig-out/bin/zlx --model GPT-OSS-20B-4bit --port 8082
    # Should reach "model loaded" stage
    ```
  </action>
  <verify>
    <automated>cat models/GPT-OSS-20B-*/config.json 2>/dev/null | jq -e '.sliding_window and .num_experts' && echo "Architecture compatible" || echo "Missing key fields or no model"</automated>
  </verify>
  <done>Config verified against implementation or discrepancies documented</done>
</task>

<task type="auto">
  <name>Task 4: Document Findings</name>
  <files>PROGRESS.md or .planning/phases/phase-12/GPT-OSS-STATUS.md</files>
  <action>
    Document GPT-OSS status for project records:
    
    **If model downloaded successfully:**
    - Repository: mlx-community/GPT-OSS-20B-4bit
    - Location: models/GPT-OSS-20B-4bit/
    - Size: X GB
    - Status: Available for testing
    
    **If model unavailable:**
    - Search attempts made
    - Reason for unavailability (not released, gated, etc.)
    - Alternative: Test implementation with similar model
    - Recommendation: Defer GPT-OSS testing until weights available
    
    **Update relevant documentation:**
    - README.md model support section
    - Phase 12-02 status notes
    - ROADMAP.md Phase 12 completion criteria
  </action>
  <verify>
    <automated>grep -i "gpt-oss\|GPT-OSS" .planning/phases/phase-12/12-SUMMARY.md README.md 2>/dev/null | head -5 || echo "Documentation update needed"</automated>
  </verify>
  <done>Status documented in project files</done>
</task>

## Success Criteria

**Success Case (Model Available):**
- [ ] GPT-OSS model found on HuggingFace mlx-community
- [ ] Model downloaded to `models/GPT-OSS-20B-*/`
- [ ] config.json valid and contains required fields
- [ ] Architecture matches Phase 12-02 implementation
- [ ] Model loads without errors

**Partial Success (Model Unavailable):**
- [ ] Comprehensive search documented
- [ ] Reason for unavailability explained
- [ ] Alternative testing approach identified
- [ ] Documentation updated to reflect status

**Failure Case:**
- [ ] Model not available AND no suitable alternative found
- [ ] Decision made to defer GPT-OSS support
- [ ] Phase 12-02 marked as "architecture only"

## Alternative Models for Testing

If GPT-OSS unavailable, these could test similar features:

| Model | Has Sliding Window | Has MoE | Similar Size |
|-------|-------------------|---------|--------------|
| Mixtral 8x7B | No | Yes (8 experts) | ~47GB |
| DeepSeek-V2 | No | Yes | ~8GB |
| Qwen2.5 (any) | Yes | No | Various |

**Recommendation:** The DeepSeek model already tests MoE. Sliding window attention in GPT-OSS implementation can be validated against Qwen2.5's sliding window as a fallback.

## Dependencies
- HuggingFace access (for search and download)
- Sufficient disk space (~20GB for 4-bit model)
- Working MLX.zig build for testing
