---
phase: phase-12
plan: FIX-02
type: execute
wave: 1
depends_on: []
files_modified: [models/Qwen2.5-Coder-1.5B-4bitq/config.json]
autonomous: true
gap_closure: true
requirements: [MOE-01]
must_haves:
  truths:
    - "Qwen2.5-Coder-1.5B-4bitq has valid JSON config"
    - "Model can be loaded without syntax errors"
    - "Config contains required fields: model_type, hidden_size, etc."
  artifacts:
    - path: "models/Qwen2.5-Coder-1.5B-4bitq/config.json"
      provides: "Valid Qwen model configuration"
      changes: ["Replace error content with valid config.json"]
  key_links:
    - from: "Config file"
      to: "JSON parser"
      pattern: "Valid JSON with model_type, vocab_size, hidden_size"
---

## Objective
Fix the corrupted config.json for Qwen2.5-Coder-1.5B-4bitq model, replacing the error page content with valid configuration.

**Purpose:** Enable the Qwen 4bit quantized model to load and be used for inference.

**Output:** Valid config.json that passes JSON parsing.

## Problem Statement

**Error:**
```
debug: Skipping Qwen2.5-Coder-1.5B-4bitq: SyntaxError
```

**Root Cause:** The config.json file contains an error page instead of valid JSON:
```
Invalid username or password.
```

This appears to be a failed download from HuggingFace that saved an authentication error page instead of the actual config.

## Technical Context

### Expected Config Structure
Qwen2.5 models require these fields in config.json:
```json
{
  "model_type": "qwen2",
  "architectures": ["Qwen2ForCausalLM"],
  "vocab_size": 151936,
  "hidden_size": 1536,
  "num_hidden_layers": 28,
  "num_attention_heads": 12,
  "num_key_value_heads": 2,
  "intermediate_size": 8960,
  "rope_theta": 1000000.0,
  "sliding_window": 32768,
  "max_position_embeddings": 32768,
  "rms_norm_eps": 1e-06,
  "attention_dropout": 0.0,
  "torch_dtype": "bfloat16",
  "quantization_config": {
    "bits": 4,
    "group_size": 128
  }
}
```

## Solution Approach

### Option A: Re-download from HuggingFace (Preferred)
Download the correct config.json from mlx-community:
```bash
curl -L -o models/Qwen2.5-Coder-1.5B-4bitq/config.json \
  "https://huggingface.co/mlx-community/Qwen2.5-Coder-1.5B-4bit/raw/main/config.json"
```

### Option B: Create from Reference
If download fails, create config based on:
1. Qwen2.5-Coder-1.5B-Instruct-4bit (non-quantized reference in models/)
2. HuggingFace documentation for Qwen2.5 architecture
3. Model card specifications

### Option C: Fix via Python Script
Use mlx-lm to inspect and regenerate config:
```python
from mlx_lm import load
model, tokenizer = load("mlx-community/Qwen2.5-Coder-1.5B-4bit")
# Extract config from loaded model
```

## Tasks

<task type="auto">
  <name>Task 1: Verify Model Exists on HuggingFace</name>
  <files>N/A (web check)</files>
  <action>
    Check if mlx-community/Qwen2.5-Coder-1.5B-4bit still exists on HuggingFace:
    1. Verify model repository accessibility
    2. Check if it requires authentication (Gated model)
    3. Identify alternative URLs if main is unavailable
    4. Document download URL
    
    Search for alternatives:
    - mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit
    - mlx-community/Qwen2.5-Coder-1.5B-4bit
    - Qwen official quantized versions
  </action>
  <verify>
    <automated>curl -sI "https://huggingface.co/mlx-community/Qwen2.5-Coder-1.5B-4bit" | head -5</automated>
  </verify>
  <done>Confirmed model repository exists and is accessible (or found alternative)</done>
</task>

<task type="auto">
  <name>Task 2: Download Correct Config</name>
  <files>models/Qwen2.5-Coder-1.5B-4bitq/config.json</files>
  <action>
    Replace the corrupted config.json with valid content:
    
    **Attempt 1: Direct download from HuggingFace**
    ```bash
    curl -L -o models/Qwen2.5-Coder-1.5B-4bitq/config.json \
      "https://huggingface.co/mlx-community/Qwen2.5-Coder-1.5B-4bit/raw/main/config.json"
    ```
    
    **Attempt 2: Download via HuggingFace CLI if available**
    ```bash
    huggingface-cli download mlx-community/Qwen2.5-Coder-1.5B-4bit \
      --local-dir models/Qwen2.5-Coder-1.5B-4bitq \
      --include "config.json"
    ```
    
    **Attempt 3: Create from reference model**
    Copy from models/Qwen2.5-Coder-1.5B-Instruct-4bit/config.json and adjust for 4bit quantization.
    
    **Validation:** Ensure downloaded file starts with `{` not error text.
  </action>
  <verify>
    <automated>head -1 models/Qwen2.5-Coder-1.5B-4bitq/config.json | grep -q "^{" && echo "Valid JSON start" || echo "Still invalid"</automated>
  </verify>
  <done>config.json contains valid JSON starting with `{`</done>
</task>

<task type="auto">
  <name>Task 3: Verify Model Loading</name>
  <files>models/Qwen2.5-Coder-1.5B-4bitq/config.json</files>
  <action>
    Test that the model can now be detected:
    1. Parse JSON to verify structure
    2. Check required fields present: model_type, vocab_size, hidden_size
    3. Run architecture detection: should return ModelType.qwen
    4. Confirm model appears in available models list
    
    Test parsing:
    ```zig
    const config = try loadConfigInfo(allocator, "models/Qwen2.5-Coder-1.5B-4bitq/config.json");
    const model_type = detectArchitecture(config);
    // Should return .qwen, not error
    ```
  </action>
  <verify>
    <automated>jq -e '.model_type and .vocab_size and .hidden_size' models/Qwen2.5-Coder-1.5B-4bitq/config.json && echo "Config valid" || echo "Missing required fields"</automated>
  </verify>
  <done>Config parses correctly and contains all required fields for model loading</done>
</task>

## Success Criteria

- [ ] config.json contains valid JSON (starts with `{`, not error text)
- [ ] JSON includes required fields: `model_type`, `vocab_size`, `hidden_size`, `num_hidden_layers`
- [ ] `detectArchitecture()` returns `ModelType.qwen` for this model
- [ ] Model appears in available models list without SyntaxError
- [ ] Can proceed to weight loading stage (if weights are also valid)

## Notes

### Why This Happened
The "Invalid username or password" error suggests the model download used an expired or invalid HuggingFace token. Gated models or models in private repositories may show this error.

### Qwen2.5-Coder-1.5B-4bitq vs Instruct
The `4bitq` suffix likely indicates quantized version. If the standard `4bit` version is available, it may be preferable:
- `Qwen2.5-Coder-1.5B-4bit` - Standard 4-bit quantization
- `Qwen2.5-Coder-1.5B-4bitq` - Alternative quantization format (possibly different group size)

### Alternative Action
If the model is no longer available on HuggingFace, consider:
1. Removing the broken model directory
2. Documenting the alternative: `Qwen2.5-Coder-1.5B-Instruct-4bit`
3. Updating model selection defaults

## Dependencies
- Working internet connection for download
- Or: Reference config from similar Qwen model in models/
