---
phase: phase-12
plan: FIX-01
type: execute
wave: 1
depends_on: []
files_modified: [src/inference/loader.zig, src/deepseek.zig]
autonomous: true
gap_closure: true
requirements: [MOE-01]
must_haves:
  truths:
    - "DeepSeek-Coder-V2-Lite loads weights without 'Key not found' errors"
    - "Weight key names match actual MLX-community safetensors structure"
    - "4-bit quantized weights with .biases/.scales suffixes are handled"
  artifacts:
    - path: "src/inference/loader.zig"
      provides: "Updated weight mapping for DeepSeek"
      changes: ["Updated key names", "Handle quantized suffixes", "Layer 0 dense vs MoE distinction"]
    - path: "src/deepseek.zig"
      provides: "Quantized weight support structure"
      changes: ["Optional bias/scales fields", "Dequantization helper"]
  key_links:
    - from: "loader.registerDeepSeekWeightKeys"
      to: "actual safetensors keys"
      pattern: "switch_mlp.*weight|.*biases|.*scales"
    - from: "loader.mapWeightsToDeepSeek"
      to: "DeepSeekWeights struct"
      pattern: "handle quantized weight groups"
---

## Objective
Fix DeepSeek weight key mapping to match actual MLX-community converted safetensors files, resolving "Key not found in weights_hash" errors.

**Purpose:** Enable DeepSeek-Coder-V2-Lite model to load and run inference.

**Output:** Working weight loading that handles the actual quantized safetensors structure.

## Problem Statement
The current implementation uses incorrect weight key names that don't match the actual DeepSeek safetensors files from mlx-community:

**Error:**
```
Key not found in weights_hash: model.layers.20.mlp.switch_mlp.up_proj.biases
```

**Root Cause:** Phase 12-01 assumed HuggingFace-style weight naming, but mlx-community uses:
- `switch_mlp` instead of `experts` for MoE weights
- `kv_a_proj_with_mqa` instead of `kv_a_proj` for attention
- Quantized weights have `.biases` and `.scales` suffixes (not just `.weight`)
- Layer 0 uses standard MLP (dense), layers 1-26 use MoE with `switch_mlp`

## Technical Context

### Actual vs Expected Key Names

| Component | Expected (Wrong) | Actual (MLX-community) |
|-----------|------------------|--------------------------|
| MoE Expert Weights | `model.layers.{i}.mlp.experts.{e}.*` | `model.layers.{i}.mlp.switch_mlp.*` |
| Quantized Weights | `.weight` only | `.weight`, `.biases`, `.scales` |
| KV Projection | `kv_a_proj` | `kv_a_proj_with_mqa` |
| New Field | Not expected | `kv_a_layernorm` per layer |
| Layer 0 MLP | `mlp.experts` | `mlp.{up,gate,down}_proj` (dense) |

### Quantized Weight Structure
Each quantized weight has 3 components:
```
model.layers.1.mlp.switch_mlp.gate_proj.weight   # The 4-bit weights
model.layers.1.mlp.switch_mlp.gate_proj.biases  # Quantization biases
model.layers.1.mlp.switch_mlp.gate_proj.scales  # Quantization scales
```

## Solution Approach

### 1. Update Weight Key Registration
Modify `registerDeepSeekWeightKeys()` to use actual mlx-community key patterns:
- Replace `experts` with `switch_mlp`
- Add `.biases` and `.scales` registrations for quantized weights
- Add `kv_a_proj_with_mqa` and `kv_a_layernorm`
- Handle Layer 0 as dense MLP (no switch_mlp)

### 2. Implement Quantized Weight Handling
Create helper to reconstruct float weights from 4-bit quantized components:
```zig
fn dequantizeWeight(weight: mlx.Array, biases: mlx.Array, scales: mlx.Array) !mlx.Array {
    // weight: int4 packed values
    // biases: per-channel biases
    // scales: per-channel scales
    // return: dequantized float32 array
}
```

### 3. Fix Layer 0 Dense vs MoE
- Layer 0 uses standard dense MLP: `mlp.{up,gate,down}_proj`
- Layers 1-26 use MoE: `mlp.{switch_mlp,shared_experts}`

## Tasks

<task type="auto">
  <name>Task 1: Analyze Actual Weight Structure</name>
  <files>models/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx/model.safetensors.index.json</files>
  <action>
    Parse the actual safetensors index to extract complete key naming pattern:
    1. Extract all unique weight key patterns
    2. Identify quantized weight groups (.weight + .biases + .scales)
    3. Map layer 0 structure (dense vs MoE)
    4. Document attention layer naming (kv_a_proj_with_mqa, kv_a_layernorm)
    5. Create comprehensive key mapping table
    
    Output: Documented key mappings in comments for reference
  </action>
  <verify>
    <automated>cat models/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx/model.safetensors.index.json | jq '.weight_map | keys' | head -50</automated>
  </verify>
  <done>Complete key mapping table created showing actual vs expected names</done>
</task>

<task type="auto">
  <name>Task 2: Update Weight Key Registration</name>
  <files>src/inference/loader.zig</files>
  <action>
    Rewrite `registerDeepSeekWeightKeys()` function:
    1. Fix Layer 0: Use `mlp.{up,gate,down}_proj.*` (dense, no MoE)
    2. Fix Layers 1-27: Use `mlp.switch_mlp.*` for experts, `mlp.shared_experts.*` for shared
    3. Add `.biases` and `.scales` suffixes for all quantized weights
    4. Fix attention: `kv_a_proj_with_mqa` instead of `kv_a_proj`
    5. Add missing: `kv_a_layernorm` per layer
    6. Fix embed_tokens: Add `.biases` and `.scales`
    7. Fix lm_head: Add `.biases` and `.scales`
    
    Key changes:
    - Replace "mlp.experts.{d}" with "mlp.switch_mlp"
    - Register both .weight/.biases/.scales for each quantized tensor
    - Handle 64 experts via switch_mlp (fused) not individual expert files
  </action>
  <verify>
    <automated>grep -n "switch_mlp\|kv_a_proj_with_mqa\|\.biases\|\.scales" src/inference/loader.zig | head -20</automated>
  </verify>
  <done>All weight keys registered match actual safetensors index structure</done>
</task>

<task type="auto">
  <name>Task 3: Implement Quantized Weight Mapping</name>
  <files>src/inference/loader.zig, src/deepseek.zig</files>
  <action>
    Update `mapWeightsToDeepSeek()` to handle quantized weight groups:
    1. For each quantized weight, load .weight, .biases, and .scales
    2. Store as tuple or struct in DeepSeekWeights
    3. Add dequantization logic or pass all components to inference
    4. Handle embed_tokens and lm_head quantization
    5. Handle per-layer MLP quantization
    
    Modify DeepSeekWeights struct if needed to store quantized components:
    ```zig
    pub const QuantizedWeight = struct {
        weight: mlx.Array,  // 4-bit packed
        biases: mlx.Array,
        scales: mlx.Array,
    };
    ```
  </action>
  <verify>
    <automated>zig build 2>&1 | grep -i "error" | head -10 || echo "Build succeeded or no errors"</automated>
  </verify>
  <done>Weight mapping correctly associates .weight/.biases/.scales groups</done>
</task>

## Success Criteria

- [ ] No more "Key not found in weights_hash" errors for DeepSeek model
- [ ] All weight keys from `model.safetensors.index.json` are registered
- [ ] Quantized weights (.weight + .biases + .scales) are handled correctly
- [ ] Layer 0 dense MLP and layers 1-26 MoE structure both supported
- [ ] `zig build` completes without errors
- [ ] Model at least reaches "weights loaded" stage (even if inference has other issues)

## Key Changes Summary

| Original Key | Fixed Key |
|--------------|-----------|
| `model.layers.0.mlp.experts.*` | `model.layers.0.mlp.{up,gate,down}_proj.*` |
| `model.layers.{i}.mlp.experts.{e}.*` | `model.layers.{i}.mlp.switch_mlp.*` |
| `model.layers.{i}.self_attn.kv_a_proj.*` | `model.layers.{i}.self_attn.kv_a_proj_with_mqa.*` |
| `*.weight` only | `*.weight`, `*.biases`, `*.scales` for quantized |
| Missing | `model.layers.{i}.self_attn.kv_a_layernorm.weight` |

## Dependencies
- MLX.zig with working safetensors loading
- DeepSeek transformer architecture (from Phase 11-04)
- Actual DeepSeek model files in `./models/`
</content>
