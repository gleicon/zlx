// gptoss_metal.metal - Metal kernels for GPT-OSS
//
// Implements:
// - MoE routing kernel (expert selection)
// - MoE application kernel (sparse expert computation)
// - Sliding window attention kernel

#include <metal_stdlib>
using namespace metal;

// FP4 lookup table for MXFP4 dequantization
constant float fp4_values[16] = {
    0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375,
    0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75
};

// ============================================================================
// MoE Routing Kernel
// ============================================================================
// Computes gate scores and selects top-k experts for each token
//
// Args:
//   hidden_states: [batch*seq_len, hidden_size]
//   gate_weight: [hidden_size, num_experts]
//   expert_indices: [batch*seq_len, top_k] - output: selected expert indices
//   expert_weights: [batch*seq_len, top_k] - output: gate scores (softmaxed)
//   num_experts: number of experts (e.g., 64)
//   top_k: number of experts to select (e.g., 6)

kernel void gptoss_moe_route(
    device const float* hidden_states [[buffer(0)]],
    device const float* gate_weight [[buffer(1)]],
    device int* expert_indices [[buffer(2)]],
    device float* expert_weights [[buffer(3)]],
    constant int& batch_seq_len [[buffer(4)]],
    constant int& hidden_size [[buffer(5)]],
    constant int& num_experts [[buffer(6)]],
    constant int& top_k [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= batch_seq_len) return;

    // Pointer to this token's hidden state
    device const float* h = hidden_states + tid * hidden_size;

    // Compute logits: hidden @ gate_weight.T
    // Store in thread-local array (assuming num_experts <= 128)
    float logits[128];  // Max 128 experts
    for (int e = 0; e < num_experts; e++) {
        float sum = 0.0;
        device const float* gw = gate_weight + e * hidden_size;
        for (int i = 0; i < hidden_size; i++) {
            sum += h[i] * gw[i];
        }
        logits[e] = sum;
    }

    // Find top-k experts using simple selection sort
    // (For small k and moderate num_experts, this is efficient)
    for (int k_idx = 0; k_idx < top_k; k_idx++) {
        int max_idx = k_idx;
        float max_val = logits[k_idx];

        for (int e = k_idx + 1; e < num_experts; e++) {
            if (logits[e] > max_val) {
                max_val = logits[e];
                max_idx = e;
            }
        }

        // Swap to front
        float tmp = logits[k_idx];
        logits[k_idx] = max_val;
        logits[max_idx] = tmp;

        // Store index
        expert_indices[tid * top_k + k_idx] = max_idx;
    }

    // Compute softmax over top-k
    float max_logit = logits[0];
    for (int k_idx = 1; k_idx < top_k; k_idx++) {
        max_logit = max(max_logit, logits[k_idx]);
    }

    float sum_exp = 0.0;
    for (int k_idx = 0; k_idx < top_k; k_idx++) {
        sum_exp += exp(logits[k_idx] - max_logit);
    }

    for (int k_idx = 0; k_idx < top_k; k_idx++) {
        expert_weights[tid * top_k + k_idx] = exp(logits[k_idx] - max_logit) / sum_exp;
    }
}

// ============================================================================
// MoE Application Kernel
// ============================================================================
// Applies selected experts to each token
// Sparse computation: only selected experts are evaluated
//
// Args:
//   hidden: [batch*seq_len, hidden_size] - input hidden states
//   expert_weights: [batch*seq_len, top_k] - weights from routing
//   expert_indices: [batch*seq_len, top_k] - selected expert indices
//   output: [batch*seq_len, hidden_size] - output hidden states
//   expert_weights_up: array of pointers to up_proj weights [num_experts, intermediate_size, hidden_size]
//   expert_weights_gate: array of pointers to gate_proj weights
//   expert_weights_down: array of pointers to down_proj weights [num_experts, hidden_size, intermediate_size]
//   num_experts: total number of experts
//   top_k: number of experts per token
//   hidden_size: hidden dimension
//   intermediate_size: intermediate dimension (typically 4*hidden_size)

kernel void gptoss_moe_apply(
    device const float* hidden [[buffer(0)]],
    device const float* expert_weights [[buffer(1)]],
    device const int* expert_indices [[buffer(2)]],
    device float* output [[buffer(3)]],
    device const float** expert_weights_up [[buffer(4)]],
    device const float** expert_weights_gate [[buffer(5)]],
    device const float** expert_weights_down [[buffer(6)]],
    constant int& batch_seq_len [[buffer(7)]],
    constant int& hidden_size [[buffer(8)]],
    constant int& intermediate_size [[buffer(9)]],
    constant int& num_experts [[buffer(10)]],
    constant int& top_k [[buffer(11)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= batch_seq_len) return;

    // Zero output for accumulation
    for (int h = 0; h < hidden_size; h++) {
        output[tid * hidden_size + h] = 0.0;
    }

    // Input for this token
    device const float* h_in = hidden + tid * hidden_size;

    // Apply each selected expert
    for (int k_idx = 0; k_idx < top_k; k_idx++) {
        int expert_idx = expert_indices[tid * top_k + k_idx];
        float weight = expert_weights[tid * top_k + k_idx];

        if (expert_idx < 0 || expert_idx >= num_experts) continue;

        // Get expert weights
        device const float* w_up = expert_weights_up[expert_idx];
        device const float* w_gate = expert_weights_gate[expert_idx];
        device const float* w_down = expert_weights_down[expert_idx];

        // Compute intermediate activations (SiLU gate)
        // up_proj(x) * sigmoid(gate_proj(x)) * down_proj
        float intermediate[14336];  // Max intermediate size for 20B model

        // Compute up projection
        for (int i = 0; i < intermediate_size; i++) {
            float sum = 0.0;
            device const float* w_up_row = w_up + i * hidden_size;
            for (int h = 0; h < hidden_size; h++) {
                sum += h_in[h] * w_up_row[h];
            }
            intermediate[i] = sum;
        }

        // Apply gate and SiLU
        for (int i = 0; i < intermediate_size; i++) {
            float gate = 0.0;
            device const float* w_gate_row = w_gate + i * hidden_size;
            for (int h = 0; h < hidden_size; h++) {
                gate += h_in[h] * w_gate_row[h];
            }
            // SiLU: x * sigmoid(x)
            float silu = gate * (1.0 / (1.0 + exp(-gate)));
            intermediate[i] *= silu;
        }

        // Down projection and accumulate with weight
        for (int h = 0; h < hidden_size; h++) {
            float sum = 0.0;
            device const float* w_down_row = w_down + h * intermediate_size;
            for (int i = 0; i < intermediate_size; i++) {
                sum += intermediate[i] * w_down_row[i];
            }
            output[tid * hidden_size + h] += weight * sum;
        }
    }
}

// ============================================================================
// Sliding Window Attention Kernel
// ============================================================================
// Computes attention only within a sliding window of past tokens
// This reduces memory from O(N^2) to O(N*window_size)
//
// Args:
//   q: [batch, num_heads, seq_len, head_dim]
//   k: [batch, num_kv_heads, seq_len, head_dim]
//   v: [batch, num_kv_heads, seq_len, head_dim]
//   output: [batch, num_heads, seq_len, head_dim]
//   window_size: size of sliding window
//   num_heads: number of query heads
//   num_kv_heads: number of key/value heads (GQA)
//   head_dim: dimension per head
//   seq_len: sequence length

kernel void gptoss_sw_attention(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant int& batch_size [[buffer(4)]],
    constant int& num_heads [[buffer(5)]],
    constant int& num_kv_heads [[buffer(6)]],
    constant int& head_dim [[buffer(7)]],
    constant int& seq_len [[buffer(8)]],
    constant int& window_size [[buffer(9)]],
    uint3 tid [[thread_position_in_grid]]
) {
    int b = tid.x;  // batch index
    int h = tid.y;  // head index
    int s = tid.z;  // seq index

    if (b >= batch_size || h >= num_heads || s >= seq_len) return;

    // Compute KV head index for GQA
    int kv_h = h / (num_heads / num_kv_heads);

    // Pointers to this query position
    int q_stride = num_heads * seq_len * head_dim;
    int k_stride = num_kv_heads * seq_len * head_dim;
    device const float* q_ptr = q + b * q_stride + h * seq_len * head_dim + s * head_dim;

    // Determine window start (only attend to last window_size tokens)
    int window_start = max(0, s - window_size + 1);
    int window_len = s - window_start + 1;

    // Compute attention scores for window
    float scores[4096];  // Max window size
    float max_score = -1e9;

    for (int t = window_start; t <= s; t++) {
        device const float* k_ptr = k + b * k_stride + kv_h * seq_len * head_dim + t * head_dim;

        float dot = 0.0;
        for (int d = 0; d < head_dim; d++) {
            dot += q_ptr[d] * k_ptr[d];
        }

        // Scale by sqrt(head_dim)
        float score = dot / sqrt(float(head_dim));
        scores[t - window_start] = score;
        max_score = max(max_score, score);
    }

    // Softmax
    float sum_exp = 0.0;
    for (int i = 0; i < window_len; i++) {
        scores[i] = exp(scores[i] - max_score);
        sum_exp += scores[i];
    }

    for (int i = 0; i < window_len; i++) {
        scores[i] /= sum_exp;
    }

    // Weighted sum of values
    for (int d = 0; d < head_dim; d++) {
        float sum = 0.0;
        for (int t = window_start; t <= s; t++) {
            device const float* v_ptr = v + b * k_stride + kv_h * seq_len * head_dim + t * head_dim;
            sum += scores[t - window_start] * v_ptr[d];
        }

        int out_idx = b * q_stride + h * seq_len * head_dim + s * head_dim + d;
        output[out_idx] = sum;
    }
}

// ============================================================================
// Yarn RoPE Kernel
// ============================================================================
// Applies Yet another RoPE extension for long context
// Scales frequencies based on position for extended context support
//
// Args:
//   x: [batch, num_heads, seq_len, head_dim] - input tensor
//   output: [batch, num_heads, seq_len, head_dim] - rotated output
//   positions: [seq_len] - position indices
//   freq_inv: [head_dim/2] - inverse frequencies
//   scale_factor: Yarn scaling factor
//   beta_fast: fast beta for Yarn
//   beta_slow: slow beta for Yarn

kernel void gptoss_yarn_rope(
    device const float* x [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const int* positions [[buffer(2)]],
    device const float* freq_inv [[buffer(3)]],
    constant int& batch_size [[buffer(4)]],
    constant int& num_heads [[buffer(5)]],
    constant int& seq_len [[buffer(6)]],
    constant int& head_dim [[buffer(7)]],
    constant float& theta [[buffer(8)]],
    constant float& scale_factor [[buffer(9)]],
    constant float& beta_fast [[buffer(10)]],
    constant float& beta_slow [[buffer(11)]],
    constant int& original_max_pos [[buffer(12)]],
    uint3 tid [[thread_position_in_grid]]
) {
    int b = tid.x;
    int h = tid.y;
    int s = tid.z;

    if (b >= batch_size || h >= num_heads || s >= seq_len) return;

    int pos = positions[s];

    // Apply rotation to each pair of dimensions
    for (int d = 0; d < head_dim / 2; d++) {
        // Base frequency
        float freq = theta * freq_inv[d];

        // Yarn scaling for long context
        if (pos > original_max_pos) {
            float ratio = float(pos) / float(original_max_pos);
            float factor;
            if (ratio > beta_fast) {
                factor = scale_factor;
            } else if (ratio < beta_slow) {
                factor = 1.0;
            } else {
                factor = 1.0 + (scale_factor - 1.0) * (ratio - beta_slow) / (beta_fast - beta_slow);
            }
            freq /= factor;
        }

        float angle = float(pos) * freq;
        float cos_a = cos(angle);
        float sin_a = sin(angle);

        int idx1 = ((b * num_heads + h) * seq_len + s) * head_dim + d * 2;
        int idx2 = idx1 + 1;

        float x1 = x[idx1];
        float x2 = x[idx2];

        output[idx1] = x1 * cos_a - x2 * sin_a;
        output[idx2] = x1 * sin_a + x2 * cos_a;
    }
}

// ============================================================================
// MXFP4 Dequantization Kernel
// ============================================================================
// Dequantizes MXFP4 format to BF16 for computation
//
// Args:
//   blocks: [num_blocks * block_size / 2] - packed FP4 values (2 per byte)
//   scales: [num_blocks] - block scale factors
//   output: [num_elements] - dequantized BF16 output
//   num_blocks: number of blocks
//   block_size: elements per block (typically 32)

kernel void mxfp4_dequantize(
    device const uchar* blocks [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant int& num_blocks [[buffer(3)]],
    constant int& block_size [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    int total_elements = num_blocks * block_size;
    if (tid >= total_elements) return;

    int block_idx = tid / block_size;
    int idx_in_block = tid % block_size;

    // Load packed byte (2 FP4 values)
    uchar packed = blocks[tid / 2];
    uchar nibble = (tid % 2 == 0) ? (packed & 0x0F) : (packed >> 4);

    // Lookup FP4 value
    float value = fp4_values[nibble];

    // Apply block scale
    float scale = scales[block_idx];
    value *= scale;

    // Store as BF16 (half precision)
    output[tid] = half(value);
}

// ============================================================================
// Helper: Shared Expert Addition
// ============================================================================
// Adds shared expert output to routed expert output
// Shared experts are always active (not sparse)
//
// Args:
//   routed_output: [batch*seq_len, hidden_size] - from MoE routing
//   shared_output: [batch*seq_len, hidden_size] - from shared experts
//   output: [batch*seq_len, hidden_size] - combined output
//   num_tokens: number of tokens
//   hidden_size: hidden dimension

kernel void gptoss_add_shared_experts(
    device const float* routed_output [[buffer(0)]],
    device const float* shared_output [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& num_tokens [[buffer(3)]],
    constant int& hidden_size [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= num_tokens * hidden_size) return;

    output[tid] = routed_output[tid] + shared_output[tid];
}
