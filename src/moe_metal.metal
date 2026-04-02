// moe_metal.metal - Metal shader for Mixture of Experts routing
// Implements sparse expert selection with top-k routing for DeepSeek-V2
//
// DeepSeek-V2 MoE Configuration:
// - 64 total experts (62 routed + 2 shared)
// - top_k=6 routed experts per token
// - Shared experts always active
//

#include <metal_stdlib>
using namespace metal;

/// Parallel bitonic sort for finding top-k elements
/// Sorts (value, index) pairs by value in descending order
///
/// @param values - Threadgroup array of values to sort
/// @param indices - Threadgroup array of corresponding indices
/// @param len - Number of elements (must be power of 2)
/// @param tid - Thread ID within group
inline void bitonic_sort(
    threadgroup float* values,
    threadgroup int* indices,
    uint len,
    uint tid
) {
    // Bitonic sort algorithm - O(log^2 n) parallel sort
    // Each thread handles one element
    if (tid >= len) return;
    
    // Outer loop: sequence length doubles each iteration
    for (uint k = 2; k <= len; k <<= 1) {
        // Inner loop: compare distance halves each iteration
        for (uint j = k >> 1; j > 0; j >>= 1) {
            // Calculate comparison partner
            uint ixj = tid ^ j;
            
            // Only process if ixj > tid (avoid double work)
            if (ixj > tid && ixj < len) {
                // Determine sort direction: ascending or descending
                bool ascending = (tid & k) == 0;
                
                // Compare and swap if needed
                if ((values[tid] < values[ixj]) == ascending) {
                    // Swap values
                    float temp_val = values[tid];
                    values[tid] = values[ixj];
                    values[ixj] = temp_val;
                    
                    // Swap indices
                    int temp_idx = indices[tid];
                    indices[tid] = indices[ixj];
                    indices[ixj] = temp_idx;
                }
            }
            
            // Synchronize threads within group
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

/// MoE routing kernel - selects top-k experts for each token
///
/// This kernel implements the gating/routing mechanism for Mixture of Experts:
/// 1. Compute gate logits: hidden_states @ gate_weight^T
/// 2. Apply softmax to get routing probabilities
/// 3. Select top-k experts with highest probability
/// 4. Normalize weights over selected experts
///
/// @param hidden_states - Input tensor [batch_size, seq_len, hidden_size]
/// @param gate_weight - Gate projection matrix [num_experts, hidden_size]
/// @param expert_indices - Output: selected expert indices [batch_size, seq_len, top_k]
/// @param expert_weights - Output: routing weights [batch_size, seq_len, top_k]
/// @param batch_size - Number of sequences in batch
/// @param seq_len - Number of tokens per sequence
/// @param hidden_size - Hidden dimension (e.g., 4096 for DeepSeek)
/// @param num_experts - Total number of routed experts (e.g., 64)
/// @param top_k - Number of experts to select per token (e.g., 6)
kernel void moe_route(
    device const float* hidden_states [[buffer(0)]],
    device const float* gate_weight [[buffer(1)]],
    device int* expert_indices [[buffer(2)]],
    device float* expert_weights [[buffer(3)]],
    constant int& batch_size [[buffer(4)]],
    constant int& seq_len [[buffer(5)]],
    constant int& hidden_size [[buffer(6)]],
    constant int& num_experts [[buffer(7)]],
    constant int& top_k [[buffer(8)]],
    uint3 tid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    // Get token position in batch
    uint batch_idx = tid.x;
    uint seq_idx = tid.y;
    
    // Bounds check
    if (batch_idx >= batch_size || seq_idx >= seq_len) {
        return;
    }
    
    // Calculate token index in flattened input
    uint token_idx = batch_idx * seq_len * hidden_size + seq_idx * hidden_size;
    
    // Threadgroup storage for parallel sorting
    // Round num_experts up to next power of 2 for bitonic sort
    const uint max_experts = 128;  // Maximum supported experts (2^7)
    threadgroup float logits[max_experts];
    threadgroup int expert_ids[max_experts];
    
    // Step 1: Compute gate logits for this token
    // logits[i] = dot(hidden[token], gate_weight[i])
    // Each thread computes one expert's logit
    for (uint expert = lid.x; expert < num_experts; expert += 32) {
        float logit = 0.0f;
        
        // Dot product: hidden[token] @ gate_weight[expert]
        for (uint h = 0; h < hidden_size; h++) {
            float hidden_val = hidden_states[token_idx + h];
            float weight_val = gate_weight[expert * hidden_size + h];
            logit += hidden_val * weight_val;
        }
        
        logits[expert] = logit;
        expert_ids[expert] = int(expert);
    }
    
    // Pad remaining entries for sorting
    for (uint expert = num_experts + lid.x; expert < max_experts; expert += 32) {
        logits[expert] = -INFINITY;
        expert_ids[expert] = -1;
    }
    
    // Synchronize before sorting
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Step 2: Sort to find top-k experts
    // Use bitonic sort on the power-of-2 padded array
    // This sorts all experts by logit value (descending)
    uint sort_len = 1;
    while (sort_len < num_experts) sort_len <<= 1;
    sort_len = min(sort_len, max_experts);
    
    bitonic_sort(logits, expert_ids, sort_len, lid.x);
    
    // Synchronize after sorting
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Step 3: Extract top-k and compute softmax normalization
    // Only thread 0 writes results
    if (lid.x == 0) {
        uint output_idx = batch_idx * seq_len * top_k + seq_idx * top_k;
        
        // Compute softmax over top-k logits
        float max_logit = logits[0];  // Sorted descending, so [0] is max
        float exp_sum = 0.0f;
        
        // Calculate exp(logit - max) for numerical stability
        float exp_vals[8];  // Fixed max for top_k (DeepSeek uses 6)
        for (uint k = 0; k < top_k; k++) {
            exp_vals[k] = exp(logits[k] - max_logit);
            exp_sum += exp_vals[k];
        }
        
        // Normalize and write results
        for (uint k = 0; k < top_k; k++) {
            expert_indices[output_idx + k] = expert_ids[k];
            expert_weights[output_idx + k] = exp_vals[k] / exp_sum;
        }
    }
}

/// MoE combine kernel - aggregates expert outputs weighted by routing weights
///
/// This kernel combines the outputs from multiple experts:
/// output[token] = sum_k (expert_weight[k] * expert_output[expert_idx[k]])
///
/// @param expert_outputs - Outputs from all experts [num_experts, batch_size, seq_len, hidden_size]
/// @param expert_indices - Selected expert indices [batch_size, seq_len, top_k]
/// @param expert_weights - Routing weights [batch_size, seq_len, top_k]
/// @param output - Combined output [batch_size, seq_len, hidden_size]
/// @param batch_size - Number of sequences
/// @param seq_len - Sequence length
/// @param hidden_size - Hidden dimension
/// @param top_k - Number of experts per token
kernel void moe_combine(
    device const float* expert_outputs [[buffer(0)]],
    device const int* expert_indices [[buffer(1)]],
    device const float* expert_weights [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant int& batch_size [[buffer(4)]],
    constant int& seq_len [[buffer(5)]],
    constant int& hidden_size [[buffer(6)]],
    constant int& num_experts [[buffer(7)]],
    constant int& top_k [[buffer(8)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint batch_idx = tid.x;
    uint seq_idx = tid.y;
    uint hidden_idx = tid.z;
    
    // Bounds check
    if (batch_idx >= batch_size || seq_idx >= seq_len || hidden_idx >= hidden_size) {
        return;
    }
    
    // Calculate indices
    uint token_idx = batch_idx * seq_len + seq_idx;
    uint routing_idx = token_idx * top_k;
    uint output_idx = token_idx * hidden_size + hidden_idx;
    
    // Weighted sum of expert outputs
    float sum = 0.0f;
    for (uint k = 0; k < top_k; k++) {
        int expert_idx = expert_indices[routing_idx + k];
        float weight = expert_weights[routing_idx + k];
        
        // Get this expert's output for this token position
        uint expert_output_idx = expert_idx * batch_size * seq_len * hidden_size +
                                token_idx * hidden_size + hidden_idx;
        sum += weight * expert_outputs[expert_output_idx];
    }
    
    output[output_idx] = sum;
}

/// Shared expert forward kernel - always active experts
///
/// Shared experts are processed separately from routed experts because:
/// 1. They are always active (not sparse)
/// 2. Can be computed in parallel with routing
/// 3. Output is added to routed expert output
///
/// @param hidden_states - Input tensor
/// @param shared_expert_weights - Weights for shared experts
/// @param output - Output tensor (accumulator)
/// @param num_shared_experts - Number of shared experts (typically 2)
kernel void moe_shared_forward(
    device const float* hidden_states [[buffer(0)]],
    device const float* shared_expert_weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& batch_size [[buffer(3)]],
    constant int& seq_len [[buffer(4)]],
    constant int& hidden_size [[buffer(5)]],
    constant int& intermediate_size [[buffer(6)]],
    constant int& num_shared_experts [[buffer(7)]],
    uint3 tid [[thread_position_in_grid]]
) {
    // Simple pass-through for now
    // Full SwiGLU implementation would be in separate kernel
    uint idx = tid.x;
    if (idx < batch_size * seq_len * hidden_size) {
        output[idx] = hidden_states[idx];
    }
}
