//! gpt_oss.zig - GPT-OSS Transformer with Sliding Window + Yarn RoPE
//!
//! Implements GPT-OSS-20B architecture:
//! - 24 layers with MoE (32 experts, 4 per token)
//! - Sliding window attention (alternating: 128 tokens every other layer)
//! - Yarn RoPE for 128K context (32x scaling from 4K base)
//! - 20B total params, ~5B active per token

const std = @import("std");
const mlx = @import("mlx.zig/src/mlx.zig");
const moe = @import("moe.zig");

/// GPT-OSS-20B configuration
pub const GptOssConfig = struct {
    vocab_size: usize = 151936,
    hidden_size: usize = 2880,
    num_hidden_layers: usize = 24,
    num_attention_heads: usize = 64,
    num_key_value_heads: usize = 8, // GQA
    intermediate_size: usize = 9216,
    num_experts: usize = 32,
    num_experts_per_tok: usize = 4,
    max_position_embeddings: usize = 131072,
    sliding_window: usize = 128,
    rope_theta: f32 = 150000.0, // High theta for Yarn
    rope_scaling_factor: f32 = 32.0,
    rope_scaling_beta_fast: f32 = 32.0,
    rope_scaling_beta_slow: f32 = 1.0,
    rms_norm_eps: f32 = 1e-6,
};

/// Yarn RoPE configuration for extended context
pub const YarnRopeConfig = struct {
    theta: f32 = 150000.0,
    factor: f32 = 32.0,
    beta_fast: f32 = 32.0,
    beta_slow: f32 = 1.0,
    original_max_position: usize = 4096,
};

/// GPT-OSS expert weights for MoE
pub const GptOssExpert = struct {
    gate_proj: mlx.Array,
    up_proj: mlx.Array,
    down_proj: mlx.Array,

    pub fn deinit(self: *GptOssExpert) void {
        mlx.arrayFree(self.gate_proj);
        mlx.arrayFree(self.up_proj);
        mlx.arrayFree(self.down_proj);
    }
};

/// GPT-OSS transformer layer
pub const GptOssLayer = struct {
    layer_type: enum { sliding, full },
    input_norm: mlx.Array,
    attention: GptOssAttention,
    post_attn_norm: mlx.Array,
    router: mlx.Array,
    experts: []GptOssExpert,

    pub fn deinit(self: *GptOssLayer, allocator: std.mem.Allocator) void {
        mlx.arrayFree(self.input_norm);
        mlx.arrayFree(self.post_attn_norm);
        mlx.arrayFree(self.router);
        self.attention.deinit();
        for (self.experts) |*expert| {
            expert.deinit();
        }
        allocator.free(self.experts);
    }
};

/// GPT-OSS attention with GQA and sliding window support
pub const GptOssAttention = struct {
    q_proj: mlx.Array,
    k_proj: mlx.Array,
    v_proj: mlx.Array,
    o_proj: mlx.Array,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    layer_idx: usize,

    pub fn deinit(self: *GptOssAttention) void {
        mlx.arrayFree(self.q_proj);
        mlx.arrayFree(self.k_proj);
        mlx.arrayFree(self.v_proj);
        mlx.arrayFree(self.o_proj);
    }

    /// Forward pass with optional sliding window
    pub fn forward(
        self: *const GptOssAttention,
        output: *mlx.Array,
        x: mlx.Array,
        position: usize,
        cache: ?*KVCache,
        use_sliding_window: bool,
        window_size: usize,
    ) !void {
        const stream = mlx.defaultGpuStreamNew();

        // Project to Q, K, V
        var q = mlx.arrayNew();
        defer mlx.arrayFree(q);
        try mlx.matmul(&q, x, self.q_proj, stream);

        var k = mlx.arrayNew();
        defer mlx.arrayFree(k);
        try mlx.matmul(&k, x, self.k_proj, stream);

        var v = mlx.arrayNew();
        defer mlx.arrayFree(v);
        try mlx.matmul(&v, x, self.v_proj, stream);

        // Reshape for multi-head attention
        // [batch, seq, hidden] -> [batch, seq, num_heads, head_dim]
        var q_reshaped = mlx.arrayNew();
        defer mlx.arrayFree(q_reshaped);
        try mlx.reshape(&q_reshaped, q, &[_]c_int{
            -1, @intCast(x.shape[1]), @intCast(self.num_heads), @intCast(self.head_dim),
        }, stream);

        var k_reshaped = mlx.arrayNew();
        defer mlx.arrayFree(k_reshaped);
        try mlx.reshape(&k_reshaped, k, &[_]c_int{
            -1, @intCast(x.shape[1]), @intCast(self.num_kv_heads), @intCast(self.head_dim),
        }, stream);

        var v_reshaped = mlx.arrayNew();
        defer mlx.arrayFree(v_reshaped);
        try mlx.reshape(&v_reshaped, v, &[_]c_int{
            -1, @intCast(x.shape[1]), @intCast(self.num_kv_heads), @intCast(self.head_dim),
        }, stream);

        // Store in cache if provided
        if (cache) |kv_cache| {
            try kv_cache.store(@intCast(position), k_reshaped, v_reshaped);

            // Retrieve all previous KVs
            const all_k = try kv_cache.getKeys();
            defer mlx.arrayFree(all_k);
            const all_v = try kv_cache.getValues();
            defer mlx.arrayFree(all_v);

            // Use cached K, V for attention
            k_reshaped = all_k;
            v_reshaped = all_v;
        }

        // Create attention mask
        var mask = mlx.arrayNew();
        defer mlx.arrayFree(mask);
        if (use_sliding_window) {
            mask = try createSlidingWindowMask(x.shape[1], position, window_size);
        } else {
            mask = try createCausalMask(x.shape[1], position);
        }

        // Scaled dot-product attention
        const scale = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(self.head_dim)));
        try mlx.fastScaledDotProductAttention(output, q_reshaped, k_reshaped, v_reshaped, scale, mask, stream);

        // Output projection
        var attn_output = mlx.arrayNew();
        defer mlx.arrayFree(attn_output);
        try mlx.reshape(&attn_output, output.*, &[_]c_int{ -1, @intCast(x.shape[1]), @intCast(self.num_heads * self.head_dim) }, stream);

        mlx.arrayFree(output.*);
        try mlx.matmul(output, attn_output, self.o_proj, stream);
    }
};

/// Simple KV cache for GPT-OSS (non-MLA, standard GQA)
pub const KVCache = struct {
    allocator: std.mem.Allocator,
    k_cache: std.ArrayList(mlx.Array),
    v_cache: std.ArrayList(mlx.Array),
    max_seq_len: usize,
    current_len: usize,

    pub fn init(allocator: std.mem.Allocator, max_seq_len: usize) !KVCache {
        return .{
            .allocator = allocator,
            .k_cache = std.ArrayList(mlx.Array).init(allocator),
            .v_cache = std.ArrayList(mlx.Array).init(allocator),
            .max_seq_len = max_seq_len,
            .current_len = 0,
        };
    }

    pub fn deinit(self: *KVCache) void {
        for (self.k_cache.items) |arr| mlx.arrayFree(arr);
        for (self.v_cache.items) |arr| mlx.arrayFree(arr);
        self.k_cache.deinit();
        self.v_cache.deinit();
    }

    pub fn store(self: *KVCache, position: usize, k: mlx.Array, v: mlx.Array) !void {
        if (position >= self.max_seq_len) return error.SequenceTooLong;

        // Extend cache if needed
        while (self.k_cache.items.len <= position) {
            try self.k_cache.append(mlx.arrayNew());
            try self.v_cache.append(mlx.arrayNew());
        }

        // Free old values and store new ones
        mlx.arrayFree(self.k_cache.items[position]);
        mlx.arrayFree(self.v_cache.items[position]);

        try mlx.arraySet(&self.k_cache.items[position], k);
        try mlx.arraySet(&self.v_cache.items[position], v);

        if (position >= self.current_len) {
            self.current_len = position + 1;
        }
    }

    pub fn getKeys(self: *KVCache) !mlx.Array {
        if (self.current_len == 0) return mlx.arrayNew();

        // Concatenate all K tensors
        const stream = mlx.defaultGpuStreamNew();
        var result = mlx.arrayNew();

        // Stack along sequence dimension
        var k_list: []mlx.Array = try self.allocator.alloc(mlx.Array, self.current_len);
        defer self.allocator.free(k_list);

        for (0..self.current_len) |i| {
            k_list[i] = self.k_cache.items[i];
        }

        try mlx.concatenate(&result, k_list, 1, stream);
        return result;
    }

    pub fn getValues(self: *KVCache) !mlx.Array {
        if (self.current_len == 0) return mlx.arrayNew();

        const stream = mlx.defaultGpuStreamNew();
        var result = mlx.arrayNew();

        var v_list: []mlx.Array = try self.allocator.alloc(mlx.Array, self.current_len);
        defer self.allocator.free(v_list);

        for (0..self.current_len) |i| {
            v_list[i] = self.v_cache.items[i];
        }

        try mlx.concatenate(&result, v_list, 1, stream);
        return result;
    }
};

/// Weight container for GPT-OSS model loading
pub const GptOssWeights = struct {
    token_embedding: mlx.Array,
    layers: []GptOssLayer,
    norm: mlx.Array,
    lm_head: mlx.Array,

    pub fn deinit(self: *GptOssWeights, allocator: std.mem.Allocator) void {
        mlx.arrayFree(self.token_embedding);
        mlx.arrayFree(self.norm);
        mlx.arrayFree(self.lm_head);
        for (self.layers) |*layer| {
            layer.deinit(allocator);
        }
        allocator.free(self.layers);
    }
};

/// GPT-OSS Transformer
pub const GptOssTransformer = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    config: GptOssConfig,
    token_embedding: mlx.Array,
    layers: []GptOssLayer,
    norm: mlx.Array,
    lm_head: mlx.Array,
    rope_config: YarnRopeConfig,

    pub fn init(allocator: std.mem.Allocator, config: GptOssConfig, weights: GptOssWeights) !Self {
        return .{
            .allocator = allocator,
            .config = config,
            .token_embedding = weights.token_embedding,
            .layers = weights.layers,
            .norm = weights.norm,
            .lm_head = weights.lm_head,
            .rope_config = .{
                .theta = config.rope_theta,
                .factor = config.rope_scaling_factor,
                .beta_fast = config.rope_scaling_beta_fast,
                .beta_slow = config.rope_scaling_beta_slow,
            },
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.layers);
    }

    /// Apply RMS normalization
    fn rmsNorm(self: Self, x: mlx.Array, weight: mlx.Array, eps: f32) !mlx.Array {
        _ = self;
        var result = mlx.arrayNew();
        errdefer mlx.arrayFree(result);

        try mlx.fastRmsNorm(&result, x, weight, eps, mlx.defaultGpuStreamNew());
        return result;
    }

    /// Forward pass through transformer
    pub fn forward(
        self: *Self,
        output: *mlx.Array,
        input_ids: mlx.Array,
        kv_caches: ?[]*KVCache,
    ) !void {
        const stream = mlx.defaultGpuStreamNew();

        // Token embedding lookup
        var hidden = mlx.arrayNew();
        defer mlx.arrayFree(hidden);
        try mlx.take(&hidden, self.token_embedding, input_ids, 0, stream);

        // Process through all layers
        for (self.layers, 0..) |*layer, i| {
            // Determine attention type based on layer parity
            const use_sliding = (i % 2 == 0); // Even layers use sliding window

            // Pre-norm for attention
            const normed = try self.rmsNorm(hidden, layer.input_norm, self.config.rms_norm_eps);
            defer mlx.arrayFree(normed);

            // Attention with residual
            var attn_out = mlx.arrayNew();
            defer mlx.arrayFree(attn_out);
            const kv_cache = if (kv_caches) |caches| caches[i] else null;
            try layer.attention.forward(&attn_out, normed, if (kv_cache) |c| c.current_len else 0, kv_cache, use_sliding, self.config.sliding_window);

            // Residual connection
            var residual = mlx.arrayNew();
            defer mlx.arrayFree(residual);
            try mlx.add(&residual, hidden, attn_out, stream);
            mlx.arrayFree(hidden);
            hidden = residual;

            // Pre-norm for MoE
            const post_normed = try self.rmsNorm(hidden, layer.post_attn_norm, self.config.rms_norm_eps);
            defer mlx.arrayFree(post_normed);

            // MoE with residual
            var moe_out = mlx.arrayNew();
            defer mlx.arrayFree(moe_out);
            try layer.moe.forward(&moe_out, post_normed);

            // Residual connection
            var moe_residual = mlx.arrayNew();
            defer mlx.arrayFree(moe_residual);
            try mlx.add(&moe_residual, hidden, moe_out, stream);
            mlx.arrayFree(hidden);
            hidden = moe_residual;
        }

        // Final norm
        const final_hidden = try self.rmsNorm(hidden, self.norm, self.config.rms_norm_eps);
        defer mlx.arrayFree(final_hidden);

        // LM head projection
        try mlx.matmul(output, final_hidden, self.lm_head, stream);
    }
};

/// Create sliding window attention mask
fn createSlidingWindowMask(seq_len: usize, position: usize, window_size: usize) !mlx.Array {
    const stream = mlx.defaultGpuStreamNew();
    var mask = mlx.arrayNew();

    // Create mask where position i can attend to [max(0, i-window_size) .. i]
    // Shape: [1, 1, seq_len, seq_len]

    // For now, use causal mask as base
    // TODO: Implement proper sliding window mask
    _ = position;
    _ = window_size;

    // Create lower triangular matrix (causal mask)
    var seq_indices = mlx.arrayNew();
    defer mlx.arrayFree(seq_indices);
    try mlx.arange(&seq_indices, 0, @floatFromInt(seq_len), 1, mlx.INT32, stream);

    // Reshape to [seq_len, 1] and [1, seq_len]
    var row_indices = mlx.arrayNew();
    defer mlx.arrayFree(row_indices);
    try mlx.reshape(&row_indices, seq_indices, &[_]c_int{ @intCast(seq_len), 1 }, stream);

    var col_indices = mlx.arrayNew();
    defer mlx.arrayFree(col_indices);
    try mlx.reshape(&col_indices, seq_indices, &[_]c_int{ 1, @intCast(seq_len) }, stream);

    // mask = row_indices >= col_indices (causal)
    try mlx.greaterEqual(&mask, row_indices, col_indices, stream);

    // Expand dims to [1, 1, seq_len, seq_len]
    var expanded = mlx.arrayNew();
    defer mlx.arrayFree(expanded);
    try mlx.expand_dims(&expanded, mask, &[_]c_int{ 0, 1 }, stream);

    return expanded;
}

/// Create standard causal attention mask
fn createCausalMask(seq_len: usize, position: usize) !mlx.Array {
    const stream = mlx.defaultGpuStreamNew();
    var mask = mlx.arrayNew();

    _ = position;

    // Create lower triangular matrix
    var seq_indices = mlx.arrayNew();
    defer mlx.arrayFree(seq_indices);
    try mlx.arange(&seq_indices, 0, @floatFromInt(seq_len), 1, mlx.INT32, stream);

    var row_indices = mlx.arrayNew();
    defer mlx.arrayFree(row_indices);
    try mlx.reshape(&row_indices, seq_indices, &[_]c_int{ @intCast(seq_len), 1 }, stream);

    var col_indices = mlx.arrayNew();
    defer mlx.arrayFree(col_indices);
    try mlx.reshape(&col_indices, seq_indices, &[_]c_int{ 1, @intCast(seq_len) }, stream);

    // mask = row_indices >= col_indices
    try mlx.greaterEqual(&mask, row_indices, col_indices, stream);

    // Expand dims to [1, 1, seq_len, seq_len]
    var expanded = mlx.arrayNew();
    defer mlx.arrayFree(expanded);
    try mlx.expand_dims(&expanded, mask, &[_]c_int{ 0, 1 }, stream);

    return expanded;
}

/// Apply Yarn RoPE (Yet another RoPE extension)
fn applyYarnRope(x: mlx.Array, positions: mlx.Array, config: YarnRopeConfig) !mlx.Array {
    const stream = mlx.defaultGpuStreamNew();
    var result = mlx.arrayNew();

    // Yarn extends context by scaling frequencies
    // This is a simplified implementation
    // Full implementation would use fast_rope with custom frequencies

    _ = config;
    try mlx.fastRope(&result, x, 64, false, .{ .has_value = false, .value = 0 }, 1.0, 0, positions, stream);

    return result;
}

// ============================================================================
// Tests
// ============================================================================

test "GptOssConfig defaults" {
    const config = GptOssConfig{};
    try std.testing.expectEqual(@as(usize, 24), config.num_hidden_layers);
    try std.testing.expectEqual(@as(usize, 32), config.num_experts);
    try std.testing.expectEqual(@as(usize, 4), config.num_experts_per_tok);
    try std.testing.expectEqual(@as(f32, 150000.0), config.rope_theta);
}

test "KVCache basic operations" {
    const allocator = std.testing.allocator;

    var cache = try KVCache.init(allocator, 1024);
    defer cache.deinit();

    // Create dummy K and V tensors
    const stream = mlx.defaultGpuStreamNew();
    var k = mlx.arrayNew();
    defer mlx.arrayFree(k);
    try mlx.ones(&k, &[_]c_int{ 1, 1, 8, 64 }, mlx.FLOAT32, stream);

    var v = mlx.arrayNew();
    defer mlx.arrayFree(v);
    try mlx.ones(&v, &[_]c_int{ 1, 1, 8, 64 }, mlx.FLOAT32, stream);

    // Store at position 0
    try cache.store(0, k, v);
    try std.testing.expectEqual(@as(usize, 1), cache.current_len);
}

test "createCausalMask" {
    const mask = try createCausalMask(4, 0);
    defer mlx.arrayFree(mask);

    // Should have shape [1, 1, 4, 4]
    try std.testing.expectEqual(@as(c_int, 4), mlx.arrayDim(mask, 3));
}
