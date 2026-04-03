//! deepseek.zig - DeepSeek-V2 Transformer with MLA + MoE
//!
//! Implements DeepSeek-V2-Lite architecture:
//! - 27 layers with MLA (Multi-head Latent Attention) and MoE (Mixture of Experts)
//! - Compressed KV cache via MLA (8x-16x memory reduction)
//! - Sparse expert activation (64 experts, top-6 per token)
//! - 15.7B total params, 2B active per token

const std = @import("std");
const mlx = @import("mlx.zig/src/mlx.zig");
const mla = @import("mlx.zig/src/mla.zig");
const moe = @import("moe.zig");

/// DeepSeek-V2-Lite configuration
pub const DeepSeekConfig = struct {
    vocab_size: usize = 102400,
    hidden_size: usize = 4096,
    num_hidden_layers: usize = 27,
    num_attention_heads: usize = 16,
    intermediate_size: usize = 11008,
    num_experts: usize = 64,
    num_shared_experts: usize = 2,
    top_k: usize = 6,
    max_position_embeddings: usize = 128000,
    rope_theta: f32 = 10000.0,
    rms_norm_eps: f32 = 1e-6,
    latent_dim: usize = 512,
};

/// DeepSeek transformer layer with MLA + MoE
pub const DeepSeekLayer = struct {
    input_norm: mlx.Array,
    mla: mla.MultiHeadLatentAttention,
    post_attn_norm: mlx.Array,
    moe: moe.MixtureOfExperts,
};

/// Weight container for DeepSeek model loading
pub const DeepSeekWeights = struct {
    token_embedding: mlx.Array,
    layers: []DeepSeekLayer,
    norm: mlx.Array,
    lm_head: mlx.Array,
};

/// DeepSeek-V2 Transformer with MLA attention and MoE FFN
pub const DeepSeekTransformer = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    config: DeepSeekConfig,
    token_embedding: mlx.Array,
    layers: []DeepSeekLayer,
    norm: mlx.Array,
    lm_head: mlx.Array,

    /// Initialize transformer with configuration and weights
    pub fn init(allocator: std.mem.Allocator, config: DeepSeekConfig, weights: DeepSeekWeights) !Self {
        return Self{
            .allocator = allocator,
            .config = config,
            .token_embedding = weights.token_embedding,
            .layers = weights.layers,
            .norm = weights.norm,
            .lm_head = weights.lm_head,
        };
    }

    /// Free all resources
    pub fn deinit(self: *Self) void {
        // Note: weights are owned by the weights struct, not freed here
        self.allocator.free(self.layers);
    }

    /// Apply RMS normalization
    fn rmsNorm(self: Self, x: mlx.Array, weight: mlx.Array, eps: f32) !mlx.Array {
        _ = self;
        var result = mlx.arrayNew();
        errdefer mlx.arrayFree(result);

        // x^2
        var squared = mlx.arrayNew();
        defer mlx.arrayFree(squared);
        try mlx.square(&squared, x, null);

        // mean(x^2, axis=-1, keepdims=true)
        var mean = mlx.arrayNew();
        defer mlx.arrayFree(mean);
        try mlx.mean(&mean, squared, &[_]c_int{-1}, true, null);

        // mean(x^2) + eps
        const eps_array = mlx.arrayNewFloat(eps);
        defer mlx.arrayFree(eps_array);
        var denominator = mlx.arrayNew();
        defer mlx.arrayFree(denominator);
        try mlx.add(&denominator, mean, eps_array, null);

        // sqrt(mean(x^2) + eps)
        var sqrt_denom = mlx.arrayNew();
        defer mlx.arrayFree(sqrt_denom);
        try mlx.sqrt(&sqrt_denom, denominator, null);

        // x / sqrt(...)
        var normalized = mlx.arrayNew();
        defer mlx.arrayFree(normalized);
        try mlx.divide(&normalized, x, sqrt_denom, null);

        // * weight
        try mlx.multiply(&result, normalized, weight, null);

        return result;
    }

    /// Forward pass through transformer
    /// Input: [batch, seq] token IDs
    /// Output: [batch, seq, vocab_size] logits
    pub fn forward(
        self: *Self,
        output: *mlx.Array,
        input_ids: mlx.Array,
        kv_caches: ?[]*mla.CompressedKVCache,
    ) !void {
        const stream = mlx.defaultGpuStreamNew();

        // Step 1: Token embedding lookup
        // input_ids: [batch, seq] -> hidden: [batch, seq, hidden_size]
        var hidden = mlx.arrayNew();
        defer mlx.arrayFree(hidden);
        try mlx.take(&hidden, self.token_embedding, input_ids, 0, stream);

        // Step 2: Process through all layers
        for (self.layers, 0..) |*layer, i| {
            // Pre-norm for attention
            const normed = try self.rmsNorm(hidden, layer.input_norm, self.config.rms_norm_eps);
            defer mlx.arrayFree(normed);

            // MLA with residual connection
            const attn_out = mlx.arrayNew();
            defer mlx.arrayFree(attn_out);
            const kv_cache = if (kv_caches) |caches| caches[i] else null;
            try layer.mla.forward(&attn_out, normed, null, kv_cache, @intCast(if (kv_cache) |c| c.seq_len else 0));

            // residual: hidden = hidden + attn_out
            var residual = mlx.arrayNew();
            defer mlx.arrayFree(residual);
            try mlx.add(&residual, hidden, attn_out, stream);
            mlx.arrayFree(hidden);
            hidden = residual;

            // Pre-norm for MoE
            const post_normed = try self.rmsNorm(hidden, layer.post_attn_norm, self.config.rms_norm_eps);
            defer mlx.arrayFree(post_normed);

            // MoE with residual connection
            const moe_out = mlx.arrayNew();
            defer mlx.arrayFree(moe_out);
            try layer.moe.forward(&moe_out, post_normed);

            // residual: hidden = hidden + moe_out
            var moe_residual = mlx.arrayNew();
            defer mlx.arrayFree(moe_residual);
            try mlx.add(&moe_residual, hidden, moe_out, stream);
            mlx.arrayFree(hidden);
            hidden = moe_residual;
        }

        // Step 3: Final layer normalization
        const final_hidden = try self.rmsNorm(hidden, self.norm, self.config.rms_norm_eps);
        defer mlx.arrayFree(final_hidden);

        // Step 4: LM head projection
        // [batch, seq, hidden] @ [vocab, hidden].T -> [batch, seq, vocab]
        try mlx.matmul(output, final_hidden, self.lm_head, stream);
    }

    /// Sample next token with temperature
    fn sample(self: Self, logits: mlx.Array, temperature: f32) !i32 {
        _ = self;
        var scaled = mlx.arrayNew();
        defer mlx.arrayFree(scaled);

        if (temperature != 1.0) {
            const temp_array = mlx.arrayNewFloat(temperature);
            defer mlx.arrayFree(temp_array);
            try mlx.divide(&scaled, logits, temp_array, null);
        } else {
            try mlx.arraySet(&scaled, logits);
        }

        var probs = mlx.arrayNew();
        defer mlx.arrayFree(probs);
        try mlx.softmax(&probs, scaled, &[_]c_int{-1}, true, null);

        var token = mlx.arrayNew();
        defer mlx.arrayFree(token);
        try mlx.argmax(&token, probs, -1, false, null);

        var result: i32 = 0;
        try mlx.item(&result, token);
        return result;
    }

    /// Generate tokens autoregressively
    /// prompt_ids: [batch, seq] initial token IDs
    /// max_tokens: maximum number of new tokens to generate
    /// temperature: sampling temperature (1.0 = greedy-ish, <1.0 = more random)
    /// Returns: generated token IDs [batch, seq + max_tokens]
    pub fn generate(
        self: *Self,
        prompt_ids: mlx.Array,
        max_tokens: usize,
        temperature: f32,
    ) !mlx.Array {
        const allocator = self.allocator;
        const num_layers = self.config.num_hidden_layers;

        // Initialize compressed KV caches for all layers
        var caches = try allocator.alloc(*mla.CompressedKVCache, num_layers);
        defer allocator.free(caches);

        for (0..num_layers) |i| {
            caches[i] = try allocator.create(mla.CompressedKVCache);
            caches[i].* = mla.CompressedKVCache.init(
                allocator,
                self.config.max_position_embeddings,
                self.config.latent_dim,
            );
        }
        defer {
            for (caches) |cache| {
                cache.deinit();
                allocator.destroy(cache);
            }
        }

        // Current token IDs (starts with prompt)
        var current_ids = mlx.arrayNew();
        defer mlx.arrayFree(current_ids);
        try mlx.arraySet(&current_ids, prompt_ids);

        // Collect output tokens
        var output_tokens = std.ArrayList(i32).init(allocator);
        defer output_tokens.deinit();

        // Generation loop
        for (0..max_tokens) |_| {
            // Forward pass to get logits
            const logits = mlx.arrayNew();
            defer mlx.arrayFree(logits);
            try self.forward(&logits, current_ids, caches);

            // Get logits for last token: [batch, seq, vocab] -> [batch, vocab]
            const last_logits = mlx.arrayNew();
            defer mlx.arrayFree(last_logits);
            try mlx.take(&last_logits, logits, mlx.int(-1), 1, null);

            // Sample next token
            const next_token = try self.sample(last_logits, temperature);

            // Append to output
            try output_tokens.append(next_token);

            // Prepare for next iteration: single token
            mlx.arrayFree(current_ids);
            const next_token_arr = mlx.arrayNewData(&next_token, .{ 1, 1 }, mlx.INT32) catch |err| {
                // Need to handle this case - create from scalar
                _ = err;
                const temp = mlx.arrayNewFloat(@floatFromInt(next_token));
                defer mlx.arrayFree(temp);
                try mlx.astype(&current_ids, temp, mlx.INT32, null);
                continue;
            };
            defer mlx.arrayFree(next_token_arr);
            try mlx.arraySet(&current_ids, next_token_arr);

            // TODO: Check for EOS token
            // For now, continue generating
        }

        // Convert output tokens to MLX array
        const output_len = output_tokens.items.len;
        if (output_len == 0) {
            return mlx.arrayNew();
        }

        var result = mlx.arrayNew();
        try mlx.arrayNewData(&result, output_tokens.items.ptr, .{ 1, output_len }, mlx.INT32);
        return result;
    }
};
