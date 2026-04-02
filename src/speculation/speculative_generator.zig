//! speculative_generator.zig - Core speculative decoding algorithm
//!
//! Implements the speculative decoding algorithm from "Fast Inference from
//! Transformers via Speculative Decoding" (arXiv:2211.17192).
//!
//! Core algorithm:
//! 1. Use a small, fast "draft" model to generate K tokens autoregressively
//! 2. Run the large "target" model once on the K draft tokens in parallel
//! 3. Compare target model probabilities with draft model probabilities
//! 4. Accept draft tokens with probability: min(1, p_target(x) / p_draft(x))
//! 5. Reject tokens that fail - restart from last accepted position
//! 6. Continue until completion

const std = @import("std");
const mlx = @import("../mlx.zig/src/mlx.zig");
const qwen = @import("../mlx.zig/src/qwen.zig");
const generator = @import("../inference/generator.zig");

/// Maximum speculation depth (prevents excessive memory usage)
pub const MAX_SPECULATION_DEPTH: usize = 8;

/// Default speculation depth (optimal for most cases)
pub const DEFAULT_SPECULATION_DEPTH: usize = 4;

/// Result from a single speculation round
pub const SpeculationResult = struct {
    accepted_tokens: usize,
    rejected_tokens: usize,
    total_generated: usize,
    draft_time_us: u64,
    target_time_us: u64,

    /// Calculate acceptance rate (0.0 to 1.0)
    pub fn getAcceptanceRate(self: SpeculationResult) f32 {
        const total = self.accepted_tokens + self.rejected_tokens;
        if (total == 0) return 0.0;
        return @as(f32, @floatFromInt(self.accepted_tokens)) / @as(f32, @floatFromInt(total));
    }

    /// Calculate average tokens per speculation
    pub fn getTokensPerSpeculation(self: SpeculationResult) f32 {
        if (self.total_generated == 0) return 0.0;
        return @as(f32, @floatFromInt(self.accepted_tokens));
    }
};

/// State tracking for ongoing speculative generation
pub const SpeculativeState = struct {
    current_tokens: std.ArrayList(u32),
    speculation_depth: usize,
    tokens_generated: usize,
    accepted_count: usize,
    rejected_count: usize,
    speculation_count: usize,
};

/// Buffer for storing draft tokens and target verification
pub const SpeculationBuffer = struct {
    draft_tokens: []u32,
    draft_count: usize,
    target_logits: ?[][]f32, // Logits for each position from target model
    draft_probs: ?[]f32, // Probabilities for each draft token

    pub fn init(allocator: std.mem.Allocator, max_depth: usize) !SpeculationBuffer {
        const draft_tokens = try allocator.alloc(u32, max_depth);
        return .{
            .draft_tokens = draft_tokens,
            .draft_count = 0,
            .target_logits = null,
            .draft_probs = null,
        };
    }

    pub fn deinit(self: *SpeculationBuffer, allocator: std.mem.Allocator) void {
        allocator.free(self.draft_tokens);
        if (self.target_logits) |logits| {
            for (logits) |pos_logits| {
                allocator.free(pos_logits);
            }
            allocator.free(logits);
        }
        if (self.draft_probs) |probs| {
            allocator.free(probs);
        }
    }

    pub fn setDraftTokens(self: *SpeculationBuffer, tokens: []const u32) !void {
        if (tokens.len > self.draft_tokens.len) {
            return error.BufferOverflow;
        }
        @memcpy(self.draft_tokens[0..tokens.len], tokens);
        self.draft_count = tokens.len;
    }

    pub fn clear(self: *SpeculationBuffer) void {
        self.draft_count = 0;
    }
};

/// Metrics for tracking speculative decoding performance
pub const SpeculativeMetrics = struct {
    total_speculations: u64,
    total_accepted: u64,
    total_rejected: u64,
    total_draft_time_us: u64,
    total_target_time_us: u64,

    pub fn init() SpeculativeMetrics {
        return .{
            .total_speculations = 0,
            .total_accepted = 0,
            .total_rejected = 0,
            .total_draft_time_us = 0,
            .total_target_time_us = 0,
        };
    }

    pub fn recordSpeculation(
        self: *SpeculativeMetrics,
        accepted: usize,
        rejected: usize,
        draft_time_us: u64,
        target_time_us: u64,
    ) void {
        self.total_speculations += 1;
        self.total_accepted += @intCast(accepted);
        self.total_rejected += @intCast(rejected);
        self.total_draft_time_us += draft_time_us;
        self.total_target_time_us += target_time_us;
    }

    pub fn getStats(self: SpeculativeMetrics) SpeculationStats {
        const total = self.total_accepted + self.total_rejected;
        const acceptance_rate = if (total == 0) 0.0 else @as(f32, @floatFromInt(self.total_accepted)) / @as(f32, @floatFromInt(total));

        const avg_tokens = if (self.total_speculations == 0) 0.0 else @as(f32, @floatFromInt(self.total_accepted)) / @as(f32, @floatFromInt(self.total_speculations));

        // Speedup estimation based on draft/target time ratio and acceptance
        const draft_overhead = if (self.total_target_time_us == 0) 0.0 else @as(f32, @floatFromInt(self.total_draft_time_us)) / @as(f32, @floatFromInt(self.total_target_time_us));
        const estimated_speedup = estimateSpeedup(4, acceptance_rate); // Use default depth

        return .{
            .total_speculations = self.total_speculations,
            .total_accepted = self.total_accepted,
            .total_rejected = self.total_rejected,
            .acceptance_rate = acceptance_rate,
            .avg_tokens_per_speculation = avg_tokens,
            .estimated_speedup = estimated_speedup,
            .draft_overhead_pct = draft_overhead * 100.0,
        };
    }
};

/// Public statistics structure
pub const SpeculationStats = struct {
    total_speculations: u64,
    total_accepted: u64,
    total_rejected: u64,
    acceptance_rate: f32,
    avg_tokens_per_speculation: f32,
    estimated_speedup: f32,
    draft_overhead_pct: f32,
};

/// Calculate acceptance probability for a token
/// Returns min(1.0, p_target / p_draft) with numerical stability
pub fn calculateAcceptanceProbability(p_target: f32, p_draft: f32) f32 {
    // Handle edge cases
    if (p_draft <= 0.0) return 1.0; // Draft probability is 0, always accept
    if (p_target >= p_draft) return 1.0; // Target is more confident
    if (p_target <= 0.0) return 0.0; // Target probability is 0, never accept

    // Normal case: p_target / p_draft
    return p_target / p_draft;
}

/// Decide whether to accept a token based on acceptance probability
/// Uses rejection sampling with uniform random value
pub fn shouldAcceptToken(acceptance_prob: f32, random_value: f32) bool {
    return random_value < acceptance_prob;
}

/// Compute softmax over logits array
pub fn softmax(allocator: std.mem.Allocator, logits: []const f32) ![]f32 {
    var probs = try allocator.alloc(f32, logits.len);
    errdefer allocator.free(probs);

    // Find max for numerical stability
    var max_logit: f32 = -std.math.inf(f32);
    for (logits) |l| {
        if (l > max_logit) max_logit = l;
    }

    // Compute exponentials
    var sum_exp: f32 = 0.0;
    for (0..logits.len) |i| {
        const exp_val = std.math.exp(logits[i] - max_logit);
        probs[i] = exp_val;
        sum_exp += exp_val;
    }

    // Normalize
    if (sum_exp > 0.0) {
        for (probs) |*p| {
            p.* /= sum_exp;
        }
    }

    return probs;
}

/// Extract probability for a specific token from logits
pub fn extractTokenProbability(
    allocator: std.mem.Allocator,
    logits: []const f32,
    token_id: u32,
) !f32 {
    const probs = try softmax(allocator, logits);
    defer allocator.free(probs);

    if (token_id >= probs.len) {
        return error.InvalidTokenId;
    }

    return probs[token_id];
}

/// Estimate speedup based on speculation depth and acceptance rate
/// Formula: speedup ~ K / (1 + α*K) where α is rejection rate (1 - acceptance)
pub fn estimateSpeedup(speculation_depth: usize, acceptance_rate: f32) f32 {
    const k = @as(f32, @floatFromInt(speculation_depth));
    const rejection_rate = 1.0 - acceptance_rate;

    // Base formula: K / (1 + α*K)
    const denominator = 1.0 + rejection_rate * k;
    if (denominator <= 0.0) return 1.0;

    return k / denominator;
}

/// SpeculativeGenerator implements the full speculative decoding algorithm
pub const SpeculativeGenerator = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    target_transformer: *qwen.Transformer,
    draft_transformer: ?*qwen.Transformer, // Optional - may be null for fallback
    target_cache: *mlx.Cache,
    draft_cache: *mlx.Cache,
    speculation_depth: usize,
    options: generator.GenerationOptions,

    // State
    current_tokens: std.ArrayList(u32),
    draft_tokens_buffer: [MAX_SPECULATION_DEPTH]u32,
    is_complete: bool = false,
    tokens_generated: usize = 0,

    // Metrics
    metrics: SpeculativeMetrics,

    /// Initialize speculative generator with target and optional draft model
    pub fn init(
        allocator: std.mem.Allocator,
        target_transformer: *qwen.Transformer,
        draft_transformer: ?*qwen.Transformer,
        speculation_depth: usize,
        options: generator.GenerationOptions,
    ) !Self {
        // Clamp speculation depth to valid range
        const depth = @min(@max(speculation_depth, 1), MAX_SPECULATION_DEPTH);

        // Initialize target cache
        const target_cache = try allocator.create(mlx.Cache);
        errdefer allocator.destroy(target_cache);
        target_cache.* = try mlx.Cache.init(allocator, target_transformer.model.layers.len, 2);

        // Initialize draft cache if draft model available
        var draft_cache: ?*mlx.Cache = null;
        if (draft_transformer) |draft| {
            draft_cache = try allocator.create(mlx.Cache);
            errdefer if (draft_cache) |dc| allocator.destroy(dc);
            draft_cache.?.* = try mlx.Cache.init(allocator, draft.model.layers.len, 2);
        }

        return .{
            .allocator = allocator,
            .target_transformer = target_transformer,
            .draft_transformer = draft_transformer,
            .target_cache = target_cache,
            .draft_cache = draft_cache orelse target_cache, // Use target cache if no draft
            .speculation_depth = depth,
            .options = options,
            .current_tokens = std.ArrayList(u32).init(allocator),
            .draft_tokens_buffer = undefined,
            .metrics = SpeculativeMetrics.init(),
        };
    }

    /// Deinitialize and free all resources
    pub fn deinit(self: *Self) void {
        // Free caches
        if (self.draft_transformer != null) {
            self.draft_cache.deinit();
            self.allocator.destroy(self.draft_cache);
        }

        self.target_cache.deinit();
        self.allocator.destroy(self.target_cache);

        self.current_tokens.deinit(self.allocator);
    }

    /// Generate the next token using speculative decoding
    /// Returns a single token (may involve multiple speculation rounds internally)
    pub fn next(self: *Self) !?generator.Token {
        if (self.is_complete or self.tokens_generated >= self.options.max_tokens) {
            return null;
        }

        // If no draft model, fall back to standard generation
        if (self.draft_transformer == null) {
            return try self.standardGeneration();
        }

        // Perform one speculation round
        const result = try self.speculate();

        // Return the first accepted token
        if (result.accepted_tokens > 0) {
            const token = self.current_tokens.items[self.current_tokens.items.len - result.accepted_tokens];
            return token;
        }

        // No tokens accepted - this shouldn't happen in normal operation
        // Fall back to standard generation for safety
        return try self.standardGeneration();
    }

    /// Perform one round of speculative decoding
    fn speculate(self: *Self) !SpeculationResult {
        // Step 1: Generate K draft tokens autoregressively
        const draft_start = std.time.microTimestamp();
        const draft_tokens = try self.generateDraftTokens(self.speculation_depth);
        defer self.allocator.free(draft_tokens);
        const draft_time = @as(u64, @intCast(std.time.microTimestamp() - draft_start));

        // Step 2: Run target model on [context + draft_tokens] to get parallel logits
        const target_start = std.time.microTimestamp();
        const target_logits = try self.verifyWithTarget(draft_tokens);
        defer {
            for (target_logits) |logits| {
                self.allocator.free(logits);
            }
            self.allocator.free(target_logits);
        }
        const target_time = @as(u64, @intCast(std.time.microTimestamp() - target_start));

        // Step 3: Accept/reject each draft token
        var accepted: usize = 0;
        var rejected: usize = 0;

        for (draft_tokens, 0..) |draft_token, i| {
            // Get probabilities from target and draft
            const target_prob = try self.getTokenProbability(target_logits[i], draft_token);
            const draft_prob = try self.getDraftProbability(i, draft_token);

            // Calculate acceptance probability
            const acceptance_prob = calculateAcceptanceProbability(target_prob, draft_prob);

            // Sample from uniform distribution
            const random_val = if (self.options.seed) |seed|
                self.getSeededRandom(seed, self.tokens_generated + i)
            else
                std.crypto.random.float(f32);

            if (shouldAcceptToken(acceptance_prob, random_val)) {
                // Token accepted - add to current sequence
                try self.current_tokens.append(self.allocator, draft_token);
                accepted += 1;

                // Check for EOS
                if (self.options.stop_on_eos and self.isEosToken(draft_token)) {
                    self.is_complete = true;
                    break;
                }
            } else {
                // Token rejected - use target distribution for this position
                const target_token = try self.sampleFromDistribution(target_logits[i]);
                try self.current_tokens.append(self.allocator, target_token);
                rejected += 1;

                // Reset caches to this position
                try self.resetCachesToPosition(self.current_tokens.items.len - 1);
                break;
            }
        }

        // Update metrics
        self.metrics.recordSpeculation(accepted, rejected, draft_time, target_time);
        self.tokens_generated += accepted + rejected;

        // Update cache positions
        try self.advanceCaches(accepted + rejected);

        return SpeculationResult{
            .accepted_tokens = accepted,
            .rejected_tokens = rejected,
            .total_generated = draft_tokens.len,
            .draft_time_us = draft_time,
            .target_time_us = target_time,
        };
    }

    /// Generate K draft tokens autoregressively using the draft model
    fn generateDraftTokens(self: *Self, count: usize) ![]u32 {
        if (self.draft_transformer == null) {
            return &[_]u32{};
        }

        const draft = self.draft_transformer.?;
        var tokens = try self.allocator.alloc(u32, count);
        errdefer self.allocator.free(tokens);

        var current_seq = try self.allocator.dupe(u32, self.current_tokens.items);
        defer self.allocator.free(current_seq);

        for (0..count) |i| {
            // Prepare input tokens
            const toks_array = try mlx.arrayNewData(
                current_seq.ptr,
                .{ 1, @as(c_int, @intCast(current_seq.len)) },
                mlx.UINT32,
            );
            defer mlx.arrayFree(toks_array);

            // Create mask
            const seq_len = mlx.arrayDim(toks_array, 1);
            var mask_array = mlx.arrayNew();
            defer mlx.arrayFree(mask_array);
            try mlx.createCausalMask(&mask_array, seq_len, self.draft_cache.offset, draft.mlx_config.dtype, draft.mlx_config.stream);

            // Forward pass
            var logits_array = mlx.arrayNew();
            defer mlx.arrayFree(logits_array);
            try draft.model.forward(&logits_array, toks_array, mask_array, self.draft_cache);

            // Sample next token (greedy or with temperature)
            const next_token = try self.sampleFromLogits(logits_array);
            tokens[i] = next_token;

            // Extend sequence for next iteration
            const new_seq = try self.allocator.alloc(u32, current_seq.len + 1);
            @memcpy(new_seq[0..current_seq.len], current_seq);
            new_seq[current_seq.len] = next_token;
            self.allocator.free(current_seq);
            current_seq = new_seq;

            // Check for early EOS
            if (self.options.stop_on_eos and self.isEosToken(next_token)) {
                // Return early with what we have
                const trimmed = try self.allocator.alloc(u32, i + 1);
                @memcpy(trimmed, tokens[0 .. i + 1]);
                self.allocator.free(tokens);
                return trimmed;
            }
        }

        return tokens;
    }

    /// Verify draft tokens with target model (parallel forward pass)
    fn verifyWithTarget(self: *Self, draft_tokens: []const u32) ![][]f32 {
        if (draft_tokens.len == 0) {
            return &[_][]f32{};
        }

        // Combine current context with draft tokens
        const total_len = self.current_tokens.items.len + draft_tokens.len;
        var combined = try self.allocator.alloc(u32, total_len);
        defer self.allocator.free(combined);

        @memcpy(combined[0..self.current_tokens.items.len], self.current_tokens.items);
        @memcpy(combined[self.current_tokens.items.len..], draft_tokens);

        // Create input array
        const toks_array = try mlx.arrayNewData(
            combined.ptr,
            .{ 1, @as(c_int, @intCast(total_len)) },
            mlx.UINT32,
        );
        defer mlx.arrayFree(toks_array);

        // Create causal mask
        var mask_array = mlx.arrayNew();
        defer mlx.arrayFree(mask_array);
        try mlx.createCausalMask(&mask_array, total_len, self.target_cache.offset, self.target_transformer.mlx_config.dtype, self.target_transformer.mlx_config.stream);

        // Run target model
        var logits_array = mlx.arrayNew();
        defer mlx.arrayFree(logits_array);
        try self.target_transformer.model.forward(&logits_array, toks_array, mask_array, self.target_cache);

        // Extract logits for each position
        var all_logits = try self.allocator.alloc([]f32, draft_tokens.len);
        errdefer {
            for (all_logits) |logits| {
                self.allocator.free(logits);
            }
            self.allocator.free(all_logits);
        }

        const vocab_size = @as(usize, @intCast(mlx.arrayDim(logits_array, 2)));

        for (0..draft_tokens.len) |i| {
            // Extract logits for position (context_len + i)
            const pos = self.current_tokens.items.len + i;

            // Take logits[:, pos, :]
            var pos_logits_array = mlx.arrayNew();
            defer mlx.arrayFree(pos_logits_array);
            try mlx.take(&pos_logits_array, logits_array, mlx.int(@intCast(pos)), 1, self.target_transformer.mlx_config.stream);

            // Evaluate and copy data
            try mlx.arrayEval(pos_logits_array);
            const logits_data: [*c]f32 = @ptrCast(@constCast(mlx.C.mlx_array_data_float32(pos_logits_array)));

            all_logits[i] = try self.allocator.alloc(f32, vocab_size);
            for (0..vocab_size) |v| {
                all_logits[i][v] = logits_data[v];
            }
        }

        return all_logits;
    }

    /// Get probability for token from draft model at position
    fn getDraftProbability(_: *Self, _: usize, _: u32) !f32 {
        // In real implementation, would retrieve from draft model's output
        // For now, return 1.0 as placeholder (assumes draft always confident)
        // TODO: Implement proper draft probability tracking during generation
        return 1.0;
    }

    /// Get probability for token from target logits
    fn getTokenProbability(self: *Self, logits: []const f32, token: u32) !f32 {
        const probs = try softmax(self.allocator, logits);
        defer self.allocator.free(probs);

        if (token >= probs.len) {
            return 0.0;
        }

        return probs[token];
    }

    /// Sample from probability distribution with temperature
    fn sampleFromLogits(self: *Self, logits_array: mlx.Array) !u32 {
        // Apply temperature if needed
        var scaled_logits = mlx.arrayNew();
        defer mlx.arrayFree(scaled_logits);

        if (self.options.temperature != 0 and self.options.temperature != 1.0) {
            const temp_scalar = mlx.float(self.options.temperature);
            try mlx.divide(&scaled_logits, logits_array, temp_scalar, self.draft_transformer.?.mlx_config.stream);
        } else {
            try mlx.arraySet(&scaled_logits, logits_array);
        }

        // Take logits for last position
        var last_logits = mlx.arrayNew();
        defer mlx.arrayFree(last_logits);
        try mlx.take(&last_logits, scaled_logits, mlx.int(-1), 1, self.draft_transformer.?.mlx_config.stream);

        // Greedy selection if temperature is 0
        if (self.options.temperature == 0) {
            var next_token_arr = mlx.arrayNew();
            defer mlx.arrayFree(next_token_arr);
            try mlx.argmax(&next_token_arr, last_logits, 1, false, self.draft_transformer.?.mlx_config.stream);

            var token: u32 = 0;
            try mlx.item(&token, next_token_arr);
            return token;
        }

        // Apply softmax and sample
        var probs_array = mlx.arrayNew();
        defer mlx.arrayFree(probs_array);
        const axes = &[_]c_int{1};
        try mlx.softmax(&probs_array, last_logits, axes, false, self.draft_transformer.?.mlx_config.stream);

        // Evaluate to get probability data
        try mlx.arrayEval(probs_array);
        const probs_data: [*c]f32 = @ptrCast(@constCast(mlx.C.mlx_array_data_float32(probs_array)));
        const vocab_size = @as(usize, @intCast(mlx.arrayDim(probs_array, 1)));

        // Sample from distribution
        const random_val = if (self.options.seed) |seed|
            self.getSeededRandom(seed, self.tokens_generated)
        else
            std.crypto.random.float(f32);

        var cumsum: f32 = 0;
        for (0..vocab_size) |i| {
            cumsum += probs_data[i];
            if (random_val <= cumsum) {
                return @intCast(i);
            }
        }

        // Fallback to last token
        return @intCast(vocab_size - 1);
    }

    /// Sample from probability distribution (CPU-side)
    fn sampleFromDistribution(self: *Self, logits: []const f32) !u32 {
        const probs = try softmax(self.allocator, logits);
        defer self.allocator.free(probs);

        const random_val = if (self.options.seed) |seed|
            self.getSeededRandom(seed, self.tokens_generated)
        else
            std.crypto.random.float(f32);

        var cumsum: f32 = 0;
        for (probs, 0..) |p, i| {
            cumsum += p;
            if (random_val <= cumsum) {
                return @intCast(i);
            }
        }

        return @intCast(probs.len - 1);
    }

    /// Generate using standard (non-speculative) method as fallback
    fn standardGeneration(self: *Self) !?generator.Token {
        // Create input from current tokens
        const toks_array = try mlx.arrayNewData(
            self.current_tokens.items.ptr,
            .{ 1, @as(c_int, @intCast(self.current_tokens.items.len)) },
            mlx.UINT32,
        );
        defer mlx.arrayFree(toks_array);

        // Create mask
        const seq_len = mlx.arrayDim(toks_array, 1);
        var mask_array = mlx.arrayNew();
        defer mlx.arrayFree(mask_array);
        try mlx.createCausalMask(&mask_array, seq_len, self.target_cache.offset, self.target_transformer.mlx_config.dtype, self.target_transformer.mlx_config.stream);

        // Forward pass
        var logits_array = mlx.arrayNew();
        defer mlx.arrayFree(logits_array);
        try self.target_transformer.model.forward(&logits_array, toks_array, mask_array, self.target_cache);

        // Sample next token
        const token = try self.sampleFromLogits(logits_array);

        try self.current_tokens.append(self.allocator, token);
        self.tokens_generated += 1;

        // Check for EOS
        if (self.options.stop_on_eos and self.isEosToken(token)) {
            self.is_complete = true;
        }

        // Advance cache
        try self.advanceCaches(1);

        return token;
    }

    /// Check if token is an EOS token
    fn isEosToken(self: *Self, token: u32) bool {
        // Standard EOS tokens for Qwen models
        const eos_tokens = [_]u32{ 151645, 151643 }; //  and <|endoftext|>
        _ = self;
        for (eos_tokens) |eos| {
            if (token == eos) return true;
        }

        return false;
    }

    /// Reset caches to specific position (used on rejection)
    fn resetCachesToPosition(self: *Self, position: usize) !void {
        // Reset target cache offset
        self.target_cache.offset = @intCast(position);

        // Reset draft cache if using separate draft
        if (self.draft_transformer != null) {
            self.draft_cache.offset = @intCast(position);
        }
    }

    /// Advance cache positions by token count
    fn advanceCaches(self: *Self, count: usize) !void {
        self.target_cache.offset += @intCast(count);

        if (self.draft_transformer != null) {
            self.draft_cache.offset += @intCast(count);
        }
    }

    /// Get deterministic random value from seed
    fn getSeededRandom(_: *Self, seed: u32, step: usize) f32 {
        // Simple LCG for reproducibility
        const a: u64 = 1103515245;
        const c: u64 = 12345;
        const combined_seed = @as(u64, seed) +% @as(u64, step);
        const state = combined_seed *% a +% c;
        return @as(f32, @floatFromInt(state & 0x7FFFFFFF)) / @as(f32, @floatFromInt(0x7FFFFFFF));
    }

    /// Get current metrics
    pub fn getMetrics(self: *Self) SpeculationStats {
        return self.metrics.getStats();
    }

    /// Check if generation is complete
    pub fn isComplete(self: *Self) bool {
        return self.is_complete;
    }

    /// Get stop reason (for compatibility with GenerationState)
    pub fn getStopReason(self: *Self) generator.StopReason {
        if (self.is_complete) {
            return .eos;
        }
        return .length;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "calculateAcceptanceProbability exact match" {
    const p_target: f32 = 0.5;
    const p_draft: f32 = 0.5;
    const result = calculateAcceptanceProbability(p_target, p_draft);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result, 0.001);
}

test "calculateAcceptanceProbability target more confident" {
    const p_target: f32 = 0.8;
    const p_draft: f32 = 0.4;
    const result = calculateAcceptanceProbability(p_target, p_draft);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result, 0.001);
}

test "calculateAcceptanceProbability draft more confident" {
    const p_target: f32 = 0.3;
    const p_draft: f32 = 0.6;
    const result = calculateAcceptanceProbability(p_target, p_draft);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), result, 0.001);
}

test "calculateAcceptanceProbability numerical stability" {
    const p_target: f32 = 1e-10;
    const p_draft: f32 = 1e-8;
    const result = calculateAcceptanceProbability(p_target, p_draft);
    try std.testing.expect(result >= 0.0 and result <= 1.0);
}

test "shouldAcceptToken basic" {
    try std.testing.expect(shouldAcceptToken(1.0, 0.5) == true);
    try std.testing.expect(shouldAcceptToken(0.3, 0.8) == false);
    try std.testing.expect(shouldAcceptToken(0.7, 0.5) == true);
}

test "softmax basic" {
    const allocator = std.testing.allocator;
    const logits = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const probs = try softmax(allocator, &logits);
    defer allocator.free(probs);

    // Probabilities should sum to 1
    var sum: f32 = 0;
    for (probs) |p| {
        sum += p;
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 0.001);

    // Higher logits should have higher probabilities
    try std.testing.expect(probs[3] > probs[2]);
    try std.testing.expect(probs[2] > probs[1]);
    try std.testing.expect(probs[1] > probs[0]);
}

test "estimateSpeedup perfect acceptance" {
    const speedup = estimateSpeedup(4, 1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), speedup, 0.1);
}

test "estimateSpeedup zero acceptance" {
    const speedup = estimateSpeedup(4, 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), speedup, 0.1);
}

test "estimateSpeedup typical acceptance" {
    const speedup = estimateSpeedup(4, 0.75);
    // With K=4, 75% acceptance: 4 / (1 + 0.25*4) = 4/2 = 2.0
    try std.testing.expect(speedup >= 1.5 and speedup <= 2.5);
}

test "SpeculationBuffer init and deinit" {
    const allocator = std.testing.allocator;
    var buffer = try SpeculationBuffer.init(allocator, 4);
    defer buffer.deinit(allocator);

    const tokens = [_]u32{ 100, 200, 300, 400 };
    try buffer.setDraftTokens(&tokens);

    try std.testing.expectEqual(@as(usize, 4), buffer.draft_count);
    try std.testing.expectEqual(@as(u32, 100), buffer.draft_tokens[0]);
}

test "SpeculativeMetrics record and getStats" {
    var metrics = SpeculativeMetrics.init();

    metrics.recordSpeculation(3, 1, 50, 150);
    metrics.recordSpeculation(4, 0, 60, 140);

    const stats = metrics.getStats();

    try std.testing.expectEqual(@as(u64, 2), stats.total_speculations);
    try std.testing.expectEqual(@as(u64, 7), stats.total_accepted);
    try std.testing.expectEqual(@as(u64, 1), stats.total_rejected);
}
