//! generator.zig - Token generation with iterator pattern
//!
//! Provides GenerationState that yields one token at a time via next().

const std = @import("std");
const mlx = @import("../mlx.zig/src/mlx.zig");
const qwen = @import("../mlx.zig/src/qwen.zig");
const mlx_tokenizer = @import("../mlx.zig/src/tokenizer.zig");

/// Token type for generation
pub const Token = u32;

/// PCG32 deterministic random number generator
/// Same seed produces same sequence — required for OpenAI API compatibility (API-05)
pub const Pcg32Rng = struct {
    state: u64,
    inc: u64,

    const MULTIPLIER: u64 = 6364136223846793005;
    const DEFAULT_INC: u64 = 1442695040888963407;

    pub fn init(seed: u32) Pcg32Rng {
        var rng = Pcg32Rng{
            .state = 0,
            .inc = (DEFAULT_INC << 1) | 1,
        };
        _ = rng.next(); // Initialize state
        rng.state = rng.state +% seed;
        _ = rng.next();
        return rng;
    }

    fn next(self: *Pcg32Rng) u32 {
        const old_state = self.state;
        self.state = old_state *% MULTIPLIER +% self.inc;
        const xor_shifted: u32 = @intCast(((old_state >> 18) ^ old_state) >> 27);
        const rot: u32 = @intCast(old_state >> 59);
        const rot_u5: u5 = @intCast(rot & 31);
        return (xor_shifted >> rot_u5) | (xor_shifted << ((~rot_u5 +% 1) & 31));
    }

    /// Return random float in range [0, 1)
    pub fn random(self: *Pcg32Rng) f32 {
        return @as(f32, @floatFromInt(self.next())) / @as(f32, @floatFromInt(@as(u32, @intCast(0xFFFFFFFF))));
    }
};

/// Single token logprob entry for top alternatives
pub const TopLogprob = struct {
    token: u32,
    token_str: []const u8,
    logprob: f32,
};

/// Logprob data for a single position
pub const LogprobEntry = struct {
    token: u32,
    token_str: []const u8,
    logprob: f32,
    top_logprobs: []const TopLogprob,

    pub fn deinit(self: *LogprobEntry, allocator: std.mem.Allocator) void {
        allocator.free(self.token_str);
        for (self.top_logprobs) |tl| {
            allocator.free(tl.token_str);
        }
        allocator.free(self.top_logprobs);
    }
};

/// Stop reason for generation
pub const StopReason = enum {
    eos,
    length,
    stop,
    timeout,
};

/// Generation options
pub const GenerationOptions = struct {
    max_tokens: usize = 256,
    temperature: f32 = 0.7,
    top_p: f32 = 0.9,
    stop_on_eos: bool = true,
    /// Optional seed for deterministic sampling (API-05)
    seed: ?u32 = null,
    /// Stop sequences to halt generation (API-01)
    stop_sequences: []const []const u8 = &[_][]const u8{},
    /// Repetition penalty, 1.0 = disabled (API-03)
    repetition_penalty: f32 = 1.0,
    /// Presence penalty, -2.0 to 2.0 (API-03)
    presence_penalty: f32 = 0.0,
    /// Frequency penalty, -2.0 to 2.0 (API-03)
    frequency_penalty: f32 = 0.0,
    /// Top-k filtering, 0 = disabled (API-03)
    top_k: u32 = 0,
    /// Minimum probability for nucleus sampling (API-03)
    min_p: f32 = 0.0,
    /// Enable logprobs tracking (API-02)
    logprobs_enabled: bool = false,
    /// Logit bias mapping (token_id -> bias value)
    logit_bias: std.AutoHashMap(u32, f32) = undefined,
};

/// State machine for token generation
pub const GenerationState = struct {
    const Self = @This();

    // Core MLX components
    allocator: std.mem.Allocator,
    transformer: ?*qwen.Transformer = null,
    cache: ?*mlx.Cache = null,

    // Generation state
    tokens_generated: usize = 0,
    max_tokens: usize,
    current_tokens: std.ArrayList(u32),
    is_complete: bool = false,
    eos_token_ids: []const u32,
    stop_reason: StopReason = .eos,

    // MLX arrays (managed)
    toks_array: mlx.Array,
    logits_array: mlx.Array,
    mask_array: mlx.Array,

    // Generation parameters
    options: GenerationOptions,

    // PCG32 RNG for deterministic sampling (API-05)
    rng: ?Pcg32Rng = null,

    // Logprobs tracking (API-02)
    logprobs_buffer: std.ArrayList(LogprobEntry),

    // Tokenizer reference for decoding tokens to strings
    tokenizer: ?*mlx_tokenizer.Tokenizer = null,

    // Stop sequence detection (API-01)
    /// Buffer to accumulate decoded text for stop sequence checking
    decoded_text_buffer: std.ArrayList(u8),
    /// Tokens since last decode (to batch decode operations)
    tokens_since_decode: std.ArrayList(u32),
    /// Final decoded text without stop sequence (when stopped by stop sequence)
    final_decoded_text: ?[]const u8 = null,

    /// Initialize generation state with a transformer and initial tokens
    pub fn init(
        allocator: std.mem.Allocator,
        transformer: *qwen.Transformer,
        initial_tokens: []const u32,
        eos_token_ids: []const u32,
        options: GenerationOptions,
        tokenizer: ?*mlx_tokenizer.Tokenizer,
    ) !Self {
        // Initialize MLX arrays
        const toks_array = blk: {
            // Create initial tokens array [1, seq_len]
            const toks_data = try allocator.dupe(u32, initial_tokens);
            defer allocator.free(toks_data);

            break :blk try mlx.arrayNewData(toks_data.ptr, .{ 1, @as(c_int, @intCast(initial_tokens.len)) }, mlx.UINT32);
        };

        // Allocate KV cache on heap so we can pass a stable pointer
        const cache = try allocator.create(mlx.Cache);
        errdefer allocator.destroy(cache);
        cache.* = try mlx.Cache.init(allocator, transformer.model.layers.len, 2);

        return Self{
            .allocator = allocator,
            .transformer = transformer,
            .cache = cache,
            .tokens_generated = 0,
            .max_tokens = options.max_tokens,
            .current_tokens = .empty,
            .is_complete = false,
            .eos_token_ids = eos_token_ids,
            .toks_array = toks_array,
            .logits_array = mlx.arrayNew(),
            .mask_array = mlx.arrayNew(),
            .options = options,
            .logprobs_buffer = .empty,
            .rng = if (options.seed) |seed| Pcg32Rng.init(seed) else null,
            .decoded_text_buffer = .empty,
            .tokens_since_decode = .empty,
            .tokenizer = tokenizer,
        };
    }

    /// Deinitialize generation state and free resources
    pub fn deinit(self: *Self) void {
        if (self.cache) |cache| {
            cache.deinit();
            self.allocator.destroy(cache);
        }

        mlx.arrayFree(self.toks_array);
        mlx.arrayFree(self.logits_array);
        mlx.arrayFree(self.mask_array);

        // Free logprobs buffer
        for (self.logprobs_buffer.items) |*entry| {
            entry.deinit(self.allocator);
        }
        self.logprobs_buffer.deinit(self.allocator);

        // Free stop sequence detection buffers
        self.decoded_text_buffer.deinit(self.allocator);
        self.tokens_since_decode.deinit(self.allocator);
        if (self.final_decoded_text) |text| {
            self.allocator.free(text);
        }

        self.current_tokens.deinit(self.allocator);
    }

    /// Decode tokens to text using the tokenizer
    fn decodeTokens(self: *Self, tokens: []const u32) ![]const u8 {
        if (self.tokenizer) |tok| {
            return try tok.decode(tokens);
        }
        return &[_]u8{}; // Return empty if no tokenizer
    }

    /// Generate the next token. Returns null when generation is complete.
    pub fn next(self: *Self) !?Token {
        if (self.is_complete or self.tokens_generated >= self.max_tokens) {
            return null;
        }

        const transformer = self.transformer.?;
        const cache = self.cache.?;

        // Create causal mask
        const seq_len = mlx.arrayDim(self.toks_array, 1);
        try mlx.createCausalMask(&self.mask_array, seq_len, cache.offset, transformer.mlx_config.dtype, transformer.mlx_config.stream);

        // Forward pass through model
        try transformer.model.forward(&self.logits_array, self.toks_array, self.mask_array, cache);

        // Take logits for last position: logits[:, -1, :]
        var last_logits = mlx.arrayNew();
        defer mlx.arrayFree(last_logits);
        try mlx.take(&last_logits, self.logits_array, mlx.int(-1), 1, transformer.mlx_config.stream);

        // Sample from the distribution
        var next_token: Token = 0;

        if (self.options.temperature == 0) {
            // D-18: Greedy selection — argmax on raw logits (before temperature/softmax)
            var next_token_arr = mlx.arrayNew();
            defer mlx.arrayFree(next_token_arr);
            try mlx.argmax(&next_token_arr, last_logits, 1, false, transformer.mlx_config.stream);
            try mlx.item(&next_token, next_token_arr);
        } else {
            // D-13, D-16, D-17: Complete sampling pipeline
            // Pipeline: logits → logit_bias → penalties → top_k → min_p → temperature → softmax → sample

            // Extract logits data from MLX array for CPU-side modifications
            const vocab_size_c = mlx.arrayDim(last_logits, 1);
            const vocab_size: usize = @intCast(vocab_size_c);

            // Get raw logits data pointer
            try mlx.arrayEval(last_logits);
            const logits_ptr: [*c]f32 = @ptrCast(@constCast(mlx.C.mlx_array_data_float32(last_logits)));

            // Check if any sampling parameters are active
            const has_logit_bias = self.options.logit_bias.count() > 0;
            const has_penalties = self.options.presence_penalty != 0.0 or
                self.options.frequency_penalty != 0.0 or
                self.options.repetition_penalty != 1.0;
            const has_top_k = self.options.top_k > 0 and self.options.top_k < vocab_size;
            const has_min_p = self.options.min_p > 0.0 and self.options.min_p <= 1.0;
            const needs_modifications = has_logit_bias or has_penalties or has_top_k or has_min_p;

            var modified_logits: mlx.Array = undefined;
            var modified_logits_owned = false;

            if (needs_modifications) {
                // Create mutable copy of logits data for modifications
                const logits_copy = try self.allocator.alloc(f32, vocab_size);
                defer self.allocator.free(logits_copy);

                // Copy logits data
                for (0..@intCast(vocab_size)) |i| {
                    logits_copy[i] = logits_ptr[i];
                }

                // D-17: Apply logit_bias first (priority)
                if (has_logit_bias) {
                    std.log.debug("Applying logit_bias to {d} tokens", .{self.options.logit_bias.count()});
                    var bias_iter = self.options.logit_bias.iterator();
                    while (bias_iter.next()) |entry| {
                        const token_id = entry.key_ptr.*;
                        const bias = entry.value_ptr.*;
                        if (token_id < vocab_size) {
                            logits_copy[token_id] += bias;
                        }
                    }
                }

                // D-16: Apply penalties
                if (has_penalties) {
                    std.log.debug("Applying penalties: presence={d}, frequency={d}, repetition={d}", .{
                        self.options.presence_penalty,
                        self.options.frequency_penalty,
                        self.options.repetition_penalty,
                    });
                    // Apply penalties directly on the slice
                    if (self.current_tokens.items.len > 0) {
                        // Count token frequencies in current sequence
                        var freq_map = std.AutoHashMap(u32, u32).init(self.allocator);
                        defer freq_map.deinit();

                        for (self.current_tokens.items) |token| {
                            const count = freq_map.get(token) orelse 0;
                            try freq_map.put(token, count + 1);
                        }

                        // Apply penalties
                        var iter = freq_map.iterator();
                        while (iter.next()) |entry| {
                            const token_id = entry.key_ptr.*;
                            const count = entry.value_ptr.*;

                            if (token_id >= vocab_size) continue;

                            // Presence penalty: applied once if token appears at all
                            if (self.options.presence_penalty != 0.0 and count > 0) {
                                logits_copy[token_id] -= self.options.presence_penalty;
                            }

                            // Frequency penalty: applied proportional to count
                            if (self.options.frequency_penalty != 0.0) {
                                logits_copy[token_id] -= self.options.frequency_penalty * @as(f32, @floatFromInt(count));
                            }

                            // Repetition penalty: multiplicative on logits
                            if (self.options.repetition_penalty > 1.0 and count > 0) {
                                if (logits_copy[token_id] > 0) {
                                    logits_copy[token_id] /= self.options.repetition_penalty;
                                } else {
                                    logits_copy[token_id] *= self.options.repetition_penalty;
                                }
                            }
                        }
                    }
                }

                // D-13: Apply top_k filtering
                if (has_top_k) {
                    std.log.debug("Applying top_k={d} filtering", .{self.options.top_k});
                    const k = self.options.top_k;

                    // Find k-th largest logit using selection algorithm
                    const sorted_logits = try self.allocator.dupe(f32, logits_copy);
                    defer self.allocator.free(sorted_logits);

                    std.mem.sort(f32, sorted_logits, {}, std.sort.desc(f32));
                    const kth_logit = sorted_logits[k - 1];

                    // Set all logits below kth to -infinity
                    for (0..vocab_size) |i| {
                        if (logits_copy[i] < kth_logit) {
                            logits_copy[i] = -std.math.inf(f32);
                        }
                    }
                }

                // D-13: Apply min_p filtering
                if (has_min_p) {
                    std.log.debug("Applying min_p={d} filtering", .{self.options.min_p});

                    // Find max logit
                    var max_logit: f32 = -std.math.inf(f32);
                    for (0..vocab_size) |i| {
                        if (logits_copy[i] > max_logit) {
                            max_logit = logits_copy[i];
                        }
                    }

                    // Compute min logit threshold
                    // min_p threshold in probability space: p >= min_p * p_max
                    // In log space: logit >= max_logit + ln(min_p)
                    const min_logit_threshold = max_logit + @log(self.options.min_p);

                    // Filter tokens below threshold
                    for (0..vocab_size) |i| {
                        if (logits_copy[i] < min_logit_threshold) {
                            logits_copy[i] = -std.math.inf(f32);
                        }
                    }
                }

                // Create new MLX array from modified logits
                modified_logits = try mlx.arrayNewData(logits_copy.ptr, .{ 1, @as(c_int, @intCast(vocab_size)) }, mlx.FLOAT32);
                modified_logits_owned = true;
            } else {
                // No modifications needed, use original logits
                try mlx.arraySet(&modified_logits, last_logits);
            }
            defer if (modified_logits_owned) mlx.arrayFree(modified_logits);

            // Apply temperature scaling to modified logits
            var scaled_logits = mlx.arrayNew();
            defer mlx.arrayFree(scaled_logits);

            if (self.options.temperature != 1.0) {
                const temp_scalar = mlx.float(self.options.temperature);
                try mlx.divide(&scaled_logits, modified_logits, temp_scalar, transformer.mlx_config.stream);
            } else {
                try mlx.arraySet(&scaled_logits, modified_logits);
            }

            // Apply softmax to get probabilities
            var probs = mlx.arrayNew();
            defer mlx.arrayFree(probs);
            const axes = &[_]c_int{1}; // Softmax over vocab dimension
            try mlx.softmax(&probs, scaled_logits, axes, false, transformer.mlx_config.stream);

            // Evaluate to get actual values for sampling
            try mlx.arrayEval(probs);

            // Get probability data
            const probs_data: [*c]f32 = @ptrCast(@constCast(mlx.C.mlx_array_data_float32(probs)));

            // Sample from the distribution using seeded RNG if available (API-05)
            const random_value = if (self.rng) |*rng| rng.random() else std.crypto.random.float(f32);
            var cumsum: f32 = 0;
            var last_idx: u32 = 0;
            for (0..@intCast(vocab_size)) |i| {
                cumsum += probs_data[i];
                last_idx = @intCast(i);
                if (random_value <= cumsum) {
                    next_token = @intCast(i);
                    break;
                }
            }
            // Fallback: if we didn't find a token (floating point edge case), use the last index
            if (next_token == 0 and random_value > cumsum) {
                next_token = last_idx;
            }
        }

        // Debug: Log token generation
        // std.log.debug("Generated token {d} at position {d}", .{ next_token, self.tokens_generated });

        // OLD: Argmax to get next token
        // var next_token_arr = mlx.arrayNew();
        // defer mlx.arrayFree(next_token_arr);
        // try mlx.argmax(&next_token_arr, last_logits, 1, false, transformer.mlx_config.stream);
        //
        // // Extract token value
        // var next_token: Token = 0;
        // try mlx.item(&next_token, next_token_arr);

        // Update state
        self.tokens_generated += 1;
        try self.current_tokens.append(self.allocator, next_token);

        // Check for EOS
        if (self.options.stop_on_eos) {
            for (self.eos_token_ids) |eos_id| {
                if (next_token == eos_id) {
                    self.is_complete = true;
                    self.stop_reason = .eos;
                    break;
                }
            }
        }

        // Stop sequence detection (API-01)
        // Accumulate tokens for potential decode+check
        try self.tokens_since_decode.append(self.allocator, next_token);

        // Decode and check stop sequences if any are configured
        if (self.options.stop_sequences.len > 0 and self.tokenizer != null) {
            // Decode accumulated tokens to text
            const decoded_chunk = try self.decodeTokens(self.tokens_since_decode.items);
            defer self.allocator.free(decoded_chunk);

            // Append to decoded text buffer
            try self.decoded_text_buffer.appendSlice(self.allocator, decoded_chunk);

            // Clear tokens since they've been decoded
            self.tokens_since_decode.clearRetainingCapacity();

            // Check if decoded text ends with any stop sequence
            if (self.checkStopSequence()) {
                // Match found — halt generation
                self.is_complete = true;
                self.stop_reason = .stop;
                self.truncateStopSequence();

                // Log for debugging
                std.log.debug("Stop sequence matched, halting generation", .{});

                return null;
            }
        }

        // Prepare tokens array for next iteration: [1, 1] with just the new token
        mlx.arrayFree(self.toks_array);
        const single_token = [_]u32{next_token};
        self.toks_array = try mlx.arrayNewData(&single_token, .{ 1, 1 }, mlx.UINT32);

        return next_token;
    }

    /// Get all tokens generated so far
    pub fn getGeneratedTokens(self: *Self) []const u32 {
        return self.current_tokens.items;
    }

    /// Check if generation is complete
    pub fn isComplete(self: *Self) bool {
        return self.is_complete;
    }

    /// Get the stop reason
    pub fn getStopReason(self: *Self) StopReason {
        return self.stop_reason;
    }

    /// Set the stop reason
    pub fn setStopReason(self: *Self, reason: StopReason) void {
        self.stop_reason = reason;
    }

    /// Get captured logprobs (returns a copy of the buffer, caller owns memory of returned slice but not entries)
    pub fn getLogprobs(self: *Self) []const LogprobEntry {
        return self.logprobs_buffer.items;
    }

    /// Check if generated text ends with any stop sequence (API-01)
    /// Returns: true if generation should stop, false otherwise
    /// Per D-01, D-03: Check after each token, match multi-character strings
    fn checkStopSequence(self: *Self) bool {
        if (self.options.stop_sequences.len == 0) return false;

        const text = self.decoded_text_buffer.items;

        for (self.options.stop_sequences) |stop_seq| {
            if (stop_seq.len == 0) continue;

            // Per D-03: Check if text ends with stop sequence
            if (text.len >= stop_seq.len) {
                const end_slice = text[text.len - stop_seq.len ..];
                if (std.mem.eql(u8, end_slice, stop_seq)) {
                    return true;
                }
            }
        }
        return false;
    }

    /// Truncate decoded text to remove matched stop sequence (API-01, D-04)
    /// Call this after checkStopSequence returns true
    fn truncateStopSequence(self: *Self) void {
        if (self.options.stop_sequences.len == 0) return;

        const text = self.decoded_text_buffer.items;

        for (self.options.stop_sequences) |stop_seq| {
            if (stop_seq.len == 0) continue;

            if (text.len >= stop_seq.len) {
                const end_slice = text[text.len - stop_seq.len ..];
                if (std.mem.eql(u8, end_slice, stop_seq)) {
                    // Remove the stop sequence from the buffer
                    self.decoded_text_buffer.shrinkAndFree(self.allocator, text.len - stop_seq.len);
                    return;
                }
            }
        }
    }

    /// Get the final decoded text without stop sequence (API-01, D-04)
    pub fn getFinalDecodedText(self: *Self, allocator: std.mem.Allocator) ![]const u8 {
        // If we stopped by stop sequence, return the truncated buffer
        // Otherwise, return a copy of the full buffer
        return try allocator.dupe(u8, self.decoded_text_buffer.items);
    }

    /// Capture top-5 logprobs from raw logits (D-07, D-12)
    fn captureLogprobs(self: *Self, logits_data: [*c]f32, vocab_size: usize) !LogprobEntry {
        // Create array of (token_id, logit) pairs
        const TokenLogitPair = struct { u32, f32 };
        var token_logits = try self.allocator.alloc(TokenLogitPair, vocab_size);
        defer self.allocator.free(token_logits);

        for (0..vocab_size) |i| {
            token_logits[i] = .{ @intCast(i), logits_data[i] };
        }

        // Sort by logit descending (highest first)
        std.mem.sort(TokenLogitPair, token_logits, {}, struct {
            fn lessThan(_: void, a: TokenLogitPair, b: TokenLogitPair) bool {
                return a[1] > b[1]; // Descending order
            }
        }.lessThan);

        // Take top 5
        const top_k = @min(5, vocab_size);
        var top_logprobs = try self.allocator.alloc(TopLogprob, top_k);
        errdefer self.allocator.free(top_logprobs);

        // Compute log_softmax for numerical stability
        const max_logit: f32 = token_logits[0][1];
        var sum_exp: f32 = 0;
        for (token_logits) |tl| {
            sum_exp += std.math.exp(tl[1] - max_logit);
        }
        const log_sum_exp = max_logit + @log(sum_exp);

        for (0..top_k) |i| {
            const token_id = token_logits[i][0];
            const logit = token_logits[i][1];
            const logprob = logit - log_sum_exp; // log_softmax

            // For now, store empty token string (will be decoded later with the tokenizer)
            top_logprobs[i] = .{
                .token = token_id,
                .token_str = &[_]u8{},
                .logprob = logprob,
            };
        }

        return LogprobEntry{
            .token = 0, // Will be filled in after sampling
            .token_str = &[_]u8{},
            .logprob = 0, // Will be filled in after sampling
            .top_logprobs = top_logprobs,
        };
    }

    /// Apply top_k filtering — keep only k highest logits (D-13)
    fn applyTopK(self: *Self, logits: *std.ArrayList(f32), vocab_size: usize) !void {
        if (self.options.top_k == 0 or self.options.top_k >= vocab_size) return;

        const k = self.options.top_k;

        // Find k-th largest logit using selection algorithm
        const sorted_logits = try self.allocator.dupe(f32, logits.items[0..vocab_size]);
        defer self.allocator.free(sorted_logits);

        std.mem.sort(f32, sorted_logits, {}, std.sort.desc(f32));
        const kth_logit = sorted_logits[k - 1];

        // Set all logits below kth to -infinity
        for (0..vocab_size) |i| {
            if (logits.items[i] < kth_logit) {
                logits.items[i] = -std.math.inf(f32);
            }
        }
    }

    /// Apply min_p filtering — tokens must have prob >= min_p * max_prob (D-13)
    fn applyMinP(self: *Self, logits: *std.ArrayList(f32), vocab_size: usize) !void {
        if (self.options.min_p <= 0.0 or self.options.min_p > 1.0) return;

        // Find max logit
        var max_logit: f32 = -std.math.inf(f32);
        for (0..vocab_size) |i| {
            if (logits.items[i] > max_logit) {
                max_logit = logits.items[i];
            }
        }

        // Compute min logit threshold
        // min_p threshold in probability space: p >= min_p * p_max
        // In log space: logit >= max_logit + log(min_p)
        const min_logit_threshold = max_logit + std.math.log(self.options.min_p);

        // Filter tokens below threshold
        for (0..vocab_size) |i| {
            if (logits.items[i] < min_logit_threshold) {
                logits.items[i] = -std.math.inf(f32);
            }
        }
    }

    /// Apply presence, frequency, and repetition penalties (D-16)
    fn applyPenalties(self: *Self, logits: *std.ArrayList(f32), vocab_size: usize) !void {
        if (self.current_tokens.items.len == 0) return;

        // Count token frequencies in current sequence
        var freq_map = std.AutoHashMap(u32, u32).init(self.allocator);
        defer freq_map.deinit();

        for (self.current_tokens.items) |token| {
            const count = freq_map.get(token) orelse 0;
            try freq_map.put(token, count + 1);
        }

        // Apply penalties
        var iter = freq_map.iterator();
        while (iter.next()) |entry| {
            const token_id = entry.key_ptr.*;
            const count = entry.value_ptr.*;

            if (token_id >= vocab_size) continue;

            const logit = &logits.items[token_id];

            // Presence penalty: applied once if token appears at all
            if (self.options.presence_penalty != 0.0 and count > 0) {
                logit.* -= self.options.presence_penalty;
            }

            // Frequency penalty: applied proportional to count
            if (self.options.frequency_penalty != 0.0) {
                logit.* -= self.options.frequency_penalty * @as(f32, @floatFromInt(count));
            }

            // Repetition penalty: multiplicative on logits (or additive on logprobs)
            // Standard approach: divide logits by repetition_penalty for seen tokens
            if (self.options.repetition_penalty > 1.0 and count > 0) {
                if (logit.* > 0) {
                    logit.* /= self.options.repetition_penalty;
                } else {
                    logit.* *= self.options.repetition_penalty;
                }
            }
        }
    }
};

/// Convenience function to generate all tokens at once (for non-streaming use)
pub fn generateAll(
    allocator: std.mem.Allocator,
    transformer: *qwen.Transformer,
    initial_tokens: []const u32,
    eos_token_ids: []const u32,
    options: GenerationOptions,
    tokenizer: ?*mlx_tokenizer.Tokenizer,
) ![]const u32 {
    var state = try GenerationState.init(allocator, transformer, initial_tokens, eos_token_ids, options, tokenizer);
    defer state.deinit();

    var result = std.ArrayList(u32).init(allocator);
    errdefer result.deinit();

    while (try state.next()) |token| {
        try result.append(token);
    }

    return result.toOwnedSlice();
}

test "GenerationState basic test" {
    // This test requires a loaded model - skip if not available
    const model_path = "./models/Qwen2.5-Coder-1.5B-4bit";
    std.fs.cwd().access(model_path, .{}) catch {
        std.debug.print("Skipping test - model not found at {s}\n", .{model_path});
        return;
    };

    const allocator = std.testing.allocator;

    // Initialize transformer (this loads the model)
    var transformer = try qwen.Transformer.init(allocator, "Qwen2.5-Coder-1.5B-4bit");
    defer transformer.deinit();

    // Create initial tokens
    const initial_tokens = [_]u32{ 151659, 750, 3974 }; // Example tokens

    const options = GenerationOptions{
        .max_tokens = 5,
        .stop_on_eos = true,
    };

    var state = try GenerationState.init(allocator, &transformer, &initial_tokens, transformer.eos_token_ids, options, null);
    defer state.deinit();

    // Generate a few tokens
    var count: usize = 0;
    while (try state.next()) |token| {
        std.debug.print("Token {d}: {d}\n", .{ count, token });
        count += 1;
        if (count >= 3) break;
    }

    try std.testing.expect(count > 0);
}
