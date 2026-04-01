//! generator.zig - Token generation with iterator pattern
//!
//! Provides GenerationState that yields one token at a time via next().

const std = @import("std");
const mlx = @import("../mlx.zig/src/mlx.zig");
const qwen = @import("../mlx.zig/src/qwen.zig");

/// Token type for generation
pub const Token = u32;

/// Generation options
pub const GenerationOptions = struct {
    max_tokens: usize = 256,
    temperature: f32 = 0.7,
    top_p: f32 = 0.9,
    stop_on_eos: bool = true,
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

    // MLX arrays (managed)
    toks_array: mlx.Array,
    logits_array: mlx.Array,
    mask_array: mlx.Array,

    // Generation parameters
    options: GenerationOptions,

    /// Initialize generation state with a transformer and initial tokens
    pub fn init(
        allocator: std.mem.Allocator,
        transformer: *qwen.Transformer,
        initial_tokens: []const u32,
        eos_token_ids: []const u32,
        options: GenerationOptions,
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

        self.current_tokens.deinit(self.allocator);
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

        // Apply temperature scaling if temperature != 0
        var scaled_logits = mlx.arrayNew();
        defer mlx.arrayFree(scaled_logits);

        if (self.options.temperature > 0 and self.options.temperature != 1.0) {
            // Divide logits by temperature: logits / temperature
            const temp_scalar = mlx.float(self.options.temperature);
            try mlx.divide(&scaled_logits, last_logits, temp_scalar, transformer.mlx_config.stream);
        } else {
            // Copy logits
            try mlx.arraySet(&scaled_logits, last_logits);
        }

        // Apply softmax to get probabilities
        var probs = mlx.arrayNew();
        defer mlx.arrayFree(probs);
        const axes = &[_]c_int{1}; // Softmax over vocab dimension
        try mlx.softmax(&probs, scaled_logits, axes, false, transformer.mlx_config.stream);

        // Sample from the distribution
        var next_token: Token = 0;

        // Evaluate to get actual values for sampling
        try mlx.arrayEval(probs);

        // Get probability data
        const probs_data: [*c]f32 = @ptrCast(@constCast(mlx.C.mlx_array_data_float32(probs)));
        const vocab_size = mlx.arrayDim(probs, 1);

        if (self.options.temperature == 0) {
            // Greedy: pick the highest probability token
            var max_prob: f32 = 0;
            var max_idx: u32 = 0;
            for (0..@intCast(vocab_size)) |i| {
                const p = probs_data[i];
                if (p > max_prob) {
                    max_prob = p;
                    max_idx = @intCast(i);
                }
            }
            next_token = max_idx;
        } else {
            // Sample from the distribution
            const random_value = std.crypto.random.float(f32);
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
                    break;
                }
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
};

/// Convenience function to generate all tokens at once (for non-streaming use)
pub fn generateAll(
    allocator: std.mem.Allocator,
    transformer: *qwen.Transformer,
    initial_tokens: []const u32,
    eos_token_ids: []const u32,
    options: GenerationOptions,
) ![]const u32 {
    var state = try GenerationState.init(allocator, transformer, initial_tokens, eos_token_ids, options);
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

    var state = try GenerationState.init(allocator, &transformer, &initial_tokens, transformer.eos_token_ids, options);
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
