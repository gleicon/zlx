//! mod.zig - Inference module public API
//!
//! Provides model loading, tokenization, and generation capabilities.

const std = @import("std");
const mlx_tokenizer = @import("../mlx.zig/src/tokenizer.zig");

// Maximum total context length to prevent memory exhaustion
// Qwen 2.5 1.5B has 28 layers, hidden_size=1536, KV cache grows quickly
// Limit to 8192 tokens total (input + output) to stay within ~8GB GPU memory
const MAX_CONTEXT_LENGTH: usize = 8192;

// Re-export submodules
pub const loader = @import("loader.zig");
pub const generator = @import("generator.zig");

// Re-export main types for convenience
pub const Tokenizer = mlx_tokenizer.Tokenizer;
pub const ModelInfo = loader.ModelInfo;
pub const ModelType = loader.ModelType;
pub const LoadError = loader.LoadError;
pub const GenerationState = generator.GenerationState;
pub const GenerationOptions = generator.GenerationOptions;
pub const Token = generator.Token;
pub const LogprobEntry = generator.LogprobEntry;
pub const StopReason = generator.StopReason;

/// Inference context that holds loaded model and tokenizer
pub const InferenceContext = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    model_path: []const u8,
    tokenizer: ?Tokenizer = null,
    model_type: ModelType,

    // Thread safety
    generation_mutex: std.Thread.Mutex,

    pub fn init(allocator: std.mem.Allocator, model_path: []const u8) !Self {
        var model_info = try loader.loadModelInfo(allocator, model_path);
        defer model_info.deinit();

        var ctx = Self{
            .allocator = allocator,
            .model_path = try allocator.dupe(u8, model_path),
            .tokenizer = null,
            .model_type = model_info.model_type,
            .generation_mutex = std.Thread.Mutex{},
        };

        // Initialize tokenizer
        ctx.tokenizer = try Tokenizer.init(allocator, model_path);

        return ctx;
    }

    pub fn deinit(self: *Self) void {
        if (self.tokenizer) |*tok| {
            tok.deinit();
        }
        self.allocator.free(self.model_path);
    }

    /// Generate text from a prompt with automatic tokenization and decoding
    /// Thread-safe: holds mutex during generation
    pub fn generate(self: *Self, prompt: []const u8, options: GenerationOptions) ![]const u8 {
        self.generation_mutex.lock();
        defer self.generation_mutex.unlock();

        // Tokenize input
        var tokenizer_ref = &self.tokenizer.?;
        var input_tokens = try tokenizer_ref.encode(prompt);
        errdefer self.allocator.free(input_tokens);

        // Calculate max tokens we can generate without exceeding context limit
        const max_new_tokens = @min(options.max_tokens, MAX_CONTEXT_LENGTH - 1);
        const max_input_tokens = MAX_CONTEXT_LENGTH - max_new_tokens;

        // Truncate input if too long (keep from the end - most recent context)
        if (input_tokens.len > max_input_tokens) {
            const start_idx = input_tokens.len - max_input_tokens;
            const truncated = try self.allocator.dupe(u32, input_tokens[start_idx..]);
            self.allocator.free(input_tokens);
            input_tokens = truncated;
            std.log.warn("Input truncated from {d} to {d} tokens to fit context limit", .{ input_tokens.len + max_input_tokens, max_input_tokens });
        }

        // Update options with adjusted max_tokens
        var adjusted_options = options;
        adjusted_options.max_tokens = max_new_tokens;

        // Generate tokens
        const output_tokens = try generator.generateAll(
            self.allocator,
            self.getTransformer(),
            input_tokens,
            self.getEosTokenIds(),
            adjusted_options,
            &self.tokenizer.?, // Pass tokenizer for stop sequence detection
        );
        defer self.allocator.free(output_tokens);
        defer self.allocator.free(input_tokens);

        // Combine input and output tokens for full context
        var all_tokens = std.ArrayList(u32).init(self.allocator);
        defer all_tokens.deinit();
        try all_tokens.appendSlice(input_tokens);
        try all_tokens.appendSlice(output_tokens);

        // Decode to text
        return tokenizer_ref.decode(all_tokens.items);
    }

    /// Generate with timeout protection (per INFRA-04)
    /// Returns partial completion if timeout occurs mid-generation
    pub fn generateWithTimeout(self: *Self, prompt: []const u8, options: GenerationOptions, timeout_ms: u64) !TimeoutResult {
        const start_time = std.time.milliTimestamp();
        var timed_out = false;

        self.generation_mutex.lock();
        defer self.generation_mutex.unlock();

        // Tokenize input
        var tokenizer_ref = &self.tokenizer.?;
        var input_tokens = try tokenizer_ref.encode(prompt);
        errdefer self.allocator.free(input_tokens);

        // Calculate limits
        const max_new_tokens = @min(options.max_tokens, MAX_CONTEXT_LENGTH - 1);
        const max_input_tokens = MAX_CONTEXT_LENGTH - max_new_tokens;

        // Truncate if needed
        if (input_tokens.len > max_input_tokens) {
            const start_idx = input_tokens.len - max_input_tokens;
            const truncated = try self.allocator.dupe(u32, input_tokens[start_idx..]);
            self.allocator.free(input_tokens);
            input_tokens = truncated;
            std.log.warn("Input truncated from {d} to {d} tokens to fit context limit", .{ input_tokens.len + max_input_tokens, max_input_tokens });
        }

        // Initialize transformer
        var transformer = try qwen.Transformer.init(self.allocator, self.model_path);
        defer transformer.deinit();

        // Update options with adjusted max_tokens
        var gen_options = options;
        gen_options.max_tokens = max_new_tokens;

        // Initialize generation state
        var state = try generator.GenerationState.init(
            self.allocator,
            &transformer,
            input_tokens,
            self.getEosTokenIds(),
            gen_options,
            &self.tokenizer.?, // Pass tokenizer for stop sequence detection
            null, // draft_model - not yet integrated
            0, // speculation_depth - disabled for now
        );
        defer state.deinit();

        // Collect tokens with timeout checking
        var output_tokens = std.ArrayList(u32).empty;
        errdefer output_tokens.deinit(self.allocator);

        while (try state.next()) |token| {
            try output_tokens.append(self.allocator, token);

            // Check timeout every token (D-33: measured from request start)
            const elapsed = @as(u64, @intCast(std.time.milliTimestamp() - start_time));
            if (elapsed >= timeout_ms) {
                std.log.warn("Generation timed out after {d}ms, returning partial result", .{elapsed});
                timed_out = true;
                state.setStopReason(.timeout);
                break;
            }
        }

        // Decode output tokens only (not full context like regular generate)
        const text = try tokenizer_ref.decode(output_tokens.items);

        // Get logprobs if enabled
        const logprobs = if (options.logprobs_enabled) state.getLogprobs() else null;

        // Set stop reason based on timeout
        var stop_reason = state.getStopReason();
        if (timed_out) {
            stop_reason = .timeout;
        }

        const result = GenerationResult{
            .text = text,
            .logprobs = logprobs,
            .prompt_tokens = @intCast(input_tokens.len),
            .completion_tokens = @intCast(output_tokens.items.len),
            .stop_reason = stop_reason,
        };

        self.allocator.free(input_tokens);

        return TimeoutResult{
            .result = result,
            .timed_out = timed_out,
        };
    }

    fn getTransformer(self: *Self) !*qwen.Transformer {
        // For now, create transformer on demand
        // In production, cache this
        const qwen_model = try qwen.Transformer.init(self.allocator, self.model_path);
        return qwen_model;
    }

    fn getEosTokenIds(self: *Self) []const u32 {
        _ = self;
        // Qwen EOS token IDs - FIM tokens now added in config loader
        return &[_]u32{ 151645, 151643 }; //   and <|endoftext|>
    }
};

/// Generate result including text and optional logprobs
pub const GenerationResult = struct {
    text: []const u8,
    logprobs: ?[]const LogprobEntry = null,
    prompt_tokens: u32,
    completion_tokens: u32,
    stop_reason: StopReason,
};

/// Result from timeout-aware generation
pub const TimeoutResult = struct {
    result: GenerationResult,
    timed_out: bool,
};

/// Generate text with logprobs tracking (API-02)
/// Caller owns the returned text and logprobs memory
pub fn generateWithLogprobs(
    allocator: std.mem.Allocator,
    tokenizer: *Tokenizer,
    transformer: *qwen.Transformer,
    prompt: []const u8,
    options: GenerationOptions,
) !GenerationResult {
    // Tokenize input
    var input_tokens = try tokenizer.encode(prompt);
    errdefer allocator.free(input_tokens);

    // Calculate max tokens we can generate without exceeding context limit
    const max_new_tokens = @min(options.max_tokens, MAX_CONTEXT_LENGTH - 1);
    const max_input_tokens = MAX_CONTEXT_LENGTH - max_new_tokens;

    // Truncate input if too long (keep from the end - most recent context)
    if (input_tokens.len > max_input_tokens) {
        const start_idx = input_tokens.len - max_input_tokens;
        const truncated = try allocator.dupe(u32, input_tokens[start_idx..]);
        allocator.free(input_tokens);
        input_tokens = truncated;
        std.log.warn("Input truncated from {d} to {d} tokens to fit context limit", .{ input_tokens.len + max_input_tokens, max_input_tokens });
    }

    // Update options with adjusted max_tokens
    var adjusted_options = options;
    adjusted_options.max_tokens = max_new_tokens;

    // Get EOS token IDs
    const eos_token_ids = &[_]u32{ 151645, 151643 };

    // Initialize generation state
    var state = try generator.GenerationState.init(
        allocator,
        transformer,
        input_tokens,
        eos_token_ids,
        adjusted_options,
        tokenizer, // Pass tokenizer for stop sequence detection
        null, // draft_model
        0, // speculation_depth
    );
    defer state.deinit();

    // Collect tokens
    var output_tokens = std.ArrayList(u32).init(allocator);
    defer output_tokens.deinit();

    var completion_tokens: u32 = 0;
    while (try state.next()) |token| {
        try output_tokens.append(token);
        completion_tokens += 1;
    }

    // Decode text
    const text = try tokenizer.decode(output_tokens.items);

    // Get logprobs if enabled
    const logprobs_entries = if (options.logprobs_enabled)
        try allocator.dupe(LogprobEntry, state.getLogprobs())
    else
        null;
    // Note: logprobs_entries entries have their own allocated strings that need freeing

    return GenerationResult{
        .text = text,
        .logprobs = logprobs_entries,
        .prompt_tokens = @intCast(input_tokens.len),
        .completion_tokens = completion_tokens,
        .stop_reason = state.getStopReason(),
    };
}

// Import qwen for transformer access
const qwen = @import("../mlx.zig/src/qwen.zig");
