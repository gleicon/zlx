//! mod.zig - Inference module public API
//!
//! Provides model loading, tokenization, and generation capabilities.

const std = @import("std");

// Re-export submodules
pub const tokenizer = @import("tokenizer.zig");
pub const loader = @import("loader.zig");
pub const generator = @import("generator.zig");

// Re-export main types for convenience
pub const Tokenizer = tokenizer.Tokenizer;
pub const ModelInfo = loader.ModelInfo;
pub const ModelType = loader.ModelType;
pub const LoadError = loader.LoadError;
pub const GenerationState = generator.GenerationState;
pub const GenerationOptions = generator.GenerationOptions;
pub const Token = generator.Token;

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
        const input_tokens = try tokenizer_ref.encode(prompt);
        defer self.allocator.free(input_tokens);

        // Generate tokens
        const output_tokens = try generator.generateAll(
            self.allocator,
            self.getTransformer(),
            input_tokens,
            self.getEosTokenIds(),
            options,
        );
        defer self.allocator.free(output_tokens);

        // Combine input and output tokens for full context
        var all_tokens = std.ArrayList(u32).init(self.allocator);
        defer all_tokens.deinit();
        try all_tokens.appendSlice(input_tokens);
        try all_tokens.appendSlice(output_tokens);

        // Decode to text
        return tokenizer_ref.decode(all_tokens.items);
    }

    fn getTransformer(self: *Self) !*qwen.Transformer {
        // For now, create transformer on demand
        // In production, cache this
        const qwen_model = try qwen.Transformer.init(self.allocator, self.model_path);
        return qwen_model;
    }

    fn getEosTokenIds(self: *Self) []const u32 {
        _ = self;
        // Qwen EOS token IDs
        return &[_]u32{ 151645, 151643 }; // <|im_end|> and <|endoftext|>
    }
};

// Import qwen for transformer access
const qwen = @import("../mlx.zig/src/qwen.zig");
