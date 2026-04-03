// Phase 14-05: Backend-aware generator

const std = @import("std");
const backends = @import("../backends/mod.zig");
const factory = @import("../backends/factory.zig");
const registry = @import("../models/registry.zig");
const mlx_tokenizer = @import("../mlx.zig/src/tokenizer.zig");
const generator = @import("generator.zig");

const GenerationOptions = generator.GenerationOptions;

/// Backend-aware generator for unified inference
/// New implementation that uses backends.Backend interface
pub const BackendGenerator = struct {
    allocator: std.mem.Allocator,
    backend: backends.Backend,
    tokenizer: mlx_tokenizer.Tokenizer,

    /// Generation statistics
    pub const Stats = struct {
        tokens_generated: u64,
        tokens_per_second: f32,
        start_time_us: i64,
    };

    /// Initialize generator with automatic backend selection
    pub fn initWithAutoBackend(
        allocator: std.mem.Allocator,
        model_path: []const u8,
        arch: registry.ModelArchitecture,
    ) !BackendGenerator {
        // Create backend via factory
        const backend = try factory.createBackend(
            allocator,
            model_path,
            arch,
            .auto, // Auto-select based on architecture
        );
        errdefer backend.deinit(allocator);

        // Load tokenizer (model-agnostic for now)
        // TODO: Make tokenizer backend-specific
        const tokenizer = try mlx_tokenizer.Tokenizer.init(allocator, model_path);
        errdefer tokenizer.deinit();

        std.log.info("BackendGenerator initialized with {s} backend for {s}", .{
            @tagName(backend.getType()),
            @tagName(arch),
        });

        return BackendGenerator{
            .allocator = allocator,
            .backend = backend,
            .tokenizer = tokenizer,
        };
    }

    /// Initialize with explicit backend preference
    pub fn initWithBackend(
        allocator: std.mem.Allocator,
        model_path: []const u8,
        arch: registry.ModelArchitecture,
        preference: factory.BackendPreference,
    ) !BackendGenerator {
        const backend = try factory.createBackend(
            allocator,
            model_path,
            arch,
            preference,
        );
        errdefer backend.deinit(allocator);

        const tokenizer = try mlx_tokenizer.Tokenizer.init(allocator, model_path);
        errdefer tokenizer.deinit();

        std.log.info("BackendGenerator initialized with {s} backend (requested: {s})", .{
            @tagName(backend.getType()),
            @tagName(preference),
        });

        return BackendGenerator{
            .allocator = allocator,
            .backend = backend,
            .tokenizer = tokenizer,
        };
    }

    /// Free generator resources
    pub fn deinit(self: *BackendGenerator) void {
        self.backend.deinit(self.allocator);
        self.tokenizer.deinit();
    }

    /// Generate text from prompt
    pub fn generate(
        self: *BackendGenerator,
        prompt: []const u8,
        options: GenerationOptions,
    ) ![]const u8 {
        // 1. Tokenize prompt
        const prompt_tokens = try self.backend.tokenize(prompt, self.allocator);
        defer self.allocator.free(prompt_tokens);

        // 2. Set up generation parameters
        const params = backends.GenerationParams{
            .temperature = options.temperature,
            .top_p = options.top_p,
            .top_k = options.top_k,
            .max_tokens = @intCast(options.max_tokens),
            .seed = options.seed orelse 0,
            .stop_sequences = if (options.stop_sequences.len > 0) options.stop_sequences else null,
            .presence_penalty = options.presence_penalty,
            .frequency_penalty = options.frequency_penalty,
            .repetition_penalty = options.repetition_penalty,
            .min_p = options.min_p,
        };

        // 3. Generate tokens
        const result = try self.backend.generate(prompt_tokens, params, self.allocator);
        defer result.deinit(self.allocator);

        // 4. Collect all tokens
        var all_tokens = std.ArrayList(u32).init(self.allocator);
        defer all_tokens.deinit();

        var token_result = try result.token_iterator.next(self.allocator);
        while (token_result != null) {
            const tr = token_result.?;
            try all_tokens.append(tr.token);

            // Check for stop sequences
            if (options.stop_sequences.len > 0) {
                // TODO: Check if generated text contains stop sequence
            }

            // Check finish reason
            if (tr.finish_reason != null) {
                break;
            }

            token_result = try result.token_iterator.next(self.allocator);
        }

        // 5. Decode tokens to text
        // TODO: Implement token-to-text decoding
        // For now, return placeholder
        return try std.fmt.allocPrint(self.allocator, "Generated {d} tokens", .{all_tokens.items.len});
    }

    /// Apply TurboQuant compression
    pub fn applyTurboQuant(
        self: *BackendGenerator,
        bits: u4,
        adaptive_layers: u8,
    ) !void {
        try self.backend.applyCompression(.{
            .enabled = true,
            .bits = bits,
            .adaptive_layers = adaptive_layers,
        });
    }

    /// Get backend type
    pub fn getBackendType(self: BackendGenerator) backends.BackendType {
        return self.backend.getType();
    }

    /// Get vocabulary size
    pub fn vocabSize(self: BackendGenerator) u32 {
        return self.backend.getVocabSize();
    }
};

/// Helper to create generator from model ID
pub fn createGeneratorForModel(
    allocator: std.mem.Allocator,
    model_id: []const u8,
) !BackendGenerator {
    // Look up model in registry
    const model_info = registry.getKnownModel(model_id) orelse {
        // Try as direct path
        return error.ModelNotFound;
    };

    // Resolve model path
    const model_path = try std.fmt.allocPrint(allocator, "./models/{s}", .{model_id});
    defer allocator.free(model_path);

    // Create generator with auto backend selection
    return BackendGenerator.initWithAutoBackend(
        allocator,
        model_path,
        model_info.architecture,
    );
}

test "BackendGenerator imports compile" {
    // Just verify the types exist
    _ = BackendGenerator.initWithAutoBackend;
    _ = BackendGenerator.initWithBackend;
    _ = createGeneratorForModel;
}
