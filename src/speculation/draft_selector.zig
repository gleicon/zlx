//! draft_selector.zig - Automatic draft model selection logic
//!
//! Provides intelligent draft model selection based on target model
//! characteristics and registry scanning.

const std = @import("std");
const registry = @import("../models/registry.zig");

/// Information about a selected draft model
pub const DraftModelInfo = struct {
    model_id: []const u8,
    path: []const u8,
    size_ratio: f32, // draft_size / target_size
    expected_acceptance: f32, // Estimated based on size ratio
    architecture_match: bool, // Same architecture family
    quantization_match: bool, // Same quantization type
};

/// DraftSelector provides automatic draft model selection
pub const DraftSelector = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    registry_ref: *registry.ModelRegistry,

    /// Selection preferences
    preferred_size_ratio_min: f32 = 0.125, // 1:8 minimum
    preferred_size_ratio_max: f32 = 0.25, // 1:4 maximum
    min_acceptance_threshold: f32 = 0.5, // Minimum expected acceptance rate

    pub fn init(allocator: std.mem.Allocator, reg: *registry.ModelRegistry) Self {
        return .{
            .allocator = allocator,
            .registry_ref = reg,
        };
    }

    /// Select best draft model for given target
    /// priority: 1) user override, 2) automatic selection, 3) null (no draft)
    pub fn selectDraftForTarget(
        self: *Self,
        target_model_id: []const u8,
        user_override: ?[]const u8,
    ) ?DraftModelInfo {
        // Priority 1: User override
        if (user_override) |override_id| {
            if (self.validateDraftModel(target_model_id, override_id)) |info| {
                return info;
            }
        }

        // Priority 2: Automatic selection
        return self.findBestDraftAutomatic(target_model_id);
    }

    /// Validate that a user-specified draft model is compatible
    fn validateDraftModel(
        self: *Self,
        target_id: []const u8,
        draft_id: []const u8,
    ) ?DraftModelInfo {
        const target = self.registry_ref.getModel(target_id) orelse return null;
        const draft = self.registry_ref.getModel(draft_id) orelse return null;

        // Check architecture compatibility
        if (!self.isCompatiblePair(target.config, draft.config)) {
            std.log.warn("Draft model '{s}' is not architecturally compatible with target '{s}'", .{ draft_id, target_id });
            return null;
        }

        // Calculate size ratio
        const target_params = target.config.estimateParameterCount();
        const draft_params = draft.config.estimateParameterCount();
        const size_ratio = @as(f32, @floatFromInt(draft_params)) / @as(f32, @floatFromInt(target_params));

        // Estimate acceptance rate (heuristic: higher ratio = higher acceptance)
        const expected_acceptance = 1.0 - (size_ratio * 0.5); // Rough heuristic

        return DraftModelInfo{
            .model_id = draft_id,
            .path = draft.path,
            .size_ratio = size_ratio,
            .expected_acceptance = expected_acceptance,
            .architecture_match = true,
            .quantization_match = target.config.quantization_bits == draft.config.quantization_bits,
        };
    }

    /// Find best draft model automatically from registry
    fn findBestDraftAutomatic(self: *Self, target_id: []const u8) ?DraftModelInfo {
        const target = self.registry_ref.getModel(target_id) orelse return null;
        const target_params = target.config.estimateParameterCount();

        // Parse target model info
        const target_info = self.parseModelName(target_id);

        var best_draft: ?DraftModelInfo = null;
        var best_score: f32 = -1.0;

        // Iterate through all models in registry
        const all_models = self.registry_ref.getAllModels(self.allocator) catch return null;
        defer self.allocator.free(all_models);

        for (all_models) |candidate| {
            // Skip self
            if (std.mem.eql(u8, candidate.id, target_id)) continue;

            // Parse candidate info
            const candidate_info = self.parseModelName(candidate.id);

            // Check architecture compatibility
            const arch_match = self.architectureFamiliesMatch(target_info.family, candidate_info.family);
            if (!arch_match) continue;

            // Check size ratio (prefer 1:4 to 1:8)
            const candidate_params = candidate.config.estimateParameterCount();
            const size_ratio = @as(f32, @floatFromInt(candidate_params)) / @as(f32, @floatFromInt(target_params));

            if (size_ratio > self.preferred_size_ratio_max) continue; // Too large
            if (size_ratio < 0.05) continue; // Too small (< 1:20)

            // Check quantization match
            const quant_match = target.config.quantization_bits == candidate.config.quantization_bits;

            // Calculate expected acceptance
            const expected_acceptance = 1.0 - (size_ratio * 0.5);
            if (expected_acceptance < self.min_acceptance_threshold) continue;

            // Score: prefer smaller ratio (faster draft) with good acceptance
            const score = expected_acceptance * (1.0 - size_ratio);

            if (score > best_score) {
                best_score = score;
                best_draft = DraftModelInfo{
                    .model_id = try self.allocator.dupe(u8, candidate.id),
                    .path = try self.allocator.dupe(u8, candidate.path),
                    .size_ratio = size_ratio,
                    .expected_acceptance = expected_acceptance,
                    .architecture_match = arch_match,
                    .quantization_match = quant_match,
                };
            }
        }

        return best_draft;
    }

    /// Parsed model name information
    const ModelNameInfo = struct {
        family: []const u8, // e.g., "Qwen", "Llama"
        variant: []const u8, // e.g., "2.5-Coder"
        size: []const u8, // e.g., "7B", "1.5B"
        quantization: ?[]const u8, // e.g., "4bit", "8bit"
    };

    /// Parse model name into components
    fn parseModelName(self: *Self, name: []const u8) ModelNameInfo {
        _ = self;

        // Default info
        var info = ModelNameInfo{
            .family = name,
            .variant = "",
            .size = "",
            .quantization = null,
        };

        // Look for size pattern (e.g., "-7B", "-1.5B")
        if (std.mem.indexOf(u8, name, "-")) |first_dash| {
            info.family = name[0..first_dash];

            // Find size pattern
            var i: usize = first_dash;
            while (i < name.len) : (i += 1) {
                if (name[i] == '-') {
                    const remaining = name[i + 1 ..];

                    // Check for size patterns
                    if (std.mem.endsWith(u8, remaining, "B")) {
                        // Look for digit before B
                        var j: usize = remaining.len - 1;
                        while (j > 0) : (j -= 1) {
                            if (remaining[j] == 'B' or remaining[j] == 'b') continue;
                            if (!std.ascii.isDigit(remaining[j]) and remaining[j] != '.') break;
                        }
                        if (j < remaining.len - 1) {
                            info.size = remaining[j + 1 ..];
                            info.variant = name[first_dash..i];
                            break;
                        }
                    }
                }
            }
        }

        // Look for quantization suffix
        if (std.mem.indexOf(u8, name, "-4bit")) |_| {
            info.quantization = "4bit";
        } else if (std.mem.indexOf(u8, name, "-8bit")) |_| {
            info.quantization = "8bit";
        }

        return info;
    }

    /// Check if two architecture families match
    fn architectureFamiliesMatch(self: *Self, family1: []const u8, family2: []const u8) bool {
        _ = self;

        // Case-insensitive comparison
        if (family1.len != family2.len) return false;

        for (family1, family2) |c1, c2| {
            if (std.ascii.toLower(c1) != std.ascii.toLower(c2)) return false;
        }

        return true;
    }

    /// Check if two model configs are compatible as draft/target pair
    pub fn isCompatiblePair(
        target_config: registry.ConfigInfo,
        draft_config: registry.ConfigInfo,
    ) bool {
        // Check architecture compatibility:
        // - Same hidden size suggests same architecture family
        // - Same num_attention_heads suggests compatible attention mechanism
        // - Same vocab_size ensures tokenizer compatibility

        // Vocab size must match for tokenizer compatibility
        if (target_config.vocab_size != draft_config.vocab_size) {
            return false;
        }

        // Hidden size should be proportionally different (draft is smaller)
        const hidden_ratio = @as(f32, @floatFromInt(draft_config.hidden_size)) / @as(f32, @floatFromInt(target_config.hidden_size));

        // Draft should be smaller but not by extreme ratio
        if (hidden_ratio > 0.5 or hidden_ratio < 0.1) {
            return false;
        }

        // Attention heads should scale proportionally
        const head_ratio = @as(f32, @floatFromInt(draft_config.num_attention_heads)) / @as(f32, @floatFromInt(target_config.num_attention_heads));
        const expected_head_ratio = hidden_ratio;

        // Allow 20% tolerance
        const tolerance = 0.2;
        if (head_ratio < expected_head_ratio * (1.0 - tolerance) or
            head_ratio > expected_head_ratio * (1.0 + tolerance))
        {
            return false;
        }

        return true;
    }

    /// Free allocated memory in DraftModelInfo
    pub fn freeDraftInfo(self: *Self, info: DraftModelInfo) void {
        self.allocator.free(info.model_id);
        self.allocator.free(info.path);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "parseModelName basic" {
    const allocator = std.testing.allocator;
    var registry_instance = registry.ModelRegistry.init(allocator);
    defer registry_instance.deinit();

    var selector = DraftSelector.init(allocator, &registry_instance);

    const info1 = selector.parseModelName("Qwen2.5-Coder-7B-4bit");
    try std.testing.expect(std.mem.eql(u8, "Qwen", info1.family));

    const info2 = selector.parseModelName("Llama-3-8B");
    try std.testing.expect(std.mem.eql(u8, "Llama", info2.family));
}

test "architectureFamiliesMatch case insensitive" {
    const allocator = std.testing.allocator;
    var registry_instance = registry.ModelRegistry.init(allocator);
    defer registry_instance.deinit();

    var selector = DraftSelector.init(allocator, &registry_instance);

    try std.testing.expect(selector.architectureFamiliesMatch("Qwen", "qwen"));
    try std.testing.expect(selector.architectureFamiliesMatch("Llama", "Llama"));
    try std.testing.expect(!selector.architectureFamiliesMatch("Qwen", "Llama"));
}

test "isCompatiblePair vocab size mismatch" {
    const target = registry.ConfigInfo{
        .hidden_size = 3584,
        .num_layers = 28,
        .num_attention_heads = 28,
        .vocab_size = 151936,
    };

    const draft = registry.ConfigInfo{
        .hidden_size = 1536,
        .num_layers = 28,
        .num_attention_heads = 16,
        .vocab_size = 32000, // Different vocab
    };

    try std.testing.expect(!DraftSelector.isCompatiblePair(target, draft));
}

test "isCompatiblePair compatible sizes" {
    const target = registry.ConfigInfo{
        .hidden_size = 3584,
        .num_layers = 28,
        .num_attention_heads = 28,
        .vocab_size = 151936,
    };

    const draft = registry.ConfigInfo{
        .hidden_size = 1536, // ~43% of target
        .num_layers = 28,
        .num_attention_heads = 12, // ~43% of target
        .vocab_size = 151936,
    };

    try std.testing.expect(DraftSelector.isCompatiblePair(target, draft));
}

test "isCompatiblePair extreme ratio" {
    const target = registry.ConfigInfo{
        .hidden_size = 3584,
        .num_layers = 28,
        .num_attention_heads = 28,
        .vocab_size = 151936,
    };

    // Draft too large (75% of target)
    const large_draft = registry.ConfigInfo{
        .hidden_size = 2688,
        .num_layers = 28,
        .num_attention_heads = 21,
        .vocab_size = 151936,
    };

    try std.testing.expect(!DraftSelector.isCompatiblePair(target, large_draft));
}
