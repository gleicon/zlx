//! speculative_generator_test.zig - Tests for speculative decoding algorithm
//!
//! TDD: Test-driven development for speculative decoding core

const std = @import("std");
const generator = @import("../inference/generator.zig");
const speculation = @import("speculative_generator.zig");

// Mock types for testing without loading actual models
const MockDraftModel = struct {
    tokens_to_generate: []const u32,
    current_index: usize,

    pub fn init(tokens: []const u32) MockDraftModel {
        return .{
            .tokens_to_generate = tokens,
            .current_index = 0,
        };
    }

    pub fn generateTokens(
        self: *MockDraftModel,
        context: []const u32,
        count: usize,
        options: generator.GenerationOptions,
    ) ![]u32 {
        _ = context;
        _ = options;

        const remaining = self.tokens_to_generate.len - self.current_index;
        const to_generate = @min(count, remaining);

        var result = try std.testing.allocator.alloc(u32, to_generate);
        for (0..to_generate) |i| {
            result[i] = self.tokens_to_generate[self.current_index + i];
        }
        self.current_index += to_generate;
        return result;
    }
};

// Mock cache for testing
const MockCache = struct {
    offset: i32 = 0,

    pub fn advance(self: *MockCache, count: usize) void {
        self.offset += @intCast(count);
    }
};

test "SpeculationResult acceptance rate calculation" {
    const result = speculation.SpeculationResult{
        .accepted_tokens = 3,
        .rejected_tokens = 1,
        .total_generated = 4,
        .draft_time_us = 100,
        .target_time_us = 200,
    };

    try std.testing.expectEqual(@as(f32, 0.75), result.getAcceptanceRate());
}

test "SpeculationResult tokens per speculation" {
    const result = speculation.SpeculationResult{
        .accepted_tokens = 4,
        .rejected_tokens = 1,
        .total_generated = 5,
        .draft_time_us = 100,
        .target_time_us = 200,
    };

    try std.testing.expectEqual(@as(f32, 4.0), result.getTokensPerSpeculation());
}

test "acceptance probability calculation - exact match" {
    // When p_target == p_draft, acceptance probability is 1.0
    const p_target: f32 = 0.5;
    const p_draft: f32 = 0.5;

    const prob = speculation.calculateAcceptanceProbability(p_target, p_draft);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), prob, 0.001);
}

test "acceptance probability calculation - target more confident" {
    // When p_target > p_draft, acceptance probability is 1.0
    const p_target: f32 = 0.8;
    const p_draft: f32 = 0.4;

    const prob = speculation.calculateAcceptanceProbability(p_target, p_draft);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), prob, 0.001);
}

test "acceptance probability calculation - draft more confident" {
    // When p_target < p_draft, acceptance probability is p_target/p_draft
    const p_target: f32 = 0.3;
    const p_draft: f32 = 0.6;

    const prob = speculation.calculateAcceptanceProbability(p_target, p_draft);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), prob, 0.001);
}

test "acceptance probability calculation - numerical stability" {
    // Handle very small probabilities without overflow/underflow
    const p_target: f32 = 1e-10;
    const p_draft: f32 = 1e-8;

    const prob = speculation.calculateAcceptanceProbability(p_target, p_draft);
    try std.testing.expect(prob >= 0.0 and prob <= 1.0);
}

test "speculative state initialization" {
    const allocator = std.testing.allocator;

    // Create mock state (actual models require MLX, so we test the struct layout)
    var state = speculation.SpeculativeState{
        .current_tokens = std.ArrayList(u32).init(allocator),
        .speculation_depth = 4,
        .tokens_generated = 0,
        .accepted_count = 0,
        .rejected_count = 0,
        .speculation_count = 0,
    };
    defer state.current_tokens.deinit();

    try std.testing.expectEqual(@as(usize, 4), state.speculation_depth);
    try std.testing.expectEqual(@as(usize, 0), state.tokens_generated);
}

test "token acceptance decision - always accept high probability" {
    // With acceptance probability 1.0 and random value 0.5, should accept
    const accepted = speculation.shouldAcceptToken(1.0, 0.5);
    try std.testing.expect(accepted);
}

test "token acceptance decision - reject when random exceeds prob" {
    // With acceptance probability 0.3 and random value 0.8, should reject
    const accepted = speculation.shouldAcceptToken(0.3, 0.8);
    try std.testing.expect(!accepted);
}

test "token acceptance decision - accept when random below prob" {
    // With acceptance probability 0.7 and random value 0.5, should accept
    const accepted = speculation.shouldAcceptToken(0.7, 0.5);
    try std.testing.expect(accepted);
}

test "softmax probability extraction" {
    // Test extracting probability for specific token from softmax distribution
    const allocator = std.testing.allocator;

    // Create a simple probability distribution
    const logits = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const token_id: u32 = 2; // Want probability for token 2

    const prob = try speculation.extractTokenProbability(allocator, &logits, token_id);

    // Softmax for [1,2,3,4] at index 2 should be around 0.32
    try std.testing.expect(prob > 0.0 and prob <= 1.0);
}

test "speedup estimation formula" {
    // Test the speedup estimation based on acceptance rate
    // Speedup ~ K / (1 + α*K) where K=speculation_depth, α=rejection_rate

    const speedup = speculation.estimateSpeedup(4, 0.75);

    // With K=4 and 75% acceptance (25% rejection):
    // Expected speedup ~ 4 / (1 + 0.25*4) = 4/2 = 2.0
    try std.testing.expect(speedup >= 1.5 and speedup <= 3.0);
}

test "speedup estimation - perfect acceptance" {
    // With 100% acceptance, speedup should equal speculation depth
    const speedup = speculation.estimateSpeedup(4, 1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), speedup, 0.1);
}

test "speedup estimation - zero acceptance" {
    // With 0% acceptance, speedup should be ~1.0 (no benefit)
    const speedup = speculation.estimateSpeedup(4, 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), speedup, 0.1);
}

test "speculation buffer management" {
    const allocator = std.testing.allocator;

    var buffer = try speculation.SpeculationBuffer.init(allocator, 4);
    defer buffer.deinit(allocator);

    // Add draft tokens
    const draft_tokens = [_]u32{ 100, 200, 300, 400 };
    try buffer.setDraftTokens(&draft_tokens);

    try std.testing.expectEqual(@as(usize, 4), buffer.draft_count);
    try std.testing.expectEqual(@as(u32, 100), buffer.draft_tokens[0]);
    try std.testing.expectEqual(@as(u32, 400), buffer.draft_tokens[3]);
}

test "metrics aggregation" {
    var metrics = speculation.SpeculativeMetrics.init();

    // Simulate multiple speculation rounds
    metrics.recordSpeculation(3, 1, 50, 150); // 3 accepted, 1 rejected
    metrics.recordSpeculation(4, 0, 60, 140); // 4 accepted, 0 rejected
    metrics.recordSpeculation(2, 2, 40, 160); // 2 accepted, 2 rejected

    const stats = metrics.getStats();

    try std.testing.expectEqual(@as(u64, 3), stats.total_speculations);
    try std.testing.expectEqual(@as(u64, 9), stats.total_accepted);
    try std.testing.expectEqual(@as(u64, 3), stats.total_rejected);
    try std.testing.expectApproxEqAbs(@as(f32, 0.75), stats.acceptance_rate, 0.01);
}
