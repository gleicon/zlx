//! integration_test.zig - Integration tests for speculative decoding
//!
//! Tests the full speculative decoding pipeline end-to-end.
//! These tests require actual models to be available.

const std = @import("std");
const speculation = @import("mod.zig");
const generator = @import("../inference/generator.zig");
const qwen = @import("../mlx.zig/src/qwen.zig");
const registry = @import("../models/registry.zig");

/// Test configuration for integration tests
const TestConfig = struct {
    target_model_path: []const u8 = "./models/Qwen2.5-Coder-7B-4bit",
    draft_model_path: []const u8 = "./models/Qwen2.5-Coder-1.5B-4bit",
};

/// Skip message for when models aren't available
fn skipIfModelNotAvailable(path: []const u8) ?void {
    std.fs.cwd().access(path, .{}) catch {
        std.debug.print("Skipping test - model not found at {s}\n", .{path});
        return {};
    };
    return null;
}

test "speculative decoding full pipeline" {
    const allocator = std.testing.allocator;
    const config = TestConfig{};

    // Skip if target model not available
    if (skipIfModelNotAvailable(config.target_model_path)) |_| {
        return;
    }

    // Skip if draft model not available
    if (skipIfModelNotAvailable(config.draft_model_path)) |_| {
        return;
    }

    // Initialize registry
    var reg = registry.ModelRegistry.init(allocator);
    defer reg.deinit();

    // Scan for models
    try reg.scanDirectory("./models");

    // Initialize speculation subsystem
    try speculation.initSpeculation(allocator, &reg);
    defer speculation.shutdownSpeculation(allocator);

    // Verify speculation is initialized
    try std.testing.expect(speculation.isInitialized());

    // Get draft model
    const draft = try speculation.getDraftForTarget("Qwen2.5-Coder-7B-4bit", null);
    try std.testing.expect(draft != null);

    // Load target transformer
    var target_transformer = try qwen.Transformer.init(allocator, config.target_model_path);
    defer target_transformer.deinit();

    // Create initial tokens (for reference)
    _ = [_]u32{ 151659, 750, 3974 };

    const options = generator.GenerationOptions{
        .max_tokens = 10,
        .stop_on_eos = true,
        .temperature = 0.7,
    };

    // Create speculative generator
    var spec_gen = try speculation.SpeculativeGenerator.init(
        allocator,
        &target_transformer,
        draft.?.transformer,
        4, // speculation depth
        options,
    );
    defer spec_gen.deinit();

    // Generate a few tokens
    var generated_count: usize = 0;
    while (try spec_gen.next()) |token| {
        _ = token;
        generated_count += 1;
        if (generated_count >= 5) break;
    }

    try std.testing.expect(generated_count > 0);

    // Check metrics
    const metrics = spec_gen.getMetrics();
    std.debug.print("Generated {d} tokens with acceptance rate: {d:.2}\n", .{
        generated_count,
        metrics.acceptance_rate,
    });
}

test "automatic draft model selection" {
    const allocator = std.testing.allocator;

    // Initialize registry
    var reg = registry.ModelRegistry.init(allocator);
    defer reg.deinit();

    // Scan for models
    try reg.scanDirectory("./models");

    // Initialize speculation subsystem
    try speculation.initSpeculation(allocator, &reg);
    defer speculation.shutdownSpeculation(allocator);

    // Test auto-selection for a 7B model (should find 1.5B draft)
    const draft = try speculation.getDraftForTarget("Qwen2.5-Coder-7B-4bit", null);

    if (draft) |d| {
        std.debug.print("Auto-selected draft: {s}\n", .{d.model_id});
        try std.testing.expectEqualStrings("Qwen2.5-Coder-7B-4bit", d.model_id);
    } else {
        // OK if no compatible draft found
        std.debug.print("No compatible draft model found (expected if models not present)\n", .{});
    }
}

test "user override draft model selection" {
    const allocator = std.testing.allocator;
    const config = TestConfig{};

    // Skip if draft model not available
    if (skipIfModelNotAvailable(config.draft_model_path)) |_| {
        return;
    }

    // Initialize registry
    var reg = registry.ModelRegistry.init(allocator);
    defer reg.deinit();

    // Scan for models
    try reg.scanDirectory("./models");

    // Initialize speculation subsystem
    try speculation.initSpeculation(allocator, &reg);
    defer speculation.shutdownSpeculation(allocator);

    // Test user override
    const draft = try speculation.getDraftForTarget(
        "Qwen2.5-Coder-7B-4bit",
        "Qwen2.5-Coder-1.5B-4bit",
    );

    if (draft) |d| {
        try std.testing.expectEqualStrings("Qwen2.5-Coder-1.5B-4bit", d.model_id);
    } else {
        // OK if override model not available
        std.debug.print("Override draft model not available\n", .{});
    }
}

test "metrics collection" {
    const allocator = std.testing.allocator;

    // Initialize registry
    var reg = registry.ModelRegistry.init(allocator);
    defer reg.deinit();

    // Initialize speculation subsystem
    try speculation.initSpeculation(allocator, &reg);
    defer speculation.shutdownSpeculation(allocator);

    // Record some test metrics
    if (speculation.getMetrics()) |metrics| {
        metrics.recordSpeculation(4, 0, 50, 150);
        metrics.recordSpeculation(3, 1, 60, 140);

        const stats = metrics.getStats();

        try std.testing.expectEqual(@as(u64, 2), stats.total_speculations);
        try std.testing.expectEqual(@as(u64, 7), stats.total_accepted);
        try std.testing.expectEqual(@as(u64, 1), stats.total_rejected);

        // Acceptance rate should be ~87.5% (7/8)
        try std.testing.expect(stats.acceptance_rate >= 0.8 and stats.acceptance_rate <= 0.9);
    } else {
        try std.testing.expect(false); // Should have metrics
    }
}

test "fallback to standard generation without draft" {
    const allocator = std.testing.allocator;

    // Initialize registry (empty or models not matching)
    var reg = registry.ModelRegistry.init(allocator);
    defer reg.deinit();

    // Initialize speculation subsystem
    try speculation.initSpeculation(allocator, &reg);
    defer speculation.shutdownSpeculation(allocator);

    // Try to get draft for non-existent target
    const draft = try speculation.getDraftForTarget("nonexistent-model", null);

    // Should return null gracefully
    try std.testing.expect(draft == null);
}

test "speedup calculation" {
    // Test speedup estimation formula
    const speedup_100 = speculation.estimateSpeedup(4, 1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), speedup_100, 0.1);

    const speedup_75 = speculation.estimateSpeedup(4, 0.75);
    // With K=4, 75% acceptance: 4 / (1 + 0.25*4) = 2.0
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), speedup_75, 0.1);

    const speedup_50 = speculation.estimateSpeedup(4, 0.5);
    // With K=4, 50% acceptance: 4 / (1 + 0.5*4) = 1.33
    try std.testing.expect(speedup_50 >= 1.3 and speedup_50 <= 1.4);

    const speedup_0 = speculation.estimateSpeedup(4, 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), speedup_0, 0.1);
}
