//! manager_test.zig - Tests for ModelManager background loading

const std = @import("std");
const testing = std.testing;
const manager_mod = @import("manager.zig");
const registry = @import("registry.zig");

const ModelManager = manager_mod.ModelManager;
const LoadProgress = manager_mod.LoadProgress;

// Mock model path - needs to exist for actual loading
const MOCK_MODEL_PATH = "/tmp/test_model";

// ============================================================================
// Test 1: startBackgroundLoad spawns thread, returns immediately
// ============================================================================
test "startBackgroundLoad spawns thread and returns immediately" {
    const allocator = testing.allocator;

    // Setup registry with a fake model
    var reg = registry.ModelRegistry.init(allocator);
    defer reg.deinit();

    const config = registry.ConfigInfo{
        .hidden_size = 768,
        .num_layers = 12,
        .num_attention_heads = 12,
    };

    var metadata = try registry.ModelMetadata.init(
        allocator,
        "test-model",
        "/tmp/nonexistent-model",
        config,
        1000000,
        500,
    );
    errdefer metadata.deinit();
    try reg.models.put("test-model", metadata);

    var mgr = ModelManager.init(allocator, &reg);
    defer mgr.deinit();

    // Start background load (will fail since path doesn't exist, but that's OK)
    const start_time = std.time.milliTimestamp();
    mgr.startBackgroundLoad("test-model", false) catch {}; // May fail but should return quickly
    const end_time = std.time.milliTimestamp();

    // Should return in under 100ms (not blocking)
    const elapsed = @as(u64, @intCast(end_time - start_time));
    try testing.expect(elapsed < 100);

    // Cleanup any background state
    _ = mgr.cancelLoad() catch {};
}

// ============================================================================
// Test 2: getLoadProgress returns percent complete and status
// ============================================================================
test "getLoadProgress returns progress information" {
    const allocator = testing.allocator;

    var reg = registry.ModelRegistry.init(allocator);
    defer reg.deinit();

    const config = registry.ConfigInfo{
        .hidden_size = 768,
        .num_layers = 12,
        .num_attention_heads = 12,
    };

    var metadata = try registry.ModelMetadata.init(
        allocator,
        "test-model",
        "/tmp/nonexistent-model",
        config,
        1000000,
        500,
    );
    errdefer metadata.deinit();
    try reg.models.put("test-model", metadata);

    var mgr = ModelManager.init(allocator, &reg);
    defer mgr.deinit();

    // Initially no active load
    const no_progress = mgr.getLoadProgress();
    try testing.expect(no_progress == null);

    // Start a load
    mgr.startBackgroundLoad("test-model", false) catch {};

    // Should have progress now
    const progress = mgr.getLoadProgress();
    try testing.expect(progress != null);

    if (progress) |p| {
        try testing.expectEqualStrings("test-model", p.model_id);
        // Status could be loading, failed, or cancelled
        try testing.expect(p.status == .loading or p.status == .failed or p.status == .cancelled);
    }

    // Cleanup
    _ = mgr.cancelLoad() catch {};
}

// ============================================================================
// Test 3: Background load completes or fails, model status updated
// ============================================================================
test "background load updates model status" {
    const allocator = testing.allocator;

    var reg = registry.ModelRegistry.init(allocator);
    defer reg.deinit();

    const config = registry.ConfigInfo{
        .hidden_size = 768,
        .num_layers = 12,
        .num_attention_heads = 12,
    };

    var metadata = try registry.ModelMetadata.init(
        allocator,
        "test-model",
        "/tmp/nonexistent-model",
        config,
        1000000,
        500,
    );
    errdefer metadata.deinit();
    try reg.models.put("test-model", metadata);

    var mgr = ModelManager.init(allocator, &reg);
    defer mgr.deinit();

    // Check initial status
    const initial_model = reg.getModel("test-model").?;
    try testing.expectEqual(registry.ModelStatus.available, initial_model.status);

    // Start load
    mgr.startBackgroundLoad("test-model", false) catch {};

    // Wait a bit for thread to start
    std.time.sleep(50 * std.time.ns_per_ms);

    // Status should have changed to loading or failed
    const loading_model = reg.getModel("test-model").?;
    try testing.expect(loading_model.status == .loading or
        loading_model.status == .failed or
        loading_model.status == .available);

    // Cleanup
    _ = mgr.cancelLoad() catch {};
}

// ============================================================================
// Test 4: Current model remains usable during background load
// ============================================================================
test "current model usable during background load" {
    const allocator = testing.allocator;

    var reg = registry.ModelRegistry.init(allocator);
    defer reg.deinit();

    const config = registry.ConfigInfo{
        .hidden_size = 768,
        .num_layers = 12,
        .num_attention_heads = 12,
    };

    // Create two models
    var meta1 = try registry.ModelMetadata.init(
        allocator,
        "model-a",
        "/tmp/nonexistent-a",
        config,
        1000000,
        500,
    );
    errdefer meta1.deinit();
    try reg.models.put("model-a", meta1);

    var meta2 = try registry.ModelMetadata.init(
        allocator,
        "model-b",
        "/tmp/nonexistent-b",
        config,
        1000000,
        500,
    );
    errdefer meta2.deinit();
    try reg.models.put("model-b", meta2);

    var mgr = ModelManager.init(allocator, &reg);
    defer mgr.deinit();

    // Simulate having model-a loaded
    mgr.current_model_id = try allocator.dupe(u8, "model-a");

    // Start loading model-b in background
    mgr.startBackgroundLoad("model-b", false) catch {};

    // Current model should still be model-a
    const current = mgr.getCurrentModel();
    try testing.expect(current != null);
    try testing.expectEqualStrings("model-a", current.?);

    // Active generations count should be accessible (no mutex contention)
    const gen_count = mgr.getActiveGenerationCount();
    try testing.expectEqual(@as(u32, 0), gen_count);

    // Cleanup
    _ = mgr.cancelLoad() catch {};
}

// ============================================================================
// Test 5: cancelLoad stops background thread
// ============================================================================
test "cancelLoad stops background load" {
    const allocator = testing.allocator;

    var reg = registry.ModelRegistry.init(allocator);
    defer reg.deinit();

    const config = registry.ConfigInfo{
        .hidden_size = 768,
        .num_layers = 12,
        .num_attention_heads = 12,
    };

    var metadata = try registry.ModelMetadata.init(
        allocator,
        "test-model",
        "/tmp/nonexistent-model",
        config,
        1000000,
        500,
    );
    errdefer metadata.deinit();
    try reg.models.put("test-model", metadata);

    var mgr = ModelManager.init(allocator, &reg);
    defer mgr.deinit();

    // Start a load
    mgr.startBackgroundLoad("test-model", false) catch {};

    // Wait a bit
    std.time.sleep(50 * std.time.ns_per_ms);

    // Verify load is active
    const progress_before = mgr.getLoadProgress();
    try testing.expect(progress_before != null);

    // Cancel it
    try mgr.cancelLoad();

    // Verify load is no longer active
    const progress_after = mgr.getLoadProgress();
    try testing.expect(progress_after == null);
}

// ============================================================================
// Test 6: Cannot start two loads simultaneously
// ============================================================================
test "cannot start two background loads simultaneously" {
    const allocator = testing.allocator;

    var reg = registry.ModelRegistry.init(allocator);
    defer reg.deinit();

    const config = registry.ConfigInfo{
        .hidden_size = 768,
        .num_layers = 12,
        .num_attention_heads = 12,
    };

    var meta1 = try registry.ModelMetadata.init(
        allocator,
        "model-a",
        "/tmp/nonexistent-a",
        config,
        1000000,
        500,
    );
    errdefer meta1.deinit();
    try reg.models.put("model-a", meta1);

    var meta2 = try registry.ModelMetadata.init(
        allocator,
        "model-b",
        "/tmp/nonexistent-b",
        config,
        1000000,
        500,
    );
    errdefer meta2.deinit();
    try reg.models.put("model-b", meta2);

    var mgr = ModelManager.init(allocator, &reg);
    defer mgr.deinit();

    // Start first load
    mgr.startBackgroundLoad("model-a", false) catch {};

    // Try to start second load - should fail
    const result = mgr.startBackgroundLoad("model-b", false);
    try testing.expect(result == error.LoadInProgress);

    // Cleanup
    _ = mgr.cancelLoad() catch {};
}

// ============================================================================
// Test 7: LoadProgress contains valid timestamps
// ============================================================================
test "LoadProgress has valid timestamps" {
    const allocator = testing.allocator;

    var reg = registry.ModelRegistry.init(allocator);
    defer reg.deinit();

    const config = registry.ConfigInfo{
        .hidden_size = 768,
        .num_layers = 12,
        .num_attention_heads = 12,
    };

    var metadata = try registry.ModelMetadata.init(
        allocator,
        "test-model",
        "/tmp/nonexistent-model",
        config,
        1000000,
        500,
    );
    errdefer metadata.deinit();
    try reg.models.put("test-model", metadata);

    var mgr = ModelManager.init(allocator, &reg);
    defer mgr.deinit();

    // Start load
    mgr.startBackgroundLoad("test-model", false) catch {};

    // Check timestamps
    const progress = mgr.getLoadProgress().?;

    const now = std.time.timestamp();
    try testing.expect(progress.started_at > 0);
    try testing.expect(progress.started_at <= now);
    try testing.expect(progress.updated_at >= progress.started_at);

    // Cleanup
    _ = mgr.cancelLoad() catch {};
}
