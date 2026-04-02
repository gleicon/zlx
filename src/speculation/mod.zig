//! mod.zig - Speculative decoding module public API
//!
//! Provides a unified interface for speculative decoding including
//! automatic draft selection, model management, and metrics tracking.

const std = @import("std");
const registry = @import("../models/registry.zig");
const draft_model = @import("../models/draft_model.zig");
const draft_selector = @import("draft_selector.zig");
const speculative_generator = @import("speculative_generator.zig");
const speculative_metrics = @import("../metrics/speculative_metrics.zig");

// Re-export main types
pub const SpeculativeGenerator = speculative_generator.SpeculativeGenerator;
pub const SpeculationResult = speculative_generator.SpeculationResult;
pub const SpeculationStats = speculative_generator.SpeculationStats;
pub const DraftSelector = draft_selector.DraftSelector;
pub const DraftModelInfo = draft_selector.DraftModelInfo;
pub const DraftModel = draft_model.DraftModel;
pub const DraftModelManager = draft_model.DraftModelManager;
pub const SpeculativeMetrics = speculative_metrics.SpeculativeMetrics;

/// Global singleton instances
var global_selector: ?*DraftSelector = null;
var global_draft_manager: ?*DraftModelManager = null;
var global_metrics: ?*SpeculativeMetrics = null;
var global_initialized: bool = false;

/// Initialize the speculation subsystem
/// Must be called before any speculative decoding operations
pub fn initSpeculation(allocator: std.mem.Allocator, reg: *registry.ModelRegistry) !void {
    if (global_initialized) return;

    // Initialize draft selector
    const selector = try allocator.create(DraftSelector);
    selector.* = DraftSelector.init(allocator, reg);
    global_selector = selector;

    // Initialize draft model manager
    const manager = try allocator.create(DraftModelManager);
    manager.* = DraftModelManager.init(allocator);
    global_draft_manager = manager;

    // Initialize global metrics
    const metrics = try allocator.create(SpeculativeMetrics);
    metrics.* = SpeculativeMetrics.init();
    global_metrics = metrics;

    global_initialized = true;

    std.log.info("Speculative decoding subsystem initialized", .{});
}

/// Shutdown the speculation subsystem and free resources
pub fn shutdownSpeculation(allocator: std.mem.Allocator) void {
    if (!global_initialized) return;

    if (global_selector) |selector| {
        allocator.destroy(selector);
        global_selector = null;
    }

    if (global_draft_manager) |manager| {
        manager.deinit();
        allocator.destroy(manager);
        global_draft_manager = null;
    }

    if (global_metrics) |metrics| {
        allocator.destroy(metrics);
        global_metrics = null;
    }

    global_initialized = false;

    std.log.info("Speculative decoding subsystem shutdown", .{});
}

/// Check if speculation subsystem is initialized
pub fn isInitialized() bool {
    return global_initialized;
}

/// Get draft model for a target model (with auto-selection)
/// Returns null if no compatible draft available or speculation disabled
pub fn getDraftForTarget(target_id: []const u8, user_override: ?[]const u8) !?*DraftModel {
    if (!global_initialized) return null;

    const manager = global_draft_manager.?;
    const selector = global_selector.?;

    return try manager.getOrLoadDraft(target_id, selector, user_override);
}

/// Get current draft model ID for a target
pub fn getCurrentDraftId(target_id: []const u8) ?[]const u8 {
    if (!global_initialized) return null;

    return global_draft_manager.?.getDraftForTarget(target_id);
}

/// Check if speculation is enabled for a target model
pub fn isSpeculationEnabled(target_id: []const u8) bool {
    if (!global_initialized) return false;

    return global_draft_manager.?.getDraftForTarget(target_id) != null;
}

/// Get global metrics for speculative decoding
pub fn getMetrics() ?*SpeculativeMetrics {
    if (!global_initialized) return null;
    return global_metrics;
}

/// Record speculation metrics (called by SpeculativeGenerator)
pub fn recordSpeculationMetrics(
    accepted: usize,
    rejected: usize,
    draft_time_us: u64,
    target_time_us: u64,
) void {
    if (!global_initialized) return;

    global_metrics.?.recordSpeculation(
        accepted,
        rejected,
        draft_time_us,
        target_time_us,
    );
}

/// Get global draft selector (for advanced use cases)
pub fn getSelector() ?*DraftSelector {
    return global_selector;
}

/// Get global draft model manager (for advanced use cases)
pub fn getManager() ?*DraftModelManager {
    return global_draft_manager;
}

/// Configuration for speculative decoding
pub const SpeculationConfig = struct {
    enabled: bool = true,
    draft_model: ?[]const u8 = null, // null = auto-select
    speculation_depth: usize = 4,
    min_acceptance_threshold: f32 = 0.3, // Disable if acceptance below this
    max_draft_memory_mb: u32 = 4096, // 4GB max for draft models
};

/// Update speculation configuration
pub fn configure(config: SpeculationConfig) void {
    if (!global_initialized) return;

    if (global_selector) |selector| {
        selector.min_acceptance_threshold = config.min_acceptance_threshold;
    }

    if (global_draft_manager) |manager| {
        manager.max_draft_memory_mb = config.max_draft_memory_mb;
    }
}

/// Unload draft model for a target (useful for memory management)
pub fn unloadDraftForTarget(target_id: []const u8) void {
    if (!global_initialized) return;

    global_draft_manager.?.unloadDraftForTarget(target_id);
}

// ============================================================================
// Tests
// ============================================================================

test "init and shutdown speculation" {
    const allocator = std.testing.allocator;

    var reg = registry.ModelRegistry.init(allocator);
    defer reg.deinit();

    try initSpeculation(allocator, &reg);
    try std.testing.expect(isInitialized());

    shutdownSpeculation(allocator);
    try std.testing.expect(!isInitialized());
}

test "getDraftForTarget without models returns null" {
    const allocator = std.testing.allocator;

    var reg = registry.ModelRegistry.init(allocator);
    defer reg.deinit();

    try initSpeculation(allocator, &reg);
    defer shutdownSpeculation(allocator);

    const draft = try getDraftForTarget("nonexistent-model", null);
    try std.testing.expect(draft == null);
}

test "isSpeculationEnabled returns false initially" {
    const allocator = std.testing.allocator;

    var reg = registry.ModelRegistry.init(allocator);
    defer reg.deinit();

    try initSpeculation(allocator, &reg);
    defer shutdownSpeculation(allocator);

    try std.testing.expect(!isSpeculationEnabled("any-model"));
}

test "getMetrics returns null when not initialized" {
    // Ensure clean state
    if (isInitialized()) {
        shutdownSpeculation(std.testing.allocator);
    }

    const metrics = getMetrics();
    try std.testing.expect(metrics == null);
}

test "recordSpeculationMetrics no crash when uninitialized" {
    // Ensure clean state
    if (isInitialized()) {
        shutdownSpeculation(std.testing.allocator);
    }

    // Should not crash even when uninitialized
    recordSpeculationMetrics(3, 1, 100, 200);
}
