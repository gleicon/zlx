//! mod.zig - Public API for the models subsystem
//!
//! Re-exports the model registry and provides global instance management.

const std = @import("std");
const registry = @import("registry.zig");

// Re-export types from registry
pub const ModelRegistry = registry.ModelRegistry;
pub const ModelMetadata = registry.ModelMetadata;
pub const ModelStatus = registry.ModelStatus;
pub const ConfigInfo = registry.ConfigInfo;

/// Global registry instance
var global_registry: ?*ModelRegistry = null;
var registry_mutex: std.Thread.Mutex = .{};

/// Initialize the global model registry and scan for models
pub fn initGlobalRegistry(allocator: std.mem.Allocator) !void {
    registry_mutex.lock();
    defer registry_mutex.unlock();

    if (global_registry != null) {
        return; // Already initialized
    }

    const reg = try allocator.create(ModelRegistry);
    errdefer allocator.destroy(reg);

    reg.* = ModelRegistry.init(allocator);

    // Scan all model directories
    try reg.scanAllModels();

    global_registry = reg;

    const count = reg.count();
    if (count > 0) {
        std.log.info("Model registry initialized with {d} model(s)", .{count});
    } else {
        std.log.warn("Model registry initialized but no models found", .{});
        std.log.info("Place models in ./models/ or ~/.cache/zlx/models/", .{});
    }
}

/// Deinitialize the global model registry
pub fn deinitGlobalRegistry(allocator: std.mem.Allocator) void {
    registry_mutex.lock();
    defer registry_mutex.unlock();

    if (global_registry) |reg| {
        reg.deinit();
        allocator.destroy(reg);
        global_registry = null;
        std.log.info("Model registry deinitialized", .{});
    }
}

/// Get the global registry instance
pub fn getGlobalRegistry() ?*ModelRegistry {
    registry_mutex.lock();
    defer registry_mutex.unlock();
    return global_registry;
}

/// Convenience function to create a new registry (for testing or custom use)
pub fn createRegistry(allocator: std.mem.Allocator) !*ModelRegistry {
    const reg = try allocator.create(ModelRegistry);
    reg.* = ModelRegistry.init(allocator);
    return reg;
}

/// Destroy a registry created with createRegistry
pub fn destroyRegistry(allocator: std.mem.Allocator, reg: *ModelRegistry) void {
    reg.deinit();
    allocator.destroy(reg);
}

// ============================================================================
// Tests
// ============================================================================

test "initGlobalRegistry creates and scans registry" {
    const allocator = std.testing.allocator;

    // Clean up any existing registry first
    if (getGlobalRegistry() != null) {
        deinitGlobalRegistry(allocator);
    }

    // Initialize registry
    try initGlobalRegistry(allocator);

    // Verify it was created
    const reg = getGlobalRegistry();
    try std.testing.expect(reg != null);

    // Cleanup
    deinitGlobalRegistry(allocator);
    try std.testing.expect(getGlobalRegistry() == null);
}

test "getGlobalRegistry returns null when not initialized" {
    // Ensure registry is not initialized
    if (getGlobalRegistry() != null) {
        deinitGlobalRegistry(std.testing.allocator);
    }

    const reg = getGlobalRegistry();
    try std.testing.expect(reg == null);
}

test "createRegistry and destroyRegistry work" {
    const allocator = std.testing.allocator;

    // Create a custom registry
    const reg = try createRegistry(allocator);

    // Verify it's usable
    try reg.scanDirectory("./models");

    // Destroy it
    destroyRegistry(allocator, reg);
}
