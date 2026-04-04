//! gptoss_manager.zig - GPT-OSS model lifecycle manager
//!
//! Manages loading, caching, and switching between GPT-OSS model variants.
//! Enforces memory budget limits and evicts least-recently-used models when needed.

const std = @import("std");
const mlx_gptoss_backend = @import("../backends/mlx_gptoss_backend.zig");

const MLXGPTOSSBackend = mlx_gptoss_backend.MLXGPTOSSBackend;

/// Estimated memory usage per model variant (bytes)
const MODEL_MEMORY_20B: usize = 12 * 1024 * 1024 * 1024; // ~12 GB
const MODEL_MEMORY_120B: usize = 60 * 1024 * 1024 * 1024; // ~60 GB

/// Default memory budget: 16 GB (fits 20B model comfortably)
const DEFAULT_MEMORY_BUDGET: usize = 16 * 1024 * 1024 * 1024;

/// Active loaded model entry
pub const ActiveModel = struct {
    name: []const u8,
    backend: MLXGPTOSSBackend,
    last_used_ns: i128,
    memory_usage: usize,
};

/// Metadata for a known model (not necessarily loaded)
pub const LoadedModel = struct {
    name: []const u8,
    path: []const u8,
    memory_usage: usize,
};

/// GPT-OSS model lifecycle manager
///
/// Usage:
///   var mgr = GPTOSSModelManager.init(allocator, .{});
///   defer mgr.deinit();
///   const backend = try mgr.getOrLoadModel("gpt-oss-20b", "/path/to/weights");
///   // ... use backend ...
pub const GPTOSSModelManager = struct {
    allocator: std.mem.Allocator,
    active_model: ?ActiveModel,
    /// Model metadata by name (tracks known paths without loading)
    registry: std.StringHashMap(LoadedModel),
    memory_budget: usize,

    pub const Config = struct {
        memory_budget: usize = DEFAULT_MEMORY_BUDGET,
    };

    /// Initialize a new model manager
    pub fn init(allocator: std.mem.Allocator, config: Config) GPTOSSModelManager {
        return .{
            .allocator = allocator,
            .active_model = null,
            .registry = std.StringHashMap(LoadedModel).init(allocator),
            .memory_budget = config.memory_budget,
        };
    }

    /// Clean up all resources
    pub fn deinit(self: *GPTOSSModelManager) void {
        if (self.active_model) |*m| {
            self.allocator.free(m.name);
            m.backend.deinit();
        }

        // Free registry keys and values
        var it = self.registry.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.name);
            self.allocator.free(entry.value_ptr.path);
        }
        self.registry.deinit();
    }

    /// Register a model path in the registry without loading it.
    pub fn registerModel(
        self: *GPTOSSModelManager,
        name: []const u8,
        path: []const u8,
    ) !void {
        const key = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(key);
        const info = LoadedModel{
            .name = try self.allocator.dupe(u8, name),
            .path = try self.allocator.dupe(u8, path),
            .memory_usage = estimateMemory(name),
        };
        try self.registry.put(key, info);
    }

    /// Get backend for model, loading if necessary.
    /// Evicts the current model if it doesn't match and the new one would fit.
    pub fn getOrLoadModel(
        self: *GPTOSSModelManager,
        name: []const u8,
        model_path: []const u8,
    ) !*MLXGPTOSSBackend {
        // Already loaded?
        if (self.active_model) |*m| {
            if (std.mem.eql(u8, m.name, name)) {
                m.last_used_ns = std.time.nanoTimestamp();
                return &m.backend;
            }
            // Different model — check memory budget
            const needed = estimateMemory(name);
            if (needed > self.memory_budget) {
                return error.InsufficientMemory;
            }
            // Unload current to make room
            self.allocator.free(m.name);
            m.backend.deinit();
            self.active_model = null;
        }

        // Load the new model
        const backend = try MLXGPTOSSBackend.init(
            self.allocator,
            model_path,
            .{}, // default BackendConfig
        );

        self.active_model = .{
            .name = try self.allocator.dupe(u8, name),
            .backend = backend,
            .last_used_ns = std.time.nanoTimestamp(),
            .memory_usage = estimateMemory(name),
        };

        return &self.active_model.?.backend;
    }

    /// Unload the currently active model if it matches name.
    pub fn unloadModel(self: *GPTOSSModelManager, name: []const u8) void {
        if (self.active_model) |*m| {
            if (std.mem.eql(u8, m.name, name)) {
                self.allocator.free(m.name);
                m.backend.deinit();
                self.active_model = null;
            }
        }
    }

    /// Switch from the current model to a different one.
    /// Unloads the current model first.
    pub fn switchModel(
        self: *GPTOSSModelManager,
        from: []const u8,
        to: []const u8,
        to_path: []const u8,
    ) !*MLXGPTOSSBackend {
        self.unloadModel(from);
        return try self.getOrLoadModel(to, to_path);
    }

    /// Returns whether a model is currently loaded.
    pub fn isLoaded(self: *const GPTOSSModelManager, name: []const u8) bool {
        if (self.active_model) |m| {
            return std.mem.eql(u8, m.name, name);
        }
        return false;
    }

    /// Returns approximate memory usage for the active model.
    pub fn currentMemoryUsage(self: *const GPTOSSModelManager) usize {
        if (self.active_model) |m| return m.memory_usage;
        return 0;
    }
};

/// Estimate memory required for a model by name.
fn estimateMemory(name: []const u8) usize {
    if (std.mem.indexOf(u8, name, "120b") != null) {
        return MODEL_MEMORY_120B;
    }
    return MODEL_MEMORY_20B; // Default to 20B estimate
}

// ── Tests ─────────────────────────────────────────────────────────────────────

test "GPTOSSModelManager init and deinit" {
    const allocator = std.testing.allocator;
    var mgr = GPTOSSModelManager.init(allocator, .{});
    defer mgr.deinit();

    try std.testing.expect(!mgr.isLoaded("gpt-oss-20b"));
    try std.testing.expectEqual(@as(usize, 0), mgr.currentMemoryUsage());
}

test "GPTOSSModelManager register model" {
    const allocator = std.testing.allocator;
    var mgr = GPTOSSModelManager.init(allocator, .{});
    defer mgr.deinit();

    try mgr.registerModel("gpt-oss-20b", "/tmp/gptoss-20b");
    try std.testing.expect(mgr.registry.contains("gpt-oss-20b"));
}

test "GPTOSSModelManager memory estimation" {
    try std.testing.expectEqual(MODEL_MEMORY_120B, estimateMemory("gpt-oss-120b"));
    try std.testing.expectEqual(MODEL_MEMORY_20B, estimateMemory("gpt-oss-20b"));
    try std.testing.expectEqual(MODEL_MEMORY_20B, estimateMemory("unknown-model"));
}
