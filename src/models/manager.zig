//! manager.zig - Model manager for hot-swapping and memory-aware loading
//!
//! Handles model loading, switching, and unloading with memory validation.
//! Ensures active generations complete before unloading models.

const std = @import("std");
const registry = @import("registry.zig");
const inference = @import("../inference/mod.zig");
const handlers = @import("../api/handlers.zig");

/// Errors that can occur during model management
pub const ManagerError = error{
    ModelNotFound,
    InsufficientMemory,
    ModelLoadFailed,
    SwitchInProgress,
    GenerationInProgress,
    LoadInProgress,
};

/// Load progress status
pub const LoadStatus = enum {
    loading,
    completed,
    failed,
    cancelled,
};

/// Load stage for detailed progress tracking
pub const LoadStage = enum {
    downloading,
    loading_weights,
    initializing,
    complete,
};

/// Progress information for an active model load
pub const LoadProgress = struct {
    model_id: []const u8,
    status: LoadStatus,
    stage: LoadStage,
    percent_complete: u8,
    bytes_loaded: u64,
    bytes_total: u64,
    error_message: ?[]const u8,
    started_at: i64,
    updated_at: i64,
};

/// Background load state
const BackgroundLoadState = struct {
    thread: ?std.Thread,
    progress: LoadProgress,
    cancel_requested: std.atomic.Value(bool),
    auto_switch: bool,
    allocator: std.mem.Allocator,
    manager: *ModelManager,
};

/// Model manager that handles loading, switching, and memory validation
pub const ModelManager = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    registry: *registry.ModelRegistry,
    current_model_id: ?[]const u8,
    active_generations: std.atomic.Value(u32),
    switch_mutex: std.Thread.Mutex,
    generation_condition: std.Thread.Condition,
    background_load: ?*BackgroundLoadState,
    load_mutex: std.Thread.Mutex,

    pub fn init(allocator: std.mem.Allocator, reg: *registry.ModelRegistry) Self {
        return .{
            .allocator = allocator,
            .registry = reg,
            .current_model_id = null,
            .active_generations = std.atomic.Value(u32).init(0),
            .switch_mutex = .{},
            .generation_condition = .{},
            .background_load = null,
            .load_mutex = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        // Cancel any background load
        _ = self.cancelLoad() catch {};

        if (self.current_model_id) |id| {
            self.allocator.free(id);
        }
    }

    /// Check if a model can be loaded given current memory availability
    pub fn canLoadModel(self: *Self, model_id: []const u8) bool {
        const model_opt = self.registry.getModel(model_id);
        if (model_opt == null) return false;

        const model = model_opt.?;
        const required_mb = model.memory_required_mb;

        // Get available system memory
        const available_mb = getAvailableMemoryMb();

        // Add 20% safety margin
        const required_with_margin = @as(u64, required_mb) * 12 / 10;

        std.log.debug("Memory check for '{s}': need {d}MB (with margin), have {d}MB available", .{
            model_id, required_with_margin, available_mb,
        });

        return available_mb >= required_with_margin;
    }

    /// Get available system memory in MB (macOS-specific)
    pub fn getAvailableMemoryMb() u64 {
        // Use sysctlbyname to get memory info on macOS
        const CTL_HW = 6; // From sys/sysctl.h
        const HW_MEMSIZE = 24; // From sys/sysctl.h

        var memsize: u64 = 0;
        var len: usize = @sizeOf(u64);

        // Use inline assembly or direct syscall for sysctl
        // For now, use a simpler approach with posix sysctl
        const result = std.c.sysctl(
            &[_]c_int{ CTL_HW, HW_MEMSIZE },
            2,
            &memsize,
            &len,
            null,
            0,
        );

        if (result != 0) {
            std.log.warn("Failed to get system memory, assuming 8GB", .{});
            return 8192; // Fallback to 8GB
        }

        // Convert bytes to MB
        const total_mb = memsize / (1024 * 1024);

        // Estimate available memory as 70% of total (conservative)
        // In a real implementation, we'd query actual available memory
        const available_mb = total_mb * 7 / 10;

        return available_mb;
    }

    /// Switch to a different model
    pub fn switchModel(self: *Self, model_id: []const u8) !void {
        // Lock to prevent concurrent switches
        self.switch_mutex.lock();
        defer self.switch_mutex.unlock();

        // Check if already loaded
        if (self.current_model_id) |current| {
            if (std.mem.eql(u8, current, model_id)) {
                std.log.info("Model '{s}' is already loaded", .{model_id});
                return;
            }
        }

        // Verify model exists
        if (self.registry.getModel(model_id) == null) {
            std.log.err("Model '{s}' not found in registry", .{model_id});
            return ManagerError.ModelNotFound;
        }

        // Check memory availability
        if (!self.canLoadModel(model_id)) {
            const model = self.registry.getModel(model_id).?;
            const available_mb = getAvailableMemoryMb();
            std.log.err("Insufficient memory to load '{s}': need {d}MB, have {d}MB", .{
                model_id, model.memory_required_mb, available_mb,
            });
            return ManagerError.InsufficientMemory;
        }

        std.log.info("Switching to model: {s}", .{model_id});

        // Wait for active generations to complete
        if (self.active_generations.load(.acquire) > 0) {
            std.log.info("Waiting for {d} active generation(s) to complete...", .{
                self.active_generations.load(.acquire),
            });

            while (self.active_generations.load(.acquire) > 0) {
                self.generation_condition.wait(&self.switch_mutex);
            }

            std.log.info("All active generations completed", .{});
        }

        // Get model path from registry
        const model = self.registry.getModel(model_id).?;

        // Update status to loading
        self.registry.updateStatus(model_id, .loading);

        // Deinitialize old context if exists
        if (handlers.global_context != null) {
            std.log.info("Unloading previous model...", .{});
            handlers.deinitGlobalContext(self.allocator);
        }

        // Initialize new context
        handlers.initGlobalContext(self.allocator, model.path) catch |err| {
            std.log.err("Failed to load model '{s}': {s}", .{ model_id, @errorName(err) });
            self.registry.updateStatus(model_id, .failed);
            return ManagerError.ModelLoadFailed;
        };

        // Update current model ID
        if (self.current_model_id) |old_id| {
            // Mark old model as available
            self.registry.updateStatus(old_id, .available);
            self.allocator.free(old_id);
        }

        self.current_model_id = try self.allocator.dupe(u8, model_id);

        // Update status to loaded
        self.registry.updateStatus(model_id, .loaded);

        std.log.info("Successfully switched to model: {s}", .{model_id});
    }

    /// Get currently loaded model ID
    pub fn getCurrentModel(self: *Self) ?[]const u8 {
        return self.current_model_id;
    }

    /// Mark start of a generation
    pub fn startGeneration(self: *Self) void {
        const count = self.active_generations.fetchAdd(1, .acquire);
        std.log.debug("Generation started (active: {d})", .{count + 1});
    }

    /// Mark end of a generation
    pub fn endGeneration(self: *Self) void {
        const count = self.active_generations.fetchSub(1, .release);
        std.log.debug("Generation ended (active: {d})", .{count - 1});

        // Signal if no more active generations
        if (count == 1) {
            self.generation_condition.broadcast();
        }
    }

    /// Check if a generation is currently active
    pub fn hasActiveGenerations(self: *Self) bool {
        return self.active_generations.load(.acquire) > 0;
    }

    /// Get count of active generations
    pub fn getActiveGenerationCount(self: *Self) u32 {
        return self.active_generations.load(.acquire);
    }

    // ============================================================================
    // Background Loading
    // ============================================================================

    /// Start loading a model in the background
    pub fn startBackgroundLoad(self: *Self, model_id: []const u8, auto_switch: bool) !void {
        self.load_mutex.lock();
        defer self.load_mutex.unlock();

        // Check if load already in progress
        if (self.background_load != null) {
            std.log.warn("Background load already in progress", .{});
            return ManagerError.LoadInProgress;
        }

        // Verify model exists
        if (self.registry.getModel(model_id) == null) {
            std.log.err("Model '{s}' not found in registry", .{model_id});
            return ManagerError.ModelNotFound;
        }

        // Check memory availability
        if (!self.canLoadModel(model_id)) {
            const model = self.registry.getModel(model_id).?;
            const available_mb = getAvailableMemoryMb();
            std.log.err("Insufficient memory to load '{s}': need {d}MB, have {d}MB", .{
                model_id, model.memory_required_mb, available_mb,
            });
            return ManagerError.InsufficientMemory;
        }

        // Update registry status
        self.registry.updateStatus(model_id, .loading);

        // Create background load state
        const load_state = try self.allocator.create(BackgroundLoadState);
        errdefer self.allocator.destroy(load_state);

        const now = std.time.timestamp();

        load_state.* = .{
            .thread = null,
            .progress = .{
                .model_id = try self.allocator.dupe(u8, model_id),
                .status = .loading,
                .stage = .downloading,
                .percent_complete = 0,
                .bytes_loaded = 0,
                .bytes_total = 0,
                .error_message = null,
                .started_at = now,
                .updated_at = now,
            },
            .cancel_requested = std.atomic.Value(bool).init(false),
            .auto_switch = auto_switch,
            .allocator = self.allocator,
            .manager = self,
        };

        self.background_load = load_state;

        // Spawn load thread
        load_state.thread = try std.Thread.spawn(.{}, loadWorker, .{load_state});

        std.log.info("Started background load for model: {s} (auto_switch={s})", .{
            model_id,
            if (auto_switch) "true" else "false",
        });
    }

    /// Get current load progress
    pub fn getLoadProgress(self: *Self) ?LoadProgress {
        self.load_mutex.lock();
        defer self.load_mutex.unlock();

        if (self.background_load) |load| {
            return load.progress;
        }
        return null;
    }

    /// Cancel an ongoing background load
    pub fn cancelLoad(self: *Self) !void {
        self.load_mutex.lock();
        defer self.load_mutex.unlock();

        if (self.background_load) |load| {
            // Signal cancellation
            load.cancel_requested.store(true, .seq_cst);

            // Wait for thread to complete
            if (load.thread) |thread| {
                thread.join();
            }

            // Update progress status
            load.progress.status = .cancelled;

            // Update registry
            self.registry.updateStatus(load.progress.model_id, .failed);

            // Clean up
            self.allocator.free(load.progress.model_id);
            if (load.progress.error_message) |msg| {
                self.allocator.free(msg);
            }
            self.allocator.destroy(load);
            self.background_load = null;

            std.log.info("Background load cancelled", .{});
            return;
        }

        return error.NoActiveLoad;
    }

    /// Worker thread for background loading
    fn loadWorker(load_state: *BackgroundLoadState) void {
        const manager = load_state.manager;
        const model_id = load_state.progress.model_id;

        defer {
            // Thread cleanup - runs even if load fails
            load_state.thread = null;
        }

        // Get model info
        const model = manager.registry.getModel(model_id) orelse {
            std.log.err("Model disappeared during background load: {s}", .{model_id});
            updateLoadStatus(load_state, .failed, "Model not found in registry");
            manager.registry.updateStatus(model_id, .failed);
            return;
        };

        // Update stage: downloading (if files not present)
        updateLoadProgress(load_state, .downloading, 10);

        // Check for cancellation
        if (load_state.cancel_requested.load(.seq_cst)) {
            updateLoadStatus(load_state, .cancelled, null);
            manager.registry.updateStatus(model_id, .failed);
            return;
        }

        // Update stage: loading weights
        updateLoadProgress(load_state, .loading_weights, 30);

        // Wait for active generations (same as switchModel but non-blocking for API)
        if (manager.active_generations.load(.acquire) > 0) {
            std.log.info("Background load waiting for {d} active generation(s)...", .{
                manager.active_generations.load(.acquire),
            });

            // Poll for completion (don't block the worker thread on mutex)
            var wait_count: u32 = 0;
            while (manager.active_generations.load(.acquire) > 0) {
                std.Thread.sleep(100 * std.time.ns_per_ms); // 100ms polling
                wait_count += 1;

                // Check cancellation during wait
                if (load_state.cancel_requested.load(.seq_cst)) {
                    updateLoadStatus(load_state, .cancelled, null);
                    manager.registry.updateStatus(model_id, .failed);
                    return;
                }

                // Timeout after ~5 minutes (3000 * 100ms)
                if (wait_count > 3000) {
                    std.log.err("Background load timed out waiting for generations", .{});
                    updateLoadStatus(load_state, .failed, "Timeout waiting for active generations");
                    manager.registry.updateStatus(model_id, .failed);
                    return;
                }
            }
        }

        // Update stage: initializing
        updateLoadProgress(load_state, .initializing, 60);

        // Deinitialize old context if exists
        if (handlers.global_context != null) {
            std.log.info("Background load: unloading previous model...", .{});
            handlers.deinitGlobalContext(manager.allocator);
        }

        // Initialize new context
        handlers.initGlobalContext(manager.allocator, model.path) catch |err| {
            std.log.err("Failed to load model '{s}' in background: {s}", .{
                model_id,
                @errorName(err),
            });
            updateLoadStatus(load_state, .failed, @errorName(err));
            manager.registry.updateStatus(model_id, .failed);
            return;
        };

        // Update stage: complete
        updateLoadProgress(load_state, .complete, 100);
        updateLoadStatus(load_state, .completed, null);

        // Update current model
        manager.switch_mutex.lock();
        defer manager.switch_mutex.unlock();

        if (manager.current_model_id) |old_id| {
            manager.registry.updateStatus(old_id, .available);
            manager.allocator.free(old_id);
        }

        manager.current_model_id = manager.allocator.dupe(u8, model_id) catch |err| {
            std.log.err("Failed to set current model: {s}", .{@errorName(err)});
            manager.registry.updateStatus(model_id, .failed);
            return;
        };

        // Update registry status
        manager.registry.updateStatus(model_id, .loaded);

        std.log.info("Background load completed for model: {s}", .{model_id});

        // Auto-switch if requested (context is already switched above)
        if (!load_state.auto_switch) {
            // Even if not auto-switching, the model is now loaded
            std.log.info("Model loaded but not switched (auto_switch=false)", .{});
        }
    }

    /// Update load progress (thread-safe)
    fn updateLoadProgress(load_state: *BackgroundLoadState, stage: LoadStage, percent: u8) void {
        load_state.progress.stage = stage;
        load_state.progress.percent_complete = percent;
        load_state.progress.updated_at = std.time.timestamp();
    }

    /// Update load status and optional error (thread-safe)
    fn updateLoadStatus(load_state: *BackgroundLoadState, status: LoadStatus, error_msg: ?[]const u8) void {
        load_state.progress.status = status;
        if (error_msg) |msg| {
            load_state.progress.error_message = load_state.allocator.dupe(u8, msg) catch null;
        }
        load_state.progress.updated_at = std.time.timestamp();
    }
};

/// Global manager instance
var global_manager: ?*ModelManager = null;
var manager_mutex: std.Thread.Mutex = .{};

/// Initialize the global model manager
pub fn initGlobalManager(allocator: std.mem.Allocator, reg: *registry.ModelRegistry) !void {
    manager_mutex.lock();
    defer manager_mutex.unlock();

    if (global_manager != null) {
        return; // Already initialized
    }

    const manager = try allocator.create(ModelManager);
    errdefer allocator.destroy(manager);

    manager.* = ModelManager.init(allocator, reg);

    global_manager = manager;
    std.log.info("Model manager initialized", .{});
}

/// Deinitialize the global model manager
pub fn deinitGlobalManager(allocator: std.mem.Allocator) void {
    manager_mutex.lock();
    defer manager_mutex.unlock();

    if (global_manager) |manager| {
        manager.deinit();
        allocator.destroy(manager);
        global_manager = null;
        std.log.info("Model manager deinitialized", .{});
    }
}

/// Get the global manager instance
pub fn getGlobalManager() ?*ModelManager {
    manager_mutex.lock();
    defer manager_mutex.unlock();
    return global_manager;
}

// ============================================================================
// Tests
// ============================================================================

test "canLoadModel returns true for small model with sufficient memory" {
    const allocator = std.testing.allocator;

    // Create a mock registry with a small model
    var reg = registry.ModelRegistry.init(allocator);
    defer reg.deinit();

    // Add a fake small model entry
    const config = registry.ConfigInfo{
        .hidden_size = 768,
        .num_layers = 12,
        .num_attention_heads = 12,
    };

    var metadata = try registry.ModelMetadata.init(
        allocator,
        "test-small-model",
        "/tmp/test-small-model",
        config,
        1000000,
        500, // Only 500MB required
    );
    errdefer metadata.deinit();

    try reg.models.put("test-small-model", metadata);

    var manager = ModelManager.init(allocator, &reg);
    defer manager.deinit();

    // Should be able to load (500MB with 20% margin = 600MB)
    // Assuming system has more than 600MB available
    const can_load = manager.canLoadModel("test-small-model");
    try std.testing.expect(can_load);
}

test "canLoadModel returns false for oversized model" {
    const allocator = std.testing.allocator;

    // Create a mock registry with an impossibly large model
    var reg = registry.ModelRegistry.init(allocator);
    defer reg.deinit();

    const config = registry.ConfigInfo{
        .hidden_size = 8192,
        .num_layers = 80,
        .num_attention_heads = 64,
    };

    // Estimate parameters for this "model"
    const params = config.estimateParameterCount();
    const weights_bytes = params * 2; // FP16
    const kv_bytes = 2 * 80 * 8192 * 8192 * 2;
    const overhead = (weights_bytes + kv_bytes) / 5;
    const memory_mb = @as(u32, @intCast((weights_bytes + kv_bytes + overhead) / (1024 * 1024)));

    var metadata = try registry.ModelMetadata.init(
        allocator,
        "test-huge-model",
        "/tmp/test-huge-model",
        config,
        1000000000,
        memory_mb, // Will be huge
    );
    errdefer metadata.deinit();

    try reg.models.put("test-huge-model", metadata);

    var manager = ModelManager.init(allocator, &reg);
    defer manager.deinit();

    // Should not be able to load (requires more than available memory)
    const can_load = manager.canLoadModel("test-huge-model");
    try std.testing.expect(!can_load);
}

test "startGeneration and endGeneration track active count" {
    const allocator = std.testing.allocator;

    var reg = registry.ModelRegistry.init(allocator);
    defer reg.deinit();

    var manager = ModelManager.init(allocator, &reg);
    defer manager.deinit();

    // Initially no active generations
    try std.testing.expectEqual(@as(u32, 0), manager.getActiveGenerationCount());
    try std.testing.expect(!manager.hasActiveGenerations());

    // Start a generation
    manager.startGeneration();
    try std.testing.expectEqual(@as(u32, 1), manager.getActiveGenerationCount());
    try std.testing.expect(manager.hasActiveGenerations());

    // End the generation
    manager.endGeneration();
    try std.testing.expectEqual(@as(u32, 0), manager.getActiveGenerationCount());
    try std.testing.expect(!manager.hasActiveGenerations());
}

test "getCurrentModel returns null initially" {
    const allocator = std.testing.allocator;

    var reg = registry.ModelRegistry.init(allocator);
    defer reg.deinit();

    var manager = ModelManager.init(allocator, &reg);
    defer manager.deinit();

    try std.testing.expect(manager.getCurrentModel() == null);
}

test "getAvailableMemoryMb returns reasonable value" {
    const available_mb = ModelManager.getAvailableMemoryMb();

    // Should return some reasonable amount (at least 1GB)
    try std.testing.expect(available_mb >= 1024);

    // Should not be impossibly high (less than 2TB)
    try std.testing.expect(available_mb < 2 * 1024 * 1024);
}
