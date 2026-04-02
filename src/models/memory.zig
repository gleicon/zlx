//! memory.zig - Memory tracking and budget management
//!
//! Tracks memory usage by component (weights, KV cache, temporaries)
//! and provides budget checking for model loading decisions.

const std = @import("std");

/// Memory component categories
pub const MemoryComponent = enum {
    weights,
    kv_cache,
    temporaries,
    overhead,
};

/// GPU memory information
pub const GpuMemoryInfo = struct {
    total_mb: u64,
    free_mb: u64,
    used_mb: u64,
    metal_enabled: bool,

    pub fn init(total: u64, free: u64, metal: bool) GpuMemoryInfo {
        return .{
            .total_mb = total,
            .free_mb = free,
            .used_mb = total - free,
            .metal_enabled = metal,
        };
    }
};

/// Budget status levels
pub const BudgetStatus = enum {
    ok,
    warning,
    critical,
};

/// Memory tracker for component-based memory accounting
pub const MemoryTracker = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    components: [4]u64, // Indexed by MemoryComponent
    peak_mb: u64,
    mutex: std.Thread.Mutex,

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .components = .{ 0, 0, 0, 0 },
            .peak_mb = 0,
            .mutex = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        _ = self;
        // Nothing to clean up
    }

    /// Record a memory allocation (thread-safe)
    pub fn recordAllocation(self: *Self, component: MemoryComponent, bytes: u64) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const mb = bytes / (1024 * 1024);
        self.components[@intFromEnum(component)] += mb;

        // Calculate total directly without calling getTotalUsage (would re-lock)
        var total: u64 = 0;
        for (self.components) |c| {
            total += c;
        }

        if (total > self.peak_mb) {
            self.peak_mb = total;
        }
    }

    /// Record a memory deallocation (thread-safe)
    pub fn recordDeallocation(self: *Self, component: MemoryComponent, bytes: u64) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const mb = bytes / (1024 * 1024);
        const idx = @intFromEnum(component);
        if (self.components[idx] >= mb) {
            self.components[idx] -= mb;
        } else {
            self.components[idx] = 0;
        }
    }

    /// Get usage for a specific component
    pub fn getComponentUsage(self: *Self, component: MemoryComponent) u64 {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.components[@intFromEnum(component)];
    }

    /// Get total memory usage across all components
    pub fn getTotalUsage(self: *Self) u64 {
        self.mutex.lock();
        defer self.mutex.unlock();

        var total: u64 = 0;
        for (self.components) |c| {
            total += c;
        }
        return total;
    }

    /// Get peak memory usage
    pub fn getPeakUsage(self: *Self) u64 {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.peak_mb;
    }

    /// Reset peak tracking
    pub fn resetPeak(self: *Self) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.peak_mb = 0;
    }

    /// Check memory budget status (threshold is 0.0 to 1.0)
    pub fn checkBudget(self: *Self, threshold: f32) BudgetStatus {
        std.debug.assert(threshold > 0.0 and threshold <= 1.0);

        self.mutex.lock();
        defer self.mutex.unlock();

        // Calculate total directly
        var total: u64 = 0;
        for (self.components) |c| {
            total += c;
        }

        const available = getAvailableMemoryMb();

        if (total >= available) return .critical;

        const usage_ratio = @as(f32, @floatFromInt(total)) / @as(f32, @floatFromInt(available));

        if (usage_ratio > threshold) {
            return .warning;
        }
        return .ok;
    }

    /// Get full component breakdown
    pub fn getBreakdown(self: *Self) ComponentBreakdown {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Calculate total directly
        var total: u64 = 0;
        for (self.components) |c| {
            total += c;
        }

        return .{
            .weights_mb = self.components[@intFromEnum(MemoryComponent.weights)],
            .kv_cache_mb = self.components[@intFromEnum(MemoryComponent.kv_cache)],
            .temporaries_mb = self.components[@intFromEnum(MemoryComponent.temporaries)],
            .overhead_mb = self.components[@intFromEnum(MemoryComponent.overhead)],
            .total_mb = total,
            .peak_mb = self.peak_mb,
        };
    }
};

/// Component breakdown structure
pub const ComponentBreakdown = struct {
    weights_mb: u64,
    kv_cache_mb: u64,
    temporaries_mb: u64,
    overhead_mb: u64,
    total_mb: u64,
    peak_mb: u64,
};

/// Get available system memory in MB
pub fn getAvailableMemoryMb() u64 {
    // Use sysctl to get memory info on macOS
    const CTL_HW = 6;
    const HW_MEMSIZE = 24;

    var memsize: u64 = 0;
    var len: usize = @sizeOf(u64);

    const result = std.c.sysctl(
        &[_]c_int{ CTL_HW, HW_MEMSIZE },
        2,
        &memsize,
        &len,
        null,
        0,
    );

    if (result != 0) {
        return 8192; // Fallback to 8GB
    }

    const total_mb = memsize / (1024 * 1024);
    // Estimate 70% available
    return total_mb * 7 / 10;
}

/// Get GPU memory info (macOS with Metal)
pub fn getGpuMemoryInfo() GpuMemoryInfo {
    // On macOS with unified memory, GPU shares system RAM
    const total = getTotalSystemMemoryMb();
    const available = getAvailableMemoryMb();

    return GpuMemoryInfo.init(total, available, true);
}

/// Get total system memory
pub fn getTotalSystemMemoryMb() u64 {
    const CTL_HW = 6;
    const HW_MEMSIZE = 24;

    var memsize: u64 = 0;
    var len: usize = @sizeOf(u64);

    const result = std.c.sysctl(
        &[_]c_int{ CTL_HW, HW_MEMSIZE },
        2,
        &memsize,
        &len,
        null,
        0,
    );

    if (result != 0) {
        return 8192;
    }

    return memsize / (1024 * 1024);
}

/// Estimate weights memory from parameter count and quantization
pub fn estimateWeightsMemory(num_params: u64, quantization_bits: u8) u64 {
    const bytes_per_param: u64 = switch (quantization_bits) {
        8 => 1,
        16 => 2,
        32 => 4,
        else => 2,
    };
    return num_params * bytes_per_param;
}

/// Estimate KV cache memory
pub fn estimateKvCacheMemory(num_layers: u32, hidden_size: u32, seq_len: u32, dtype_bytes: u8) u64 {
    // KV cache: 2 (K and V) * num_layers * hidden_size * seq_len * bytes_per_element
    return 2 * @as(u64, num_layers) * @as(u64, hidden_size) * @as(u64, seq_len) * @as(u64, dtype_bytes);
}

/// Estimate temporary memory for activations
pub fn estimateTemporaryMemory(batch_size: u32, hidden_size: u32, num_layers: u32) u64 {
    // Rough estimate: batch * hidden * layers * 4 bytes * 2 (for intermediate activations)
    return @as(u64, batch_size) * @as(u64, hidden_size) * @as(u64, num_layers) * 8;
}

/// Global tracker instance
var global_tracker: ?*MemoryTracker = null;
var tracker_mutex: std.Thread.Mutex = .{};

/// Initialize global memory tracker
pub fn initGlobalTracker(allocator: std.mem.Allocator) !void {
    tracker_mutex.lock();
    defer tracker_mutex.unlock();

    if (global_tracker != null) return;

    const tracker = try allocator.create(MemoryTracker);
    tracker.* = MemoryTracker.init(allocator);

    global_tracker = tracker;
    std.log.info("Memory tracker initialized", .{});
}

/// Deinitialize global memory tracker
pub fn deinitGlobalTracker(allocator: std.mem.Allocator) void {
    tracker_mutex.lock();
    defer tracker_mutex.unlock();

    if (global_tracker) |tracker| {
        tracker.deinit();
        allocator.destroy(tracker);
        global_tracker = null;
        std.log.info("Memory tracker deinitialized", .{});
    }
}

/// Get global memory tracker
pub fn getGlobalTracker() ?*MemoryTracker {
    tracker_mutex.lock();
    defer tracker_mutex.unlock();
    return global_tracker;
}

// ============================================================================
// Tests
// ============================================================================

test "MemoryTracker recordAllocation updates component" {
    const allocator = std.testing.allocator;

    var tracker = MemoryTracker.init(allocator);
    defer tracker.deinit();

    // Record allocation
    tracker.recordAllocation(.weights, 1024 * 1024 * 1024); // 1GB

    try std.testing.expectEqual(@as(u64, 1024), tracker.getComponentUsage(.weights));
    try std.testing.expectEqual(@as(u64, 1024), tracker.getTotalUsage());
}

test "MemoryTracker recordDeallocation updates component" {
    const allocator = std.testing.allocator;

    var tracker = MemoryTracker.init(allocator);
    defer tracker.deinit();

    // Allocate and then deallocate
    tracker.recordAllocation(.weights, 1024 * 1024 * 1024);
    tracker.recordDeallocation(.weights, 512 * 1024 * 1024); // Free 512MB

    try std.testing.expectEqual(@as(u64, 512), tracker.getComponentUsage(.weights));
}

test "MemoryTracker tracks peak usage" {
    const allocator = std.testing.allocator;

    var tracker = MemoryTracker.init(allocator);
    defer tracker.deinit();

    tracker.recordAllocation(.weights, 1024 * 1024 * 1024);
    try std.testing.expectEqual(@as(u64, 1024), tracker.getPeakUsage());

    tracker.recordDeallocation(.weights, 512 * 1024 * 1024);
    // Peak should remain at 1024
    try std.testing.expectEqual(@as(u64, 1024), tracker.getPeakUsage());
}

test "estimateKvCacheMemory returns reasonable value" {
    // 28 layers, 1536 hidden, 8192 seq len, FP16 (2 bytes)
    const kv_memory = estimateKvCacheMemory(28, 1536, 8192, 2);

    // Should be approximately 1.4GB
    const expected_approx: u64 = 1 * 1024 * 1024 * 1024; // ~1GB minimum
    try std.testing.expect(kv_memory > expected_approx);
    try std.testing.expect(kv_memory < 3 * 1024 * 1024 * 1024); // Less than 3GB
}

test "estimateWeightsMemory calculates correctly" {
    // 1.5B parameters at FP16
    const weights_memory = estimateWeightsMemory(1_500_000_000, 16);

    // Should be ~3GB
    const expected = 1_500_000_000 * 2;
    try std.testing.expectEqual(expected, weights_memory);
}

test "getAvailableMemoryMb returns reasonable value" {
    const available = getAvailableMemoryMb();

    // Should be at least 1GB
    try std.testing.expect(available >= 1024);

    // Should be less than 2TB
    try std.testing.expect(available < 2 * 1024 * 1024);
}
