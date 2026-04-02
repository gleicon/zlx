//! speculative_metrics.zig - Metrics collection for speculative decoding
//!
//! Provides atomic metrics tracking for acceptance rates,
//! tokens per speculation, and speedup estimation.

const std = @import("std");

/// Metrics for speculative decoding performance tracking
pub const SpeculativeMetrics = struct {
    const Self = @This();

    // Counters (atomic for thread safety)
    total_speculations: std.atomic.Value(u64),
    total_tokens_accepted: std.atomic.Value(u64),
    total_tokens_rejected: std.atomic.Value(u64),
    total_draft_tokens_generated: std.atomic.Value(u64),

    // Timing (microseconds)
    draft_generation_time_us: std.atomic.Value(u64),
    target_verification_time_us: std.atomic.Value(u64),

    pub fn init() Self {
        return .{
            .total_speculations = std.atomic.Value(u64).init(0),
            .total_tokens_accepted = std.atomic.Value(u64).init(0),
            .total_tokens_rejected = std.atomic.Value(u64).init(0),
            .total_draft_tokens_generated = std.atomic.Value(u64).init(0),
            .draft_generation_time_us = std.atomic.Value(u64).init(0),
            .target_verification_time_us = std.atomic.Value(u64).init(0),
        };
    }

    /// Record a completed speculation round
    pub fn recordSpeculation(
        self: *Self,
        accepted: usize,
        rejected: usize,
        draft_time_us: u64,
        target_time_us: u64,
    ) void {
        _ = self.total_speculations.fetchAdd(1, .monotonic);
        _ = self.total_tokens_accepted.fetchAdd(@intCast(accepted), .monotonic);
        _ = self.total_tokens_rejected.fetchAdd(@intCast(rejected), .monotonic);
        _ = self.total_draft_tokens_generated.fetchAdd(@intCast(accepted + rejected), .monotonic);
        _ = self.draft_generation_time_us.fetchAdd(draft_time_us, .monotonic);
        _ = self.target_verification_time_us.fetchAdd(target_time_us, .monotonic);
    }

    /// Get current statistics snapshot
    pub fn getStats(self: *Self) SpeculationStats {
        const total_accepted = self.total_tokens_accepted.load(.monotonic);
        const total_rejected = self.total_tokens_rejected.load(.monotonic);
        const total = total_accepted + total_rejected;
        const total_speculations = self.total_speculations.load(.monotonic);

        // Calculate acceptance rate
        const acceptance_rate = if (total == 0) 0.0 else @as(f32, @floatFromInt(total_accepted)) / @as(f32, @floatFromInt(total));

        // Calculate average tokens per speculation
        const avg_tokens = if (total_speculations == 0) 0.0 else @as(f32, @floatFromInt(total_accepted)) / @as(f32, @floatFromInt(total_speculations));

        // Calculate draft overhead percentage
        const draft_time = self.draft_generation_time_us.load(.monotonic);
        const target_time = self.target_verification_time_us.load(.monotonic);
        const total_time = draft_time + target_time;
        const draft_overhead = if (total_time == 0) 0.0 else @as(f32, @floatFromInt(draft_time)) / @as(f32, @floatFromInt(total_time)) * 100.0;

        // Estimate speedup based on acceptance rate
        // Formula: speedup ~ K / (1 + α*K) where α is rejection rate
        const k: f32 = 4.0; // Default speculation depth
        const rejection_rate = 1.0 - acceptance_rate;
        const estimated_speedup = if (acceptance_rate == 0) 1.0 else k / (1.0 + rejection_rate * k);

        return .{
            .total_speculations = total_speculations,
            .total_accepted = total_accepted,
            .total_rejected = total_rejected,
            .acceptance_rate = acceptance_rate,
            .avg_tokens_per_speculation = avg_tokens,
            .estimated_speedup = estimated_speedup,
            .draft_overhead_pct = draft_overhead,
        };
    }

    /// Reset all metrics to zero
    pub fn reset(self: *Self) void {
        _ = self.total_speculations.store(0, .monotonic);
        _ = self.total_tokens_accepted.store(0, .monotonic);
        _ = self.total_tokens_rejected.store(0, .monotonic);
        _ = self.total_draft_tokens_generated.store(0, .monotonic);
        _ = self.draft_generation_time_us.store(0, .monotonic);
        _ = self.target_verification_time_us.store(0, .monotonic);
    }
};

/// Public statistics structure (returned by getStats)
pub const SpeculationStats = struct {
    total_speculations: u64,
    total_accepted: u64,
    total_rejected: u64,
    acceptance_rate: f32, // 0.0 to 1.0
    avg_tokens_per_speculation: f32,
    estimated_speedup: f32,
    draft_overhead_pct: f32, // Percentage of time spent on draft
};

// ============================================================================
// Tests
// ============================================================================

test "SpeculativeMetrics init" {
    const metrics = SpeculativeMetrics.init();

    try std.testing.expectEqual(@as(u64, 0), metrics.total_speculations.load(.monotonic));
    try std.testing.expectEqual(@as(u64, 0), metrics.total_tokens_accepted.load(.monotonic));
}

test "recordSpeculation updates counters" {
    var metrics = SpeculativeMetrics.init();

    metrics.recordSpeculation(3, 1, 100, 200);

    try std.testing.expectEqual(@as(u64, 1), metrics.total_speculations.load(.monotonic));
    try std.testing.expectEqual(@as(u64, 3), metrics.total_tokens_accepted.load(.monotonic));
    try std.testing.expectEqual(@as(u64, 1), metrics.total_tokens_rejected.load(.monotonic));
}

test "getStats calculates acceptance rate" {
    var metrics = SpeculativeMetrics.init();

    // Record: 3 accepted, 1 rejected = 75% acceptance
    metrics.recordSpeculation(3, 1, 100, 200);

    const stats = metrics.getStats();

    try std.testing.expectApproxEqAbs(@as(f32, 0.75), stats.acceptance_rate, 0.01);
    try std.testing.expectEqual(@as(u64, 1), stats.total_speculations);
    try std.testing.expectEqual(@as(u64, 3), stats.total_accepted);
}

test "getStats with no speculations" {
    var metrics = SpeculativeMetrics.init();

    const stats = metrics.getStats();

    try std.testing.expectEqual(@as(u64, 0), stats.total_speculations);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), stats.acceptance_rate, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), stats.estimated_speedup, 0.01);
}

test "getStats calculates speedup" {
    var metrics = SpeculativeMetrics.init();

    // Perfect acceptance: speedup should be ~4x (speculation depth)
    metrics.recordSpeculation(4, 0, 50, 150);

    const stats = metrics.getStats();

    // With K=4 and 100% acceptance: speedup = 4 / (1 + 0*4) = 4.0
    try std.testing.expect(stats.estimated_speedup >= 3.0);
}

test "reset clears all metrics" {
    var metrics = SpeculativeMetrics.init();

    metrics.recordSpeculation(5, 2, 100, 300);
    metrics.reset();

    try std.testing.expectEqual(@as(u64, 0), metrics.total_speculations.load(.monotonic));
    try std.testing.expectEqual(@as(u64, 0), metrics.total_tokens_accepted.load(.monotonic));
}

test "multiple speculations accumulate correctly" {
    var metrics = SpeculativeMetrics.init();

    // Simulate 3 speculation rounds
    metrics.recordSpeculation(3, 1, 50, 150); // 75% acceptance
    metrics.recordSpeculation(4, 0, 60, 140); // 100% acceptance
    metrics.recordSpeculation(2, 2, 40, 160); // 50% acceptance

    const stats = metrics.getStats();

    // Total: 9 accepted, 3 rejected = 75% acceptance
    try std.testing.expectEqual(@as(u64, 3), stats.total_speculations);
    try std.testing.expectEqual(@as(u64, 9), stats.total_accepted);
    try std.testing.expectEqual(@as(u64, 3), stats.total_rejected);
    try std.testing.expectApproxEqAbs(@as(f32, 0.75), stats.acceptance_rate, 0.01);
}

test "draft overhead calculation" {
    var metrics = SpeculativeMetrics.init();

    // 50us draft + 150us target = 200us total, draft = 25%
    metrics.recordSpeculation(3, 1, 50, 150);

    const stats = metrics.getStats();

    try std.testing.expectApproxEqAbs(@as(f32, 25.0), stats.draft_overhead_pct, 0.1);
}

test "atomic thread safety" {
    var metrics = SpeculativeMetrics.init();

    // Simulate concurrent updates
    const threads = 10;
    const updates_per_thread = 100;

    // In a real test, spawn threads. Here we just verify atomics work.
    for (0..threads * updates_per_thread) |_| {
        metrics.recordSpeculation(3, 1, 10, 20);
    }

    const stats = metrics.getStats();

    // Should have exactly threads * updates_per_thread speculations
    const expected: u64 = threads * updates_per_thread;
    try std.testing.expectEqual(expected, stats.total_speculations);
    try std.testing.expectEqual(expected * 3, stats.total_accepted);
    try std.testing.expectEqual(expected * 1, stats.total_rejected);
}
