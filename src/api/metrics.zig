//! metrics.zig - Performance metrics tracking and logging
//!
//! Tracks tokens/sec, TTFT, memory usage, and request statistics.
//! Logs periodically to show server performance.

const std = @import("std");

/// Global metrics state
var g_metrics: ?Metrics = null;
var g_mutex: std.Thread.Mutex = .{};

/// Performance metrics structure
pub const Metrics = struct {
    /// Total requests processed
    total_requests: u64 = 0,
    /// Total tokens generated
    total_tokens: u64 = 0,
    /// Total prompt tokens processed
    total_prompt_tokens: u64 = 0,
    /// Total generation time in milliseconds
    total_generation_time_ms: u64 = 0,
    /// Number of active generations
    active_generations: u32 = 0,
    /// Time to first token accumulator (microseconds)
    ttft_sum_us: u64 = 0,
    /// Count of generations for TTFT average
    ttft_count: u64 = 0,
    /// Peak memory usage (if available)
    peak_memory_mb: u64 = 0,

    /// Allocator for metrics
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Metrics {
        return .{ .allocator = allocator };
    }

    /// Record a completed request
    pub fn recordRequest(self: *Metrics, prompt_tokens: u32, completion_tokens: u32, generation_time_ms: u64, ttft_us: u64) void {
        g_mutex.lock();
        defer g_mutex.unlock();

        self.total_requests += 1;
        self.total_prompt_tokens += prompt_tokens;
        self.total_tokens += completion_tokens;
        self.total_generation_time_ms += generation_time_ms;

        if (ttft_us > 0) {
            self.ttft_sum_us += ttft_us;
            self.ttft_count += 1;
        }

        self.active_generations = if (self.active_generations > 0) self.active_generations - 1 else 0;
    }

    /// Mark start of generation
    pub fn startGeneration(self: *Metrics) void {
        g_mutex.lock();
        defer g_mutex.unlock();
        self.active_generations += 1;
    }

    /// Get current stats snapshot
    pub fn getStats(self: *Metrics) Stats {
        g_mutex.lock();
        defer g_mutex.unlock();

        const avg_tokens_per_sec = if (self.total_generation_time_ms > 0)
            @as(f64, @floatFromInt(self.total_tokens)) / (@as(f64, @floatFromInt(self.total_generation_time_ms)) / 1000.0)
        else
            0;

        const avg_ttft_ms = if (self.ttft_count > 0)
            @as(f64, @floatFromInt(self.ttft_sum_us)) / @as(f64, @floatFromInt(self.ttft_count)) / 1000.0
        else
            0;

        return .{
            .total_requests = self.total_requests,
            .total_tokens = self.total_tokens,
            .avg_tokens_per_sec = avg_tokens_per_sec,
            .avg_ttft_ms = avg_ttft_ms,
            .active_generations = self.active_generations,
        };
    }
};

/// Stats snapshot for logging
pub const Stats = struct {
    total_requests: u64,
    total_tokens: u64,
    avg_tokens_per_sec: f64,
    avg_ttft_ms: f64,
    active_generations: u32,
};

/// Initialize global metrics
pub fn initMetrics(allocator: std.mem.Allocator) void {
    g_mutex.lock();
    defer g_mutex.unlock();
    g_metrics = Metrics.init(allocator);
}

/// Get global metrics
pub fn getMetrics() ?*Metrics {
    g_mutex.lock();
    defer g_mutex.unlock();
    return if (g_metrics) |*m| m else null;
}

/// Record a request completion
pub fn recordRequest(prompt_tokens: u32, completion_tokens: u32, generation_time_ms: u64, ttft_us: u64) void {
    if (getMetrics()) |m| {
        m.recordRequest(prompt_tokens, completion_tokens, generation_time_ms, ttft_us);
    }
}

/// Mark start of generation
pub fn startGeneration() void {
    if (getMetrics()) |m| {
        m.startGeneration();
    }
}

/// Start periodic stats logging
pub fn startStatsLogging(allocator: std.mem.Allocator, interval_seconds: u64) !std.Thread {
    return try std.Thread.spawn(.{}, statsLogger, .{ allocator, interval_seconds });
}

/// Stats logging thread
fn statsLogger(_: std.mem.Allocator, interval_seconds: u64) void {
    var last_requests: u64 = 0;
    var last_tokens: u64 = 0;
    var last_time = std.time.milliTimestamp();

    while (true) {
        std.Thread.sleep(interval_seconds * std.time.ns_per_s);

        if (getMetrics()) |m| {
            const stats = m.getStats();
            const now = std.time.milliTimestamp();
            const elapsed_ms = now - last_time;

            // Calculate interval stats
            const requests_delta = stats.total_requests - last_requests;
            const tokens_delta = stats.total_tokens - last_tokens;
            const interval_tps = if (elapsed_ms > 0)
                @as(f64, @floatFromInt(tokens_delta)) / (@as(f64, @floatFromInt(elapsed_ms)) / 1000.0)
            else
                0;

            // Format and log
            var buf: [512]u8 = undefined;
            const msg = std.fmt.bufPrint(&buf, "[STATS] Requests: {d} (+{d}) | Tokens: {d} (+{d}) | Avg TPS: {d:.1} | Interval TPS: {d:.1} | Avg TTFT: {d:.1}ms | Active: {d}", .{
                stats.total_requests,
                requests_delta,
                stats.total_tokens,
                tokens_delta,
                stats.avg_tokens_per_sec,
                interval_tps,
                stats.avg_ttft_ms,
                stats.active_generations,
            }) catch |err| {
                std.log.err("Failed to format stats: {s}", .{@errorName(err)});
                continue;
            };

            std.log.info("{s}", .{msg});

            last_requests = stats.total_requests;
            last_tokens = stats.total_tokens;
            last_time = now;
        }
    }
}
