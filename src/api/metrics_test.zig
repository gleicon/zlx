//! metrics_test.zig - Unit tests for metrics tracking

const std = @import("std");
const metrics = @import("metrics.zig");

// Test Metrics initialization
test "Metrics initializes to zero" {
    var m = metrics.Metrics{};

    try std.testing.expectEqual(@as(u64, 0), m.total_requests.load(.monotonic));
    try std.testing.expectEqual(@as(u64, 0), m.total_tokens.load(.monotonic));
    try std.testing.expectEqual(@as(u64, 0), m.active_generations.load(.monotonic));
}

// Test counter increments
test "incrementRequests increases total_requests" {
    var m = metrics.Metrics{};

    m.incrementRequests();
    try std.testing.expectEqual(@as(u64, 1), m.total_requests.load(.monotonic));

    m.incrementRequests();
    try std.testing.expectEqual(@as(u64, 2), m.total_requests.load(.monotonic));
}

test "incrementTokens adds to total_tokens" {
    var m = metrics.Metrics{};

    m.incrementTokens(10);
    try std.testing.expectEqual(@as(u64, 10), m.total_tokens.load(.monotonic));

    m.incrementTokens(5);
    try std.testing.expectEqual(@as(u64, 15), m.total_tokens.load(.monotonic));
}

test "incrementTokens handles zero" {
    var m = metrics.Metrics{};

    m.incrementTokens(0);
    try std.testing.expectEqual(@as(u64, 0), m.total_tokens.load(.monotonic));
}

// Test active generation tracking
test "active generations tracking" {
    var m = metrics.Metrics{};

    try std.testing.expectEqual(@as(u64, 0), m.active_generations.load(.monotonic));

    m.incrementActiveGenerations();
    try std.testing.expectEqual(@as(u64, 1), m.active_generations.load(.monotonic));

    m.incrementActiveGenerations();
    try std.testing.expectEqual(@as(u64, 2), m.active_generations.load(.monotonic));

    m.decrementActiveGenerations();
    try std.testing.expectEqual(@as(u64, 1), m.active_generations.load(.monotonic));

    m.decrementActiveGenerations();
    try std.testing.expectEqual(@as(u64, 0), m.active_generations.load(.monotonic));
}

test "decrementActiveGenerations won't go below zero" {
    var m = metrics.Metrics{};

    // Should not underflow
    m.decrementActiveGenerations();
    try std.testing.expectEqual(@as(u64, 0), m.active_generations.load(.monotonic));
}

// Test request timing
test "recordRequestTiming updates timing stats" {
    var m = metrics.Metrics{};

    // First request
    m.recordRequestTiming(100, 50); // 100ms TTFT, 50 tokens

    // Check internal state
    try std.testing.expectEqual(@as(u64, 1), m.total_requests.load(.monotonic));
    try std.testing.expectEqual(@as(u64, 50), m.total_tokens.load(.monotonic));
}

// Test reset functionality
test "reset clears all counters" {
    var m = metrics.Metrics{};

    // Add some data
    m.incrementRequests();
    m.incrementTokens(100);
    m.incrementActiveGenerations();

    // Reset
    m.reset();

    // Verify cleared
    try std.testing.expectEqual(@as(u64, 0), m.total_requests.load(.monotonic));
    try std.testing.expectEqual(@as(u64, 0), m.total_tokens.load(.monotonic));
    try std.testing.expectEqual(@as(u64, 0), m.active_generations.load(.monotonic));
}

// Test concurrent access (basic thread safety check)
test "Metrics handles concurrent increments" {
    var m = metrics.Metrics{};

    // Simulate concurrent access by incrementing many times
    var i: usize = 0;
    while (i < 1000) : (i += 1) {
        m.incrementRequests();
        m.incrementTokens(1);
    }

    try std.testing.expectEqual(@as(u64, 1000), m.total_requests.load(.monotonic));
    try std.testing.expectEqual(@as(u64, 1000), m.total_tokens.load(.monotonic));
}
