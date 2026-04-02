const turboquant = @import("turboquant");
const std = @import("std");

test "turboquant import and EngineConfig" {
    const config = turboquant.EngineConfig{ .dim = 1024, .seed = 42 };
    try std.testing.expectEqual(@as(usize, 1024), config.dim);
    try std.testing.expectEqual(@as(u32, 42), config.seed);
}

test "turboquant Engine init/deinit" {
    const allocator = std.testing.allocator;
    const config = turboquant.EngineConfig{ .dim = 1024, .seed = 42 };
    var engine = try turboquant.Engine.init(allocator, config);
    defer engine.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 1024), engine.dim);
}

test "turboquant encode/decode round-trip" {
    const allocator = std.testing.allocator;
    const config = turboquant.EngineConfig{ .dim = 1024, .seed = 42 };
    var engine = try turboquant.Engine.init(allocator, config);
    defer engine.deinit(allocator);

    // Create test data
    const data = try allocator.alloc(f32, 1024);
    defer allocator.free(data);
    for (0..1024) |i| {
        data[i] = @floatFromInt(i % 100);
    }

    // Encode
    const encoded = try engine.encode(allocator, data);
    defer allocator.free(encoded);

    // Decode
    const decoded = try engine.decode(allocator, encoded);
    defer allocator.free(decoded);

    // Verify dimension
    try std.testing.expectEqual(@as(usize, 1024), decoded.len);
}
