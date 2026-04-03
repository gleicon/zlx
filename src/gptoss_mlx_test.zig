//! gptoss_mlx_test.zig - Tests for GPT-OSS MLX implementation

const std = @import("std");
const gptoss_mlx = @import("gptoss_mlx.zig");
const sliding_window = @import("sliding_window_mlx.zig");

const GPTOSSConfig = gptoss_mlx.GPTOSSConfig;
const GPTOSSTransformer = gptoss_mlx.GPTOSSTransformer;
const SlidingWindowConfig = sliding_window.SlidingWindowConfig;
const SlidingKVCache = sliding_window.SlidingKVCache;
const YarnRoPE = sliding_window.YarnRoPE;

test "GPTOSSConfig gptoss20b" {
    const config = GPTOSSConfig.gptoss20b();
    try std.testing.expectEqual(@as(usize, 151936), config.vocab_size);
    try std.testing.expectEqual(@as(usize, 5120), config.hidden_size);
    try std.testing.expectEqual(@as(usize, 40), config.num_hidden_layers);
    try std.testing.expectEqual(@as(usize, 32), config.num_experts);
    try std.testing.expectEqual(@as(usize, 4), config.top_k);
}

test "GPTOSSConfig gptoss120b" {
    const config = GPTOSSConfig.gptoss120b();
    try std.testing.expectEqual(@as(usize, 6656), config.hidden_size);
    try std.testing.expectEqual(@as(usize, 56), config.num_hidden_layers);
    try std.testing.expectEqual(@as(usize, 64), config.num_experts);
    try std.testing.expectEqual(@as(usize, 6), config.top_k);
}

test "YarnRoPE frequency scaling" {
    const rope = YarnRoPE{
        .theta = 1000000.0,
        .factor = 32.0,
        .beta_fast = 32.0,
        .beta_slow = 1.0,
        .original_max_position = 4096,
    };

    const freq_short = rope.getFrequency(0, 1000);
    const freq_long = rope.getFrequency(0, 8192);
    try std.testing.expect(freq_long < freq_short);
}

test "SlidingKVCache initialization" {
    const allocator = std.testing.allocator;
    var cache = try SlidingKVCache.init(allocator, 2, 4, 128, 64);
    defer cache.deinit();
    try std.testing.expectEqual(@as(usize, 128), cache.window_size);
}

test "GPTOSSTransformer initialization" {
    const allocator = std.testing.allocator;
    const mlx = @import("mlx.zig/src/mlx.zig");
    const stream = mlx.newStream(mlx.CPU);
    defer mlx.streamFree(stream);

    const config = GPTOSSConfig.gptoss20b();
    var transformer = try GPTOSSTransformer.init(allocator, config, stream);
    defer transformer.deinit();

    try std.testing.expectEqual(config.vocab_size, transformer.config.vocab_size);
}
