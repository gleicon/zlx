//! streaming_test.zig - Unit tests for SSE streaming

const std = @import("std");
const streaming = @import("streaming.zig");

// Test SSE chunk formatting
test "formatSseChunk creates valid SSE format" {
    const allocator = std.testing.allocator;

    const chunk = try streaming.formatSseChunk(allocator, "test data");
    defer allocator.free(chunk);

    try std.testing.expect(std.mem.startsWith(u8, chunk, "data: "));
    try std.testing.expect(std.mem.endsWith(u8, chunk, "\n\n"));
    try std.testing.expect(std.mem.indexOf(u8, chunk, "test data") != null);
}

test "formatSseChunk handles empty data" {
    const allocator = std.testing.allocator;

    const chunk = try streaming.formatSseChunk(allocator, "");
    defer allocator.free(chunk);

    try std.testing.expectEqualStrings("data: \n\n", chunk);
}

test "formatSseChunk handles JSON data" {
    const allocator = std.testing.allocator;

    const json = "{\"id\":\"1\",\"object\":\"chat.completion.chunk\"}";
    const chunk = try streaming.formatSseChunk(allocator, json);
    defer allocator.free(chunk);

    try std.testing.expect(std.mem.indexOf(u8, chunk, json) != null);
}

// Test SSE termination
test "formatSseTermination creates [DONE] marker" {
    const allocator = std.testing.allocator;

    const chunk = try streaming.formatSseTermination(allocator);
    defer allocator.free(chunk);

    try std.testing.expectEqualStrings("data: [DONE]\n\n", chunk);
}

// Test UTF-8 boundary handling
test "findLastValidUtf8Boundary finds valid boundary" {
    // "Hello" in ASCII - all bytes are valid UTF-8 boundaries
    const text = "Hello";
    const boundary = streaming.findLastValidUtf8Boundary(text, text.len);
    try std.testing.expectEqual(@as(usize, 5), boundary);
}

test "findLastValidUtf8Boundary handles multi-byte characters" {
    // "Hello 世界" - "世" is 3 bytes (E4 B8 96), "界" is 3 bytes (E7 95 8C)
    const text = "Hello 世界";

    // If we try to split at byte 8 (middle of "世"), should back up to 6
    const boundary = streaming.findLastValidUtf8Boundary(text, 8);
    try std.testing.expectEqual(@as(usize, 6), boundary); // After "Hello "
}

test "findLastValidUtf8Boundary handles empty string" {
    const text = "";
    const boundary = streaming.findLastValidUtf8Boundary(text, 0);
    try std.testing.expectEqual(@as(usize, 0), boundary);
}

test "findLastValidUtf8Boundary handles single multi-byte character" {
    // Just "世" (3 bytes)
    const text = "世";

    const boundary = streaming.findLastValidUtf8Boundary(text, 1);
    try std.testing.expectEqual(@as(usize, 0), boundary);

    const boundary2 = streaming.findLastValidUtf8Boundary(text, 2);
    try std.testing.expectEqual(@as(usize, 0), boundary2);

    const boundary3 = streaming.findLastValidUtf8Boundary(text, 3);
    try std.testing.expectEqual(@as(usize, 3), boundary3);
}

// Test special token stripping
test "stripSpecialTokens removes end tokens" {
    const allocator = std.testing.allocator;

    const text = "Hello world<|endoftext|>";
    const stripped = try streaming.stripSpecialTokens(allocator, text);
    defer allocator.free(stripped);

    try std.testing.expectEqualStrings("Hello world", stripped);
}

test "stripSpecialTokens removes multiple tokens" {
    const allocator = std.testing.allocator;

    const text = "<|im_start|>Hello<|im_end|> world<|endoftext|>";
    const stripped = try streaming.stripSpecialTokens(allocator, text);
    defer allocator.free(stripped);

    try std.testing.expectEqualStrings("Hello world", stripped);
}

test "stripSpecialTokens handles no special tokens" {
    const allocator = std.testing.allocator;

    const text = "Just regular text";
    const stripped = try streaming.stripSpecialTokens(allocator, text);
    defer allocator.free(stripped);

    try std.testing.expectEqualStrings("Just regular text", stripped);
}

test "stripSpecialTokens handles empty string" {
    const allocator = std.testing.allocator;

    const text = "";
    const stripped = try streaming.stripSpecialTokens(allocator, text);
    defer allocator.free(stripped);

    try std.testing.expectEqualStrings("", stripped);
}

// Test streaming buffer operations
test "StreamingBuffer accumulates tokens" {
    const allocator = std.testing.allocator;

    var buffer = streaming.StreamingBuffer.init(allocator, 1024);
    defer buffer.deinit();

    try buffer.append("Hello");
    try buffer.append(" ");
    try buffer.append("world");

    const content = buffer.getContent();
    try std.testing.expectEqualStrings("Hello world", content);
}

test "StreamingBuffer handles UTF-8 correctly" {
    const allocator = std.testing.allocator;

    var buffer = streaming.StreamingBuffer.init(allocator, 1024);
    defer buffer.deinit();

    try buffer.append("Hello ");
    try buffer.append("世界");

    const content = buffer.getContent();
    try std.testing.expectEqualStrings("Hello 世界", content);
}
