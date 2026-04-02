//! types_test.zig - Unit tests for API types and JSON serialization

const std = @import("std");
const types = @import("types.zig");
const JsonFloat = types.JsonFloat;
const MessageContent = types.MessageContent;
const ChatCompletionRequest = types.ChatCompletionRequest;

// Test JsonFloat parsing from various formats
test "JsonFloat parses from integer" {
    const allocator = std.testing.allocator;
    const json = "0";
    const parsed = try std.json.parseFromSlice(JsonFloat, allocator, json, .{});
    defer parsed.deinit();
    try std.testing.expectEqual(@as(f64, 0.0), parsed.value.value);
}

test "JsonFloat parses from float" {
    const allocator = std.testing.allocator;
    const json = "0.7";
    const parsed = try std.json.parseFromSlice(JsonFloat, allocator, json, .{});
    defer parsed.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 0.7), parsed.value.value, 0.0001);
}

test "JsonFloat parses high temperature" {
    const allocator = std.testing.allocator;
    const json = "2.0";
    const parsed = try std.json.parseFromSlice(JsonFloat, allocator, json, .{});
    defer parsed.deinit();
    try std.testing.expectEqual(@as(f64, 2.0), parsed.value.value);
}

// Test MessageContent parsing
test "MessageContent parses string content" {
    const allocator = std.testing.allocator;
    const json = "\"Hello world\"";
    const parsed = try std.json.parseFromSlice(MessageContent, allocator, json, .{});
    defer parsed.deinit();

    switch (parsed.value) {
        .string => |s| try std.testing.expectEqualStrings("Hello world", s),
        .array => try std.testing.expect(false), // Should not be array
    }
}

test "MessageContent parses array content" {
    const allocator = std.testing.allocator;
    const json = "[{\"type\":\"text\",\"text\":\"Hello\"}]";
    const parsed = try std.json.parseFromSlice(MessageContent, allocator, json, .{});
    defer parsed.deinit();

    switch (parsed.value) {
        .string => try std.testing.expect(false), // Should not be string
        .array => |arr| {
            try std.testing.expectEqual(@as(usize, 1), arr.items.len);
            try std.testing.expectEqualStrings("text", arr.items[0].type);
            try std.testing.expectEqualStrings("Hello", arr.items[0].text);
        },
    }
}

// Test ChatCompletionRequest parsing
test "ChatCompletionRequest parses minimal request" {
    const allocator = std.testing.allocator;
    const json =
        \\{
        \\  "model": "qwen2.5-coder-1.5b",
        \\  "messages": [{"role":"user","content":"Hello"}]
        \\}
    ;

    const parsed = try std.json.parseFromSlice(ChatCompletionRequest, allocator, json, .{});
    defer parsed.deinit();

    try std.testing.expectEqualStrings("qwen2.5-coder-1.5b", parsed.value.model);
    try std.testing.expectEqual(@as(usize, 1), parsed.value.messages.len);
    try std.testing.expectEqualStrings("user", parsed.value.messages[0].role);
}

test "ChatCompletionRequest parses with all options" {
    const allocator = std.testing.allocator;
    const json =
        \\{
        \\  "model": "qwen2.5-coder-7b",
        \\  "messages": [
        \\    {"role":"system","content":"You are a coding assistant"},
        \\    {"role":"user","content":"Write a function"}
        \\  ],
        \\  "max_tokens": 100,
        \\  "temperature": 0.7,
        \\  "stream": true
        \\}
    ;

    const parsed = try std.json.parseFromSlice(ChatCompletionRequest, allocator, json, .{});
    defer parsed.deinit();

    try std.testing.expectEqualStrings("qwen2.5-coder-7b", parsed.value.model);
    try std.testing.expectEqual(@as(usize, 2), parsed.value.messages.len);
    try std.testing.expectEqual(@as(?u32, 100), parsed.value.max_tokens);
    try std.testing.expectApproxEqAbs(@as(f64, 0.7), parsed.value.temperature.?.value, 0.0001);
    try std.testing.expectEqual(@as(?bool, true), parsed.value.stream);
}

test "ChatCompletionRequest default values" {
    const allocator = std.testing.allocator;
    const json =
        \\{
        \\  "model": "test-model",
        \\  "messages": [{"role":"user","content":"Hi"}]
        \\}
    ;

    const parsed = try std.json.parseFromSlice(ChatCompletionRequest, allocator, json, .{});
    defer parsed.deinit();

    // Check defaults
    try std.testing.expectEqual(@as(?u32, null), parsed.value.max_tokens);
    try std.testing.expectEqual(@as(?JsonFloat, null), parsed.value.temperature);
    try std.testing.expectEqual(@as(?bool, null), parsed.value.stream);
}

// Test serialization round-trip
test "ChatCompletionRequest serializes and deserializes" {
    const allocator = std.testing.allocator;

    var request = ChatCompletionRequest{
        .model = "test-model",
        .messages = try allocator.alloc(types.ChatMessage, 1),
    };
    defer allocator.free(request.messages);

    request.messages[0] = .{
        .role = "user",
        .content = .{ .string = "Hello" },
    };

    const json = try std.json.stringifyAlloc(allocator, request, .{});
    defer allocator.free(json);

    // Parse it back
    const parsed = try std.json.parseFromSlice(ChatCompletionRequest, allocator, json, .{});
    defer parsed.deinit();

    try std.testing.expectEqualStrings("test-model", parsed.value.model);
    try std.testing.expectEqual(@as(usize, 1), parsed.value.messages.len);
}
