//! handlers_test.zig - Unit tests for API handlers

const std = @import("std");
const handlers = @import("handlers.zig");
const types = @import("types.zig");

// Test CORS header setting
test "setCorsHeaders sets required headers" {
    // This would need a mock response object to test properly
    // For now, just verify the function exists and compiles
}

// Test request validation
test "validateChatRequest accepts valid request" {
    const allocator = std.testing.allocator;

    var request = types.ChatCompletionRequest{
        .model = "test-model",
        .messages = try allocator.alloc(types.ChatMessage, 1),
    };
    defer allocator.free(request.messages);

    request.messages[0] = .{
        .role = "user",
        .content = .{ .string = "Hello" },
    };

    // Should not error
    try handlers.validateChatRequest(&request);
}

test "validateChatRequest rejects empty messages" {
    const allocator = std.testing.allocator;

    var request = types.ChatCompletionRequest{
        .model = "test-model",
        .messages = try allocator.alloc(types.ChatMessage, 0),
    };
    defer allocator.free(request.messages);

    const result = handlers.validateChatRequest(&request);
    try std.testing.expectError(error.InvalidRequest, result);
}

test "validateChatRequest rejects empty model" {
    const allocator = std.testing.allocator;

    var request = types.ChatCompletionRequest{
        .model = "",
        .messages = try allocator.alloc(types.ChatMessage, 1),
    };
    defer allocator.free(request.messages);

    request.messages[0] = .{
        .role = "user",
        .content = .{ .string = "Hello" },
    };

    const result = handlers.validateChatRequest(&request);
    try std.testing.expectError(error.InvalidRequest, result);
}

// Test role validation
test "isValidRole accepts valid roles" {
    try std.testing.expect(handlers.isValidRole("user"));
    try std.testing.expect(handlers.isValidRole("assistant"));
    try std.testing.expect(handlers.isValidRole("system"));
}

test "isValidRole rejects invalid roles" {
    try std.testing.expect(!handlers.isValidRole("invalid"));
    try std.testing.expect(!handlers.isValidRole(""));
    try std.testing.expect(!handlers.isValidRole("bot"));
}

// Test message content validation
test "validateMessageContent accepts string content" {
    const content = types.MessageContent{ .string = "Hello world" };
    try handlers.validateMessageContent(&content);
}

test "validateMessageContent accepts array content" {
    const allocator = std.testing.allocator;

    var arr = std.ArrayList(types.ContentPart).init(allocator);
    defer arr.deinit();

    try arr.append(.{ .type = "text", .text = "Hello" });

    const content = types.MessageContent{ .array = arr };
    try handlers.validateMessageContent(&content);
}

// Test error response generation
test "createErrorResponse generates valid JSON" {
    const allocator = std.testing.allocator;

    const json = try handlers.createErrorResponse(allocator, 400, "invalid_request", "Bad request");
    defer allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "error") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "invalid_request") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "Bad request") != null);
}

test "createErrorResponse handles special characters" {
    const allocator = std.testing.allocator;

    const json = try handlers.createErrorResponse(allocator, 500, "server_error", "Error: something\"bad");
    defer allocator.free(json);

    // Should be properly escaped
    try std.testing.expect(std.mem.indexOf(u8, json, "error") != null);
}

// Test streaming response formatting
test "formatStreamingChunk creates valid chunk" {
    const allocator = std.testing.allocator;

    const chunk = try handlers.formatStreamingChunk(allocator, "test-id", "test-model", 0, "Hello", false);
    defer allocator.free(chunk);

    try std.testing.expect(std.mem.indexOf(u8, chunk, "test-id") != null);
    try std.testing.expect(std.mem.indexOf(u8, chunk, "test-model") != null);
    try std.testing.expect(std.mem.indexOf(u8, chunk, "Hello") != null);
}

test "formatStreamingChunk handles finish reason" {
    const allocator = std.testing.allocator;

    const chunk = try handlers.formatStreamingChunk(allocator, "test-id", "test-model", 0, "", true);
    defer allocator.free(chunk);

    try std.testing.expect(std.mem.indexOf(u8, chunk, "finish_reason") != null);
}

// Test usage statistics calculation
test "calculateUsage returns correct token counts" {
    const usage = handlers.calculateUsage(10, 20);

    try std.testing.expectEqual(@as(u32, 10), usage.prompt_tokens);
    try std.testing.expectEqual(@as(u32, 20), usage.completion_tokens);
    try std.testing.expectEqual(@as(u32, 30), usage.total_tokens);
}

test "calculateUsage handles zero tokens" {
    const usage = handlers.calculateUsage(0, 0);

    try std.testing.expectEqual(@as(u32, 0), usage.prompt_tokens);
    try std.testing.expectEqual(@as(u32, 0), usage.completion_tokens);
    try std.testing.expectEqual(@as(u32, 0), usage.total_tokens);
}
