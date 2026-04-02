//! templates.zig - Chat prompt templates for different model architectures
//!
//! Provides model-specific prompt formatting for OpenAI-compatible chat completions.
//! Each architecture may have unique formatting requirements.

const std = @import("std");
const types = @import("../api/types.zig");

/// Supported model architectures for template selection
pub const ModelArchitecture = enum {
    qwen,
    llama,
    phi,
    deepseek_v2_moe,
    unknown,
};

/// Format messages using DeepSeek chat template
/// Format: System prompt at top, then "User: " and "Assistant: " prefixes
/// Adds trailing "Assistant: " for generation
pub fn formatDeepSeekChat(allocator: std.mem.Allocator, messages: []const types.Message) ![]const u8 {
    var result = std.ArrayList(u8).init(allocator);
    defer result.deinit();

    for (messages) |msg| {
        switch (msg.role) {
            .system, .developer => {
                // System prompt at top, no special marker
                try result.appendSlice(msg.content.text);
                try result.append('\n');
            },
            .user => {
                try result.appendSlice("User: ");
                try result.appendSlice(msg.content.text);
                try result.append('\n');
            },
            .assistant => {
                try result.appendSlice("Assistant: ");
                try result.appendSlice(msg.content.text);
                try result.append('\n');
            },
            .tool => {
                // Tool messages treated as system context
                try result.appendSlice("System: Tool result: ");
                try result.appendSlice(msg.content.text);
                try result.append('\n');
            },
        }
    }

    // Add assistant prefix for generation
    try result.appendSlice("Assistant: ");

    return result.toOwnedSlice();
}

/// Format messages using Qwen chat template (ChatML format)
/// Uses <|im_start|>system/user/assistant<|im_end|> markers
pub fn formatQwenChat(allocator: std.mem.Allocator, messages: []const types.Message) ![]const u8 {
    var result = std.ArrayList(u8).init(allocator);
    defer result.deinit();

    for (messages) |msg| {
        switch (msg.role) {
            .system, .developer => {
                try result.appendSlice("<|im_start|>system\n");
                try result.appendSlice(msg.content.text);
                try result.appendSlice("<|im_end|>\n");
            },
            .user => {
                try result.appendSlice("<|im_start|>user\n");
                try result.appendSlice(msg.content.text);
                try result.appendSlice("<|im_end|>\n");
            },
            .assistant => {
                try result.appendSlice("<|im_start|>assistant\n");
                try result.appendSlice(msg.content.text);
                try result.appendSlice("<|im_end|>\n");
            },
            .tool => {
                try result.appendSlice("<|im_start|>system\nTool result: ");
                try result.appendSlice(msg.content.text);
                try result.appendSlice("<|im_end|>\n");
            },
        }
    }

    // Add final assistant prefix
    try result.appendSlice("<|im_start|>assistant\n");

    return result.toOwnedSlice();
}

/// Format messages using Llama chat template
/// Uses [INST] and <<SYS>> markers
pub fn formatLlamaChat(allocator: std.mem.Allocator, messages: []const types.Message) ![]const u8 {
    var result = std.ArrayList(u8).init(allocator);
    defer result.deinit();

    // Find system message if present
    var system_prompt: ?[]const u8 = null;
    for (messages) |msg| {
        if (msg.role == .system or msg.role == .developer) {
            system_prompt = msg.content.text;
            break;
        }
    }

    // Build conversation
    var first_user = true;
    for (messages) |msg| {
        switch (msg.role) {
            .system, .developer => {
                // Skip - handled above
                continue;
            },
            .user => {
                if (first_user and system_prompt != null) {
                    // First user message with system prompt
                    try result.appendSlice("[INST] <<SYS>>\n");
                    try result.appendSlice(system_prompt.?);
                    try result.appendSlice("\n<</SYS>>\n\n");
                    try result.appendSlice(msg.content.text);
                    try result.appendSlice(" [/INST]");
                    first_user = false;
                } else {
                    try result.appendSlice("[INST] ");
                    try result.appendSlice(msg.content.text);
                    try result.appendSlice(" [/INST]");
                }
            },
            .assistant => {
                try result.appendSlice(msg.content.text);
            },
            .tool => {
                // Tool results appended to context
                try result.appendSlice("\n[TOOL RESULT] ");
                try result.appendSlice(msg.content.text);
            },
        }
    }

    // Add assistant marker for generation if last message was from user
    const last_was_user = messages.len > 0 and messages[messages.len - 1].role == .user;
    if (last_was_user) {
        try result.appendSlice(" ");
    }

    return result.toOwnedSlice();
}

/// Select and apply appropriate template based on model architecture
pub fn formatChatByArchitecture(
    allocator: std.mem.Allocator,
    arch: ModelArchitecture,
    messages: []const types.Message,
) ![]const u8 {
    return switch (arch) {
        .deepseek_v2_moe => formatDeepSeekChat(allocator, messages),
        .qwen => formatQwenChat(allocator, messages),
        .llama => formatLlamaChat(allocator, messages),
        .phi, .unknown => formatQwenChat(allocator, messages), // Default to Qwen format
    };
}

// ============================================================================
// Tests
// ============================================================================

test "formatDeepSeekChat - system prompt at top" {
    const allocator = std.testing.allocator;

    const messages = &[_]types.Message{
        .{ .role = .system, .content = .{ .text = "You are a coding assistant" } },
        .{ .role = .user, .content = .{ .text = "Write a Python function" } },
    };

    const prompt = try formatDeepSeekChat(allocator, messages);
    defer allocator.free(prompt);

    // System prompt should be at top without special marker
    try std.testing.expect(std.mem.startsWith(u8, prompt, "You are a coding assistant\n"));
    // User message should have "User: " prefix
    try std.testing.expect(std.mem.indexOf(u8, prompt, "User: Write a Python function") != null);
    // Should end with "Assistant: " for generation
    try std.testing.expect(std.mem.endsWith(u8, prompt, "Assistant: "));
}

test "formatDeepSeekChat - user assistant alternation" {
    const allocator = std.testing.allocator;

    const messages = &[_]types.Message{
        .{ .role = .user, .content = .{ .text = "Hello" } },
        .{ .role = .assistant, .content = .{ .text = "Hi there!" } },
        .{ .role = .user, .content = .{ .text = "How are you?" } },
    };

    const prompt = try formatDeepSeekChat(allocator, messages);
    defer allocator.free(prompt);

    // Check user prefixes
    try std.testing.expect(std.mem.indexOf(u8, prompt, "User: Hello") != null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "User: How are you?") != null);

    // Check assistant prefix
    try std.testing.expect(std.mem.indexOf(u8, prompt, "Assistant: Hi there!") != null);

    // Should end with "Assistant: " for generation
    try std.testing.expect(std.mem.endsWith(u8, prompt, "Assistant: "));
}

test "formatDeepSeekChat - tool message handling" {
    const allocator = std.testing.allocator;

    const messages = &[_]types.Message{
        .{ .role = .user, .content = .{ .text = "Search for something" } },
        .{ .role = .tool, .content = .{ .text = "Found 3 results" } },
        .{ .role = .user, .content = .{ .text = "Show me" } },
    };

    const prompt = try formatDeepSeekChat(allocator, messages);
    defer allocator.free(prompt);

    // Tool result should be formatted as system context
    try std.testing.expect(std.mem.indexOf(u8, prompt, "System: Tool result: Found 3 results") != null);
}

test "formatQwenChat - ChatML format" {
    const allocator = std.testing.allocator;

    const messages = &[_]types.Message{
        .{ .role = .system, .content = .{ .text = "You are helpful." } },
        .{ .role = .user, .content = .{ .text = "Hello!" } },
    };

    const prompt = try formatQwenChat(allocator, messages);
    defer allocator.free(prompt);

    // Should use ChatML format
    try std.testing.expect(std.mem.indexOf(u8, prompt, "<|im_start|>system") != null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "<|im_start|>user") != null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "<|im_end|>") != null);
}

test "formatLlamaChat - INST format" {
    const allocator = std.testing.allocator;

    const messages = &[_]types.Message{
        .{ .role = .system, .content = .{ .text = "Be helpful." } },
        .{ .role = .user, .content = .{ .text = "Hello!" } },
    };

    const prompt = try formatLlamaChat(allocator, messages);
    defer allocator.free(prompt);

    // Should use Llama instruct format
    try std.testing.expect(std.mem.indexOf(u8, prompt, "[INST]") != null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "[/INST]") != null);
}

test "formatChatByArchitecture - dispatches to correct template" {
    const allocator = std.testing.allocator;

    const messages = &[_]types.Message{
        .{ .role = .user, .content = .{ .text = "Hello" } },
    };

    // Test DeepSeek template dispatch
    const deepseek_prompt = try formatChatByArchitecture(allocator, .deepseek_v2_moe, messages);
    defer allocator.free(deepseek_prompt);
    try std.testing.expect(std.mem.startsWith(u8, deepseek_prompt, "User: "));

    // Test Qwen template dispatch
    const qwen_prompt = try formatChatByArchitecture(allocator, .qwen, messages);
    defer allocator.free(qwen_prompt);
    try std.testing.expect(std.mem.startsWith(u8, qwen_prompt, "<|im_start|>user"));
}
