//! harmony_test.zig - Tests for Harmony format implementation

const std = @import("std");
const harmony = @import("harmony.zig");
const parser = @import("parser.zig");
const template = @import("template.zig");

const HarmonyMessage = harmony.HarmonyMessage;
const HarmonyConversation = harmony.HarmonyConversation;
const HarmonyRole = harmony.HarmonyRole;
const ToolCall = harmony.ToolCall;
const ToolDefinition = harmony.ToolDefinition;
const OpenAIMessage = harmony.OpenAIMessage;
const ReasoningEffort = harmony.ReasoningEffort;
const HarmonyTemplate = template.HarmonyTemplate;
const HarmonyParser = parser.HarmonyParser;

// ============================================================================
// Harmony Types Tests
// ============================================================================

test "HarmonyMessage text creation" {
    const allocator = std.testing.allocator;

    var msg = try HarmonyMessage.initText(allocator, .user, "Hello!");
    defer msg.deinit(allocator);

    try std.testing.expectEqual(.user, msg.role);
    try std.testing.expectEqualStrings("Hello!", msg.content.text);
}

test "HarmonyMessage tool call creation" {
    const allocator = std.testing.allocator;

    var msg = try HarmonyMessage.initToolCall(
        allocator,
        "browser",
        "{\"url\": \"https://example.com\"}",
        "browser",
    );
    defer msg.deinit(allocator);

    try std.testing.expectEqual(.assistant, msg.role);
    try std.testing.expectEqualStrings("browser", msg.content.tool_call.name);
    try std.testing.expectEqualStrings("browser", msg.recipient.?);
}

test "HarmonyConversation memory management" {
    const allocator = std.testing.allocator;

    var conv = HarmonyConversation.init(allocator);
    defer conv.deinit();

    var msg1 = try HarmonyMessage.initText(allocator, .user, "Hello!");
    try conv.addMessage(msg1);

    var msg2 = try HarmonyMessage.initText(allocator, .assistant, "Hi there!");
    try conv.addMessage(msg2);

    try std.testing.expectEqual(@as(usize, 2), conv.messages.items.len);
}

test "HarmonyConversation hasToolCalls detection" {
    const allocator = std.testing.allocator;

    var conv = HarmonyConversation.init(allocator);
    defer conv.deinit();

    // Add regular message
    var msg1 = try HarmonyMessage.initText(allocator, .user, "Hello!");
    try conv.addMessage(msg1);

    try std.testing.expect(!conv.hasToolCalls());

    // Add tool call
    var msg2 = try HarmonyMessage.initToolCall(
        allocator,
        "browser",
        "{}",
        "browser",
    );
    try conv.addMessage(msg2);

    try std.testing.expect(conv.hasToolCalls());
}

// ============================================================================
// Parser Tests
// ============================================================================

test "Parse simple user message" {
    const allocator = std.testing.allocator;

    const harmony_text = "<|user|>\nHello!\n<|/user|>";

    var p = HarmonyParser.init(allocator);
    var conv = try p.parse(harmony_text);
    defer conv.deinit();

    try std.testing.expectEqual(@as(usize, 1), conv.messages.items.len);
    try std.testing.expectEqual(.user, conv.messages.items[0].role);
}

test "Parse conversation with multiple messages" {
    const allocator = std.testing.allocator;

    const harmony_text =
        "<|user|>\n" ++
        "What's the weather?\n" ++
        "<|/user|>\n" ++
        "<|assistant|>\n" ++
        "I need to check.\n" ++
        "<|/assistant|>";

    var p = HarmonyParser.init(allocator);
    var conv = try p.parse(harmony_text);
    defer conv.deinit();

    try std.testing.expectEqual(@as(usize, 2), conv.messages.items.len);
    try std.testing.expectEqual(.user, conv.messages.items[0].role);
    try std.testing.expectEqual(.assistant, conv.messages.items[1].role);
}

test "Parse tool call with recipient" {
    const allocator = std.testing.allocator;

    const harmony_text =
        "<|recipient|>browser<|/recipient|>\n" ++
        "<|tool_call|>\n" ++
        "{\"url\": \"https://example.com\"}\n" ++
        "<|/tool_call|>";

    var p = HarmonyParser.init(allocator);
    var conv = try p.parse(harmony_text);
    defer conv.deinit();

    try std.testing.expect(conv.hasToolCalls());
}

test "Extract tool calls from text" {
    const allocator = std.testing.allocator;

    const text_with_tools =
        "Let me search for that.\n" ++
        "<|recipient|>browser<|/recipient|>\n" ++
        "<|tool_call|>{\"action\": \"search\", \"query\": \"test\"}<|/tool_call|>";

    var p = HarmonyParser.init(allocator);
    const tool_calls = try p.extractToolCalls(text_with_tools);

    defer {
        for (tool_calls) |*tc| tc.deinit(allocator);
        allocator.free(tool_calls);
    }

    try std.testing.expectEqual(@as(usize, 1), tool_calls.len);
    try std.testing.expectEqualStrings("browser", tool_calls[0].name);
}

// ============================================================================
// Template Tests
// ============================================================================

test "Format simple user message to Harmony" {
    const allocator = std.testing.allocator;

    const messages = &[_]OpenAIMessage{
        .{
            .role = "user",
            .content = "Hello!",
        },
    };

    var tmpl = HarmonyTemplate.init(
        allocator,
        "harmony_gpt_oss",
        .low,
        false,
        null,
    );

    const harmony = try tmpl.formatHarmonyChat(messages, null);
    defer allocator.free(harmony);

    try std.testing.expect(std.mem.contains(u8, harmony, "<|user|>"));
    try std.testing.expect(std.mem.contains(u8, harmony, "Hello!"));
    try std.testing.expect(std.mem.contains(u8, harmony, "<|/user|>"));
}

test "Format conversation with system prompt" {
    const allocator = std.testing.allocator;

    const messages = &[_]OpenAIMessage{
        .{
            .role = "system",
            .content = "You are helpful.",
        },
        .{
            .role = "user",
            .content = "Hello!",
        },
    };

    var tmpl = HarmonyTemplate.init(
        allocator,
        "harmony_gpt_oss",
        .low,
        false,
        null,
    );

    const harmony = try tmpl.formatHarmonyChat(messages, null);
    defer allocator.free(harmony);

    try std.testing.expect(std.mem.contains(u8, harmony, "<|system|>"));
    try std.testing.expect(std.mem.contains(u8, harmony, "You are helpful."));
}

test "Format with tool definitions" {
    const allocator = std.testing.allocator;

    const tool_schema = std.json.Value{ .object = std.json.ObjectMap.init(allocator) };
    defer tool_schema.object.deinit();

    const tools = &[_]ToolDefinition{
        .{
            .name = "browser",
            .description = "Search and browse the web",
            .parameters = tool_schema,
        },
    };

    var tmpl = HarmonyTemplate.init(
        allocator,
        "harmony_gpt_oss",
        .low,
        true,
        tools,
    );

    const messages = &[_]OpenAIMessage{};
    const harmony = try tmpl.formatHarmonyChat(messages, tools);
    defer allocator.free(harmony);

    try std.testing.expect(std.mem.contains(u8, harmony, "browser"));
    try std.testing.expect(std.mem.contains(u8, harmony, "Search and browse the web"));
}

test "Format tool result" {
    const allocator = std.testing.allocator;

    var tmpl = HarmonyTemplate.init(
        allocator,
        "harmony_gpt_oss",
        .low,
        false,
        null,
    );

    const result = try tmpl.formatToolResult("browser", "San Francisco: 72°F, sunny", false);
    defer allocator.free(result);

    try std.testing.expect(std.mem.contains(u8, result, "<|recipient|>browser<|/recipient|>"));
    try std.testing.expect(std.mem.contains(u8, result, "<|tool_result|>"));
    try std.testing.expect(std.mem.contains(u8, result, "San Francisco: 72°F, sunny"));
}

test "Format error tool result" {
    const allocator = std.testing.allocator;

    var tmpl = HarmonyTemplate.init(
        allocator,
        "harmony_gpt_oss",
        .low,
        false,
        null,
    );

    const result = try tmpl.formatToolResult("python", "Syntax error", true);
    defer allocator.free(result);

    try std.testing.expect(std.mem.contains(u8, result, "Error:"));
}

test "Reasoning markers - low effort" {
    const allocator = std.testing.allocator;

    var tmpl = HarmonyTemplate.init(
        allocator,
        "harmony_gpt_oss",
        .low,
        false,
        null,
    );

    const content = "Simple response";
    const result = try tmpl.addReasoningMarkers(content);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("Simple response", result);
}

test "Reasoning markers - medium effort" {
    const allocator = std.testing.allocator;

    var tmpl = HarmonyTemplate.init(
        allocator,
        "harmony_gpt_oss",
        .medium,
        false,
        null,
    );

    const content = "Thinking through this...";
    const result = try tmpl.addReasoningMarkers(content);
    defer allocator.free(result);

    try std.testing.expect(std.mem.contains(u8, result, "<|reasoning|>"));
    try std.testing.expect(std.mem.contains(u8, result, "<|/reasoning|>"));
}

test "Format assistant with tool calls" {
    const allocator = std.testing.allocator;

    const tc = harmony.ToolCall{
        .name = "browser",
        .arguments = "{\"action\": \"search\"}",
    };

    const messages = &[_]OpenAIMessage{
        .{
            .role = "assistant",
            .content = "",
            .tool_calls = &[_]harmony.ToolCall{tc},
        },
    };

    var tmpl = HarmonyTemplate.init(
        allocator,
        "harmony_gpt_oss",
        .low,
        true,
        null,
    );

    const harmony_text = try tmpl.formatHarmonyChat(messages, null);
    defer allocator.free(harmony_text);

    try std.testing.expect(std.mem.contains(u8, harmony_text, "<|tool_call|>"));
    try std.testing.expect(std.mem.contains(u8, harmony_text, "browser"));
}

// ============================================================================
// Integration Tests
// ============================================================================

test "Round-trip: OpenAI -> Harmony -> Parse" {
    const allocator = std.testing.allocator;

    // Start with OpenAI format
    const openai_messages = &[_]OpenAIMessage{
        .{
            .role = "system",
            .content = "You are helpful.",
        },
        .{
            .role = "user",
            .content = "What's 2+2?",
        },
    };

    // Format to Harmony
    var tmpl = HarmonyTemplate.init(
        allocator,
        "harmony_gpt_oss",
        .low,
        false,
        null,
    );

    const harmony_text = try tmpl.formatHarmonyChat(openai_messages, null);
    defer allocator.free(harmony_text);

    // Parse back
    var p = HarmonyParser.init(allocator);
    var conv = try p.parse(harmony_text);
    defer conv.deinit();

    // Verify we have the messages
    try std.testing.expect(conv.messages.items.len >= 2);
}

test "Special token constants exist" {
    // Verify all expected tokens are defined
    try std.testing.expect(parser.SpecialTokens.startoftext.len > 0);
    try std.testing.expect(parser.SpecialTokens.endoftext.len > 0);
    try std.testing.expect(parser.SpecialTokens.tool_call_start.len > 0);
    try std.testing.expect(parser.SpecialTokens.tool_call_end.len > 0);
    try std.testing.expect(parser.SpecialTokens.recipient_start.len > 0);
    try std.testing.expect(parser.SpecialTokens.recipient_end.len > 0);
}
