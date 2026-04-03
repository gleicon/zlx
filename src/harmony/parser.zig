//! harmony_parser.zig - Parse Harmony format from text

const std = @import("std");
const harmony = @import("harmony.zig");

const HarmonyMessage = harmony.HarmonyMessage;
const HarmonyContent = harmony.HarmonyContent;
const HarmonyConversation = harmony.HarmonyConversation;
const HarmonyRole = harmony.HarmonyRole;
const ToolCall = harmony.ToolCall;
const ToolResult = harmony.ToolResult;

/// Special tokens in Harmony format
pub const SpecialTokens = struct {
    pub const startoftext = "<|startoftext|>";
    pub const endoftext = "<|endoftext|>";
    pub const tool_call_start = "<|tool_call|>";
    pub const tool_call_end = "<|/tool_call|>";
    pub const tool_result_start = "<|tool_result|>";
    pub const tool_result_end = "<|/tool_result|>";
    pub const recipient_start = "<|recipient|>";
    pub const recipient_end = "<|/recipient|>";
    pub const reasoning_start = "<|reasoning|>";
    pub const reasoning_end = "<|/reasoning|>";
    pub const user_start = "<|user|>";
    pub const user_end = "<|/user|>";
    pub const assistant_start = "<|assistant|>";
    pub const assistant_end = "<|/assistant|>";
    pub const system_start = "<|system|>";
    pub const system_end = "<|/system|>";
};

/// Parser state machine
const ParserState = enum {
    start,
    in_user,
    in_assistant,
    in_system,
    in_tool_call,
    in_tool_result,
    in_reasoning,
    in_recipient,
};

/// Harmony format parser
pub const HarmonyParser = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) HarmonyParser {
        return .{ .allocator = allocator };
    }

    /// Parse Harmony-encoded text into conversation
    pub fn parse(self: *HarmonyParser, text: []const u8) !HarmonyConversation {
        var conversation = HarmonyConversation.init(self.allocator);
        errdefer conversation.deinit();

        var state: ParserState = .start;
        var current_role: ?HarmonyRole = null;
        var current_content = std.ArrayList(u8).init(self.allocator);
        defer current_content.deinit();
        var current_recipient: ?[]const u8 = null;

        var i: usize = 0;
        while (i < text.len) {
            // Check for special tokens
            if (try self.parseSpecialToken(text, &i)) |token| {
                switch (token) {
                    .user_start => {
                        try self.finishMessage(&conversation, current_role, &current_content, current_recipient);
                        state = .in_user;
                        current_role = .user;
                        current_recipient = null;
                    },
                    .assistant_start => {
                        try self.finishMessage(&conversation, current_role, &current_content, current_recipient);
                        state = .in_assistant;
                        current_role = .assistant;
                        current_recipient = null;
                    },
                    .system_start => {
                        try self.finishMessage(&conversation, current_role, &current_content, current_recipient);
                        state = .in_system;
                        current_role = .system;
                        current_recipient = null;
                    },
                    .user_end, .assistant_end, .system_end => {
                        _ = state;
                        try self.finishMessage(&conversation, current_role, &current_content, current_recipient);
                        state = .start;
                        current_role = null;
                    },
                    .tool_call_start => {
                        try self.finishMessage(&conversation, current_role, &current_content, current_recipient);
                        state = .in_tool_call;
                        // Parse tool call
                        const tool_call = try self.parseToolCall(text, &i);
                        try conversation.addMessage(tool_call);
                        state = .start;
                    },
                    .tool_result_start => {
                        try self.finishMessage(&conversation, current_role, &current_content, current_recipient);
                        state = .in_tool_result;
                        // Parse tool result
                        const tool_result = try self.parseToolResult(text, &i);
                        try conversation.addMessage(tool_result);
                        state = .start;
                    },
                    .recipient_start => {
                        // Extract recipient name
                        current_recipient = try self.parseRecipient(text, &i);
                    },
                    .reasoning_start => {
                        state = .in_reasoning;
                    },
                    .reasoning_end => {
                        if (current_role) |role| {
                            const reasoning_text = try current_content.toOwnedSlice();
                            const msg = try HarmonyMessage.init(
                                self.allocator,
                                role,
                                .{ .reasoning = reasoning_text },
                                current_recipient,
                            );
                            try conversation.addMessage(msg);
                            current_content = std.ArrayList(u8).init(self.allocator);
                        }
                    },
                    else => {},
                }
            } else {
                // Regular text
                try current_content.append(text[i]);
                i += 1;
            }
        }

        // Finish any pending message
        try self.finishMessage(&conversation, current_role, &current_content, current_recipient);

        return conversation;
    }

    /// Extract tool calls from assistant output
    pub fn extractToolCalls(self: *HarmonyParser, text: []const u8) ![]ToolCall {
        var tool_calls = std.ArrayList(ToolCall).init(self.allocator);
        errdefer {
            for (tool_calls.items) |*tc| tc.deinit(self.allocator);
            tool_calls.deinit();
        }

        var i: usize = 0;
        while (i < text.len) {
            if (std.mem.indexOfPos(u8, text, i, SpecialTokens.tool_call_start)) |start| {
                const end = std.mem.indexOfPos(u8, text, start, SpecialTokens.tool_call_end) orelse break;
                const json_content = text[start + SpecialTokens.tool_call_start.len .. end];

                // Parse JSON
                const parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, json_content, .{});
                defer parsed.deinit();

                // Extract tool name and arguments
                const name = if (parsed.value.object.get("name")) |n|
                    try self.allocator.dupe(u8, n.string)
                else
                    try self.allocator.dupe(u8, "unknown");

                try tool_calls.append(.{
                    .name = name,
                    .arguments = try self.allocator.dupe(u8, json_content),
                });

                i = end + SpecialTokens.tool_call_end.len;
            } else {
                break;
            }
        }

        return tool_calls.toOwnedSlice();
    }

    /// Parse tool call from text
    fn parseToolCall(self: *HarmonyParser, text: []const u8, pos: *usize) !HarmonyMessage {
        // Find the closing tag
        const content_start = pos.*;
        if (std.mem.indexOfPos(u8, text, content_start, SpecialTokens.tool_call_end)) |end| {
            const json_content = text[content_start..end];
            pos.* = end + SpecialTokens.tool_call_end.len;

            // Parse JSON to extract name and args
            const parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, json_content, .{});
            defer parsed.deinit();

            const name = if (parsed.value.object.get("name")) |n|
                try self.allocator.dupe(u8, n.string)
            else
                try self.allocator.dupe(u8, "unknown");

            return HarmonyMessage{
                .role = .assistant,
                .content = .{ .tool_call = .{
                    .name = name,
                    .arguments = try self.allocator.dupe(u8, json_content),
                } },
                .recipient = try self.allocator.dupe(u8, name),
            };
        }

        return error.InvalidToolCall;
    }

    /// Parse tool result from text
    fn parseToolResult(self: *HarmonyParser, text: []const u8, pos: *usize) !HarmonyMessage {
        const content_start = pos.*;
        if (std.mem.indexOfPos(u8, text, content_start, SpecialTokens.tool_result_end)) |end| {
            const result_content = text[content_start..end];
            pos.* = end + SpecialTokens.tool_result_end.len;

            // Extract tool name from preceding recipient if available
            const tool_name = try self.allocator.dupe(u8, "tool");

            return HarmonyMessage{
                .role = .tool,
                .content = .{ .tool_result = .{
                    .tool_name = tool_name,
                    .result = try self.allocator.dupe(u8, result_content),
                } },
            };
        }

        return error.InvalidToolResult;
    }

    /// Parse recipient name
    fn parseRecipient(self: *HarmonyParser, text: []const u8, pos: *usize) !?[]const u8 {
        const content_start = pos.* + SpecialTokens.recipient_start.len;
        if (std.mem.indexOfPos(u8, text, content_start, SpecialTokens.recipient_end)) |end| {
            const recipient = text[content_start..end];
            pos.* = end + SpecialTokens.recipient_end.len;
            return try self.allocator.dupe(u8, recipient);
        }
        return null;
    }

    /// Parse special token at current position
    fn parseSpecialToken(self: *HarmonyParser, text: []const u8, pos: *usize) !?SpecialToken {
        _ = self;
        const tokens = &[_]struct { []const u8, SpecialToken }{
            .{ SpecialTokens.startoftext, .startoftext },
            .{ SpecialTokens.endoftext, .endoftext },
            .{ SpecialTokens.tool_call_start, .tool_call_start },
            .{ SpecialTokens.tool_call_end, .tool_call_end },
            .{ SpecialTokens.tool_result_start, .tool_result_start },
            .{ SpecialTokens.tool_result_end, .tool_result_end },
            .{ SpecialTokens.recipient_start, .recipient_start },
            .{ SpecialTokens.recipient_end, .recipient_end },
            .{ SpecialTokens.reasoning_start, .reasoning_start },
            .{ SpecialTokens.reasoning_end, .reasoning_end },
            .{ SpecialTokens.user_start, .user_start },
            .{ SpecialTokens.user_end, .user_end },
            .{ SpecialTokens.assistant_start, .assistant_start },
            .{ SpecialTokens.assistant_end, .assistant_end },
            .{ SpecialTokens.system_start, .system_start },
            .{ SpecialTokens.system_end, .system_end },
        };

        for (tokens) |entry| {
            const token_str = entry[0];
            const token_enum = entry[1];
            if (std.mem.startsWith(u8, text[pos.*..], token_str)) {
                pos.* += token_str.len;
                return token_enum;
            }
        }

        return null;
    }

    /// Finish current message and add to conversation
    fn finishMessage(
        self: *HarmonyParser,
        conversation: *HarmonyConversation,
        role: ?HarmonyRole,
        content: *std.ArrayList(u8),
        recipient: ?[]const u8,
    ) !void {
        if (role) |r| {
            if (content.items.len > 0) {
                const text = try content.toOwnedSlice();
                const msg = try HarmonyMessage.initText(self.allocator, r, text);
                if (recipient) |rec| {
                    msg.recipient = rec;
                }
                try conversation.addMessage(msg);
                content.* = std.ArrayList(u8).init(self.allocator);
            }
        }
    }
};

/// Special token types
const SpecialToken = enum {
    startoftext,
    endoftext,
    tool_call_start,
    tool_call_end,
    tool_result_start,
    tool_result_end,
    recipient_start,
    recipient_end,
    reasoning_start,
    reasoning_end,
    user_start,
    user_end,
    assistant_start,
    assistant_end,
    system_start,
    system_end,
};
