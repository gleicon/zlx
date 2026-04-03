//! parser.zig - Parse Harmony format from text
//!
//! Parses Harmony-encoded text (with special tokens) into structured
//! HarmonyConversation objects. Handles tool calls, tool results, and
//! reasoning chains.

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

    /// Parse Harmony-encoded text into a structured conversation
    pub fn parse(self: *HarmonyParser, text: []const u8) !HarmonyConversation {
        var conversation = HarmonyConversation.init(self.allocator);
        errdefer conversation.deinit();

        var state: ParserState = .start;
        var current_role: ?HarmonyRole = null;
        var current_content: std.ArrayListUnmanaged(u8) = .empty;
        defer current_content.deinit(self.allocator);
        var current_recipient: ?[]const u8 = null;

        var i: usize = 0;
        while (i < text.len) {
            if (parseSpecialToken(text, &i)) |token| {
                switch (token) {
                    .startoftext, .endoftext => {
                        // Skip document markers
                    },
                    .user_start => {
                        try self.finishMessage(&conversation, current_role, &current_content, current_recipient);
                        if (current_recipient) |r| self.allocator.free(r);
                        current_recipient = null;
                        state = .in_user;
                        current_role = .user;
                    },
                    .assistant_start => {
                        try self.finishMessage(&conversation, current_role, &current_content, current_recipient);
                        if (current_recipient) |r| self.allocator.free(r);
                        current_recipient = null;
                        state = .in_assistant;
                        current_role = .assistant;
                    },
                    .system_start => {
                        try self.finishMessage(&conversation, current_role, &current_content, current_recipient);
                        if (current_recipient) |r| self.allocator.free(r);
                        current_recipient = null;
                        state = .in_system;
                        current_role = .system;
                    },
                    .user_end, .assistant_end, .system_end => {
                        try self.finishMessage(&conversation, current_role, &current_content, current_recipient);
                        if (current_recipient) |r| self.allocator.free(r);
                        state = .start;
                        current_role = null;
                        current_recipient = null;
                    },
                    .tool_call_start => {
                        try self.finishMessage(&conversation, current_role, &current_content, current_recipient);
                        state = .in_tool_call;
                        const prev_recipient = current_recipient;
                        current_recipient = null;
                        const tool_call = try self.parseToolCall(text, &i, prev_recipient);
                        if (prev_recipient) |r| self.allocator.free(r);
                        try conversation.addMessage(tool_call);
                        state = .start;
                    },
                    .tool_result_start => {
                        try self.finishMessage(&conversation, current_role, &current_content, current_recipient);
                        state = .in_tool_result;
                        const prev_recipient = current_recipient;
                        current_recipient = null;
                        const tool_result = try self.parseToolResult(text, &i, prev_recipient);
                        if (prev_recipient) |r| self.allocator.free(r);
                        try conversation.addMessage(tool_result);
                        state = .start;
                    },
                    .recipient_start => {
                        // Free previous recipient if any
                        if (current_recipient) |r| self.allocator.free(r);
                        current_recipient = try self.parseRecipient(text, &i);
                    },
                    .reasoning_start => {
                        state = .in_reasoning;
                    },
                    .reasoning_end => {
                        if (current_content.items.len > 0) {
                            const reasoning_text = try current_content.toOwnedSlice(self.allocator);
                            errdefer self.allocator.free(reasoning_text);
                            const msg = HarmonyMessage{
                                .role = current_role orelse .assistant,
                                .content = .{ .reasoning = reasoning_text },
                                .recipient = current_recipient,
                            };
                            current_recipient = null;
                            try conversation.addMessage(msg);
                        } else {
                            if (current_recipient) |r| self.allocator.free(r);
                            current_recipient = null;
                        }
                        state = .start;
                    },
                    else => {
                        // Ignore unknown tokens
                    },
                }
            } else {
                // Regular text - accumulate
                try current_content.append(self.allocator, text[i]);
                i += 1;
            }
        }

        // Finish any pending message
        try self.finishMessage(&conversation, current_role, &current_content, current_recipient);
        if (current_recipient) |r| self.allocator.free(r);

        return conversation;
    }

    /// Extract all tool calls from assistant output text
    pub fn extractToolCalls(self: *HarmonyParser, text: []const u8) ![]ToolCall {
        var tool_calls: std.ArrayListUnmanaged(ToolCall) = .empty;
        errdefer {
            for (tool_calls.items) |*tc| tc.deinit(self.allocator);
            tool_calls.deinit(self.allocator);
        }

        var i: usize = 0;
        while (i < text.len) {
            if (std.mem.indexOfPos(u8, text, i, SpecialTokens.tool_call_start)) |start| {
                const content_start = start + SpecialTokens.tool_call_start.len;
                const end = std.mem.indexOfPos(u8, text, content_start, SpecialTokens.tool_call_end) orelse break;
                const json_content = std.mem.trim(u8, text[content_start..end], " \t\n\r");

                // Try to find a recipient before this tool call to get the name
                const name = try self.extractRecipientBefore(text, start) orelse
                    try self.extractNameFromJson(json_content) orelse
                    try self.allocator.dupe(u8, "unknown");

                try tool_calls.append(self.allocator, .{
                    .name = name,
                    .arguments = try self.allocator.dupe(u8, json_content),
                });

                i = end + SpecialTokens.tool_call_end.len;
            } else {
                break;
            }
        }

        return tool_calls.toOwnedSlice(self.allocator);
    }

    /// Parse messages from completion output tokens (text)
    pub fn parseMessagesFromTokens(self: *HarmonyParser, text: []const u8) !HarmonyConversation {
        return self.parse(text);
    }

    /// Parse Harmony encoding name from text header
    pub fn parseHarmonyEncoding(text: []const u8) ?[]const u8 {
        // Look for encoding identifier in header
        const prefix = "harmony_encoding:";
        if (std.mem.indexOf(u8, text, prefix)) |start| {
            const value_start = start + prefix.len;
            const end = std.mem.indexOfAny(u8, text[value_start..], " \t\n\r") orelse
                return text[value_start..];
            return text[value_start .. value_start + end];
        }
        return null;
    }

    // ========================================================================
    // Private helpers
    // ========================================================================

    /// Parse tool call content (caller already consumed <|tool_call|>)
    fn parseToolCall(
        self: *HarmonyParser,
        text: []const u8,
        pos: *usize,
        recipient: ?[]const u8,
    ) !HarmonyMessage {
        const content_start = pos.*;
        if (std.mem.indexOfPos(u8, text, content_start, SpecialTokens.tool_call_end)) |end| {
            const json_content = std.mem.trim(u8, text[content_start..end], " \t\n\r");
            pos.* = end + SpecialTokens.tool_call_end.len;

            // Determine tool name: from recipient, or from JSON "name" field
            const name = if (recipient) |r|
                try self.allocator.dupe(u8, r)
            else
                try self.extractNameFromJson(json_content) orelse
                    try self.allocator.dupe(u8, "unknown");

            const msg_recipient = if (recipient) |r|
                try self.allocator.dupe(u8, r)
            else
                try self.allocator.dupe(u8, name);

            return HarmonyMessage{
                .role = .assistant,
                .content = .{ .tool_call = .{
                    .name = name,
                    .arguments = try self.allocator.dupe(u8, json_content),
                } },
                .recipient = msg_recipient,
            };
        }

        return error.InvalidToolCall;
    }

    /// Parse tool result content (caller already consumed <|tool_result|>)
    fn parseToolResult(
        self: *HarmonyParser,
        text: []const u8,
        pos: *usize,
        recipient: ?[]const u8,
    ) !HarmonyMessage {
        const content_start = pos.*;
        if (std.mem.indexOfPos(u8, text, content_start, SpecialTokens.tool_result_end)) |end| {
            const result_content = std.mem.trim(u8, text[content_start..end], " \t\n\r");
            pos.* = end + SpecialTokens.tool_result_end.len;

            const tool_name = if (recipient) |r|
                try self.allocator.dupe(u8, r)
            else
                try self.allocator.dupe(u8, "tool");

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

    /// Parse recipient name between <|recipient|> and <|/recipient|>
    /// Assumes pos is AFTER the <|recipient|> token was already consumed
    fn parseRecipient(self: *HarmonyParser, text: []const u8, pos: *usize) !?[]const u8 {
        const content_start = pos.*;
        if (std.mem.indexOfPos(u8, text, content_start, SpecialTokens.recipient_end)) |end| {
            const recipient = text[content_start..end];
            pos.* = end + SpecialTokens.recipient_end.len;
            return try self.allocator.dupe(u8, std.mem.trim(u8, recipient, " \t\n\r"));
        }
        return null;
    }

    /// Find recipient token immediately before given position
    fn extractRecipientBefore(self: *HarmonyParser, text: []const u8, pos: usize) !?[]const u8 {
        // Scan backwards for <|recipient|>...<|/recipient|> pattern just before pos
        const search_range = if (pos > 128) text[pos - 128 .. pos] else text[0..pos];
        if (std.mem.lastIndexOf(u8, search_range, SpecialTokens.recipient_start)) |r_start| {
            const r_content_start = r_start + SpecialTokens.recipient_start.len;
            if (std.mem.indexOfPos(u8, search_range, r_content_start, SpecialTokens.recipient_end)) |r_end| {
                const name = search_range[r_content_start..r_end];
                return try self.allocator.dupe(u8, std.mem.trim(u8, name, " \t\n\r"));
            }
        }
        return null;
    }

    /// Try to extract "name" field from JSON object string
    fn extractNameFromJson(self: *HarmonyParser, json_content: []const u8) !?[]const u8 {
        if (json_content.len == 0) return null;
        const parsed = std.json.parseFromSlice(std.json.Value, self.allocator, json_content, .{}) catch return null;
        defer parsed.deinit();
        if (parsed.value != .object) return null;
        const name_val = parsed.value.object.get("name") orelse return null;
        if (name_val != .string) return null;
        return try self.allocator.dupe(u8, name_val.string);
    }

    /// Finish current message and add to conversation if it has content
    fn finishMessage(
        self: *HarmonyParser,
        conversation: *HarmonyConversation,
        role: ?HarmonyRole,
        content: *std.ArrayListUnmanaged(u8),
        recipient: ?[]const u8,
    ) !void {
        if (role) |r| {
            const trimmed = std.mem.trim(u8, content.items, " \t\n\r");
            if (trimmed.len > 0) {
                const text = try self.allocator.dupe(u8, trimmed);
                errdefer self.allocator.free(text);
                var msg = HarmonyMessage{
                    .role = r,
                    .content = .{ .text = text },
                    .recipient = if (recipient) |rec| try self.allocator.dupe(u8, rec) else null,
                };
                errdefer msg.deinit(self.allocator);
                try conversation.addMessage(msg);
            }
            content.clearRetainingCapacity();
        }
    }

    /// Check for a special token at current pos; advances pos if found
    fn parseSpecialToken(text: []const u8, pos: *usize) ?SpecialToken {
        const entries = [_]struct { []const u8, SpecialToken }{
            .{ SpecialTokens.startoftext, .startoftext },
            .{ SpecialTokens.endoftext, .endoftext },
            .{ SpecialTokens.tool_call_end, .tool_call_end }, // Must come before tool_call_start
            .{ SpecialTokens.tool_call_start, .tool_call_start },
            .{ SpecialTokens.tool_result_end, .tool_result_end },
            .{ SpecialTokens.tool_result_start, .tool_result_start },
            .{ SpecialTokens.recipient_end, .recipient_end },
            .{ SpecialTokens.recipient_start, .recipient_start },
            .{ SpecialTokens.reasoning_end, .reasoning_end },
            .{ SpecialTokens.reasoning_start, .reasoning_start },
            .{ SpecialTokens.user_end, .user_end },
            .{ SpecialTokens.user_start, .user_start },
            .{ SpecialTokens.assistant_end, .assistant_end },
            .{ SpecialTokens.assistant_start, .assistant_start },
            .{ SpecialTokens.system_end, .system_end },
            .{ SpecialTokens.system_start, .system_start },
        };

        const remaining = text[pos.*..];
        for (entries) |entry| {
            const token_str = entry[0];
            const token_enum = entry[1];
            if (std.mem.startsWith(u8, remaining, token_str)) {
                pos.* += token_str.len;
                return token_enum;
            }
        }

        return null;
    }
};
