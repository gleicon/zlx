//! harmony.zig - Core Harmony types and interfaces
//!
//! Harmony is GPT-OSS's native conversation format with special tokens for:
//! - Tool calls and results
//! - Recipient routing
//! - Reasoning chains

const std = @import("std");

/// Role in a Harmony conversation
pub const HarmonyRole = enum {
    user,
    assistant,
    system,
    tool,
};

/// Content type in Harmony messages
pub const HarmonyContent = union(enum) {
    text: []const u8,
    tool_call: ToolCall,
    tool_result: ToolResult,
    reasoning: []const u8,

    pub fn deinit(self: *HarmonyContent, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .text => |text| allocator.free(text),
            .tool_call => |*tc| tc.deinit(allocator),
            .tool_result => |*tr| tr.deinit(allocator),
            .reasoning => |r| allocator.free(r),
        }
    }
};

/// Tool call specification
pub const ToolCall = struct {
    name: []const u8,
    arguments: []const u8,
    id: ?[]const u8 = null,

    pub fn deinit(self: *ToolCall, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.arguments);
        if (self.id) |id| allocator.free(id);
    }
};

/// Tool execution result
pub const ToolResult = struct {
    tool_name: []const u8,
    result: []const u8,
    is_error: bool = false,

    pub fn deinit(self: *ToolResult, allocator: std.mem.Allocator) void {
        allocator.free(self.tool_name);
        allocator.free(self.result);
    }
};

/// Single message in a Harmony conversation
pub const HarmonyMessage = struct {
    role: HarmonyRole,
    content: HarmonyContent,
    recipient: ?[]const u8 = null,

    pub fn initText(
        allocator: std.mem.Allocator,
        role: HarmonyRole,
        text: []const u8,
    ) !HarmonyMessage {
        return .{
            .role = role,
            .content = .{ .text = try allocator.dupe(u8, text) },
        };
    }

    pub fn initToolCall(
        allocator: std.mem.Allocator,
        name: []const u8,
        args: []const u8,
        recipient: []const u8,
    ) !HarmonyMessage {
        return .{
            .role = .assistant,
            .content = .{ .tool_call = .{
                .name = try allocator.dupe(u8, name),
                .arguments = try allocator.dupe(u8, args),
            } },
            .recipient = try allocator.dupe(u8, recipient),
        };
    }

    pub fn initToolResult(
        allocator: std.mem.Allocator,
        tool_name: []const u8,
        result: []const u8,
    ) !HarmonyMessage {
        return .{
            .role = .tool,
            .content = .{ .tool_result = .{
                .tool_name = try allocator.dupe(u8, tool_name),
                .result = try allocator.dupe(u8, result),
            } },
        };
    }

    pub fn deinit(self: *HarmonyMessage, allocator: std.mem.Allocator) void {
        self.content.deinit(allocator);
        if (self.recipient) |r| allocator.free(r);
    }
};

/// Complete Harmony conversation
pub const HarmonyConversation = struct {
    messages: std.ArrayList(HarmonyMessage),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) HarmonyConversation {
        return .{
            .messages = std.ArrayList(HarmonyMessage).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn addMessage(self: *HarmonyConversation, message: HarmonyMessage) !void {
        try self.messages.append(message);
    }

    pub fn deinit(self: *HarmonyConversation) void {
        for (self.messages.items) |*m| m.deinit(self.allocator);
        self.messages.deinit();
    }

    /// Check if conversation has tool calls pending
    pub fn hasToolCalls(self: *const HarmonyConversation) bool {
        for (self.messages.items) |msg| {
            if (msg.content == .tool_call) return true;
        }
        return false;
    }

    /// Get last assistant message
    pub fn getLastAssistantMessage(self: *const HarmonyConversation) ?*const HarmonyMessage {
        var i: usize = self.messages.items.len;
        while (i > 0) {
            i -= 1;
            if (self.messages.items[i].role == .assistant) {
                return &self.messages.items[i];
            }
        }
        return null;
    }
};

/// Tool definition for system prompts
pub const ToolDefinition = struct {
    name: []const u8,
    description: []const u8,
    parameters: std.json.Value,

    pub fn deinit(self: *ToolDefinition, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.description);
        // JSON value cleanup handled by parsed document
        _ = allocator;
    }
};

/// Reasoning effort level for model
pub const ReasoningEffort = enum {
    low,
    medium,
    high,
};

/// OpenAI message format for conversion
pub const OpenAIMessage = struct {
    role: []const u8,
    content: []const u8,
    tool_calls: ?[]ToolCall = null,
    tool_call_id: ?[]const u8 = null,
};
