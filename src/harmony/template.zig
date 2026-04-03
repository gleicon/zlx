//! harmony_template.zig - Format conversations to Harmony format

const std = @import("std");
const harmony = @import("harmony.zig");

const HarmonyMessage = harmony.HarmonyMessage;
const HarmonyConversation = harmony.HarmonyConversation;
const HarmonyRole = harmony.HarmonyRole;
const ToolDefinition = harmony.ToolDefinition;
const OpenAIMessage = harmony.OpenAIMessage;
const ReasoningEffort = harmony.ReasoningEffort;

/// Harmony chat template for formatting conversations
pub const HarmonyTemplate = struct {
    allocator: std.mem.Allocator,
    encoding_name: []const u8,
    reasoning_effort: ReasoningEffort,
    include_tools: bool,
    tool_definitions: ?[]const ToolDefinition,

    pub fn init(
        allocator: std.mem.Allocator,
        encoding_name: []const u8,
        reasoning_effort: ReasoningEffort,
        include_tools: bool,
        tool_definitions: ?[]const ToolDefinition,
    ) HarmonyTemplate {
        return .{
            .allocator = allocator,
            .encoding_name = encoding_name,
            .reasoning_effort = reasoning_effort,
            .include_tools = include_tools,
            .tool_definitions = tool_definitions,
        };
    }

    /// Convert OpenAI messages to Harmony format
    pub fn formatHarmonyChat(
        self: *HarmonyTemplate,
        messages: []const OpenAIMessage,
        system_tools: ?[]const ToolDefinition,
    ) ![]u8 {
        var output = std.ArrayList(u8).init(self.allocator);
        errdefer output.deinit();

        // Start marker
        try output.appendSlice("<|startoftext|>\n");

        // System message with tools
        if (system_tools) |tools| {
            try output.appendSlice("<|system|>\n");
            try output.appendSlice("You are a helpful assistant.\n\n");
            try output.appendSlice("You have access to the following tools:\n");

            for (tools) |tool| {
                try std.fmt.format(output.writer(), "\n- {s}: {s}\n", .{
                    tool.name,
                    tool.description,
                });
                try std.json.stringify(tool.parameters, .{}, output.writer());
                try output.appendSlice("\n");
            }

            try output.appendSlice("<|/system|>\n");
        }

        // Format each message
        for (messages) |msg| {
            if (std.mem.eql(u8, msg.role, "system")) {
                try output.appendSlice("<|system|>\n");
                try output.appendSlice(msg.content);
                try output.appendSlice("\n<|/system|>\n");
            } else if (std.mem.eql(u8, msg.role, "user")) {
                try output.appendSlice("<|recipient|>user<|/recipient|>\n");
                try output.appendSlice("<|user|>\n");
                try output.appendSlice(msg.content);
                try output.appendSlice("\n<|/user|>\n");
            } else if (std.mem.eql(u8, msg.role, "assistant")) {
                if (msg.tool_calls) |tool_calls| {
                    for (tool_calls) |tc| {
                        try std.fmt.format(
                            output.writer(),
                            "<|recipient|>{s}<|/recipient|>\n",
                            .{tc.name},
                        );
                        try output.appendSlice("<|tool_call|>\n");
                        try output.appendSlice(tc.arguments);
                        try output.appendSlice("\n<|/tool_call|>\n");
                    }
                } else {
                    try output.appendSlice("<|recipient|>user<|/recipient|>\n");
                    try output.appendSlice("<|assistant|>\n");
                    try output.appendSlice(msg.content);
                    try output.appendSlice("\n<|/assistant|>\n");
                }
            } else if (std.mem.eql(u8, msg.role, "tool")) {
                const tool_name = msg.tool_call_id orelse "tool";
                try std.fmt.format(
                    output.writer(),
                    "<|recipient|>{s}<|/recipient|>\n",
                    .{tool_name},
                );
                try output.appendSlice("<|tool_result|>\n");
                try output.appendSlice(msg.content);
                try output.appendSlice("\n<|/tool_result|>\n");
            }
        }

        // Add assistant prompt for generation
        try output.appendSlice("<|recipient|>user<|/recipient|>\n");
        try output.appendSlice("<|assistant|>\n");

        return output.toOwnedSlice();
    }

    /// Format a tool execution result as Harmony
    pub fn formatToolResult(
        self: *HarmonyTemplate,
        tool_name: []const u8,
        result: []const u8,
        is_error: bool,
    ) ![]u8 {
        var output = std.ArrayList(u8).init(self.allocator);
        errdefer output.deinit();

        try std.fmt.format(
            output.writer(),
            "<|recipient|>{s}<|/recipient|>\n",
            .{tool_name},
        );
        try output.appendSlice("<|tool_result|>\n");

        if (is_error) {
            try output.appendSlice("Error: ");
        }

        try output.appendSlice(result);
        try output.appendSlice("\n<|/tool_result|>\n");

        // Add assistant continuation prompt
        try output.appendSlice("<|recipient|>user<|/recipient|>\n");
        try output.appendSlice("<|assistant|>\n");

        return output.toOwnedSlice();
    }

    /// Format tool definitions for system prompt
    pub fn formatToolDefinitions(
        self: *HarmonyTemplate,
        tools: []const ToolDefinition,
    ) ![]u8 {
        var output = std.ArrayList(u8).init(self.allocator);
        errdefer output.deinit();

        try output.appendSlice("Available tools:\n");

        for (tools) |tool| {
            try std.fmt.format(output.writer(), "\n{s}:\n", .{tool.name});
            try output.appendSlice(tool.description);
            try output.appendSlice("\nParameters: ");
            try std.json.stringify(tool.parameters, .{}, output.writer());
            try output.appendSlice("\n");
        }

        return output.toOwnedSlice();
    }

    /// Add reasoning markers based on effort level
    pub fn addReasoningMarkers(
        self: *HarmonyTemplate,
        content: []const u8,
    ) ![]u8 {
        switch (self.reasoning_effort) {
            .low => return self.allocator.dupe(u8, content),
            .medium => {
                var output = std.ArrayList(u8).init(self.allocator);
                errdefer output.deinit();
                try output.appendSlice("<|reasoning|>\n");
                try output.appendSlice(content);
                try output.appendSlice("\n<|/reasoning|>\n");
                return output.toOwnedSlice();
            },
            .high => {
                var output = std.ArrayList(u8).init(self.allocator);
                errdefer output.deinit();
                try output.appendSlice("<|reasoning|>\n");
                try output.appendSlice("Let me think through this step by step:\n");
                try output.appendSlice(content);
                try output.appendSlice("\n<|/reasoning|>\n");
                return output.toOwnedSlice();
            },
        }
    }

    /// Render conversation for model completion
    pub fn renderConversationForCompletion(
        self: *HarmonyTemplate,
        conversation: HarmonyConversation,
        target_role: HarmonyRole,
    ) ![]u8 {
        _ = self;
        _ = conversation;
        _ = target_role;
        // TODO: Implement conversation rendering with special tokens
        return self.allocator.dupe(u8, "");
    }
};

/// Format a single message to Harmony string
pub fn formatMessage(
    allocator: std.mem.Allocator,
    msg: HarmonyMessage,
) ![]u8 {
    var output = std.ArrayList(u8).init(allocator);
    errdefer output.deinit();

    if (msg.recipient) |recipient| {
        try std.fmt.format(
            output.writer(),
            "<|recipient|>{s}<|/recipient|>\n",
            .{recipient},
        );
    }

    switch (msg.role) {
        .user => {
            try output.appendSlice("<|user|>\n");
            switch (msg.content) {
                .text => |text| try output.appendSlice(text),
                else => {},
            }
            try output.appendSlice("\n<|/user|>\n");
        },
        .assistant => {
            try output.appendSlice("<|assistant|>\n");
            switch (msg.content) {
                .text => |text| try output.appendSlice(text),
                .tool_call => |tc| {
                    try output.appendSlice("<|tool_call|>\n");
                    try std.fmt.format(output.writer(), "{{\"name\": \"{s}\", \"arguments\": {s}}}\n", .{
                        tc.name,
                        tc.arguments,
                    });
                    try output.appendSlice("<|/tool_call|>\n");
                },
                .reasoning => |r| {
                    try output.appendSlice("<|reasoning|>\n");
                    try output.appendSlice(r);
                    try output.appendSlice("\n<|/reasoning|>\n");
                },
                else => {},
            }
            try output.appendSlice("<|/assistant|>\n");
        },
        .system => {
            try output.appendSlice("<|system|>\n");
            switch (msg.content) {
                .text => |text| try output.appendSlice(text),
                else => {},
            }
            try output.appendSlice("\n<|/system|>\n");
        },
        .tool => {
            switch (msg.content) {
                .tool_result => |tr| {
                    try output.appendSlice("<|tool_result|>\n");
                    if (tr.is_error) {
                        try output.appendSlice("Error: ");
                    }
                    try output.appendSlice(tr.result);
                    try output.appendSlice("\n<|/tool_result|>\n");
                },
                else => {},
            }
        },
    }

    return output.toOwnedSlice();
}
