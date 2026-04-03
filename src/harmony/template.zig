//! template.zig - Format conversations to Harmony format
//!
//! Converts OpenAI-format messages into Harmony-encoded text suitable for
//! GPT-OSS models. Handles tool definitions in system prompts, tool call
//! formatting, tool result injection, and reasoning chain markers.

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

    /// Convert OpenAI-format messages to Harmony format string.
    /// The output ends with an open <|assistant|> tag ready for model generation.
    pub fn formatHarmonyChat(
        self: *HarmonyTemplate,
        messages: []const OpenAIMessage,
        system_tools: ?[]const ToolDefinition,
    ) ![]u8 {
        var output: std.ArrayListUnmanaged(u8) = .empty;
        errdefer output.deinit(self.allocator);

        // Start marker
        try output.appendSlice(self.allocator, "<|startoftext|>\n");

        // Optional system message with tool definitions
        if (system_tools) |tools| {
            try output.appendSlice(self.allocator, "<|system|>\n");
            try output.appendSlice(self.allocator, "You are a helpful assistant.\n\n");
            try output.appendSlice(self.allocator, "You have access to the following tools:\n");

            for (tools) |tool| {
                try std.fmt.format(output.writer(self.allocator), "\n- {s}: {s}\n", .{
                    tool.name,
                    tool.description,
                });
                try std.fmt.format(output.writer(self.allocator), "{f}\n", .{std.json.fmt(tool.parameters, .{})});
            }

            try output.appendSlice(self.allocator, "<|/system|>\n");
        }

        // Format each OpenAI message
        for (messages) |msg| {
            if (std.mem.eql(u8, msg.role, "system")) {
                try output.appendSlice(self.allocator, "<|system|>\n");
                try output.appendSlice(self.allocator, msg.content);
                try output.appendSlice(self.allocator, "\n<|/system|>\n");
            } else if (std.mem.eql(u8, msg.role, "user")) {
                try output.appendSlice(self.allocator, "<|recipient|>user<|/recipient|>\n");
                try output.appendSlice(self.allocator, "<|user|>\n");
                try output.appendSlice(self.allocator, msg.content);
                try output.appendSlice(self.allocator, "\n<|/user|>\n");
            } else if (std.mem.eql(u8, msg.role, "assistant")) {
                if (msg.tool_calls) |tool_calls| {
                    for (tool_calls) |tc| {
                        try std.fmt.format(
                            output.writer(self.allocator),
                            "<|recipient|>{s}<|/recipient|>\n",
                            .{tc.name},
                        );
                        try output.appendSlice(self.allocator, "<|tool_call|>\n");
                        try output.appendSlice(self.allocator, tc.arguments);
                        try output.appendSlice(self.allocator, "\n<|/tool_call|>\n");
                    }
                } else {
                    try output.appendSlice(self.allocator, "<|recipient|>user<|/recipient|>\n");
                    try output.appendSlice(self.allocator, "<|assistant|>\n");
                    try output.appendSlice(self.allocator, msg.content);
                    try output.appendSlice(self.allocator, "\n<|/assistant|>\n");
                }
            } else if (std.mem.eql(u8, msg.role, "tool")) {
                const tool_name = msg.tool_call_id orelse "tool";
                try std.fmt.format(
                    output.writer(self.allocator),
                    "<|recipient|>{s}<|/recipient|>\n",
                    .{tool_name},
                );
                try output.appendSlice(self.allocator, "<|tool_result|>\n");
                try output.appendSlice(self.allocator, msg.content);
                try output.appendSlice(self.allocator, "\n<|/tool_result|>\n");
            }
        }

        // Add open assistant prompt for model generation
        try output.appendSlice(self.allocator, "<|recipient|>user<|/recipient|>\n");
        try output.appendSlice(self.allocator, "<|assistant|>\n");

        return output.toOwnedSlice(self.allocator);
    }

    /// Render a complete HarmonyConversation for model completion
    pub fn renderConversation(
        self: *HarmonyTemplate,
        conversation: HarmonyConversation,
    ) ![]u8 {
        var output: std.ArrayListUnmanaged(u8) = .empty;
        errdefer output.deinit(self.allocator);

        try output.appendSlice(self.allocator, "<|startoftext|>\n");

        for (conversation.messages.items) |msg| {
            const msg_text = try formatMessage(self.allocator, msg);
            defer self.allocator.free(msg_text);
            try output.appendSlice(self.allocator, msg_text);
        }

        return output.toOwnedSlice(self.allocator);
    }

    /// Format a tool execution result as Harmony with assistant continuation prompt
    pub fn formatToolResult(
        self: *HarmonyTemplate,
        tool_name: []const u8,
        result: []const u8,
        is_error: bool,
    ) ![]u8 {
        var output: std.ArrayListUnmanaged(u8) = .empty;
        errdefer output.deinit(self.allocator);

        try std.fmt.format(
            output.writer(self.allocator),
            "<|recipient|>{s}<|/recipient|>\n",
            .{tool_name},
        );
        try output.appendSlice(self.allocator, "<|tool_result|>\n");

        if (is_error) {
            try output.appendSlice(self.allocator, "Error: ");
        }

        try output.appendSlice(self.allocator, result);
        try output.appendSlice(self.allocator, "\n<|/tool_result|>\n");

        // Add assistant continuation prompt
        try output.appendSlice(self.allocator, "<|recipient|>user<|/recipient|>\n");
        try output.appendSlice(self.allocator, "<|assistant|>\n");

        return output.toOwnedSlice(self.allocator);
    }

    /// Format tool definitions for inclusion in a system prompt
    pub fn formatToolDefinitions(
        self: *HarmonyTemplate,
        tools: []const ToolDefinition,
    ) ![]u8 {
        var output: std.ArrayListUnmanaged(u8) = .empty;
        errdefer output.deinit(self.allocator);

        try output.appendSlice(self.allocator, "Available tools:\n");

        for (tools) |tool| {
            try std.fmt.format(output.writer(self.allocator), "\n{s}:\n", .{tool.name});
            try output.appendSlice(self.allocator, tool.description);
            try std.fmt.format(output.writer(self.allocator), "\nParameters: {f}\n", .{std.json.fmt(tool.parameters, .{})});
        }

        return output.toOwnedSlice(self.allocator);
    }

    /// Add reasoning chain markers based on configured effort level
    pub fn addReasoningMarkers(
        self: *HarmonyTemplate,
        content: []const u8,
    ) ![]u8 {
        switch (self.reasoning_effort) {
            .low => return self.allocator.dupe(u8, content),
            .medium => {
                var output: std.ArrayListUnmanaged(u8) = .empty;
                errdefer output.deinit(self.allocator);
                try output.appendSlice(self.allocator, "<|reasoning|>\n");
                try output.appendSlice(self.allocator, content);
                try output.appendSlice(self.allocator, "\n<|/reasoning|>\n");
                return output.toOwnedSlice(self.allocator);
            },
            .high => {
                var output: std.ArrayListUnmanaged(u8) = .empty;
                errdefer output.deinit(self.allocator);
                try output.appendSlice(self.allocator, "<|reasoning|>\n");
                try output.appendSlice(self.allocator, "Let me think through this step by step:\n");
                try output.appendSlice(self.allocator, content);
                try output.appendSlice(self.allocator, "\n<|/reasoning|>\n");
                return output.toOwnedSlice(self.allocator);
            },
        }
    }
};

/// Format a single HarmonyMessage into its Harmony text representation
pub fn formatMessage(
    allocator: std.mem.Allocator,
    msg: HarmonyMessage,
) ![]u8 {
    var output: std.ArrayListUnmanaged(u8) = .empty;
    errdefer output.deinit(allocator);

    if (msg.recipient) |recipient| {
        try std.fmt.format(
            output.writer(allocator),
            "<|recipient|>{s}<|/recipient|>\n",
            .{recipient},
        );
    }

    switch (msg.role) {
        .user => {
            try output.appendSlice(allocator, "<|user|>\n");
            switch (msg.content) {
                .text => |text| try output.appendSlice(allocator, text),
                else => {},
            }
            try output.appendSlice(allocator, "\n<|/user|>\n");
        },
        .assistant => {
            switch (msg.content) {
                .text => |text| {
                    try output.appendSlice(allocator, "<|assistant|>\n");
                    try output.appendSlice(allocator, text);
                    try output.appendSlice(allocator, "\n<|/assistant|>\n");
                },
                .tool_call => |tc| {
                    try output.appendSlice(allocator, "<|tool_call|>\n");
                    try std.fmt.format(output.writer(allocator), "{{\"name\": \"{s}\", \"arguments\": {s}}}\n", .{
                        tc.name,
                        tc.arguments,
                    });
                    try output.appendSlice(allocator, "<|/tool_call|>\n");
                },
                .reasoning => |r| {
                    try output.appendSlice(allocator, "<|reasoning|>\n");
                    try output.appendSlice(allocator, r);
                    try output.appendSlice(allocator, "\n<|/reasoning|>\n");
                },
                else => {},
            }
        },
        .system => {
            try output.appendSlice(allocator, "<|system|>\n");
            switch (msg.content) {
                .text => |text| try output.appendSlice(allocator, text),
                else => {},
            }
            try output.appendSlice(allocator, "\n<|/system|>\n");
        },
        .tool => {
            switch (msg.content) {
                .tool_result => |tr| {
                    try output.appendSlice(allocator, "<|tool_result|>\n");
                    if (tr.is_error) {
                        try output.appendSlice(allocator, "Error: ");
                    }
                    try output.appendSlice(allocator, tr.result);
                    try output.appendSlice(allocator, "\n<|/tool_result|>\n");
                },
                else => {},
            }
        },
    }

    return output.toOwnedSlice(allocator);
}
