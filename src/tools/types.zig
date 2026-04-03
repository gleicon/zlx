//! tools_types.zig - Shared types for tools

const std = @import("std");

/// Tool call request from model
pub const ToolCallRequest = struct {
    name: []const u8,
    arguments: []const u8, // JSON
    id: ?[]const u8 = null,
    allocator: ?std.mem.Allocator = null,

    pub fn init(name: []const u8, arguments: []const u8) ToolCallRequest {
        return .{
            .name = name,
            .arguments = arguments,
        };
    }

    pub fn deinit(self: *ToolCallRequest) void {
        if (self.allocator) |allocator| {
            allocator.free(self.name);
            allocator.free(self.arguments);
            if (self.id) |id| allocator.free(id);
        }
    }

    /// Parse arguments JSON to typed struct
    pub fn parseArguments(self: *const ToolCallRequest, comptime T: type) !T {
        return try std.json.parseFromSlice(T, std.heap.page_allocator, self.arguments, .{});
    }
};

/// Tool execution result
pub const ToolExecutionResult = struct {
    success: bool,
    output: []const u8,
    error_message: ?[]const u8 = null,
    execution_time_ms: u64 = 0,

    pub fn deinit(self: *ToolExecutionResult, allocator: std.mem.Allocator) void {
        allocator.free(self.output);
        if (self.error_message) |msg| allocator.free(msg);
    }

    /// Convert to Harmony format
    pub fn toHarmony(
        self: *const ToolExecutionResult,
        allocator: std.mem.Allocator,
        tool_name: []const u8,
    ) ![]u8 {
        var output = std.ArrayList(u8).init(allocator);
        errdefer output.deinit();

        try std.fmt.format(
            output.writer(),
            "<|recipient|>{s}<|/recipient|>\n<|tool_result|>\n",
            .{tool_name},
        );

        if (!self.success and self.error_message != null) {
            try output.appendSlice("Error: ");
            try output.appendSlice(self.error_message.?);
        } else {
            try output.appendSlice(self.output);
        }

        try output.appendSlice("\n<|/tool_result|>\n");
        try output.appendSlice("<|recipient|>user<|/recipient|>\n<|assistant|>\n");

        return output.toOwnedSlice();
    }
};

/// Tool definition for system prompts
pub const ToolDefinition = struct {
    name: []const u8,
    description: []const u8,
    parameters: []const u8, // JSON schema string

    pub fn deinit(self: *ToolDefinition, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.description);
        allocator.free(self.parameters);
    }

    /// Convert to JSON for system prompt
    pub fn toJSON(self: *const ToolDefinition, allocator: std.mem.Allocator) ![]u8 {
        var output = std.ArrayList(u8).init(allocator);
        errdefer output.deinit();

        try std.fmt.format(output.writer(),
            \\{{"type": "function", "function": {{"name": "{s}", "description": "{s}", "parameters": {s}}}}
        , .{
            self.name,
            self.description,
            self.parameters,
        });

        return output.toOwnedSlice();
    }
};

/// Tool trait interface
pub const Tool = struct {
    vtable: *const VTable,
    ptr: *anyopaque,

    pub const VTable = struct {
        execute: *const fn (ctx: *anyopaque, request: ToolCallRequest, allocator: std.mem.Allocator) anyerror!ToolExecutionResult,
        getDefinition: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator) anyerror!ToolDefinition,
        deinit: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator) void,
    };

    pub fn execute(self: Tool, request: ToolCallRequest, allocator: std.mem.Allocator) !ToolExecutionResult {
        return self.vtable.execute(self.ptr, request, allocator);
    }

    pub fn getDefinition(self: Tool, allocator: std.mem.Allocator) !ToolDefinition {
        return self.vtable.getDefinition(self.ptr, allocator);
    }

    pub fn deinit(self: Tool, allocator: std.mem.Allocator) void {
        self.vtable.deinit(self.ptr, allocator);
    }
};

/// Tool registry for managing available tools
pub const ToolRegistry = struct {
    tools: std.StringHashMap(Tool),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) ToolRegistry {
        return .{
            .tools = std.StringHashMap(Tool).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ToolRegistry) void {
        var iter = self.tools.valueIterator();
        while (iter.next()) |tool| {
            tool.deinit(self.allocator);
        }
        self.tools.deinit();
    }

    pub fn register(self: *ToolRegistry, name: []const u8, tool: Tool) !void {
        try self.tools.put(name, tool);
    }

    pub fn get(self: *const ToolRegistry, name: []const u8) ?Tool {
        return self.tools.get(name);
    }

    pub fn hasTool(self: *const ToolRegistry, name: []const u8) bool {
        return self.tools.contains(name);
    }

    pub fn getAllDefinitions(self: *const ToolRegistry, allocator: std.mem.Allocator) ![]ToolDefinition {
        var definitions = std.ArrayList(ToolDefinition).init(allocator);
        errdefer definitions.deinit();

        var iter = self.tools.valueIterator();
        while (iter.next()) |tool| {
            const def = try tool.getDefinition(allocator);
            try definitions.append(def);
        }

        return definitions.toOwnedSlice();
    }
};
