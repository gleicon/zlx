//! tool_executor.zig - Tool execution orchestrator

const std = @import("std");
const tools_types = @import("types.zig");
const browser = @import("browser.zig");
const python = @import("python.zig");

const Tool = tools_types.Tool;
const ToolCallRequest = tools_types.ToolCallRequest;
const ToolExecutionResult = tools_types.ToolExecutionResult;
const ToolDefinition = tools_types.ToolDefinition;
const ToolRegistry = tools_types.ToolRegistry;
const BrowserTool = browser.BrowserTool;
const PythonTool = python.PythonTool;

/// Tool orchestrator for managing and executing tools
pub const ToolExecutor = struct {
    allocator: std.mem.Allocator,
    registry: ToolRegistry,

    pub fn init(allocator: std.mem.Allocator) ToolExecutor {
        return .{
            .allocator = allocator,
            .registry = ToolRegistry.init(allocator),
        };
    }

    pub fn deinit(self: *ToolExecutor) void {
        self.registry.deinit();
    }

    /// Register the browser tool
    pub fn registerBrowser(self: *ToolExecutor, config: BrowserTool.BrowserConfig, backend: BrowserTool.SearchBackend) !void {
        const browser_tool = try self.allocator.create(BrowserTool);
        browser_tool.* = try BrowserTool.init(self.allocator, config, backend);
        try self.registry.register("browser", browser_tool.asTool());
    }

    /// Register the Python tool
    pub fn registerPython(self: *ToolExecutor, config: PythonTool.PythonConfig) !void {
        const python_tool = try self.allocator.create(PythonTool);
        python_tool.* = PythonTool.init(self.allocator, config);
        try self.registry.register("python", python_tool.asTool());
    }

    /// Execute a single tool call
    pub fn executeToolCall(self: *ToolExecutor, request: ToolCallRequest) !ToolExecutionResult {
        const tool = self.registry.get(request.name) orelse {
            return .{
                .success = false,
                .output = &[_]u8{},
                .error_message = try std.fmt.allocPrint(
                    self.allocator,
                    "Unknown tool: {s}",
                    .{request.name},
                ),
            };
        };

        return try tool.execute(request, self.allocator);
    }

    /// Execute multiple tool calls in sequence
    pub fn executeToolCalls(
        self: *ToolExecutor,
        requests: []const ToolCallRequest,
    ) ![]ToolExecutionResult {
        var results = std.ArrayList(ToolExecutionResult).init(self.allocator);
        errdefer results.deinit();

        for (requests) |request| {
            const result = try self.executeToolCall(request);
            try results.append(result);
        }

        return results.toOwnedSlice();
    }

    /// Format results to Harmony format
    pub fn resultsToHarmony(
        self: *ToolExecutor,
        results: []const ToolExecutionResult,
        tool_names: []const []const u8,
    ) ![]u8 {
        var output = std.ArrayList(u8).init(self.allocator);
        errdefer output.deinit();

        for (results, tool_names) |result, name| {
            const harmony = try result.toHarmony(self.allocator, name);
            defer self.allocator.free(harmony);
            try output.appendSlice(harmony);
        }

        return output.toOwnedSlice();
    }

    /// Get all registered tool definitions
    pub fn getToolDefinitions(self: *ToolExecutor) ![]ToolDefinition {
        return try self.registry.getAllDefinitions(self.allocator);
    }

    /// Check if a tool is registered
    pub fn hasTool(self: *const ToolExecutor, name: []const u8) bool {
        return self.registry.hasTool(name);
    }
};
