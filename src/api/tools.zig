//! tools_api.zig - HTTP API endpoints for tools

const std = @import("std");
fn ArrayList(comptime T: type) type { return std.array_list.AlignedManaged(T, null); }
const httpz = @import("httpz");
const tools_types = @import("../tools/types.zig");
const tool_executor = @import("../tools/tool_executor.zig");

const ToolExecutor = tool_executor.ToolExecutor;
const ToolExecutionResult = tools_types.ToolExecutionResult;

/// Browser tool request
const BrowserRequest = struct {
    action: []const u8,
    query: ?[]const u8 = null,
    url: ?[]const u8 = null,
};

/// Python tool request
const PythonRequest = struct {
    code: []const u8,
};

/// Tool response
const ToolResponse = struct {
    success: bool,
    output: []const u8,
    err_msg: ?[]const u8 = null,
    execution_time_ms: u64 = 0,
};

/// Tools API handler
pub const ToolsAPI = struct {
    executor: *ToolExecutor,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, executor: *ToolExecutor) ToolsAPI {
        return .{
            .allocator = allocator,
            .executor = executor,
        };
    }

    /// POST /v1/tools/browser - Execute browser tool
    pub fn browserToolHandler(self: *ToolsAPI, req: *httpz.Request, res: *httpz.Response) !void {
        // Parse request body
        const body = req.body() orelse {
            res.status = 400;
            try res.json(.{ .@"error" ="Missing request body" }, .{});
            return;
        };

        const parsed_result = std.json.parseFromSlice(BrowserRequest, self.allocator, body, .{}) catch {
            res.status = 400;
            try res.json(.{ .@"error" ="Invalid JSON" }, .{});
            return;
        };
        defer parsed_result.deinit();
        const parsed = parsed_result.value;

        // Execute browser tool
        const args = try std.json.Stringify.valueAlloc(self.allocator, parsed, .{});
        defer self.allocator.free(args);

        const request = tools_types.ToolCallRequest{
            .name = "browser",
            .arguments = args,
        };

        var result = self.executor.executeToolCall(request) catch |err| {
            res.status = 500;
            try res.json(.{
                .@"error" ="Tool execution failed",
                .details = @errorName(err),
            }, .{});
            return;
        };
        defer result.deinit(self.allocator);

        // Return response
        try res.json(.{
            .success = result.success,
            .output = result.output,
            .@"error" =result.error_message,
            .execution_time_ms = result.execution_time_ms,
        }, .{});
    }

    /// POST /v1/tools/python - Execute Python tool
    pub fn pythonToolHandler(self: *ToolsAPI, req: *httpz.Request, res: *httpz.Response) !void {
        const body = req.body() orelse {
            res.status = 400;
            try res.json(.{ .@"error" ="Missing request body" }, .{});
            return;
        };

        const parsed_result = std.json.parseFromSlice(PythonRequest, self.allocator, body, .{}) catch {
            res.status = 400;
            try res.json(.{ .@"error" ="Invalid JSON" }, .{});
            return;
        };
        defer parsed_result.deinit();
        const parsed = parsed_result.value;

        // Execute Python tool
        const args = try std.json.Stringify.valueAlloc(self.allocator, parsed, .{});
        defer self.allocator.free(args);

        const request = tools_types.ToolCallRequest{
            .name = "python",
            .arguments = args,
        };

        var result = self.executor.executeToolCall(request) catch |err| {
            res.status = 500;
            try res.json(.{
                .@"error" ="Tool execution failed",
                .details = @errorName(err),
            }, .{});
            return;
        };
        defer result.deinit(self.allocator);

        try res.json(.{
            .success = result.success,
            .output = result.output,
            .@"error" =result.error_message,
            .execution_time_ms = result.execution_time_ms,
        }, .{});
    }

    /// GET /v1/tools - List available tools
    pub fn listToolsHandler(self: *ToolsAPI, req: *httpz.Request, res: *httpz.Response) !void {
        _ = req;

        const definitions = self.executor.getToolDefinitions() catch {
            res.status = 500;
            try res.json(.{ .@"error" ="Failed to get tool definitions" }, .{});
            return;
        };

        var tools = ArrayList(struct {
            name: []const u8,
            description: []const u8,
            parameters: []const u8,
        }).init(self.allocator);
        defer {
            for (tools.items) |*t| {
                self.allocator.free(t.name);
                self.allocator.free(t.description);
                self.allocator.free(t.parameters);
            }
            tools.deinit();
        }

        for (definitions) |def| {
            try tools.append(.{
                .name = def.name,
                .description = def.description,
                .parameters = def.parameters,
            });
        }

        try res.json(.{ .tools = tools.items }, .{});
    }
};
