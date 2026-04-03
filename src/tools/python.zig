//! python.zig - Python tool for code execution in Docker sandbox

const std = @import("std");
const tools_types = @import("types.zig");

const ToolCallRequest = tools_types.ToolCallRequest;
const ToolExecutionResult = tools_types.ToolExecutionResult;
const ToolDefinition = tools_types.ToolDefinition;
const Tool = tools_types.Tool;

/// Python execution result
pub const ExecutionResult = struct {
    stdout: []const u8,
    stderr: []const u8,
    exit_code: i32,
    execution_time_ms: u64,

    pub fn deinit(self: *ExecutionResult, allocator: std.mem.Allocator) void {
        allocator.free(self.stdout);
        allocator.free(self.stderr);
    }
};

/// Python tool configuration
pub const PythonConfig = struct {
    docker_image: []const u8 = "python:3.11-slim",
    timeout_ms: u64 = 60000,
    memory_limit_mb: usize = 512,
    cpu_limit: f32 = 1.0,
    network_enabled: bool = false,
    allowed_modules: []const []const u8 = &.{
        "math",        "random",    "datetime",  "json",       "re",    "string",
        "collections", "itertools", "functools", "statistics", "numpy", "pandas",
    },
    forbidden_patterns: []const []const u8 = &.{
        "__import__",
        "eval(",
        "exec(",
        "compile(",
        "open(",
        "os.system",
        "subprocess",
        "socket",
        "urllib",
    },
};

/// Python tool for sandboxed code execution
pub const PythonTool = struct {
    allocator: std.mem.Allocator,
    config: PythonConfig,

    pub fn init(allocator: std.mem.Allocator, config: PythonConfig) PythonTool {
        return .{
            .allocator = allocator,
            .config = config,
        };
    }

    pub fn deinit(self: *PythonTool) void {
        _ = self;
    }

    /// Execute Python code in Docker container
    pub fn execute(self: *PythonTool, code: []const u8) !ExecutionResult {
        const start_time = std.time.milliTimestamp();

        // Validate code for security
        try self.validateCode(code);

        // Build Docker command
        const docker_args = try self.buildDockerCommand(code);
        defer {
            for (docker_args) |arg| self.allocator.free(arg);
            self.allocator.free(docker_args);
        }

        // Execute Docker
        var stdout = std.ArrayList(u8).init(self.allocator);
        defer stdout.deinit();

        var stderr = std.ArrayList(u8).init(self.allocator);
        defer stderr.deinit();

        var process = std.process.Child.init(docker_args, self.allocator);
        process.stdout_behavior = .Pipe;
        process.stderr_behavior = .Pipe;

        try process.spawn();

        // Read output with timeout
        const stdout_reader = process.stdout.?.reader();
        const stderr_reader = process.stderr.?.reader();

        var stdout_buffer: [4096]u8 = undefined;
        var stderr_buffer: [4096]u8 = undefined;

        var stdout_done = false;
        var stderr_done = false;

        const deadline = start_time + @as(i64, @intCast(self.config.timeout_ms));

        while (!stdout_done or !stderr_done) {
            if (std.time.milliTimestamp() > deadline) {
                _ = try process.kill();
                return error.Timeout;
            }

            if (!stdout_done) {
                const n = try stdout_reader.read(&stdout_buffer);
                if (n == 0) {
                    stdout_done = true;
                } else {
                    try stdout.appendSlice(stdout_buffer[0..n]);
                }
            }

            if (!stderr_done) {
                const n = try stderr_reader.read(&stderr_buffer);
                if (n == 0) {
                    stderr_done = true;
                } else {
                    try stderr.appendSlice(stderr_buffer[0..n]);
                }
            }
        }

        const result = try process.wait();
        const end_time = std.time.milliTimestamp();

        return .{
            .stdout = try stdout.toOwnedSlice(),
            .stderr = try stderr.toOwnedSlice(),
            .exit_code = switch (result) {
                .Exited => |code| @intCast(code),
                .Signal => |sig| @intCast(sig),
                .Stopped => |sig| @intCast(sig),
                .Unknown => |sig| @intCast(sig),
            },
            .execution_time_ms = @intCast(end_time - start_time),
        };
    }

    /// Validate code for forbidden patterns
    pub fn validateCode(self: *const PythonTool, code: []const u8) !void {
        for (self.config.forbidden_patterns) |pattern| {
            if (std.mem.indexOf(u8, code, pattern) != null) {
                std.log.err("Python code contains forbidden pattern: {s}", .{pattern});
                return error.ForbiddenPattern;
            }
        }
    }

    /// Build Docker command for execution
    fn buildDockerCommand(self: *PythonTool, code: []const u8) ![][]const u8 {
        var args = std.ArrayList([]const u8).init(self.allocator);
        errdefer {
            for (args.items) |arg| self.allocator.free(arg);
            args.deinit();
        }

        try args.append(try self.allocator.dupe(u8, "docker"));
        try args.append(try self.allocator.dupe(u8, "run"));
        try args.append(try self.allocator.dupe(u8, "--rm"));
        try args.append(try self.allocator.dupe(u8, "-i"));

        // Network
        if (!self.config.network_enabled) {
            try args.append(try self.allocator.dupe(u8, "--network"));
            try args.append(try self.allocator.dupe(u8, "none"));
        }

        // Memory limit
        var mem_arg = try std.fmt.allocPrint(self.allocator, "--memory={d}m", .{self.config.memory_limit_mb});
        try args.append(mem_arg);

        // CPU limit
        var cpu_arg = try std.fmt.allocPrint(self.allocator, "--cpus={d:.1}", .{self.config.cpu_limit});
        try args.append(cpu_arg);

        // Timeout
        var timeout_arg = try std.fmt.allocPrint(self.allocator, "--stop-timeout={d}", .{
            self.config.timeout_ms / 1000,
        });
        try args.append(timeout_arg);

        // Security options
        try args.append(try self.allocator.dupe(u8, "--security-opt"));
        try args.append(try self.allocator.dupe(u8, "no-new-privileges:true"));
        try args.append(try self.allocator.dupe(u8, "--cap-drop"));
        try args.append(try self.allocator.dupe(u8, "ALL"));

        // Image
        try args.append(try self.allocator.dupe(u8, self.config.docker_image));

        // Python command
        try args.append(try self.allocator.dupe(u8, "python"));
        try args.append(try self.allocator.dupe(u8, "-c"));
        try args.append(try self.allocator.dupe(u8, code));

        return args.toOwnedSlice();
    }

    /// Execute tool call
    pub fn executeToolCall(self: *PythonTool, request: ToolCallRequest, allocator: std.mem.Allocator) !ToolExecutionResult {
        const Args = struct {
            code: []const u8,
        };

        const args = try std.json.parseFromSlice(Args, allocator, request.arguments, .{});
        defer std.json.parseFree(Args, allocator, args);

        const result = try self.execute(args.code);
        defer result.deinit(allocator);

        var output = std.ArrayList(u8).init(allocator);
        errdefer output.deinit();

        if (result.stdout.len > 0) {
            try output.appendSlice("Output:\n");
            try output.appendSlice(result.stdout);
        }

        if (result.stderr.len > 0) {
            if (output.items.len > 0) try output.appendSlice("\n\n");
            try output.appendSlice("Errors:\n");
            try output.appendSlice(result.stderr);
        }

        const success = result.exit_code == 0;

        return .{
            .success = success,
            .output = try output.toOwnedSlice(),
            .error_message = if (!success) try allocator.dupe(u8, "Execution failed") else null,
            .execution_time_ms = result.execution_time_ms,
        };
    }

    /// Get tool definition
    pub fn getToolDefinition(self: *PythonTool, allocator: std.mem.Allocator) !ToolDefinition {
        _ = self;
        return .{
            .name = try allocator.dupe(u8, "python"),
            .description = try allocator.dupe(u8, "Execute Python code in a sandboxed Docker environment. Supports math, data analysis with numpy/pandas. No network access."),
            .parameters = try allocator.dupe(u8,
                \\{{"type": "object", "properties": {{"code": {{"type": "string", "description": "Python code to execute"}}}}, "required": ["code"]}}
            ),
        };
    }

    // Tool interface implementation

    pub fn asTool(self: *PythonTool) Tool {
        return .{
            .vtable = &.{
                .execute = toolExecute,
                .getDefinition = toolGetDefinition,
                .deinit = toolDeinit,
            },
            .ptr = self,
        };
    }

    fn toolExecute(ctx: *anyopaque, request: ToolCallRequest, allocator: std.mem.Allocator) anyerror!ToolExecutionResult {
        const self = @as(*PythonTool, @ptrCast(@alignCast(ctx)));
        return self.executeToolCall(request, allocator);
    }

    fn toolGetDefinition(ctx: *anyopaque, allocator: std.mem.Allocator) anyerror!ToolDefinition {
        const self = @as(*PythonTool, @ptrCast(@alignCast(ctx)));
        return self.getToolDefinition(allocator);
    }

    fn toolDeinit(ctx: *anyopaque, allocator: std.mem.Allocator) void {
        const self = @as(*PythonTool, @ptrCast(@alignCast(ctx)));
        self.deinit();
        allocator.destroy(self);
    }
};
