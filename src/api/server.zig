//! server.zig - HTTP server setup with httpz
//!
//! Configures httpz server with proper routing and middleware.

const std = @import("std");
const httpz = @import("httpz");
const handlers = @import("handlers.zig");
const chat_gptoss = @import("chat_gptoss.zig");
const tools_api_mod = @import("tools.zig");
const tool_executor_mod = @import("../tools/tool_executor.zig");
const mlx_gptoss_backend_mod = @import("../backends/mlx_gptoss_backend.zig");

/// Server configuration
pub const ServerConfig = struct {
    /// Port to listen on
    port: u16 = 8080,
    /// Address to bind to
    address: []const u8 = "127.0.0.1",
    /// Request timeout in milliseconds (0 = no timeout)
    timeout_ms: u32 = 0,
    /// Maximum request body size in bytes
    max_body_size: usize = 10 * 1024 * 1024, // 10MB
    /// Request timeout in seconds (default 60s per D-32)
    timeout_seconds: u32 = 60,
};

/// Global server instance for cleanup
var g_server: ?httpz.Server(void) = null;

/// Module-level MLXGPTOSSBackend — initialized in Server.init
var g_gptoss_backend: ?mlx_gptoss_backend_mod.MLXGPTOSSBackend = null;
/// Module-level ChatGPTOSSHandler — requires g_gptoss_backend to be initialized first
var g_chat_gptoss_handler: ?chat_gptoss.ChatGPTOSSHandler = null;
/// Module-level ToolsAPI instance for /v1/tools/* route dispatch
var g_tools_api: ?tools_api_mod.ToolsAPI = null;
/// Module-level ToolExecutor backing g_tools_api
var g_tool_executor: ?tool_executor_mod.ToolExecutor = null;

/// HTTP server state
pub const Server = struct {
    /// The underlying httpz server
    http_server: httpz.Server(void),
    /// Allocator used by the server
    allocator: std.mem.Allocator,

    /// Initialize and start the HTTP server
    pub fn init(allocator: std.mem.Allocator, config: ServerConfig) !Server {
        // Create httpz server config
        const httpz_config = httpz.Config{
            .address = if (std.mem.eql(u8, config.address, "0.0.0.0"))
                .all(config.port)
            else
                .{ .ip = .{ .host = config.address, .port = config.port } },
        };

        // Create httpz server
        var server = try httpz.Server(void).init(allocator, httpz_config, {});

        // Initialize GPT-OSS backend and handler (model loaded lazily on first request)
        g_gptoss_backend = try mlx_gptoss_backend_mod.MLXGPTOSSBackend.init(
            allocator,
            "", // model_path — empty placeholder; real path set at load time via /v1/models/switch
            .{},
        );
        g_chat_gptoss_handler = chat_gptoss.ChatGPTOSSHandler.init(allocator, &g_gptoss_backend.?);

        // Initialize tool executor and ToolsAPI for /v1/tools/* routes
        g_tool_executor = tool_executor_mod.ToolExecutor.init(allocator);
        g_tools_api = tools_api_mod.ToolsAPI.init(allocator, &g_tool_executor.?);

        // Configure router
        var router = try server.router(.{});

        // Register routes
        // POST /v1/chat/completions
        router.post("/v1/chat/completions", handleChatCompletions, .{});

        // GET /v1/models
        router.get("/v1/models", handleListModels, .{});

        // POST /v1/models/switch (model switching endpoint)
        router.post("/v1/models/switch", handleSwitchModel, .{});
        router.options("/v1/models/switch", handleOptions, .{});

        // GET /v1/health (health check)
        router.get("/v1/health", handleHealth, .{});

        // GET /v1/metrics (Prometheus metrics)
        router.get("/v1/metrics", handleMetrics, .{});

        // POST /v1/models/load (background model loading)
        router.post("/v1/models/load", handleLoadModel, .{});
        router.options("/v1/models/load", handleOptions, .{});

        // GET /v1/models/load-status (check background load progress)
        router.get("/v1/models/load-status", handleLoadStatus, .{});
        router.options("/v1/models/load-status", handleOptions, .{});

        // POST /v1/models/load/cancel (cancel background load)
        router.post("/v1/models/load/cancel", handleCancelLoad, .{});
        router.options("/v1/models/load/cancel", handleOptions, .{});

        // OPTIONS handler for CORS preflight (handled in individual handlers)
        router.options("/v1/chat/completions", handleOptions, .{});
        router.options("/v1/models", handleOptions, .{});

        // POST /v1/tools/browser - Browser tool endpoint
        router.post("/v1/tools/browser", handleBrowserTool, .{});
        router.options("/v1/tools/browser", handleOptions, .{});

        // POST /v1/tools/python - Python tool endpoint
        router.post("/v1/tools/python", handlePythonTool, .{});
        router.options("/v1/tools/python", handleOptions, .{});

        // Store for cleanup
        g_server = server;

        std.log.info("HTTP server configured on {s}:{d}", .{ config.address, config.port });

        return Server{
            .http_server = server,
            .allocator = allocator,
        };
    }

    /// Start listening for connections (blocking)
    pub fn start(self: *Server) !void {
        std.log.info("Starting HTTP server...", .{});
        try self.http_server.listen();
    }

    /// Stop the server
    pub fn stop(self: *Server) void {
        std.log.info("Stopping HTTP server...", .{});
        self.http_server.stop();
        if (g_gptoss_backend) |*b| b.deinit();
        g_gptoss_backend = null;
        g_chat_gptoss_handler = null;
        if (g_tool_executor) |*te| te.deinit();
        g_tools_api = null;
        g_tool_executor = null;
        g_server = null;
    }
};

/// Wrapper for chat completions handler
/// Peeks at the model field and dispatches to ChatGPTOSSHandler for gpt-oss models.
fn handleChatCompletions(req: *httpz.Request, res: *httpz.Response) !void {
    // Peek at model field to dispatch GPT-OSS requests to dedicated handler
    const body = req.body() orelse {
        try handlers.handleChatCompletions(req, res);
        return;
    };

    const ModelPeek = struct { model: []const u8 = "" };
    const peek = std.json.parseFromSlice(ModelPeek, req.arena, body, .{ .ignore_unknown_fields = true }) catch {
        // Parse failed — fall through to generic handler which will return proper error
        try handlers.handleChatCompletions(req, res);
        return;
    };
    defer peek.deinit();

    if (std.mem.startsWith(u8, peek.value.model, "gpt-oss") or
        std.mem.startsWith(u8, peek.value.model, "gptoss"))
    {
        if (g_chat_gptoss_handler) |*handler| {
            // ChatGPTOSSHandler.handle is an instance method — must call on *handler
            try handler.handle(req, res);
        } else {
            res.status = 503;
            try res.json(.{ .@"error" = "GPT-OSS handler not initialized" }, .{});
        }
    } else {
        try handlers.handleChatCompletions(req, res);
    }
}

/// Wrapper for list models handler
fn handleListModels(req: *httpz.Request, res: *httpz.Response) !void {
    try handlers.handleListModels(req, res);
}

/// Wrapper for switch model handler
fn handleSwitchModel(req: *httpz.Request, res: *httpz.Response) !void {
    try handlers.handleSwitchModel(req, res);
}

/// Wrapper for load model handler (background loading)
fn handleLoadModel(req: *httpz.Request, res: *httpz.Response) !void {
    try handlers.handleLoadModel(req, res);
}

/// Wrapper for load status handler
fn handleLoadStatus(req: *httpz.Request, res: *httpz.Response) !void {
    try handlers.handleLoadStatus(req, res);
}

/// Wrapper for cancel load handler
fn handleCancelLoad(req: *httpz.Request, res: *httpz.Response) !void {
    try handlers.handleCancelLoad(req, res);
}

/// Wrapper for health check handler
fn handleHealth(req: *httpz.Request, res: *httpz.Response) !void {
    try handlers.handleHealth(req, res);
}

/// Wrapper for metrics handler
fn handleMetrics(req: *httpz.Request, res: *httpz.Response) !void {
    try handlers.handleMetrics(req, res);
}

/// Options handler for CORS preflight
fn handleOptions(req: *httpz.Request, res: *httpz.Response) !void {
    _ = req;

    // Set CORS headers
    res.header("Access-Control-Allow-Origin", "*");
    res.header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.header("Access-Control-Allow-Headers", "Content-Type, Authorization");

    res.status = 204; // No content
}

/// Browser tool endpoint wrapper
fn handleBrowserTool(req: *httpz.Request, res: *httpz.Response) !void {
    if (g_tools_api) |*api| {
        try api.browserToolHandler(req, res);
    } else {
        res.status = 503;
        try res.json(.{ .@"error" = "Tools not initialized" }, .{});
    }
}

/// Python tool endpoint wrapper
fn handlePythonTool(req: *httpz.Request, res: *httpz.Response) !void {
    if (g_tools_api) |*api| {
        try api.pythonToolHandler(req, res);
    } else {
        res.status = 503;
        try res.json(.{ .@"error" = "Tools not initialized" }, .{});
    }
}

/// Run the server with the given configuration
/// This is a convenience function that creates, starts, and manages the server
pub fn runServer(allocator: std.mem.Allocator, config: ServerConfig) !void {
    var server = try Server.init(allocator, config);

    // Set up signal handler for graceful shutdown
    const sigaction = std.posix.Sigaction{
        .handler = .{ .handler = signalHandler },
        .mask = std.posix.sigemptyset(),
        .flags = 0,
    };
    std.posix.sigaction(std.posix.SIG.INT, &sigaction, null);
    std.posix.sigaction(std.posix.SIG.TERM, &sigaction, null);

    std.log.info("Server running at http://{s}:{d}/v1", .{ config.address, config.port });
    std.log.info("Press Ctrl+C to stop", .{});

    // Start server (blocking)
    try server.start();
}

/// Signal handler for graceful shutdown
fn signalHandler(sig: c_int) callconv(.c) void {
    _ = sig;
    std.log.info("\nShutdown signal received", .{});
    if (g_server) |*server| {
        server.stop();
    }
}

/// Create a simple test server that can be used for testing
pub fn createTestServer(allocator: std.mem.Allocator, port: u16) !Server {
    return Server.init(allocator, .{
        .port = port,
        .address = "127.0.0.1",
    });
}
