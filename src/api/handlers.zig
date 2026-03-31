//! handlers.zig - HTTP request handlers for OpenAI-compatible API
//!
//! Implements:
//! - POST /v1/chat/completions - Chat completions (streaming and non-streaming)
//! - GET /v1/models - List available models

const std = @import("std");
const types = @import("types.zig");
const streaming = @import("streaming.zig");
const inference = @import("../inference/mod.zig");

/// Global inference context - initialized at server startup
pub var global_context: ?*inference.InferenceContext = null;

/// CORS headers for all responses
const CORS_HEADERS = &[_]struct { []const u8, []const u8 }{
    .{ "Access-Control-Allow-Origin", "*" },
    .{ "Access-Control-Allow-Methods", "GET, POST, OPTIONS" },
    .{ "Access-Control-Allow-Headers", "Content-Type, Authorization" },
};

/// Handle POST /v1/chat/completions
pub fn handleChatCompletions(req: anytype, res: anytype) !void {
    // Set CORS headers
    setCorsHeaders(res);

    // Handle preflight OPTIONS request
    if (req.method == .OPTIONS) {
        res.status = 204;
        return;
    }

    // Ensure context is available
    const ctx = global_context orelse {
        try sendError(res, 500, "Server not initialized", "server_error");
        return;
    };

    // Get request body
    const body = req.body() orelse {
        std.log.err("Failed to read request body", .{});
        try sendError(res, 400, "Invalid request body", "invalid_request");
        return;
    };

    // Parse JSON request
    const parsed = std.json.parseFromSlice(
        types.ChatCompletionRequest,
        ctx.allocator,
        body,
        .{ .ignore_unknown_fields = true },
    ) catch |err| {
        std.log.err("Failed to parse request JSON: {s}", .{@errorName(err)});
        try sendError(res, 400, "Invalid JSON in request body", "invalid_request");
        return;
    };
    defer parsed.deinit();

    const request = parsed.value;

    // Validate model matches loaded model
    // For now, we support only the loaded model
    const loaded_model_name = std.fs.path.basename(ctx.model_path);
    if (!std.mem.eql(u8, request.model, loaded_model_name)) {
        // Allow if model name is a substring or vice versa (flexible matching)
        const model_matches = std.mem.indexOf(u8, loaded_model_name, request.model) != null or
            std.mem.indexOf(u8, request.model, loaded_model_name) != null;

        if (!model_matches) {
            try sendError(res, 404, "Model not found", "model_not_found");
            return;
        }
    }

    // Route to streaming or non-streaming handler
    if (request.stream) {
        try handleStreamingRequest(req, res, request, ctx);
    } else {
        try handleNonStreamingRequest(res, request, ctx);
    }
}

/// Handle streaming chat completion request
fn handleStreamingRequest(req: anytype, res: anytype, request: types.ChatCompletionRequest, ctx: *inference.InferenceContext) !void {
    _ = req; // Request already parsed, not used directly

    // Set SSE headers
    res.status = 200;
    res.content_type = .EVENTS;
    res.header("Cache-Control", "no-cache");
    res.header("Connection", "keep-alive");

    // Get response writer
    const writer = res.writer();

    // Generate streaming response
    // Note: This is a simplified implementation that writes directly
    // In production, we'd use async/await or proper streaming
    streaming.streamResponse(writer, request, ctx) catch |err| {
        std.log.err("Streaming error: {s}", .{@errorName(err)});
        try streaming.streamError(writer, "Generation failed", "server_error");
    };
}

/// Handle non-streaming chat completion request
fn handleNonStreamingRequest(res: anytype, request: types.ChatCompletionRequest, ctx: *inference.InferenceContext) !void {
    // Generate response
    const response_json = streaming.generateNonStreamingResponse(
        ctx.allocator,
        request,
        ctx,
    ) catch |err| {
        std.log.err("Generation error: {s}", .{@errorName(err)});
        try sendError(res, 500, "Generation failed", "server_error");
        return;
    };
    defer ctx.allocator.free(response_json);

    // Send response
    res.status = 200;
    res.content_type = .JSON;

    const writer = res.writer();
    try writer.writeAll(response_json);
}

/// Handle GET /v1/models
pub fn handleListModels(req: anytype, res: anytype) !void {
    // Set CORS headers
    setCorsHeaders(res);

    // Handle preflight OPTIONS request
    if (req.method == .OPTIONS) {
        res.status = 204;
        return;
    }

    // Get context to know which model is loaded
    const ctx = global_context orelse {
        // Return empty list if no context
        try sendJsonResponse(res, 200, "{\"object\":\"list\",\"data\":[]}");
        return;
    };

    // Get the loaded model name
    const model_name = std.fs.path.basename(ctx.model_path);

    // Build models response
    var json = std.ArrayList(u8).empty;
    errdefer json.deinit(ctx.allocator);

    const writer = json.writer(ctx.allocator);

    try writer.writeAll("{\"object\":\"list\",\"data\":[");

    // Add the loaded model
    const created = std.time.timestamp();
    try writer.print("{{\"id\":\"{s}\",\"object\":\"model\",\"created\":{d},\"owned_by\":\"local\"}}", .{
        model_name,
        created,
    });

    try writer.writeAll("]}");

    try sendJsonResponse(res, 200, json.items);
}

/// Handle GET /v1/health (health check endpoint)
pub fn handleHealth(req: anytype, res: anytype) !void {
    _ = req;

    setCorsHeaders(res);

    res.status = 200;
    res.content_type = .JSON;

    const ctx = global_context;
    const status = if (ctx != null) "healthy" else "initializing";
    const model = if (ctx) |c| std.fs.path.basename(c.model_path) else "none";

    var json_buf: [512]u8 = undefined;
    const response = try std.fmt.bufPrint(&json_buf, "{{\"status\":\"{s}\",\"model\":\"{s}\"}}", .{ status, model });

    const writer = res.writer();
    try writer.writeAll(response);
}

/// Send JSON error response
fn sendError(res: anytype, status: u16, message: []const u8, error_type: []const u8) !void {
    res.status = status;
    res.content_type = .JSON;
    setCorsHeaders(res);

    var buf: [1024]u8 = undefined;
    const json = std.fmt.bufPrint(&buf, "{{\"error\":{{\"message\":\"{s}\",\"type\":\"{s}\"}}}}", .{ message, error_type }) catch {
        // Fallback if formatting fails
        try res.writer().writeAll("{\"error\":{\"message\":\"Internal error\",\"type\":\"server_error\"}}");
        return;
    };

    try res.writer().writeAll(json);
}

/// Send JSON response with given body
fn sendJsonResponse(res: anytype, status: u16, body: []const u8) !void {
    res.status = status;
    res.content_type = .JSON;

    const writer = res.writer();
    try writer.writeAll(body);
}

/// Set CORS headers on response
fn setCorsHeaders(res: anytype) void {
    res.header("Access-Control-Allow-Origin", "*");
    res.header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.header("Access-Control-Allow-Headers", "Content-Type, Authorization");
}

/// Initialize global inference context
pub fn initGlobalContext(allocator: std.mem.Allocator, model_path: []const u8) !void {
    if (global_context != null) {
        // Already initialized
        return;
    }

    const ctx = try allocator.create(inference.InferenceContext);
    errdefer allocator.destroy(ctx);

    ctx.* = try inference.InferenceContext.init(allocator, model_path);

    global_context = ctx;
    std.log.info("Inference context initialized for model: {s}", .{model_path});
}

/// Deinitialize global inference context
pub fn deinitGlobalContext(allocator: std.mem.Allocator) void {
    if (global_context) |ctx| {
        ctx.deinit();
        allocator.destroy(ctx);
        global_context = null;
        std.log.info("Inference context deinitialized", .{});
    }
}
