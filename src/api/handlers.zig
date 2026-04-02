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
        try sendError(res, 500, "Server not initialized", "server_error", null);
        return;
    };

    // Get request body
    const body = req.body() orelse {
        std.log.err("Failed to read request body", .{});
        try sendError(res, 400, "Invalid request body", "invalid_request", null);
        return;
    };

    // Log the actual body for debugging (truncate if very large)
    const log_len = @min(body.len, 500);
    std.log.info("Request body length: {d}", .{body.len});
    if (body.len > 1000) {
        std.log.info("Request body (truncated): {s}...", .{body[0..log_len]});
    } else {
        std.log.info("Request body: {s}", .{body});
    }

    // Log first 200 chars as hex for debugging encoding issues
    if (body.len > 0) {
        const hex_len: usize = @min(body.len, 200);
        var hex_buf: [401]u8 = undefined;
        for (0..hex_len) |i| {
            _ = std.fmt.bufPrint(hex_buf[i * 2 .. i * 2 + 2], "{x:0>2}", .{body[i]}) catch break;
        }
        const slice_end: usize = hex_len * 2;
        std.log.info("Request body hex (first {d} bytes): {s}", .{ hex_len, hex_buf[0..slice_end] });
    }

    // Parse JSON request
    const parsed = std.json.parseFromSlice(
        types.ChatCompletionRequest,
        ctx.allocator,
        body,
        .{ .ignore_unknown_fields = true },
    ) catch |err| {
        std.log.err("Failed to parse request JSON: {s}", .{@errorName(err)});
        std.log.err("Body length: {d}", .{body.len});

        // Only log first 1000 chars of body on error to avoid log spam
        const err_log_len = @min(body.len, 1000);
        std.log.err("Body was: {s}", .{body[0..err_log_len]});
        if (body.len > 1000) {
            std.log.err("... (truncated, total length: {d})", .{body.len});
        }

        // Try to identify specific error position
        if (err == error.SyntaxError or err == error.UnexpectedToken) {
            // Find position of error by trying to parse with standard parser
            const dummy_parsed = std.json.parseFromSlice(
                std.json.Value,
                ctx.allocator,
                body,
                .{ .ignore_unknown_fields = true },
            ) catch |syntax_err| {
                std.log.err("Raw JSON syntax error: {s}", .{@errorName(syntax_err)});
                // Try to find where the error might be
                if (std.mem.indexOf(u8, body, "\x00") != null) {
                    std.log.err("ERROR: Body contains null bytes!", .{});
                }
                if (std.mem.indexOf(u8, body, "\n") != null) {
                    std.log.info("Body contains newlines (this is OK)", .{});
                }
                // Check for unmatched quotes
                var quote_count: usize = 0;
                var in_escape = false;
                for (body) |c| {
                    if (c == '"' and !in_escape) {
                        quote_count += 1;
                    }
                    in_escape = (c == '\\' and !in_escape);
                }
                if (quote_count % 2 != 0) {
                    std.log.err("ERROR: Unmatched quotes in JSON! Count: {d}", .{quote_count});
                }

                // Show body without newlines for easier debugging
                var clean_buf = try ctx.allocator.alloc(u8, body.len);
                defer ctx.allocator.free(clean_buf);
                for (body, 0..) |c, i| {
                    clean_buf[i] = if (c == '\n') ' ' else c;
                }
                std.log.err("Body (newlines replaced with spaces): {s}", .{clean_buf});

                return; // Return from the error handling
            };
            defer dummy_parsed.deinit();
            std.log.info("Raw JSON is valid, issue is with ChatCompletionRequest struct mapping", .{});

            // Try to identify which field is causing the issue
            const root = dummy_parsed.value;

            // List all fields in the JSON
            std.log.info("All fields in request:", .{});
            var field_iter = root.object.iterator();
            while (field_iter.next()) |entry| {
                std.log.info("  Field: {s} = {s}", .{ entry.key_ptr.*, @tagName(entry.value_ptr.*) });
            }

            // Try to validate each expected field
            std.log.info("Validating field types...", .{});

            // Check messages array structure
            if (root.object.get("messages")) |messages_val| {
                if (messages_val == .array) {
                    std.log.info("messages is array with {d} items", .{messages_val.array.items.len});
                    for (messages_val.array.items, 0..) |msg, i| {
                        if (msg == .object) {
                            if (msg.object.get("role")) |role_val| {
                                std.log.info("  Message {d} role type: {s}", .{ i, @tagName(role_val) });
                                if (role_val == .string) {
                                    std.log.info("  Message {d} role value: {s}", .{ i, role_val.string });
                                }
                            }
                            if (msg.object.get("content")) |content_val| {
                                std.log.info("  Message {d} content type: {s}, len: {d}", .{ i, @tagName(content_val), if (content_val == .string) content_val.string.len else 0 });
                            }
                        }
                    }
                }
            }

            if (root.object.get("model")) |model_val| {
                std.log.info("model field type: {s}", .{@typeName(@TypeOf(model_val))});
            }
            if (root.object.get("messages")) |messages_val| {
                std.log.info("messages field is present, type: {s}", .{@typeName(@TypeOf(messages_val))});
                if (messages_val == .array) {
                    std.log.info("messages is array with {d} items", .{messages_val.array.items.len});
                    if (messages_val.array.items.len > 0) {
                        const first_msg = messages_val.array.items[0];
                        if (first_msg == .object) {
                            if (first_msg.object.get("role")) |role_val| {
                                std.log.info("First message role: {s}", .{if (role_val == .string) role_val.string else "not string"});
                            }
                        }
                    }
                }
            }
            if (root.object.get("max_tokens")) |tokens_val| {
                std.log.info("max_tokens type: {s}", .{@typeName(@TypeOf(tokens_val))});
            }
            if (root.object.get("temperature")) |temp_val| {
                std.log.info("temperature type: {s}", .{@typeName(@TypeOf(temp_val))});
            }
            if (root.object.get("top_p")) |topp_val| {
                std.log.info("top_p type tag: {s}", .{@tagName(topp_val)});
            }
            if (root.object.get("stream")) |stream_val| {
                std.log.info("stream type: {s}", .{@typeName(@TypeOf(stream_val))});
            }
        }

        try sendError(res, 400, "Invalid JSON in request body", "invalid_request", null);
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
            try sendError(res, 404, "Model not found", "model_not_found", null);
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
        try sendError(res, 500, "Generation failed", "server_error", null);
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

/// Send JSON error response with request ID
fn sendError(res: anytype, status: u16, message: []const u8, error_type: []const u8, request_id: ?[]const u8) !void {
    res.status = status;
    res.content_type = .JSON;
    setCorsHeaders(res);

    var buf: [2048]u8 = undefined;
    const json = if (request_id) |rid|
        std.fmt.bufPrint(&buf, "{{\"error\":{{\"message\":\"{s}\",\"type\":\"{s}\"}},\"request_id\":\"{s}\"}}", .{ message, error_type, rid }) catch null
    else
        std.fmt.bufPrint(&buf, "{{\"error\":{{\"message\":\"{s}\",\"type\":\"{s}\"}}}}", .{ message, error_type }) catch null;

    if (json) |j| {
        try res.writer().writeAll(j);
    } else {
        try res.writer().writeAll("{\"error\":{\"message\":\"Internal error\",\"type\":\"server_error\"}}");
    }
}

/// Send 400 Bad Request error (per D-25)
fn sendBadRequestError(res: anytype, message: []const u8, param: ?[]const u8, request_id: []const u8) !void {
    res.status = 400;
    res.content_type = .JSON;
    setCorsHeaders(res);

    var buf: [2048]u8 = undefined;
    const param_field = if (param) |p| std.fmt.bufPrint(&buf, ",\"param\":\"{s}\"", .{p}) catch "" else "";
    const json = std.fmt.bufPrint(&buf, "{{\"error\":{{\"message\":\"{s}\",\"type\":\"invalid_request_error\"{s}}},\"request_id\":\"{s}\"}}", .{ message, param_field, request_id }) catch null;

    if (json) |j| {
        try res.writer().writeAll(j);
    }
}

/// Send 408 Request Timeout error (per D-26)
fn sendTimeoutError(res: anytype, message: []const u8, request_id: []const u8) !void {
    res.status = 408;
    res.content_type = .JSON;
    setCorsHeaders(res);

    var buf: [1024]u8 = undefined;
    const json = std.fmt.bufPrint(&buf, "{{\"error\":{{\"message\":\"{s}\",\"type\":\"timeout_error\"}},\"request_id\":\"{s}\"}}", .{ message, request_id }) catch null;

    if (json) |j| {
        try res.writer().writeAll(j);
    }
}

/// Send 500 Server Error with request ID (per D-27, D-31)
fn sendServerError(res: anytype, message: []const u8, request_id: []const u8) !void {
    res.status = 500;
    res.content_type = .JSON;
    setCorsHeaders(res);

    var buf: [1024]u8 = undefined;
    const json = std.fmt.bufPrint(&buf, "{{\"error\":{{\"message\":\"{s}\",\"type\":\"server_error\"}},\"request_id\":\"{s}\"}}", .{ message, request_id }) catch null;

    if (json) |j| {
        try res.writer().writeAll(j);
    }
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
