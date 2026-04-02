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

/// Global timeout configuration (default 60s per D-32)
pub var request_timeout_seconds: u32 = 60;

/// Get timeout in milliseconds
fn getTimeoutMs() u64 {
    return @as(u64, request_timeout_seconds) * 1000;
}

/// CORS headers for all responses
const CORS_HEADERS = &[_]struct { []const u8, []const u8 }{
    .{ "Access-Control-Allow-Origin", "*" },
    .{ "Access-Control-Allow-Methods", "GET, POST, OPTIONS" },
    .{ "Access-Control-Allow-Headers", "Content-Type, Authorization" },
};

/// Handle POST /v1/chat/completions
pub fn handleChatCompletions(req: anytype, res: anytype) !void {
    // Generate request ID for tracking this request (per D-28)
    const request_id = types.generateRequestId(std.heap.page_allocator) catch "req-unknown";
    defer if (request_id.ptr != "req-unknown".ptr) {
        std.heap.page_allocator.free(request_id);
    };

    // Set CORS headers
    setCorsHeaders(res);

    // Handle preflight OPTIONS request
    if (req.method == .OPTIONS) {
        res.status = 204;
        return;
    }

    // Ensure context is available
    const ctx = global_context orelse {
        std.log.err("[{s}] Server not initialized", .{request_id});
        try sendServerError(res, "Server not initialized", request_id);
        return;
    };

    // Get request body
    const body = req.body() orelse {
        std.log.err("[{s}] Failed to read request body", .{request_id});
        try sendBadRequestError(res, "Invalid request body", null, request_id);
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
        std.log.err("[{s}] Failed to parse request JSON: {s}", .{ request_id, @errorName(err) });
        std.log.err("[{s}] Body length: {d}", .{ request_id, body.len });

        // Only log first 1000 chars of body on error to avoid log spam
        const err_log_len = @min(body.len, 1000);
        std.log.err("[{s}] Body was: {s}", .{ request_id, body[0..err_log_len] });
        if (body.len > 1000) {
            std.log.err("[{s}] ... (truncated, total length: {d})", .{ request_id, body.len });
        }

        // Provide helpful error message based on error type (per D-25)
        const error_message = switch (err) {
            error.SyntaxError => "Invalid JSON syntax in request body",
            error.UnexpectedToken => "Unexpected token in JSON",
            error.InvalidNumber => "Invalid number in JSON",
            error.InvalidCharacters => "Invalid characters in JSON",
            error.InvalidUtf8 => "Invalid UTF-8 in request body",
            else => "Invalid JSON in request body",
        };

        try sendBadRequestError(res, error_message, null, request_id);
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
            std.log.warn("[{s}] Model not found: requested={s}, loaded={s}", .{ request_id, request.model, loaded_model_name });
            try sendError(res, 404, "Model not found", "model_not_found", request_id);
            return;
        }
    }

    // Route to streaming or non-streaming handler with request_id
    if (request.stream) {
        try handleStreamingRequest(req, res, request, ctx, request_id);
    } else {
        try handleNonStreamingRequest(res, request, ctx, request_id);
    }
}

/// Handle streaming chat completion request
fn handleStreamingRequest(req: anytype, res: anytype, request: types.ChatCompletionRequest, ctx: *inference.InferenceContext, request_id: []const u8) !void {
    _ = req; // Request already parsed, not used directly
    std.log.info("[{s}] Starting streaming generation", .{request_id});

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
        std.log.err("[{s}] Streaming error: {s}", .{ request_id, @errorName(err) });
        try streaming.streamError(writer, "Generation failed", "server_error");
    };
}

/// Handle non-streaming chat completion request
fn handleNonStreamingRequest(res: anytype, request: types.ChatCompletionRequest, ctx: *inference.InferenceContext, request_id: []const u8) !void {
    // Build prompt from messages
    const prompt = try types.buildPromptFromMessages(ctx.allocator, request.messages);
    defer ctx.allocator.free(prompt);

    // Create generation options
    const gen_options = inference.GenerationOptions{
        .max_tokens = request.getMaxTokens(),
        .temperature = request.getTemperature(),
        .top_p = request.getTopP(),
        .stop_on_eos = true,
        .seed = if (request.seed) |s| @intCast(s) else null,
        .top_k = request.getTopK(),
        .min_p = request.getMinP(),
        .presence_penalty = request.getPresencePenalty(),
        .frequency_penalty = request.getFrequencyPenalty(),
        .repetition_penalty = request.getRepetitionPenalty(),
        .logprobs_enabled = request.logprobs orelse false,
    };

    // Generate with timeout
    const timeout_ms = getTimeoutMs();
    const timeout_result = ctx.generateWithTimeout(prompt, gen_options, timeout_ms) catch |err| {
        std.log.err("[{s}] Generation failed: {s}", .{ request_id, @errorName(err) });
        try sendServerError(res, "Generation failed", request_id);
        return;
    };
    defer {
        ctx.allocator.free(timeout_result.result.text);
        if (timeout_result.result.logprobs) |lp| {
            for (lp) |*entry| entry.deinit(ctx.allocator);
            ctx.allocator.free(lp);
        }
    }

    // Check if timed out and return 408 if so
    if (timeout_result.timed_out) {
        std.log.warn("[{s}] Request timed out", .{request_id});
        try sendTimeoutError(res, "Request exceeded timeout limit", request_id);
        return;
    }

    // Build and send the response
    const response_json = buildChatCompletionResponse(
        ctx.allocator,
        request,
        timeout_result.result,
    ) catch |err| {
        std.log.err("[{s}] Failed to build response: {s}", .{ request_id, @errorName(err) });
        try sendServerError(res, "Failed to build response", request_id);
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

/// Build chat completion response JSON from generation result
fn buildChatCompletionResponse(
    allocator: std.mem.Allocator,
    request: types.ChatCompletionRequest,
    result: inference.GenerationResult,
) ![]const u8 {
    const completion_id = try types.generateCompletionId(allocator);
    defer allocator.free(completion_id);

    const created_timestamp = std.time.timestamp();

    // Determine finish_reason
    const finish_reason = switch (result.stop_reason) {
        .eos => "stop",
        .length => "length",
        .stop => "stop",
        .timeout => "timeout",
    };

    var json = std.ArrayList(u8).empty;
    errdefer json.deinit(allocator);

    const writer = json.writer(allocator);

    try writer.writeAll("{\"id\":\"");
    try writeJsonString(writer, completion_id);
    try writer.print("\",\"object\":\"chat.completion\",\"created\":{d},\"model\":\"", .{created_timestamp});
    try writeJsonString(writer, request.model);
    try writer.writeAll("\",\"choices\":[");

    // Single choice
    try writer.writeAll("{\"index\":0,\"message\":{");
    try writer.writeAll("\"role\":\"assistant\",\"content\":\"");
    try writeJsonString(writer, result.text);
    try writer.writeAll("\"},\"finish_reason\":\"");
    try writer.writeAll(finish_reason);
    try writer.writeAll("\"}");

    try writer.writeAll("],\"usage\":");

    // Usage stats
    try writer.print("{{\"prompt_tokens\":{d},\"completion_tokens\":{d},\"total_tokens\":{d}}}}}", .{
        result.prompt_tokens,
        result.completion_tokens,
        result.prompt_tokens + result.completion_tokens,
    });

    return json.toOwnedSlice(allocator);
}

/// Write a string with JSON escaping
fn writeJsonString(writer: anytype, str: []const u8) !void {
    for (str) |c| {
        switch (c) {
            '"' => try writer.writeAll("\\\""),
            '\\' => try writer.writeAll("\\\\"),
            0x08 => try writer.writeAll("\\b"), // backspace
            0x0C => try writer.writeAll("\\f"), // form feed
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x0B, 0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F => try writer.print("\\u{x:0>4}", .{c}),
            else => try writer.writeByte(c),
        }
    }
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
