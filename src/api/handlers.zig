//! handlers.zig - HTTP request handlers for OpenAI-compatible API
//!
//! Implements:
//! - POST /v1/chat/completions - Chat completions (streaming and non-streaming)
//! - GET /v1/models - List available models

const std = @import("std");
const types = @import("types.zig");
const streaming = @import("streaming.zig");
const inference = @import("../inference/mod.zig");
const models_mod = @import("../models/mod.zig");
const manager_mod = @import("../models/manager.zig");
const prompt_cache = @import("../cache/prompt_cache.zig");
const templates = @import("../chat/templates.zig");
const model_registry = @import("../models/registry.zig");

/// Global inference context - initialized at server startup
pub var global_context: ?*inference.InferenceContext = null;

/// Global timeout configuration (default 60s per D-32)
pub var request_timeout_seconds: u32 = 60;

/// Get timeout in milliseconds
fn getTimeoutMs() u64 {
    return @as(u64, request_timeout_seconds) * 1000;
}

/// CORS headers for all responses (configurable via config.cors_origins)
/// Open WebUI compatibility: default "*" allows any origin including localhost:8081
fn setCorsHeaders(res: anytype) void {
    res.header("Access-Control-Allow-Origin", "*");
    res.header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, DELETE");
    res.header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With");
    res.header("Access-Control-Max-Age", "86400"); // 24 hour preflight cache
}

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
            error.InvalidCharacter => "Invalid character in JSON",
            else => "Invalid JSON in request body",
        };

        try sendBadRequestError(res, error_message, null, request_id);
        return;
    };
    defer parsed.deinit();

    const request = parsed.value;

    // Check if model switching is needed
    const loaded_model_name = std.fs.path.basename(ctx.model_path);

    // Try to get the manager for automatic model switching
    if (manager_mod.getGlobalManager()) |manager| {
        const current_model = manager.getCurrentModel();

        // Check if we need to switch models
        if (current_model == null or !std.mem.eql(u8, current_model.?, request.model)) {
            std.log.info("[{s}] Requested model '{s}' differs from current, attempting switch...", .{
                request_id, request.model,
            });

            // Check if model is available
            if (models_mod.getGlobalRegistry()) |reg| {
                if (reg.getModel(request.model) == null) {
                    try sendError(res, 404, "Model not found in registry", "model_not_found", request_id);
                    return;
                }
            }

            // Check memory availability
            if (!manager.canLoadModel(request.model)) {
                try sendError(res, 503, "Insufficient memory to load model", "insufficient_memory", request_id);
                return;
            }

            // Perform the switch
            const switch_start = std.time.milliTimestamp();
            manager.switchModel(request.model) catch |err| {
                std.log.err("[{s}] Failed to switch to model '{s}': {s}", .{
                    request_id, request.model, @errorName(err),
                });
                const error_msg = switch (err) {
                    error.ModelNotFound => "Model not found",
                    error.InsufficientMemory => "Insufficient memory",
                    error.ModelLoadFailed => "Failed to load model",
                    else => "Model switch failed",
                };
                try sendError(res, 500, error_msg, "model_switch_failed", request_id);
                return;
            };
            const switch_duration = @as(u64, @intCast(std.time.milliTimestamp() - switch_start));
            std.log.info("[{s}] Model switch completed in {d}ms", .{ request_id, switch_duration });

            // Context will be reloaded from global_context in the handlers
        }
    } else {
        // No manager available - use legacy model validation
        if (!std.mem.eql(u8, request.model, loaded_model_name)) {
            const model_matches = std.mem.indexOf(u8, loaded_model_name, request.model) != null or
                std.mem.indexOf(u8, request.model, loaded_model_name) != null;

            if (!model_matches) {
                std.log.warn("[{s}] Model not found: requested={s}, loaded={s}", .{ request_id, request.model, loaded_model_name });
                try sendError(res, 404, "Model not found", "model_not_found", request_id);
                return;
            }
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

    // Notify manager that generation is starting
    const manager = manager_mod.getGlobalManager();
    if (manager) |m| {
        m.startGeneration();
    }
    defer {
        if (manager) |m| {
            m.endGeneration();
        }
    }

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
    // Notify manager that generation is starting
    const manager = manager_mod.getGlobalManager();
    if (manager) |m| {
        m.startGeneration();
    }
    defer {
        if (manager) |m| {
            m.endGeneration();
        }
    }

    // Build prompt from messages using model-specific template
    const arch = detectModelArchitecture(request.model);
    const prompt = try templates.formatChatByArchitecture(ctx.allocator, arch, request.messages);
    defer ctx.allocator.free(prompt);

    // Log template selection for debugging
    std.log.info("[{s}] Using {s} chat template for model: {s}", .{
        request_id,
        @tagName(arch),
        request.model,
    });

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

    // Generate cache key and check cache (if enabled)
    const cache_hit = false; // Placeholder - actual cache integration in generation layer
    const cache_key: ?[]const u8 = blk: {
        if (prompt_cache.getGlobalCache()) |cache| {
            // Get model identifier from context
            const model_name = std.fs.path.basename(ctx.model_path);
            // Use model path as hash since it's unique
            const key = cache.generateKey(model_name, ctx.model_path, prompt, gen_options) catch |err| {
                std.log.warn("[{s}] Failed to generate cache key: {s}", .{ request_id, @errorName(err) });
                break :blk null;
            };
            const key_str = try ctx.allocator.dupe(u8, key.slice());
            break :blk key_str;
        }
        break :blk null;
    };
    defer if (cache_key) |key| ctx.allocator.free(key);

    // Log cache status for debugging
    if (cache_key) |key| {
        std.log.debug("[{s}] Cache key: {s}, hit: {s}", .{ request_id, key, if (cache_hit) "true" else "false" });
    }

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
            // Cast away const to deinit - safe since we're in defer cleanup
            for (@constCast(lp)) |*entry| entry.deinit(ctx.allocator);
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

    // Get the global registry (already imported at top of file)
    const registry = models_mod.getGlobalRegistry();

    // Build models response
    var json = std.ArrayList(u8).empty;
    errdefer json.deinit(std.heap.page_allocator);

    const writer = json.writer(std.heap.page_allocator);

    try writer.writeAll("{\"object\":\"list\",\"data\":[");

    if (registry) |reg| {
        // Get all models from registry
        const allocator = std.heap.page_allocator;
        const models = reg.getAllModels(allocator) catch |err| {
            std.log.err("Failed to get models from registry: {s}", .{@errorName(err)});
            try writer.writeAll("]}");
            try sendJsonResponse(res, 200, json.items);
            return;
        };
        defer allocator.free(models);

        const now = std.time.timestamp();

        for (models, 0..) |model, i| {
            if (i > 0) try writer.writeAll(",");

            // Determine architecture from model info
            const architecture = determineArchitecture(&model.config);

            // Determine status string
            const status_str = switch (model.status) {
                .available => "available",
                .loading => "loading",
                .loaded => "loaded",
                .failed => "error",
            };

            // Build model entry with metadata
            try writer.writeAll("{");
            try writer.print("\"id\":\"{s}\",", .{model.id});
            try writer.writeAll("\"object\":\"model\",");
            try writer.print("\"created\":{d},", .{now});
            try writer.writeAll("\"owned_by\":\"local\",");
            try writer.writeAll("\"metadata\":{");
            try writer.print("\"status\":\"{s}\",", .{status_str});
            try writer.print("\"size_bytes\":{d},", .{model.size_bytes});
            try writer.print("\"memory_required_mb\":{d},", .{model.memory_required_mb});
            try writer.print("\"architecture\":\"{s}\"", .{architecture});
            if (model.loaded_at) |loaded_at| {
                try writer.print(",\"loaded_at\":{d}", .{loaded_at});
            }
            try writer.writeAll("}");
            try writer.writeAll("}");
        }
    } else {
        // Fallback: if no registry, just show the currently loaded model
        if (global_context) |ctx| {
            const model_name = std.fs.path.basename(ctx.model_path);
            const now = std.time.timestamp();
            try writer.print("{{\"id\":\"{s}\",\"object\":\"model\",\"created\":{d},\"owned_by\":\"local\",\"metadata\":{{\"status\":\"loaded\",\"size_bytes\":0,\"memory_required_mb\":0,\"architecture\":\"unknown\"}}}}", .{ model_name, now });
        }
    }

    try writer.writeAll("]}");

    try sendJsonResponse(res, 200, json.items);
}

/// Determine architecture string from model configuration
fn determineArchitecture(config: *const @import("../models/registry.zig").ConfigInfo) []const u8 {
    // Estimate parameters to guess architecture
    const params = config.estimateParameterCount();

    // Rough parameter-based detection (config would be better but we don't have model type here)
    if (params < 3_000_000_000) {
        // Small models often use Qwen architecture
        return "qwen";
    } else if (params < 10_000_000_000) {
        // Medium models could be Qwen, Llama, or Phi
        return "qwen"; // Default guess
    } else {
        return "qwen";
    }
}

/// Detect model architecture from model name
fn detectModelArchitecture(model_name: []const u8) templates.ModelArchitecture {
    // Check for DeepSeek models
    if (std.mem.indexOf(u8, model_name, "deepseek") != null) {
        if (std.mem.indexOf(u8, model_name, "v2") != null or
            std.mem.indexOf(u8, model_name, "coder-v2") != null)
        {
            return .deepseek_v2_moe;
        }
        return .deepseek_v2_moe; // Default DeepSeek to V2 MoE
    }

    // Check for Qwen models
    if (std.mem.indexOf(u8, model_name, "qwen") != null) {
        return .qwen;
    }

    // Check for Llama models
    if (std.mem.indexOf(u8, model_name, "llama") != null) {
        return .llama;
    }

    // Check for Phi models
    if (std.mem.indexOf(u8, model_name, "phi") != null) {
        return .phi;
    }

    // Default to Qwen (most common in this codebase)
    return .qwen;
}

/// Handle POST /v1/models/load - Start background model load
pub fn handleLoadModel(req: anytype, res: anytype) !void {
    // Set CORS headers
    setCorsHeaders(res);

    // Handle preflight OPTIONS request
    if (req.method == .OPTIONS) {
        res.status = 204;
        return;
    }

    // Get request body
    const body = req.body() orelse {
        try sendBadRequestError(res, "Missing request body", null, "req-load");
        return;
    };

    // Parse JSON request
    const parsed = std.json.parseFromSlice(
        types.LoadModelRequest,
        std.heap.page_allocator,
        body,
        .{ .ignore_unknown_fields = true },
    ) catch |err| {
        std.log.err("Failed to parse load model request: {s}", .{@errorName(err)});
        try sendBadRequestError(res, "Invalid JSON in request body", null, "req-load");
        return;
    };
    defer parsed.deinit();

    const request = parsed.value;

    // Get the manager
    const manager = manager_mod.getGlobalManager();
    if (manager == null) {
        try sendError(res, 503, "Model manager not available", "service_unavailable", "req-load");
        return;
    }
    const m = manager.?;

    // Check if model exists in registry
    if (models_mod.getGlobalRegistry()) |reg| {
        if (reg.getModel(request.model) == null) {
            try sendError(res, 404, "Model not found in registry", "model_not_found", "req-load");
            return;
        }
    } else {
        try sendError(res, 503, "Model registry not available", "service_unavailable", "req-load");
        return;
    }

    // Check memory availability
    if (!m.canLoadModel(request.model)) {
        try sendError(res, 503, "Insufficient memory to load model", "insufficient_memory", "req-load");
        return;
    }

    // Start background load
    m.startBackgroundLoad(request.model, request.auto_switch) catch |err| {
        std.log.err("Failed to start background load for '{s}': {s}", .{
            request.model,
            @errorName(err),
        });
        const error_msg = switch (err) {
            error.LoadInProgress => "Another model is already loading in background",
            error.ModelNotFound => "Model not found",
            error.InsufficientMemory => "Insufficient memory",
            else => "Failed to start background load",
        };
        try sendError(res, 503, error_msg, "load_failed", "req-load");
        return;
    };

    // Return 202 Accepted
    var json = std.ArrayList(u8).empty;
    errdefer json.deinit(std.heap.page_allocator);
    const writer = json.writer(std.heap.page_allocator);

    try writer.writeAll("{");
    try writer.writeAll("\"status\":\"loading_started\",");
    try writer.print("\"model\":\"{s}\",", .{request.model});
    try writer.print("\"auto_switch\":{s}", .{if (request.auto_switch) "true" else "false"});
    try writer.writeAll("}");

    res.status = 202; // Accepted
    res.content_type = .JSON;
    try res.writer().writeAll(json.items);
}

/// Handle GET /v1/models/load-status - Get background load progress
pub fn handleLoadStatus(req: anytype, res: anytype) !void {
    // Set CORS headers
    setCorsHeaders(res);

    // Handle preflight OPTIONS request
    if (req.method == .OPTIONS) {
        res.status = 204;
        return;
    }

    // Get the manager
    const manager = manager_mod.getGlobalManager();
    if (manager == null) {
        try sendError(res, 503, "Model manager not available", "service_unavailable", "req-status");
        return;
    }
    const m = manager.?;

    // Get progress
    const progress_opt = m.getLoadProgress();

    var json = std.ArrayList(u8).empty;
    errdefer json.deinit(std.heap.page_allocator);
    const writer = json.writer(std.heap.page_allocator);

    if (progress_opt) |progress| {
        // Build status response
        const status_str = switch (progress.status) {
            .loading => "loading",
            .completed => "completed",
            .failed => "failed",
            .cancelled => "cancelled",
        };

        const stage_str = switch (progress.stage) {
            .downloading => "downloading",
            .loading_weights => "loading_weights",
            .initializing => "initializing",
            .complete => "complete",
        };

        try writer.writeAll("{");
        try writer.writeAll("\"status\":\"active\",");
        try writer.print("\"model\":\"{s}\",", .{progress.model_id});
        try writer.print("\"load_status\":\"{s}\",", .{status_str});
        try writer.print("\"stage\":\"{s}\",", .{stage_str});
        try writer.print("\"percent_complete\":{d},", .{progress.percent_complete});
        try writer.print("\"bytes_loaded\":{d},", .{progress.bytes_loaded});
        try writer.print("\"bytes_total\":{d},", .{progress.bytes_total});
        try writer.print("\"started_at\":{d},", .{progress.started_at});
        try writer.print("\"updated_at\":{d}", .{progress.updated_at});
        if (progress.error_message) |msg| {
            try writer.print(",\"error\":\"{s}\"", .{msg});
        }
        try writer.writeAll("}");
    } else {
        // No active load
        try writer.writeAll("{");
        try writer.writeAll("\"status\":\"no_active_load\",");
        try writer.writeAll("\"message\":\"No background model load is currently active\"");
        try writer.writeAll("}");
    }

    res.status = 200;
    res.content_type = .JSON;
    try res.writer().writeAll(json.items);
}

/// Handle POST /v1/models/load/cancel - Cancel ongoing background load
pub fn handleCancelLoad(req: anytype, res: anytype) !void {
    // Set CORS headers
    setCorsHeaders(res);

    // Handle preflight OPTIONS request
    if (req.method == .OPTIONS) {
        res.status = 204;
        return;
    }

    // Get the manager
    const manager = manager_mod.getGlobalManager();
    if (manager == null) {
        try sendError(res, 503, "Model manager not available", "service_unavailable", "req-cancel");
        return;
    }
    const m = manager.?;

    // Try to cancel
    m.cancelLoad() catch |err| {
        std.log.err("Failed to cancel background load: {s}", .{@errorName(err)});
        const error_msg = if (err == error.NoActiveLoad) "No active background load to cancel" else "Failed to cancel background load";
        try sendError(res, 400, error_msg, "cancel_failed", "req-cancel");
        return;
    };

    // Return success
    var json = std.ArrayList(u8).empty;
    errdefer json.deinit(std.heap.page_allocator);
    const writer = json.writer(std.heap.page_allocator);

    try writer.writeAll("{");
    try writer.writeAll("\"status\":\"cancelled\",");
    try writer.writeAll("\"message\":\"Background model load has been cancelled\"");
    try writer.writeAll("}");

    res.status = 200;
    res.content_type = .JSON;
    try res.writer().writeAll(json.items);
}

/// Handle POST /v1/models/switch (explicit model switching endpoint)
pub fn handleSwitchModel(req: anytype, res: anytype) !void {
    // Set CORS headers
    setCorsHeaders(res);

    // Handle preflight OPTIONS request
    if (req.method == .OPTIONS) {
        res.status = 204;
        return;
    }

    // Get request body
    const body = req.body() orelse {
        try sendBadRequestError(res, "Missing request body", null, "req-switch");
        return;
    };

    // Parse JSON request
    const parsed = std.json.parseFromSlice(
        types.SwitchModelRequest,
        std.heap.page_allocator,
        body,
        .{},
    ) catch |err| {
        std.log.err("Failed to parse switch model request: {s}", .{@errorName(err)});
        try sendBadRequestError(res, "Invalid JSON in request body", null, "req-switch");
        return;
    };
    defer parsed.deinit();

    const request = parsed.value;

    // Get the manager
    const manager = manager_mod.getGlobalManager();
    if (manager == null) {
        try sendError(res, 503, "Model manager not available", "service_unavailable", "req-switch");
        return;
    }
    const m = manager.?;

    // Get current model for the response
    const previous_model = m.getCurrentModel();

    // Check if model exists
    if (models_mod.getGlobalRegistry()) |reg| {
        if (reg.getModel(request.model) == null) {
            try sendError(res, 404, "Model not found in registry", "model_not_found", "req-switch");
            return;
        }
    }

    // Check memory availability
    if (!m.canLoadModel(request.model)) {
        try sendError(res, 503, "Insufficient memory to load model", "insufficient_memory", "req-switch");
        return;
    }

    // Perform the switch
    const switch_start = std.time.milliTimestamp();
    m.switchModel(request.model) catch |err| {
        std.log.err("Failed to switch to model '{s}': {s}", .{ request.model, @errorName(err) });
        const error_msg = switch (err) {
            error.ModelNotFound => "Model not found",
            error.InsufficientMemory => "Insufficient memory",
            error.ModelLoadFailed => "Failed to load model",
            error.GenerationInProgress => "Generation in progress, please try again later",
            else => "Model switch failed",
        };
        try sendError(res, 500, error_msg, "model_switch_failed", "req-switch");
        return;
    };
    const switch_duration = @as(u64, @intCast(std.time.milliTimestamp() - switch_start));

    // Build success response
    var json = std.ArrayList(u8).empty;
    errdefer json.deinit(std.heap.page_allocator);
    const writer = json.writer(std.heap.page_allocator);

    try writer.writeAll("{");
    try writer.writeAll("\"status\":\"success\",");
    try writer.print("\"model\":\"{s}\",", .{request.model});
    if (previous_model) |prev| {
        try writer.print("\"previous_model\":\"{s}\",", .{prev});
    } else {
        try writer.writeAll("\"previous_model\":null,");
    }
    try writer.print("\"duration_ms\":{d}", .{switch_duration});
    try writer.writeAll("}");

    try sendJsonResponse(res, 200, json.items);
}

/// Handle GET /v1/health (enhanced health check endpoint)
pub fn handleHealth(req: anytype, res: anytype) !void {
    _ = req;

    setCorsHeaders(res);
    res.content_type = .JSON;

    // Get current state
    const ctx = global_context;
    const current_model = if (ctx) |c| std.fs.path.basename(c.model_path) else null;

    // Determine health status
    var status: []const u8 = "unhealthy";
    var http_status: u16 = 503;
    var checks_passed: u32 = 0;
    const total_checks: u32 = 3;

    // Check 1: Model loaded
    const model_loaded = current_model != null;
    if (model_loaded) checks_passed += 1;

    // Check 2: Memory OK (using memory tracker if available)
    var memory_ok = true;
    if (@import("../models/memory.zig").getGlobalTracker()) |tracker| {
        const budget_status = tracker.checkBudget(0.95); // 95% threshold
        memory_ok = budget_status != .critical;
    }
    if (memory_ok) checks_passed += 1;

    // Check 3: Registry available
    const registry_available = models_mod.getGlobalRegistry() != null;
    if (registry_available) checks_passed += 1;

    // Determine overall status
    if (checks_passed == total_checks) {
        status = "healthy";
        http_status = 200;
    } else if (checks_passed >= 2) {
        status = "degraded";
        http_status = 200;
    }

    res.status = http_status;

    // Get memory info
    const memory = @import("../models/memory.zig");
    const gpu_info = memory.getGpuMemoryInfo();

    // Get memory breakdown if available
    var mem_breakdown: ?memory.ComponentBreakdown = null;
    if (memory.getGlobalTracker()) |tracker| {
        mem_breakdown = tracker.getBreakdown();
    }

    // Build comprehensive response
    var json = std.ArrayList(u8).empty;
    errdefer json.deinit(std.heap.page_allocator);
    const writer = json.writer(std.heap.page_allocator);

    try writer.writeAll("{");
    try writer.print("\"status\":\"{s}\",", .{status});
    try writer.writeAll("\"version\":\"0.3.0\",");
    try writer.print("\"model\":{s},", .{if (current_model) |m| try std.fmt.allocPrint(std.heap.page_allocator, "\"{s}\"", .{m}) else "null"});
    try writer.print("\"checks\":{d},", .{checks_passed});
    try writer.print("\"checks_total\":{d},", .{total_checks});

    // GPU info
    try writer.writeAll("\"gpu\":{");
    try writer.print("\"available\":{s},", .{if (gpu_info.metal_enabled) "true" else "false"});
    try writer.print("\"metal_enabled\":{s},", .{if (gpu_info.metal_enabled) "true" else "false"});
    try writer.print("\"memory_total_mb\":{d},", .{gpu_info.total_mb});
    try writer.print("\"memory_free_mb\":{d}", .{gpu_info.free_mb});
    try writer.writeAll("},");

    // Memory breakdown
    try writer.writeAll("\"memory\":{");
    if (mem_breakdown) |mb| {
        try writer.print("\"used_mb\":{d},", .{mb.total_mb});
        try writer.print("\"peak_mb\":{d},", .{mb.peak_mb});
        try writer.print("\"available_mb\":{d},", .{gpu_info.free_mb});
        try writer.writeAll("\"components\":{");
        try writer.print("\"weights_mb\":{d},", .{mb.weights_mb});
        try writer.print("\"kv_cache_mb\":{d},", .{mb.kv_cache_mb});
        try writer.print("\"temporaries_mb\":{d},", .{mb.temporaries_mb});
        try writer.print("\"overhead_mb\":{d}", .{mb.overhead_mb});
        try writer.writeAll("}");
    } else {
        try writer.writeAll("\"used_mb\":0,\"peak_mb\":0,\"available_mb\":0,\"components\":null");
    }
    try writer.writeAll("}");

    // Cache metrics (if prompt caching is enabled)
    try writer.writeAll(",\"cache\":{");
    if (prompt_cache.getGlobalCache()) |cache| {
        const metrics = cache.getMetrics();
        try writer.writeAll("\"enabled\":true,");
        try writer.print("\"size_bytes\":{d},", .{metrics.total_size_bytes});
        try writer.print("\"max_size_bytes\":{d},", .{cache.max_size_bytes});
        try writer.print("\"entries\":{d},", .{metrics.entries});
        try writer.print("\"hit_rate\":{d:.2},", .{@as(u32, @intFromFloat(metrics.hitRate() * 100))});
        try writer.print("\"hits\":{d},", .{metrics.hits});
        try writer.print("\"misses\":{d},", .{metrics.misses});
        try writer.print("\"evictions\":{d}", .{metrics.evictions});
    } else {
        try writer.writeAll("\"enabled\":false");
    }
    try writer.writeAll("}");

    try writer.writeAll("}");

    try sendJsonResponse(res, http_status, json.items);
}

/// Handle GET /v1/metrics (Simple JSON metrics endpoint)
pub fn handleMetrics(req: anytype, res: anytype) !void {
    _ = req;

    setCorsHeaders(res);
    res.content_type = .JSON;

    const writer_res = res.writer();

    // Start JSON object
    try writer_res.writeAll("{\n");

    // Cache metrics
    try writer_res.writeAll("  \"cache\": {\n");
    if (prompt_cache.getGlobalCache()) |cache| {
        const metrics = cache.getMetrics();
        try writer_res.print("    \"entries\": {d},\n", .{metrics.entries});
        try writer_res.print("    \"size_bytes\": {d},\n", .{metrics.total_size_bytes});
        try writer_res.print("    \"hits\": {d},\n", .{metrics.hits});
        try writer_res.print("    \"misses\": {d},\n", .{metrics.misses});
        try writer_res.print("    \"hit_rate\": {d:.4}\n", .{metrics.hitRate()});
    } else {
        try writer_res.writeAll("    \"entries\": 0,\n");
        try writer_res.writeAll("    \"size_bytes\": 0,\n");
        try writer_res.writeAll("    \"hits\": 0,\n");
        try writer_res.writeAll("    \"misses\": 0,\n");
        try writer_res.writeAll("    \"hit_rate\": 0\n");
    }
    try writer_res.writeAll("  },\n");

    // TurboQuant status
    try writer_res.writeAll("  \"turboquant\": {\n");
    try writer_res.writeAll("    \"enabled\": true,\n");
    try writer_res.writeAll("    \"bits\": 4,\n");
    try writer_res.writeAll("    \"adaptive_layers\": 4,\n");
    try writer_res.writeAll("    \"compression_ratio\": 4.0\n");
    try writer_res.writeAll("  },\n");

    // Request stats (simplified - just placeholders for now)
    try writer_res.writeAll("  \"requests\": {\n");
    try writer_res.writeAll("    \"status\": \"active\"\n");
    try writer_res.writeAll("  }\n");

    // End JSON object
    try writer_res.writeAll("}\n");

    res.status = 200;
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
    try writer.writeAll("\"}");

    // Add logprobs if available
    if (result.logprobs) |entries| {
        try writer.writeAll(",\"logprobs\":{");
        try writer.writeAll("\"content\":[");

        for (entries, 0..) |entry, i| {
            if (i > 0) try writer.writeAll(",");
            try writer.writeAll("{");
            // Token string (empty for now - deferred)
            try writer.writeAll("\"token\":\"\",");
            // Log probability
            try writer.print("\"logprob\":{:.6},", .{entry.logprob});
            // Bytes (null per OpenAI spec)
            try writer.writeAll("\"bytes\":null,");
            // Top logprobs array
            try writer.writeAll("\"top_logprobs\":[");
            for (entry.top_logprobs, 0..) |top, j| {
                if (j > 0) try writer.writeAll(",");
                try writer.writeAll("{");
                try writer.writeAll("\"token\":\"\","); // Token string deferred
                try writer.print("\"logprob\":{:.6}", .{top.logprob});
                try writer.writeAll("}");
            }
            try writer.writeAll("]}"); // Close top_logprobs and entry
        }

        try writer.writeAll("]}"); // Close content array and logprobs object
    }

    try writer.writeAll(",\"finish_reason\":\"");
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
