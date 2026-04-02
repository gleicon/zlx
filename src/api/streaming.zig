//! streaming.zig - Server-Sent Events (SSE) for streaming chat completions
//!
//! Implements OpenAI-compatible SSE streaming for chat completions.
//! Format: data: {...}\n\n with flush after each chunk.

const std = @import("std");
const types = @import("types.zig");
const inference = @import("../inference/mod.zig");
const qwen = @import("../mlx.zig/src/qwen.zig");
const mlx = @import("../mlx.zig/src/mlx.zig");
const generator = @import("../inference/generator.zig");
const metrics = @import("metrics.zig");

/// Check if a string is valid UTF-8
fn isValidUtf8(str: []const u8) bool {
    return std.unicode.utf8ValidateSlice(str);
}

/// Check if a string ends with a complete UTF-8 sequence
fn isCompleteUtf8Ending(str: []const u8) bool {
    if (str.len == 0) return true;

    // Walk backwards to count continuation bytes
    var i = str.len;
    var continuation_bytes: usize = 0;

    while (i > 0) {
        i -= 1;
        const b = str[i];
        if ((b & 0xC0) == 0x80) {
            continuation_bytes += 1;
        } else {
            break;
        }
    }

    if (continuation_bytes == 0) return true;

    const start_byte = str[str.len - continuation_bytes - 1];
    const masked = start_byte & 0xF0;
    var expected: usize = 0;
    if (masked == 0xF0) {
        expected = 3; // 4 byte sequence
    } else if (masked == 0xE0) {
        expected = 2; // 3 byte sequence
    } else if (masked == 0xC0) {
        expected = 1; // 2 byte sequence
    }

    return continuation_bytes == expected;
}

pub const SSE_HEADERS = .{
    .{ "Content-Type", "text/event-stream" },
    .{ "Cache-Control", "no-cache" },
    .{ "Connection", "keep-alive" },
    .{ "Access-Control-Allow-Origin", "*" },
    .{ "Access-Control-Allow-Methods", "GET, POST, OPTIONS" },
    .{ "Access-Control-Allow-Headers", "Content-Type, Authorization" },
};

/// Stream a chat completion response using SSE
///
/// Takes the HTTP response writer, request, and inference context.
/// Yields tokens one at a time as SSE chunks.
pub fn streamResponse(
    writer: anytype,
    request: types.ChatCompletionRequest,
    ctx: *inference.InferenceContext,
) !void {
    const allocator = ctx.allocator;

    // Start timing
    const start_time = std.time.milliTimestamp();
    metrics.startGeneration();

    // Generate completion ID
    const completion_id = try types.generateCompletionId(allocator);
    defer allocator.free(completion_id);

    // Build prompt from messages
    const prompt = try types.buildPromptFromMessages(allocator, request.messages);
    defer allocator.free(prompt);

    // Tokenize the prompt
    const tokenizer_ref = &ctx.tokenizer.?;
    var input_tokens = try tokenizer_ref.encode(prompt);
    errdefer allocator.free(input_tokens);
    const prompt_token_count: u32 = @intCast(input_tokens.len);

    // Create generation options with context limit
    const MAX_CONTEXT_LENGTH: usize = 8192;
    const requested_max = request.getMaxTokens();
    const max_new_tokens = @min(requested_max, MAX_CONTEXT_LENGTH - 1);
    const max_input_tokens = MAX_CONTEXT_LENGTH - max_new_tokens;

    // Truncate input if too long (keep from the end)
    if (input_tokens.len > max_input_tokens) {
        const start_idx = input_tokens.len - max_input_tokens;
        const truncated = try allocator.dupe(u32, input_tokens[start_idx..]);
        allocator.free(input_tokens);
        input_tokens = truncated;
        std.log.warn("Streaming input truncated from {d} to {d} tokens", .{ input_tokens.len + max_input_tokens, max_input_tokens });
    }

    // Create generation options with all sampling parameters
    const gen_options = generator.GenerationOptions{
        .max_tokens = max_new_tokens,
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

    // Initialize transformer (we need it for generation)
    var transformer = try qwen.Transformer.init(allocator, ctx.model_path);
    defer transformer.deinit();

    // Use generation state with full sampling support
    var state = try generator.GenerationState.init(
        allocator,
        &transformer,
        input_tokens,
        transformer.eos_token_ids,
        gen_options,
        &ctx.tokenizer.?, // Pass tokenizer for stop sequence detection
        null, // draft_model
        0, // speculation_depth
    );
    defer state.deinit();

    // Collect all tokens using the iterator
    var generated_tokens = std.ArrayList(u32).empty;
    defer generated_tokens.deinit(allocator);

    while (try state.next()) |token| {
        try generated_tokens.append(allocator, token);
    }

    // Decode generated tokens
    const generated_text = try tokenizer_ref.decode(generated_tokens.items);
    defer allocator.free(generated_text);

    const created_timestamp = std.time.timestamp();

    // Get logprobs if enabled
    const logprobs_entries = if (gen_options.logprobs_enabled) state.getLogprobs() else null;

    // Stream the generated text in chunks
    // Strip special tokens from generated text
    const cleaned_text = try stripSpecialTokens(allocator, generated_text);
    defer allocator.free(cleaned_text);

    // Stream the entire cleaned text at once (simpler and more reliable)
    if (cleaned_text.len > 0) {
        const chunk = try buildStreamingChunk(
            allocator,
            completion_id,
            created_timestamp,
            request.model,
            cleaned_text,
            true,
            null,
            null, // No logprobs in content chunk
        );
        defer allocator.free(chunk);

        try writer.writeAll("data: ");
        try writer.writeAll(chunk);
        try writer.writeAll("\n\n");
    }

    // Write final chunk with finish_reason and logprobs if enabled
    const final_chunk = try buildStreamingChunk(
        allocator,
        completion_id,
        created_timestamp,
        request.model,
        "", // No content in final chunk
        false,
        "stop", // Generation stopped
        logprobs_entries, // Include logprobs in final chunk if enabled
    );
    defer allocator.free(final_chunk);

    try writer.writeAll("data: ");
    try writer.writeAll(final_chunk);
    try writer.writeAll("\n\n");

    // Write [DONE] marker as per OpenAI protocol
    try writer.writeAll("data: [DONE]\n\n");

    // Record metrics
    const end_time = std.time.milliTimestamp();
    const generation_time_ms = @as(u64, @intCast(end_time - start_time));
    const completion_token_count: u32 = @intCast(generated_tokens.items.len);
    // For streaming, we don't have true TTFT since we wait for all tokens, so use 0
    metrics.recordRequest(prompt_token_count, completion_token_count, generation_time_ms, 0);
}

/// Build a streaming chunk JSON string
fn buildStreamingChunk(
    allocator: std.mem.Allocator,
    id: []const u8,
    created: i64,
    model: []const u8,
    content: []const u8,
    is_first: bool,
    finish_reason: ?[]const u8,
    logprobs: ?[]const generator.LogprobEntry,
) ![]const u8 {
    var json = std.ArrayList(u8).empty;
    errdefer json.deinit(allocator);

    const writer = json.writer(allocator);

    // Build the chunk structure
    try writer.writeAll("{\"id\":\"");
    try writeJsonString(writer, id);
    try writer.print("\",\"object\":\"chat.completion.chunk\",\"created\":{d},\"model\":\"", .{created});
    try writeJsonString(writer, model);
    try writer.writeAll("\",\"choices\":[");

    // Single choice for now
    try writer.writeAll("{\"index\":0,\"delta\":");

    if (is_first) {
        // First chunk includes role
        try writer.writeAll("{\"role\":\"assistant\"");
        if (content.len > 0) {
            try writer.writeAll(",\"content\":\"");
            try writeJsonString(writer, content);
            try writer.writeByte('"');
        }
        try writer.writeByte('}');
    } else if (content.len > 0) {
        // Subsequent chunks just have content
        try writer.writeAll("{\"content\":\"");
        try writeJsonString(writer, content);
        try writer.writeAll("\"}");
    } else {
        // Final chunk - empty delta
        try writer.writeAll("{}");
    }

    // Add logprobs if present (include in final chunk for MVP)
    if (logprobs) |entries| {
        try writer.writeAll(",\"logprobs\":{");
        try writer.writeAll("\"content\":[");

        for (entries, 0..) |entry, i| {
            if (i > 0) try writer.writeAll(",");
            try writer.writeAll("{");
            // Token string (empty for now - deferred)
            try writer.writeAll("\"token\":\"\",");
            // Log probability
            try writer.print("\"logprob\":{:.6}", .{entry.logprob});
            // Top logprobs array
            try writer.writeAll(",\"top_logprobs\":[");
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

    // Add finish_reason inside the choice object
    if (finish_reason) |reason| {
        try writer.print(",\"finish_reason\":\"{s}\"", .{reason});
    } else {
        try writer.writeAll(",\"finish_reason\":null");
    }

    // Close choice object, choices array, and root object
    try writer.writeAll("}]}");

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

/// Stream error response as SSE
pub fn streamError(
    writer: anytype,
    error_message: []const u8,
    error_type: []const u8,
) !void {
    var json_buf: [1024]u8 = undefined;
    const json = try std.fmt.bufPrint(&json_buf, "{{\"error\":{{\"message\":\"{s}\",\"type\":\"{s}\"}}}}", .{ error_message, error_type });

    try writer.writeAll("data: ");
    try writer.writeAll(json);
    try writer.writeAll("\n\n");
    try writer.writeAll("data: [DONE]\n\n");
}

/// Non-streaming response generation
///
/// Collects all tokens and returns complete JSON response
pub fn generateNonStreamingResponse(
    allocator: std.mem.Allocator,
    request: types.ChatCompletionRequest,
    ctx: *inference.InferenceContext,
) ![]const u8 {
    // Start timing
    const start_time = std.time.milliTimestamp();
    const generation_start_micro = std.time.microTimestamp();
    metrics.startGeneration();

    // Generate completion ID
    const completion_id = try types.generateCompletionId(allocator);
    defer allocator.free(completion_id);

    // Build prompt from messages
    const prompt = try types.buildPromptFromMessages(allocator, request.messages);
    defer allocator.free(prompt);

    // Tokenize the prompt
    const tokenizer_ref = &ctx.tokenizer.?;
    var input_tokens = try tokenizer_ref.encode(prompt);
    errdefer allocator.free(input_tokens);
    const prompt_tokens: u32 = @intCast(input_tokens.len);

    // Create generation options with context limit
    const MAX_CONTEXT_LENGTH: usize = 8192;
    const requested_max = request.getMaxTokens();
    const max_new_tokens = @min(requested_max, MAX_CONTEXT_LENGTH - 1);
    const max_input_tokens = MAX_CONTEXT_LENGTH - max_new_tokens;

    // Truncate input if too long (keep from the end)
    if (input_tokens.len > max_input_tokens) {
        const start_idx = input_tokens.len - max_input_tokens;
        const truncated = try allocator.dupe(u32, input_tokens[start_idx..]);
        allocator.free(input_tokens);
        input_tokens = truncated;
        std.log.warn("Non-streaming input truncated from {d} to {d} tokens", .{ input_tokens.len + max_input_tokens, max_input_tokens });
    }

    const gen_options = generator.GenerationOptions{
        .max_tokens = max_new_tokens,
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

    // Initialize transformer
    var transformer = try qwen.Transformer.init(allocator, ctx.model_path);
    defer transformer.deinit();

    // Get EOS token IDs
    const eos_token_ids = transformer.eos_token_ids;

    // Initialize generation state
    var state = try generator.GenerationState.init(
        allocator,
        &transformer,
        input_tokens,
        eos_token_ids,
        gen_options,
        &ctx.tokenizer.?, // Pass tokenizer for stop sequence detection
        null, // draft_model
        0, // speculation_depth
    );
    defer state.deinit();

    // Collect all tokens
    var output_tokens = std.ArrayList(u32).empty;
    defer output_tokens.deinit(allocator);

    var completion_tokens: u32 = 0;
    var first_token = true;
    var first_token_time_micro: i64 = 0;

    while (try state.next()) |token| {
        if (first_token) {
            first_token_time_micro = std.time.microTimestamp();
            first_token = false;
        }
        try output_tokens.append(allocator, token);
        completion_tokens += 1;
    }

    // Decode all tokens at once for efficiency
    const raw_content = try tokenizer_ref.decode(output_tokens.items);
    defer allocator.free(raw_content);

    // Strip special tokens from output
    const content = try stripSpecialTokens(allocator, raw_content);
    defer allocator.free(content);

    // Note: logprobs are retrieved but not used in this function's response format
    // The handlers.zig buildChatCompletionResponse handles logprobs for non-streaming
    _ = if (gen_options.logprobs_enabled) state.getLogprobs() else null;

    // Build response
    const created_timestamp = std.time.timestamp();

    // Record metrics
    const end_time = std.time.milliTimestamp();
    const generation_time_ms = @as(u64, @intCast(end_time - start_time));
    const ttft_us = if (first_token_time_micro > generation_start_micro)
        @as(u64, @intCast(first_token_time_micro - generation_start_micro))
    else
        0;
    metrics.recordRequest(prompt_tokens, completion_tokens, generation_time_ms, ttft_us);

    // Build the response JSON
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
    try writeJsonString(writer, content);
    try writer.writeAll("\"},");
    try writer.writeAll("\"finish_reason\":\"stop\"}");

    try writer.writeAll("],\"usage\":");

    // Usage stats
    try writer.print("{{\"prompt_tokens\":{d},\"completion_tokens\":{d},\"total_tokens\":{d}}}}}", .{
        prompt_tokens,
        completion_tokens,
        prompt_tokens + completion_tokens,
    });

    return json.toOwnedSlice(allocator);
}

/// Strip special tokens from generated text
fn stripSpecialTokens(allocator: std.mem.Allocator, text: []const u8) ![]const u8 {
    // Special tokens to remove
    const special_tokens = &[_][]const u8{
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>", // This is token 151645
        "  ", // Legacy - might also appear
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
        "<|fim_pad|>",
        "<|repo_name|>",
        "<|file_sep|>",
    };

    var result = try allocator.dupe(u8, text);
    errdefer allocator.free(result);

    // Iteratively remove special tokens
    var changed = true;
    while (changed) {
        changed = false;
        for (special_tokens) |token| {
            if (std.mem.indexOf(u8, result, token)) |pos| {
                // Found token, remove it
                const new_len = result.len - token.len;
                var new_result = try allocator.alloc(u8, new_len);

                // Copy before token
                @memcpy(new_result[0..pos], result[0..pos]);
                // Copy after token
                if (pos + token.len < result.len) {
                    @memcpy(new_result[pos..], result[pos + token.len ..]);
                }

                allocator.free(result);
                result = new_result;
                changed = true;
                break; // Restart search after modification
            }
        }
    }

    return result;
}
