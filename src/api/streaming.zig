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

/// SSE headers for streaming response
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

    // Generate completion ID
    const completion_id = try types.generateCompletionId(allocator);
    defer allocator.free(completion_id);

    // Build prompt from messages
    const prompt = try types.buildPromptFromMessages(allocator, request.messages);
    defer allocator.free(prompt);

    // Tokenize the prompt
    const tokenizer_ref = &ctx.tokenizer.?;
    const input_tokens = try tokenizer_ref.encode(prompt);
    defer allocator.free(input_tokens);

    // Create generation options
    const gen_options = generator.GenerationOptions{
        .max_tokens = request.getMaxTokens(),
        .temperature = request.getTemperature(),
        .top_p = request.getTopP(),
        .stop_on_eos = true,
    };

    // Initialize transformer (we need it for generation)
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
    );
    defer state.deinit();

    // Track tokens and content
    var token_buffer = std.ArrayList(u32).empty;
    defer token_buffer.deinit(allocator);

    var total_tokens: u32 = 0;
    const created_timestamp = std.time.timestamp();

    // Stream tokens one at a time
    var is_first_chunk = true;
    while (try state.next()) |token| {
        total_tokens += 1;
        try token_buffer.append(allocator, token);

        // Decode the token to text
        const token_slice = &[_]u32{token};
        const text = try tokenizer_ref.decode(token_slice);
        defer allocator.free(text);

        // Build the SSE chunk
        const chunk = try buildStreamingChunk(
            allocator,
            completion_id,
            created_timestamp,
            request.model,
            text,
            is_first_chunk,
            null, // Not finished yet
        );
        defer allocator.free(chunk);

        // Write SSE format: data: {...}\n\n
        try writer.writeAll("data: ");
        try writer.writeAll(chunk);
        try writer.writeAll("\n\n");

        // Flush to ensure client receives immediately
        // Note: httpz writer may need explicit flush - we'll handle this in the caller

        is_first_chunk = false;
    }

    // Write final chunk with finish_reason
    const final_chunk = try buildStreamingChunk(
        allocator,
        completion_id,
        created_timestamp,
        request.model,
        "", // No content in final chunk
        false,
        "stop", // Generation stopped
    );
    defer allocator.free(final_chunk);

    try writer.writeAll("data: ");
    try writer.writeAll(final_chunk);
    try writer.writeAll("\n\n");

    // Write [DONE] marker as per OpenAI protocol
    try writer.writeAll("data: [DONE]\n\n");
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

    try writer.writeByte('}');

    if (finish_reason) |reason| {
        try writer.print(",\"finish_reason\":\"{s}\"", .{reason});
    } else {
        try writer.writeAll(",\"finish_reason\":null");
    }

    try writer.writeAll("]}]}");

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
    // Generate completion ID
    const completion_id = try types.generateCompletionId(allocator);
    defer allocator.free(completion_id);

    // Build prompt from messages
    const prompt = try types.buildPromptFromMessages(allocator, request.messages);
    defer allocator.free(prompt);

    // Tokenize the prompt
    const tokenizer_ref = &ctx.tokenizer.?;
    const input_tokens = try tokenizer_ref.encode(prompt);
    defer allocator.free(input_tokens);

    // Create generation options
    const gen_options = generator.GenerationOptions{
        .max_tokens = request.getMaxTokens(),
        .temperature = request.getTemperature(),
        .top_p = request.getTopP(),
        .stop_on_eos = true,
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
    );
    defer state.deinit();

    // Collect all tokens
    var output_tokens = std.ArrayList(u32).empty;
    defer output_tokens.deinit(allocator);

    const prompt_tokens: u32 = @intCast(input_tokens.len);
    var completion_tokens: u32 = 0;

    while (try state.next()) |token| {
        try output_tokens.append(allocator, token);
        completion_tokens += 1;
    }

    // Decode all tokens at once for efficiency
    const content = try tokenizer_ref.decode(output_tokens.items);
    defer allocator.free(content);

    // Build response
    const created_timestamp = std.time.timestamp();

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
