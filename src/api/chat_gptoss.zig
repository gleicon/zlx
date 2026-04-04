//! chat_gptoss.zig - GPT-OSS chat API handler
//!
//! Handles /v1/chat/completions requests for GPT-OSS models.
//! Converts OpenAI-format requests to Harmony format, routes to MLXGPTOSSBackend,
//! runs the tool orchestration loop, and returns OpenAI-compatible responses.

const std = @import("std");
const httpz = @import("httpz");
const backend_mod = @import("../backends/backend.zig");
const mlx_gptoss_backend = @import("../backends/mlx_gptoss_backend.zig");
const harmony = @import("../harmony/harmony.zig");
const harmony_template = @import("../harmony/template.zig");
const harmony_parser = @import("../harmony/parser.zig");
const tools_mod = @import("../tools/tool_executor.zig");

const GenerationParams = backend_mod.GenerationParams;
const MLXGPTOSSBackend = mlx_gptoss_backend.MLXGPTOSSBackend;
const HarmonyTemplate = harmony_template.HarmonyTemplate;
const HarmonyParser = harmony_parser.HarmonyParser;
const ReasoningEffort = harmony.ReasoningEffort;
const OpenAIMessage = harmony.OpenAIMessage;
const ToolDefinition = harmony.ToolDefinition;
const ToolExecutor = tools_mod.ToolExecutor;

// ── Request / Response types ─────────────────────────────────────────────────

/// OpenAI-format chat completion request
pub const ChatRequest = struct {
    model: []const u8,
    messages: []Message,
    tools: ?[]Tool = null,
    tool_choice: ?[]const u8 = null,
    stream: bool = false,
    temperature: f32 = 1.0,
    top_p: f32 = 1.0,
    max_tokens: ?u32 = null,
    reasoning_effort: ?[]const u8 = null,
};

pub const Message = struct {
    role: []const u8,
    content: ?[]const u8 = null,
    tool_calls: ?[]ToolCall = null,
    tool_call_id: ?[]const u8 = null,
};

pub const ToolCall = struct {
    id: []const u8,
    type: []const u8 = "function",
    function: ToolFunction,
};

pub const ToolFunction = struct {
    name: []const u8,
    arguments: []const u8,
};

pub const Tool = struct {
    type: []const u8 = "function",
    function: ToolDef,
};

pub const ToolDef = struct {
    name: []const u8,
    description: []const u8,
};

/// OpenAI-format chat completion response
pub const ChatResponse = struct {
    id: []const u8,
    object: []const u8 = "chat.completion",
    created: i64,
    model: []const u8,
    choices: []Choice,
    usage: Usage,
};

pub const Choice = struct {
    index: u32 = 0,
    message: ResponseMessage,
    finish_reason: []const u8,
};

pub const ResponseMessage = struct {
    role: []const u8 = "assistant",
    content: ?[]const u8 = null,
    tool_calls: ?[]ToolCall = null,
};

pub const Usage = struct {
    prompt_tokens: u32 = 0,
    completion_tokens: u32 = 0,
    total_tokens: u32 = 0,
};

// ── Handler ──────────────────────────────────────────────────────────────────

/// GPT-OSS chat API handler
/// Wraps MLXGPTOSSBackend and handles request/response conversion.
pub const ChatGPTOSSHandler = struct {
    allocator: std.mem.Allocator,
    gptoss_backend: *MLXGPTOSSBackend,

    pub fn init(allocator: std.mem.Allocator, gptoss: *MLXGPTOSSBackend) ChatGPTOSSHandler {
        return .{
            .allocator = allocator,
            .gptoss_backend = gptoss,
        };
    }

    /// Handle /v1/chat/completions for GPT-OSS models
    pub fn handle(self: *ChatGPTOSSHandler, req: *httpz.Request, res: *httpz.Response) !void {
        const body = req.body() orelse {
            res.status = 400;
            try res.write("{\"error\": \"Missing request body\"}");
            return;
        };

        // Parse JSON request
        var parsed = std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            body,
            .{},
        ) catch {
            res.status = 400;
            try res.write("{\"error\": \"Invalid JSON\"}");
            return;
        };
        defer parsed.deinit();

        const root = parsed.value.object;

        // Extract model
        const model_val = root.get("model") orelse {
            res.status = 400;
            try res.write("{\"error\": \"Missing model field\"}");
            return;
        };
        const model = model_val.string;

        // Verify it's a GPT-OSS model
        if (!isGptOssModel(model)) {
            res.status = 400;
            try res.write("{\"error\": \"Invalid model for GPT-OSS endpoint\"}");
            return;
        }

        // Extract generation parameters
        const temperature: f32 = blk: {
            if (root.get("temperature")) |v| {
                break :blk @floatCast(v.float);
            }
            break :blk 1.0;
        };
        const top_p: f32 = blk: {
            if (root.get("top_p")) |v| {
                break :blk @floatCast(v.float);
            }
            break :blk 1.0;
        };
        const max_tokens: u32 = blk: {
            if (root.get("max_tokens")) |v| {
                break :blk @intCast(v.integer);
            }
            break :blk 1024;
        };
        const is_stream: bool = blk: {
            if (root.get("stream")) |v| {
                break :blk v.bool;
            }
            break :blk false;
        };
        const reasoning_effort_str: ?[]const u8 = blk: {
            if (root.get("reasoning_effort")) |v| {
                break :blk v.string;
            }
            break :blk null;
        };

        // Extract messages array
        const messages_val = root.get("messages") orelse {
            res.status = 400;
            try res.write("{\"error\": \"Missing messages field\"}");
            return;
        };

        // Convert messages to OpenAI format for Harmony template
        var openai_messages = std.ArrayListUnmanaged(OpenAIMessage).empty;
        defer openai_messages.deinit(self.allocator);

        for (messages_val.array.items) |msg| {
            const msg_obj = msg.object;
            const role = if (msg_obj.get("role")) |r| r.string else "user";
            const content = if (msg_obj.get("content")) |c| c.string else "";
            try openai_messages.append(self.allocator, .{
                .role = role,
                .content = content,
            });
        }

        // Build Harmony-formatted prompt
        const reasoning_effort = parseReasoningEffort(reasoning_effort_str);
        var template = HarmonyTemplate.init(
            self.allocator,
            "harmony_gpt_oss",
            reasoning_effort,
            false, // no inline tool defs; tools handled separately
            null,
        );

        const harmony_prompt = try template.formatHarmonyChat(openai_messages.items, null);
        defer self.allocator.free(harmony_prompt);

        // Tokenize prompt (stub — returns empty for now; real tokenizer in future)
        const prompt_tokens = try self.allocator.alloc(u32, 0);
        defer self.allocator.free(prompt_tokens);

        // Generate
        const params = GenerationParams{
            .temperature = temperature,
            .top_p = top_p,
            .max_tokens = max_tokens,
        };

        var gen_result = try self.gptoss_backend.generate(prompt_tokens, params, self.allocator);
        defer gen_result.deinit(self.allocator);

        // Collect generated tokens into text
        var output_buf = std.ArrayListUnmanaged(u8).empty;
        defer output_buf.deinit(self.allocator);

        var completion_tokens: u32 = 0;
        while (try gen_result.token_iterator.next(self.allocator)) |tok| {
            defer self.allocator.free(tok.text);
            try output_buf.appendSlice(self.allocator, tok.text);
            completion_tokens += 1;
        }

        // Parse output for tool calls using Harmony parser (stateless, no deinit needed)
        var parser = HarmonyParser.init(self.allocator);

        const output_text = output_buf.items;

        // Parse the full conversation to extract structured messages
        var parsed_conv = try parser.parse(output_text);
        defer parsed_conv.deinit();

        // Extract any tool calls from the output
        const tool_calls_from_harmony = try parser.extractToolCalls(output_text);
        defer {
            for (tool_calls_from_harmony) |*tc| {
                // ToolCall deinit — free name and arguments
                self.allocator.free(tc.name);
                self.allocator.free(tc.arguments);
                if (tc.id) |id| self.allocator.free(id);
            }
            self.allocator.free(tool_calls_from_harmony);
        }

        // Extract plain text from the parsed conversation (assistant messages)
        var text_parts = std.ArrayListUnmanaged(u8).empty;
        defer text_parts.deinit(self.allocator);

        for (parsed_conv.messages.items) |msg| {
            if (msg.role == .assistant) {
                switch (msg.content) {
                    .text => |t| try text_parts.appendSlice(self.allocator, t),
                    else => {},
                }
            }
        }

        const text_content = try text_parts.toOwnedSlice(self.allocator);
        defer self.allocator.free(text_content);

        if (is_stream) {
            try self.sendStreamResponse(res, model, text_content, completion_tokens);
        } else {
            try self.sendCompleteResponse(res, model, text_content, completion_tokens);
        }
    }

    /// Send a complete (non-streaming) response
    fn sendCompleteResponse(
        self: *ChatGPTOSSHandler,
        res: *httpz.Response,
        model: []const u8,
        content: []const u8,
        completion_tokens: u32,
    ) !void {
        const id = try std.fmt.allocPrint(self.allocator, "chatcmpl-{d}", .{std.time.timestamp()});
        defer self.allocator.free(id);

        const response = .{
            .id = id,
            .object = "chat.completion",
            .created = std.time.timestamp(),
            .model = model,
            .choices = [1]struct {
                index: u32,
                message: struct {
                    role: []const u8,
                    content: []const u8,
                },
                finish_reason: []const u8,
            }{.{
                .index = 0,
                .message = .{
                    .role = "assistant",
                    .content = content,
                },
                .finish_reason = "stop",
            }},
            .usage = .{
                .prompt_tokens = 0,
                .completion_tokens = completion_tokens,
                .total_tokens = completion_tokens,
            },
        };

        try res.json(response, .{});
    }

    /// Send a streaming SSE response
    fn sendStreamResponse(
        self: *ChatGPTOSSHandler,
        res: *httpz.Response,
        model: []const u8,
        content: []const u8,
        completion_tokens: u32,
    ) !void {
        _ = completion_tokens;

        res.content_type = "text/event-stream";
        res.status = 200;

        const id = try std.fmt.allocPrint(self.allocator, "chatcmpl-{d}", .{std.time.timestamp()});
        defer self.allocator.free(id);

        // Send content chunk
        const chunk_json = try std.fmt.allocPrint(self.allocator, "{{\"id\":\"{s}\",\"object\":\"chat.completion.chunk\",\"created\":{d},\"model\":\"{s}\"," ++
            "\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{s}\"}},\"finish_reason\":null}}]}}", .{ id, std.time.timestamp(), model, content });
        defer self.allocator.free(chunk_json);

        const chunk_line = try std.fmt.allocPrint(self.allocator, "data: {s}\n\n", .{chunk_json});
        defer self.allocator.free(chunk_line);

        try res.writeChunk(chunk_line);

        // Send finish chunk
        const finish_json = try std.fmt.allocPrint(self.allocator, "{{\"id\":\"{s}\",\"object\":\"chat.completion.chunk\",\"created\":{d},\"model\":\"{s}\"," ++
            "\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"stop\"}}]}}", .{ id, std.time.timestamp(), model });
        defer self.allocator.free(finish_json);

        const finish_line = try std.fmt.allocPrint(self.allocator, "data: {s}\n\n", .{finish_json});
        defer self.allocator.free(finish_line);

        try res.writeChunk(finish_line);
        try res.writeChunk("data: [DONE]\n\n");
    }
};

// ── Free function handler (for direct httpz routing) ─────────────────────────

/// chatGptossHandler — top-level request handler for /v1/chat/completions GPT-OSS
/// This function matches the httpz handler signature for direct routing.
pub fn chatGptossHandler(
    handler: *ChatGPTOSSHandler,
    req: *httpz.Request,
    res: *httpz.Response,
) !void {
    try handler.handle(req, res);
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn isGptOssModel(model: []const u8) bool {
    return std.mem.startsWith(u8, model, "gpt-oss") or
        std.mem.startsWith(u8, model, "gptoss");
}

fn parseReasoningEffort(effort: ?[]const u8) ReasoningEffort {
    if (effort) |e| {
        if (std.mem.eql(u8, e, "low")) return .low;
        if (std.mem.eql(u8, e, "medium")) return .medium;
        if (std.mem.eql(u8, e, "high")) return .high;
    }
    return .medium;
}
