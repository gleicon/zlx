//! chat_deepseek.zig - DeepSeek chat API handler
//!
//! Handles /v1/chat/completions requests for DeepSeek models.
//! Uses LlamaBackend (llama.cpp) for GGUF inference — per D-01 handler-per-model pattern.
//! No factory abstraction: server.zig owns the globals and dispatches directly.

const std = @import("std");
const httpz = @import("httpz");
const backend_mod = @import("../backends/backend.zig");
const llama_cpp_mod = @import("../backends/llama_cpp.zig");

const GenerationParams = backend_mod.GenerationParams;
const LlamaBackend = llama_cpp_mod.LlamaBackend;

// ── Request / Response types ─────────────────────────────────────────────────

/// OpenAI-format chat completion request
pub const ChatRequest = struct {
    model: []const u8,
    messages: []Message,
    stream: bool = false,
    temperature: f32 = 1.0,
    top_p: f32 = 1.0,
    max_tokens: ?u32 = null,
};

pub const Message = struct {
    role: []const u8,
    content: ?[]const u8 = null,
};

// ── Handler ──────────────────────────────────────────────────────────────────

/// DeepSeek chat API handler — per D-01 handler-per-model pattern
/// Wraps LlamaBackend (llama.cpp) for DeepSeek GGUF models.
pub const ChatDeepSeekHandler = struct {
    allocator: std.mem.Allocator,
    backend: *?LlamaBackend,
    model_path: []const u8 = "",
    loaded: bool = false,

    /// Initialize with a pointer to the module-level optional LlamaBackend
    pub fn init(allocator: std.mem.Allocator, backend_ptr: *?LlamaBackend) ChatDeepSeekHandler {
        return .{
            .allocator = allocator,
            .backend = backend_ptr,
        };
    }

    /// Handle /v1/chat/completions for DeepSeek models
    pub fn handle(self: *ChatDeepSeekHandler, req: *httpz.Request, res: *httpz.Response) !void {
        const body = req.body() orelse {
            res.status = 400;
            try res.writer().writeAll("{\"error\": \"Missing request body\"}");
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
            try res.writer().writeAll("{\"error\": \"Invalid JSON\"}");
            return;
        };
        defer parsed.deinit();

        const root = parsed.value.object;

        // Extract model
        const model_val = root.get("model") orelse {
            res.status = 400;
            try res.writer().writeAll("{\"error\": \"Missing model field\"}");
            return;
        };
        const model = model_val.string;

        // Verify it's a DeepSeek model
        if (!isDeepSeekModel(model)) {
            res.status = 400;
            try res.writer().writeAll("{\"error\": \"Invalid model for DeepSeek endpoint\"}");
            return;
        }

        // Lazy-load backend on first request — per D-01
        if (!self.loaded) {
            if (self.model_path.len == 0) {
                res.status = 503;
                try res.json(.{
                    .@"error" = "DeepSeek model not loaded — use /v1/models/switch to select a model path",
                }, .{});
                return;
            }
            self.backend.* = try LlamaBackend.init(self.allocator, self.model_path);
            self.loaded = true;
        }

        var llama_backend = &self.backend.*.?;

        // Log token metadata from backend accessors — per D-06, never hardcode
        const vocab_sz = llama_backend.vocabSize(); // per D-06 read from backend accessors
        const eos_tok = llama_backend.eosToken(); // per D-06 read from backend accessors
        const bos_tok = llama_backend.bosToken(); // per D-06 read from backend accessors
        std.log.info("DeepSeek backend: vocab_size={d}, eos={d}, bos={d}", .{ vocab_sz, eos_tok, bos_tok });

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

        // Extract messages array
        const messages_val = root.get("messages") orelse {
            res.status = 400;
            try res.writer().writeAll("{\"error\": \"Missing messages field\"}");
            return;
        };

        // Build prompt using DeepSeek-Coder instruct format
        var prompt_buf = std.ArrayListUnmanaged(u8).empty;
        defer prompt_buf.deinit(self.allocator);

        for (messages_val.array.items) |msg| {
            const msg_obj = msg.object;
            const role = if (msg_obj.get("role")) |r| r.string else "user";
            const content = if (msg_obj.get("content")) |c| c.string else "";

            if (std.mem.eql(u8, role, "system")) {
                try prompt_buf.appendSlice(self.allocator, "### System:\n");
                try prompt_buf.appendSlice(self.allocator, content);
                try prompt_buf.appendSlice(self.allocator, "\n");
            } else if (std.mem.eql(u8, role, "user")) {
                try prompt_buf.appendSlice(self.allocator, "### Instruction:\n");
                try prompt_buf.appendSlice(self.allocator, content);
                try prompt_buf.appendSlice(self.allocator, "\n### Response:\n");
            } else if (std.mem.eql(u8, role, "assistant")) {
                try prompt_buf.appendSlice(self.allocator, content);
                try prompt_buf.appendSlice(self.allocator, "\n");
            }
        }

        const prompt = prompt_buf.items;
        std.log.debug("DeepSeek prompt ({d} chars): {s}", .{ prompt.len, prompt[0..@min(80, prompt.len)] });

        // Tokenize prompt — tokenize() returns []llama_c.llama_token (i32); convert to []u32
        const llama_tokens = try llama_backend.tokenize(prompt);
        defer self.allocator.free(llama_tokens);

        var u32_tokens = try self.allocator.alloc(u32, llama_tokens.len);
        defer self.allocator.free(u32_tokens);
        for (llama_tokens, 0..) |t, i| {
            u32_tokens[i] = @intCast(t);
        }

        // Build generation params
        const params = GenerationParams{
            .temperature = temperature,
            .top_p = top_p,
            .max_tokens = max_tokens,
        };

        // Generate via llama.cpp vtable wrapper
        var gen_result = try llama_cpp_mod.llamaGenerate(
            @ptrCast(llama_backend),
            u32_tokens,
            params,
            self.allocator,
        );
        defer gen_result.deinit(self.allocator);

        // Collect generated token text
        var output_buf = std.ArrayListUnmanaged(u8).empty;
        defer output_buf.deinit(self.allocator);

        var completion_tokens: u32 = 0;
        while (try gen_result.token_iterator.next(self.allocator)) |tok| {
            defer self.allocator.free(tok.text);
            try output_buf.appendSlice(self.allocator, tok.text);
            completion_tokens += 1;
        }

        const output_text = output_buf.items;

        if (is_stream) {
            try self.sendStreamResponse(res, model, output_text, completion_tokens);
        } else {
            try self.sendCompleteResponse(res, model, output_text, completion_tokens);
        }
    }

    /// Send a complete (non-streaming) response
    fn sendCompleteResponse(
        self: *ChatDeepSeekHandler,
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
        self: *ChatDeepSeekHandler,
        res: *httpz.Response,
        model: []const u8,
        content: []const u8,
        completion_tokens: u32,
    ) !void {
        _ = completion_tokens;

        res.content_type = .EVENTS;
        res.status = 200;

        const id = try std.fmt.allocPrint(self.allocator, "chatcmpl-{d}", .{std.time.timestamp()});
        defer self.allocator.free(id);

        // Send content chunk
        const chunk_json = try std.fmt.allocPrint(
            self.allocator,
            "{{\"id\":\"{s}\",\"object\":\"chat.completion.chunk\",\"created\":{d},\"model\":\"{s}\"," ++
                "\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{s}\"}},\"finish_reason\":null}}]}}",
            .{ id, std.time.timestamp(), model, content },
        );
        defer self.allocator.free(chunk_json);

        const chunk_line = try std.fmt.allocPrint(self.allocator, "data: {s}\n\n", .{chunk_json});
        defer self.allocator.free(chunk_line);

        try res.chunk(chunk_line);

        // Send finish chunk
        const finish_json = try std.fmt.allocPrint(
            self.allocator,
            "{{\"id\":\"{s}\",\"object\":\"chat.completion.chunk\",\"created\":{d},\"model\":\"{s}\"," ++
                "\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"stop\"}}]}}",
            .{ id, std.time.timestamp(), model },
        );
        defer self.allocator.free(finish_json);

        const finish_line = try std.fmt.allocPrint(self.allocator, "data: {s}\n\n", .{finish_json});
        defer self.allocator.free(finish_line);

        try res.chunk(finish_line);
        try res.chunk("data: [DONE]\n\n");
    }
};

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Returns true for any deepseek-prefixed model name
pub fn isDeepSeekModel(model: []const u8) bool {
    return std.mem.startsWith(u8, model, "deepseek");
}
