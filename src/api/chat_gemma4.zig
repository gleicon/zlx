//! chat_gemma4.zig - Gemma 4 E4B chat API handler
//!
//! Handles /v1/chat/completions requests for Gemma 4 E4B models.
//! Uses LlamaBackend (llama.cpp) for GGUF inference — per D-01 handler-per-model pattern.
//! No factory abstraction: server.zig owns the globals and dispatches directly.

const std = @import("std");
const httpz = @import("httpz");
const backend_mod = @import("../backends/backend.zig");
const llama_cpp_mod = @import("../backends/llama_cpp.zig");

const GenerationParams = backend_mod.GenerationParams;
const LlamaBackend = llama_cpp_mod.LlamaBackend;

// ── Gemma 4 Chat Template Control Tokens ───────────────────────────────────────

// Turn delimiters
const TURN_START = "<|turn>";
const TURN_END = "<turn|>\n";

// Channel/no-think tokens (per D-04)
const CHANNEL_START = "<|channel>";
const CHANNEL_END = "<channel|>";
const NO_THINK_SUFFIX = "<|channel>thought\n<channel|>";

// Role names in Gemma 4 template (maps OpenAI "assistant" to "model")
const ROLE_SYSTEM = "system";
const ROLE_USER = "user";
const ROLE_MODEL = "model";

// Default temperature for verbosity reduction (per D-04)
const DEFAULT_TEMPERATURE: f32 = 0.3;

// ── Request / Response types ───────────────────────────────────────────────────

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

// ── Handler ─────────────────────────────────────────────────────────────────────

/// Gemma 4 E4B chat API handler — per D-01 handler-per-model pattern
/// Wraps LlamaBackend (llama.cpp) for Gemma 4 GGUF models.
pub const ChatGemma4Handler = struct {
    allocator: std.mem.Allocator,
    backend: *?LlamaBackend,
    model_path: []const u8 = "",
    loaded: bool = false,

    /// Initialize with a pointer to the module-level optional LlamaBackend
    pub fn init(allocator: std.mem.Allocator, backend_ptr: *?LlamaBackend, model_path: []const u8) ChatGemma4Handler {
        return .{
            .allocator = allocator,
            .backend = backend_ptr,
            .model_path = model_path,
        };
    }

    /// Handle /v1/chat/completions for Gemma 4 models
    pub fn handle(self: *ChatGemma4Handler, req: *httpz.Request, res: *httpz.Response) !void {
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

        // Verify it's a Gemma 4 model
        if (!isGemma4Model(model)) {
            res.status = 400;
            try res.writer().writeAll("{\"error\": \"Invalid model for Gemma 4 endpoint\"}");
            return;
        }

        // Lazy-load backend on first request — per D-01
        if (!self.loaded) {
            if (self.model_path.len == 0) {
                res.status = 503;
                try res.json(.{
                    .@"error" = "Gemma 4 model not loaded — use /v1/models/switch to select a model path",
                }, .{});
                return;
            }
            self.backend.* = try LlamaBackend.init(self.allocator, self.model_path);
            self.loaded = true;

            // Log verbosity reduction mode at load time (per D-04)
            std.log.info("[gemma4] loaded {s}, verbosity reduction: always-on (no-think mode, temperature floor {d})", .{ self.model_path, DEFAULT_TEMPERATURE });
        }

        var llama_backend = &self.backend.*.?;

        // Log token metadata from backend accessors — per D-06, never hardcode
        const vocab_sz = llama_backend.vocabSize();
        const eos_tok = llama_backend.eosToken();
        const bos_tok = llama_backend.bosToken();
        std.log.info("Gemma 4 backend: vocab_size={d}, eos={d}, bos={d}", .{ vocab_sz, eos_tok, bos_tok });

        // Extract generation parameters
        // Temperature: use request value if provided, otherwise DEFAULT_TEMPERATURE (per D-04)
        const temperature: f32 = blk: {
            if (root.get("temperature")) |v| {
                break :blk @floatCast(v.float);
            }
            break :blk DEFAULT_TEMPERATURE;
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

        // Build prompt using Gemma 4 chat template with no-think mode always-on
        const prompt = try buildGemma4Prompt(
            self.allocator,
            messages_val,
            true, // add_generation_prompt
            false, // enable_thinking: always false per D-04
        );
        defer self.allocator.free(prompt);

        std.log.debug("Gemma 4 prompt ({d} chars): {s}", .{ prompt.len, prompt[0..@min(80, prompt.len)] });

        // Tokenize prompt
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

        // Generate via llama.cpp
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
        self: *ChatGemma4Handler,
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
            }{
                .{
                    .index = 0,
                    .message = .{
                        .role = "assistant",
                        .content = content,
                    },
                    .finish_reason = "stop",
                },
            },
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
        self: *ChatGemma4Handler,
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

/// Build prompt using Gemma 4 E4B chat template
///
/// Format per gemma4.jinja:
/// - System turn: <|turn>system\n{content}<turn|>\n
/// - User turn: <|turn>user\n{content}<turn|>\n
/// - Assistant turn: <|turn>model\n{content}<turn|>\n
/// - Generation prompt: <|turn>model\n + NO_THINK_SUFFIX (if !enable_thinking)
fn buildGemma4Prompt(
    allocator: std.mem.Allocator,
    messages: std.json.Value,
    add_generation_prompt: bool,
    enable_thinking: bool, // per D-04: always false (no-think always on)
) ![]const u8 {
    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(allocator);

    // BOS token is handled by llama.cpp tokenizer, not prepended here

    for (messages.array.items) |msg| {
        const msg_obj = msg.object;
        const role = if (msg_obj.get("role")) |r| r.string else "user";
        const content = if (msg_obj.get("content")) |c| c.string else "";

        // Map OpenAI roles to Gemma 4 roles
        const gemma_role = if (std.mem.eql(u8, role, "assistant")) ROLE_MODEL else role;

        // Format: <|turn>{role}\n{content}<turn|>\n
        try buf.appendSlice(allocator, TURN_START);
        try buf.appendSlice(allocator, gemma_role);
        try buf.appendSlice(allocator, "\n");
        try buf.appendSlice(allocator, content);
        try buf.appendSlice(allocator, TURN_END);
    }

    // Add generation prompt with no-think suffix (per D-04)
    if (add_generation_prompt) {
        try buf.appendSlice(allocator, TURN_START);
        try buf.appendSlice(allocator, ROLE_MODEL);
        try buf.appendSlice(allocator, "\n");

        // No-think mode: suppress thinking channel
        if (!enable_thinking) {
            try buf.appendSlice(allocator, NO_THINK_SUFFIX);
        }
    }

    return try buf.toOwnedSlice(allocator);
}

/// Returns true for any gemma4 or gemma-4 prefixed model name
pub fn isGemma4Model(model: []const u8) bool {
    return std.mem.startsWith(u8, model, "gemma4") or
        std.mem.startsWith(u8, model, "gemma-4") or
        std.mem.startsWith(u8, model, "gemma4-") or
        std.mem.startsWith(u8, model, "gemma-");
}
