//! types.zig - OpenAI-compatible API types
//!
//! JSON structs matching OpenAI API specification for chat completions.
//! Uses @"field-name" syntax for snake_case JSON field names.

const std = @import("std");

/// JsonFloat is a float type that can be parsed from either JSON integers or floats
pub const JsonFloat = struct {
    value: f64,

    pub fn jsonParse(allocator: std.mem.Allocator, source: anytype, options: std.json.ParseOptions) std.json.ParseError(@TypeOf(source.*))!JsonFloat {
        _ = allocator;
        _ = options;

        const token = try source.next();
        switch (token) {
            .number => |num_str| {
                const val = try std.fmt.parseFloat(f64, num_str);
                return JsonFloat{ .value = val };
            },
            .allocated_number => |num_str| {
                const val = try std.fmt.parseFloat(f64, num_str);
                return JsonFloat{ .value = val };
            },
            else => return error.UnexpectedToken,
        }
    }
};

/// Message role in chat completion
pub const Role = enum {
    system,
    developer,
    user,
    assistant,
    tool,
};

/// Message content can be a string or an array (for multimodal)
pub const MessageContent = struct {
    text: []const u8,

    pub fn jsonParse(allocator: std.mem.Allocator, source: anytype, options: std.json.ParseOptions) std.json.ParseError(@TypeOf(source.*))!MessageContent {
        _ = options;

        std.log.info("MessageContent.jsonParse called", .{});

        // Handle potentially large strings that come as partial_string tokens
        var buffer = std.ArrayList(u8).empty;
        errdefer buffer.deinit(allocator);

        while (true) {
            const token = try source.next();
            std.log.info("MessageContent got token: {s}", .{@tagName(token)});

            switch (token) {
                .string => |str| {
                    std.log.info("MessageContent: complete string len={d}", .{str.len});
                    if (buffer.items.len == 0) {
                        // Small string, return directly
                        return MessageContent{ .text = try allocator.dupe(u8, str) };
                    } else {
                        // Part of a larger string we were collecting
                        try buffer.appendSlice(allocator, str);
                        return MessageContent{ .text = try buffer.toOwnedSlice(allocator) };
                    }
                },
                .allocated_string => |str| {
                    std.log.info("MessageContent: allocated_string len={d}", .{str.len});
                    if (buffer.items.len == 0) {
                        return MessageContent{ .text = try allocator.dupe(u8, str) };
                    } else {
                        try buffer.appendSlice(allocator, str);
                        return MessageContent{ .text = try buffer.toOwnedSlice(allocator) };
                    }
                },
                .partial_string => |str| {
                    std.log.info("MessageContent: partial_string len={d}, accumulating", .{str.len});
                    try buffer.appendSlice(allocator, str);
                    // Continue to collect more parts
                },
                .partial_string_escaped_1 => |arr| {
                    std.log.info("MessageContent: partial_string_escaped_1", .{});
                    try buffer.appendSlice(allocator, &arr);
                },
                .partial_string_escaped_2 => |arr| {
                    std.log.info("MessageContent: partial_string_escaped_2", .{});
                    try buffer.appendSlice(allocator, &arr);
                },
                .partial_string_escaped_3 => |arr| {
                    std.log.info("MessageContent: partial_string_escaped_3", .{});
                    try buffer.appendSlice(allocator, &arr);
                },
                .partial_string_escaped_4 => |arr| {
                    std.log.info("MessageContent: partial_string_escaped_4", .{});
                    try buffer.appendSlice(allocator, &arr);
                },
                .array_begin => {
                    std.log.info("MessageContent: array_begin - consuming array", .{});
                    // Empty array or array of content parts - treat as empty string for now
                    var depth: usize = 1;
                    while (depth > 0) {
                        const inner = try source.next();
                        switch (inner) {
                            .array_begin => depth += 1,
                            .array_end => depth -= 1,
                            else => {},
                        }
                    }
                    return MessageContent{ .text = try allocator.dupe(u8, "") };
                },
                else => {
                    std.log.info("MessageContent: unexpected token {s}", .{@tagName(token)});
                    return error.UnexpectedToken;
                },
            }
        }
    }
};

/// Chat message structure
pub const Message = struct {
    role: Role,
    content: MessageContent,
    // Optional name field (for distinguishing between multiple users)
    name: ?[]const u8 = null,
};

/// Chat completion request body
pub const ChatCompletionRequest = struct {
    /// Model identifier to use for completion
    model: []const u8,
    /// List of messages for the conversation
    messages: []const Message,
    /// Whether to stream the response (default: false)
    stream: bool = false,
    /// Maximum tokens to generate (default: 256)
    max_tokens: ?u32 = null,
    /// Sampling temperature 0.0-2.0 (default: 0.7)
    temperature: ?JsonFloat = null,
    /// Nucleus sampling parameter 0.0-1.0 (default: 0.9)
    top_p: ?JsonFloat = null,
    /// Stop sequences to end generation
    stop: ?StopSequence = null,
    /// Number of completions to generate (default: 1)
    n: ?u32 = null,
    /// Whether to return log probabilities
    logprobs: ?bool = null,
    /// Seed for deterministic sampling
    seed: ?i32 = null,
    /// Presence penalty -2.0 to 2.0
    presence_penalty: ?JsonFloat = null,
    /// Frequency penalty -2.0 to 2.0
    frequency_penalty: ?JsonFloat = null,
    /// Top-k sampling (0 = disabled, API-03)
    top_k: ?u32 = null,
    /// Min-p sampling parameter (API-03)
    min_p: ?JsonFloat = null,
    /// Repetition penalty (API-03)
    repetition_penalty: ?JsonFloat = null,
    /// Logit bias map (API-03)
    logit_bias: ?std.json.Value = null,

    // Fields that OpenCode sends but we don't use (defined to avoid parse errors)
    /// Tools for function calling (not implemented, ignored)
    tools: ?std.json.Value = null,
    /// Tool choice (not implemented, ignored)
    tool_choice: ?std.json.Value = null,
    /// Stream options (not implemented, ignored)
    stream_options: ?std.json.Value = null,
    /// Response format (not implemented, ignored)
    response_format: ?std.json.Value = null,

    /// Get effective max_tokens (capped at 4096 to prevent memory exhaustion)
    pub fn getMaxTokens(self: ChatCompletionRequest) u32 {
        const requested = self.max_tokens orelse 256;
        return @min(requested, 4096);
    }

    /// Get effective temperature
    pub fn getTemperature(self: ChatCompletionRequest) f32 {
        return if (self.temperature) |t| @floatCast(t.value) else 0.7;
    }

    /// Get effective top_p
    pub fn getTopP(self: ChatCompletionRequest) f32 {
        return if (self.top_p) |tp| @floatCast(tp.value) else 0.9;
    }

    /// Get effective top_k (0 = disabled)
    pub fn getTopK(self: ChatCompletionRequest) u32 {
        return self.top_k orelse 0;
    }

    /// Get effective min_p
    pub fn getMinP(self: ChatCompletionRequest) f32 {
        return if (self.min_p) |mp| @floatCast(mp.value) else 0.0;
    }

    /// Get effective presence_penalty (-2.0 to 2.0, default 0.0)
    pub fn getPresencePenalty(self: ChatCompletionRequest) f32 {
        return if (self.presence_penalty) |pp| @floatCast(pp.value) else 0.0;
    }

    /// Get effective frequency_penalty (-2.0 to 2.0, default 0.0)
    pub fn getFrequencyPenalty(self: ChatCompletionRequest) f32 {
        return if (self.frequency_penalty) |fp| @floatCast(fp.value) else 0.0;
    }

    /// Get effective repetition_penalty (1.0 = disabled)
    pub fn getRepetitionPenalty(self: ChatCompletionRequest) f32 {
        return if (self.repetition_penalty) |rp| @floatCast(rp.value) else 1.0;
    }

    /// Parse logit_bias from JSON value into hashmap
    /// Caller owns the returned hashmap
    pub fn getLogitBias(self: ChatCompletionRequest, allocator: std.mem.Allocator) !std.AutoHashMap(u32, f32) {
        var map = std.AutoHashMap(u32, f32).init(allocator);
        errdefer map.deinit();

        if (self.logit_bias) |bias_value| {
            if (bias_value == .object) {
                var it = bias_value.object.iterator();
                while (it.next()) |entry| {
                    const token_id = try std.fmt.parseInt(u32, entry.key_ptr.*, 10);
                    const bias_val: f32 = switch (entry.value_ptr.*) {
                        .number => |num_str| try std.fmt.parseFloat(f32, num_str),
                        .integer => |int_val| @floatFromInt(int_val),
                        .float => |float_val| @floatCast(float_val),
                        else => 0.0,
                    };
                    try map.put(token_id, bias_val);
                }
            }
        }

        return map;
    }
};

/// Stop sequence can be a single string or array of strings
pub const StopSequence = union(enum) {
    single: []const u8,
    multiple: []const []const u8,
};

/// Choice in chat completion response
pub const Choice = struct {
    /// Index of this choice
    index: u32 = 0,
    /// The generated message
    message: Message,
    /// Reason generation stopped
    finish_reason: ?[]const u8 = null,
    /// Log probabilities (optional)
    logprobs: ?LogProbs = null,
};

/// Choice for streaming responses
pub const StreamingChoice = struct {
    /// Index of this choice
    index: u32 = 0,
    /// Delta content for this chunk
    delta: DeltaMessage,
    /// Reason generation stopped (only in final chunk)
    finish_reason: ?[]const u8 = null,
    /// Log probabilities (optional)
    logprobs: ?LogProbs = null,
};

/// Delta message for streaming (partial content)
pub const DeltaMessage = struct {
    /// Role (usually only in first chunk)
    role: ?Role = null,
    /// Content fragment
    content: ?[]const u8 = null,
};

/// Token log probabilities — OpenAI compatible format
pub const LogProbs = struct {
    /// Tokens generated (strings)
    tokens: []const []const u8,
    /// Log probability of each generated token
    token_logprobs: []const f32,
    /// Top-k alternative tokens at each position - array of objects mapping token->logprob
    top_logprobs: ?[]const []const TopLogprobAlternative = null,
    /// Character offset for each token in the output
    text_offset: []const u32,
};

/// Alternative token with logprob — for top_logprobs
pub const TopLogprobAlternative = struct {
    token: []const u8,
    logprob: f32,
    bytes: ?[]const u8 = null, // Raw bytes if different from token string
};

/// Token usage statistics
pub const Usage = struct {
    /// Tokens in the prompt
    prompt_tokens: u32,
    /// Tokens in the completion
    completion_tokens: u32,
    /// Total tokens used
    total_tokens: u32,
};

/// Chat completion response (non-streaming)
pub const ChatCompletionResponse = struct {
    /// Unique identifier for this completion
    id: []const u8,
    /// Object type (always "chat.completion")
    object: []const u8 = "chat.completion",
    /// Unix timestamp of creation
    created: i64,
    /// Model used for completion
    model: []const u8,
    /// List of completion choices
    choices: []const Choice,
    /// Token usage statistics
    usage: Usage,
    /// System fingerprint (optional)
    system_fingerprint: ?[]const u8 = null,
};

/// Chat completion chunk for streaming
pub const ChatCompletionChunk = struct {
    /// Unique identifier for this completion
    id: []const u8,
    /// Object type (always "chat.completion.chunk")
    object: []const u8 = "chat.completion.chunk",
    /// Unix timestamp of creation
    created: i64,
    /// Model used for completion
    model: []const u8,
    /// List of completion choices (usually 1)
    choices: []const StreamingChoice,
    /// System fingerprint (optional)
    system_fingerprint: ?[]const u8 = null,
};

/// Model information for /v1/models endpoint
pub const ModelInfo = struct {
    /// Model identifier
    id: []const u8,
    /// Object type (always "model")
    object: []const u8 = "model",
    /// Unix timestamp of creation
    created: i64,
    /// Owner of the model
    owned_by: []const u8 = "local",
};

/// Models list response
pub const ModelsResponse = struct {
    /// Object type (always "list")
    object: []const u8 = "list",
    /// List of available models
    data: []const ModelInfo,
};

/// Error response structure
pub const ErrorResponse = struct {
    @"error": ApiError,
};

/// API error details
pub const ApiError = struct {
    /// Error message
    message: []const u8,
    /// Error type
    type: []const u8 = "invalid_request_error",
    /// Parameter that caused the error (optional)
    param: ?[]const u8 = null,
    /// Error code (optional)
    code: ?[]const u8 = null,
};

/// Generate a unique completion ID
pub fn generateCompletionId(allocator: std.mem.Allocator) ![]const u8 {
    const timestamp = std.time.milliTimestamp();
    const random = std.crypto.random.int(u32);
    return std.fmt.allocPrint(allocator, "chatcmpl-{x}-{x}", .{ timestamp, random });
}

/// Build chat prompt from messages array
pub fn buildPromptFromMessages(allocator: std.mem.Allocator, messages: []const Message) ![]const u8 {
    var result = std.ArrayList(u8).empty;
    errdefer result.deinit(allocator);

    for (messages) |msg| {
        switch (msg.role) {
            .system, .developer => {
                try result.appendSlice(allocator, "<|im_start|>system\n");
                try result.appendSlice(allocator, msg.content.text);
                try result.appendSlice(allocator, "  \n");
            },
            .user => {
                try result.appendSlice(allocator, "<|im_start|>user\n");
                try result.appendSlice(allocator, msg.content.text);
                try result.appendSlice(allocator, "  \n");
            },
            .assistant => {
                try result.appendSlice(allocator, "<|im_start|>assistant\n");
                try result.appendSlice(allocator, msg.content.text);
                try result.appendSlice(allocator, "  \n");
            },
            .tool => {
                // Tool messages are treated as system context for the model
                try result.appendSlice(allocator, "<|im_start|>system\nTool result: ");
                try result.appendSlice(allocator, msg.content.text);
                try result.appendSlice(allocator, "  \n");
            },
        }
    }

    // Add final assistant prefix to prompt generation
    try result.appendSlice(allocator, "<|im_start|>assistant\n");

    return result.toOwnedSlice(allocator);
}

test "types - prompt building" {
    const allocator = std.testing.allocator;

    const messages = &[_]Message{
        .{ .role = .system, .content = .{ .text = "You are a helpful assistant." } },
        .{ .role = .user, .content = .{ .text = "Hello!" } },
    };

    const prompt = try buildPromptFromMessages(allocator, messages);
    defer allocator.free(prompt);

    try std.testing.expect(std.mem.indexOf(u8, prompt, "<|im_start|>system") != null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "You are a helpful assistant.") != null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "<|im_start|>user") != null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "Hello!") != null);
}
