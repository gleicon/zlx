//! types.zig - OpenAI-compatible API types
//!
//! JSON structs matching OpenAI API specification for chat completions.
//! Uses @"field-name" syntax for snake_case JSON field names.

const std = @import("std");

/// Message role in chat completion
pub const Role = enum {
    system,
    user,
    assistant,
};

/// Chat message structure
pub const Message = struct {
    role: Role,
    content: []const u8,
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
    temperature: ?f32 = null,
    /// Nucleus sampling parameter 0.0-1.0 (default: 0.9)
    top_p: ?f32 = null,
    /// Stop sequences to end generation
    stop: ?StopSequence = null,
    /// Number of completions to generate (default: 1)
    n: ?u32 = null,
    /// Whether to return log probabilities
    logprobs: ?bool = null,
    /// Seed for deterministic sampling
    seed: ?i32 = null,
    /// Presence penalty -2.0 to 2.0
    presence_penalty: ?f32 = null,
    /// Frequency penalty -2.0 to 2.0
    frequency_penalty: ?f32 = null,

    /// Get effective max_tokens
    pub fn getMaxTokens(self: ChatCompletionRequest) u32 {
        return self.max_tokens orelse 256;
    }

    /// Get effective temperature
    pub fn getTemperature(self: ChatCompletionRequest) f32 {
        return self.temperature orelse 0.7;
    }

    /// Get effective top_p
    pub fn getTopP(self: ChatCompletionRequest) f32 {
        return self.top_p orelse 0.9;
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

/// Token log probabilities
pub const LogProbs = struct {
    tokens: []const []const u8,
    token_logprobs: []const f32,
    top_logprobs: ?[]const std.StringHashMap(f32) = null,
    text_offset: []const u32,
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
            .system => {
                try result.appendSlice(allocator, "<|im_start|>system\n");
                try result.appendSlice(allocator, msg.content);
                try result.appendSlice(allocator, "<|im_end|>\n");
            },
            .user => {
                try result.appendSlice(allocator, "<|im_start|>user\n");
                try result.appendSlice(allocator, msg.content);
                try result.appendSlice(allocator, "<|im_end|>\n");
            },
            .assistant => {
                try result.appendSlice(allocator, "<|im_start|>assistant\n");
                try result.appendSlice(allocator, msg.content);
                try result.appendSlice(allocator, "<|im_end|>\n");
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
        .{ .role = .system, .content = "You are a helpful assistant." },
        .{ .role = .user, .content = "Hello!" },
    };

    const prompt = try buildPromptFromMessages(allocator, messages);
    defer allocator.free(prompt);

    try std.testing.expect(std.mem.indexOf(u8, prompt, "<|im_start|>system") != null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "You are a helpful assistant.") != null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "<|im_start|>user") != null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "Hello!") != null);
}
