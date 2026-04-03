//! llama_cpp.zig - llama.cpp backend implementation
//!
//! Implements the Backend interface using llama.cpp for DeepSeek and GPT-OSS models.

const std = @import("std");
const backend = @import("backend.zig");

/// llama.cpp backend state
pub const LlamaBackend = struct {
    // TODO(14-03, 14-04): Implement llama.cpp backend
    // This will wrap llama.cpp C API and provide Backend interface

    allocator: std.mem.Allocator,
    model_path: []const u8,

    pub fn init(allocator: std.mem.Allocator, model_path: []const u8) !LlamaBackend {
        return LlamaBackend{
            .allocator = allocator,
            .model_path = try allocator.dupe(u8, model_path),
        };
    }

    pub fn deinit(self: *LlamaBackend) void {
        self.allocator.free(self.model_path);
    }
};

/// Tokenize using llama.cpp tokenizer
pub fn llamaTokenize(ptr: *anyopaque, text: []const u8, allocator: std.mem.Allocator) anyerror![]u32 {
    _ = ptr;
    _ = text;
    // TODO(14-03): Implement llama.cpp tokenization
    return allocator.alloc(u32, 0);
}

/// Generate using llama.cpp
pub fn llamaGenerate(
    ptr: *anyopaque,
    tokens: []const u32,
    params: backend.GenerationParams,
    allocator: std.mem.Allocator,
) anyerror!backend.GenerationResult {
    _ = ptr;
    _ = tokens;
    _ = params;
    _ = allocator;
    // TODO(14-03): Implement llama.cpp generation
    return backend.GenerationResult{
        .token_iterator = undefined,
        .tokens_generated = 0,
        .finish_reason = .error_status,
    };
}

/// Deinit llama.cpp backend
pub fn llamaDeinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
    const backend_ptr = @as(*LlamaBackend, @ptrCast(@alignCast(ptr)));
    backend_ptr.deinit();
    allocator.destroy(backend_ptr);
}

/// Get vocabulary size
pub fn llamaGetVocabSize(ptr: *anyopaque) u32 {
    _ = ptr;
    // TODO(14-03): Get actual vocab size from llama.cpp model
    return 32000;
}

/// Get EOS token
pub fn llamaEosToken(ptr: *anyopaque) u32 {
    _ = ptr;
    // TODO(14-03): Get actual EOS token from llama.cpp tokenizer
    return 2;
}

/// Get BOS token
pub fn llamaBosToken(ptr: *anyopaque) u32 {
    _ = ptr;
    // TODO(14-03): Get actual BOS token from llama.cpp tokenizer
    return 1;
}

/// Get KV cache handle
pub fn llamaGetKvCache(ptr: *anyopaque) ?backend.KvCacheHandle {
    _ = ptr;
    // TODO(14-03): Implement llama.cpp KV cache access
    return null;
}

/// Apply compression to llama.cpp KV cache
pub fn llamaApplyCompression(ptr: *anyopaque, params: backend.CompressionParams) anyerror!void {
    _ = ptr;
    // llama.cpp has built-in quantization, but for TurboQuant integration:
    std.log.info("llama.cpp TurboQuant compression: bits={d}, adaptive_layers={d}", .{
        params.bits,
        params.adaptive_layers,
    });
    // TODO(14-05): Implement TurboQuant compression for llama.cpp KV cache
}
