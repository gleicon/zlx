//! backend.zig - Unified Backend interface for MLX.zig and llama.cpp
//!
//! Provides backend-agnostic types for inference across different backends.

const std = @import("std");

/// Backend type enumeration
pub const BackendType = enum {
    mlx,
    llama_cpp,
    mlx_gptoss, // Native MLX GPT-OSS backend (Phase 15)
};

/// Backend preference for factory creation
pub const BackendPreference = enum {
    auto, // Auto-select based on architecture
    mlx, // Force MLX.zig backend
    llama_cpp, // Force llama.cpp backend
};

/// Generation parameters (backend-agnostic)
pub const GenerationParams = struct {
    temperature: f32 = 0.7,
    top_p: f32 = 0.9,
    top_k: u32 = 40,
    max_tokens: u32 = 1024,
    seed: u64 = 0,
    stop_sequences: ?[]const []const u8 = null,
    presence_penalty: f32 = 0.0,
    frequency_penalty: f32 = 0.0,
    repetition_penalty: f32 = 1.0,
    min_p: f32 = 0.0,
};

/// Stop reason for generation
pub const StopReason = enum {
    eos,
    length,
    stop,
    timeout,
    error_status,
};

/// Single token result with metadata
pub const TokenResult = struct {
    token: u32,
    text: []const u8,
    logprob: f32,
    finish_reason: ?StopReason,
};

/// Generation result with token iterator
pub const GenerationResult = struct {
    token_iterator: TokenIterator,
    tokens_generated: u32 = 0,
    finish_reason: StopReason = .length,

    pub fn deinit(self: *GenerationResult, allocator: std.mem.Allocator) void {
        self.token_iterator.deinit(allocator);
    }
};

/// Token iterator for streaming generation
pub const TokenIterator = struct {
    /// Function pointer to get next token
    next_fn: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator) anyerror!?TokenResult,
    /// Context pointer for iterator state
    ctx: *anyopaque,
    /// Cleanup function
    deinit_fn: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator) void,

    pub fn next(self: *TokenIterator, allocator: std.mem.Allocator) anyerror!?TokenResult {
        return self.next_fn(self.ctx, allocator);
    }

    pub fn deinit(self: *TokenIterator, allocator: std.mem.Allocator) void {
        self.deinit_fn(self.ctx, allocator);
    }
};

/// KV cache handle for TurboQuant integration
pub const KvCacheHandle = struct {
    /// Opaque pointer to backend-specific KV cache
    ptr: *anyopaque,
    backend_type: BackendType,

    /// Get raw pointer for compression engine
    pub fn getRaw(self: KvCacheHandle) *anyopaque {
        return self.ptr;
    }

    /// Get backend type for type-safe operations
    pub fn getType(self: KvCacheHandle) BackendType {
        return self.backend_type;
    }
};

/// Compression parameters for TurboQuant
pub const CompressionParams = struct {
    enabled: bool,
    bits: u4, // 3 or 4
    adaptive_layers: u8,
    group_size: u32 = 64,
};

/// Model load result
pub const ModelLoadResult = struct {
    vocab_size: u32,
    eos_token: u32,
    bos_token: u32,
    context_size: u32,
    layers: u32,
};

/// Select backend type based on model name
/// GPT-OSS models → mlx_gptoss, GGUF models → llama_cpp, others → mlx
pub fn selectBackend(model_name: []const u8) BackendType {
    if (std.mem.startsWith(u8, model_name, "gpt-oss") or
        std.mem.startsWith(u8, model_name, "gptoss"))
    {
        return .mlx_gptoss;
    }
    if (std.mem.endsWith(u8, model_name, ".gguf") or
        std.mem.startsWith(u8, model_name, "deepseek"))
    {
        return .llama_cpp;
    }
    return .mlx;
}

/// Backend union - holds MLX, llama.cpp, or mlx_gptoss backend
pub const Backend = union(BackendType) {
    mlx: *anyopaque,       // Pointer to MlxBackend
    llama_cpp: *anyopaque, // Pointer to LlamaBackend
    mlx_gptoss: *anyopaque, // Pointer to MLXGPTOSSBackend

    /// Get backend type
    pub fn getType(self: Backend) BackendType {
        return switch (self) {
            .mlx => .mlx,
            .llama_cpp => .llama_cpp,
            .mlx_gptoss => .mlx_gptoss,
        };
    }

    /// Tokenize text into token IDs
    pub fn tokenize(
        self: Backend,
        text: []const u8,
        allocator: std.mem.Allocator,
    ) anyerror![]u32 {
        return switch (self) {
            .mlx => |ptr| mlxTokenize(ptr, text, allocator),
            .llama_cpp => |ptr| llamaTokenize(ptr, text, allocator),
            .mlx_gptoss => |ptr| gptossTokenize(ptr, text, allocator),
        };
    }

    /// Generate tokens from prompt
    pub fn generate(
        self: Backend,
        tokens: []const u32,
        params: GenerationParams,
        allocator: std.mem.Allocator,
    ) anyerror!GenerationResult {
        return switch (self) {
            .mlx => |ptr| mlxGenerate(ptr, tokens, params, allocator),
            .llama_cpp => |ptr| llamaGenerate(ptr, tokens, params, allocator),
            .mlx_gptoss => |ptr| gptossGenerate(ptr, tokens, params, allocator),
        };
    }

    /// Free backend resources
    pub fn deinit(self: Backend, allocator: std.mem.Allocator) void {
        switch (self) {
            .mlx => |ptr| mlxDeinit(ptr, allocator),
            .llama_cpp => |ptr| llamaDeinit(ptr, allocator),
            .mlx_gptoss => |ptr| gptosDeinit(ptr, allocator),
        }
    }

    /// Get vocabulary size
    pub fn getVocabSize(self: Backend) u32 {
        return switch (self) {
            .mlx => |ptr| mlxGetVocabSize(ptr),
            .llama_cpp => |ptr| llamaGetVocabSize(ptr),
            .mlx_gptoss => |ptr| gptossGetVocabSize(ptr),
        };
    }

    /// Get end-of-sequence token ID
    pub fn eosToken(self: Backend) u32 {
        return switch (self) {
            .mlx => |ptr| mlxEosToken(ptr),
            .llama_cpp => |ptr| llamaEosToken(ptr),
            .mlx_gptoss => |ptr| gptossEosToken(ptr),
        };
    }

    /// Get beginning-of-sequence token ID
    pub fn bosToken(self: Backend) u32 {
        return switch (self) {
            .mlx => |ptr| mlxBosToken(ptr),
            .llama_cpp => |ptr| llamaBosToken(ptr),
            .mlx_gptoss => |ptr| gptosBosToken(ptr),
        };
    }

    /// Get KV cache handle for TurboQuant
    pub fn getKvCache(self: Backend) ?KvCacheHandle {
        return switch (self) {
            .mlx => |ptr| mlxGetKvCache(ptr),
            .llama_cpp => |ptr| llamaGetKvCache(ptr),
            .mlx_gptoss => null, // GPT-OSS KV cache not yet integrated with TurboQuant
        };
    }

    /// Apply compression to KV cache
    pub fn applyCompression(
        self: Backend,
        params: CompressionParams,
    ) anyerror!void {
        if (!params.enabled) return;

        return switch (self) {
            .mlx => |ptr| mlxApplyCompression(ptr, params),
            .llama_cpp => |ptr| llamaApplyCompression(ptr, params),
            .mlx_gptoss => {}, // GPT-OSS compression not yet implemented
        };
    }

    // VTable function declarations - implemented in respective backend modules
    extern fn mlxTokenize(*anyopaque, []const u8, std.mem.Allocator) anyerror![]u32;
    extern fn llamaTokenize(*anyopaque, []const u8, std.mem.Allocator) anyerror![]u32;
    extern fn gptossTokenize(*anyopaque, []const u8, std.mem.Allocator) anyerror![]u32;
    extern fn mlxGenerate(*anyopaque, []const u32, GenerationParams, std.mem.Allocator) anyerror!GenerationResult;
    extern fn llamaGenerate(*anyopaque, []const u32, GenerationParams, std.mem.Allocator) anyerror!GenerationResult;
    extern fn gptossGenerate(*anyopaque, []const u32, GenerationParams, std.mem.Allocator) anyerror!GenerationResult;
    extern fn mlxDeinit(*anyopaque, std.mem.Allocator) void;
    extern fn llamaDeinit(*anyopaque, std.mem.Allocator) void;
    extern fn gptosDeinit(*anyopaque, std.mem.Allocator) void;
    extern fn mlxGetVocabSize(*anyopaque) u32;
    extern fn llamaGetVocabSize(*anyopaque) u32;
    extern fn gptossGetVocabSize(*anyopaque) u32;
    extern fn mlxEosToken(*anyopaque) u32;
    extern fn llamaEosToken(*anyopaque) u32;
    extern fn gptossEosToken(*anyopaque) u32;
    extern fn mlxBosToken(*anyopaque) u32;
    extern fn llamaBosToken(*anyopaque) u32;
    extern fn gptosBosToken(*anyopaque) u32;
    extern fn mlxGetKvCache(*anyopaque) ?KvCacheHandle;
    extern fn llamaGetKvCache(*anyopaque) ?KvCacheHandle;
    extern fn mlxApplyCompression(*anyopaque, CompressionParams) anyerror!void;
    extern fn llamaApplyCompression(*anyopaque, CompressionParams) anyerror!void;
};

/// Backend capabilities (optional features)
pub const BackendCapabilities = struct {
    supports_turboquant: bool,
    supports_speculative_decoding: bool,
    supports_prompt_caching: bool,
    max_context_length: u32,
    preferred_quantization: []const u8,
};

/// Backend statistics for monitoring
pub const BackendStats = struct {
    tokens_generated: u64,
    tokens_per_second: f32,
    memory_used_mb: u64,
    kv_cache_size_mb: u64,
    gpu_layers: u32,
};
