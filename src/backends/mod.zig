//! mod.zig - Backend module entry point
//!
//! Re-exports public types from the backends module.

const std = @import("std");

// Core backend types
pub const Backend = @import("backend.zig").Backend;
pub const BackendType = @import("backend.zig").BackendType;
pub const BackendPreference = @import("backend.zig").BackendPreference;
pub const GenerationParams = @import("backend.zig").GenerationParams;
pub const GenerationResult = @import("backend.zig").GenerationResult;
pub const TokenIterator = @import("backend.zig").TokenIterator;
pub const TokenResult = @import("backend.zig").TokenResult;
pub const StopReason = @import("backend.zig").StopReason;
pub const KvCacheHandle = @import("backend.zig").KvCacheHandle;
pub const CompressionParams = @import("backend.zig").CompressionParams;
pub const ModelLoadResult = @import("backend.zig").ModelLoadResult;
pub const BackendCapabilities = @import("backend.zig").BackendCapabilities;
pub const BackendStats = @import("backend.zig").BackendStats;

// factory: removed — backlog item, evaluate after core inference is stable
// speculative-decoding: removed — re-evaluate as dedicated phase after core inference is stable

// Backend implementations (real, live-path backends only)
pub const llama_cpp_backend = @import("llama_cpp.zig");

/// Backend module version
pub const VERSION = "0.1.0";

/// Backend initialization error set
pub const BackendError = error{
    ModelLoadFailed,
    ContextCreationFailed,
    TokenizationFailed,
    GenerationFailed,
    CompressionFailed,
    UnsupportedArchitecture,
    CannotAutoDetect,
    OutOfMemory,
};

/// Initialize the backends module
pub fn init() void {
    std.log.info("Backends module v{s} initialized", .{VERSION});
}

/// Get backend description string
pub fn getBackendDescription(backend_type: BackendType) []const u8 {
    return switch (backend_type) {
        // mlx backend: removed stub — Qwen uses inference/mod.zig + MLX.zig directly
        .llama_cpp => "llama.cpp - GGML inference engine",
        .mlx_gptoss => "MLX GPT-OSS - Native MLX GPT-OSS backend",
    };
}

/// Check if backend supports a specific feature
pub fn supportsFeature(backend_type: BackendType, feature: []const u8) bool {
    if (std.mem.eql(u8, feature, "turboquant")) {
        return true; // All backends support TurboQuant
    }
    if (std.mem.eql(u8, feature, "speculative_decoding")) {
        // speculative-decoding: removed — re-evaluate as dedicated phase after core inference is stable
        return false;
    }
    if (std.mem.eql(u8, feature, "prompt_caching")) {
        return true; // All backends support prompt caching
    }
    if (std.mem.eql(u8, feature, "gguf")) {
        return backend_type == .llama_cpp; // Only llama.cpp supports GGUF
    }
    return false;
}
