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

// Factory for backend creation
pub const factory = @import("factory.zig");

// Backend implementations (when available)
// These will be populated in subsequent plans
pub const mlx_backend = @import("mlx_backend.zig");
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
        .mlx => "MLX.zig - Apple's ML framework",
        .llama_cpp => "llama.cpp - GGML inference engine",
    };
}

/// Check if backend supports a specific feature
pub fn supportsFeature(backend_type: BackendType, feature: []const u8) bool {
    if (std.mem.eql(u8, feature, "turboquant")) {
        return true; // All backends support TurboQuant
    }
    if (std.mem.eql(u8, feature, "speculative_decoding")) {
        return backend_type == .mlx; // Only MLX supports speculative decoding currently
    }
    if (std.mem.eql(u8, feature, "prompt_caching")) {
        return true; // All backends support prompt caching
    }
    if (std.mem.eql(u8, feature, "gguf")) {
        return backend_type == .llama_cpp; // Only llama.cpp supports GGUF
    }
    return false;
}

// Test helper: check if backends compile
test "backends module compiles" {
    const testing = std.testing;

    // Verify all types are exported
    _ = BackendType.mlx;
    _ = BackendType.llama_cpp;
    _ = BackendPreference.auto;

    // Verify backend descriptions
    try testing.expectEqualStrings("MLX.zig - Apple's ML framework", getBackendDescription(.mlx));
    try testing.expectEqualStrings("llama.cpp - GGML inference engine", getBackendDescription(.llama_cpp));

    // Verify feature support
    try testing.expect(supportsFeature(.mlx, "prompt_caching"));
    try testing.expect(supportsFeature(.llama_cpp, "prompt_caching"));
    try testing.expect(supportsFeature(.llama_cpp, "gguf"));
    try testing.expect(!supportsFeature(.mlx, "gguf"));
}
