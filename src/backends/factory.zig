//! factory.zig - Backend factory for automatic backend selection
//!
//! Provides factory pattern for creating backends based on model architecture.

const std = @import("std");
const backend = @import("backend.zig");
const registry = @import("../models/registry.zig");

/// Re-export BackendPreference from backend module
pub const BackendPreference = backend.BackendPreference;

/// Detect which backend to use for a given architecture
///
/// Routing decisions:
/// - Qwen -> MLX.zig (proven stable, optimized)
/// - DeepSeek MoE -> llama.cpp (better MoE support, GGUF ecosystem)
/// - GPT-OSS -> llama.cpp (better sliding window attn support)
/// - Llama -> llama.cpp (native support)
/// - Phi -> MLX.zig (works well with MLX)
pub fn detectBackendType(arch: registry.ModelArchitecture) BackendPreference {
    return switch (arch) {
        .qwen => .mlx, // Qwen works best with MLX.zig - proven stable
        .phi => .mlx, // Phi works well with MLX.zig
        .deepseek_v2_moe => .llama_cpp, // DeepSeek MoE has better support in llama.cpp
        .deepseek_v1 => .llama_cpp, // DeepSeek V1 via llama.cpp
        .gpt_oss => .llama_cpp, // GPT-OSS needs llama.cpp for sliding window attention
        .llama => .llama_cpp, // Native llama support in llama.cpp
        .unknown => .auto, // Cannot determine, let caller decide
    };
}

/// Create appropriate backend for model
///
/// Arguments:
///   - allocator: Memory allocator for backend creation
///   - model_path: Path to model weights (GGUF or safetensors)
///   - arch: Model architecture from registry
///   - preference: Backend preference (auto will use detectBackendType)
///
/// Returns: Initialized Backend union
///
/// Errors:
///   - error.CannotAutoDetect if preference is .auto and arch is .unknown
///   - error.ModelLoadFailed if backend creation fails
///   - error.OutOfMemory on allocation failure
pub fn createBackend(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    arch: registry.ModelArchitecture,
    preference: BackendPreference,
) !backend.Backend {
    const backend_type = if (preference == .auto)
        detectBackendType(arch)
    else
        preference;

    if (backend_type == .auto) {
        return error.CannotAutoDetect;
    }

    std.log.info("Creating {s} backend for {s} model at {s}", .{
        @tagName(backend_type),
        @tagName(arch),
        model_path,
    });

    return switch (backend_type) {
        .mlx => createMlxBackend(allocator, model_path, arch),
        .llama_cpp => createLlamaBackend(allocator, model_path, arch),
        .auto => error.CannotAutoDetect,
    };
}

/// Detect model file format from path
pub const ModelFormat = enum {
    gguf,
    safetensors,
    unknown,
};

/// Detect model format from file path
pub fn detectModelFormat(model_path: []const u8) ModelFormat {
    if (std.mem.endsWith(u8, model_path, ".gguf")) {
        return .gguf;
    } else if (std.mem.endsWith(u8, model_path, ".safetensors")) {
        return .safetensors;
    } else if (std.mem.indexOf(u8, model_path, "model.safetensors") != null) {
        return .safetensors;
    } else if (std.mem.indexOf(u8, model_path, ".gguf") != null) {
        return .gguf;
    }
    return .unknown;
}

/// Backend creation options
pub const BackendOptions = struct {
    /// Context size in tokens
    context_size: u32 = 8192,
    /// Number of GPU layers to offload (0 = CPU only, 1000 = all)
    gpu_layers: u32 = 1000,
    /// Thread count for CPU inference
    threads: u32 = 0, // 0 = auto-detect
    /// Batch size for prompt processing
    batch_size: u32 = 512,
    /// Random seed for deterministic generation
    seed: u64 = 0,
};

/// Get default backend options
pub fn defaultOptions() BackendOptions {
    return .{};
}

/// Create MLX backend (placeholder for 14-03, 14-04)
fn createMlxBackend(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    arch: registry.ModelArchitecture,
) !backend.Backend {
    _ = arch;

    std.log.info("Initializing MLX backend for {s}", .{model_path});

    // TODO(14-05): Implement actual MLX backend creation
    // For now, return stub that will be implemented in subsequent plans

    // Create placeholder opaque pointer
    const ptr = try allocator.create(u8);
    ptr.* = 0; // Just a marker

    return backend.Backend{ .mlx = ptr };
}

/// Create llama.cpp backend (placeholder for 14-03, 14-04)
fn createLlamaBackend(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    arch: registry.ModelArchitecture,
) !backend.Backend {
    _ = arch;

    std.log.info("Initializing llama.cpp backend for {s}", .{model_path});

    // TODO(14-03, 14-04): Implement actual llama.cpp backend creation
    // For now, return stub that will be implemented in subsequent plans

    // Create placeholder opaque pointer
    const ptr = try allocator.create(u8);
    ptr.* = 0; // Just a marker

    return backend.Backend{ .llama_cpp = ptr };
}

/// Get backend capabilities for a given architecture
pub fn getCapabilities(
    arch: registry.ModelArchitecture,
    preference: BackendPreference,
) backend.BackendCapabilities {
    const backend_type = if (preference == .auto)
        detectBackendType(arch)
    else
        preference;

    return switch (backend_type) {
        .mlx => .{
            .supports_turboquant = true,
            .supports_speculative_decoding = true,
            .supports_prompt_caching = true,
            .max_context_length = 32768,
            .preferred_quantization = "4bit",
        },
        .llama_cpp => .{
            .supports_turboquant = true,
            .supports_speculative_decoding = false, // Not yet implemented
            .supports_prompt_caching = true,
            .max_context_length = 131072, // GPT-OSS supports 128K
            .preferred_quantization = "Q4_K_M",
        },
        .auto => .{
            .supports_turboquant = true,
            .supports_speculative_decoding = false,
            .supports_prompt_caching = true,
            .max_context_length = 8192,
            .preferred_quantization = "4bit",
        },
    };
}

/// Log backend information
pub fn logBackendInfo(backend_type: BackendPreference, arch: registry.ModelArchitecture) void {
    const caps = getCapabilities(arch, backend_type);

    std.log.info("Backend: {s}", .{@tagName(backend_type)});
    std.log.info("Architecture: {s}", .{@tagName(arch)});
    std.log.info("TurboQuant: {}", .{caps.supports_turboquant});
    std.log.info("Speculative Decoding: {}", .{caps.supports_speculative_decoding});
    std.log.info("Prompt Caching: {}", .{caps.supports_prompt_caching});
    std.log.info("Max Context: {d}", .{caps.max_context_length});
    std.log.info("Preferred Quantization: {s}", .{caps.preferred_quantization});
}

// Test factory functionality
test "factory detects correct backend for architectures" {
    const testing = std.testing;

    // Qwen should select MLX
    try testing.expectEqual(
        BackendPreference.mlx,
        detectBackendType(.qwen),
    );

    // DeepSeek MoE should select llama.cpp
    try testing.expectEqual(
        BackendPreference.llama_cpp,
        detectBackendType(.deepseek_v2_moe),
    );

    // GPT-OSS should select llama.cpp
    try testing.expectEqual(
        BackendPreference.llama_cpp,
        detectBackendType(.gpt_oss),
    );

    // Llama should select llama.cpp
    try testing.expectEqual(
        BackendPreference.llama_cpp,
        detectBackendType(.llama),
    );

    // Phi should select MLX
    try testing.expectEqual(
        BackendPreference.mlx,
        detectBackendType(.phi),
    );

    // Unknown should return auto (needs manual selection)
    try testing.expectEqual(
        BackendPreference.auto,
        detectBackendType(.unknown),
    );
}

test "detectModelFormat identifies file types" {
    const testing = std.testing;

    try testing.expectEqual(
        ModelFormat.gguf,
        detectModelFormat("model.gguf"),
    );

    try testing.expectEqual(
        ModelFormat.gguf,
        detectModelFormat("deepseek-coder-v2-lite.Q4_K_M.gguf"),
    );

    try testing.expectEqual(
        ModelFormat.safetensors,
        detectModelFormat("model.safetensors"),
    );

    try testing.expectEqual(
        ModelFormat.safetensors,
        detectModelFormat("model-00001-of-00002.safetensors"),
    );

    try testing.expectEqual(
        ModelFormat.unknown,
        detectModelFormat("model.bin"),
    );
}

test "getCapabilities returns correct values" {
    const testing = std.testing;

    // MLX capabilities
    const mlx_caps = getCapabilities(.qwen, .mlx);
    try testing.expect(mlx_caps.supports_turboquant);
    try testing.expect(mlx_caps.supports_speculative_decoding);
    try testing.expectEqual(@as(u32, 32768), mlx_caps.max_context_length);

    // llama.cpp capabilities
    const llama_caps = getCapabilities(.deepseek_v2_moe, .llama_cpp);
    try testing.expect(llama_caps.supports_turboquant);
    try testing.expect(!llama_caps.supports_speculative_decoding);
    try testing.expectEqual(@as(u32, 131072), llama_caps.max_context_length);
}
