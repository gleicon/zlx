//! test_backend_integration.zig - Integration tests for backend abstraction layer
//!
//! Tests the unified backend interface with both MLX.zig and llama.cpp backends.
// factory: removed — backlog item, evaluate after core inference is stable
// Tests that depended on factory.zig (detectBackendType, detectModelFormat, getCapabilities)
// have been removed. The factory abstraction is deferred to a future phase.

const std = @import("std");
const backends = @import("backends/mod.zig");
const registry = @import("models/registry.zig");
const backend_generator = @import("inference/backend_generator.zig");

test "backend type conversions" {
    const testing = std.testing;

    // Verify BackendType enum values — mlx removed, llama_cpp and mlx_gptoss remain
    try testing.expect(@intFromEnum(backends.BackendType.llama_cpp) >= 0);
    try testing.expect(@intFromEnum(backends.BackendType.mlx_gptoss) >= 0);
}

test "registry has DeepSeek with llama.cpp preference" {
    const testing = std.testing;

    const model = registry.getKnownModel("deepseek-coder-v2-lite") orelse {
        try testing.expect(false); // Should find the model
        return;
    };

    try testing.expectEqual(registry.ModelArchitecture.deepseek_v2_moe, model.architecture);
    try testing.expect(model.preferred_backend != null);
    try testing.expectEqual(registry.BackendType.llama_cpp, model.preferred_backend.?);
    try testing.expect(model.download_urls != null);
    try testing.expect(model.gguf_filename != null);
}

test "registry has GPT-OSS with llama.cpp preference" {
    const testing = std.testing;

    const model = registry.getKnownModel("gpt-oss-20b") orelse {
        try testing.expect(false); // Should find the model
        return;
    };

    try testing.expectEqual(registry.ModelArchitecture.gpt_oss, model.architecture);
    try testing.expect(model.preferred_backend != null);
    try testing.expectEqual(registry.BackendType.llama_cpp, model.preferred_backend.?);
    try testing.expect(model.download_urls != null);
    try testing.expect(model.gguf_filename != null);
    try testing.expect(model.memory_required_gb > 10.0); // 11GB model
}

test "getDownloadInfo returns correct DeepSeek info" {
    const testing = std.testing;

    const info = registry.getDownloadInfo("deepseek-coder-v2-lite") orelse {
        try testing.expect(false); // Should have download info
        return;
    };

    try testing.expect(info.urls.len > 0);
    try testing.expectEqualStrings("deepseek-coder-v2-lite.Q4_K_M.gguf", info.filename);
    try testing.expect(info.expected_size_bytes > 4_000_000_000); // ~4.5GB
}

test "getDownloadInfo returns correct GPT-OSS info" {
    const testing = std.testing;

    const info = registry.getDownloadInfo("gpt-oss-20b") orelse {
        try testing.expect(false); // Should have download info
        return;
    };

    try testing.expect(info.urls.len > 0);
    try testing.expectEqualStrings("GPT-OSS-20B-Q4_K_M.gguf", info.filename);
    try testing.expect(info.expected_size_bytes > 10_000_000_000); // ~11GB
}

test "GenerationParams defaults" {
    const testing = std.testing;

    const params = backends.GenerationParams{};
    try testing.expectApproxEqAbs(@as(f32, 0.7), params.temperature, 0.001);
    try testing.expectApproxEqAbs(@as(f32, 0.9), params.top_p, 0.001);
    try testing.expectEqual(@as(u32, 40), params.top_k);
    try testing.expectEqual(@as(u32, 1024), params.max_tokens);
    try testing.expectEqual(@as(u64, 0), params.seed);
    try testing.expectEqual(@as(f32, 0.0), params.presence_penalty);
    try testing.expectEqual(@as(f32, 0.0), params.frequency_penalty);
    try testing.expectEqual(@as(f32, 1.0), params.repetition_penalty);
}

test "CompressionParams struct" {
    const testing = std.testing;

    const params = backends.CompressionParams{
        .enabled = true,
        .bits = 4,
        .adaptive_layers = 4,
        .group_size = 64,
    };

    try testing.expect(params.enabled);
    try testing.expectEqual(@as(u4, 4), params.bits);
    try testing.expectEqual(@as(u8, 4), params.adaptive_layers);
    try testing.expectEqual(@as(u32, 64), params.group_size);
}

test "backend descriptions" {
    const testing = std.testing;

    const llama_desc = backends.getBackendDescription(.llama_cpp);
    try testing.expect(std.mem.indexOf(u8, llama_desc, "llama.cpp") != null);

    const gptoss_desc = backends.getBackendDescription(.mlx_gptoss);
    try testing.expect(std.mem.indexOf(u8, gptoss_desc, "MLX") != null);
}

test "backend feature support" {
    const testing = std.testing;

    // Both backends support turboquant
    try testing.expect(backends.supportsFeature(.llama_cpp, "turboquant"));
    try testing.expect(backends.supportsFeature(.mlx_gptoss, "turboquant"));

    // speculative-decoding: removed — re-evaluate as dedicated phase after core inference is stable
    try testing.expect(!backends.supportsFeature(.llama_cpp, "speculative_decoding"));
    try testing.expect(!backends.supportsFeature(.mlx_gptoss, "speculative_decoding"));

    // Both support prompt caching
    try testing.expect(backends.supportsFeature(.llama_cpp, "prompt_caching"));
    try testing.expect(backends.supportsFeature(.mlx_gptoss, "prompt_caching"));

    // Only llama.cpp supports GGUF
    try testing.expect(!backends.supportsFeature(.mlx_gptoss, "gguf"));
    try testing.expect(backends.supportsFeature(.llama_cpp, "gguf"));
}

test "BackendGenerator types compile" {
    // Verify BackendGenerator types are accessible
    _ = backend_generator.BackendGenerator;
    _ = backend_generator.createGeneratorForModel;
    _ = backend_generator.BackendGenerator.initWithAutoBackend;
    _ = backend_generator.BackendGenerator.initWithBackend;
    _ = backend_generator.BackendGenerator.deinit;
    _ = backend_generator.BackendGenerator.generate;
}
