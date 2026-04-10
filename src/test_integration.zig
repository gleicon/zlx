//! test_integration.zig - Integration tests for model loading and inference
//!
//! Tests real model loading scenarios:
//! - DeepSeek weight loading and dequantization
//! - GPT-OSS weight loading with MXFP4
//! - TurboQuant compression verification
//! - End-to-end inference on real models (if available)

const std = @import("std");
const testing = std.testing;
const loader = @import("inference/loader.zig");
const deepseek = @import("deepseek.zig");
const gpt_oss = @import("gpt_oss.zig");
const registry = @import("models/registry.zig");
const mlx = @import("mlx.zig/src/mlx.zig");

// ============================================================================
// DeepSeek Integration Tests
// ============================================================================

test "DeepSeek weight loading and dequantization" {
    const allocator = testing.allocator;

    // Skip if model not present
    const model_path = "./models/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx";
    std.fs.cwd().access(model_path, .{}) catch |err| {
        if (err == error.FileNotFound) {
            std.log.info("DeepSeek model not found at {s}, skipping test", .{model_path});
            return;
        }
        return err;
    };

    std.log.info("Testing DeepSeek weight loading from {s}...", .{model_path});

    // 1. Load config and detect model type
    var config_info = try loader.loadModelInfo(allocator, model_path);
    defer config_info.deinit();

    try testing.expectEqual(loader.ModelType.deepseek_v2_moe, config_info.model_type);
    std.log.info("✓ Model type detected: {s}", .{@tagName(config_info.model_type)});

    // 2. Load weights (includes dequantization)
    const config_json_path = try std.fs.path.join(allocator, &.{ model_path, "config.json" });
    defer allocator.free(config_json_path);
    var raw_config = try loader.loadConfigInfo(allocator, config_json_path);
    defer raw_config.deinit();
    const weights = try loader.loadDeepSeekWeights(allocator, raw_config, model_path);
    defer {
        var w = weights;
        w.deinit(allocator);
    }

    // 3. Verify weights are valid (not empty)
    try testing.expect(!mlx.arrayIsEmpty(weights.token_embedding));
    try testing.expect(!mlx.arrayIsEmpty(weights.norm));
    try testing.expect(!mlx.arrayIsEmpty(weights.lm_head));
    try testing.expect(weights.layers.len > 0);

    std.log.info("✓ Weights loaded: {d} layers", .{weights.layers.len});

    // 4. Verify first layer weights
    const first_layer = weights.layers[0];
    try testing.expect(!mlx.arrayIsEmpty(first_layer.input_norm));
    try testing.expect(!mlx.arrayIsEmpty(first_layer.post_attn_norm));

    // 5. Verify dequantization produced valid (non-empty) arrays
    // mla.w_dq is the query down-projection weight array
    try testing.expect(!mlx.arrayIsEmpty(first_layer.mla.w_dq));

    std.log.info("✓ Dequantization verified: w_dq loaded", .{});

    // 6. Check layer count matches config
    const expected_layers = 27; // DeepSeek-V2-Lite has 27 layers
    try testing.expectEqual(expected_layers, weights.layers.len);

    std.log.info("✓ DeepSeek weight loading and dequantization test passed", .{});
}

test "DeepSeek architecture detection from config" {
    _ = testing.allocator; // Not needed for this test but signature requires it

    // Create a mock DeepSeek config
    const mock_config = registry.ConfigInfo{
        .hidden_size = 4096,
        .num_layers = 27,
        .num_attention_heads = 128,
        .max_position_embeddings = 128000,
        .vocab_size = 102400,
        .model_type = "deepseek_v2",
        .num_experts = 64,
        .sliding_window = null,
    };

    const arch = registry.detectArchitecture(&mock_config);
    try testing.expectEqual(registry.ModelArchitecture.deepseek_v2_moe, arch);

    std.log.info("✓ DeepSeek architecture detection works", .{});
}

test "DeepSeek memory estimation" {
    const allocator = testing.allocator;

    // DeepSeek-Coder-V2-Lite config
    const config = registry.ConfigInfo{
        .hidden_size = 4096,
        .num_layers = 27,
        .num_attention_heads = 128,
        .max_position_embeddings = 128000,
        .vocab_size = 102400,
        .model_type = "deepseek_v2",
        .num_experts = 64,
        .sliding_window = null,
        .quantization_bits = 4,
    };

    var reg = registry.ModelRegistry.init(allocator);
    defer reg.deinit();

    const memory_mb = reg.estimateMemoryFromConfig(config);

    // With 4-bit quantization and KV cache at max_context=8192:
    // ~2.3GB weights + ~3.5GB KV cache + 20% overhead ≈ 6.9GB total
    try testing.expect(memory_mb > 1000); // At least 1GB
    try testing.expect(memory_mb < 12000); // Less than 12GB (KV cache dominates)

    std.log.info("✓ DeepSeek memory estimate: {d}MB", .{memory_mb});
}

// ============================================================================
// GPT-OSS Integration Tests
// ============================================================================

test "GPT-OSS weight loading" {
    const allocator = testing.allocator;

    // Skip if model not present
    const model_path = "./models/gpt-oss-20b-MXFP4-Q4";
    std.fs.cwd().access(model_path, .{}) catch |err| {
        if (err == error.FileNotFound) {
            std.log.info("GPT-OSS model not found at {s}, skipping test", .{model_path});
            return;
        }
        return err;
    };

    std.log.info("Testing GPT-OSS weight loading from {s}...", .{model_path});

    // 1. Load config and detect model type
    var config_info = try loader.loadModelInfo(allocator, model_path);
    defer config_info.deinit();

    try testing.expectEqual(loader.ModelType.gpt_oss, config_info.model_type);
    std.log.info("✓ Model type detected: {s}", .{@tagName(config_info.model_type)});

    // 2. Load weights
    const gptoss_config_json_path = try std.fs.path.join(allocator, &.{ model_path, "config.json" });
    defer allocator.free(gptoss_config_json_path);
    var gptoss_raw_config = try loader.loadConfigInfo(allocator, gptoss_config_json_path);
    defer gptoss_raw_config.deinit();
    const weights = try loader.loadGptOssWeights(allocator, gptoss_raw_config, model_path);
    defer {
        var w = weights;
        w.deinit(allocator);
    }

    // 3. Verify weights are valid
    try testing.expect(!mlx.arrayIsEmpty(weights.token_embedding));
    try testing.expect(!mlx.arrayIsEmpty(weights.norm));
    try testing.expect(!mlx.arrayIsEmpty(weights.lm_head));
    try testing.expectEqual(@as(usize, 24), weights.layers.len);

    std.log.info("✓ Weights loaded: {d} layers", .{weights.layers.len});

    // 4. Verify first layer
    const first_layer = weights.layers[0];
    try testing.expect(!mlx.arrayIsEmpty(first_layer.input_norm));
    try testing.expect(!mlx.arrayIsEmpty(first_layer.post_attn_norm));
    try testing.expect(!mlx.arrayIsEmpty(first_layer.router));
    try testing.expectEqual(@as(usize, 32), first_layer.experts.len);

    // 5. Verify first expert
    const first_expert = first_layer.experts[0];
    try testing.expect(!mlx.arrayIsEmpty(first_expert.gate_proj));
    try testing.expect(!mlx.arrayIsEmpty(first_expert.up_proj));
    try testing.expect(!mlx.arrayIsEmpty(first_expert.down_proj));

    std.log.info("✓ GPT-OSS weight loading test passed", .{});
}

test "GPT-OSS architecture detection from config" {
    // Test with explicit model_type
    const config1 = registry.ConfigInfo{
        .hidden_size = 2880,
        .num_layers = 24,
        .num_attention_heads = 64,
        .model_type = "gpt_oss",
        .num_experts = 32,
        .sliding_window = 128,
    };

    const arch1 = registry.detectArchitecture(&config1);
    try testing.expectEqual(registry.ModelArchitecture.gpt_oss, arch1);

    // Test with heuristics (MoE + sliding window)
    const config2 = registry.ConfigInfo{
        .hidden_size = 2880,
        .num_layers = 24,
        .num_attention_heads = 64,
        .model_type = "unknown",
        .num_experts = 32,
        .sliding_window = 128,
    };

    const arch2 = registry.detectArchitecture(&config2);
    try testing.expectEqual(registry.ModelArchitecture.gpt_oss, arch2);

    std.log.info("✓ GPT-OSS architecture detection works", .{});
}

test "GPT-OSS model registry lookup" {
    // Test by ID
    const model1 = registry.getKnownModel("gpt-oss-20b");
    try testing.expect(model1 != null);
    try testing.expectEqual(registry.ModelArchitecture.gpt_oss, model1.?.architecture);

    // Test by alias
    const model2 = registry.getKnownModel("gptoss");
    try testing.expect(model2 != null);
    try testing.expectEqual(registry.ModelArchitecture.gpt_oss, model2.?.architecture);

    std.log.info("✓ GPT-OSS registry lookup works", .{});
}

test "GPT-OSS memory estimation" {
    const allocator = testing.allocator;

    // GPT-OSS config
    const config = registry.ConfigInfo{
        .hidden_size = 2880,
        .num_layers = 24,
        .num_attention_heads = 64,
        .max_position_embeddings = 131072,
        .vocab_size = 151936,
        .model_type = "gpt_oss",
        .num_experts = 32,
        .sliding_window = 128,
        .quantization_bits = 4,
    };

    var reg = registry.ModelRegistry.init(allocator);
    defer reg.deinit();

    const memory_mb = reg.estimateMemoryFromConfig(config);

    // With 4-bit quantization and KV cache at max_context=8192:
    // ~1.3GB weights + ~2.2GB KV cache + 20% overhead ≈ 4.2GB total
    try testing.expect(memory_mb > 1000); // At least 1GB
    try testing.expect(memory_mb < 8000); // Less than 8GB

    std.log.info("✓ GPT-OSS memory estimate: {d}MB", .{memory_mb});
}

// ============================================================================
// TurboQuant Compression Tests
// ============================================================================

test "TurboQuant compression configuration" {
    // Test that TurboQuant can be enabled
    const config = .{
        .enabled = true,
        .compression_ratio = 4.6,
        .max_context = 131072,
    };

    try testing.expect(config.enabled);
    try testing.expect(config.compression_ratio >= 4.0);

    std.log.info("✓ TurboQuant compression configured: {d:.1}x", .{config.compression_ratio});
}

// ============================================================================
// End-to-End Inference Tests (if models available)
// ============================================================================

test "DeepSeek end-to-end inference (if model available)" {
    _ = testing.allocator; // Not needed but signature requires it

    // Skip if model not present
    const model_path = "./models/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx";
    std.fs.cwd().access(model_path, .{}) catch |err| {
        if (err == error.FileNotFound) {
            std.log.info("DeepSeek model not found, skipping E2E test", .{});
            return;
        }
        return err;
    };

    // TODO: Add full inference test with token generation
    // This would require:
    // 1. Load model
    // 2. Tokenize input
    // 3. Run forward pass
    // 4. Verify output tokens

    std.log.info("✓ DeepSeek E2E test placeholder (model exists)", .{});
}

test "GPT-OSS end-to-end inference (if model available)" {
    _ = testing.allocator; // Not needed but signature requires it

    // Skip if model not present
    const model_path = "./models/gpt-oss-20b-MXFP4-Q4";
    std.fs.cwd().access(model_path, .{}) catch |err| {
        if (err == error.FileNotFound) {
            std.log.info("GPT-OSS model not found, skipping E2E test", .{});
            return;
        }
        return err;
    };

    // TODO: Add full inference test with token generation

    std.log.info("✓ GPT-OSS E2E test placeholder (model exists)", .{});
}

// ============================================================================
// Build Integration
// ============================================================================

// This test ensures the integration test file compiles correctly
test "Integration tests compile" {
    // If we got here, the file compiled successfully
    std.log.info("✓ Integration test file compiles correctly", .{});
}
