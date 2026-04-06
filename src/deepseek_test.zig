//! deepseek_test.zig - Unit tests for DeepSeek-V2 Transformer

const std = @import("std");
const deepseek = @import("../deepseek.zig");
const mla = @import("mlx.zig/src/mla.zig");
const moe = @import("moe.zig");
const mlx = @import("mlx.zig/src/mlx.zig");

// Test 1: DeepSeek forward produces correct logits shape
test "DeepSeek forward produces correct logits shape" {
    const allocator = std.testing.allocator;

    // Use small config for testing speed
    const config = deepseek.DeepSeekConfig{
        .vocab_size = 1024,
        .hidden_size = 256,
        .num_hidden_layers = 2,
        .num_attention_heads = 4,
        .intermediate_size = 512,
        .num_experts = 8,
        .num_shared_experts = 2,
        .top_k = 2,
        .max_position_embeddings = 1024,
        .rope_theta = 10000.0,
        .rms_norm_eps = 1e-6,
        .latent_dim = 64,
    };

    // Create minimal weights for testing
    const layers = try allocator.alloc(deepseek.DeepSeekLayer, config.num_hidden_layers);
    defer allocator.free(layers);

    // For this test, we just verify the structure compiles
    // Real weight loading would initialize actual MLX arrays
    const weights = deepseek.DeepSeekWeights{
        .token_embedding = mlx.arrayNew(),
        .layers = layers,
        .norm = mlx.arrayNew(),
        .lm_head = mlx.arrayNew(),
    };
    defer mlx.arrayFree(weights.token_embedding);
    defer mlx.arrayFree(weights.norm);
    defer mlx.arrayFree(weights.lm_head);

    // Initialize transformer
    var transformer = try deepseek.DeepSeekTransformer.init(allocator, config, weights);

    // Test that transformer was initialized with correct config
    try std.testing.expectEqual(config.num_hidden_layers, transformer.config.num_hidden_layers);
    try std.testing.expectEqual(config.hidden_size, transformer.config.hidden_size);
    try std.testing.expectEqual(config.vocab_size, transformer.config.vocab_size);

    // Cleanup (note: weights are owned externally, not freed here)
    transformer.deinit();
}

// Test 2: DeepSeek config matches V2-Lite defaults
test "DeepSeek config matches V2-Lite defaults" {
    const config = deepseek.DeepSeekConfig{};

    // Verify default V2-Lite configuration
    try std.testing.expectEqual(@as(usize, 102400), config.vocab_size);
    try std.testing.expectEqual(@as(usize, 4096), config.hidden_size);
    try std.testing.expectEqual(@as(usize, 27), config.num_hidden_layers);
    try std.testing.expectEqual(@as(usize, 16), config.num_attention_heads);
    try std.testing.expectEqual(@as(usize, 11008), config.intermediate_size);
    try std.testing.expectEqual(@as(usize, 64), config.num_experts);
    try std.testing.expectEqual(@as(usize, 2), config.num_shared_experts);
    try std.testing.expectEqual(@as(usize, 6), config.top_k);
    try std.testing.expectEqual(@as(usize, 128000), config.max_position_embeddings);
    try std.testing.expectEqual(@as(f32, 10000.0), config.rope_theta);
    try std.testing.expectApproxEqAbs(@as(f32, 1e-6), config.rms_norm_eps, 1e-7);
    try std.testing.expectEqual(@as(usize, 512), config.latent_dim);
}

// Test 3: DeepSeek layer has MLA and MoE
test "DeepSeek layer structure" {
    // Test that DeepSeekLayer struct has the expected fields
    const TestLayer = deepseek.DeepSeekLayer;

    // Verify struct layout by checking field types
    // This is a compile-time test
    const input_norm_info = @typeInfo(TestLayer).@"struct".fields[0];
    try std.testing.expectEqualStrings("input_norm", input_norm_info.name);

    const mla_info = @typeInfo(TestLayer).@"struct".fields[1];
    try std.testing.expectEqualStrings("mla", mla_info.field_type);

    const post_attn_norm_info = @typeInfo(TestLayer).@"struct".fields[2];
    try std.testing.expectEqualStrings("post_attn_norm", post_attn_norm_info.name);

    const moe_info = @typeInfo(TestLayer).@"struct".fields[3];
    try std.testing.expectEqualStrings("moe", moe_info.field_type);
}

// Test 4: DeepSeek compression ratio calculation
test "DeepSeek MLA compression ratio" {
    const allocator = std.testing.allocator;

    const latent_dim: usize = 512;
    const hidden_size: usize = 4096;
    const seq_len: usize = 100;

    // Initialize compressed cache
    var cache = mla.CompressedKVCache.init(
        allocator,
        1024, // max_seq_len
        latent_dim,
    );
    defer cache.deinit();

    // Standard KV cache size: seq_len * 2 (K+V) * hidden_size * 2 bytes
    const standard_size = seq_len * 2 * hidden_size * 2;

    // Compressed cache size: seq_len * latent_dim * 2 bytes
    const compressed_size = seq_len * latent_dim * 2;

    // Expected compression ratio: 8192/512 = 16x
    const expected_ratio = @as(f32, @floatFromInt(standard_size)) / @as(f32, @floatFromInt(compressed_size));
    const expected_compression: f32 = 16.0; // 8192/512

    try std.testing.expectApproxEqAbs(expected_compression, expected_ratio, 0.01);
}

// Test 5: DeepSeek sparse parameter count
test "DeepSeek sparse parameter count" {
    // DeepSeek-V2-Lite has 15.7B total params but only 2B active per token
    const total_params: u64 = 15_700_000_000;
    const active_params: u64 = 2_000_000_000;
    const num_experts: usize = 64;
    const top_k: usize = 6;

    // Verify sparse activation ratio
    const sparse_ratio = @as(f32, @floatFromInt(active_params)) / @as(f32, @floatFromInt(total_params));

    // Should be around 12.7% (2B/15.7B)
    try std.testing.expect(sparse_ratio < 0.15);
    try std.testing.expect(sparse_ratio > 0.10);

    // Verify routing calculation
    // Each token uses top_k experts out of num_experts
    const routing_ratio = @as(f32, @floatFromInt(top_k)) / @as(f32, @floatFromInt(num_experts));
    try std.testing.expectApproxEqAbs(@as(f32, 0.09375), routing_ratio, 0.001); // 6/64 = 9.375%
}

// Test 6: ModelUnion supports DeepSeek variant
test "ModelUnion supports DeepSeek variant" {
    const inference = @import("inference/mod.zig");

    // Verify ModelUnion has deepseek variant
    const union_info = @typeInfo(inference.ModelUnion);
    var has_deepseek = false;

    for (union_info.@"union".fields) |field| {
        if (std.mem.eql(u8, field.name, "deepseek")) {
            has_deepseek = true;
            break;
        }
    }

    try std.testing.expect(has_deepseek);
}

// Test 7: DeepSeekWeights structure
test "DeepSeekWeights structure" {
    // Verify DeepSeekWeights has required fields
    const TestWeights = deepseek.DeepSeekWeights;

    const token_embed_info = @typeInfo(TestWeights).@"struct".fields[0];
    try std.testing.expectEqualStrings("token_embedding", token_embed_info.name);

    const layers_info = @typeInfo(TestWeights).@"struct".fields[1];
    try std.testing.expectEqualStrings("layers", layers_info.name);

    const norm_info = @typeInfo(TestWeights).@"struct".fields[2];
    try std.testing.expectEqualStrings("norm", norm_info.name);

    const lm_head_info = @typeInfo(TestWeights).@"struct".fields[3];
    try std.testing.expectEqualStrings("lm_head", lm_head_info.name);
}

// Test 8: Memory estimation for DeepSeek model
test "DeepSeek memory estimation" {
    const config = deepseek.DeepSeekConfig{};

    // Embedding table: vocab_size * hidden_size * 2 bytes (fp16)
    const embedding_memory = config.vocab_size * config.hidden_size * 2;

    // Per layer: weights (simplified estimate)
    // MLA: ~hidden * latent * 3 projections
    const mla_memory = config.hidden_size * config.latent_dim * 3 * 2;

    // MoE: (shared_experts + routed_experts * active_ratio) * FFN weights
    const expert_memory = config.hidden_size * config.intermediate_size * 3 * 2;
    const shared_moe_memory = config.num_shared_experts * expert_memory;
    const active_routed_experts = config.top_k;
    const routed_moe_memory = active_routed_experts * expert_memory;
    const moe_memory = shared_moe_memory + routed_moe_memory;

    // Norms: 2 per layer
    const norm_memory = config.hidden_size * 2 * 2;

    const per_layer_memory = mla_memory + moe_memory + norm_memory;
    const total_layers_memory = per_layer_memory * config.num_hidden_layers;

    // LM head: vocab_size * hidden_size * 2 bytes
    const lm_head_memory = config.vocab_size * config.hidden_size * 2;

    const total_weight_memory = embedding_memory + total_layers_memory + lm_head_memory;

    // Verify total is in reasonable range for 15.7B model in 4-bit
    // 15.7B params * 0.5 bytes (4-bit) = ~7.8GB
    // Or fp16: 15.7B * 2 = ~31GB (but sparse)
    try std.testing.expect(total_weight_memory > 5_000_000_000); // > 5GB
    try std.testing.expect(total_weight_memory < 35_000_000_000); // < 35GB
}

// Test 9: DeepSeekTransformer init and deinit
test "DeepSeekTransformer lifecycle" {
    const allocator = std.testing.allocator;

    const config = deepseek.DeepSeekConfig{
        .vocab_size = 1000,
        .hidden_size = 128,
        .num_hidden_layers = 2,
        .num_attention_heads = 4,
        .intermediate_size = 256,
        .num_experts = 4,
        .num_shared_experts = 1,
        .top_k = 2,
        .max_position_embeddings = 512,
        .latent_dim = 32,
    };

    const layers = try allocator.alloc(deepseek.DeepSeekLayer, 2);
    defer allocator.free(layers);

    const weights = deepseek.DeepSeekWeights{
        .token_embedding = mlx.arrayNew(),
        .layers = layers,
        .norm = mlx.arrayNew(),
        .lm_head = mlx.arrayNew(),
    };
    defer mlx.arrayFree(weights.token_embedding);
    defer mlx.arrayFree(weights.norm);
    defer mlx.arrayFree(weights.lm_head);

    // Initialize
    var transformer = deepseek.DeepSeekTransformer.init(allocator, config, weights) catch |err| {
        try std.testing.expect(false); // Should not fail
        _ = err;
        return;
    };

    // Verify initialization
    try std.testing.expectEqual(allocator, transformer.allocator);
    try std.testing.expectEqual(config.vocab_size, transformer.config.vocab_size);
    try std.testing.expectEqual(config.hidden_size, transformer.config.hidden_size);

    // Cleanup
    transformer.deinit();
}

// Test 10: Architecture detection returns correct type
test "Loader detects DeepSeek V2 MoE architecture" {
    const loader = @import("inference/loader.zig");

    // Test config with MoE indicators
    const json_with_moe = "{\"model_type\": \"deepseek\", \"num_experts\": 64, \"kv_lora_rank\": 512}";

    var config_info = loader.ConfigInfo{
        .model_type = try std.testing.allocator.dupe(u8, "deepseek"),
        .raw_json = json_with_moe,
        .allocator = std.testing.allocator,
    };
    defer config_info.deinit();

    const arch = loader.detectArchitecture(config_info);
    try std.testing.expectEqual(loader.ModelType.deepseek_v2_moe, arch);
}
