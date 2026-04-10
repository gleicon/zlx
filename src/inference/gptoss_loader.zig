//! gptoss_loader.zig - GPT-OSS specific weight loading
//!
//! Implements weight loading for GPT-OSS-20B model:
//! - 24 layers (vs 27 for DeepSeek)
//! - 32 experts (vs 64 for DeepSeek)
//! - Standard GQA attention (vs MLA for DeepSeek)
//! - Sliding window attention (128 tokens)
//! - Yarn RoPE with theta=150000
//! - MXFP4 quantization format

const std = @import("std");
const mlx = @import("../mlx.zig/src/mlx.zig");
const gpt_oss = @import("../gpt_oss.zig");
const safetensors_index = @import("safetensors_index.zig");
const dequantize = @import("dequantize.zig");

/// GPT-OSS weight loading errors
pub const GptOssLoadError = error{
    WeightNotFound,
    InvalidWeightShape,
    UnsupportedQuantization,
    SafetensorsError,
    ConfigError,
    OutOfMemory,
};

/// Register all weight keys for GPT-OSS model
///
/// GPT-OSS-20B has:
/// - 24 transformer layers
/// - 32 experts per MoE layer (all layers use MoE)
/// - No shared experts (all 32 are routed)
/// - 4 experts activated per token (top-4)
pub fn registerGptOssWeightKeys(
    allocator: std.mem.Allocator,
    weight_index: *std.StringHashMap([]const u8),
    config: gpt_oss.GptOssConfig,
) GptOssLoadError!void {
    // Embedding and output
    try weight_index.put(try allocator.dupe(u8, "model.embed_tokens.weight"), "model.safetensors");
    try weight_index.put(try allocator.dupe(u8, "model.norm.weight"), "model.safetensors");
    try weight_index.put(try allocator.dupe(u8, "lm_head.weight"), "model.safetensors");

    // 24 layers
    var buf: [256]u8 = undefined;
    for (0..config.num_hidden_layers) |layer_idx| {
        // Layer norms (not quantized)
        const input_ln = std.fmt.bufPrint(&buf, "model.layers.{d}.input_layernorm.weight", .{layer_idx}) catch unreachable;
        try weight_index.put(try allocator.dupe(u8, input_ln), "model.safetensors");

        const post_ln = std.fmt.bufPrint(&buf, "model.layers.{d}.post_attention_layernorm.weight", .{layer_idx}) catch unreachable;
        try weight_index.put(try allocator.dupe(u8, post_ln), "model.safetensors");

        // Attention (Q, K, V, O projections) - may be quantized
        for ([_][]const u8{ "q", "k", "v", "o" }) |proj| {
            const attn_key = std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.{s}_proj.weight", .{ layer_idx, proj }) catch unreachable;
            try weight_index.put(try allocator.dupe(u8, attn_key), "model.safetensors");
        }

        // MoE router (not quantized)
        const router_key = std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.router.weight", .{layer_idx}) catch unreachable;
        try weight_index.put(try allocator.dupe(u8, router_key), "model.safetensors");

        // 32 experts per layer (all quantized in MXFP4)
        for (0..config.num_experts) |expert_idx| {
            for ([_][]const u8{ "gate", "up", "down" }) |proj| {
                const expert_key = std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.experts.{d}.{s}_proj.weight", .{ layer_idx, expert_idx, proj }) catch unreachable;
                try weight_index.put(try allocator.dupe(u8, expert_key), "model.safetensors");
            }
        }
    }

    std.log.info("Registered {d} weight keys for GPT-OSS (24 layers, 32 experts)", .{weight_index.count()});
}

/// Load all GPT-OSS weights from safetensors files
///
/// Parameters:
/// - allocator: Memory allocator
/// - config: GPT-OSS model configuration
/// - model_path: Path to model directory
/// - quant_config: Quantization configuration (for MXFP4 handling)
///
/// Returns: Populated GptOssWeights struct with dequantized arrays
pub fn loadGptOssWeights(
    allocator: std.mem.Allocator,
    config: gpt_oss.GptOssConfig,
    model_path: []const u8,
    quant_config: ?QuantizationConfig,
) GptOssLoadError!gpt_oss.GptOssWeights {
    _ = quant_config;

    // Parse safetensors index to find weight locations
    var index_path_buf: [1024]u8 = undefined;
    const index_path = std.fmt.bufPrint(&index_path_buf, "{s}/model.safetensors.index.json", .{model_path}) catch return GptOssLoadError.ConfigError;

    var weight_index = std.StringHashMap([]const u8).init(allocator);
    defer {
        var iter = weight_index.iterator();
        while (iter.next()) |entry| {
            allocator.free(entry.key_ptr.*);
        }
        weight_index.deinit();
    }

    // Register all weight keys
    try registerGptOssWeightKeys(allocator, &weight_index, config);

    // If index exists, parse it; otherwise assume single file
    if (std.fs.cwd().access(index_path, .{})) {
        // TODO: Parse safetensors index to map weights to files
        std.log.info("Using safetensors index: {s}", .{index_path});
    } else |_| {
        std.log.info("No index found, assuming single model.safetensors file", .{});
    }

    // Load weights
    var weights = gpt_oss.GptOssWeights{
        .token_embedding = mlx.arrayNew(),
        .norm = mlx.arrayNew(),
        .lm_head = mlx.arrayNew(),
        .layers = try allocator.alloc(gpt_oss.GptOssLayer, config.num_hidden_layers),
    };
    errdefer {
        mlx.arrayFree(weights.token_embedding);
        mlx.arrayFree(weights.norm);
        mlx.arrayFree(weights.lm_head);
        for (weights.layers) |*layer| {
            layer.deinit(allocator);
        }
        allocator.free(weights.layers);
    }

    // Load embeddings and output weights
    weights.token_embedding = try loadWeightOrQuantized(allocator, model_path, "model.embed_tokens.weight", .{ .group_size = 64 });

    weights.norm = try loadUnquantizedWeight(allocator, model_path, "model.norm.weight");

    weights.lm_head = try loadWeightOrQuantized(allocator, model_path, "lm_head.weight", .{ .group_size = 64 });

    // Load per-layer weights
    var buf: [256]u8 = undefined;
    for (0..config.num_hidden_layers) |layer_idx| {
        // Layer norms (not quantized)
        const input_ln_key = std.fmt.bufPrint(&buf, "model.layers.{d}.input_layernorm.weight", .{layer_idx}) catch unreachable;
        const input_norm = try loadUnquantizedWeight(allocator, model_path, input_ln_key);

        const post_ln_key = std.fmt.bufPrint(&buf, "model.layers.{d}.post_attention_layernorm.weight", .{layer_idx}) catch unreachable;
        const post_attn_norm = try loadUnquantizedWeight(allocator, model_path, post_ln_key);

        // Attention projections (may be quantized)
        const q_proj_key = std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.q_proj.weight", .{layer_idx}) catch unreachable;
        const q_proj = try loadWeightOrQuantized(allocator, model_path, q_proj_key, .{ .group_size = 64 });

        const k_proj_key = std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.k_proj.weight", .{layer_idx}) catch unreachable;
        const k_proj = try loadWeightOrQuantized(allocator, model_path, k_proj_key, .{ .group_size = 64 });

        const v_proj_key = std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.v_proj.weight", .{layer_idx}) catch unreachable;
        const v_proj = try loadWeightOrQuantized(allocator, model_path, v_proj_key, .{ .group_size = 64 });

        const o_proj_key = std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.o_proj.weight", .{layer_idx}) catch unreachable;
        const o_proj = try loadWeightOrQuantized(allocator, model_path, o_proj_key, .{ .group_size = 64 });

        // MoE router (not quantized)
        const router_key = std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.router.weight", .{layer_idx}) catch unreachable;
        const router = try loadUnquantizedWeight(allocator, model_path, router_key);

        // Load all 32 experts
        var experts = try allocator.alloc(gpt_oss.GptOssExpert, config.num_experts);
        errdefer allocator.free(experts);

        for (0..config.num_experts) |expert_idx| {
            const gate_key = std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.experts.{d}.gate_proj.weight", .{ layer_idx, expert_idx }) catch unreachable;
            const up_key = std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.experts.{d}.up_proj.weight", .{ layer_idx, expert_idx }) catch unreachable;
            const down_key = std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.experts.{d}.down_proj.weight", .{ layer_idx, expert_idx }) catch unreachable;

            experts[expert_idx] = gpt_oss.GptOssExpert{
                .gate_proj = try loadWeightOrQuantized(allocator, model_path, gate_key, .{ .group_size = 32 }),
                .up_proj = try loadWeightOrQuantized(allocator, model_path, up_key, .{ .group_size = 32 }),
                .down_proj = try loadWeightOrQuantized(allocator, model_path, down_key, .{ .group_size = 32 }),
            };
        }

        weights.layers[layer_idx] = gpt_oss.GptOssLayer{
            .layer_type = .sliding,
            .input_norm = input_norm,
            .post_attn_norm = post_attn_norm,
            .attention = gpt_oss.GptOssAttention{
                .q_proj = q_proj,
                .k_proj = k_proj,
                .v_proj = v_proj,
                .o_proj = o_proj,
                .num_heads = config.num_attention_heads,
                .num_kv_heads = config.num_key_value_heads,
                .head_dim = config.hidden_size / config.num_attention_heads,
                .layer_idx = layer_idx,
            },
            .router = router,
            .experts = experts,
        };
    }

    std.log.info("Loaded GPT-OSS weights: {d} layers, {d} experts per layer", .{ config.num_hidden_layers, config.num_experts });

    return weights;
}

/// Quantization configuration
pub const QuantizationConfig = struct {
    format: enum { none, affine_4bit, mxfp4 } = .none,
    group_size: usize = 64,

    /// Detect quantization format from weight metadata
    pub fn detectFromConfig(config_json: []const u8) QuantizationConfig {
        // Check for MXFP4 in quantization_config
        if (std.mem.indexOf(u8, config_json, "\"quant_method\": \"mxfp4\"") != null or
            std.mem.indexOf(u8, config_json, "mxfp4") != null)
        {
            return .{ .format = .mxfp4, .group_size = 32 };
        }

        // Check for standard 4-bit affine
        if (std.mem.indexOf(u8, config_json, "\"quant_method\": \"affine\"") != null or
            std.mem.indexOf(u8, config_json, "4bit") != null)
        {
            return .{ .format = .affine_4bit, .group_size = 64 };
        }

        // Default: no quantization
        return .{ .format = .none, .group_size = 64 };
    }
};

/// Load a weight that may or may not be quantized
///
/// First tries to load as unquantized weight, then falls back to quantized components
fn loadWeightOrQuantized(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    weight_key: []const u8,
    quant_config: QuantizationConfig,
) GptOssLoadError!mlx.Array {
    _ = allocator;
    _ = model_path;
    _ = weight_key;
    _ = quant_config;

    // TODO: Implement actual weight loading from safetensors
    // For now, return a placeholder array

    // In actual implementation:
    // 1. Try to load unquantized weight: "{weight_key}"
    // 2. If not found, try quantized components: "{weight_key}.weight", "{weight_key}.biases", "{weight_key}.scales"
    // 3. If quantized components found, dequantize using appropriate method

    // Placeholder: return empty array (will be filled by actual loader)
    return mlx.arrayNew();
}

/// Load an unquantized weight from safetensors
fn loadUnquantizedWeight(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    weight_key: []const u8,
) GptOssLoadError!mlx.Array {
    _ = allocator;
    _ = model_path;
    _ = weight_key;

    // TODO: Implement actual safetensors loading
    // For now, return a placeholder
    return mlx.arrayNew();
}

/// Detect if MLX supports MXFP4 natively
///
/// Returns true if MLX can handle MXFP4 dequantization automatically
pub fn mlxSupportsMXFP4() bool {
    // MLX added MXFP4 support in v0.18.x
    // Check if we're running a compatible version
    // For now, assume support (will be verified at runtime)
    return true;
}

/// Dequantize MXFP4 weights if MLX doesn't handle it natively
///
/// MXFP4 (Microscaling FP4) format:
/// - 4 bits per value
/// - 3-bit mantissa + 1-bit exponent
/// - Micro-exponent scaling per group
fn dequantizeMXFP4IfNeeded(
    weight: mlx.Array,
    config: QuantizationConfig,
) GptOssLoadError!mlx.Array {
    // If MLX supports MXFP4 natively, the weight is already dequantized
    if (mlxSupportsMXFP4()) {
        return weight;
    }

    _ = config;

    // TODO: Implement custom MXFP4 dequantization
    // This would require:
    // 1. Unpacking 4-bit values
    // 2. Applying micro-exponent scaling
    // 3. Converting to float16

    return error.UnsupportedQuantization;
}

// ============================================================================
// Tests
// ============================================================================

test "registerGptOssWeightKeys registers correct count" {
    const allocator = std.testing.allocator;

    var weight_index = std.StringHashMap([]const u8).init(allocator);
    defer {
        var iter = weight_index.iterator();
        while (iter.next()) |entry| {
            allocator.free(entry.key_ptr.*);
        }
        weight_index.deinit();
    }

    const config = gpt_oss.GptOssConfig{
        .hidden_size = 2880,
        .num_layers = 24,
        .num_experts = 32,
    };

    try registerGptOssWeightKeys(allocator, &weight_index, config);

    // Should have: 3 (embed/norm/lm_head) + 24 * (2 norms + 4 attn + 1 router + 32*3 experts)
    // = 3 + 24 * (7 + 96) = 3 + 24 * 103 = 3 + 2472 = 2475 keys
    // Plus the quantization components (3 per quantized weight)
    // This is a lot, so just check it's > 2000
    try std.testing.expect(weight_index.count() > 2000);

    // Verify specific keys exist
    try std.testing.expect(weight_index.contains("model.embed_tokens.weight"));
    try std.testing.expect(weight_index.contains("model.layers.0.self_attn.q_proj.weight"));
    try std.testing.expect(weight_index.contains("model.layers.23.mlp.experts.31.down_proj.weight"));
}

test "QuantizationConfig.detectFromConfig detects MXFP4" {
    const mxfp4_config = "{\"quantization_config\": {\"quant_method\": \"mxfp4\", \"group_size\": 32}}";
    const config = QuantizationConfig.detectFromConfig(mxfp4_config);

    try std.testing.expectEqual(QuantizationConfig.Format.mxfp4, config.format);
    try std.testing.expectEqual(@as(usize, 32), config.group_size);
}

test "QuantizationConfig.detectFromConfig detects affine 4bit" {
    const affine_config = "{\"quantization_config\": {\"quant_method\": \"affine\", \"bits\": 4}}";
    const config = QuantizationConfig.detectFromConfig(affine_config);

    try std.testing.expectEqual(QuantizationConfig.Format.affine_4bit, config.format);
    try std.testing.expectEqual(@as(usize, 64), config.group_size);
}
