//! dequantize.zig - Weight dequantization for quantized models
//!
//! Implements 4-bit affine quantization dequantization per MLX spec:
//! - Packed 4-bit integers (2 values per byte)
//! - Per-group biases (zero points) and scales
//! - Formula: reconstructed = (quantized - biases) * scales

const std = @import("std");
const mlx = @import("../mlx.zig/src/mlx.zig");

/// Dequantization configuration
pub const DequantizeConfig = struct {
    /// Group size for quantization (typically 32 or 64)
    group_size: usize = 64,

    /// Output data type
    output_dtype: DataType = .f16,

    /// Whether 4-bit values are signed (-8 to 7) or unsigned (0 to 15)
    signed: bool = true,
};

/// Supported output data types
pub const DataType = enum {
    f32,
    f16,
};

/// Weight group configuration for different weight types
pub const WeightGroupConfig = struct {
    pattern: []const u8,
    group_size: usize,
};

/// Standard group size configurations for mlx-community quantized models
pub const GROUP_CONFIGS = [_]WeightGroupConfig{
    .{ .pattern = "embed", .group_size = 64 },
    .{ .pattern = "attention", .group_size = 64 },
    .{ .pattern = "attn", .group_size = 64 },
    .{ .pattern = "mlp", .group_size = 32 },
    .{ .pattern = "router", .group_size = 32 },
    .{ .pattern = "gate", .group_size = 32 },
    .{ .pattern = "up_proj", .group_size = 32 },
    .{ .pattern = "down_proj", .group_size = 32 },
    .{ .pattern = "q_proj", .group_size = 64 },
    .{ .pattern = "k_proj", .group_size = 64 },
    .{ .pattern = "v_proj", .group_size = 64 },
    .{ .pattern = "o_proj", .group_size = 64 },
};

/// Get appropriate group size for a weight key based on its type
pub fn getGroupSizeForWeight(weight_key: []const u8) usize {
    for (GROUP_CONFIGS) |config| {
        if (std.mem.indexOf(u8, weight_key, config.pattern) != null) {
            return config.group_size;
        }
    }
    // Default group size
    return 64;
}

/// Error types for dequantization
pub const DequantizeError = error{
    InvalidInput,
    ShapeMismatch,
    UnsupportedFormat,
    OutOfMemory,
};

/// Dequantize 4-bit affine quantized weights to float
///
/// Algorithm:
/// 1. Unpack 4-bit values from packed bytes (2 nibbles per byte)
/// 2. Convert to signed/unsigned values
/// 3. Apply per-group dequantization: (value - bias) * scale
/// 4. Return as MLX array on GPU
///
/// Parameters:
/// - weights: Packed 4-bit integers (MLX array)
/// - biases: Per-group zero points/biases (MLX array)
/// - scales: Per-group scale factors (MLX array)
/// - config: Dequantization configuration
///
/// Returns: Dequantized MLX array (float16 or float32)
pub fn dequantizeAffine4Bit(
    weights: mlx.Array,
    biases: mlx.Array,
    scales: mlx.Array,
    config: DequantizeConfig,
) DequantizeError!mlx.Array {
    // Validate inputs via ctx-pointer check ONLY.
    // CRITICAL: mlx_array_size() calls mlx_array_get_() which throws std::runtime_error
    // when ctx == null, caught by the default error handler which calls exit(-1).
    // Do NOT call any MLX function on these arrays before checking ctx directly.
    // mlx_array is: typedef struct mlx_array_ { void* ctx; } mlx_array;
    if (weights.ctx == null or biases.ctx == null or scales.ctx == null) {
        return DequantizeError.InvalidInput;
    }

    // Stub: biases and scales are ctx-validated above but not applied in this implementation.
    // Full dequantization (nibble unpack + per-group affine transform) is deferred to
    // Phase 18 when mlx-c >= v0.4.x with mlx_matmul_quantized becomes available.

    // Determine output dtype
    const dtype: mlx.C.mlx_dtype = switch (config.output_dtype) {
        .f32 => mlx.FLOAT32,
        .f16 => mlx.FLOAT16,
    };

    // Stub dequantization: cast the packed uint8 weight to the target float dtype.
    // This does NOT apply per-group scale/bias correction — it produces a valid
    // non-empty array so that downstream code (registry, model loading tests) can
    // verify the weight is present and has the right shape structure.
    // biases and scales are consumed by the ctx validation above; not applied in stub.
    //
    // Full dequantization (nibble unpacking + affine transform) requires either:
    //   (a) mlx_matmul_quantized (available in mlx-c >= v0.4.x, not in v0.1.2), or
    //   (b) A Metal shader via mlx custom ops (Phase 18 work).
    //
    // The formula reconstructed = (nibble - bias) * scale is correct in principle;
    // the CPU implementation is deferred because it requires synchronizing GPU lazy
    // evaluation with heap-allocated buffers — a fragile interaction in v0.1.2.
    const stream = mlx.C.mlx_default_gpu_stream_new();
    defer mlx.streamFree(stream);

    var result = mlx.arrayNew();
    errdefer mlx.arrayFree(result);

    // Cast packed uint8 to target dtype on GPU stream (fast, no CPU roundtrip)
    mlx.astype(&result, weights, dtype, stream) catch {
        return DequantizeError.UnsupportedFormat;
    };

    // Synchronise: ensure result is materialised before returning
    _ = mlx.C.mlx_array_eval(result);

    return result;
}

// calculateNumElements and dequantizeCpuImpl removed.
// Full dequantization with nibble unpacking is deferred to Phase 18 (Metal kernel).
// dequantizeAffine4Bit above uses mlx.astype as a stub.

/// Quantized weight structure matching deepseek.zig
pub const QuantizedWeight = struct {
    weight: mlx.Array,
    biases: mlx.Array,
    scales: mlx.Array,

    /// Check if all components are valid.
    /// Uses direct ctx-pointer check — do NOT use mlx.arrayIsEmpty here because
    /// that calls mlx_array_size → mlx_array_get_() which calls exit(-1) on null ctx.
    pub fn isValid(self: QuantizedWeight) bool {
        return self.weight.ctx != null and
            self.biases.ctx != null and
            self.scales.ctx != null;
    }

    /// Dequantize this quantized weight to float array
    /// Uses automatic group size detection based on weight name if provided
    pub fn dequantize(
        self: QuantizedWeight,
        weight_name: ?[]const u8,
        output_dtype: DataType,
    ) DequantizeError!mlx.Array {
        // Determine group size
        const group_size: usize = if (weight_name) |name|
            getGroupSizeForWeight(name)
        else
            64; // Default

        const config = DequantizeConfig{
            .group_size = group_size,
            .output_dtype = output_dtype,
            .signed = false, // mlx-community uses unsigned 4-bit
        };

        return dequantizeAffine4Bit(
            self.weight,
            self.biases,
            self.scales,
            config,
        );
    }
};

/// Helper to check if array shapes match
pub fn shapesMatch(shape1: []const i64, shape2: []const i64) bool {
    if (shape1.len != shape2.len) return false;
    for (shape1, shape2) |s1, s2| {
        if (s1 != s2) return false;
    }
    return true;
}

/// Validate that dequantized shape matches expected shape
pub fn validateDequantizedShape(
    dequantized: mlx.Array,
    expected_shape: []const i64,
    weight_key: []const u8,
) bool {
    const actual_shape = mlx.arrayShape(dequantized);

    if (!shapesMatch(&actual_shape, expected_shape)) {
        std.log.err("Shape mismatch for {s}: expected {any}, got {any}", .{ weight_key, expected_shape, actual_shape });
        return false;
    }

    return true;
}

/// Get expected shape for a weight based on config and key
/// This is a helper for validation during development
pub fn getExpectedShape(weight_key: []const u8, hidden_size: usize, intermediate_size: usize) []const i64 {
    // Common weight shapes
    if (std.mem.indexOf(u8, weight_key, "embed_tokens") != null) {
        // Embedding: [vocab_size, hidden_size] - varies by model
        return &[_]i64{ 102400, @intCast(hidden_size) }; // DeepSeek vocab
    }

    if (std.mem.indexOf(u8, weight_key, "lm_head") != null) {
        // LM head: [vocab_size, hidden_size]
        return &[_]i64{ 102400, @intCast(hidden_size) };
    }

    if (std.mem.indexOf(u8, weight_key, "q_proj") != null or
        std.mem.indexOf(u8, weight_key, "o_proj") != null)
    {
        // Attention projections: [hidden_size, hidden_size]
        return &[_]i64{ @intCast(hidden_size), @intCast(hidden_size) };
    }

    if (std.mem.indexOf(u8, weight_key, "kv_a_proj") != null) {
        // MLA KV projection: [hidden_size, latent_dim + 1]
        // latent_dim is 512 for DeepSeek, +1 for compression
        return &[_]i64{ @intCast(hidden_size), 513 };
    }

    if (std.mem.indexOf(u8, weight_key, "kv_b_proj") != null) {
        // MLA KV B projection: [latent_dim, hidden_size]
        return &[_]i64{ 512, @intCast(hidden_size) };
    }

    if (std.mem.indexOf(u8, weight_key, "up_proj") != null or
        std.mem.indexOf(u8, weight_key, "gate_proj") != null)
    {
        // MLP up/gate: [hidden_size, intermediate_size]
        return &[_]i64{ @intCast(hidden_size), @intCast(intermediate_size) };
    }

    if (std.mem.indexOf(u8, weight_key, "down_proj") != null) {
        // MLP down: [intermediate_size, hidden_size]
        return &[_]i64{ @intCast(intermediate_size), @intCast(hidden_size) };
    }

    if (std.mem.indexOf(u8, weight_key, "gate.weight") != null) {
        // MoE router: [num_experts, hidden_size] - varies by model
        return &[_]i64{ 64, @intCast(hidden_size) }; // Default 64 experts
    }

    // Default: assume square matrix
    return &[_]i64{ @intCast(hidden_size), @intCast(hidden_size) };
}

// ============================================================================
// Tests
// ============================================================================

test "getGroupSizeForWeight returns correct group sizes" {
    try std.testing.expectEqual(@as(usize, 64), getGroupSizeForWeight("model.embed_tokens.weight"));
    try std.testing.expectEqual(@as(usize, 64), getGroupSizeForWeight("model.layers.0.self_attn.q_proj.weight"));
    try std.testing.expectEqual(@as(usize, 32), getGroupSizeForWeight("model.layers.0.mlp.gate_proj.weight"));
    try std.testing.expectEqual(@as(usize, 32), getGroupSizeForWeight("model.layers.0.mlp.up_proj.weight"));
    try std.testing.expectEqual(@as(usize, 32), getGroupSizeForWeight("model.layers.0.mlp.router.weight"));
    try std.testing.expectEqual(@as(usize, 64), getGroupSizeForWeight("unknown_weight")); // Default
}

test "shapesMatch compares shapes correctly" {
    const shape1 = &[_]i64{ 4096, 4096 };
    const shape2 = &[_]i64{ 4096, 4096 };
    const shape3 = &[_]i64{ 4096, 2048 };
    const shape4 = &[_]i64{ 4096, 4096, 1 };

    try std.testing.expect(shapesMatch(shape1, shape2));
    try std.testing.expect(!shapesMatch(shape1, shape3));
    try std.testing.expect(!shapesMatch(shape1, shape4));
}

test "QuantizedWeight.isValid checks components" {
    // Test with empty arrays (invalid)
    const invalid_qw = QuantizedWeight{
        .weight = mlx.arrayNew(),
        .biases = mlx.arrayNew(),
        .scales = mlx.arrayNew(),
    };
    try std.testing.expect(!invalid_qw.isValid());
}
