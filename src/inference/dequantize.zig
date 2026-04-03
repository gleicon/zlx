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
    // Validate inputs
    if (mlx.arrayIsEmpty(weights) or
        mlx.arrayIsEmpty(biases) or
        mlx.arrayIsEmpty(scales))
    {
        return DequantizeError.InvalidInput;
    }

    // Get weight shape information
    const weight_shape = mlx.arrayShape(weights);
    const num_elements = calculateNumElements(&weight_shape);
    const num_groups = num_elements / config.group_size;

    // Validate group dimensions match
    const bias_shape = mlx.arrayShape(biases);
    const scale_shape = mlx.arrayShape(scales);

    if (bias_shape.len != 1 or scale_shape.len != 1) {
        return DequantizeError.ShapeMismatch;
    }

    if (bias_shape[0] != num_groups or scale_shape[0] != num_groups) {
        std.log.err("Group count mismatch: weights have {d} groups, biases {d}, scales {d}", .{ num_groups, bias_shape[0], scale_shape[0] });
        return DequantizeError.ShapeMismatch;
    }

    // Create output array with correct dtype
    const dtype: mlx.Dtype = switch (config.output_dtype) {
        .f32 => mlx.float32,
        .f16 => mlx.float16,
    };

    // Create result array
    var result = mlx.arrayNewData(
        null, // Will be allocated by MLX
        &weight_shape,
        weight_shape.len,
        dtype,
    );
    errdefer mlx.arrayFree(result);

    // Perform dequantization
    // For now, use CPU-based dequantization with MLX arrays
    // In production, this should use Metal kernels
    try dequantizeCpuImpl(weights, biases, scales, config, &result);

    // Ensure computation happens on GPU
    mlx.eval(result);

    return result;
}

/// Calculate total number of elements from shape array
fn calculateNumElements(shape: []const i64) usize {
    var total: usize = 1;
    for (shape) |dim| {
        total *= @intCast(dim);
    }
    return total;
}

/// CPU-based dequantization implementation
/// Note: In production, this should be replaced with Metal GPU kernels
fn dequantizeCpuImpl(
    weights: mlx.Array,
    biases: mlx.Array,
    scales: mlx.Array,
    config: DequantizeConfig,
    out_result: *mlx.Array,
) DequantizeError!void {

    // Get raw data pointers
    const weight_data = mlx.arrayDataUint8(weights);
    const bias_data = mlx.arrayDataFloat32(biases);
    const scale_data = mlx.arrayDataFloat32(scales);

    if (weight_data == null or bias_data == null or scale_data == null) {
        return DequantizeError.InvalidInput;
    }

    // Get array dimensions
    const weight_shape = mlx.arrayShape(weights);
    const num_elements = calculateNumElements(&weight_shape);
    const num_groups = num_elements / config.group_size;

    // Create temporary buffer for dequantized values
    const allocator = std.heap.page_allocator;
    const dequantized = allocator.alloc(f32, num_elements) catch {
        return DequantizeError.OutOfMemory;
    };
    defer allocator.free(dequantized);

    // Dequantize each element
    var group_idx: usize = 0;
    var element_idx: usize = 0;

    while (group_idx < num_groups) : (group_idx += 1) {
        const scale = scale_data[group_idx];
        const bias = bias_data[group_idx];

        var group_element: usize = 0;
        while (group_element < config.group_size and element_idx < num_elements) : ({
            group_element += 1;
            element_idx += 1;
        }) {
            // Get packed byte and unpack nibble
            const packed_idx = element_idx / 2;
            const is_high_nibble = element_idx % 2 == 0;

            const packed_byte = weight_data[packed_idx];
            const nibble: u8 = if (is_high_nibble)
                (packed_byte >> 4) & 0xF
            else
                packed_byte & 0xF;

            // Convert to signed if needed (mlx-community uses unsigned 0-15)
            const signed_val: f32 = if (config.signed)
                @as(f32, @floatFromInt(@as(i8, @intCast(nibble)) - 8))
            else
                @as(f32, @floatFromInt(nibble));

            // Apply dequantization formula: (value - bias) * scale
            dequantized[element_idx] = (signed_val - bias) * scale;
        }
    }

    // Create new MLX array from dequantized data
    // Note: In actual implementation, this would use MLX operations
    // For now, we create a new array and copy the data
    out_result.* = mlx.arrayNewData(
        dequantized.ptr,
        &weight_shape,
        weight_shape.len,
        mlx.float32,
    );

    // Cast to desired output dtype if needed
    if (config.output_dtype == .f16) {
        const casted = mlx.arrayAstype(out_result.*, mlx.float16);
        mlx.arrayFree(out_result.*);
        out_result.* = casted;
    }
}

/// Quantized weight structure matching deepseek.zig
pub const QuantizedWeight = struct {
    weight: mlx.Array,
    biases: mlx.Array,
    scales: mlx.Array,

    /// Check if all components are valid
    pub fn isValid(self: QuantizedWeight) bool {
        return !mlx.arrayIsEmpty(self.weight) and
            !mlx.arrayIsEmpty(self.biases) and
            !mlx.arrayIsEmpty(self.scales);
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
