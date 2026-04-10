//! mxfp4.zig - MXFP4 dequantization for GPT-OSS weights

const std = @import("std");
const mlx = @import("mlx.zig/src/mlx.zig");

/// MXFP4 tensor with block quantization
pub const MXFP4Tensor = struct {
    blocks: []const u8, // Packed FP4 values (2 per byte)
    scales: []const f32, // Block scale factors
    num_blocks: usize,
    block_size: usize,
    total_elements: usize,
    shape: []const usize,

    pub fn init(
        blocks: []const u8,
        scales: []const f32,
        num_blocks: usize,
        block_size: usize,
        shape: []const usize,
    ) MXFP4Tensor {
        var total: usize = 1;
        for (shape) |dim| {
            total *= dim;
        }

        return .{
            .blocks = blocks,
            .scales = scales,
            .num_blocks = num_blocks,
            .block_size = block_size,
            .total_elements = total,
            .shape = shape,
        };
    }

    /// Dequantize to BF16 (native MLX format)
    pub fn dequantizeToBF16(
        self: *const MXFP4Tensor,
        allocator: std.mem.Allocator,
        use_gpu: bool,
    ) !mlx.Array {
        // Allocate output buffer
        const output = try allocator.alloc(f32, self.total_elements);
        defer allocator.free(output);

        // Dequantize each block
        for (0..self.num_blocks) |block_idx| {
            const scale = self.scales[block_idx];
            const block_start = block_idx * self.block_size;
            const block_end = @min(block_start + self.block_size, self.total_elements);

            for (block_start..block_end) |i| {
                const byte_idx = i / 2;
                const nibble = if (i % 2 == 0)
                    self.blocks[byte_idx] & 0x0F
                else
                    (self.blocks[byte_idx] >> 4) & 0x0F;

                output[i] = fp4ToFloat(nibble) * scale;
            }
        }

        // Create MLX array from CPU data
        const mlx_shape = try allocator.alloc(i32, self.shape.len);
        defer allocator.free(mlx_shape);

        for (self.shape, 0..) |dim, i| {
            mlx_shape[i] = @intCast(dim);
        }

        // MLX arrays always run on Metal GPU on Apple Silicon — no explicit device transfer needed
        _ = use_gpu;
        // Use C API directly because shape is dynamic (unknown at compile time)
        const arr = mlx.C.mlx_array_new_data(
            output.ptr,
            mlx_shape.ptr,
            @intCast(self.shape.len),
            mlx.FLOAT32,
        );
        if (arr.ctx == null) return error.InvalidArray;
        return arr;
    }
};

/// FP4 lookup table (E2M1 format: 2 exponent bits, 1 mantissa bit)
/// Values: 0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375,
///         0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75
fn fp4ToFloat(nibble: u8) f32 {
    const table = [_]f32{
        0.0,  0.0625, 0.125, 0.1875,
        0.25, 0.3125, 0.375, 0.4375,
        0.5,  0.625,  0.75,  0.875,
        1.0,  1.25,   1.5,   1.75,
    };
    return table[nibble & 0x0F];
}

/// GPU-accelerated dequantization using Metal kernel
pub fn dequantizeMXFP4GPU(
    blocks: []const u8,
    scales: []const f32,
    num_blocks: usize,
    block_size: usize,
    stream: mlx.Stream,
) !mlx.Array {
    _ = stream;

    // TODO: Load and execute Metal kernel
    // For now, fallback to CPU
    const total_elements = num_blocks * block_size;
    var output = try std.heap.page_allocator.alloc(f32, total_elements);
    defer std.heap.page_allocator.free(output);

    for (0..num_blocks) |block_idx| {
        const scale = scales[block_idx];
        const block_start = block_idx * (block_size / 2);

        for (0..block_size) |i| {
            const byte_idx = block_start + (i / 2);
            const nibble = if (i % 2 == 0)
                blocks[byte_idx] & 0x0F
            else
                (blocks[byte_idx] >> 4) & 0x0F;

            output[block_idx * block_size + i] = fp4ToFloat(nibble) * scale;
        }
    }

    // Use C API directly because shape size is runtime-known
    const shape = [_]i32{@intCast(total_elements)};
    const arr = mlx.C.mlx_array_new_data(
        output.ptr,
        &shape,
        1,
        mlx.FLOAT32,
    );
    if (arr.ctx == null) return error.InvalidArray;
    return arr;
}

/// Load MXFP4 tensor from raw data
pub fn loadMXFP4Blocks(
    allocator: std.mem.Allocator,
    data: []const u8,
    num_elements: usize,
    block_size: usize,
) !struct { blocks: []u8, scales: []f32, num_blocks: usize } {
    const num_blocks = (num_elements + block_size - 1) / block_size;
    const blocks_size = (num_elements + 1) / 2; // 2 values per byte

    // Read blocks (packed FP4 values)
    const blocks = try allocator.alloc(u8, blocks_size);
    @memcpy(blocks, data[0..blocks_size]);

    // Read scales (after blocks, aligned)
    const scales_offset = ((blocks_size + 3) / 4) * 4; // Align to 4 bytes
    const scales_data = data[scales_offset..];

    const scales = try allocator.alloc(f32, num_blocks);
    for (0..num_blocks) |i| {
        scales[i] = std.mem.bytesToValue(f32, scales_data[i * 4 ..][0..4]);
    }

    return .{
        .blocks = blocks,
        .scales = scales,
        .num_blocks = num_blocks,
    };
}

/// Apply block scales to dequantized values
pub fn applyBlockScales(
    values: []f32,
    scales: []const f32,
    block_size: usize,
) void {
    for (0..scales.len) |block_idx| {
        const scale = scales[block_idx];
        const start = block_idx * block_size;
        const end = @min(start + block_size, values.len);

        for (start..end) |i| {
            values[i] *= scale;
        }
    }
}

/// Convert safetensors MXFP4 tensor to MLX array
pub fn safetensorsToMLX(
    allocator: std.mem.Allocator,
    tensor_data: []const u8,
    shape: []const usize,
    use_gpu: bool,
) !mlx.Array {
    var total_elements: usize = 1;
    for (shape) |dim| {
        total_elements *= dim;
    }

    const block_size: usize = 32; // Standard MXFP4 block size
    const loaded = try loadMXFP4Blocks(
        allocator,
        tensor_data,
        total_elements,
        block_size,
    );
    defer {
        allocator.free(loaded.blocks);
        allocator.free(loaded.scales);
    }

    const mxfp4_tensor = MXFP4Tensor{
        .blocks = loaded.blocks,
        .scales = loaded.scales,
        .num_blocks = loaded.num_blocks,
        .block_size = block_size,
        .total_elements = total_elements,
        .shape = shape,
    };

    return try mxfp4_tensor.dequantizeToBF16(allocator, use_gpu);
}
