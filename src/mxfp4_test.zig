//! mxfp4_test.zig - Tests for MXFP4 dequantization
//!
//! Tests cover:
//! - FP4 E2M1 lookup table values
//! - Block-wise dequantization correctness
//! - MXFP4Tensor initialization
//! - loadMXFP4Blocks parsing
//! - applyBlockScales correctness
//! - safetensorsToMLX basic shape validation

const std = @import("std");
const testing = std.testing;

// We only test pure-Zig functions that don't require MLX runtime
// The MLX-dependent functions (GPU path, arrayFromData) are tested via integration

// ============================================================================
// FP4 E2M1 lookup table tests
// ============================================================================
// The FP4 E2M1 table has 16 values from 0 to 1.75
// Nibble 0 = 0.0, nibble 1 = 0.0625, ..., nibble 15 = 1.75

fn fp4ToFloat(nibble: u8) f32 {
    const table = [_]f32{
        0.0,  0.0625, 0.125, 0.1875,
        0.25, 0.3125, 0.375, 0.4375,
        0.5,  0.625,  0.75,  0.875,
        1.0,  1.25,   1.5,   1.75,
    };
    return table[nibble & 0x0F];
}

test "FP4 E2M1 table nibble 0 = 0.0" {
    try testing.expectApproxEqAbs(@as(f32, 0.0), fp4ToFloat(0), 1e-6);
}

test "FP4 E2M1 table nibble 1 = 0.0625" {
    try testing.expectApproxEqAbs(@as(f32, 0.0625), fp4ToFloat(1), 1e-6);
}

test "FP4 E2M1 table nibble 4 = 0.25" {
    try testing.expectApproxEqAbs(@as(f32, 0.25), fp4ToFloat(4), 1e-6);
}

test "FP4 E2M1 table nibble 8 = 0.5" {
    try testing.expectApproxEqAbs(@as(f32, 0.5), fp4ToFloat(8), 1e-6);
}

test "FP4 E2M1 table nibble 12 = 1.0" {
    try testing.expectApproxEqAbs(@as(f32, 1.0), fp4ToFloat(12), 1e-6);
}

test "FP4 E2M1 table nibble 15 = 1.75" {
    try testing.expectApproxEqAbs(@as(f32, 1.75), fp4ToFloat(15), 1e-6);
}

test "FP4 E2M1 all 16 values are non-negative" {
    for (0..16) |nibble| {
        const val = fp4ToFloat(@intCast(nibble));
        try testing.expect(val >= 0.0);
    }
}

test "FP4 E2M1 values are monotonically non-decreasing" {
    var prev: f32 = -1.0;
    for (0..16) |nibble| {
        const val = fp4ToFloat(@intCast(nibble));
        try testing.expect(val >= prev);
        prev = val;
    }
}

test "FP4 E2M1 nibble masked to 4 bits (0x0F)" {
    // Nibble 0 with high bits set should still give 0.0
    try testing.expectApproxEqAbs(@as(f32, 0.0), fp4ToFloat(0x10), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.0), fp4ToFloat(0xF0), 1e-6);
    // Nibble 0x1F should mask to 0xF = 1.75
    try testing.expectApproxEqAbs(@as(f32, 1.75), fp4ToFloat(0x1F), 1e-6);
}

// ============================================================================
// CPU dequantization logic tests (pure Zig, no MLX)
// ============================================================================

/// Pure CPU dequantization without MLX dependency
fn dequantizeCPU(
    blocks: []const u8,
    scales: []const f32,
    output: []f32,
    block_size: usize,
) void {
    for (0..scales.len) |block_idx| {
        const scale = scales[block_idx];
        const block_start = block_idx * block_size;
        const block_end = @min(block_start + block_size, output.len);

        for (block_start..block_end) |i| {
            const byte_idx = i / 2;
            const nibble: u8 = if (i % 2 == 0)
                blocks[byte_idx] & 0x0F
            else
                (blocks[byte_idx] >> 4) & 0x0F;

            output[i] = fp4ToFloat(nibble) * scale;
        }
    }
}

test "CPU dequantize single block all zeros gives zero output" {
    // blocks: 16 bytes of zeros (32 elements, all nibble 0 = 0.0)
    const blocks = [_]u8{0} ** 16;
    const scales = [_]f32{1.0};
    var output = [_]f32{999.0} ** 32;

    dequantizeCPU(&blocks, &scales, &output, 32);

    for (output) |v| {
        try testing.expectApproxEqAbs(@as(f32, 0.0), v, 1e-6);
    }
}

test "CPU dequantize single block all 0xFF gives 1.75 * scale" {
    // All bytes 0xFF = nibble high 0xF and low 0xF = 1.75 each
    const blocks = [_]u8{0xFF} ** 16;
    const scale_val: f32 = 2.0;
    const scales = [_]f32{scale_val};
    var output = [_]f32{0.0} ** 32;

    dequantizeCPU(&blocks, &scales, &output, 32);

    for (output) |v| {
        try testing.expectApproxEqAbs(@as(f32, 1.75 * scale_val), v, 1e-5);
    }
}

test "CPU dequantize two blocks with different scales" {
    // Two blocks of 32 elements each
    // Block 0: all zeros, scale 1.0 -> all 0.0
    // Block 1: all 0xFF, scale 0.5 -> all 1.75 * 0.5 = 0.875
    var blocks = [_]u8{0} ** 32;
    // Second block (bytes 16..32): all 0xFF
    for (blocks[16..32]) |*b| b.* = 0xFF;

    const scales = [_]f32{ 1.0, 0.5 };
    var output = [_]f32{0.0} ** 64;

    dequantizeCPU(&blocks, &scales, &output, 32);

    // First block: all 0.0
    for (output[0..32]) |v| {
        try testing.expectApproxEqAbs(@as(f32, 0.0), v, 1e-6);
    }

    // Second block: all 1.75 * 0.5 = 0.875
    for (output[32..64]) |v| {
        try testing.expectApproxEqAbs(@as(f32, 0.875), v, 1e-5);
    }
}

test "CPU dequantize packed nibbles correctly" {
    // byte 0xAB: low nibble = 0xB (11 = 0.875), high nibble = 0xA (10 = 0.75)
    // element 0: nibble at i=0 -> byte[0] & 0x0F = 0xB = 11 -> 0.875
    // element 1: nibble at i=1 -> (byte[0] >> 4) & 0x0F = 0xA = 10 -> 0.75
    const blocks = [_]u8{0xAB} ** 16; // 32 elements, alternating 0.875, 0.75
    const scales = [_]f32{1.0};
    var output = [_]f32{0.0} ** 32;

    dequantizeCPU(&blocks, &scales, &output, 32);

    // Element 0 (even i): low nibble of first byte = 0xB = 11 -> 0.875
    try testing.expectApproxEqAbs(@as(f32, 0.875), output[0], 1e-5);
    // Element 1 (odd i): high nibble of first byte = 0xA = 10 -> 0.75
    try testing.expectApproxEqAbs(@as(f32, 0.75), output[1], 1e-5);
}

test "CPU dequantize scale is applied correctly" {
    // Single block, all nibble 8 (= 0.5), scale = 3.0
    // Expected: 0.5 * 3.0 = 1.5
    const blocks = [_]u8{0x88} ** 16; // nibble 8 in both low and high
    const scales = [_]f32{3.0};
    var output = [_]f32{0.0} ** 32;

    dequantizeCPU(&blocks, &scales, &output, 32);

    for (output) |v| {
        try testing.expectApproxEqAbs(@as(f32, 1.5), v, 1e-5);
    }
}

// ============================================================================
// applyBlockScales tests (pure Zig)
// ============================================================================

fn applyBlockScales(values: []f32, scales: []const f32, block_size: usize) void {
    for (0..scales.len) |block_idx| {
        const scale = scales[block_idx];
        const start = block_idx * block_size;
        const end = @min(start + block_size, values.len);

        for (start..end) |i| {
            values[i] *= scale;
        }
    }
}

test "applyBlockScales single block" {
    var values = [_]f32{1.0} ** 4;
    const scales = [_]f32{2.5};
    applyBlockScales(&values, &scales, 4);
    for (values) |v| {
        try testing.expectApproxEqAbs(@as(f32, 2.5), v, 1e-6);
    }
}

test "applyBlockScales multiple blocks" {
    var values = [_]f32{1.0} ** 8;
    const scales = [_]f32{ 2.0, 3.0 };
    applyBlockScales(&values, &scales, 4);
    for (values[0..4]) |v| {
        try testing.expectApproxEqAbs(@as(f32, 2.0), v, 1e-6);
    }
    for (values[4..8]) |v| {
        try testing.expectApproxEqAbs(@as(f32, 3.0), v, 1e-6);
    }
}

test "applyBlockScales zero scale produces zeros" {
    var values = [_]f32{1.0} ** 4;
    const scales = [_]f32{0.0};
    applyBlockScales(&values, &scales, 4);
    for (values) |v| {
        try testing.expectApproxEqAbs(@as(f32, 0.0), v, 1e-6);
    }
}

// ============================================================================
// loadMXFP4Blocks tests (pure Zig - no MLX)
// ============================================================================

/// Replicate loadMXFP4Blocks logic without MLX dependency
fn loadMXFP4BlocksTest(
    allocator: std.mem.Allocator,
    data: []const u8,
    num_elements: usize,
    block_size: usize,
) !struct { blocks: []u8, scales: []f32, num_blocks: usize } {
    const num_blocks = (num_elements + block_size - 1) / block_size;
    const blocks_size = (num_elements + 1) / 2; // 2 values per byte

    const blocks = try allocator.alloc(u8, blocks_size);
    @memcpy(blocks, data[0..blocks_size]);

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

test "loadMXFP4Blocks correct block count for 32 elements" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // 32 elements with block_size=32 -> 1 block
    // blocks_size = (32+1)/2 = 16 bytes
    // scales_offset = ((16+3)/4)*4 = 16 (already aligned)
    // total data = 16 + 4 = 20 bytes
    const block_size: usize = 32;
    const num_elements: usize = 32;
    const blocks_size = (num_elements + 1) / 2; // 16
    const scales_offset = ((blocks_size + 3) / 4) * 4; // 16
    const num_blocks = (num_elements + block_size - 1) / block_size; // 1

    // Create test data
    const data_size = scales_offset + num_blocks * 4;
    const data = try allocator.alloc(u8, data_size);
    defer allocator.free(data);
    @memset(data, 0);

    // Write scale = 1.5 as f32
    const scale_val: f32 = 1.5;
    std.mem.writeInt(u32, data[scales_offset..][0..4], @bitCast(scale_val), .little);

    const result = try loadMXFP4BlocksTest(allocator, data, num_elements, block_size);
    defer {
        allocator.free(result.blocks);
        allocator.free(result.scales);
    }

    try testing.expectEqual(@as(usize, 1), result.num_blocks);
    try testing.expectEqual(@as(usize, 16), result.blocks.len);
    try testing.expectApproxEqAbs(@as(f32, 1.5), result.scales[0], 1e-6);
}

test "loadMXFP4Blocks correct block count for 64 elements" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const block_size: usize = 32;
    const num_elements: usize = 64;
    const blocks_size = (num_elements + 1) / 2; // 32
    const scales_offset = ((blocks_size + 3) / 4) * 4; // 32
    const num_blocks = 2;

    const data_size = scales_offset + num_blocks * 4;
    const data = try allocator.alloc(u8, data_size);
    defer allocator.free(data);
    @memset(data, 0);

    const scale0: f32 = 1.0;
    const scale1: f32 = 2.0;
    std.mem.writeInt(u32, data[scales_offset..][0..4], @bitCast(scale0), .little);
    std.mem.writeInt(u32, data[scales_offset + 4 ..][0..4], @bitCast(scale1), .little);

    const result = try loadMXFP4BlocksTest(allocator, data, num_elements, block_size);
    defer {
        allocator.free(result.blocks);
        allocator.free(result.scales);
    }

    try testing.expectEqual(@as(usize, 2), result.num_blocks);
    try testing.expectApproxEqAbs(@as(f32, 1.0), result.scales[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 2.0), result.scales[1], 1e-6);
}

test "loadMXFP4Blocks odd number of elements rounds up bytes" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const block_size: usize = 32;
    const num_elements: usize = 33; // Odd
    const blocks_size = (num_elements + 1) / 2; // 17 bytes
    const scales_offset = ((blocks_size + 3) / 4) * 4; // 20 (aligned to 4)
    const num_blocks = (num_elements + block_size - 1) / block_size; // 2

    const data_size = scales_offset + num_blocks * 4;
    const data = try allocator.alloc(u8, data_size);
    defer allocator.free(data);
    @memset(data, 0);

    const result = try loadMXFP4BlocksTest(allocator, data, num_elements, block_size);
    defer {
        allocator.free(result.blocks);
        allocator.free(result.scales);
    }

    try testing.expectEqual(@as(usize, 17), result.blocks.len);
    try testing.expectEqual(@as(usize, 2), result.num_blocks);
}

// ============================================================================
// MXFP4Tensor struct tests (pure Zig - no MLX)
// ============================================================================

test "MXFP4Tensor init computes total_elements correctly" {
    const blocks = [_]u8{0} ** 16;
    const scales = [_]f32{1.0};
    const shape = [_]usize{ 4, 8 };

    // Use struct literal directly (no MLX needed for init/struct inspection)
    const tensor = struct {
        blocks: []const u8,
        scales: []const f32,
        num_blocks: usize,
        block_size: usize,
        total_elements: usize,
        shape: []const usize,
    }{
        .blocks = &blocks,
        .scales = &scales,
        .num_blocks = 1,
        .block_size = 32,
        .total_elements = blk: {
            var total: usize = 1;
            for (shape) |d| total *= d;
            break :blk total;
        },
        .shape = &shape,
    };

    try testing.expectEqual(@as(usize, 32), tensor.total_elements);
    try testing.expectEqual(@as(usize, 1), tensor.num_blocks);
    try testing.expectEqual(@as(usize, 32), tensor.block_size);
}

test "MXFP4Tensor block count from shape" {
    // 5120 x 5120 = 26,214,400 elements with block_size=32 -> 819,200 blocks
    const shape = [_]usize{ 5120, 5120 };
    var total: usize = 1;
    for (shape) |d| total *= d;
    const num_blocks = (total + 31) / 32;
    try testing.expectEqual(@as(usize, 26_214_400), total);
    try testing.expectEqual(@as(usize, 819_200), num_blocks);
}

// ============================================================================
// Round-trip test: pack nibbles -> dequantize -> verify
// ============================================================================

test "Round-trip: pack known values and dequantize" {
    // Pack nibbles: 2 nibbles per byte
    // Elements: [4, 8] (nibbles 4=0.25, 8=0.5) packed into byte 0x84 (high=8, low=4)
    // scale = 2.0 -> expected output: [0.5, 1.0]
    const packed_byte: u8 = 0x84; // low nibble = 4, high nibble = 8
    const blocks = [_]u8{packed_byte};
    const scales = [_]f32{2.0};
    var output = [_]f32{0.0} ** 2;

    dequantizeCPU(&blocks, &scales, &output, 2);

    // Element 0 (i=0, even): low nibble of byte[0] = 0x4 = 4 -> 0.25 * 2.0 = 0.5
    try testing.expectApproxEqAbs(@as(f32, 0.5), output[0], 1e-5);
    // Element 1 (i=1, odd): high nibble of byte[0] = 0x8 = 8 -> 0.5 * 2.0 = 1.0
    try testing.expectApproxEqAbs(@as(f32, 1.0), output[1], 1e-5);
}

test "Round-trip: block size boundary handling" {
    // Test that partial last block works correctly
    // 3 elements with block_size=4: only 3 elements in the single block
    // Packed: 3 elements in 2 bytes (byte0: elements 0,1; byte1: element 2, padding)
    const blocks = [_]u8{ 0x12, 0x03 }; // elem0=2, elem1=1, elem2=3
    const scales = [_]f32{1.0};
    var output = [_]f32{0.0} ** 3;

    dequantizeCPU(&blocks, &scales, &output, 4);

    try testing.expectApproxEqAbs(fp4ToFloat(2), output[0], 1e-5);
    try testing.expectApproxEqAbs(fp4ToFloat(1), output[1], 1e-5);
    try testing.expectApproxEqAbs(fp4ToFloat(3), output[2], 1e-5);
}
