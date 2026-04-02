//! MLX Array Bridge
//!
//! Utilities for converting between MLX GPU arrays and CPU f32 buffers.
//! Required because TurboQuant operates on CPU f32 slices.

const std = @import("std");
const mlx = @import("mlx.zig/src/mlx.zig");
pub const c = @import("c.zig").mlx;

/// Managed buffer for temporary f32 data
pub const MlxBuffer = struct {
    data: []f32,
    allocator: std.mem.Allocator,

    pub fn alloc(allocator: std.mem.Allocator, size: usize) !MlxBuffer {
        const data = try allocator.alloc(f32, size);
        return MlxBuffer{ .data = data, .allocator = allocator };
    }

    pub fn free(self: *MlxBuffer) void {
        self.allocator.free(self.data);
        self.data = &.{};
    }
};

/// MLX Bridge Error types
pub const BridgeError = error{
    InvalidDtype,
    BufferTooSmall,
    NullDataPointer,
    ArrayCreationFailed,
    EvalFailed,
    ShapeMismatch,
};

/// Extract f32 data from MLX array (GPU → CPU)
///
/// Parameters:
///   - arr: MLX array (must be float32 dtype, will be evaluated)
///   - buffer: Pre-allocated f32 buffer
///
/// Returns: Number of elements copied, or error
pub fn arrayToF32(arr: mlx.Array, buffer: []f32) BridgeError!usize {
    // Ensure array is evaluated (GPU → CPU sync)
    mlx.arrayEval(arr) catch {
        return BridgeError.EvalFailed;
    };

    // Get array properties
    const dtype = c.mlx_array_dtype(arr);
    if (dtype != c.MLX_FLOAT32) {
        std.log.err("arrayToF32: expected float32, got dtype {d}", .{dtype});
        return BridgeError.InvalidDtype;
    }

    const nelements = c.mlx_array_size(arr);
    if (nelements > buffer.len) {
        return BridgeError.BufferTooSmall;
    }

    // Get raw data pointer from MLX
    const data_ptr = c.mlx_array_data_float32(arr);
    if (data_ptr == null) {
        return BridgeError.NullDataPointer;
    }

    // Copy data to buffer
    const src = data_ptr.?[0..nelements];
    @memcpy(buffer[0..nelements], src);

    return nelements;
}

/// Create MLX array from f32 data (CPU → GPU)
///
/// Parameters:
///   - data: Source f32 data
///   - shape: Target shape for the array
///   - stream: MLX stream for device placement
///
/// Returns: New MLX array on GPU
pub fn f32ToArray(data: []const f32, shape: []const i64, stream: mlx.Stream) BridgeError!mlx.Array {
    _ = stream; // Stream parameter reserved for future use with explicit device placement
    // Create array from CPU data (will be moved to GPU by MLX)
    // Convert i64 shape to c_int for MLX C API
    var shape_c: [8]c_int = undefined;
    if (shape.len > 8) {
        return BridgeError.ShapeMismatch;
    }
    for (0..shape.len) |i| {
        shape_c[i] = @intCast(shape[i]);
    }

    const arr = c.mlx_array_new_data(
        data.ptr,
        &shape_c,
        @intCast(shape.len),
        c.MLX_FLOAT32,
    );

    if (arr.ctx == null) {
        return BridgeError.ArrayCreationFailed;
    }

    // Evaluate to move to stream (GPU)
    mlx.arrayEval(arr) catch {
        c.mlx_array_free(arr);
        return BridgeError.EvalFailed;
    };

    return arr;
}

/// Get total number of elements in array
pub fn arrayNumel(arr: mlx.Array) usize {
    return c.mlx_array_size(arr);
}

/// Get array nbytes (total bytes)
pub fn arrayNbytes(arr: mlx.Array) usize {
    return c.mlx_array_nbytes(arr);
}

/// Get array dtype
pub fn arrayDtype(arr: mlx.Array) c.mlx_dtype {
    return c.mlx_array_dtype(arr);
}

/// Get shape as slice of i64 (allocated, caller must free)
pub fn arrayShapeSlice(arr: mlx.Array, allocator: std.mem.Allocator) ![]i64 {
    const ndim = c.mlx_array_ndim(arr);
    const shape_ptr = c.mlx_array_shape(arr);
    if (shape_ptr == null) {
        return BridgeError.NullDataPointer;
    }

    const shape = try allocator.alloc(i64, ndim);
    for (0..ndim) |i| {
        shape[i] = shape_ptr[i];
    }

    return shape;
}

/// Get number of dimensions
pub fn arrayNDim(arr: mlx.Array) usize {
    return @intCast(c.mlx_array_ndim(arr));
}

/// Get size of specific dimension
pub fn arrayDim(arr: mlx.Array, dim_idx: usize) i64 {
    return c.mlx_array_dim(arr, @intCast(dim_idx));
}

// =============================================================================
// Tests
// =============================================================================

test "MlxBuffer alloc/free" {
    const allocator = std.testing.allocator;

    var buffer = try MlxBuffer.alloc(allocator, 1024);
    defer buffer.free();

    try std.testing.expectEqual(@as(usize, 1024), buffer.data.len);
}
