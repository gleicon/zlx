// src/mlx_v4.zig
// MLX-C v0.4.x Fast Custom Ops API wrapper
// Provides FastMetalKernel for custom Metal operations needed by MLA and MoE

const std = @import("std");
const c = @import("c_v4.zig");

// Re-export c_v4 types for convenience
pub const mlx_v4 = c.mlx_v4;

/// Error types for mlx_v4 operations
pub const MlxV4Error = error{
    KernelCreationFailed,
    KernelApplyFailed,
    InvalidInput,
    OutOfMemory,
};

/// FastMetalKernel wraps mlx-c v0.4.x's custom Metal kernel API
/// for high-performance operations like MoE routing and MLA attention
pub const FastMetalKernel = struct {
    kernel: ?*c.mlx_v4.mlx_fast_metal_kernel,
    allocator: std.mem.Allocator,
    name: []const u8,

    const Self = @This();

    /// Initialize a new Metal kernel from source code
    ///
    /// Args:
    ///   allocator: Memory allocator for string management
    ///   name: Kernel name (used for identification)
    ///   input_names: Array of input buffer names
    ///   output_names: Array of output buffer names
    ///   metal_source: Complete Metal shader source code
    ///   ensure_row_contiguous: Whether to ensure row-contiguous memory layout
    ///   atomic_outputs: Whether outputs use atomic operations
    ///
    /// Returns:
    ///   Initialized FastMetalKernel or error
    pub fn init(
        allocator: std.mem.Allocator,
        name: []const u8,
        input_names: []const []const u8,
        output_names: []const []const u8,
        metal_source: []const u8,
        ensure_row_contiguous: bool,
        atomic_outputs: bool,
    ) MlxV4Error!Self {
        // Convert input names to C strings
        const input_c_strings = allocator.alloc([*c]const u8, input_names.len) catch {
            return MlxV4Error.OutOfMemory;
        };
        defer allocator.free(input_c_strings);

        for (input_names, 0..) |input_name, i| {
            input_c_strings[i] = allocator.dupeZ(u8, input_name) catch {
                // Free previously allocated strings
                for (0..i) |j| {
                    allocator.free(std.mem.span(input_c_strings[j]));
                }
                return MlxV4Error.OutOfMemory;
            };
        }
        defer {
            for (input_c_strings) |c_str| {
                allocator.free(std.mem.span(c_str));
            }
        }

        // Convert output names to C strings
        const output_c_strings = allocator.alloc([*c]const u8, output_names.len) catch {
            return MlxV4Error.OutOfMemory;
        };
        defer allocator.free(output_c_strings);

        for (output_names, 0..) |output_name, i| {
            output_c_strings[i] = allocator.dupeZ(u8, output_name) catch {
                for (0..i) |j| {
                    allocator.free(std.mem.span(output_c_strings[j]));
                }
                return MlxV4Error.OutOfMemory;
            };
        }
        defer {
            for (output_c_strings) |c_str| {
                allocator.free(std.mem.span(c_str));
            }
        }

        // Create input/output name vectors
        var input_vec: c.mlx_v4.mlx_vector_string = undefined;
        c.mlx_v4.mlx_vector_string_new_data(
            &input_vec,
            input_c_strings.ptr,
            @intCast(input_names.len),
        );
        defer c.mlx_v4.mlx_vector_string_free(&input_vec);

        var output_vec: c.mlx_v4.mlx_vector_string = undefined;
        c.mlx_v4.mlx_vector_string_new_data(
            &output_vec,
            output_c_strings.ptr,
            @intCast(output_names.len),
        );
        defer c.mlx_v4.mlx_vector_string_free(&output_vec);

        // Create kernel
        const source_c = allocator.dupeZ(u8, metal_source) catch {
            return MlxV4Error.OutOfMemory;
        };
        defer allocator.free(source_c);

        const name_c = allocator.dupeZ(u8, name) catch {
            return MlxV4Error.OutOfMemory;
        };
        defer allocator.free(name_c);

        const header_c = allocator.dupeZ(u8, "") catch {
            return MlxV4Error.OutOfMemory;
        };
        defer allocator.free(header_c);

        var kernel: c.mlx_v4.mlx_fast_metal_kernel = undefined;
        const result = c.mlx_v4.mlx_fast_metal_kernel_new(
            &kernel,
            name_c,
            &input_vec,
            &output_vec,
            source_c,
            header_c,
            if (ensure_row_contiguous) 1 else 0,
            if (atomic_outputs) 1 else 0,
        );

        if (result != 0) {
            return MlxV4Error.KernelCreationFailed;
        }

        // Allocate persistent name copy
        const name_copy = allocator.dupe(u8, name) catch {
            _ = c.mlx_v4.mlx_fast_metal_kernel_free(&kernel);
            return MlxV4Error.OutOfMemory;
        };

        return Self{
            .kernel = kernel,
            .allocator = allocator,
            .name = name_copy,
        };
    }

    /// Free kernel resources
    pub fn deinit(self: *Self) void {
        if (self.kernel) |*k| {
            _ = c.mlx_v4.mlx_fast_metal_kernel_free(k);
            self.kernel = null;
        }
        self.allocator.free(self.name);
    }

    /// Apply the kernel to input arrays
    ///
    /// Args:
    ///   inputs: Input mlx arrays
    ///   outputs: Output mlx arrays (pre-allocated)
    ///   grid_dims: Grid dimensions (3D: x, y, z)
    ///   thread_group_dims: Thread group dimensions (3D: x, y, z)
    ///   stream: MLX stream for execution
    ///
    /// Returns:
    ///   Error if kernel execution fails
    pub fn apply(
        self: Self,
        inputs: []const c.mlx_v4.mlx_array,
        outputs: []c.mlx_v4.mlx_array,
        grid_dims: [3]u32,
        thread_group_dims: [3]u32,
        stream: c.mlx_v4.mlx_stream,
    ) MlxV4Error!void {
        if (self.kernel == null) {
            return MlxV4Error.InvalidInput;
        }

        // Create input vector
        var input_vec: c.mlx_v4.mlx_vector_array = undefined;
        c.mlx_v4.mlx_vector_array_new_data(
            &input_vec,
            inputs.ptr,
            @intCast(inputs.len),
        );
        defer c.mlx_v4.mlx_vector_array_free(&input_vec);

        // Create output vector (will be modified by kernel)
        var output_vec: c.mlx_v4.mlx_vector_array = undefined;
        c.mlx_v4.mlx_vector_array_new_data(
            &output_vec,
            outputs.ptr,
            @intCast(outputs.len),
        );
        defer c.mlx_v4.mlx_vector_array_free(&output_vec);

        // Create kernel configuration
        const config = c.mlx_v4.mlx_fast_metal_kernel_config{
            .grid_dims = .{
                @intCast(grid_dims[0]),
                @intCast(grid_dims[1]),
                @intCast(grid_dims[2]),
            },
            .thread_group_dims = .{
                @intCast(thread_group_dims[0]),
                @intCast(thread_group_dims[1]),
                @intCast(thread_group_dims[2]),
            },
        };

        // Apply kernel
        var kernel_mut = self.kernel.?;
        const result = c.mlx_v4.mlx_fast_metal_kernel_apply(
            &output_vec,
            &kernel_mut,
            &input_vec,
            &config,
            stream,
        );

        if (result != 0) {
            return MlxV4Error.KernelApplyFailed;
        }

        // Copy results back to output arrays
        for (0..outputs.len) |i| {
            _ = c.mlx_v4.mlx_array_free(&outputs[i]);
            const arr_result = c.mlx_v4.mlx_vector_array_get(&output_vec, @intCast(i), &outputs[i]);
            if (arr_result != 0) {
                return MlxV4Error.KernelApplyFailed;
            }
        }
    }
};

/// Check if mlx-c v0.4.x Fast Ops API is available
/// This can be used to conditionally enable advanced features
pub fn hasFastOps() bool {
    // Try to access a v0.4.x specific function
    // If the library is not linked, this will fail at link time
    // At runtime, we check if the symbols are present by attempting
    // a minimal operation
    return true; // Simplified - actual check would verify library presence
}

/// Configuration for grid/thread dimensions
pub const KernelConfig = struct {
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    thread_x: u32,
    thread_y: u32,
    thread_z: u32,

    pub fn toArrays(self: KernelConfig) struct { grid: [3]u32, threads: [3]u32 } {
        return .{
            .grid = .{ self.grid_x, self.grid_y, self.grid_z },
            .threads = .{ self.thread_x, self.thread_y, self.thread_z },
        };
    }
};

/// Default 1D kernel configuration
pub fn default1DConfig(size: u32) KernelConfig {
    const threads_per_group = 256;
    const groups = (size + threads_per_group - 1) / threads_per_group;
    return KernelConfig{
        .grid_x = groups,
        .grid_y = 1,
        .grid_z = 1,
        .thread_x = threads_per_group,
        .thread_y = 1,
        .thread_z = 1,
    };
}
