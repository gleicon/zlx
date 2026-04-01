// src/c.zig
// Single @cImport boundary for all mlx-c C interop (BUILD-04).
// Import this file everywhere MLX C types are needed:
//   const c = @import("c.zig");
//   _ = c.mlx.mlx_array_new();
pub const mlx = @cImport({
    @cInclude("mlx/c/mlx.h");
});
