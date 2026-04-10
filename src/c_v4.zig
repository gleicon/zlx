// src/c_v4.zig
// Separate @cImport boundary for mlx-c v0.4.x to avoid conflicts with v0.1.2
// This imports the Fast Custom Ops API available in v0.4.x
pub const mlx_v4 = @cImport({
    @cDefine("MLX_C_VERSION", "0.4.1");
    @cInclude("mlx/c/mlx.h");
});
