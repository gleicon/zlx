//! Generic KV Cache Compression Interface
//!
//! Provides a pluggable compression framework for KV cache optimization.
//! Supports multiple compression backends through a common interface.

const std = @import("std");
const mlx = @import("../mlx.zig/src/mlx.zig");
const turboquant_engine = @import("turboquant_engine.zig");
const mlx_bridge = @import("../mlx_bridge.zig");

/// Compression backend types
pub const CompressionType = enum {
    NoOp, // No compression (default)
    TurboQuant, // TurboQuant compression via botirk38/turboquant
    FutureMethod, // Placeholder for future compression methods
};

/// Compression configuration parameters
pub const CompressionConfig = struct {
    compression_type: CompressionType = .NoOp,
    bits: u4 = 4, // Quantization bits (3-4 for TurboQuant)
    adaptive_layers: u8 = 4, // First/last N layers kept in FP16
    enabled: bool = false, // Master enable flag
};

/// Result of compression operation
pub const CompressionResult = struct {
    k_compressed: []const u8, // Compressed key tensor data
    v_compressed: []const u8, // Compressed value tensor data
    k_metadata: KMetadata, // Key compression metadata
    v_metadata: VMetadata, // Value compression metadata

    pub const KMetadata = struct {
        original_shape: []const i64,
        compressed_bits: u4,
        algorithm: CompressionType,
    };

    pub const VMetadata = struct {
        original_shape: []const i64,
        compressed_bits: u4,
        algorithm: CompressionType,
    };
};

/// Statistics tracked by the compressor
pub const CompressionStats = struct {
    bytes_compressed: u64 = 0,
    bytes_original: u64 = 0,
    compress_calls: u64 = 0,
    layers_compressed: u64 = 0,
    layers_total: u64 = 0,
};

/// Generic KV cache compressor interface
///
/// This interface abstracts different compression backends.
/// Use NoOp for no compression (default), or TurboQuant for real compression.
pub const KvCompressor = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    config: CompressionConfig,
    backend: BackendUnion,
    backend_type: CompressionType,
    stats: CompressionStats,

    const BackendUnion = union {
        noop: void,
        turboquant: *turboquant_engine.TurboQuantEngine,
    };

    /// Initialize a new KV cache compressor
    ///
    /// Parameters:
    ///   - allocator: Memory allocator for the compressor
    ///   - config: Compression configuration
    ///
    /// Returns: Initialized compressor or error
    pub fn init(allocator: std.mem.Allocator, config: CompressionConfig) !Self {
        // NoOp backend - no compression applied
        if (config.compression_type == .NoOp or !config.enabled) {
            return Self{
                .allocator = allocator,
                .config = config,
                .backend = .{ .noop = {} },
                .backend_type = .NoOp,
                .stats = .{},
            };
        }

        // TurboQuant backend - real implementation
        if (config.compression_type == .TurboQuant) {
            // Create TurboQuant engine
            const engine = try allocator.create(turboquant_engine.TurboQuantEngine);
            engine.* = turboquant_engine.TurboQuantEngine.init(allocator, 42); // seed=42

            return Self{
                .allocator = allocator,
                .config = config,
                .backend = .{ .turboquant = engine },
                .backend_type = .TurboQuant,
                .stats = .{},
            };
        }

        // Future methods - not yet implemented
        return error.NotImplemented;
    }

    /// Deinitialize the compressor and free resources
    pub fn deinit(self: *Self) void {
        switch (self.backend_type) {
            .NoOp => {},
            .TurboQuant => {
                if (self.backend.turboquant) |engine| {
                    engine.deinit();
                    self.allocator.destroy(engine);
                }
            },
            .FutureMethod => {},
        }
    }

    /// Check if a layer should be compressed based on adaptive settings
    fn isLayerCompressed(self: *Self, layer_idx: usize, total_layers: usize) bool {
        if (!self.config.enabled) return false;
        if (self.config.compression_type == .NoOp) return false;

        const adaptive = self.config.adaptive_layers;
        if (adaptive == 0) return true; // Compress all layers

        // Don't compress first 'adaptive' layers
        if (layer_idx < adaptive) return false;

        // Don't compress last 'adaptive' layers
        if (layer_idx >= total_layers - adaptive) return false;

        return true;
    }

    /// Compress KV cache tensors
    ///
    /// Parameters:
    ///   - k: Key tensor (modified in place for NoOp, would compress for others)
    ///   - v: Value tensor (modified in place for NoOp, would compress for others)
    ///   - layer_idx: Layer index for adaptive compression
    ///   - total_layers: Total number of layers in model
    ///
    /// Returns: CompressionResult with compressed data and metadata
    pub fn compress(
        self: *Self,
        k: *mlx.Array,
        v: *mlx.Array,
        layer_idx: usize,
        total_layers: usize,
    ) !CompressionResult {
        switch (self.backend_type) {
            .NoOp => {
                // NoOp - parameters intentionally unused
                return CompressionResult{
                    .k_compressed = &.{},
                    .v_compressed = &.{},
                    .k_metadata = .{
                        .original_shape = &.{},
                        .compressed_bits = 16,
                        .algorithm = .NoOp,
                    },
                    .v_metadata = .{
                        .original_shape = &.{},
                        .compressed_bits = 16,
                        .algorithm = .NoOp,
                    },
                };
            },
            .TurboQuant => {
                // Check adaptive layer logic
                if (!self.isLayerCompressed(layer_idx, total_layers)) {
                    // Return NoOp result for adaptive layers (uncompressed)
                    return CompressionResult{
                        .k_compressed = &.{},
                        .v_compressed = &.{},
                        .k_metadata = .{
                            .original_shape = &.{},
                            .compressed_bits = 16,
                            .algorithm = .NoOp,
                        },
                        .v_metadata = .{
                            .original_shape = &.{},
                            .compressed_bits = 16,
                            .algorithm = .NoOp,
                        },
                    };
                }

                const engine = self.backend.turboquant;

                // Get array shapes
                const k_shape = try mlx_bridge.arrayShapeSlice(k.*, self.allocator);
                defer self.allocator.free(k_shape);

                const v_shape = try mlx_bridge.arrayShapeSlice(v.*, self.allocator);
                defer self.allocator.free(v_shape);

                // Allocate temp buffers
                const k_numel = mlx_bridge.arrayNumel(k.*);
                const v_numel = mlx_bridge.arrayNumel(v.*);

                var k_buffer = try mlx_bridge.MlxBuffer.alloc(self.allocator, k_numel);
                defer k_buffer.free();

                var v_buffer = try mlx_bridge.MlxBuffer.alloc(self.allocator, v_numel);
                defer v_buffer.free();

                // Convert MLX arrays to f32
                _ = try mlx_bridge.arrayToF32(k.*, k_buffer.data);
                _ = try mlx_bridge.arrayToF32(v.*, v_buffer.data);

                // Get dimension (typically last dimension of K)
                const dim = @as(usize, @intCast(k_shape[k_shape.len - 1]));

                // Compress via TurboQuant
                const layer_result = try turboquant_engine.compressLayer(
                    engine,
                    k_buffer.data,
                    v_buffer.data,
                    k_shape,
                    v_shape,
                    dim,
                    self.allocator,
                );

                // Track stats
                self.stats.compress_calls += 1;
                self.stats.bytes_original += k_numel * 4 + v_numel * 4; // f32 = 4 bytes
                self.stats.bytes_compressed += layer_result.k_compressed.len + layer_result.v_compressed.len;
                self.stats.layers_compressed += 1;

                // Return result with allocated slices (caller owns memory)
                return CompressionResult{
                    .k_compressed = layer_result.k_compressed,
                    .v_compressed = layer_result.v_compressed,
                    .k_metadata = .{
                        .original_shape = layer_result.k_shape,
                        .compressed_bits = self.config.bits,
                        .algorithm = .TurboQuant,
                    },
                    .v_metadata = .{
                        .original_shape = layer_result.v_shape,
                        .compressed_bits = self.config.bits,
                        .algorithm = .TurboQuant,
                    },
                };
            },
            .FutureMethod => return error.NotImplemented,
        }
    }

    /// Decompress KV cache tensors
    ///
    /// Parameters:
    ///   - compressed: CompressionResult from compress()
    ///   - k: Output key tensor (restored from compressed data)
    ///   - v: Output value tensor (restored from compressed data)
    pub fn decompress(self: *Self, compressed: CompressionResult, k: *mlx.Array, v: *mlx.Array) !void {
        switch (self.backend_type) {
            .NoOp => {
                // NoOp - parameters intentionally unused
                return;
            },
            .TurboQuant => {
                if (compressed.k_metadata.algorithm == .NoOp) {
                    // Adaptive layer - no decompression needed
                    return;
                }

                const engine = self.backend.turboquant;

                // Get dimension from shape
                const dim = @as(usize, @intCast(compressed.k_metadata.original_shape[
                    compressed.k_metadata.original_shape.len - 1
                ]));

                // Reconstruct layer result
                const layer_result = turboquant_engine.LayerCompressionResult{
                    .k_compressed = compressed.k_compressed,
                    .v_compressed = compressed.v_compressed,
                    .k_shape = compressed.k_metadata.original_shape,
                    .v_shape = compressed.v_metadata.original_shape,
                    .dim = dim,
                };

                // Decompress
                const decoded = try turboquant_engine.decompressLayer(engine, layer_result, self.allocator);
                defer self.allocator.free(decoded.k);
                defer self.allocator.free(decoded.v);

                // Convert back to MLX arrays
                k.* = try mlx_bridge.f32ToArray(decoded.k, compressed.k_metadata.original_shape, undefined);
                v.* = try mlx_bridge.f32ToArray(decoded.v, compressed.v_metadata.original_shape, undefined);
            },
            .FutureMethod => return error.NotImplemented,
        }
    }

    /// Check if compression is enabled
    pub fn isEnabled(self: *Self) bool {
        return self.config.enabled and self.backend_type != .NoOp;
    }

    /// Get the compression type
    pub fn getType(self: *Self) CompressionType {
        return self.backend_type;
    }

    /// Get compression ratio estimate
    ///
    /// Returns estimated compression ratio for this configuration.
    /// NoOp returns 1.0 (no compression).
    /// TurboQuant returns ~5.5x compression.
    pub fn getCompressionRatio(self: *Self) f32 {
        return switch (self.backend_type) {
            .NoOp => 1.0,
            .TurboQuant => 5.5, // TurboQuant achieves ~5.5x compression
            .FutureMethod => 1.0,
        };
    }

    /// Get compression statistics
    pub fn getStats(self: *Self) CompressionStats {
        return self.stats;
    }
};

// =============================================================================
// Tests
// =============================================================================

test "KvCompressor.init creates NoOp compressor" {
    const allocator = std.testing.allocator;

    const config = CompressionConfig{
        .compression_type = .NoOp,
        .enabled = false,
    };

    var compressor = try KvCompressor.init(allocator, config);
    defer compressor.deinit();

    try std.testing.expectEqual(CompressionType.NoOp, compressor.getType());
    try std.testing.expect(!compressor.isEnabled());
    try std.testing.expectEqual(@as(f32, 1.0), compressor.getCompressionRatio());
}

test "KvCompressor.init creates TurboQuant compressor" {
    const allocator = std.testing.allocator;

    const config = CompressionConfig{
        .compression_type = .TurboQuant,
        .enabled = true,
    };

    var compressor = try KvCompressor.init(allocator, config);
    defer compressor.deinit();

    try std.testing.expectEqual(CompressionType.TurboQuant, compressor.getType());
    try std.testing.expect(compressor.isEnabled());
    try std.testing.expectApproxEqAbs(@as(f32, 5.5), compressor.getCompressionRatio(), 0.1);
}

test "KvCompressor.compress returns original data with NoOp" {
    const allocator = std.testing.allocator;

    const config = CompressionConfig{
        .compression_type = .NoOp,
        .enabled = false,
    };

    var compressor = try KvCompressor.init(allocator, config);
    defer compressor.deinit();

    var k: mlx.Array = undefined;
    var v: mlx.Array = undefined;

    const result = try compressor.compress(&k, &v, 0, 24);

    // NoOp should return empty compressed data
    try std.testing.expectEqual(@as(usize, 0), result.k_compressed.len);
    try std.testing.expectEqual(@as(usize, 0), result.v_compressed.len);
    try std.testing.expectEqual(CompressionType.NoOp, result.k_metadata.algorithm);
    try std.testing.expectEqual(CompressionType.NoOp, result.v_metadata.algorithm);
}

test "KvCompressor.decompress with NoOp does nothing" {
    const allocator = std.testing.allocator;

    const config = CompressionConfig{
        .compression_type = .NoOp,
        .enabled = false,
    };

    var compressor = try KvCompressor.init(allocator, config);
    defer compressor.deinit();

    const result = CompressionResult{
        .k_compressed = &.{},
        .v_compressed = &.{},
        .k_metadata = .{
            .original_shape = &.{},
            .compressed_bits = 16,
            .algorithm = .NoOp,
        },
        .v_metadata = .{
            .original_shape = &.{},
            .compressed_bits = 16,
            .algorithm = .NoOp,
        },
    };

    var k: mlx.Array = undefined;
    var v: mlx.Array = undefined;

    // Should not error
    try compressor.decompress(result, &k, &v);
}

test "CompressionType enum includes all variants" {
    try std.testing.expectEqual(@as(usize, 3), @typeInfo(CompressionType).Enum.fields.len);
    try std.testing.expectEqualStrings("NoOp", @tagName(CompressionType.NoOp));
    try std.testing.expectEqualStrings("TurboQuant", @tagName(CompressionType.TurboQuant));
    try std.testing.expectEqualStrings("FutureMethod", @tagName(CompressionType.FutureMethod));
}

test "KvCompressor.deinit frees resources" {
    const allocator = std.testing.allocator;

    const config = CompressionConfig{
        .compression_type = .NoOp,
        .enabled = false,
    };

    var compressor = try KvCompressor.init(allocator, config);
    compressor.deinit();

    // Test passes if no memory leaks
}

test "TurboQuant compressor init/deinit" {
    const allocator = std.testing.allocator;

    const config = CompressionConfig{
        .compression_type = .TurboQuant,
        .enabled = true,
    };

    var compressor = try KvCompressor.init(allocator, config);
    defer compressor.deinit();

    try std.testing.expect(compressor.isEnabled());
    try std.testing.expectEqual(CompressionType.TurboQuant, compressor.getType());
}

test "CompressionConfig defaults" {
    const config = CompressionConfig{};

    try std.testing.expectEqual(CompressionType.NoOp, config.compression_type);
    try std.testing.expectEqual(@as(u4, 4), config.bits);
    try std.testing.expectEqual(@as(u8, 4), config.adaptive_layers);
    try std.testing.expect(!config.enabled);
}

test "KvCompressor.getCompressionRatio for TurboQuant" {
    const allocator = std.testing.allocator;

    const config = CompressionConfig{
        .compression_type = .TurboQuant,
        .enabled = true,
    };

    var compressor = try KvCompressor.init(allocator, config);
    defer compressor.deinit();

    const ratio = compressor.getCompressionRatio();
    try std.testing.expectApproxEqAbs(@as(f32, 5.5), ratio, 0.1);
}

test "isLayerCompressed respects adaptive settings" {
    const allocator = std.testing.allocator;

    const config = CompressionConfig{
        .compression_type = .TurboQuant,
        .enabled = true,
        .adaptive_layers = 2,
    };

    var compressor = try KvCompressor.init(allocator, config);
    defer compressor.deinit();

    // With 24 total layers and adaptive=2:
    // - Layers 0, 1 (first 2) should NOT be compressed
    // - Layers 22, 23 (last 2) should NOT be compressed
    // - Layers 2-21 should be compressed

    try std.testing.expect(!compressor.isLayerCompressed(0, 24)); // First layer
    try std.testing.expect(!compressor.isLayerCompressed(1, 24)); // Second layer
    try std.testing.expect(compressor.isLayerCompressed(2, 24)); // Third layer
    try std.testing.expect(compressor.isLayerCompressed(10, 24)); // Middle layer
    try std.testing.expect(compressor.isLayerCompressed(21, 24)); // Layer before last 2
    try std.testing.expect(!compressor.isLayerCompressed(22, 24)); // Second to last
    try std.testing.expect(!compressor.isLayerCompressed(23, 24)); // Last layer
}

test "isLayerCompressed with adaptive=0 compresses all" {
    const allocator = std.testing.allocator;

    const config = CompressionConfig{
        .compression_type = .TurboQuant,
        .enabled = true,
        .adaptive_layers = 0,
    };

    var compressor = try KvCompressor.init(allocator, config);
    defer compressor.deinit();

    // All layers should be compressed when adaptive=0
    try std.testing.expect(compressor.isLayerCompressed(0, 24));
    try std.testing.expect(compressor.isLayerCompressed(23, 24));
}
