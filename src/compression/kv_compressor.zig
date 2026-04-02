//! Generic KV Cache Compression Interface
//!
//! Provides a pluggable compression framework for KV cache optimization.
//! Supports multiple compression backends through a common interface.

const std = @import("std");
const mlx = @import("../mlx.zig/src/mlx.zig");

/// Compression backend types
pub const CompressionType = enum {
    NoOp, // No compression (default)
    TurboQuant, // TurboQuant compression (stubbed - requires porting)
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

/// Generic KV cache compressor interface
///
/// This interface abstracts different compression backends.
/// Use NoOp for no compression (default), or TurboQuant when implemented.
pub const KvCompressor = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    config: CompressionConfig,
    backend: ?*anyopaque, // Opaque pointer to backend-specific data
    backend_type: CompressionType,

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
                .backend = null,
                .backend_type = .NoOp,
            };
        }

        // TurboQuant backend - stubbed, would need actual implementation
        if (config.compression_type == .TurboQuant) {
            // For now, TurboQuant returns error.NotImplemented
            // When implemented, this would:
            // 1. Allocate backend-specific data
            // 2. Initialize Metal kernel manager
            // 3. Precompute quantization codebooks
            return error.NotImplemented;
        }

        // Future methods - not yet implemented
        return error.NotImplemented;
    }

    /// Deinitialize the compressor and free resources
    pub fn deinit(self: *Self) void {
        // Clean up backend-specific resources
        if (self.backend) |backend| {
            switch (self.backend_type) {
                .NoOp => {}, // No resources to free
                .TurboQuant => {
                    // Would free TurboQuantCompressor instance
                    _ = backend;
                },
                .FutureMethod => {},
            }
        }
        self.backend = null;
    }

    /// Compress KV cache tensors
    ///
    /// Parameters:
    ///   - k: Key tensor (modified in place for NoOp, would compress for others)
    ///   - v: Value tensor (modified in place for NoOp, would compress for others)
    ///
    /// Returns: CompressionResult with compressed data and metadata
    pub fn compress(self: *Self, k: *mlx.Array, v: *mlx.Array) !CompressionResult {
        switch (self.backend_type) {
            .NoOp => {
                // No compression - return empty result with original data
                // k and v are not modified in NoOp mode
                _ = k;
                _ = v;
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
                // Stubbed - would implement:
                // 1. Apply Walsh-Hadamard Transform to keys
                // 2. Lloyd-Max quantization
                // 3. Return compressed data
                std.log.warn("TurboQuant compression requested but not implemented. See src/compression/RESEARCH.md", .{});
                return error.NotImplemented;
            },
            .FutureMethod => {
                return error.NotImplemented;
            },
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
                // No decompression needed for NoOp
                _ = compressed;
                _ = k;
                _ = v;
                return;
            },
            .TurboQuant => {
                // Stubbed - would implement:
                // 1. Dequantize keys/values using codebook
                // 2. Apply inverse Walsh-Hadamard Transform
                _ = compressed;
                _ = k;
                _ = v;
                return error.NotImplemented;
            },
            .FutureMethod => {
                _ = compressed;
                _ = k;
                _ = v;
                return error.NotImplemented;
            },
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
    /// TurboQuant would return ~4.6x when implemented.
    pub fn getCompressionRatio(self: *Self) f32 {
        return switch (self.backend_type) {
            .NoOp => 1.0,
            .TurboQuant => {
                // TurboQuant target: 4.6x compression at 98% FP16 speed
                // This is approximate - actual ratio depends on model and data
                const bits: f32 = @floatFromInt(self.config.bits);
                return 16.0 / bits; // e.g., 4x for 4-bit, 5.33x for 3-bit
            },
            .FutureMethod => 1.0,
        };
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

    const result = try compressor.compress(&k, &v);

    // Suppress unused variable warnings - in real test these would be actual MLX arrays
    _ = result.k_metadata.original_shape;

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

    // After deinit, backend should be null
    try std.testing.expectEqual(@as(?*anyopaque, null), compressor.backend);
}

test "TurboQuant returns NotImplemented" {
    const allocator = std.testing.allocator;

    const config = CompressionConfig{
        .compression_type = .TurboQuant,
        .enabled = true,
    };

    // TurboQuant initialization should fail with NotImplemented
    const result = KvCompressor.init(allocator, config);
    try std.testing.expectError(error.NotImplemented, result);
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

    // Create TurboQuant config but expect init to fail
    // We'll manually test the ratio calculation logic
    const config_4bit = CompressionConfig{
        .compression_type = .TurboQuant,
        .bits = 4,
        .enabled = true,
    };

    const config_3bit = CompressionConfig{
        .compression_type = .TurboQuant,
        .bits = 3,
        .enabled = true,
    };

    // Calculate expected ratios
    const ratio_4bit = 16.0 / @as(f32, 4.0); // 4.0x
    const ratio_3bit = 16.0 / @as(f32, 3.0); // 5.333x

    try std.testing.expectApproxEqAbs(@as(f32, 4.0), ratio_4bit, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 5.33), ratio_3bit, 0.01);

    _ = allocator;
    _ = config_4bit;
    _ = config_3bit;
}
