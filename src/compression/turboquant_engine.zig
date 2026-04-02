//! TurboQuant Engine Integration
//!
//! Wrapper around botirk38/turboquant that provides:
//! - Engine caching per dimension (amortizes init cost)
//! - Compression/decompression for MLX-compatible buffers
//! - Thread-safe engine access

const std = @import("std");
const turboquant = @import("turboquant");

/// Cached TurboQuant engine for a specific dimension
pub const CachedEngine = struct {
    dim: usize,
    engine: turboquant.Engine,
    refcount: std.atomic.Value(u32),

    pub fn init(allocator: std.mem.Allocator, dim: usize, seed: u32) !CachedEngine {
        const config = turboquant.EngineConfig{
            .dim = dim,
            .seed = seed,
        };
        const engine = try turboquant.Engine.init(allocator, config);

        return CachedEngine{
            .dim = dim,
            .engine = engine,
            .refcount = std.atomic.Value(u32).init(1),
        };
    }

    pub fn deinit(self: *CachedEngine, allocator: std.mem.Allocator) void {
        self.engine.deinit(allocator);
    }
};

/// Manages TurboQuant engines with dimension-based caching
pub const TurboQuantEngine = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    engines: std.AutoHashMap(usize, *CachedEngine),
    mutex: std.Thread.Mutex,
    seed: u32,

    pub fn init(allocator: std.mem.Allocator, seed: u32) Self {
        return Self{
            .allocator = allocator,
            .engines = std.AutoHashMap(usize, *CachedEngine).init(allocator),
            .mutex = std.Thread.Mutex{},
            .seed = seed,
        };
    }

    pub fn deinit(self: *Self) void {
        // Clean up all cached engines
        var it = self.engines.valueIterator();
        while (it.next()) |engine_ptr| {
            engine_ptr.*.deinit(self.allocator);
            self.allocator.destroy(engine_ptr.*);
        }
        self.engines.deinit();
    }

    /// Get or create engine for given dimension
    pub fn getEngineForDimension(self: *Self, dim: usize) !*CachedEngine {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Check if engine exists
        if (self.engines.get(dim)) |engine| {
            _ = engine.refcount.fetchAdd(1, .monotonic);
            return engine;
        }

        // Create new engine
        const engine = try self.allocator.create(CachedEngine);
        engine.* = try CachedEngine.init(self.allocator, dim, self.seed);

        try self.engines.put(dim, engine);
        return engine;
    }

    /// Compress f32 data using TurboQuant
    ///
    /// Parameters:
    ///   - data: Input f32 slice
    ///   - dim: Dimension for engine selection
    ///
    /// Returns: Allocated compressed bytes (caller must free)
    pub fn compress(self: *Self, data: []const f32, dim: usize) ![]u8 {
        const engine = try self.getEngineForDimension(dim);
        defer _ = engine.refcount.fetchSub(1, .monotonic);

        return try engine.engine.encode(self.allocator, data);
    }

    /// Decompress TurboQuant data back to f32
    ///
    /// Parameters:
    ///   - compressed: Compressed bytes from compress()
    ///   - dim: Dimension for engine selection
    ///
    /// Returns: Allocated f32 slice (caller must free)
    pub fn decompress(self: *Self, compressed: []const u8, dim: usize) ![]f32 {
        const engine = try self.getEngineForDimension(dim);
        defer _ = engine.refcount.fetchSub(1, .monotonic);

        return try engine.engine.decode(self.allocator, compressed);
    }

    /// Fast dot product without full decode (for query scoring)
    ///
    /// Parameters:
    ///   - query: Query vector (f32)
    ///   - compressed: Compressed data
    ///   - dim: Dimension for engine selection
    ///
    /// Returns: Dot product score
    pub fn dot(self: *Self, query: []const f32, compressed: []const u8, dim: usize) !f32 {
        const engine = try self.getEngineForDimension(dim);
        defer _ = engine.refcount.fetchSub(1, .monotonic);

        return engine.engine.dot(query, compressed);
    }

    /// Get compression ratio for a dimension
    pub fn getCompressionRatio(self: *Self, dim: usize) f32 {
        // TurboQuant uses 3 bits/dim = ~6x compression
        // Plus header overhead, so ~5.5-6x in practice
        _ = self;
        _ = dim;
        return 5.5; // Conservative estimate
    }
};

/// Layer compression result (matches kv_compressor.CompressionResult structure)
pub const LayerCompressionResult = struct {
    k_compressed: []u8,
    v_compressed: []u8,
    k_shape: []i64,
    v_shape: []i64,
    dim: usize,

    pub fn deinit(self: *LayerCompressionResult, allocator: std.mem.Allocator) void {
        allocator.free(self.k_compressed);
        allocator.free(self.v_compressed);
        allocator.free(self.k_shape);
        allocator.free(self.v_shape);
    }
};

/// Compress a full layer's K and V tensors
///
/// This is the high-level API that the KvCompressor will call.
pub fn compressLayer(
    engine: *TurboQuantEngine,
    k_data: []const f32,
    v_data: []const f32,
    k_shape: []const i64,
    v_shape: []const i64,
    dim: usize,
    allocator: std.mem.Allocator,
) !LayerCompressionResult {
    // k_shape and v_shape are preserved in result for reconstruction
    // The actual dimension used is 'dim' for the engine

    // Compress K and V separately
    const k_compressed = try engine.compress(k_data, dim);
    errdefer allocator.free(k_compressed);

    const v_compressed = try engine.compress(v_data, dim);
    errdefer allocator.free(v_compressed);

    // Copy shapes
    const k_shape_copy = try allocator.dupe(i64, k_shape);
    errdefer allocator.free(k_shape_copy);

    const v_shape_copy = try allocator.dupe(i64, v_shape);

    return LayerCompressionResult{
        .k_compressed = k_compressed,
        .v_compressed = v_compressed,
        .k_shape = k_shape_copy,
        .v_shape = v_shape_copy,
        .dim = dim,
    };
}

/// Decompress a full layer
pub fn decompressLayer(
    engine: *TurboQuantEngine,
    compressed: LayerCompressionResult,
    allocator: std.mem.Allocator,
) !struct { k: []f32, v: []f32 } {
    const k_data = try engine.decompress(compressed.k_compressed, compressed.dim);
    errdefer allocator.free(k_data);

    const v_data = try engine.decompress(compressed.v_compressed, compressed.dim);

    return .{ .k = k_data, .v = v_data };
}

// =============================================================================
// Tests
// =============================================================================

test "CachedEngine init/deinit" {
    const allocator = std.testing.allocator;

    var engine = try CachedEngine.init(allocator, 1024, 42);
    defer engine.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 1024), engine.dim);
    try std.testing.expectEqual(@as(u32, 1), engine.refcount.load(.monotonic));
}

test "TurboQuantEngine engine caching" {
    const allocator = std.testing.allocator;

    var manager = TurboQuantEngine.init(allocator, 42);
    defer manager.deinit();

    // Get engine for dimension 1024
    const engine1 = try manager.getEngineForDimension(1024);
    try std.testing.expectEqual(@as(u32, 1), engine1.refcount.load(.monotonic));

    // Get same engine again - should return cached instance
    const engine2 = try manager.getEngineForDimension(1024);
    try std.testing.expectEqual(engine1, engine2); // Same pointer
    try std.testing.expectEqual(@as(u32, 2), engine2.refcount.load(.monotonic));

    // Release both references
    _ = engine1.refcount.fetchSub(1, .monotonic);
    _ = engine2.refcount.fetchSub(1, .monotonic);
}

test "TurboQuantEngine compress/decompress round-trip" {
    const allocator = std.testing.allocator;

    var manager = TurboQuantEngine.init(allocator, 42);
    defer manager.deinit();

    // Create test data (must match engine dimension)
    const dim = 1024;
    const data = try allocator.alloc(f32, dim);
    defer allocator.free(data);
    for (0..dim) |i| {
        data[i] = @floatFromInt(i % 100);
    }

    // Compress
    const compressed = try manager.compress(data, dim);
    defer allocator.free(compressed);

    // Verify compression ratio (should be significantly smaller)
    const original_size = dim * @sizeOf(f32);
    try std.testing.expect(compressed.len < original_size);

    // Decompress
    const decompressed = try manager.decompress(compressed, dim);
    defer allocator.free(decompressed);

    // Verify dimension preserved
    try std.testing.expectEqual(dim, decompressed.len);
}

test "compressLayer/decompressLayer round-trip" {
    const allocator = std.testing.allocator;

    var engine = TurboQuantEngine.init(allocator, 42);
    defer engine.deinit();

    const dim = 1024;

    // Create test K and V data
    const k_data = try allocator.alloc(f32, dim);
    defer allocator.free(k_data);
    const v_data = try allocator.alloc(f32, dim);
    defer allocator.free(v_data);

    for (0..dim) |i| {
        k_data[i] = @floatFromInt(i % 50);
        v_data[i] = @floatFromInt(i % 50 + 25);
    }

    const k_shape = &[_]i64{@intCast(dim)};
    const v_shape = &[_]i64{@intCast(dim)};

    // Compress layer
    var compressed = try compressLayer(&engine, k_data, v_data, k_shape, v_shape, dim, allocator);
    defer compressed.deinit(allocator);

    // Verify both K and V were compressed
    try std.testing.expect(compressed.k_compressed.len > 0);
    try std.testing.expect(compressed.v_compressed.len > 0);
    try std.testing.expectEqual(dim, compressed.dim);

    // Decompress layer
    const decoded = try decompressLayer(&engine, compressed, allocator);
    defer allocator.free(decoded.k);
    defer allocator.free(decoded.v);

    // Verify dimensions
    try std.testing.expectEqual(dim, decoded.k.len);
    try std.testing.expectEqual(dim, decoded.v.len);
}

test "getCompressionRatio returns expected value" {
    const allocator = std.testing.allocator;

    var engine = TurboQuantEngine.init(allocator, 42);
    defer engine.deinit();

    const ratio = engine.getCompressionRatio(1024);
    try std.testing.expectApproxEqAbs(@as(f32, 5.5), ratio, 0.1);
}
