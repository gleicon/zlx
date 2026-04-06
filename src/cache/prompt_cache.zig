//! prompt_cache.zig - Prompt caching for fast repeated inference
//!
//! Implements LRU-based prompt caching to achieve sub-second TTFT for repeated prompts.
//! Caches KV cache state to disk and restores it on cache hits.

const std = @import("std");

/// Opaque pointer to cached KV state (actual type depends on MLX implementation)
pub const CachedKvState = opaque {};

/// Unique cache key generated from model + prompt + sampling params
pub const CacheKey = struct {
    /// Hex-encoded SHA256 hash: {model_hash}_{prompt_hash}_{params_hash}
    /// Each SHA256 hash is 32 bytes = 64 hex chars. Total: 64*3 + 2 underscores = 194
    key_str: [194]u8,
    key_len: u8,

    const KEY_LEN: usize = 194; // 64 + 1 + 64 + 1 + 64 = 194

    pub fn init(
        allocator: std.mem.Allocator,
        model_name: []const u8,
        model_hash: []const u8,
        prompt: []const u8,
        params: anytype, // GenerationOptions
    ) !CacheKey {
        var key: CacheKey = undefined;

        // Compute model hash portion (SHA256 of model identifier)
        var model_hash_buf: [32]u8 = undefined;
        var model_hasher = std.crypto.hash.sha2.Sha256.init(.{});
        model_hasher.update(model_name);
        model_hasher.update(model_hash);
        model_hasher.final(&model_hash_buf);

        // Compute prompt hash portion
        var prompt_hash_buf: [32]u8 = undefined;
        var prompt_hasher = std.crypto.hash.sha2.Sha256.init(.{});
        prompt_hasher.update(prompt);
        prompt_hasher.final(&prompt_hash_buf);

        // Compute params hash portion
        var params_hash_buf: [32]u8 = undefined;
        var params_hasher = std.crypto.hash.sha2.Sha256.init(.{});
        // Serialize sampling parameters
        const temp_bytes = std.mem.toBytes(params.temperature);
        const top_p_bytes = std.mem.toBytes(params.top_p);
        const top_k_bytes = std.mem.toBytes(params.top_k);
        const min_p_bytes = std.mem.toBytes(params.min_p);
        const presence_bytes = std.mem.toBytes(params.presence_penalty);
        const frequency_bytes = std.mem.toBytes(params.frequency_penalty);
        const repetition_bytes = std.mem.toBytes(params.repetition_penalty);

        params_hasher.update(&temp_bytes);
        params_hasher.update(&top_p_bytes);
        params_hasher.update(&top_k_bytes);
        params_hasher.update(&min_p_bytes);
        params_hasher.update(&presence_bytes);
        params_hasher.update(&frequency_bytes);
        params_hasher.update(&repetition_bytes);
        params_hasher.final(&params_hash_buf);

        // Format as hex string with underscores
        var pos: usize = 0;
        pos += try formatHex(&key.key_str, pos, &model_hash_buf);
        key.key_str[pos] = '_';
        pos += 1;
        pos += try formatHex(&key.key_str, pos, &prompt_hash_buf);
        key.key_str[pos] = '_';
        pos += 1;
        pos += try formatHex(&key.key_str, pos, &params_hash_buf);

        key.key_len = @intCast(pos);
        _ = allocator; // Used for future extensibility
        return key;
    }

    pub fn slice(self: *const CacheKey) []const u8 {
        return self.key_str[0..self.key_len];
    }

    fn formatHex(buf: []u8, pos: usize, bytes: []const u8) !usize {
        for (bytes, 0..) |byte, i| {
            _ = try std.fmt.bufPrint(buf[pos + i * 2 .. pos + i * 2 + 2], "{x:0>2}", .{byte});
        }
        return bytes.len * 2;
    }
};

/// Cached entry metadata
pub const CacheEntry = struct {
    key: CacheKey,
    file_path: []const u8,
    size_bytes: u64,
    created_at: i64,
    last_accessed: std.atomic.Value(i64),
    access_count: std.atomic.Value(u64),

    pub fn deinit(self: *CacheEntry, allocator: std.mem.Allocator) void {
        allocator.free(self.file_path);
    }
};

/// Cache metrics for monitoring
pub const CacheMetrics = struct {
    hits: u64,
    misses: u64,
    evictions: u64,
    total_size_bytes: u64,
    entries: u64,

    pub fn hitRate(self: CacheMetrics) f32 {
        const total = self.hits + self.misses;
        if (total == 0) return 0.0;
        return @as(f32, @floatFromInt(self.hits)) / @as(f32, @floatFromInt(total));
    }
};

/// LRU list node for eviction ordering
const LruNode = struct {
    key: []const u8,
    prev: ?*LruNode,
    next: ?*LruNode,
};

/// Prompt cache with LRU eviction
pub const PromptCache = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    cache_dir: []const u8,
    max_size_bytes: u64,
    current_size_bytes: std.atomic.Value(u64),

    // Hash map for O(1) lookup
    entries: std.StringHashMap(CacheEntry),

    // LRU tracking
    lru_head: ?*LruNode,
    lru_tail: ?*LruNode,
    lru_nodes: std.StringHashMap(*LruNode),

    // Thread safety
    mutex: std.Thread.Mutex,

    // Metrics
    hits: std.atomic.Value(u64),
    misses: std.atomic.Value(u64),
    evictions: std.atomic.Value(u64),

    pub fn init(allocator: std.mem.Allocator, cache_dir: []const u8, max_size_gb: u32) !Self {
        // Ensure cache directory exists
        try ensureCacheDir(cache_dir);

        const max_size_bytes = @as(u64, max_size_gb) * 1024 * 1024 * 1024;

        var self = Self{
            .allocator = allocator,
            .cache_dir = try allocator.dupe(u8, cache_dir),
            .max_size_bytes = max_size_bytes,
            .current_size_bytes = std.atomic.Value(u64).init(0),
            .entries = std.StringHashMap(CacheEntry).init(allocator),
            .lru_head = null,
            .lru_tail = null,
            .lru_nodes = std.StringHashMap(*LruNode).init(allocator),
            .mutex = .{},
            .hits = std.atomic.Value(u64).init(0),
            .misses = std.atomic.Value(u64).init(0),
            .evictions = std.atomic.Value(u64).init(0),
        };

        // Load existing cache entries from disk
        try self.loadIndex();

        return self;
    }

    pub fn deinit(self: *Self) void {
        // Save index before shutting down
        self.saveIndex() catch |err| {
            std.log.warn("Failed to save cache index: {s}", .{@errorName(err)});
        };

        // Free all entries
        var iter = self.entries.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.deinit(self.allocator);
        }
        self.entries.deinit();

        // Free LRU nodes
        var lru_iter = self.lru_nodes.iterator();
        while (lru_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.destroy(entry.value_ptr.*);
        }
        self.lru_nodes.deinit();

        self.allocator.free(self.cache_dir);
    }

    /// Generate a cache key from request parameters
    pub fn generateKey(
        self: *const Self,
        model_name: []const u8,
        model_hash: []const u8,
        prompt: []const u8,
        params: anytype,
    ) !CacheKey {
        _ = self;
        return try CacheKey.init(std.heap.page_allocator, model_name, model_hash, prompt, params);
    }

    /// Look up a cache entry. Returns the file path if found, null otherwise.
    /// On hit, updates LRU ordering and metrics.
    /// Caller is responsible for loading the actual MLX cache from the returned path.
    pub fn lookup(self: *Self, key: CacheKey) ?[]const u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        const key_str = key.slice();

        if (self.entries.getPtr(key_str)) |entry| {
            // Cache hit!
            _ = self.hits.fetchAdd(1, .monotonic);

            // Update access stats
            entry.last_accessed.store(std.time.timestamp(), .monotonic);
            _ = entry.access_count.fetchAdd(1, .monotonic);

            // Update LRU ordering
            self.moveToFront(key_str);

            // Return the file path for the caller to load
            return entry.file_path;
        }

        // Cache miss
        _ = self.misses.fetchAdd(1, .monotonic);
        return null;
    }

    /// Save a KV cache file path reference.
    /// The caller is responsible for saving the actual MLX cache data to disk first.
    pub fn save(self: *Self, key: CacheKey, file_path: []const u8, size_bytes: u64) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Check if we need to evict to make space
        try self.makeSpace(size_bytes);

        // Create entry
        const key_owned = try self.allocator.dupe(u8, key.slice());
        errdefer self.allocator.free(key_owned);

        const path_owned = try self.allocator.dupe(u8, file_path);
        errdefer self.allocator.free(path_owned);

        const entry = CacheEntry{
            .key = key,
            .file_path = path_owned,
            .size_bytes = size_bytes,
            .created_at = std.time.timestamp(),
            .last_accessed = std.atomic.Value(i64).init(std.time.timestamp()),
            .access_count = std.atomic.Value(u64).init(1),
        };

        // Store in hash map
        try self.entries.put(key_owned, entry);

        // Add to LRU list
        try self.addToLru(key_owned);

        // Update size
        _ = self.current_size_bytes.fetchAdd(size_bytes, .monotonic);

        std.log.debug("Cached prompt with key {s}, size: {d} MB", .{
            key.slice(),
            size_bytes / (1024 * 1024),
        });
    }

    /// Get the path where a cache entry should be stored
    pub fn getCacheFilePath(self: *Self, key: CacheKey) ![]const u8 {
        return try std.fs.path.join(self.allocator, &.{ self.cache_dir, "entries", key.slice() });
    }

    /// Get current metrics
    pub fn getMetrics(self: *Self) CacheMetrics {
        return .{
            .hits = self.hits.load(.monotonic),
            .misses = self.misses.load(.monotonic),
            .evictions = self.evictions.load(.monotonic),
            .total_size_bytes = self.current_size_bytes.load(.monotonic),
            .entries = self.entries.count(),
        };
    }

    // =========================================================================
    // Internal helpers
    // =========================================================================

    /// Make space in cache by evicting old entries
    fn makeSpace(self: *Self, needed_bytes: u64) !void {
        const current = self.current_size_bytes.load(.monotonic);
        if (current + needed_bytes <= self.max_size_bytes) {
            return; // Enough space
        }

        // Need to evict
        var to_evict: u64 = (current + needed_bytes) - self.max_size_bytes;

        while (to_evict > 0 and self.lru_tail != null) {
            const tail = self.lru_tail.?;
            const key_str = tail.key;

            if (self.entries.get(key_str)) |entry| {
                // Delete file
                std.fs.deleteFileAbsolute(entry.file_path) catch |err| {
                    std.log.warn("Failed to delete cache file {s}: {s}", .{ entry.file_path, @errorName(err) });
                };

                to_evict -= entry.size_bytes;
                _ = self.current_size_bytes.fetchSub(entry.size_bytes, .monotonic);
                _ = self.evictions.fetchAdd(1, .monotonic);

                std.log.debug("Evicted cache entry {s}, freed {d} MB", .{
                    key_str,
                    entry.size_bytes / (1024 * 1024),
                });
            }

            // Remove from all structures
            try self.removeFromLru(tail);
            if (self.entries.fetchRemove(key_str)) |kv| {
                kv.value.deinit(self.allocator);
                self.allocator.free(kv.key);
            }
        }
    }

    /// Add key to front of LRU list
    fn addToLru(self: *Self, key: []const u8) !void {
        const node = try self.allocator.create(LruNode);
        node.* = .{
            .key = key,
            .prev = null,
            .next = self.lru_head,
        };

        if (self.lru_head) |head| {
            head.prev = node;
        }
        self.lru_head = node;

        if (self.lru_tail == null) {
            self.lru_tail = node;
        }

        try self.lru_nodes.put(key, node);
    }

    /// Move key to front of LRU list
    fn moveToFront(self: *Self, key: []const u8) void {
        const node = self.lru_nodes.get(key) orelse return;

        if (node.prev == null) {
            return; // Already at front
        }

        // Remove from current position
        if (node.prev) |prev| {
            prev.next = node.next;
        }
        if (node.next) |next| {
            next.prev = node.prev;
        } else {
            self.lru_tail = node.prev;
        }

        // Add to front
        node.prev = null;
        node.next = self.lru_head;
        if (self.lru_head) |head| {
            head.prev = node;
        }
        self.lru_head = node;
    }

    /// Remove node from LRU list
    fn removeFromLru(self: *Self, node: *LruNode) !void {
        if (node.prev) |prev| {
            prev.next = node.next;
        } else {
            self.lru_head = node.next;
        }

        if (node.next) |next| {
            next.prev = node.prev;
        } else {
            self.lru_tail = node.prev;
        }

        _ = self.lru_nodes.remove(node.key);
        self.allocator.destroy(node);
    }

    /// Get size of file in bytes
    fn getFileSize(self: *const Self, path: []const u8) !u64 {
        _ = self;
        const file = try std.fs.openFileAbsolute(path, .{});
        defer file.close();
        const stat = try file.stat();
        return stat.size;
    }

    /// Get file path for a cache entry
    fn getEntryPath(self: *Self, key: CacheKey) ![]const u8 {
        return try std.fs.path.join(self.allocator, &.{ self.cache_dir, "entries", key.slice() });
    }

    /// Load cache index from disk
    // real index.json load per D-05 — called once at init, no per-request I/O
    fn loadIndex(self: *Self) !void {
        const index_path = try std.fs.path.join(self.allocator, &.{ self.cache_dir, "index.json" });
        defer self.allocator.free(index_path);

        // Check if index exists
        std.fs.accessAbsolute(index_path, .{}) catch {
            std.log.info("No existing cache index found, starting fresh", .{});
            return;
        };

        const file = try std.fs.openFileAbsolute(index_path, .{});
        defer file.close();

        const content = try file.readToEndAlloc(self.allocator, 1024 * 1024); // 1MB max
        defer self.allocator.free(content);

        const parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, content, .{});
        defer parsed.deinit();

        const root_obj = parsed.value.object;
        const entries_val = root_obj.get("entries") orelse return;
        const entries_arr = entries_val.array;

        try self.entries.ensureTotalCapacity(@intCast(entries_arr.items.len));

        for (entries_arr.items) |item| {
            const obj = item.object;
            const key_str = obj.get("key").?.string;
            const size_bytes: u64 = @intCast(obj.get("size_bytes").?.integer);
            const created_at: i64 = obj.get("created_at").?.integer;
            const access_count_val: u64 = @intCast(obj.get("access_count").?.integer);

            // Dupe key string before parsed.deinit() frees it
            const key_copy = try self.allocator.dupe(u8, key_str);
            errdefer self.allocator.free(key_copy);

            // file_path is not stored in index.json — default to empty string
            const file_path_copy = try self.allocator.dupe(u8, "");
            errdefer self.allocator.free(file_path_copy);

            const now = std.time.timestamp();
            const entry = CacheEntry{
                .key = undefined, // key is reconstructed from map key; not serialized
                .file_path = file_path_copy,
                .size_bytes = size_bytes,
                .created_at = created_at,
                .last_accessed = std.atomic.Value(i64).init(now),
                .access_count = std.atomic.Value(u64).init(access_count_val),
            };

            try self.entries.put(key_copy, entry);

            // Also add to LRU tracking so eviction works correctly
            try self.addToLru(key_copy);
        }

        std.log.info("Loaded {d} cache entries from {s}", .{ entries_arr.items.len, index_path });
    }

    /// Save cache index to disk
    fn saveIndex(self: *Self) !void {
        const index_path = try std.fs.path.join(self.allocator, &.{ self.cache_dir, "index.json" });
        defer self.allocator.free(index_path);

        const file = try std.fs.createFileAbsolute(index_path, .{});
        defer file.close();

        // Build JSON in memory first, then write
        var json = std.ArrayList(u8){};
        defer json.deinit(self.allocator);
        const writer = json.writer(self.allocator);

        try writer.writeAll("{\n  \"entries\": [\n");

        var iter = self.entries.iterator();
        var first = true;
        while (iter.next()) |entry| {
            if (!first) try writer.writeAll(",\n");
            first = false;

            try writer.print("    {{\n      \"key\": \"{s}\",\n", .{entry.key_ptr.*});
            try writer.print("      \"size_bytes\": {d},\n", .{entry.value_ptr.size_bytes});
            try writer.print("      \"created_at\": {d},\n", .{entry.value_ptr.created_at});
            try writer.print("      \"access_count\": {d}\n    }}", .{entry.value_ptr.access_count.load(.monotonic)});
        }

        try writer.writeAll("\n  ],\n");

        const metrics = self.getMetrics();
        try writer.print("  \"hits\": {d},\n", .{metrics.hits});
        try writer.print("  \"misses\": {d},\n", .{metrics.misses});
        try writer.print("  \"evictions\": {d},\n", .{metrics.evictions});
        try writer.print("  \"total_size_bytes\": {d}\n", .{metrics.total_size_bytes});
        try writer.writeAll("}\n");

        try file.writeAll(json.items);

        std.log.info("Saved cache index with {d} entries", .{self.entries.count()});
    }

    /// Ensure cache directory structure exists
    fn ensureCacheDir(cache_dir: []const u8) !void {
        try std.fs.cwd().makePath(cache_dir);
        const entries_dir = try std.fs.path.join(std.heap.page_allocator, &.{ cache_dir, "entries" });
        defer std.heap.page_allocator.free(entries_dir);
        try std.fs.cwd().makePath(entries_dir);
    }
};

/// Header for KV cache binary format
const KvCacheHeader = struct {
    const MAGIC: u32 = 0x4B56434B; // "KVCK"

    magic: u32,
    version: u32,
    num_layers: u32,
    prompt_len: u32,
    hidden_size: u32,
    dtype: u32,
};

// ============================================================================
// Global cache instance
// ============================================================================

var global_cache: ?*PromptCache = null;
var global_mutex: std.Thread.Mutex = .{};

/// Initialize global prompt cache
pub fn initGlobalCache(allocator: std.mem.Allocator, cache_dir: []const u8, max_size_gb: u32) !void {
    global_mutex.lock();
    defer global_mutex.unlock();

    if (global_cache != null) return;

    const cache = try allocator.create(PromptCache);
    cache.* = try PromptCache.init(allocator, cache_dir, max_size_gb);

    global_cache = cache;
    std.log.info("Prompt cache initialized at {s} (max: {d} GB)", .{ cache_dir, max_size_gb });
}

/// Deinitialize global prompt cache
pub fn deinitGlobalCache(allocator: std.mem.Allocator) void {
    global_mutex.lock();
    defer global_mutex.unlock();

    if (global_cache) |cache| {
        cache.deinit();
        allocator.destroy(cache);
        global_cache = null;
        std.log.info("Prompt cache deinitialized", .{});
    }
}

/// Get global prompt cache
pub fn getGlobalCache() ?*PromptCache {
    global_mutex.lock();
    defer global_mutex.unlock();
    return global_cache;
}

// ============================================================================
// Tests
// ============================================================================

test "CacheKey generates identical keys for same inputs" {
    const allocator = std.testing.allocator;

    const model_name = "qwen-7b";
    const model_hash = "abc123";
    const prompt = "Hello, world!";
    const params = .{
        .temperature = 0.7,
        .top_p = 0.9,
        .top_k = 50,
        .min_p = 0.0,
        .presence_penalty = 0.0,
        .frequency_penalty = 0.0,
        .repetition_penalty = 1.0,
    };

    const key1 = try CacheKey.init(allocator, model_name, model_hash, prompt, params);
    const key2 = try CacheKey.init(allocator, model_name, model_hash, prompt, params);

    try std.testing.expectEqualStrings(key1.slice(), key2.slice());
}

test "CacheKey generates different keys for different prompts" {
    const allocator = std.testing.allocator;

    const model_name = "qwen-7b";
    const model_hash = "abc123";
    const params = .{
        .temperature = 0.7,
        .top_p = 0.9,
        .top_k = 50,
        .min_p = 0.0,
        .presence_penalty = 0.0,
        .frequency_penalty = 0.0,
        .repetition_penalty = 1.0,
    };

    const key1 = try CacheKey.init(allocator, model_name, model_hash, "Hello!", params);
    const key2 = try CacheKey.init(allocator, model_name, model_hash, "Goodbye!", params);

    try std.testing.expect(!std.mem.eql(u8, key1.slice(), key2.slice()));
}

test "CacheKey generates different keys for different params" {
    const allocator = std.testing.allocator;

    const model_name = "qwen-7b";
    const model_hash = "abc123";
    const prompt = "Hello!";

    const params1 = .{
        .temperature = @as(f32, 0.7),
        .top_p = @as(f32, 0.9),
        .top_k = @as(u32, 50),
        .min_p = @as(f32, 0.0),
        .presence_penalty = @as(f32, 0.0),
        .frequency_penalty = @as(f32, 0.0),
        .repetition_penalty = @as(f32, 1.0),
    };

    const params2 = .{
        .temperature = @as(f32, 0.8), // Different temperature
        .top_p = @as(f32, 0.9),
        .top_k = @as(u32, 50),
        .min_p = @as(f32, 0.0),
        .presence_penalty = @as(f32, 0.0),
        .frequency_penalty = @as(f32, 0.0),
        .repetition_penalty = @as(f32, 1.0),
    };

    const key1 = try CacheKey.init(allocator, model_name, model_hash, prompt, params1);
    const key2 = try CacheKey.init(allocator, model_name, model_hash, prompt, params2);

    try std.testing.expect(!std.mem.eql(u8, key1.slice(), key2.slice()));
}

test "PromptCache metrics tracking" {
    const allocator = std.testing.allocator;

    // Use temporary directory for testing
    const test_dir = "/tmp/zlx_cache_test";
    try std.fs.cwd().makePath(test_dir);
    defer std.fs.deleteTreeAbsolute(test_dir) catch {};

    var cache = try PromptCache.init(allocator, test_dir, 1); // 1GB max
    defer cache.deinit();

    // Check initial metrics
    const metrics = cache.getMetrics();
    try std.testing.expectEqual(@as(u64, 0), metrics.hits);
    try std.testing.expectEqual(@as(u64, 0), metrics.misses);
    try std.testing.expectEqual(@as(f32, 0.0), metrics.hitRate());
}

test "PromptCache hit rate calculation" {
    // Create cache metrics manually
    const metrics = CacheMetrics{
        .hits = 75,
        .misses = 25,
        .evictions = 0,
        .total_size_bytes = 1000,
        .entries = 1,
    };

    try std.testing.expectApproxEqAbs(@as(f32, 0.75), metrics.hitRate(), 0.01);
}
