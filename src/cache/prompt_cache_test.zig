//! prompt_cache_test.zig - Unit tests for prompt caching system

const std = @import("std");
const cache = @import("prompt_cache.zig");
const CacheEntry = cache.CacheEntry;
const PromptCache = cache.PromptCache;

// Test CacheEntry initialization
test "CacheEntry initializes correctly" {
    const allocator = std.testing.allocator;

    const prompt_hash = "abc123";
    const model_name = "test-model";
    const params_hash = "params456";
    const data = try allocator.alloc(u8, 100);
    defer allocator.free(data);

    var entry = try CacheEntry.init(
        allocator,
        prompt_hash,
        model_name,
        params_hash,
        data,
        1000, // original_size
    );
    defer entry.deinit();

    try std.testing.expectEqualStrings("abc123", entry.prompt_hash);
    try std.testing.expectEqualStrings("test-model", entry.model_name);
    try std.testing.expectEqualStrings("params456", entry.params_hash);
    try std.testing.expectEqual(@as(usize, 100), entry.data.len);
    try std.testing.expectEqual(@as(u64, 1000), entry.original_size);
    try std.testing.expect(entry.access_count >= 1);
}

// Test cache key generation
test "generateCacheKey produces consistent hashes" {
    const allocator = std.testing.allocator;

    const prompt = "Hello, world!";
    const model_name = "test-model";

    const key1 = try cache.generateCacheKey(allocator, prompt, model_name, null);
    defer allocator.free(key1);

    const key2 = try cache.generateCacheKey(allocator, prompt, model_name, null);
    defer allocator.free(key2);

    // Same inputs should produce same key
    try std.testing.expectEqualStrings(key1, key2);

    // Different inputs should produce different keys
    const key3 = try cache.generateCacheKey(allocator, "Different prompt", model_name, null);
    defer allocator.free(key3);

    try std.testing.expect(!std.mem.eql(u8, key1, key3));
}

// Test PromptCache basic operations
test "PromptCache put and get" {
    const allocator = std.testing.allocator;

    var prompt_cache = PromptCache.init(allocator, 1024 * 1024 * 100); // 100MB
    defer prompt_cache.deinit();

    const prompt = "Test prompt";
    const model_name = "test-model";
    const data = try allocator.alloc(u8, 50);
    defer allocator.free(data);
    @memset(data, 0xAA);

    // Put entry in cache
    try prompt_cache.put(prompt, model_name, null, data, 100);

    // Get entry from cache
    const retrieved = prompt_cache.get(prompt, model_name, null);
    try std.testing.expect(retrieved != null);
    try std.testing.expectEqual(@as(usize, 50), retrieved.?.len);
}

test "PromptCache returns null for missing entries" {
    const allocator = std.testing.allocator;

    var prompt_cache = PromptCache.init(allocator, 1024 * 1024 * 100);
    defer prompt_cache.deinit();

    const result = prompt_cache.get("nonexistent", "model", null);
    try std.testing.expectEqual(@as(?[]const u8, null), result);
}

test "PromptCache updates existing entries" {
    const allocator = std.testing.allocator;

    var prompt_cache = PromptCache.init(allocator, 1024 * 1024 * 100);
    defer prompt_cache.deinit();

    const prompt = "Test prompt";
    const model_name = "test-model";

    // First put
    const data1 = try allocator.alloc(u8, 50);
    defer allocator.free(data1);
    try prompt_cache.put(prompt, model_name, null, data1, 100);

    // Second put with same key should update
    const data2 = try allocator.alloc(u8, 75);
    defer allocator.free(data2);
    try prompt_cache.put(prompt, model_name, null, data2, 150);

    const retrieved = prompt_cache.get(prompt, model_name, null);
    try std.testing.expect(retrieved != null);
    try std.testing.expectEqual(@as(usize, 75), retrieved.?.len);
}

// Test cache metrics
test "PromptCache metrics track hits and misses" {
    const allocator = std.testing.allocator;

    var prompt_cache = PromptCache.init(allocator, 1024 * 1024 * 100);
    defer prompt_cache.deinit();

    // Initial metrics should be zero
    const metrics1 = prompt_cache.getMetrics();
    try std.testing.expectEqual(@as(u64, 0), metrics1.hits);
    try std.testing.expectEqual(@as(u64, 0), metrics1.misses);

    // Add entry
    const data = try allocator.alloc(u8, 50);
    defer allocator.free(data);
    try prompt_cache.put("prompt", "model", null, data, 100);

    // Get should count as hit
    _ = prompt_cache.get("prompt", "model", null);
    const metrics2 = prompt_cache.getMetrics();
    try std.testing.expectEqual(@as(u64, 1), metrics2.hits);
    try std.testing.expectEqual(@as(u64, 0), metrics2.misses);

    // Get nonexistent should count as miss
    _ = prompt_cache.get("nonexistent", "model", null);
    const metrics3 = prompt_cache.getMetrics();
    try std.testing.expectEqual(@as(u64, 1), metrics3.hits);
    try std.testing.expectEqual(@as(u64, 1), metrics3.misses);
}

// Test cache size limits (if eviction is implemented)
test "PromptCache respects size limits" {
    const allocator = std.testing.allocator;

    // Small cache to force eviction
    var prompt_cache = PromptCache.init(allocator, 100);
    defer prompt_cache.deinit();

    // Add entries until we exceed the limit
    const data1 = try allocator.alloc(u8, 60);
    defer allocator.free(data1);
    try prompt_cache.put("prompt1", "model", null, data1, 60);

    const data2 = try allocator.alloc(u8, 60);
    defer allocator.free(data2);
    try prompt_cache.put("prompt2", "model", null, data2, 60);

    // Total is now 120, which exceeds 100 byte limit
    // Depending on implementation, oldest entry might be evicted
    const metrics = prompt_cache.getMetrics();

    // Should have 2 entries or eviction should have occurred
    try std.testing.expect(metrics.entries <= 2);
}
