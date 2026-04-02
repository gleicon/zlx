//! config_test.zig - TDD tests for configuration system

const std = @import("std");
const config = @import("config.zig");
const testing = std.testing;

// Test 1: loadConfig with no files uses defaults
test "loadConfig uses defaults when no config files exist" {
    const allocator = testing.allocator;

    // Ensure we're not picking up any user config
    const cfg = try config.loadConfig(allocator);

    // Verify defaults
    try testing.expectEqual(@as(u16, 8080), cfg.port);
    try testing.expectEqualStrings("127.0.0.1", cfg.host);
    try testing.expectEqual(@as(u32, 60), cfg.timeout_seconds);
    try testing.expectEqualStrings("~/.cache/zlx/prompts", cfg.cache_dir);
    try testing.expectEqual(@as(u32, 10), cfg.cache_size_gb);
    try testing.expectEqual(@as(bool, true), cfg.cache_enabled);
    try testing.expectEqual(@as(bool, false), cfg.turboquant_enabled);
    try testing.expectEqual(@as(u4, 4), cfg.turboquant_bits);
    try testing.expectEqual(@as(u8, 4), cfg.turboquant_adaptive);
    try testing.expectEqual(@as(usize, 4), cfg.speculation_depth);
    try testing.expectEqual(@as(bool, false), cfg.no_speculation);
    try testing.expectEqualStrings("*", cfg.cors_origins);
    try testing.expect(cfg.model == null);
    try testing.expect(cfg.draft_model == null);
}

// Test 2: loadConfigFromPath returns error on nonexistent file
test "loadConfigFromPath returns error on nonexistent file" {
    const allocator = testing.allocator;

    // This should fail with file not found
    const result = config.loadConfigFromPath(allocator, "/nonexistent/path/config.json");
    // We expect this to fail with file not found
    try testing.expectError(error.FileNotFound, result);
}

// Test 3: expandPath handles home directory
test "expandPath expands tilde to HOME" {
    const allocator = testing.allocator;

    // Get HOME env var
    const home = std.process.getEnvVarOwned(allocator, "HOME") catch |err| {
        // Skip test if HOME not set
        if (err == error.EnvironmentVariableNotFound) return;
        return err;
    };
    defer allocator.free(home);

    const expanded = try config.expandPath(allocator, "~/.config/zlx");
    defer allocator.free(expanded);

    const expected = try std.fmt.allocPrint(allocator, "{s}/.config/zlx", .{home});
    defer allocator.free(expected);

    try testing.expectEqualStrings(expected, expanded);
}

// Test 4: expandPath handles just tilde
test "expandPath handles just tilde" {
    const allocator = testing.allocator;

    const home = std.process.getEnvVarOwned(allocator, "HOME") catch |err| {
        if (err == error.EnvironmentVariableNotFound) return;
        return err;
    };
    defer allocator.free(home);

    const expanded = try config.expandPath(allocator, "~");
    defer allocator.free(expanded);

    try testing.expectEqualStrings(home, expanded);
}

// Test 5: expandPath leaves absolute paths unchanged
test "expandPath leaves absolute paths unchanged" {
    const allocator = testing.allocator;

    const expanded = try config.expandPath(allocator, "/absolute/path/to/config");
    defer allocator.free(expanded);

    try testing.expectEqualStrings("/absolute/path/to/config", expanded);
}

// Test 6: expandPath leaves relative paths without tilde unchanged
test "expandPath leaves relative paths unchanged" {
    const allocator = testing.allocator;

    const expanded = try config.expandPath(allocator, "relative/path");
    defer allocator.free(expanded);

    try testing.expectEqualStrings("relative/path", expanded);
}

// Test 7: Config struct has correct default values
test "Config struct default values are correct" {
    const cfg = config.Config{};

    try testing.expectEqual(@as(u16, 8080), cfg.port);
    try testing.expectEqualStrings("127.0.0.1", cfg.host);
    try testing.expectEqual(@as(u32, 60), cfg.timeout_seconds);
    try testing.expectEqualStrings("~/.cache/zlx/prompts", cfg.cache_dir);
    try testing.expectEqual(@as(u32, 10), cfg.cache_size_gb);
    try testing.expectEqual(@as(bool, true), cfg.cache_enabled);
    try testing.expectEqual(@as(bool, false), cfg.turboquant_enabled);
    try testing.expectEqual(@as(u4, 4), cfg.turboquant_bits);
    try testing.expectEqual(@as(u8, 4), cfg.turboquant_adaptive);
    try testing.expectEqual(@as(usize, 4), cfg.speculation_depth);
    try testing.expectEqual(@as(bool, false), cfg.no_speculation);
    try testing.expectEqualStrings("*", cfg.cors_origins);
    try testing.expect(cfg.model == null);
    try testing.expect(cfg.draft_model == null);
    try testing.expect(cfg.config_file == null);
}
