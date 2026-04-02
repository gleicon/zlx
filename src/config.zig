//! config.zig - Configuration file system with priority chain
//!
//! Priority order (highest to lowest):
//! 1. CLI flags (--model, --port, etc.)
//! 2. Environment variables (ZLX_MODEL, ZLX_PORT, etc.)
//! 3. Config file (~/.config/zlx/config.json or ./zlx.json)
//! 4. Built-in defaults

const std = @import("std");

/// Configuration errors
pub const ConfigError = error{
    InvalidPort,
    InvalidTimeout,
    InvalidCacheSize,
    InvalidTurboquantBits,
    InvalidSpeculationDepth,
    InvalidConfigFile,
    MissingRequiredField,
};

/// Configuration structure with all CLI flags as optional fields with defaults
pub const Config = struct {
    model: ?[]const u8 = null,
    port: u16 = 8080,
    host: []const u8 = "127.0.0.1",
    timeout_seconds: u32 = 60,
    cache_dir: []const u8 = "~/.cache/zlx/prompts",
    cache_size_gb: u32 = 10,
    cache_enabled: bool = true,
    turboquant_enabled: bool = false,
    turboquant_bits: u4 = 4,
    turboquant_adaptive: u8 = 4,
    draft_model: ?[]const u8 = null,
    speculation_depth: usize = 4,
    no_speculation: bool = false,
    cors_origins: []const u8 = "*", // Allow all for WebUI compatibility
    config_file: ?[]const u8 = null, // Track source for debugging
};

/// File config struct for JSON parsing (all optional)
const FileConfig = struct {
    model: ?[]const u8 = null,
    port: ?u16 = null,
    host: ?[]const u8 = null,
    timeout_seconds: ?u32 = null,
    cache_dir: ?[]const u8 = null,
    cache_size_gb: ?u32 = null,
    cache_enabled: ?bool = null,
    turboquant_enabled: ?bool = null,
    turboquant_bits: ?u4 = null,
    turboquant_adaptive: ?u8 = null,
    draft_model: ?[]const u8 = null,
    speculation_depth: ?usize = null,
    no_speculation: ?bool = null,
    cors_origins: ?[]const u8 = null,
};

/// Load configuration with priority chain: Defaults -> Config File -> Env -> CLI
pub fn loadConfig(allocator: std.mem.Allocator) !Config {
    var config = Config{}; // Start with defaults

    // 1. Load from file if exists (lowest priority after defaults)
    if (try loadConfigFile(allocator)) |file_config| {
        config = mergeConfig(config, file_config);
    }

    // 2. Override from environment variables
    config = try applyEnvOverrides(allocator, config);

    // 3. Validate final config
    try validateConfig(config);

    return config;
}

/// Load config from explicit file path
pub fn loadConfigFromPath(allocator: std.mem.Allocator, path: []const u8) !Config {
    var config = Config{};

    const content = std.fs.cwd().readFileAlloc(allocator, path, 1024 * 1024) catch |err| { // Max 1MB
        // Don't leak if file read fails
        return err;
    };
    defer allocator.free(content);

    const file_config = try parseConfigJson(allocator, content);
    config = mergeConfig(config, file_config);
    config.config_file = try allocator.dupe(u8, path);

    // Apply env and validate
    config = try applyEnvOverrides(allocator, config);
    try validateConfig(config);

    return config;
}

/// Try to load config file from standard locations
fn loadConfigFile(allocator: std.mem.Allocator) !?FileConfig {
    // Try ~/.config/zlx/config.json first
    const home = std.process.getEnvVarOwned(allocator, "HOME") catch null;
    if (home) |h| {
        defer allocator.free(h);
        const primary_path = try std.fmt.allocPrint(allocator, "{s}/.config/zlx/config.json", .{h});
        defer allocator.free(primary_path);

        if (try loadConfigFileFromPath(allocator, primary_path)) |cfg| {
            return cfg;
        }
    }

    // Fallback to ./zlx.json
    if (try loadConfigFileFromPath(allocator, "./zlx.json")) |cfg| {
        return cfg;
    }

    return null;
}

/// Load config from a specific file path
fn loadConfigFileFromPath(allocator: std.mem.Allocator, path: []const u8) !?FileConfig {
    const content = std.fs.cwd().readFileAlloc(allocator, path, 1024 * 1024) catch |err| {
        if (err == error.FileNotFound) return null;
        std.log.warn("Failed to read config file {s}: {s}", .{ path, @errorName(err) });
        return null;
    };
    defer allocator.free(content);

    return parseConfigJson(allocator, content) catch |err| {
        std.log.err("Failed to parse config file {s}: {s}", .{ path, @errorName(err) });
        return ConfigError.InvalidConfigFile;
    };
}

/// Parse JSON config content
fn parseConfigJson(allocator: std.mem.Allocator, content: []const u8) !FileConfig {
    const parsed = std.json.parseFromSlice(
        FileConfig,
        allocator,
        content,
        .{ .ignore_unknown_fields = true },
    ) catch |err| {
        std.log.err("JSON parse error: {s}", .{@errorName(err)});
        return ConfigError.InvalidConfigFile;
    };
    defer parsed.deinit();

    return parsed.value;
}

/// Merge file config into base config (file values override defaults)
fn mergeConfig(base: Config, file: FileConfig) Config {
    var result = base;

    if (file.model) |v| result.model = v;
    if (file.port) |v| result.port = v;
    if (file.host) |v| result.host = v;
    if (file.timeout_seconds) |v| result.timeout_seconds = v;
    if (file.cache_dir) |v| result.cache_dir = v;
    if (file.cache_size_gb) |v| result.cache_size_gb = v;
    if (file.cache_enabled) |v| result.cache_enabled = v;
    if (file.turboquant_enabled) |v| result.turboquant_enabled = v;
    if (file.turboquant_bits) |v| result.turboquant_bits = v;
    if (file.turboquant_adaptive) |v| result.turboquant_adaptive = v;
    if (file.draft_model) |v| result.draft_model = v;
    if (file.speculation_depth) |v| result.speculation_depth = v;
    if (file.no_speculation) |v| result.no_speculation = v;
    if (file.cors_origins) |v| result.cors_origins = v;

    return result;
}

/// Apply environment variable overrides
fn applyEnvOverrides(allocator: std.mem.Allocator, config: Config) !Config {
    var result = config;

    if (try getEnvVar(allocator, "ZLX_MODEL")) |value| {
        defer allocator.free(value);
        result.model = try allocator.dupe(u8, value);
    }

    if (try getEnvVar(allocator, "ZLX_PORT")) |value| {
        defer allocator.free(value);
        if (std.fmt.parseInt(u16, value, 10)) |port| {
            result.port = port;
        } else |_| {
            std.log.warn("Invalid ZLX_PORT value: {s}, using {d}", .{ value, result.port });
        }
    }

    if (try getEnvVar(allocator, "ZLX_HOST")) |value| {
        defer allocator.free(value);
        result.host = try allocator.dupe(u8, value);
    }

    if (try getEnvVar(allocator, "ZLX_TIMEOUT")) |value| {
        defer allocator.free(value);
        if (std.fmt.parseInt(u32, value, 10)) |timeout| {
            if (timeout > 0 and timeout <= 3600) {
                result.timeout_seconds = timeout;
            } else {
                std.log.warn("ZLX_TIMEOUT out of range (1-3600): {d}, using {d}", .{ timeout, result.timeout_seconds });
            }
        } else |_| {
            std.log.warn("Invalid ZLX_TIMEOUT value: {s}", .{value});
        }
    }

    if (try getEnvVar(allocator, "ZLX_CACHE_DIR")) |value| {
        defer allocator.free(value);
        result.cache_dir = try allocator.dupe(u8, value);
    }

    if (try getEnvVar(allocator, "ZLX_CACHE_SIZE")) |value| {
        defer allocator.free(value);
        if (std.fmt.parseInt(u32, value, 10)) |size| {
            if (size > 0 and size <= 1000) {
                result.cache_size_gb = size;
            } else {
                std.log.warn("ZLX_CACHE_SIZE out of range (1-1000): {d}, using {d}", .{ size, result.cache_size_gb });
            }
        } else |_| {
            std.log.warn("Invalid ZLX_CACHE_SIZE value: {s}", .{value});
        }
    }

    if (try getEnvVar(allocator, "ZLX_CACHE_ENABLED")) |value| {
        defer allocator.free(value);
        result.cache_enabled = std.mem.eql(u8, value, "true") or std.mem.eql(u8, value, "1");
    }

    if (try getEnvVar(allocator, "ZLX_TURBOQUANT")) |value| {
        defer allocator.free(value);
        result.turboquant_enabled = std.mem.eql(u8, value, "true") or std.mem.eql(u8, value, "1");
    }

    if (try getEnvVar(allocator, "ZLX_TURBOQUANT_BITS")) |value| {
        defer allocator.free(value);
        if (std.fmt.parseInt(u4, value, 10)) |bits| {
            if (bits == 3 or bits == 4) {
                result.turboquant_bits = bits;
            } else {
                std.log.warn("ZLX_TURBOQUANT_BITS must be 3 or 4: {d}, using {d}", .{ bits, result.turboquant_bits });
            }
        } else |_| {
            std.log.warn("Invalid ZLX_TURBOQUANT_BITS value: {s}", .{value});
        }
    }

    if (try getEnvVar(allocator, "ZLX_DRAFT_MODEL")) |value| {
        defer allocator.free(value);
        result.draft_model = try allocator.dupe(u8, value);
    }

    if (try getEnvVar(allocator, "ZLX_SPECULATION_DEPTH")) |value| {
        defer allocator.free(value);
        if (std.fmt.parseInt(usize, value, 10)) |depth| {
            if (depth >= 1 and depth <= 8) {
                result.speculation_depth = depth;
            } else {
                std.log.warn("ZLX_SPECULATION_DEPTH must be 1-8: {d}, using {d}", .{ depth, result.speculation_depth });
            }
        } else |_| {
            std.log.warn("Invalid ZLX_SPECULATION_DEPTH value: {s}", .{value});
        }
    }

    if (try getEnvVar(allocator, "ZLX_NO_SPECULATION")) |value| {
        defer allocator.free(value);
        result.no_speculation = std.mem.eql(u8, value, "true") or std.mem.eql(u8, value, "1");
    }

    if (try getEnvVar(allocator, "ZLX_CORS_ORIGINS")) |value| {
        defer allocator.free(value);
        result.cors_origins = try allocator.dupe(u8, value);
    }

    return result;
}

/// Helper to get environment variable
fn getEnvVar(allocator: std.mem.Allocator, key: []const u8) !?[]const u8 {
    return std.process.getEnvVarOwned(allocator, key) catch |err| {
        if (err == error.EnvironmentVariableNotFound) return null;
        return err;
    };
}

/// Validate configuration values
fn validateConfig(config: Config) ConfigError!void {
    // Port: 1-65535 (0 is reserved)
    if (config.port == 0) {
        std.log.err("Invalid port: {d}. Port must be between 1 and 65535.", .{config.port});
        return ConfigError.InvalidPort;
    }

    // Timeout: 1-3600 seconds
    if (config.timeout_seconds == 0 or config.timeout_seconds > 3600) {
        std.log.err("Invalid timeout: {d}. Must be between 1 and 3600 seconds.", .{config.timeout_seconds});
        return ConfigError.InvalidTimeout;
    }

    // Cache size: 1-1000 GB
    if (config.cache_size_gb == 0 or config.cache_size_gb > 1000) {
        std.log.err("Invalid cache size: {d}GB. Must be between 1 and 1000 GB.", .{config.cache_size_gb});
        return ConfigError.InvalidCacheSize;
    }

    // TurboQuant bits: 3 or 4
    if (config.turboquant_bits != 3 and config.turboquant_bits != 4) {
        std.log.err("Invalid turboquant_bits: {d}. Must be 3 or 4.", .{config.turboquant_bits});
        return ConfigError.InvalidTurboquantBits;
    }

    // Speculation depth: 1-8
    if (config.speculation_depth < 1 or config.speculation_depth > 8) {
        std.log.err("Invalid speculation_depth: {d}. Must be between 1 and 8.", .{config.speculation_depth});
        return ConfigError.InvalidSpeculationDepth;
    }

    // Note: model is optional at config file level (can be provided via CLI)
}

/// Helper to expand ~ in paths
pub fn expandPath(allocator: std.mem.Allocator, path: []const u8) ![]const u8 {
    if (path.len > 0 and path[0] == '~') {
        const home = std.process.getEnvVarOwned(allocator, "HOME") catch {
            std.log.warn("Could not expand ~ in path, HOME not set: {s}", .{path});
            return try allocator.dupe(u8, path);
        };
        defer allocator.free(home);

        if (path.len == 1) {
            return try allocator.dupe(u8, home);
        } else if (path[1] == '/') {
            return try std.fmt.allocPrint(allocator, "{s}{s}", .{ home, path[1..] });
        }
    }
    return try allocator.dupe(u8, path);
}
