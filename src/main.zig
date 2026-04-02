// src/main.zig
// Phase 3: HTTP API Server - OpenAI-compatible inference server
// Serves chat completions via /v1/chat/completions endpoint

const std = @import("std");
const api = @import("api/server.zig");
const handlers = @import("api/handlers.zig");
const metrics = @import("api/metrics.zig");
const models_mod = @import("models/mod.zig");
const manager_mod = @import("models/manager.zig");
const memory = @import("models/memory.zig");
const prompt_cache = @import("cache/prompt_cache.zig");

const USAGE =
    "Usage: zlx [OPTIONS]\n" ++
    "\n" ++
    "zlx - Local inference server for OpenAI-compatible API\n" ++
    "\n" ++
    "Options:\n" ++
    "  --model <NAME>          Model name or path (required)\n" ++
    "  --port <PORT>           Server port (default: 8080)\n" ++
    "  --host <HOST>           Bind address (default: 127.0.0.1)\n" ++
    "  --timeout <SECONDS>     Request timeout in seconds (default: 60)\n" ++
    "  --cache-dir <DIR>       Directory for prompt cache (default: ~/.cache/zlx/prompts)\n" ++
    "  --cache-size <GB>       Maximum cache size in GB (default: 10)\n" ++
    "  --cache-enabled         Enable prompt caching (default: true)\n" ++
    "  --help                  Show this help message\n" ++
    "\n" ++
    "Examples:\n" ++
    "  zlx --model qwen2.5-coder-1.5b\n" ++
    "  zlx --model ./models/my-model --port 9000 --timeout 120\n" ++
    "  zlx --model qwen2.5-coder --cache-size 5 --cache-dir ~/.cache/zlx-small\n" ++
    "\n" ++
    "The server exposes OpenAI-compatible endpoints:\n" ++
    "  POST /v1/chat_completions    Chat completions\n" ++
    "  GET  /v1/models              List available models\n" ++
    "  POST /v1/models/switch       Switch to different model\n" ++
    "  GET  /v1/health              Health check\n" ++
    "  GET  /v1/metrics             Prometheus metrics\n" ++
    "\n";

const Config = struct {
    model_name: ?[]const u8 = null,
    model_path: ?[]const u8 = null,
    port: u16 = 8080,
    host: []const u8 = "127.0.0.1",
    timeout_seconds: u32 = 60, // Default 60s per D-32
    cache_dir: []const u8 = "~/.cache/zlx/prompts", // Default cache directory
    cache_size_gb: u32 = 10, // Default 10GB
    cache_enabled: bool = true, // Default enabled
};

fn printUsage() void {
    std.debug.print("{s}", .{USAGE});
}

fn parseArgs(allocator: std.mem.Allocator) !Config {
    var config = Config{};
    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    // Skip program name
    _ = args.next();

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--help")) {
            printUsage();
            std.process.exit(0);
        } else if (std.mem.eql(u8, arg, "--model")) {
            const value = args.next() orelse {
                std.log.err("Expected value after --model", .{});
                return error.MissingArgument;
            };
            config.model_name = value;
        } else if (std.mem.eql(u8, arg, "--port")) {
            const value = args.next() orelse {
                std.log.err("Expected value after --port", .{});
                return error.MissingArgument;
            };
            config.port = try std.fmt.parseInt(u16, value, 10);
        } else if (std.mem.eql(u8, arg, "--host")) {
            const value = args.next() orelse {
                std.log.err("Expected value after --host", .{});
                return error.MissingArgument;
            };
            config.host = try allocator.dupe(u8, value);
        } else if (std.mem.eql(u8, arg, "--timeout")) { // Per D-37
            const value = args.next() orelse {
                std.log.err("Expected value after --timeout", .{});
                return error.MissingArgument;
            };
            config.timeout_seconds = try std.fmt.parseInt(u32, value, 10);
            if (config.timeout_seconds == 0 or config.timeout_seconds > 3600) {
                std.log.err("Timeout must be between 1 and 3600 seconds", .{});
                return error.InvalidTimeout;
            }
        } else if (std.mem.eql(u8, arg, "--cache-dir")) {
            const value = args.next() orelse {
                std.log.err("Expected value after --cache-dir", .{});
                return error.MissingArgument;
            };
            config.cache_dir = try allocator.dupe(u8, value);
        } else if (std.mem.eql(u8, arg, "--cache-size")) {
            const value = args.next() orelse {
                std.log.err("Expected value after --cache-size", .{});
                return error.MissingArgument;
            };
            config.cache_size_gb = try std.fmt.parseInt(u32, value, 10);
            if (config.cache_size_gb == 0 or config.cache_size_gb > 1000) {
                std.log.err("Cache size must be between 1 and 1000 GB", .{});
                return error.InvalidCacheSize;
            }
        } else if (std.mem.eql(u8, arg, "--cache-enabled")) {
            const value = args.next() orelse {
                std.log.err("Expected value after --cache-enabled", .{});
                return error.MissingArgument;
            };
            if (std.mem.eql(u8, value, "true")) {
                config.cache_enabled = true;
            } else if (std.mem.eql(u8, value, "false")) {
                config.cache_enabled = false;
            } else {
                std.log.err("--cache-enabled must be 'true' or 'false'", .{});
                return error.InvalidCacheEnabled;
            }
        }
    }

    // Build model path from name if not explicitly set
    if (config.model_path == null) {
        if (config.model_name) |name| {
            // Try to find model in ./models/{name} or use as-is if it contains /
            if (std.mem.indexOf(u8, name, "/")) |_| {
                config.model_path = try allocator.dupe(u8, name);
            } else {
                config.model_path = try std.fmt.allocPrint(allocator, "./models/{s}", .{name});
            }
        }
    }

    return config;
}

fn expandHomeDir(allocator: std.mem.Allocator, path: []const u8) ![]const u8 {
    // Check if path starts with ~
    if (path.len == 0 or path[0] != '~') {
        return allocator.dupe(u8, path);
    }

    // Get HOME environment variable
    const home = std.process.getEnvVarOwned(allocator, "HOME") catch |err| {
        std.log.warn("Could not get HOME environment variable: {s}. Using path as-is.", .{@errorName(err)});
        return allocator.dupe(u8, path);
    };
    defer allocator.free(home);

    // Replace ~ with home directory
    if (path.len == 1) {
        return allocator.dupe(u8, home);
    } else if (path.len > 1 and path[1] == '/') {
        return try std.fmt.allocPrint(allocator, "{s}{s}", .{ home, path[1..] });
    } else {
        // ~something - treat as literal path
        return allocator.dupe(u8, path);
    }
}

fn resolveModelPath(allocator: std.mem.Allocator, name_or_path: []const u8) ![]const u8 {
    // If it contains a slash, treat as explicit path
    if (std.mem.indexOf(u8, name_or_path, "/")) |_| {
        return allocator.dupe(u8, name_or_path);
    }

    // Otherwise, look in ./models/
    return try std.fmt.allocPrint(allocator, "./models/{s}", .{name_or_path});
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const config = parseArgs(allocator) catch |err| {
        if (err == error.MissingArgument) {
            printUsage();
            std.process.exit(1);
        }
        return err;
    };

    // Validate model is specified
    if (config.model_path == null) {
        std.log.err("No model specified. Use --model <name>", .{});
        printUsage();
        std.process.exit(1);
    }

    const model_path = config.model_path.?;
    defer allocator.free(model_path);

    // Validate model path exists
    std.fs.cwd().access(model_path, .{}) catch {
        std.log.err("Model not found at: {s}", .{model_path});
        std.log.err("Make sure the model is downloaded to ./models/", .{});
        std.process.exit(1);
    };

    std.log.info("zlx - Local inference server", .{});

    // Initialize memory tracker first (tracks all subsequent allocations)
    try memory.initGlobalTracker(allocator);
    defer memory.deinitGlobalTracker(allocator);

    // Initialize prompt cache (if enabled)
    if (config.cache_enabled) {
        const cache_dir_expanded = try expandHomeDir(allocator, config.cache_dir);
        defer allocator.free(cache_dir_expanded);

        prompt_cache.initGlobalCache(allocator, cache_dir_expanded, config.cache_size_gb) catch |err| {
            std.log.warn("Failed to initialize prompt cache: {s}. Continuing without caching.", .{@errorName(err)});
            // Continue without cache - not fatal
        };
    } else {
        std.log.info("Prompt caching disabled", .{});
    }
    defer prompt_cache.deinitGlobalCache(allocator);

    // Initialize model registry (scans for available models)
    try models_mod.initGlobalRegistry(allocator);
    defer models_mod.deinitGlobalRegistry(allocator);

    // Initialize model manager
    if (models_mod.getGlobalRegistry()) |registry| {
        try manager_mod.initGlobalManager(allocator, registry);
    }
    defer manager_mod.deinitGlobalManager(allocator);

    // Get model name for status update
    const model_name = config.model_name orelse model_path;

    std.log.info("Loading model from: {s}", .{model_path});

    // Initialize inference context via manager for proper tracking
    if (manager_mod.getGlobalManager()) |manager| {
        // Use manager to load initial model (enables hot-swap later)
        try manager.switchModel(model_name);
        std.log.info("Model loaded successfully via manager!", .{});
    } else {
        // Fallback: direct loading without manager
        try handlers.initGlobalContext(allocator, model_path);

        // Update registry status to show model is loaded
        if (models_mod.getGlobalRegistry()) |reg| {
            reg.updateStatus(model_name, .loaded);
        }

        std.log.info("Model loaded successfully!", .{});
    }

    // Initialize metrics tracking
    metrics.initMetrics(allocator);

    // Start periodic stats logging (every 10 seconds)
    _ = metrics.startStatsLogging(allocator, 10) catch |err| {
        std.log.warn("Failed to start stats logging: {s}", .{@errorName(err)});
    };

    std.log.info("Stats logging enabled - logs every 10 seconds", .{});

    // Configure and run the HTTP server
    const server_config = api.ServerConfig{
        .port = config.port,
        .address = config.host,
        .timeout_seconds = config.timeout_seconds,
    };

    try api.runServer(allocator, server_config);
}
