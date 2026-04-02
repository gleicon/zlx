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
const compression = @import("compression/mod.zig");
const speculation = @import("speculation/mod.zig");
const config_mod = @import("config.zig");
const download = @import("download/mod.zig");

const USAGE =
    "Usage: zlx [OPTIONS]\n" ++
    "\n" ++
    "zlx - Local inference server for OpenAI-compatible API\n" ++
    "\n" ++
    "Configuration:\n" ++
    "  Settings are loaded from (in order of priority):\n" ++
    "    1. CLI flags (highest priority)\n" ++
    "    2. Environment variables (ZLX_*)\n" ++
    "    3. Config file: ~/.config/zlx/config.json or ./zlx.json\n" ++
    "    4. Built-in defaults\n" ++
    "\n" ++
    "Options:\n" ++
    "  --config <PATH>         Load configuration from file\n" ++
    "  --model <NAME>          Model name or path (required)\n" ++
    "  --port <PORT>           Server port (default: 8080)\n" ++
    "  --host <HOST>           Bind address (default: 127.0.0.1)\n" ++
    "  --timeout <SECONDS>     Request timeout in seconds (default: 60)\n" ++
    "  --cache-dir <DIR>       Directory for prompt cache (default: ~/.cache/zlx/prompts)\n" ++
    "  --cache-size <GB>       Maximum cache size in GB (default: 10)\n" ++
    "  --cache-enabled         Enable prompt caching (default: true)\n" ++
    "  --turboquant            Enable TurboQuant KV cache compression (BETA)\n" ++
    "  --turboquant-bits N     Quantization bits: 3 or 4 (default: 4)\n" ++
    "  --turboquant-adaptive N Keep first/last N layers in FP16 (default: 4)\n" ++
    "  --draft-model <NAME>   Draft model for speculative decoding (auto-select if not set)\n" ++
    "  --speculation-depth N   Tokens to speculate ahead: 1-8 (default: 4)\n" ++
    "  --no-speculation        Disable speculative decoding\n" ++
    "  --list-models           List available model shortcuts\n" ++
    "  --configure-opencode    Auto-configure OpenCode to use this server\n" ++
    "  --help                  Show this help message\n" ++
    "\n" ++
    "Environment Variables:\n" ++
    "  ZLX_MODEL               Model name (overrides config file)\n" ++
    "  ZLX_PORT                Server port\n" ++
    "  ZLX_HOST                Bind address\n" ++
    "  ZLX_TIMEOUT             Request timeout\n" ++
    "  ZLX_CACHE_SIZE          Cache size in GB\n" ++
    "  ZLX_TURBOQUANT          Enable TurboQuant (true/1)\n" ++
    "  ZLX_CORS_ORIGINS        CORS allowed origins (default: *)\n" ++
    "\n" ++
    "Examples:\n" ++
    "  zlx --model qwen2.5-coder-1.5b\n" ++
    "  zlx --model ./models/my-model --port 9000 --timeout 120\n" ++
    "  zlx --model qwen2.5-coder --cache-size 5 --cache-dir ~/.cache/zlx-small\n" ++
    "  zlx --model qwen2.5-coder --turboquant --turboquant-bits 4\n" ++
    "  zlx --model qwen2.5-coder-7b --draft-model qwen2.5-coder-1.5b --speculation-depth 4\n" ++
    "  zlx --model gpt-oss-20b                     # OpenAI GPT-OSS (11.2 GB)\n" ++
    "  zlx --list-models                           # Show all model shortcuts\n" ++
    "  zlx --download-model qwen2.5-coder-1.5b       # Download a model\n" ++
    "\n" ++
    "Model Shortcuts:\n" ++
    "  Use short names like 'qwen2.5-coder-1.5b' or 'gpt-oss-20b'\n" ++
    "  Run 'zlx --list-models' to see all available shortcuts\n" ++
    "\n" ++
    "The server exposes OpenAI-compatible endpoints:\n" ++
    "  POST /v1/chat_completions       Chat completions\n" ++
    "  GET  /v1/models                 List available models\n" ++
    "  POST /v1/models/load            Start background model load\n" ++
    "  GET  /v1/models/load-status     Check background load progress\n" ++
    "  POST /v1/models/load/cancel     Cancel ongoing background load\n" ++
    "  POST /v1/models/switch          Switch to different model (blocking)\n" ++
    "  GET  /v1/health                 Health check\n" ++
    "  GET  /v1/metrics                Prometheus metrics\n" ++
    "  GET  /v1/metrics/speculative    Speculative decoding metrics\n" ++
    "\n" ++
    "CORS Configuration:\n" ++
    "  Default: CORS is enabled for all origins (*) for Open WebUI compatibility.\n" ++
    "  Set cors_origins in ~/.config/zlx/config.json to restrict (e.g., \"http://localhost:8081\").\n" ++
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
    turboquant_enabled: bool = false, // TurboQuant compression (EXPERIMENTAL)
    turboquant_bits: u4 = 4, // Quantization bits (3 or 4)
    turboquant_adaptive: u8 = 4, // First/last N layers kept in FP16
    draft_model: ?[]const u8 = null, // Draft model for speculation (null = auto)
    speculation_depth: usize = 4, // Tokens to speculate ahead (default: 4)
    no_speculation: bool = false, // Disable speculative decoding
    configure_opencode: bool = false, // Configure OpenCode and exit
};

fn printUsage() void {
    std.debug.print("{s}", .{USAGE});
}

fn parseArgs(allocator: std.mem.Allocator) !Config {
    // First, check for --config flag before loading any config
    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    // Skip program name
    _ = args.next();

    var explicit_config_path: ?[]const u8 = null;

    // First pass: look for --config flag
    var args_copy = try std.process.argsWithAllocator(allocator);
    defer args_copy.deinit();
    _ = args_copy.next(); // Skip program name

    while (args_copy.next()) |arg| {
        if (std.mem.eql(u8, arg, "--config")) {
            explicit_config_path = args_copy.next() orelse {
                std.log.err("Expected value after --config", .{});
                return error.MissingArgument;
            };
            break;
        }
    }

    // Load config from file (either explicit path or default locations)
    var config = Config{};
    var config_source: []const u8 = "defaults";

    if (explicit_config_path) |path| {
        // Load from explicit path
        const file_config = config_mod.loadConfigFromPath(allocator, path) catch |err| {
            std.log.err("Failed to load config from {s}: {s}", .{ path, @errorName(err) });
            return err;
        };
        config = convertFileConfig(file_config);
        config_source = try allocator.dupe(u8, path);
    } else {
        // Try loading from default locations
        const file_config = config_mod.loadConfig(allocator) catch |err| blk: {
            if (err == config_mod.ConfigError.InvalidConfigFile) {
                std.log.err("Config file has errors. Please fix and try again.", .{});
                return err;
            }
            // Other errors just mean no config file found, use defaults
            break :blk config_mod.Config{};
        };
        config = convertFileConfig(file_config);
        if (file_config.config_file) |path| {
            config_source = try allocator.dupe(u8, path);
        }
    }

    // Second pass: Apply CLI overrides (highest priority)
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--help")) {
            printUsage();
            std.process.exit(0);
        } else if (std.mem.eql(u8, arg, "--config")) {
            // Already handled in first pass, just skip the value
            _ = args.next();
        } else if (std.mem.eql(u8, arg, "--model")) {
            const value = args.next() orelse {
                std.log.err("Expected value after --model", .{});
                return error.MissingArgument;
            };
            config.model_name = try allocator.dupe(u8, value);
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
        } else if (std.mem.eql(u8, arg, "--turboquant")) {
            config.turboquant_enabled = true;
        } else if (std.mem.eql(u8, arg, "--turboquant-bits")) {
            const value = args.next() orelse {
                std.log.err("Expected value after --turboquant-bits", .{});
                return error.MissingArgument;
            };
            const bits = try std.fmt.parseInt(u8, value, 10);
            if (bits < 3 or bits > 4) {
                std.log.warn("TurboQuant only supports 3-4 bits, got {d}. Using 4 bits.", .{bits});
                config.turboquant_bits = 4;
            } else {
                config.turboquant_bits = @intCast(bits);
            }
        } else if (std.mem.eql(u8, arg, "--turboquant-adaptive")) {
            const value = args.next() orelse {
                std.log.err("Expected value after --turboquant-adaptive", .{});
                return error.MissingArgument;
            };
            config.turboquant_adaptive = try std.fmt.parseInt(u8, value, 10);
            if (config.turboquant_adaptive > 32) {
                std.log.warn("Adaptive layers >32 seems high, but accepting value: {d}", .{config.turboquant_adaptive});
            }
        } else if (std.mem.eql(u8, arg, "--draft-model")) {
            const value = args.next() orelse {
                std.log.err("Expected value after --draft-model", .{});
                return error.MissingArgument;
            };
            config.draft_model = try allocator.dupe(u8, value);
        } else if (std.mem.eql(u8, arg, "--speculation-depth")) {
            const value = args.next() orelse {
                std.log.err("Expected value after --speculation-depth", .{});
                return error.MissingArgument;
            };
            config.speculation_depth = try std.fmt.parseInt(usize, value, 10);
            if (config.speculation_depth < 1 or config.speculation_depth > 8) {
                std.log.err("Speculation depth must be 1-8, got {d}", .{config.speculation_depth});
                return error.InvalidSpeculationDepth;
            }
        } else if (std.mem.eql(u8, arg, "--no-speculation")) {
            config.no_speculation = true;
        } else if (std.mem.eql(u8, arg, "--list-models")) {
            // List available model shortcuts and exit
            download.listKnownModels();
            std.process.exit(0);
        } else if (std.mem.eql(u8, arg, "--download-model")) {
            const value = args.next() orelse {
                std.log.err("Expected model name after --download-model", .{});
                return error.MissingArgument;
            };
            // Resolve alias to HF ID
            const model_id = download.resolveModelAlias(value);
            std.log.info("Downloading model: {s} (resolved from: {s})", .{ model_id, value });
            // TODO: Actually download the model
            std.log.info("Download not yet implemented - manually download from:", .{});
            std.log.info("  https://huggingface.co/{s}", .{model_id});
            std.process.exit(0);
        } else if (std.mem.eql(u8, arg, "--configure-opencode")) {
            config.configure_opencode = true;
        }
    }

    // Build model path from name if not explicitly set
    if (config.model_path == null) {
        if (config.model_name) |name| {
            // Resolve model alias to HF ID or path
            const resolved_name = download.resolveModelAlias(name);

            // Check if it's a HuggingFace ID (contains /) or local path
            if (std.mem.indexOf(u8, resolved_name, "/")) |_| {
                // HuggingFace ID - for now use as-is (download manager will handle)
                config.model_path = try allocator.dupe(u8, resolved_name);
                std.log.info("Model resolved from '{s}' to HuggingFace ID: {s}", .{ name, resolved_name });
            } else {
                // Local path
                config.model_path = try std.fmt.allocPrint(allocator, "./models/{s}", .{resolved_name});
            }
        }
    }

    // Handle --configure-opencode (must come after model_name is set)
    if (config.configure_opencode) {
        if (config.model_name == null) {
            std.log.err("--configure-opencode requires --model to be specified", .{});
            std.log.info("Example: zlx --model qwen2.5-coder-1.5b --configure-opencode", .{});
            return error.MissingModelArgument;
        }
        try configureOpenCode(allocator, config.port, config.model_name);
        std.process.exit(0);
    }

    // Log configuration source
    std.log.info("Configuration loaded from: {s}", .{config_source});
    if (config.port != 8080) {
        std.log.info("Port overridden to {d}", .{config.port});
    }
    if (config.turboquant_enabled) {
        std.log.info("TurboQuant KV cache compression enabled ({d} bits, adaptive: {d})", .{
            config.turboquant_bits,
            config.turboquant_adaptive,
        });
    }
    if (config.draft_model) |draft| {
        std.log.info("Speculative decoding enabled with draft model: {s} (depth: {d})", .{
            draft,
            config.speculation_depth,
        });
    } else if (!config.no_speculation) {
        std.log.info("Speculative decoding enabled (auto-select draft model, depth: {d})", .{
            config.speculation_depth,
        });
    }

    return config;
}

/// Convert from config_mod.Config to main.Config
fn convertFileConfig(file_config: config_mod.Config) Config {
    return .{
        .model_name = if (file_config.model) |m| m else null,
        .model_path = null, // Will be built from model_name
        .port = file_config.port,
        .host = file_config.host,
        .timeout_seconds = file_config.timeout_seconds,
        .cache_dir = file_config.cache_dir,
        .cache_size_gb = file_config.cache_size_gb,
        .cache_enabled = file_config.cache_enabled,
        .turboquant_enabled = file_config.turboquant_enabled,
        .turboquant_bits = file_config.turboquant_bits,
        .turboquant_adaptive = file_config.turboquant_adaptive,
        .draft_model = file_config.draft_model,
        .speculation_depth = file_config.speculation_depth,
        .no_speculation = file_config.no_speculation,
    };
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

/// Configure OpenCode to use this zlx server
fn configureOpenCode(allocator: std.mem.Allocator, port: u16, model_name: ?[]const u8) !void {
    const home = std.process.getEnvVarOwned(allocator, "HOME") catch |err| {
        if (err == error.EnvironmentVariableNotFound) {
            std.log.err("HOME environment variable not set", .{});
            return error.HomeNotSet;
        }
        return err;
    };
    defer allocator.free(home);

    // OpenCode config directory
    const config_dir = try std.fs.path.join(allocator, &.{ home, ".config", "opencode" });
    defer allocator.free(config_dir);

    // Create directory if it doesn't exist
    std.fs.cwd().makeDir(config_dir) catch |err| {
        if (err != error.PathAlreadyExists) {
            std.log.err("Failed to create config directory: {s}", .{@errorName(err)});
            return err;
        }
    };

    const config_path = try std.fs.path.join(allocator, &.{ config_dir, "config.json" });
    defer allocator.free(config_path);

    // Check if config already exists
    const existing_config = std.fs.cwd().access(config_path, .{}) catch null;
    if (existing_config != null) {
        std.log.warn("OpenCode config already exists at: {s}", .{config_path});
        std.log.info("Backing up to: {s}.backup", .{config_path});

        // Simple backup - just rename (or you could copy)
        const backup_path = try std.fmt.allocPrint(allocator, "{s}.backup", .{config_path});
        defer allocator.free(backup_path);

        std.fs.cwd().rename(config_path, backup_path) catch |err| {
            std.log.warn("Failed to backup existing config: {s}", .{@errorName(err)});
            std.log.info("Continuing without backup...", .{});
        };
    }

    // Determine model name to use in config
    const model_display_name = blk: {
        if (model_name) |name| {
            // Extract basename if it's a path
            if (std.mem.lastIndexOf(u8, name, "/")) |last_slash| {
                break :blk name[last_slash + 1 ..];
            }
            break :blk name;
        }
        break :blk "local-model";
    };

    // Create config content
    const config_content = try std.fmt.allocPrint(allocator,
        \\{{
        \\  "model": "openai/{s}",
        \\  "server": {{
        \\    "baseUrl": "http://127.0.0.1:{d}/v1"
        \\  }}
        \\}}
    , .{ model_display_name, port });
    defer allocator.free(config_content);

    // Write config file
    const file = try std.fs.cwd().createFile(config_path, .{});
    defer file.close();

    try file.writeAll(config_content);

    std.log.info("OpenCode configured successfully!", .{});
    std.log.info("Config written to: {s}", .{config_path});
    std.log.info("", .{});
    std.log.info("To use OpenCode with zlx:", .{});
    std.log.info("  1. Make sure zlx is running: ./zig-out/bin/zlx --model {s} --port {d}", .{ model_name orelse "<your-model>", port });
    std.log.info("  2. Run: opencode", .{});
    std.log.info("", .{});
    std.log.info("To restore previous config:", .{});
    std.log.info("  mv {s}.backup {s}", .{ config_path, config_path });
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

    // Initialize compression configuration (stubbed for now)
    const compression_config = compression.CompressionConfig{
        .compression_type = if (config.turboquant_enabled) .TurboQuant else .NoOp,
        .bits = config.turboquant_bits,
        .adaptive_layers = config.turboquant_adaptive,
        .enabled = config.turboquant_enabled,
    };

    if (config.turboquant_enabled) {
        std.log.info("TurboQuant compression active: {d} bits, {d} adaptive layers", .{
            config.turboquant_bits,
            config.turboquant_adaptive,
        });
        std.log.info("Expected compression ratio: ~{d:.1}x", .{
            16.0 / @as(f32, @floatFromInt(config.turboquant_bits)),
        });
    }

    // Store compression config for handlers (currently unused but available for future)
    _ = compression_config;

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

        // Initialize speculative decoding subsystem (after registry is available)
        if (!config.no_speculation) {
            speculation.initSpeculation(allocator, registry) catch |err| {
                std.log.warn("Failed to initialize speculative decoding: {s}. Continuing without speculation.", .{@errorName(err)});
            };

            // Configure speculation settings
            if (speculation.isInitialized()) {
                const spec_config = speculation.SpeculationConfig{
                    .enabled = true,
                    .draft_model = config.draft_model,
                    .speculation_depth = config.speculation_depth,
                };
                speculation.configure(spec_config);

                std.log.info("Speculative decoding enabled (depth: {d})", .{config.speculation_depth});
                if (config.draft_model) |dm| {
                    std.log.info("Draft model: {s}", .{dm});
                } else {
                    std.log.info("Draft model: auto-select", .{});
                }
            }
        } else {
            std.log.info("Speculative decoding disabled by user", .{});
        }
    }
    defer {
        speculation.shutdownSpeculation(allocator);
        manager_mod.deinitGlobalManager(allocator);
    }

    // Get model name for status update (extract basename from path if needed)
    const raw_model_name = config.model_name orelse model_path;
    const model_name = blk: {
        // If the model name contains path separators, extract just the basename
        if (std.mem.indexOf(u8, raw_model_name, "/")) |_| {
            if (std.mem.lastIndexOf(u8, raw_model_name, "/")) |last_slash| {
                break :blk raw_model_name[last_slash + 1 ..];
            }
        }
        break :blk raw_model_name;
    };

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
