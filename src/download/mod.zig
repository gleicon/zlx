//! mod.zig - Public API for model download subsystem
//!
//! Provides convenient functions for downloading models from HuggingFace.

const std = @import("std");

pub const huggingface = @import("huggingface.zig");
pub const manager = @import("manager.zig");
pub const deepseek = @import("deepseek.zig");

// Re-export key types
pub const HuggingFaceId = huggingface.HuggingFaceId;
pub const DownloadManager = manager.DownloadManager;
pub const DownloadStatus = manager.DownloadStatus;
pub const DownloadProgress = manager.DownloadProgress;
pub const DownloadTask = manager.DownloadTask;
pub const FileInfo = huggingface.FileInfo;

// Re-export deepseek functions
pub const downloadDeepSeekModel = deepseek.downloadDeepSeekModel;
pub const verifyDeepSeekModel = deepseek.verifyDeepSeekModel;

// Re-export global functions
pub const initGlobalManager = manager.initGlobalManager;
pub const deinitGlobalManager = manager.deinitGlobalManager;
pub const getGlobalManager = manager.getGlobalManager;

/// Model aliases for convenient short names
/// Maps user-friendly names to HuggingFace model IDs
pub const ModelAlias = struct {
    alias: []const u8,
    hf_id: []const u8,
    description: []const u8,
    size_gb: f32,
};

/// Known model aliases for popular models
pub const KNOWN_MODELS = [_]ModelAlias{
    .{
        .alias = "qwen2.5-coder-1.5b",
        .hf_id = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
        .description = "Qwen 2.5 Coder 1.5B - Fast coding model",
        .size_gb = 1.0,
    },
    .{
        .alias = "qwen2.5-coder-7b",
        .hf_id = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        .description = "Qwen 2.5 Coder 7B - Powerful coding model",
        .size_gb = 4.2,
    },
    .{
        .alias = "gpt-oss-20b",
        .hf_id = "mlx-community/gpt-oss-20b-MXFP4-Q4",
        .description = "OpenAI GPT-OSS 20B - General chat & tool use (PENDING: requires MoE support)",
        .size_gb = 11.2,
    },
    .{
        .alias = "deepseek-coder-v2-lite",
        .hf_id = "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx",
        .description = "DeepSeek Coder V2 Lite - MoE coding model (PENDING: requires MoE support)",
        .size_gb = 8.84,
    },
};

/// Resolve a model alias to HuggingFace ID
/// Returns the original string if not found (assumes it's already a HF ID)
pub fn resolveModelAlias(alias: []const u8) []const u8 {
    for (KNOWN_MODELS) |model| {
        if (std.mem.eql(u8, alias, model.alias)) {
            return model.hf_id;
        }
    }
    // Not an alias, return original
    return alias;
}

/// List all known model aliases
pub fn listKnownModels() void {
    std.debug.print("Available model shortcuts:\n", .{});
    for (KNOWN_MODELS) |model| {
        std.debug.print("  {s:<25} - {s} ({d:.1} GB)\n", .{
            model.alias,
            model.description,
            model.size_gb,
        });
    }
}

/// Check if a string looks like a HuggingFace model ID (contains "/")
fn isHuggingFaceId(str: []const u8) bool {
    return std.mem.indexOf(u8, str, "/") != null;
}

/// Download a model from HuggingFace if it's not already available locally
/// Returns true if download was started, false if model already exists
pub fn downloadModelIfNeeded(
    allocator: std.mem.Allocator,
    model_id: []const u8,
) !bool {
    // Check if it looks like a HuggingFace ID
    if (!isHuggingFaceId(model_id)) {
        // Not a HF ID, assume it's a local path
        return false;
    }

    // Check if already exists in cache
    const home = std.process.getEnvVarOwned(allocator, "HOME") catch |err| {
        if (err == error.EnvironmentVariableNotFound) {
            std.log.err("HOME environment variable not set", .{});
            return error.HomeNotSet;
        }
        return err;
    };
    defer allocator.free(home);

    // Parse the HF ID to get org/repo
    const hf_id = huggingface.HuggingFaceId.parse(model_id) catch |err| {
        std.log.err("Invalid HuggingFace model ID format '{s}': {s}", .{ model_id, @errorName(err) });
        return err;
    };

    // Check if model directory exists
    const model_dir = try std.fmt.allocPrint(allocator, "{s}/.cache/zlx/models/{s}/{s}", .{
        home, hf_id.org, hf_id.repo,
    });
    defer allocator.free(model_dir);

    // Try to access the directory
    const dir = std.fs.cwd().openDir(model_dir, .{}) catch |err| {
        if (err == error.FileNotFound) {
            // Model not cached, need to download
            return true;
        }
        return err;
    };
    dir.close();

    // Check for essential files (config.json)
    const config_path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{model_dir});
    defer allocator.free(config_path);

    std.fs.cwd().access(config_path, .{}) catch |err| {
        if (err == error.FileNotFound) {
            // Model directory exists but incomplete
            return true;
        }
        return err;
    };

    // Model appears to be fully cached
    std.log.info("Model {s} found in local cache at {s}", .{ model_id, model_dir });
    return false;
}

/// Start download for a model from HuggingFace
/// This blocks until download completes (for MVP simplicity)
pub fn downloadModelBlocking(
    _allocator: std.mem.Allocator,
    model_id: []const u8,
) !void {
    _ = _allocator; // Reserved for future use
    const mgr = getGlobalManager() orelse {
        std.log.err("Download manager not initialized", .{});
        return error.ManagerNotInitialized;
    };

    // Queue the download
    try mgr.queueDownload(model_id);

    // Wait for completion
    std.log.info("Downloading model {s}...", .{model_id});

    var last_progress: u64 = 0;
    while (true) {
        if (mgr.getProgress(model_id)) |progress| {
            const current_bytes = progress.bytes_downloaded;
            const total_bytes = progress.bytes_total;

            // Log progress every MB or on completion
            if (current_bytes != last_progress and (current_bytes / (1024 * 1024) > last_progress / (1024 * 1024) or current_bytes == total_bytes)) {
                if (total_bytes > 0) {
                    const percent = @as(f64, @floatFromInt(current_bytes)) / @as(f64, @floatFromInt(total_bytes)) * 100.0;
                    std.log.info("Download progress: {d:.1}% ({d}/{d} MB)", .{
                        percent,
                        current_bytes / (1024 * 1024),
                        total_bytes / (1024 * 1024),
                    });
                } else {
                    std.log.info("Downloaded {d} MB...", .{current_bytes / (1024 * 1024)});
                }
                last_progress = current_bytes;
            }

            // Check if complete
            if (progress.files_completed == progress.files_total and progress.files_total > 0) {
                std.log.info("Download completed: {s}", .{model_id});
                return;
            }
        } else {
            // No progress tracking, wait a bit and assume done
            std.log.warn("No download progress available", .{});
            std.time.sleep(100 * std.time.ns_per_ms);
            return;
        }

        // Poll every 100ms
        std.time.sleep(100 * std.time.ns_per_ms);
    }
}
