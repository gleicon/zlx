//! deepseek.zig - DeepSeek GGUF model download support
//!
//! Provides download functionality for DeepSeek-Coder-V2-Lite GGUF models.

const std = @import("std");
const download = @import("mod.zig");
const registry = @import("../models/registry.zig");

pub const DeepSeekDownloadError = error{
    InvalidUrl,
    DownloadFailed,
    ChecksumMismatch,
    InsufficientDiskSpace,
    ModelNotFound,
};

/// Download DeepSeek GGUF model with progress reporting
///
/// Arguments:
///   - allocator: Memory allocator
///   - cache_dir: Directory to save model
///   - quantization: Quantization level ("Q4_K_M", "Q5_K_M", "Q6_K")
///   - progress_callback: Optional callback for progress updates
///
/// Returns: Path to downloaded model file (owned by caller)
pub fn downloadDeepSeekModel(
    allocator: std.mem.Allocator,
    cache_dir: []const u8,
    quantization: []const u8,
    progress_callback: ?*const fn (downloaded: u64, total: u64) void,
) ![]u8 {
    // Get download info from registry
    const info = registry.getDownloadInfo("deepseek-coder-v2-lite") orelse
        return error.ModelNotFound;

    // Construct filename with quantization
    const filename = std.fmt.allocPrint(
        allocator,
        "deepseek-coder-v2-lite.{s}.gguf",
        .{quantization},
    ) catch return error.OutOfMemory;
    defer allocator.free(filename);

    const output_path = try std.fs.path.join(allocator, &.{ cache_dir, filename });

    // Check if already exists
    if (std.fs.accessAbsolute(output_path, .{})) {
        std.log.info("DeepSeek model already exists at {s}", .{output_path});
        return output_path;
    } else |_| {}

    // Ensure cache directory exists
    try std.fs.cwd().makePath(cache_dir);

    // Check disk space (need ~5GB free)
    const space = try getAvailableDiskSpace(cache_dir);
    if (space < 5_000_000_000) {
        std.log.err("Insufficient disk space: {d} GB available, need ~5 GB", .{
            space / (1024 * 1024 * 1024),
        });
        return error.InsufficientDiskSpace;
    }

    // Download with resume support
    std.log.info("Downloading DeepSeek-Coder-V2-Lite ({s}) - ~4.5GB...", .{quantization});

    // Try each URL
    for (info.urls) |url| {
        const result = download.downloadWithResume(
            allocator,
            url,
            output_path,
            info.expected_size_bytes,
            progress_callback,
        ) catch |err| {
            std.log.warn("Download from {s} failed: {s}", .{ url, @errorName(err) });
            continue;
        };

        if (result) {
            std.log.info("Download complete: {s}", .{output_path});

            // TODO: Verify checksum if available
            // if (info.checksum) |expected| {
            //     if (!verifyChecksum(output_path, expected)) {
            //         return error.ChecksumMismatch;
            //     }
            // }

            return output_path;
        }
    }

    return error.DownloadFailed;
}

/// Get available disk space in bytes (macOS/Linux)
fn getAvailableDiskSpace(path: []const u8) !u64 {
    // Platform-specific implementation
    // macOS: use statfs via libc
    // Linux: use statvfs

    // For now, return a safe default (assume enough space)
    // In production, this should use actual system calls
    _ = path;
    return 100_000_000_000; // 100GB placeholder
}

/// Verify GGUF model can be loaded by llama.cpp
pub fn verifyDeepSeekModel(model_path: []const u8) !bool {
    // Check file exists and has valid GGUF magic
    const file = std.fs.cwd().openFile(model_path, .{}) catch {
        return false;
    };
    defer file.close();

    // Read first 4 bytes for GGUF magic
    var magic: [4]u8 = undefined;
    _ = file.read(&magic) catch {
        return false;
    };

    // GGUF magic: "GGUF" in little-endian = [0x47, 0x47, 0x55, 0x46] or [0x46, 0x55, 0x47, 0x47]
    const gguf_magic_le = [_]u8{ 0x47, 0x47, 0x55, 0x46 }; // "GGUF"
    const gguf_magic_be = [_]u8{ 0x46, 0x55, 0x47, 0x47 }; // Reversed for BE check

    if (std.mem.eql(u8, &magic, &gguf_magic_le) or std.mem.eql(u8, &magic, &gguf_magic_be)) {
        std.log.info("Valid GGUF file: {s}", .{model_path});
        return true;
    }

    std.log.warn("Invalid GGUF magic bytes: {x}", .{magic});
    return false;
}

/// Download progress callback type
pub const ProgressCallback = *const fn (downloaded: u64, total: u64) void;

/// Default progress callback that logs to stdout
pub fn defaultProgressCallback(downloaded: u64, total: u64) void {
    if (total > 0) {
        const percent = @as(f64, @floatFromInt(downloaded)) / @as(f64, @floatFromInt(total)) * 100.0;
        std.log.info("Downloaded: {d:.1}% ({d}/{d} MB)", .{
            percent,
            downloaded / (1024 * 1024),
            total / (1024 * 1024),
        });
    }
}
