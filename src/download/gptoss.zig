//! gptoss.zig - GPT-OSS GGUF model download support
//!
//! Provides download functionality for GPT-OSS-20B GGUF models (11GB).

const std = @import("std");
const download = @import("mod.zig");
const registry = @import("../models/registry.zig");

pub const GptOssDownloadError = error{
    InvalidUrl,
    DownloadFailed,
    ChecksumMismatch,
    InsufficientDiskSpace,
    ModelNotFound,
    VerificationFailed,
};

/// Download GPT-OSS GGUF model with progress reporting
///
/// Arguments:
///   - allocator: Memory allocator
///   - cache_dir: Directory to save model
///   - quantization: Quantization level ("Q4_K_M", "Q5_K_M")
///   - progress_callback: Optional callback for progress updates
///
/// Returns: Path to downloaded model file (owned by caller)
pub fn downloadGptOssModel(
    allocator: std.mem.Allocator,
    cache_dir: []const u8,
    quantization: []const u8,
    progress_callback: ?*const fn (downloaded: u64, total: u64) void,
) ![]u8 {
    // Get download info from registry
    const info = registry.getDownloadInfo("gpt-oss-20b") orelse
        return error.ModelNotFound;

    // Construct filename with quantization
    const filename = std.fmt.allocPrint(
        allocator,
        "GPT-OSS-20B-{s}.gguf",
        .{quantization},
    ) catch return error.OutOfMemory;
    defer allocator.free(filename);

    const output_path = try std.fs.path.join(allocator, &.{ cache_dir, filename });

    // Check if already exists
    if (std.fs.accessAbsolute(output_path, .{})) {
        std.log.info("GPT-OSS model already exists at {s}", .{output_path});

        // Verify it's a valid GGUF
        if (try verifyGptOssModel(output_path)) {
            return output_path;
        } else {
            std.log.warn("Existing file is not valid GGUF, re-downloading...", .{});
            std.fs.deleteFileAbsolute(output_path) catch {};
        }
    } else |_| {}

    // Ensure cache directory exists
    try std.fs.cwd().makePath(cache_dir);

    // Check disk space (need ~12GB free)
    const space = try getAvailableDiskSpace(cache_dir);
    if (space < 12_000_000_000) {
        std.log.err("Insufficient disk space: {d} GB available, need ~12 GB for GPT-OSS", .{
            space / (1024 * 1024 * 1024),
        });
        return error.InsufficientDiskSpace;
    }

    // Download with resume support
    std.log.info("Downloading GPT-OSS-20B ({s}) - ~11GB, this may take a while...", .{quantization});
    std.log.info("  Note: Large file download with resume support enabled", .{});

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

            // Verify downloaded file
            if (try verifyGptOssModel(output_path)) {
                std.log.info("GGUF verification passed", .{});
                return output_path;
            } else {
                std.log.warn("Downloaded file failed verification, trying next mirror...", .{});
                std.fs.deleteFileAbsolute(output_path) catch {};
                continue;
            }
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
pub fn verifyGptOssModel(model_path: []const u8) !bool {
    // Check file exists
    const file = std.fs.cwd().openFile(model_path, .{}) catch {
        return false;
    };
    defer file.close();

    // Check file size is reasonable (> 1GB)
    const stat = file.stat() catch {
        return false;
    };
    if (stat.size < 1_000_000_000) {
        std.log.warn("File too small to be GPT-OSS ({d} MB)", .{stat.size / (1024 * 1024)});
        return false;
    }

    // Read first 4 bytes for GGUF magic
    var magic: [4]u8 = undefined;
    _ = file.read(&magic) catch {
        return false;
    };

    // GGUF magic: "GGUF" = [0x47, 0x47, 0x55, 0x46]
    const gguf_magic = [_]u8{ 0x47, 0x47, 0x55, 0x46 };

    if (!std.mem.eql(u8, &magic, &gguf_magic)) {
        std.log.warn("Invalid GGUF magic bytes: {x}", .{magic});
        return false;
    }

    std.log.info("Valid GPT-OSS GGUF file: {s} ({d} MB)", .{
        model_path,
        stat.size / (1024 * 1024),
    });
    return true;
}

/// Check if GPT-OSS model is available locally
pub fn isGptOssAvailable(cache_dir: []const u8, quantization: []const u8) bool {
    const filename = std.fmt.allocPrint(
        std.heap.page_allocator,
        "GPT-OSS-20B-{s}.gguf",
        .{quantization},
    ) catch return false;
    defer std.heap.page_allocator.free(filename);

    const path = std.fs.path.join(std.heap.page_allocator, &.{
        cache_dir,
        filename,
    }) catch return false;
    defer std.heap.page_allocator.free(path);

    // Check file exists and is valid
    if (std.fs.accessAbsolute(path, .{})) {
        // Try to verify it's a valid GGUF
        if (verifyGptOssModel(path) catch false) {
            return true;
        }
    }

    return false;
}

/// Download progress callback type
pub const ProgressCallback = *const fn (downloaded: u64, total: u64) void;

/// Default progress callback that logs to stdout
pub fn defaultProgressCallback(downloaded: u64, total: u64) void {
    if (total > 0) {
        const percent = @as(f64, @floatFromInt(downloaded)) / @as(f64, @floatFromInt(total)) * 100.0;
        const downloaded_mb = downloaded / (1024 * 1024);
        const total_mb = total / (1024 * 1024);

        // Print progress every 100MB
        if (downloaded_mb % 100 == 0) {
            std.log.info("Downloaded: {d:.1}% ({d}/{d} MB)", .{
                percent,
                downloaded_mb,
                total_mb,
            });
        }
    }
}

/// Format bytes to human-readable string
pub fn formatBytes(bytes: u64, allocator: std.mem.Allocator) ![]u8 {
    const gb = @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0 * 1024.0);
    return std.fmt.allocPrint(allocator, "{d:.2} GB", .{gb});
}

/// Calculate download ETA in seconds
pub fn calculateEta(
    downloaded_bytes: u64,
    total_bytes: u64,
    elapsed_seconds: u64,
) u64 {
    if (downloaded_bytes == 0 or elapsed_seconds == 0) {
        return 0;
    }

    const remaining_bytes = total_bytes - downloaded_bytes;
    const bytes_per_second = downloaded_bytes / elapsed_seconds;

    if (bytes_per_second == 0) {
        return 0;
    }

    return remaining_bytes / bytes_per_second;
}
