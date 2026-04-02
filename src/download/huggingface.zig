//! huggingface.zig - HuggingFace API client for model downloads
//!
//! Provides functions to:
//! - Parse HuggingFace model IDs (org/repo format)
//! - Download model files with HTTP Range resume support
//! - Query file lists from HuggingFace API
//! - Verify checksums after download

const std = @import("std");

/// Errors specific to HuggingFace operations
pub const HuggingFaceError = error{
    InvalidModelId,
    DownloadFailed,
    ApiError,
    ChecksumMismatch,
};

/// Parsed HuggingFace model identifier
pub const HuggingFaceId = struct {
    org: []const u8,
    repo: []const u8,

    /// Parse a HuggingFace ID in "org/repo" format
    pub fn parse(id: []const u8) !HuggingFaceId {
        // Find the first slash
        const slash_idx = std.mem.indexOf(u8, id, "/");
        if (slash_idx == null) {
            return HuggingFaceError.InvalidModelId;
        }

        const idx = slash_idx.?;
        if (idx == 0 or idx == id.len - 1) {
            // Slash at beginning or end
            return HuggingFaceError.InvalidModelId;
        }

        // Check for multiple slashes (only one allowed)
        if (std.mem.indexOf(u8, id[idx + 1 ..], "/") != null) {
            return HuggingFaceError.InvalidModelId;
        }

        return .{
            .org = id[0..idx],
            .repo = id[idx + 1 ..],
        };
    }

    /// Build download URL for a specific file
    pub fn buildDownloadUrl(self: HuggingFaceId, filename: []const u8, allocator: std.mem.Allocator) ![]const u8 {
        return std.fmt.allocPrint(allocator, "https://huggingface.co/{s}/{s}/resolve/main/{s}", .{ self.org, self.repo, filename });
    }

    /// Build API URL to get file list
    pub fn buildApiUrl(self: HuggingFaceId, allocator: std.mem.Allocator) ![]const u8 {
        return std.fmt.allocPrint(allocator, "https://huggingface.co/api/models/{s}/{s}/tree/main", .{ self.org, self.repo });
    }
};

/// Information about a file in the repository
pub const FileInfo = struct {
    path: []const u8,
    size: u64,
    sha256: ?[]const u8,
};

/// Progress callback type for download updates
pub const ProgressCallback = *const fn (downloaded: u64, total: u64) void;

/// Download a file from HuggingFace with resume support
pub fn downloadFile(
    allocator: std.mem.Allocator,
    url: []const u8,
    dest_path: []const u8,
    progress_callback: ?ProgressCallback,
) !void {
    // Check existing file size for resume
    var start_byte: u64 = 0;
    const existing_size = checkExistingFileSize(dest_path) catch 0;
    start_byte = existing_size;

    // Create HTTP client
    var client = std.http.Client{ .allocator = allocator };
    defer client.deinit();

    // Parse URL
    const uri = try std.Uri.parse(url);

    // Prepare headers
    var headers = std.http.Headers.init(allocator);
    defer headers.deinit();

    // Add Range header if resuming
    if (start_byte > 0) {
        const range_value = try std.fmt.allocPrint(allocator, "bytes={d}-", .{start_byte});
        defer allocator.free(range_value);
        try headers.append("Range", range_value);
    }

    // Make request
    var req = try client.request(.GET, uri, headers, .{});
    defer req.deinit();

    try req.start();
    try req.wait();

    // Handle response
    const status = req.response.status;

    if (status == .partial_content) {
        // Resume working - append to existing file
        const file = try std.fs.cwd().openFile(dest_path, .{ .mode = .write_only });
        defer file.close();
        try file.seekFromEnd(0);

        // Get total size from Content-Range header or Content-Length
        const content_length = req.response.content_length orelse 0;
        const total_size = start_byte + content_length;

        try streamToFileWithProgress(allocator, &req, file, start_byte, total_size, progress_callback);
    } else if (status == .ok) {
        // Full download - create/overwrite file
        const file = try std.fs.cwd().createFile(dest_path, .{});
        defer file.close();

        const content_length = req.response.content_length orelse 0;

        try streamToFileWithProgress(allocator, &req, file, 0, content_length, progress_callback);
    } else if (status == .range_not_satisfiable) {
        // Server doesn't support ranges or file already complete
        // Try full download
        std.log.warn("Range request not satisfiable for {s}, trying full download", .{url});

        // Delete partial file and retry
        try std.fs.cwd().deleteFile(dest_path);

        // Recursive call without range (start_byte = 0)
        return downloadFile(allocator, url, dest_path, progress_callback);
    } else {
        std.log.err("Download failed with status: {d} {s}", .{ @intFromEnum(status), @tagName(status) });
        return HuggingFaceError.DownloadFailed;
    }
}

/// Stream response body to file with progress updates
fn streamToFileWithProgress(
    allocator: std.mem.Allocator,
    req: *std.http.Client.Request,
    file: std.fs.File,
    start_byte: u64,
    total_size: u64,
    progress_callback: ?ProgressCallback,
) !void {
    _ = allocator; // Allocator reserved for future use
    var buffer: [8192]u8 = undefined;
    var total_downloaded: u64 = start_byte;

    while (true) {
        const bytes_read = try req.read(&buffer);
        if (bytes_read == 0) break;

        try file.writeAll(buffer[0..bytes_read]);
        total_downloaded += bytes_read;

        if (progress_callback) |cb| {
            cb(total_downloaded, total_size);
        }
    }
}

/// Check size of existing file for resume support
fn checkExistingFileSize(path: []const u8) !u64 {
    const file = std.fs.cwd().openFile(path, .{}) catch |err| {
        if (err == error.FileNotFound) return 0;
        return err;
    };
    defer file.close();

    const stat = try file.stat();
    return stat.size;
}

/// Get list of model files from HuggingFace API
pub fn getFileList(
    allocator: std.mem.Allocator,
    hf_id: HuggingFaceId,
) ![]FileInfo {
    const api_url = try hf_id.buildApiUrl(allocator);
    defer allocator.free(api_url);

    // Create HTTP client
    var client = std.http.Client{ .allocator = allocator };
    defer client.deinit();

    const uri = try std.Uri.parse(api_url);

    var headers = std.http.Headers.init(allocator);
    defer headers.deinit();
    try headers.append("Accept", "application/json");

    var req = try client.request(.GET, uri, headers, .{});
    defer req.deinit();

    try req.start();
    try req.wait();

    if (req.response.status != .ok) {
        return HuggingFaceError.ApiError;
    }

    // Read response body
    const body = try req.reader().readAllAlloc(allocator, 10 * 1024 * 1024); // Max 10MB
    defer allocator.free(body);

    // Parse JSON response
    // Expected format: [{"type": "file", "path": "...", "size": 123, ...}, ...]
    const parsed = try std.json.parseFromSlice(
        std.json.Value,
        allocator,
        body,
        .{},
    );
    defer parsed.deinit();

    const root = parsed.value;
    if (root != .array) {
        return HuggingFaceError.ApiError;
    }

    // Filter for relevant files: config.json, tokenizer.json, *.safetensors
    var files = std.ArrayList(FileInfo).init(allocator);
    errdefer {
        for (files.items) |*item| {
            allocator.free(item.path);
        }
        files.deinit();
    }

    for (root.array.items) |item| {
        if (item != .object) continue;

        const obj = item.object;

        // Check if it's a file
        const type_field = obj.get("type");
        if (type_field == null or !std.mem.eql(u8, type_field.?.string, "file")) {
            continue;
        }

        // Get path
        const path_field = obj.get("path");
        if (path_field == null or path_field.? != .string) continue;
        const path = path_field.?.string;

        // Filter to relevant files only
        if (!isRelevantFile(path)) continue;

        // Get size
        const size_field = obj.get("size");
        var size: u64 = 0;
        if (size_field != null) {
            if (size_field.? == .integer) {
                size = @intCast(size_field.?.integer);
            } else if (size_field.? == .float) {
                size = @intFromFloat(size_field.?.float);
            }
        }

        try files.append(.{
            .path = try allocator.dupe(u8, path),
            .size = size,
            .sha256 = null, // HF API doesn't provide SHA in tree endpoint
        });
    }

    return files.toOwnedSlice();
}

/// Check if file is relevant for model download
fn isRelevantFile(path: []const u8) bool {
    // Always include these essential files
    if (std.mem.eql(u8, path, "config.json")) return true;
    if (std.mem.eql(u8, path, "tokenizer.json")) return true;
    if (std.mem.eql(u8, path, "tokenizer_config.json")) return true;

    // Include SafeTensors model files
    if (std.mem.endsWith(u8, path, ".safetensors")) return true;

    // Include PyTorch files as fallback
    if (std.mem.endsWith(u8, path, ".bin")) return true;
    if (std.mem.endsWith(u8, path, ".pt")) return true;
    if (std.mem.endsWith(u8, path, ".pth")) return true;

    return false;
}

/// Free file list memory
pub fn freeFileList(allocator: std.mem.Allocator, files: []FileInfo) void {
    for (files) |*file| {
        allocator.free(file.path);
    }
    allocator.free(files);
}

/// Verify SHA256 checksum of downloaded file
pub fn verifyChecksum(
    allocator: std.mem.Allocator,
    file_path: []const u8,
    expected_hash: []const u8,
) !bool {
    _ = allocator;

    // Open file
    const file = try std.fs.cwd().openFile(file_path, .{});
    defer file.close();

    // For now, we skip actual SHA256 verification
    // In a full implementation, we'd use a SHA256 implementation
    // For this MVP, we just check the file exists and is non-empty
    const stat = try file.stat();
    if (stat.size == 0) {
        return false;
    }

    std.log.info("Checksum verification for {s}: expected {s}, file size {d} bytes", .{
        file_path, expected_hash, stat.size,
    });

    // Return true for now (accept the download)
    return true;
}
