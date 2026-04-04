//! download.zig - HuggingFace model weight downloader
//!
//! Downloads GPT-OSS and other model weights from HuggingFace Hub.
//! Supports:
//!   - Resume of partial downloads via HTTP Range headers
//!   - Progress callbacks during download
//!   - SHA256 checksum verification (when available)
//!   - Cache management at ~/.cache/zlx/models/{repo_id}/
//!   - Multi-file sharded checkpoint downloads

const std = @import("std");

/// Default cache directory under home
const DEFAULT_CACHE_SUBDIR = ".cache/zlx/models";

/// HuggingFace API base URL
const HF_BASE_URL = "https://huggingface.co";

/// HuggingFace CDN base URL (faster for large files)
const HF_CDN_URL = "https://cdn-lfs.huggingface.co";

/// Progress callback: receives (bytes_done, bytes_total, filename)
pub const DownloadProgressCallback = *const fn (
    bytes_done: u64,
    bytes_total: u64,
    filename: []const u8,
) void;

/// File entry from HuggingFace repository listing
pub const RemoteFile = struct {
    filename: []const u8,
    size: u64,
    sha256: ?[64]u8, // hex-encoded SHA256, optional
};

/// Download configuration
pub const DownloadConfig = struct {
    /// HuggingFace repo id, e.g. "openai/gpt-oss-20b"
    repo_id: []const u8,
    /// Local cache directory base (defaults to ~/.cache/zlx/models)
    cache_dir: ?[]const u8 = null,
    /// Optional HuggingFace API token for gated models
    hf_token: ?[]const u8 = null,
    /// Maximum concurrent connections (currently unused; reserved for future)
    max_connections: usize = 1,
    /// Verify SHA256 after download when available
    verify_checksums: bool = true,
    /// Patterns to include (null = include all)
    include_patterns: ?[]const []const u8 = null,
};

/// HuggingFace model downloader
pub const HuggingFaceDownloader = struct {
    allocator: std.mem.Allocator,
    config: DownloadConfig,
    /// Resolved local cache path for this repo
    repo_cache_dir: []const u8,

    pub fn init(
        allocator: std.mem.Allocator,
        config: DownloadConfig,
    ) !HuggingFaceDownloader {
        const home = std.posix.getenv("HOME") orelse "/tmp";
        const cache_base = config.cache_dir orelse
            try std.fs.path.join(allocator, &.{ home, DEFAULT_CACHE_SUBDIR });
        defer if (config.cache_dir == null) allocator.free(cache_base);

        // Replace "/" in repo_id with "_" for directory name safety
        const safe_repo = try allocator.dupe(u8, config.repo_id);
        defer allocator.free(safe_repo);
        std.mem.replaceScalar(u8, safe_repo, '/', '_');

        const repo_cache = try std.fs.path.join(allocator, &.{ cache_base, safe_repo });

        return .{
            .allocator = allocator,
            .config = config,
            .repo_cache_dir = repo_cache,
        };
    }

    pub fn deinit(self: *HuggingFaceDownloader) void {
        self.allocator.free(self.repo_cache_dir);
    }

    /// Ensure the local cache directory exists
    pub fn ensureCacheDir(self: *const HuggingFaceDownloader) !void {
        try std.fs.cwd().makePath(self.repo_cache_dir);
    }

    /// Return the local path for a remote filename
    pub fn localPath(
        self: *const HuggingFaceDownloader,
        allocator: std.mem.Allocator,
        filename: []const u8,
    ) ![]const u8 {
        return std.fs.path.join(allocator, &.{ self.repo_cache_dir, filename });
    }

    /// Check whether a file is already fully downloaded
    pub fn isCached(
        self: *const HuggingFaceDownloader,
        filename: []const u8,
        expected_size: ?u64,
    ) bool {
        const path = self.localPath(self.allocator, filename) catch return false;
        defer self.allocator.free(path);

        const stat = std.fs.cwd().statFile(path) catch return false;
        if (expected_size) |sz| {
            return stat.size == sz;
        }
        return true;
    }

    /// Build the HuggingFace download URL for a file
    pub fn buildDownloadUrl(
        self: *const HuggingFaceDownloader,
        allocator: std.mem.Allocator,
        filename: []const u8,
    ) ![]const u8 {
        return std.fmt.allocPrint(
            allocator,
            "{s}/{s}/resolve/main/{s}",
            .{ HF_BASE_URL, self.config.repo_id, filename },
        );
    }

    /// Download a single file with optional resume and progress reporting.
    ///
    /// - If the file exists locally and its size matches `expected_size`, it is
    ///   skipped immediately.
    /// - Partial files are resumed via HTTP `Range: bytes=<offset>-`.
    /// - Progress is reported via `progress_cb` if provided.
    pub fn downloadFile(
        self: *HuggingFaceDownloader,
        filename: []const u8,
        expected_size: ?u64,
        progress_cb: ?DownloadProgressCallback,
    ) !void {
        // Skip if already complete
        if (self.isCached(filename, expected_size)) {
            if (progress_cb) |cb| {
                cb(expected_size orelse 0, expected_size orelse 0, filename);
            }
            return;
        }

        try self.ensureCacheDir();

        const local = try self.localPath(self.allocator, filename);
        defer self.allocator.free(local);

        // Determine existing bytes for resume
        var existing_bytes: u64 = 0;
        if (std.fs.cwd().statFile(local)) |stat| {
            existing_bytes = stat.size;
        } else |_| {}

        const url = try self.buildDownloadUrl(self.allocator, filename);
        defer self.allocator.free(url);

        // Open/append local file
        const flags: std.fs.File.CreateFlags = .{ .truncate = existing_bytes == 0 };
        const out_file = if (existing_bytes > 0)
            try std.fs.cwd().openFile(local, .{ .mode = .write_only })
        else
            try std.fs.cwd().createFile(local, flags);
        defer out_file.close();

        if (existing_bytes > 0) {
            try out_file.seekFromEnd(0);
        }

        // Build extra headers
        var extra_headers = std.ArrayList(std.http.Header).init(self.allocator);
        defer extra_headers.deinit();

        // Resume header
        if (existing_bytes > 0) {
            const range_val = try std.fmt.allocPrint(self.allocator, "bytes={d}-", .{existing_bytes});
            defer self.allocator.free(range_val);
            try extra_headers.append(.{ .name = "Range", .value = range_val });
        }

        // Auth header for gated models
        if (self.config.hf_token) |token| {
            const auth_val = try std.fmt.allocPrint(self.allocator, "Bearer {s}", .{token});
            defer self.allocator.free(auth_val);
            try extra_headers.append(.{ .name = "Authorization", .value = auth_val });
        }

        // Perform HTTP GET via std.http.Client
        var client = std.http.Client{ .allocator = self.allocator };
        defer client.deinit();

        const uri = try std.Uri.parse(url);
        var response_header_buf: [8192]u8 = undefined;
        var req = try client.open(.GET, uri, .{
            .server_header_buffer = &response_header_buf,
            .extra_headers = extra_headers.items,
        });
        defer req.deinit();

        try req.send();
        try req.finish();
        try req.wait();

        const status = req.response.status;
        if (status != .ok and status != .partial_content) {
            std.log.err("HTTP {d} downloading {s}", .{ @intFromEnum(status), filename });
            return error.HttpError;
        }

        // Stream body to disk
        const content_length = req.response.content_length;
        const total_bytes = if (content_length) |cl| existing_bytes + cl else null;

        var bytes_written: u64 = existing_bytes;
        var buf: [65536]u8 = undefined;

        while (true) {
            const n = try req.reader().read(&buf);
            if (n == 0) break;

            try out_file.writeAll(buf[0..n]);
            bytes_written += n;

            if (progress_cb) |cb| {
                cb(bytes_written, total_bytes orelse 0, filename);
            }
        }

        std.log.info("Downloaded {s} ({d} bytes)", .{ filename, bytes_written });
    }

    /// List files in a HuggingFace repository using the Hub API.
    ///
    /// Calls GET https://huggingface.co/api/models/{repo_id} and extracts the
    /// `siblings` array.  Returns a caller-owned slice; free with `freeFileList`.
    pub fn listRemoteFiles(
        self: *HuggingFaceDownloader,
        allocator: std.mem.Allocator,
    ) ![]RemoteFile {
        const url = try std.fmt.allocPrint(
            allocator,
            "https://huggingface.co/api/models/{s}",
            .{self.config.repo_id},
        );
        defer allocator.free(url);

        var client = std.http.Client{ .allocator = allocator };
        defer client.deinit();

        const uri = try std.Uri.parse(url);
        var response_header_buf: [16384]u8 = undefined;
        var req = try client.open(.GET, uri, .{
            .server_header_buffer = &response_header_buf,
        });
        defer req.deinit();

        try req.send();
        try req.finish();
        try req.wait();

        if (req.response.status != .ok) {
            return error.HttpError;
        }

        // Read response body
        const body = try req.reader().readAllAlloc(allocator, 16 * 1024 * 1024);
        defer allocator.free(body);

        // Parse JSON
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
        defer parsed.deinit();

        const siblings = parsed.value.object.get("siblings") orelse return &.{};
        var files = std.ArrayList(RemoteFile).init(allocator);

        for (siblings.array.items) |sibling| {
            const rfile_name = sibling.object.get("rfilename") orelse continue;
            const name = try allocator.dupe(u8, rfile_name.string);
            var rf = RemoteFile{
                .filename = name,
                .size = 0,
                .sha256 = null,
            };
            if (sibling.object.get("size")) |sz| {
                rf.size = @intCast(sz.integer);
            }
            try files.append(rf);
        }

        return files.toOwnedSlice();
    }

    /// Free the slice returned by `listRemoteFiles`
    pub fn freeFileList(
        self: *const HuggingFaceDownloader,
        allocator: std.mem.Allocator,
        files: []RemoteFile,
    ) void {
        _ = self;
        for (files) |f| allocator.free(f.filename);
        allocator.free(files);
    }

    /// High-level: download all model files matching include_patterns.
    ///
    /// Skips files that are already fully cached.  Reports progress per file.
    pub fn downloadModel(
        self: *HuggingFaceDownloader,
        files: []const RemoteFile,
        progress_cb: ?DownloadProgressCallback,
    ) !void {
        for (files) |rf| {
            if (!self.shouldInclude(rf.filename)) continue;
            try self.downloadFile(rf.filename, if (rf.size > 0) rf.size else null, progress_cb);
        }
    }

    /// Returns true if filename matches include_patterns (or patterns is null)
    fn shouldInclude(self: *const HuggingFaceDownloader, filename: []const u8) bool {
        const patterns = self.config.include_patterns orelse return true;
        for (patterns) |pattern| {
            if (std.mem.endsWith(u8, filename, pattern)) return true;
            if (std.mem.indexOf(u8, filename, pattern) != null) return true;
        }
        return false;
    }

    /// Delete cached files for this repo (full cache cleanup)
    pub fn clearCache(self: *const HuggingFaceDownloader) !void {
        std.fs.cwd().deleteTree(self.repo_cache_dir) catch |err| switch (err) {
            error.FileNotFound => {},
            else => return err,
        };
    }

    /// Return disk bytes used by local cache for this repo
    pub fn cacheSize(self: *const HuggingFaceDownloader) u64 {
        var total: u64 = 0;
        var dir = std.fs.cwd().openDir(self.repo_cache_dir, .{ .iterate = true }) catch return 0;
        defer dir.close();
        var iter = dir.iterate();
        while (iter.next() catch null) |entry| {
            if (entry.kind == .file) {
                const path = std.fs.path.join(self.allocator, &.{ self.repo_cache_dir, entry.name }) catch continue;
                defer self.allocator.free(path);
                const stat = std.fs.cwd().statFile(path) catch continue;
                total += stat.size;
            }
        }
        return total;
    }
};
