//! manager.zig - Download manager for model downloads
//!
//! Manages download queue, progress tracking, and background downloads.
//! Integrates with the model registry to track download status.

const std = @import("std");
const huggingface = @import("huggingface.zig");
const registry = @import("../models/registry.zig");

/// Download status states
pub const DownloadStatus = enum {
    queued,
    downloading,
    completed,
    failed,
    cancelled,
};

/// Progress information for an active download
pub const DownloadProgress = struct {
    files_total: usize,
    files_completed: usize,
    bytes_downloaded: u64,
    bytes_total: u64,
    current_file: ?[]const u8,
    started_at: i64,
    updated_at: i64,
};

/// A single download task
pub const DownloadTask = struct {
    model_id: []const u8,
    hf_id: huggingface.HuggingFaceId,
    dest_dir: []const u8,
    status: DownloadStatus,
    progress: DownloadProgress,
    error_message: ?[]const u8,
    thread: ?std.Thread,
    cancelled: std.atomic.Value(bool),
    allocator: std.mem.Allocator,

    pub fn init(
        allocator: std.mem.Allocator,
        model_id: []const u8,
        hf_id: huggingface.HuggingFaceId,
        dest_dir: []const u8,
    ) !*DownloadTask {
        const task = try allocator.create(DownloadTask);

        // Copy strings
        const model_id_copy = try allocator.dupe(u8, model_id);
        errdefer allocator.free(model_id_copy);

        const org_copy = try allocator.dupe(u8, hf_id.org);
        errdefer allocator.free(org_copy);

        const repo_copy = try allocator.dupe(u8, hf_id.repo);
        errdefer allocator.free(repo_copy);

        const dest_dir_copy = try allocator.dupe(u8, dest_dir);
        errdefer allocator.free(dest_dir_copy);

        task.* = .{
            .model_id = model_id_copy,
            .hf_id = .{ .org = org_copy, .repo = repo_copy },
            .dest_dir = dest_dir_copy,
            .status = .queued,
            .progress = .{
                .files_total = 0,
                .files_completed = 0,
                .bytes_downloaded = 0,
                .bytes_total = 0,
                .current_file = null,
                .started_at = std.time.timestamp(),
                .updated_at = std.time.timestamp(),
            },
            .error_message = null,
            .thread = null,
            .cancelled = std.atomic.Value(bool).init(false),
            .allocator = allocator,
        };

        return task;
    }

    pub fn deinit(self: *DownloadTask) void {
        self.allocator.free(self.model_id);
        self.allocator.free(self.hf_id.org);
        self.allocator.free(self.hf_id.repo);
        self.allocator.free(self.dest_dir);
        if (self.error_message) |msg| {
            self.allocator.free(msg);
        }
        if (self.progress.current_file) |file| {
            self.allocator.free(file);
        }
        self.allocator.destroy(self);
    }
};

/// Download manager that handles queue and active downloads
pub const DownloadManager = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    queue: std.ArrayList(*DownloadTask),
    active: ?*DownloadTask,
    mutex: std.Thread.Mutex,
    registry: *registry.ModelRegistry,

    pub fn init(allocator: std.mem.Allocator, model_registry: *registry.ModelRegistry) Self {
        return .{
            .allocator = allocator,
            .queue = std.ArrayList(*DownloadTask).init(allocator),
            .active = null,
            .mutex = .{},
            .registry = model_registry,
        };
    }

    pub fn deinit(self: *Self) void {
        // Cancel active download
        if (self.active) |task| {
            self.cancelDownload(task.model_id) catch {};
        }

        // Cancel queued downloads
        for (self.queue.items) |task| {
            self.cancelDownload(task.model_id) catch {};
        }

        self.queue.deinit();
    }

    /// Queue a model for download
    pub fn queueDownload(self: *Self, model_id: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Parse HuggingFace ID
        const hf_id = try huggingface.HuggingFaceId.parse(model_id);

        // Create destination path: ~/.cache/zlx/models/{org}/{repo}/
        const home = std.process.getEnvVarOwned(self.allocator, "HOME") catch |err| {
            if (err == error.EnvironmentVariableNotFound) {
                std.log.err("HOME environment variable not set", .{});
                return error.HomeNotSet;
            }
            return err;
        };
        defer self.allocator.free(home);

        const dest_dir = try std.fmt.allocPrint(self.allocator, "{s}/.cache/zlx/models/{s}/{s}", .{
            home, hf_id.org, hf_id.repo,
        });
        defer self.allocator.free(dest_dir);

        // Create task
        const task = try DownloadTask.init(
            self.allocator,
            try self.allocator.dupe(u8, model_id),
            hf_id,
            try self.allocator.dupe(u8, dest_dir),
        );

        // Add to queue
        try self.queue.append(task);

        // Update registry
        try self.registry.insert(model_id, .{
            .id = try self.allocator.dupe(u8, model_id),
            .path = try self.allocator.dupe(u8, dest_dir),
            .memory_required_mb = 0, // Unknown until loaded
            .status = .loading,
            .architecture = .unknown,
        });

        std.log.info("Queued download for model: {s}", .{model_id});

        // If no active download, start this one
        if (self.active == null) {
            try self.startDownload(task);
        }
    }

    /// Start downloading a task
    fn startDownload(self: *Self, task: *DownloadTask) !void {
        task.status = .downloading;
        task.progress.started_at = std.time.timestamp();
        task.progress.updated_at = std.time.timestamp();
        self.active = task;

        // Spawn download thread
        task.thread = try std.Thread.spawn(.{}, downloadWorker, .{ self, task });

        std.log.info("Started download thread for model: {s}", .{task.model_id});
    }

    /// Download worker function (runs in separate thread)
    fn downloadWorker(self: *Self, task: *DownloadTask) void {
        defer {
            // Thread cleanup
            task.thread = null;
            self.active = null;

            // Process next queued download
            self.processNextQueued();
        }

        // Create destination directory
        std.fs.cwd().makePath(task.dest_dir) catch |err| {
            std.log.err("Failed to create destination directory {s}: {s}", .{
                task.dest_dir, @errorName(err),
            });
            task.status = .failed;
            task.error_message = task.allocator.dupe(u8, "Failed to create destination directory") catch null;
            self.registry.updateModelStatus(task.model_id, .failed);
            return;
        };

        // Get file list from HuggingFace
        const files = huggingface.getFileList(task.allocator, task.hf_id) catch |err| {
            std.log.err("Failed to get file list for {s}: {s}", .{
                task.model_id, @errorName(err),
            });
            task.status = .failed;
            task.error_message = task.allocator.dupe(u8, "Failed to get file list from HuggingFace") catch null;
            self.registry.updateModelStatus(task.model_id, .failed);
            return;
        };
        defer huggingface.freeFileList(task.allocator, files);

        task.progress.files_total = files.len;

        // Download each file
        for (files) |file_info| {
            // Check if cancelled
            if (task.cancelled.load(.seq_cst)) {
                task.status = .cancelled;
                self.registry.updateModelStatus(task.model_id, .failed);
                return;
            }

            // Update current file
            if (task.progress.current_file) |old| {
                task.allocator.free(old);
            }
            task.progress.current_file = task.allocator.dupe(u8, file_info.path) catch null;
            task.progress.bytes_total += file_info.size;
            task.progress.updated_at = std.time.timestamp();

            // Build URLs
            const download_url = task.hf_id.buildDownloadUrl(file_info.path, task.allocator) catch |err| {
                std.log.err("Failed to build download URL: {s}", .{@errorName(err)});
                continue;
            };
            defer task.allocator.free(download_url);

            const dest_path = std.fs.path.join(task.allocator, &.{ task.dest_dir, file_info.path }) catch |err| {
                std.log.err("Failed to build destination path: {s}", .{@errorName(err)});
                continue;
            };
            defer task.allocator.free(dest_path);

            // Download file with progress callback
            const progress_ctx = ProgressContext{
                .task = task,
            };

            huggingface.downloadFile(
                task.allocator,
                download_url,
                dest_path,
                makeProgressCallback(&progress_ctx),
            ) catch |err| {
                std.log.err("Failed to download {s}: {s}", .{ file_info.path, @errorName(err) });
                continue;
            };

            task.progress.files_completed += 1;
            task.progress.bytes_downloaded += file_info.size;
        }

        // Mark as complete
        task.status = .completed;
        self.registry.updateModelStatus(task.model_id, .available);
        std.log.info("Download completed for model: {s}", .{task.model_id});
    }

    /// Process next queued download
    fn processNextQueued(self: *Self) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.queue.items.len > 0) {
            const next_task = self.queue.orderedRemove(0);
            self.startDownload(next_task) catch |err| {
                std.log.err("Failed to start next download: {s}", .{@errorName(err)});
                next_task.status = .failed;
                self.registry.updateModelStatus(next_task.model_id, .failed);
            };
        }
    }

    /// Get progress for a specific download
    pub fn getProgress(self: *Self, model_id: []const u8) ?DownloadProgress {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Check active download
        if (self.active) |task| {
            if (std.mem.eql(u8, task.model_id, model_id)) {
                return task.progress;
            }
        }

        // Check queued downloads
        for (self.queue.items) |task| {
            if (std.mem.eql(u8, task.model_id, model_id)) {
                return task.progress;
            }
        }

        return null;
    }

    /// Cancel a download
    pub fn cancelDownload(self: *Self, model_id: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Check active download
        if (self.active) |task| {
            if (std.mem.eql(u8, task.model_id, model_id)) {
                task.cancelled.store(true, .seq_cst);
                if (task.thread) |thread| {
                    thread.join();
                }
                task.status = .cancelled;
                self.registry.updateModelStatus(model_id, .failed);
                return;
            }
        }

        // Check queued downloads
        for (self.queue.items, 0..) |task, idx| {
            if (std.mem.eql(u8, task.model_id, model_id)) {
                task.status = .cancelled;
                _ = self.queue.orderedRemove(idx);
                self.registry.updateModelStatus(model_id, .failed);
                return;
            }
        }

        return error.DownloadNotFound;
    }
};

/// Context for progress callback
const ProgressContext = struct {
    task: *DownloadTask,
};

/// Make a progress callback from context
fn makeProgressCallback(ctx: *ProgressContext) ?huggingface.ProgressCallback {
    const S = struct {
        fn cb(downloaded: u64, total: u64) void {
            _ = downloaded;
            _ = total;
            // Progress updates happen through the task struct
        }
    };

    _ = ctx;
    return S.cb;
}

// Global download manager instance
var global_manager: ?*DownloadManager = null;

/// Initialize global download manager
pub fn initGlobalManager(allocator: std.mem.Allocator, model_registry: *registry.ModelRegistry) !void {
    const manager = try allocator.create(DownloadManager);
    manager.* = DownloadManager.init(allocator, model_registry);
    global_manager = manager;
}

/// Deinitialize global download manager
pub fn deinitGlobalManager(allocator: std.mem.Allocator) void {
    if (global_manager) |manager| {
        manager.deinit();
        allocator.destroy(manager);
        global_manager = null;
    }
}

/// Get global download manager
pub fn getGlobalManager() ?*DownloadManager {
    return global_manager;
}
