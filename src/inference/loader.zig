//! loader.zig - Model loading infrastructure
//!
//! Loads model weights and configuration from disk.

const std = @import("std");
const mlx = @import("../mlx.zig/src/mlx.zig");
const qwen = @import("../mlx.zig/src/qwen.zig");

pub const ModelType = enum {
    qwen,
    llama,
    phi,
};

/// Model configuration wrapper that handles different model types
pub const ModelConfig = union(ModelType) {
    qwen: qwen.ModelConfig,
    // llama: @import("../mlx.zig/src/llama.zig").ModelConfig,
    // phi: @import("../mlx.zig/src/phi.zig").ModelConfig,
};

/// Model loading error types
pub const LoadError = error{
    ModelNotFound,
    ConfigNotFound,
    WeightsNotFound,
    InvalidConfig,
    UnsupportedModelType,
};

/// Information about a model directory
pub const ModelInfo = struct {
    allocator: std.mem.Allocator,
    path: []const u8,
    model_type: ModelType,
    config: ModelConfig,

    pub fn deinit(self: *ModelInfo) void {
        self.allocator.free(self.path);
    }
};

/// Detect model type from config.json contents
fn detectModelType(config_path: []const u8) !ModelType {
    const file = try std.fs.cwd().openFile(config_path, .{});
    defer file.close();

    const content = try file.readToEndAlloc(std.heap.page_allocator, 1024 * 1024);
    defer std.heap.page_allocator.free(content);

    // Simple string-based detection
    if (std.mem.indexOf(u8, content, "Qwen") != null) return .qwen;
    if (std.mem.indexOf(u8, content, "Llama") != null) return .llama;
    if (std.mem.indexOf(u8, content, "Phi") != null) return .phi;

    // Default to qwen if can't detect
    return .qwen;
}

/// Check if a model directory exists and has required files
pub fn validateModelPath(path: []const u8) !ModelType {
    // Check directory exists
    var dir = std.fs.cwd().openDir(path, .{}) catch {
        return LoadError.ModelNotFound;
    };
    defer dir.close();

    // Check for config.json
    dir.access("config.json", .{}) catch {
        return LoadError.ConfigNotFound;
    };

    // Check for at least one safetensors file
    var has_weights = false;
    var iter = dir.iterate();
    while (try iter.next()) |entry| {
        if (std.mem.endsWith(u8, entry.name, ".safetensors")) {
            has_weights = true;
            break;
        }
    }

    if (!has_weights) {
        return LoadError.WeightsNotFound;
    }

    // Detect model type from config
    var buf: [1024]u8 = undefined;
    const config_path = try std.fmt.bufPrint(&buf, "{s}/config.json", .{path});
    return detectModelType(config_path);
}

/// Load model information from a path
pub fn loadModelInfo(allocator: std.mem.Allocator, path: []const u8) !ModelInfo {
    const model_type = try validateModelPath(path);

    const path_copy = try allocator.dupe(u8, path);
    errdefer allocator.free(path_copy);

    // For now, we only return type info.
    // The actual MLX config is loaded by the Transformer itself.
    return ModelInfo{
        .allocator = allocator,
        .path = path_copy,
        .model_type = model_type,
        .config = undefined, // Loaded by Transformer
    };
}

/// Default model paths relative to executable
pub fn getDefaultModelPath(name: []const u8) ![]const u8 {
    // Look in ./models/{name}
    var buf: [256]u8 = undefined;
    return try std.fmt.bufPrint(&buf, "./models/{s}", .{name});
}

test "loader - validate model path" {
    const allocator = std.testing.allocator;
    _ = allocator;

    // Test with non-existent path
    const result = validateModelPath("./models/non-existent-model");
    try std.testing.expectError(LoadError.ModelNotFound, result);
}
