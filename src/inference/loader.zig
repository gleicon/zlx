//! loader.zig - Model loading infrastructure
//!
//! Loads model weights and configuration from disk.

const std = @import("std");
const mlx = @import("../mlx.zig/src/mlx.zig");
const qwen = @import("../mlx.zig/src/qwen.zig");
const deepseek = @import("../deepseek.zig");

pub const ModelType = enum {
    qwen,
    llama,
    phi,
    deepseek_v1,
    deepseek_v2_moe,
};

/// Configuration info from config.json for architecture detection
pub const ConfigInfo = struct {
    model_type: []const u8,
    raw_json: []const u8,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *ConfigInfo) void {
        self.allocator.free(self.model_type);
        self.allocator.free(self.raw_json);
    }

    /// Check if config has a specific key
    pub fn hasKey(self: ConfigInfo, key: []const u8) bool {
        return std.mem.indexOf(u8, self.raw_json, key) != null;
    }
};

/// Model configuration wrapper that handles different model types
pub const ModelConfig = union(ModelType) {
    qwen: qwen.ModelConfig,
    llama: void, // TODO: Add llama ModelConfig
    phi: void, // TODO: Add phi ModelConfig
    deepseek_v1: void,
    deepseek_v2_moe: deepseek.DeepSeekConfig,
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

/// Load and parse config.json
fn loadConfigInfo(allocator: std.mem.Allocator, config_path: []const u8) !ConfigInfo {
    const file = try std.fs.cwd().openFile(config_path, .{});
    defer file.close();

    const content = try file.readToEndAlloc(allocator, 1024 * 1024);
    errdefer allocator.free(content);

    // Extract model_type from JSON
    var model_type: []const u8 = "";
    if (std.mem.indexOf(u8, content, "\"model_type\"") != null) {
        const start_idx = std.mem.indexOf(u8, content, "\"model_type\"").? + 13;
        const quote_start = std.mem.indexOfPos(u8, content, start_idx, "\"").? + 1;
        const quote_end = std.mem.indexOfPos(u8, content, quote_start, "\"").?;
        model_type = try allocator.dupe(u8, content[quote_start..quote_end]);
    } else {
        model_type = try allocator.dupe(u8, "unknown");
    }
    errdefer allocator.free(model_type);

    return ConfigInfo{
        .model_type = model_type,
        .raw_json = content,
        .allocator = allocator,
    };
}

/// Detect architecture from config info
pub fn detectArchitecture(config: ConfigInfo) ModelType {
    // Check for DeepSeek models
    if (std.mem.indexOf(u8, config.model_type, "deepseek")) |_| {
        // Check if V2 with MLA/MoE
        if (config.hasKey("num_experts") or config.hasKey("n_routed_experts")) {
            return .deepseek_v2_moe;
        }
        // Check for MLA specific keys
        if (config.hasKey("kv_lora_rank") or config.hasKey("q_lora_rank")) {
            return .deepseek_v2_moe;
        }
        return .deepseek_v1;
    }

    // Standard model detection
    if (std.mem.indexOf(u8, config.model_type, "qwen") != null) return .qwen;
    if (std.mem.indexOf(u8, config.model_type, "llama") != null) return .llama;
    if (std.mem.indexOf(u8, config.model_type, "phi") != null) return .phi;

    // Default to qwen
    return .qwen;
}

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
    if (std.mem.indexOf(u8, content, "deepseek") != null) {
        // Check for MoE indicators
        if (std.mem.indexOf(u8, content, "num_experts") != null or
            std.mem.indexOf(u8, content, "n_routed_experts") != null or
            std.mem.indexOf(u8, content, "kv_lora_rank") != null)
        {
            return .deepseek_v2_moe;
        }
        return .deepseek_v1;
    }

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

/// Load DeepSeek weights from model directory
pub fn loadDeepSeekWeights(
    allocator: std.mem.Allocator,
    config_info: ConfigInfo,
) !deepseek.DeepSeekWeights {
    _ = config_info;
    // TODO: Implement actual weight loading
    // For now, return a placeholder
    return deepseek.DeepSeekWeights{
        .token_embedding = mlx.arrayNew(),
        .layers = try allocator.alloc(deepseek.DeepSeekLayer, 27),
        .norm = mlx.arrayNew(),
        .lm_head = mlx.arrayNew(),
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

test "loader - detect DeepSeek V2 MoE" {
    // Test that we can detect DeepSeek V2 MoE architecture
    const json_with_moe = "{\"model_type\": \"deepseek\", \"num_experts\": 64}";
    const config = ConfigInfo{
        .model_type = "deepseek",
        .raw_json = json_with_moe,
        .allocator = std.testing.allocator,
    };

    const arch = detectArchitecture(config);
    try std.testing.expectEqual(ModelType.deepseek_v2_moe, arch);
}
