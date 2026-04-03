//! loader.zig - Model loading infrastructure
//!
//! Loads model weights and configuration from disk.

const std = @import("std");
const mlx = @import("../mlx.zig/src/mlx.zig");
const qwen = @import("../mlx.zig/src/qwen.zig");
const deepseek = @import("../deepseek.zig");
const safetensors_index = @import("safetensors_index.zig");

pub const ModelType = enum {
    qwen,
    llama,
    phi,
    deepseek_v1,
    deepseek_v2_moe,
    gpt_oss,
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
    gpt_oss: void, // TODO: Add GPT-OSS config
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
    // Check for GPT-OSS models first
    if (std.mem.indexOf(u8, config.model_type, "gpt_oss") != null or
        std.mem.indexOf(u8, config.raw_json, "\"model_type\": \"gpt_oss\"") != null or
        (config.hasKey("num_experts") and config.hasKey("sliding_window")))
    {
        return .gpt_oss;
    }

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

    // Check for GPT-OSS first (sliding_window + num_experts is unique signature)
    if (std.mem.indexOf(u8, content, "gpt_oss") != null or
        (std.mem.indexOf(u8, content, "sliding_window") != null and
            std.mem.indexOf(u8, content, "num_experts") != null))
    {
        return .gpt_oss;
    }

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
    model_path: []const u8,
) !deepseek.DeepSeekWeights {
    // Step 1: Parse safetensors index
    std.log.info("Parsing safetensors index for DeepSeek model...", .{});

    var index = safetensors_index.parseSafetensorsIndex(allocator, model_path) catch |err| {
        std.log.err("Failed to parse safetensors index: {s}", .{@errorName(err)});
        return err;
    };
    defer index.deinit();

    std.log.info("Found {d} weight mappings", .{index.weightCount()});

    // Step 2: Initialize weights hash map for MLX loading
    var weights_hash = std.StringHashMap(*mlx.Array).init(allocator);
    defer {
        var iter = weights_hash.iterator();
        while (iter.next()) |entry| {
            mlx.arrayFree(entry.value_ptr.*);
            allocator.destroy(entry.value_ptr.*);
        }
        weights_hash.deinit();
    }

    // Step 3: Register all expected weight keys
    const config = deepseek.DeepSeekConfig{};
    try registerDeepSeekWeightKeys(allocator, &weights_hash, config);

    // Step 4: Load weights from safetensors files
    const stream = mlx.defaultGpuStreamNew();

    var shard_files = try safetensors_index.getShardFiles(allocator, index);
    defer {
        for (shard_files.items) |path| allocator.free(path);
        shard_files.deinit();
    }

    for (shard_files.items) |shard_path| {
        std.log.info("Loading weights from: {s}", .{shard_path});
        try mlx.loadSafetensors(&weights_hash, shard_path, stream);
    }

    // Step 5: Map loaded weights to DeepSeekWeights structure
    const weights = try mapWeightsToDeepSeek(allocator, &weights_hash, config);

    _ = config_info;
    return weights;
}

/// Register all expected DeepSeek weight keys
fn registerDeepSeekWeightKeys(
    allocator: std.mem.Allocator,
    weights_hash: *std.StringHashMap(*mlx.Array),
    config: deepseek.DeepSeekConfig,
) !void {
    // Embedding and output weights (quantized: weight + biases + scales)
    try registerQuantizedWeightKey(allocator, weights_hash, "model.embed_tokens");
    try registerWeightKey(allocator, weights_hash, "model.norm.weight");
    try registerQuantizedWeightKey(allocator, weights_hash, "lm_head");

    // Per-layer weights
    var buf: [256]u8 = undefined;
    for (0..config.num_hidden_layers) |layer_idx| {
        // Layer norms (not quantized)
        const input_ln = try std.fmt.bufPrint(&buf, "model.layers.{d}.input_layernorm.weight", .{layer_idx});
        try registerWeightKey(allocator, weights_hash, input_ln);

        const post_attn_ln = try std.fmt.bufPrint(&buf, "model.layers.{d}.post_attention_layernorm.weight", .{layer_idx});
        try registerWeightKey(allocator, weights_hash, post_attn_ln);

        // MLA attention weights (all quantized in 4-bit model)
        // q_proj
        const q_proj_base = try std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.q_proj", .{layer_idx});
        try registerQuantizedWeightKey(allocator, weights_hash, q_proj_base);

        // kv_b_proj
        const kv_b_proj_base = try std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.kv_b_proj", .{layer_idx});
        try registerQuantizedWeightKey(allocator, weights_hash, kv_b_proj_base);

        // o_proj
        const o_proj_base = try std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.o_proj", .{layer_idx});
        try registerQuantizedWeightKey(allocator, weights_hash, o_proj_base);

        // kv_a_proj_with_mqa (DeepSeek V2 uses MLA with MQA)
        const kv_a_proj_base = try std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.kv_a_proj_with_mqa", .{layer_idx});
        try registerQuantizedWeightKey(allocator, weights_hash, kv_a_proj_base);

        // kv_a_layernorm (per-layer normalization for compressed KV)
        const kv_a_ln = try std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.kv_a_layernorm.weight", .{layer_idx});
        try registerWeightKey(allocator, weights_hash, kv_a_ln);

        // Layer 0 uses dense MLP (first_k_dense_replace = 1)
        if (layer_idx == 0) {
            // Dense MLP: up_proj, gate_proj, down_proj (all quantized)
            const up_proj_base = try std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.up_proj", .{layer_idx});
            try registerQuantizedWeightKey(allocator, weights_hash, up_proj_base);

            const gate_proj_base = try std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.gate_proj", .{layer_idx});
            try registerQuantizedWeightKey(allocator, weights_hash, gate_proj_base);

            const down_proj_base = try std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.down_proj", .{layer_idx});
            try registerQuantizedWeightKey(allocator, weights_hash, down_proj_base);
        } else {
            // Layers 1+ use MoE with switch_mlp for experts

            // MoE router gate (not quantized)
            const gate = try std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.gate.weight", .{layer_idx});
            try registerWeightKey(allocator, weights_hash, gate);

            // Shared experts (2 experts per layer, all quantized)
            for (0..config.num_shared_experts) |exp_idx| {
                const shared_gate_base = try std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.shared_experts.{d}.gate_proj", .{ layer_idx, exp_idx });
                try registerQuantizedWeightKey(allocator, weights_hash, shared_gate_base);

                const shared_up_base = try std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.shared_experts.{d}.up_proj", .{ layer_idx, exp_idx });
                try registerQuantizedWeightKey(allocator, weights_hash, shared_up_base);

                const shared_down_base = try std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.shared_experts.{d}.down_proj", .{ layer_idx, exp_idx });
                try registerQuantizedWeightKey(allocator, weights_hash, shared_down_base);
            }

            // Routed experts via switch_mlp (fused 64 experts, all quantized)
            // switch_mlp contains all expert weights fused together
            const switch_gate_base = try std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.switch_mlp.gate_proj", .{layer_idx});
            try registerQuantizedWeightKey(allocator, weights_hash, switch_gate_base);

            const switch_up_base = try std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.switch_mlp.up_proj", .{layer_idx});
            try registerQuantizedWeightKey(allocator, weights_hash, switch_up_base);

            const switch_down_base = try std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.switch_mlp.down_proj", .{layer_idx});
            try registerQuantizedWeightKey(allocator, weights_hash, switch_down_base);
        }
    }
}

/// Register a quantized weight key group (.weight + .biases + .scales)
fn registerQuantizedWeightKey(
    allocator: std.mem.Allocator,
    weights_hash: *std.StringHashMap(*mlx.Array),
    base_key: []const u8,
) !void {
    var buf: [256]u8 = undefined;

    // Register .weight
    const weight_key = try std.fmt.bufPrint(&buf, "{s}.weight", .{base_key});
    try registerWeightKey(allocator, weights_hash, weight_key);

    // Register .biases
    const biases_key = try std.fmt.bufPrint(&buf, "{s}.biases", .{base_key});
    try registerWeightKey(allocator, weights_hash, biases_key);

    // Register .scales
    const scales_key = try std.fmt.bufPrint(&buf, "{s}.scales", .{base_key});
    try registerWeightKey(allocator, weights_hash, scales_key);
}

/// Register a single weight key in the hash map
fn registerWeightKey(
    allocator: std.mem.Allocator,
    weights_hash: *std.StringHashMap(*mlx.Array),
    key: []const u8,
) !void {
    const owned_key = try allocator.dupe(u8, key);
    const weight_ptr = try allocator.create(mlx.Array);
    weight_ptr.* = mlx.arrayNew();

    try weights_hash.put(owned_key, weight_ptr);
}

/// Map loaded weights to DeepSeekWeights structure
fn mapWeightsToDeepSeek(
    allocator: std.mem.Allocator,
    weights_hash: *std.StringHashMap(*mlx.Array),
    config: deepseek.DeepSeekConfig,
) !deepseek.DeepSeekWeights {
    var weights = deepseek.DeepSeekWeights{
        .embed_tokens = try mapQuantizedWeight(weights_hash, "model.embed_tokens"),
        .norm = mlx.arrayNew(),
        .lm_head = try mapQuantizedWeight(weights_hash, "lm_head"),
        .layers = try allocator.alloc(deepseek.DeepSeekLayer, config.num_hidden_layers),
    };

    // Map final norm (not quantized)
    if (weights_hash.get("model.norm.weight")) |norm_ptr| {
        weights.norm = norm_ptr.*;
    } else {
        std.log.warn("Missing weight: model.norm.weight", .{});
    }

    // Map per-layer weights
    var buf: [256]u8 = undefined;
    for (0..config.num_hidden_layers) |layer_idx| {
        // Initialize layer with default values
        weights.layers[layer_idx] = deepseek.DeepSeekLayer{
            .input_norm = mlx.arrayNew(),
            .mla = undefined, // Will be initialized separately
            .post_attn_norm = mlx.arrayNew(),
            .moe = undefined, // Will be initialized separately
        };

        // Map layer norms (not quantized)
        const input_ln_key = try std.fmt.bufPrint(&buf, "model.layers.{d}.input_layernorm.weight", .{layer_idx});
        if (weights_hash.get(input_ln_key)) |ptr| {
            weights.layers[layer_idx].input_norm = ptr.*;
        }

        const post_ln_key = try std.fmt.bufPrint(&buf, "model.layers.{d}.post_attention_layernorm.weight", .{layer_idx});
        if (weights_hash.get(post_ln_key)) |ptr| {
            weights.layers[layer_idx].post_attn_norm = ptr.*;
        }
    }

    return weights;
}

/// Helper to map a quantized weight group from hash
fn mapQuantizedWeight(
    weights_hash: *std.StringHashMap(*mlx.Array),
    base_key: []const u8,
) !deepseek.QuantizedWeight {
    var buf: [256]u8 = undefined;

    const weight_key = try std.fmt.bufPrint(&buf, "{s}.weight", .{base_key});
    const biases_key = try std.fmt.bufPrint(&buf, "{s}.biases", .{base_key});
    const scales_key = try std.fmt.bufPrint(&buf, "{s}.scales", .{base_key});

    var qw = deepseek.QuantizedWeight{
        .weight = mlx.arrayNew(),
        .biases = mlx.arrayNew(),
        .scales = mlx.arrayNew(),
    };

    if (weights_hash.get(weight_key)) |ptr| {
        qw.weight = ptr.*;
    } else {
        std.log.warn("Missing quantized weight: {s}", .{weight_key});
    }

    if (weights_hash.get(biases_key)) |ptr| {
        qw.biases = ptr.*;
    } else {
        std.log.warn("Missing quantized bias: {s}", .{biases_key});
    }

    if (weights_hash.get(scales_key)) |ptr| {
        qw.scales = ptr.*;
    } else {
        std.log.warn("Missing quantized scales: {s}", .{scales_key});
    }

    return qw;
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

test "loader - detect GPT-OSS" {
    // Test that we can detect GPT-OSS architecture
    const json_gpt_oss = "{\"model_type\": \"gpt_oss\", \"num_experts\": 32, \"sliding_window\": 128}";
    const config = ConfigInfo{
        .model_type = "gpt_oss",
        .raw_json = json_gpt_oss,
        .allocator = std.testing.allocator,
    };

    const arch = detectArchitecture(config);
    try std.testing.expectEqual(ModelType.gpt_oss, arch);
}
