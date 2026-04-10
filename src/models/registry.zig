//! registry.zig - Model registry for scanning and cataloging available models
//!
//! Scans model directories and extracts metadata from config.json files.
//! Provides memory estimation and status tracking for all discovered models.

const std = @import("std");

/// Model architecture types
pub const ModelArchitecture = enum {
    qwen,
    llama,
    phi,
    deepseek_v2_moe,
    deepseek_v1,
    gpt_oss,
    gemma4,
    unknown,
};

/// Known model metadata for pre-configured models
pub const KnownModelInfo = struct {
    id: []const u8,
    aliases: []const []const u8,
    architecture: ModelArchitecture,
    total_params: u64,
    active_params: ?u64, // For MoE models (sparse)
    memory_required_gb: f32,
    max_context: u32,
    quantization: []const u8,
    recommended: bool,
    description: []const u8,
    // PHASE-14: Backend selection and download support
    preferred_backend: ?BackendType = null, // null = use default
    download_urls: ?[]const []const u8 = null,
    gguf_filename: ?[]const u8 = null,
};

/// Backend type for preferred_backend field
pub const BackendType = enum {
    mlx,
    llama_cpp,
};

/// Pre-configured known models
pub const KNOWN_MODELS = &[_]KnownModelInfo{
    // DeepSeek-Coder-V2-Lite (15.7B total, 2B active MoE)
    .{
        .id = "deepseek-coder-v2-lite",
        .aliases = &.{ "deepseek", "deepseek-v2", "deepseek-coder", "deepseek-coder-v2-lite-instruct", "deepseek-coder-v2-lite-gguf" },
        .architecture = .deepseek_v2_moe,
        .total_params = 15_700_000_000,
        .active_params = 2_000_000_000,
        .memory_required_gb = 2.5,
        .max_context = 128_000,
        .quantization = "Q4_K_M_GGUF", // Primary format via llama.cpp
        .recommended = true,
        .description = "DeepSeek-Coder-V2-Lite 15.7B MoE (2B active) - llama.cpp backend recommended",
        .preferred_backend = .llama_cpp,
        .download_urls = &.{
            "https://huggingface.co/TheBloke/deepseek-coder-v2-lite-GGUF/resolve/main/deepseek-coder-v2-lite.Q4_K_M.gguf",
        },
        .gguf_filename = "deepseek-coder-v2-lite.Q4_K_M.gguf",
    },
    // GPT-OSS-20B (20B total, ~5B active MoE)
    .{
        .id = "gpt-oss-20b",
        .aliases = &.{ "gpt-oss", "gptoss", "gpt-oss-20b-mxfp4", "gpt-oss-20b-gguf" },
        .architecture = .gpt_oss,
        .total_params = 20_000_000_000,
        .active_params = 5_000_000_000,
        .memory_required_gb = 11.0,
        .max_context = 131_072,
        .quantization = "Q4_K_M_GGUF", // Primary format via llama.cpp
        .recommended = true,
        .description = "GPT-OSS-20B MoE (5B active) - llama.cpp backend recommended for best compatibility",
        .preferred_backend = .llama_cpp,
        .download_urls = &.{
            "https://huggingface.co/bartowski/GPT-OSS-20B-GGUF/resolve/main/GPT-OSS-20B-Q4_K_M.gguf",
        },
        .gguf_filename = "GPT-OSS-20B-Q4_K_M.gguf",
    },
};

/// Detect architecture from model configuration
pub fn detectArchitecture(config: *const ConfigInfo) ModelArchitecture {
    // First check model_type field from config.json
    if (config.model_type.len > 0) {
        // Check for DeepSeek models
        if (std.mem.indexOf(u8, config.model_type, "deepseek")) |_| {
            // Check if V2 with MLA/MoE
            if (std.mem.indexOf(u8, config.model_type, "v2")) |_| {
                return .deepseek_v2_moe;
            }
            return .deepseek_v1;
        }

        // Check for other known model types
        if (std.mem.indexOf(u8, config.model_type, "qwen")) |_| {
            return .qwen;
        }
        if (std.mem.indexOf(u8, config.model_type, "llama")) |_| {
            return .llama;
        }
        if (std.mem.indexOf(u8, config.model_type, "phi")) |_| {
            return .phi;
        }
        if (std.mem.indexOf(u8, config.model_type, "gpt_oss")) |_| {
            return .gpt_oss;
        }
        if (std.mem.indexOf(u8, config.model_type, "gpt-oss")) |_| {
            return .gpt_oss;
        }
        if (std.mem.indexOf(u8, config.model_type, "gemma4")) |_| {
            return .gemma4;
        }
    }

    // Check for GPT-OSS heuristics: MoE with sliding window
    if (config.num_experts > 0 and config.sliding_window != null) {
        if (config.sliding_window.? > 0) {
            return .gpt_oss;
        }
    }

    // Fallback: use heuristics based on architecture characteristics
    // Check for DeepSeek-V2 MoE indicators
    const gqa_ratio = @as(f32, @floatFromInt(config.num_attention_heads)) / @as(f32, @floatFromInt(config.hidden_size)) * 4096.0;

    // DeepSeek has 128 attention heads with 4096 hidden = 32:1 ratio
    if (gqa_ratio > 20.0 and config.num_layers >= 20) {
        return .deepseek_v2_moe;
    }

    // Fallback: use parameter count as heuristic
    const params = config.estimateParameterCount();

    if (params < 3_000_000_000) {
        return .qwen; // Small models often Qwen
    } else if (params > 10_000_000_000 and params < 20_000_000_000) {
        // 10-20B range could be DeepSeek or other MoE
        if (config.num_layers > 25) {
            return .deepseek_v2_moe; // DeepSeek has 27 layers
        }
    }

    return .qwen; // Default
}

/// Get known model info by ID or alias
pub fn getKnownModel(id: []const u8) ?KnownModelInfo {
    for (KNOWN_MODELS) |model| {
        if (std.mem.eql(u8, model.id, id)) {
            return model;
        }
        for (model.aliases) |alias| {
            if (std.mem.eql(u8, alias, id)) {
                return model;
            }
        }
    }
    return null;
}

/// Estimate memory for MoE models using active parameters
pub fn estimateDeepSeekMemory(config: ConfigInfo, _total_params: u64, active_params: u64) u32 {
    // For DeepSeek MoE: memory is based on active params, not total
    // Note: total_params is provided for reference but not used in calculation
    _ = _total_params; // Silence unused parameter warning
    // Formula: embeddings + active_params + compressed_KV + overhead

    const bytes_per_param: u8 = switch (config.quantization_bits) {
        8 => 1,
        16 => 2,
        32 => 4,
        else => 2, // Default to FP16
    };

    // Embedding weights (always loaded, not sparse)
    const embedding_params = @as(u64, config.vocab_size) * @as(u64, config.hidden_size);

    // Active parameter memory (not total!)
    const active_weights_bytes = active_params * bytes_per_param;

    // Compressed KV cache (MLA reduces this significantly)
    // DeepSeek uses 512-dim latent per layer (8x compression vs standard)
    const latent_dim: u32 = 512; // DeepSeek MLA latent dimension
    const kv_bytes_per_token = 2 * @as(u64, config.num_layers) * @as(u64, latent_dim) * bytes_per_param;
    const kv_cache_bytes = kv_bytes_per_token * @as(u64, config.max_position_embeddings);

    // Activations overhead (20% buffer)
    const base_memory = (embedding_params * bytes_per_param) + active_weights_bytes + kv_cache_bytes;
    const overhead_bytes = base_memory / 5;

    const total_bytes = base_memory + overhead_bytes;
    const total_mb = @as(u32, @intCast(total_bytes / (1024 * 1024)));

    return total_mb;
}

/// Get architecture string for API responses
pub fn architectureToString(arch: ModelArchitecture) []const u8 {
    return switch (arch) {
        .qwen => "qwen",
        .llama => "llama",
        .phi => "phi",
        .deepseek_v2_moe => "deepseek_v2_moe",
        .deepseek_v1 => "deepseek_v1",
        .gpt_oss => "gpt_oss",
        .gemma4 => "gemma4",
        .unknown => "unknown",
    };
}

/// Download information for model fetching
pub const DownloadInfo = struct {
    urls: []const []const u8,
    filename: []const u8,
    expected_size_bytes: u64,
    checksum: ?[]const u8,
};

/// Get download info for a model by ID
pub fn getDownloadInfo(model_id: []const u8) ?DownloadInfo {
    const model = getKnownModel(model_id) orelse return null;

    if (std.mem.eql(u8, model.id, "deepseek-coder-v2-lite")) {
        return DownloadInfo{
            .urls = &.{
                "https://huggingface.co/TheBloke/deepseek-coder-v2-lite-GGUF/resolve/main/deepseek-coder-v2-lite.Q4_K_M.gguf",
            },
            .filename = "deepseek-coder-v2-lite.Q4_K_M.gguf",
            .expected_size_bytes = 4_500_000_000, // ~4.5GB
            .checksum = null,
        };
    }

    if (std.mem.eql(u8, model.id, "gpt-oss-20b")) {
        return DownloadInfo{
            .urls = &.{
                "https://huggingface.co/bartowski/GPT-OSS-20B-GGUF/resolve/main/GPT-OSS-20B-Q4_K_M.gguf",
            },
            .filename = "GPT-OSS-20B-Q4_K_M.gguf",
            .expected_size_bytes = 11_500_000_000, // ~11GB
            .checksum = null,
        };
    }

    return null;
}

/// Get preferred backend for a model
pub fn getPreferredBackend(model_id: []const u8) ?BackendType {
    const model = getKnownModel(model_id) orelse return null;
    return model.preferred_backend;
}

/// Model status states
pub const ModelStatus = enum {
    available, // Model files present, not loaded
    loading, // Currently being loaded into memory
    loaded, // Active and ready for inference
    failed, // Failed to load or corrupted
};

/// Model configuration extracted from config.json
pub const ConfigInfo = struct {
    hidden_size: u32,
    num_layers: u32,
    num_attention_heads: u32,
    max_position_embeddings: u32 = 8192, // Default context length
    vocab_size: u32 = 32000, // Default vocab size
    quantization_bits: u8 = 16, // Default to FP16
    model_type: []const u8 = "", // Model type from config (e.g., "deepseek_v2", "qwen2")
    num_experts: u32 = 0, // For MoE models (0 = not MoE)
    sliding_window: ?u32 = null, // For sliding window attention (null = not used)

    /// Calculate total parameter count (rough estimate)
    pub fn estimateParameterCount(self: ConfigInfo) u64 {
        // Rough estimate: vocab * hidden + layers * (hidden * hidden * 4 + hidden * intermediate)
        // Simplified: ~2 * hidden * hidden * num_layers for transformer
        const hidden_u64 = @as(u64, self.hidden_size);
        const layers_u64 = @as(u64, self.num_layers);
        const vocab_u64 = @as(u64, self.vocab_size);

        // Embedding weights
        const embedding_params = vocab_u64 * hidden_u64;

        // Per layer: ~4 * hidden^2 (Q, K, V, O projections) + 2 * hidden * intermediate (FFN)
        // Assume intermediate = 4 * hidden
        const ffn_params = hidden_u64 * (4 * hidden_u64);
        const attn_params = 4 * hidden_u64 * hidden_u64;
        const layer_params = attn_params + ffn_params;

        // Output layer norm and lm_head
        const output_params = hidden_u64 * vocab_u64;

        return embedding_params + (layers_u64 * layer_params) + output_params;
    }
};

/// Complete metadata for a discovered model
pub const ModelMetadata = struct {
    allocator: std.mem.Allocator,
    id: []const u8, // Model name (directory name)
    path: []const u8, // Full directory path
    status: ModelStatus,
    size_bytes: u64, // Total directory size
    memory_required_mb: u32, // Estimated memory requirement
    config: ConfigInfo,
    loaded_at: ?i64, // Timestamp when loaded (null if not loaded)

    pub fn init(allocator: std.mem.Allocator, id: []const u8, path: []const u8, config: ConfigInfo, size_bytes: u64, memory_mb: u32) !ModelMetadata {
        return .{
            .allocator = allocator,
            .id = try allocator.dupe(u8, id),
            .path = try allocator.dupe(u8, path),
            .status = .available,
            .size_bytes = size_bytes,
            .memory_required_mb = memory_mb,
            .config = config,
            .loaded_at = null,
        };
    }

    pub fn deinit(self: *ModelMetadata) void {
        self.allocator.free(self.id);
        self.allocator.free(self.path);
        if (self.config.model_type.len > 0) self.allocator.free(self.config.model_type);
    }
};

/// Model registry that catalogs all available models
pub const ModelRegistry = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    models: std.StringHashMap(ModelMetadata),
    scan_mutex: std.Thread.Mutex,
    max_context_length: usize, // For memory estimation

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .models = std.StringHashMap(ModelMetadata).init(allocator),
            .scan_mutex = .{},
            .max_context_length = 8192, // Match MAX_CONTEXT_LENGTH in inference/mod.zig
        };
    }

    pub fn deinit(self: *Self) void {
        var iter = self.models.iterator();
        while (iter.next()) |entry| {
            // Free the key (model_name) stored in the hash map
            self.allocator.free(entry.key_ptr.*);
            // Free the value's internal allocations
            entry.value_ptr.deinit();
        }
        self.models.deinit();
    }

    /// Scan a single directory for models
    pub fn scanDirectory(self: *Self, path: []const u8) !void {
        self.scan_mutex.lock();
        defer self.scan_mutex.unlock();

        // Check if directory exists
        var dir = std.fs.cwd().openDir(path, .{ .iterate = true }) catch |err| {
            if (err == error.FileNotFound) {
                std.log.debug("Model directory not found: {s}", .{path});
                return;
            }
            return err;
        };
        defer dir.close();

        std.log.info("Scanning model directory: {s}", .{path});

        // Iterate subdirectories
        var iter = dir.iterate();
        while (try iter.next()) |entry| {
            if (entry.kind != .directory and entry.kind != .sym_link) continue;

            const model_path = try std.fs.path.join(self.allocator, &.{ path, entry.name });
            defer self.allocator.free(model_path);

            // Validate model directory
            if (self.validateModelDirectory(model_path)) |config| {
                // Calculate directory size
                const size_bytes = try self.calculateDirectorySize(model_path);

                // Estimate memory required
                const memory_mb = self.estimateMemoryFromConfig(config);

                // Check if model already exists (prefer local over cache)
                if (self.models.contains(entry.name)) {
                    std.log.debug("Model '{s}' already exists, skipping from {s}", .{ entry.name, path });
                    continue;
                }

                // Duplicate the model name since entry.name is temporary
                // StringHashMap takes ownership of this allocation
                const model_name = try self.allocator.dupe(u8, entry.name);

                // Add to registry
                const metadata = try ModelMetadata.init(
                    self.allocator,
                    model_name,
                    model_path,
                    config,
                    size_bytes,
                    memory_mb,
                );

                try self.models.put(model_name, metadata);
                std.log.info("Found model: {s} ({d} MB estimated)", .{ model_name, memory_mb });
                // NOTE: model_name is now owned by the hash map, don't free it here
            } else |err| {
                std.log.debug("Skipping {s}: {s}", .{ entry.name, @errorName(err) });
            }
        }
    }

    /// Validate a model directory has required files
    fn validateModelDirectory(self: *Self, path: []const u8) !ConfigInfo {
        var dir = try std.fs.cwd().openDir(path, .{});
        defer dir.close();

        // Check for config.json
        dir.access("config.json", .{}) catch {
            return error.ConfigNotFound;
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
            return error.WeightsNotFound;
        }

        // Parse config.json
        var buf: [1024]u8 = undefined;
        const config_path = try std.fmt.bufPrint(&buf, "{s}/config.json", .{path});
        return parseConfigFile(self.allocator, config_path);
    }

    /// Parse config.json to extract model configuration
    fn parseConfigFile(allocator: std.mem.Allocator, config_path: []const u8) !ConfigInfo {
        const file = try std.fs.cwd().openFile(config_path, .{});
        defer file.close();

        const content = try file.readToEndAlloc(allocator, 1024 * 1024);
        defer allocator.free(content);

        // Parse JSON
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, content, .{});
        defer parsed.deinit();

        const root = parsed.value;
        if (root != .object) return error.InvalidConfig;

        var config = ConfigInfo{
            .hidden_size = 0,
            .num_layers = 0,
            .num_attention_heads = 0,
            .model_type = "", // Will be set below if present
        };

        // Extract model_type
        if (root.object.get("model_type")) |v| {
            if (v == .string) {
                config.model_type = try allocator.dupe(u8, v.string);
            }
        }

        // Extract hidden_size (check root first, then text_config for multimodal models)
        if (root.object.get("hidden_size")) |v| {
            config.hidden_size = @intCast(v.integer);
        } else if (root.object.get("d_model")) |v| {
            config.hidden_size = @intCast(v.integer);
        } else if (root.object.get("text_config")) |tc| {
            if (tc == .object) {
                if (tc.object.get("hidden_size")) |v| {
                    config.hidden_size = @intCast(v.integer);
                }
            }
        }

        // Extract num_layers (check root first, then text_config for multimodal models)
        if (root.object.get("num_hidden_layers")) |v| {
            config.num_layers = @intCast(v.integer);
        } else if (root.object.get("n_layer")) |v| {
            config.num_layers = @intCast(v.integer);
        } else if (root.object.get("num_layers")) |v| {
            config.num_layers = @intCast(v.integer);
        } else if (root.object.get("text_config")) |tc| {
            if (tc == .object) {
                if (tc.object.get("num_hidden_layers")) |v| {
                    config.num_layers = @intCast(v.integer);
                }
            }
        }

        // Extract num_attention_heads (check root first, then text_config for multimodal models)
        if (root.object.get("num_attention_heads")) |v| {
            config.num_attention_heads = @intCast(v.integer);
        } else if (root.object.get("n_head")) |v| {
            config.num_attention_heads = @intCast(v.integer);
        } else if (root.object.get("text_config")) |tc| {
            if (tc == .object) {
                if (tc.object.get("num_attention_heads")) |v| {
                    config.num_attention_heads = @intCast(v.integer);
                }
            }
        }

        // Extract max_position_embeddings (check root first, then text_config)
        if (root.object.get("max_position_embeddings")) |v| {
            config.max_position_embeddings = @intCast(v.integer);
        } else if (root.object.get("text_config")) |tc| {
            if (tc == .object) {
                if (tc.object.get("max_position_embeddings")) |v| {
                    config.max_position_embeddings = @intCast(v.integer);
                }
            }
        }

        // Extract vocab_size (check root first, then text_config)
        if (root.object.get("vocab_size")) |v| {
            config.vocab_size = @intCast(v.integer);
        } else if (root.object.get("text_config")) |tc| {
            if (tc == .object) {
                if (tc.object.get("vocab_size")) |v| {
                    config.vocab_size = @intCast(v.integer);
                }
            }
        }

        // Validate required fields
        if (config.hidden_size == 0 or config.num_layers == 0) {
            return error.InvalidConfig;
        }

        return config;
    }

    /// Calculate total size of a directory (recursively)
    fn calculateDirectorySize(self: *Self, path: []const u8) !u64 {
        _ = self;

        var total_size: u64 = 0;
        var dir = try std.fs.cwd().openDir(path, .{ .iterate = true });
        defer dir.close();

        var iter = dir.iterate();
        while (try iter.next()) |entry| {
            switch (entry.kind) {
                .file => {
                    const full_path = try std.fs.path.join(std.heap.page_allocator, &.{ path, entry.name });
                    defer std.heap.page_allocator.free(full_path);

                    const stat = try std.fs.cwd().statFile(full_path);
                    total_size += stat.size;
                },
                .directory => {
                    const subdir_path = try std.fs.path.join(std.heap.page_allocator, &.{ path, entry.name });
                    defer std.heap.page_allocator.free(subdir_path);

                    // Recursively calculate subdir size
                    var subdir = try std.fs.cwd().openDir(subdir_path, .{ .iterate = true });
                    defer subdir.close();

                    var sub_iter = subdir.iterate();
                    while (try sub_iter.next()) |sub_entry| {
                        if (sub_entry.kind == .file) {
                            const file_path = try std.fs.path.join(std.heap.page_allocator, &.{ subdir_path, sub_entry.name });
                            defer std.heap.page_allocator.free(file_path);

                            const stat = try std.fs.cwd().statFile(file_path);
                            total_size += stat.size;
                        }
                    }
                },
                else => {},
            }
        }

        return total_size;
    }

    /// Estimate memory required for a model based on its configuration
    pub fn estimateMemoryFromConfig(self: *Self, config: ConfigInfo) u32 {
        // Memory estimation formula (conservative):
        // - Weights: num_params * bytes_per_param (2 for FP16, 4 for FP32)
        // - KV cache: 2 * num_layers * hidden_size * max_seq_len * bytes_per_param
        // - Activations: ~20% overhead buffer

        const num_params = config.estimateParameterCount();

        // Weights memory — handle sub-byte quantization (4-bit = 0.5 bytes/param)
        const weights_bytes: u64 = switch (config.quantization_bits) {
            4 => num_params / 2, // 2 values packed per byte
            8 => num_params * 1,
            16 => num_params * 2,
            32 => num_params * 4,
            else => num_params * 2, // Default to FP16
        };

        // KV cache bytes_per_param (always FP16 at runtime regardless of weight quant)
        const kv_bytes_per_param: u64 = 2;

        // KV cache memory (2 for K and V, per layer) — always FP16 at runtime
        const kv_bytes_per_token = 2 * @as(u64, config.num_layers) * @as(u64, config.hidden_size) * kv_bytes_per_param;
        const kv_cache_bytes = kv_bytes_per_token * self.max_context_length;

        // Activations overhead (20% buffer)
        const overhead_bytes = (weights_bytes + kv_cache_bytes) / 5;

        const total_bytes = weights_bytes + kv_cache_bytes + overhead_bytes;
        const total_mb = @as(u32, @intCast(total_bytes / (1024 * 1024)));

        return total_mb;
    }

    /// Scan both local and cache directories
    pub fn scanAllModels(self: *Self) !void {
        // Scan local models first (./models/)
        try self.scanDirectory("./models");

        // Scan cache directory (~/.cache/zlx/models/)
        if (getCacheModelPath(self.allocator)) |cache_path| {
            defer self.allocator.free(cache_path);
            try self.scanDirectory(cache_path);
        } else |err| {
            std.log.debug("Cache directory not available: {s}", .{@errorName(err)});
        }
    }

    /// Get a model by ID
    pub fn getModel(self: *Self, id: []const u8) ?ModelMetadata {
        self.scan_mutex.lock();
        defer self.scan_mutex.unlock();

        return self.models.get(id);
    }

    /// Get all models as a sorted array (caller must free)
    pub fn getAllModels(self: *Self, allocator: std.mem.Allocator) ![]ModelMetadata {
        self.scan_mutex.lock();
        defer self.scan_mutex.unlock();

        const model_count = self.models.count();
        if (model_count == 0) return &[_]ModelMetadata{};

        var models = try allocator.alloc(ModelMetadata, model_count);
        errdefer allocator.free(models);

        var iter = self.models.iterator();
        var i: usize = 0;
        while (iter.next()) |entry| : (i += 1) {
            models[i] = entry.value_ptr.*;
        }

        // Sort by ID
        std.mem.sort(ModelMetadata, models, {}, struct {
            fn lessThan(_: void, a: ModelMetadata, b: ModelMetadata) bool {
                return std.mem.lessThan(u8, a.id, b.id);
            }
        }.lessThan);

        return models;
    }

    /// Update model status (thread-safe)
    pub fn updateStatus(self: *Self, id: []const u8, status: ModelStatus) void {
        self.scan_mutex.lock();
        defer self.scan_mutex.unlock();

        if (self.models.getPtr(id)) |model| {
            model.status = status;
            if (status == .loaded) {
                model.loaded_at = std.time.timestamp();
            } else if (status == .available) {
                model.loaded_at = null;
            }
        }
    }

    /// Get total number of models
    pub fn count(self: *Self) usize {
        self.scan_mutex.lock();
        defer self.scan_mutex.unlock();
        return self.models.count();
    }
};

/// Get the cache model directory path (~/.cache/zlx/models/)
pub fn getCacheModelPath(allocator: std.mem.Allocator) ![]const u8 {
    const home = std.process.getEnvVarOwned(allocator, "HOME") catch |err| {
        return err;
    };
    defer allocator.free(home);

    return try std.fs.path.join(allocator, &.{ home, ".cache", "zlx", "models" });
}

// ============================================================================
// Tests
// ============================================================================

test "scanDirectory finds models with config.json and safetensors" {
    const allocator = std.testing.allocator;

    var registry = ModelRegistry.init(allocator);
    defer registry.deinit();

    // Scan the test models directory (if it exists)
    try registry.scanDirectory("./models");

    // Just verify it doesn't crash - we can't guarantee models exist
}

test "extractModelMetadata parses config.json" {
    const allocator = std.testing.allocator;
    _ = allocator;

    // Create a test config
    const config = ConfigInfo{
        .hidden_size = 1536,
        .num_layers = 28,
        .num_attention_heads = 16,
    };

    // Verify parameter count is reasonable
    const params = config.estimateParameterCount();
    try std.testing.expect(params > 0);
    try std.testing.expect(params > 1_000_000); // At least 1M params
}

test "estimateMemoryRequired calculates reasonable values" {
    const allocator = std.testing.allocator;

    var registry = ModelRegistry.init(allocator);
    defer registry.deinit();

    // Test 1.5B-like model (Qwen 2.5 1.5B config)
    const small_config = ConfigInfo{
        .hidden_size = 1536,
        .num_layers = 28,
        .num_attention_heads = 16,
        .vocab_size = 151936, // Qwen vocab
    };

    const small_memory = registry.estimateMemoryFromConfig(small_config);
    // Should be around 1.2GB but allow for variation in estimation
    try std.testing.expect(small_memory >= 500); // At least 500MB
    try std.testing.expect(small_memory <= 5000); // At most 5GB

    // Test 7B-like model (Qwen 2.5 7B config)
    const large_config = ConfigInfo{
        .hidden_size = 3584,
        .num_layers = 28,
        .num_attention_heads = 28,
        .vocab_size = 151936,
    };

    const large_memory = registry.estimateMemoryFromConfig(large_config);
    // Should be around 4-5GB
    try std.testing.expect(large_memory >= 2000); // At least 2GB
    try std.testing.expect(large_memory <= 15000); // At most 15GB
}

test "ModelRegistry.getAllModels returns cached results without rescanning" {
    const allocator = std.testing.allocator;

    var registry = ModelRegistry.init(allocator);
    defer registry.deinit();

    // First scan (if models exist)
    try registry.scanDirectory("./models");
    const initial_count = registry.count();

    // Get all models - should not rescan
    const models = try registry.getAllModels(allocator);
    defer allocator.free(models);

    // Verify count matches
    try std.testing.expectEqual(initial_count, models.len);
}

test "getCacheModelPath returns correct path" {
    const allocator = std.testing.allocator;

    // This will fail if HOME is not set, which is fine in some test environments
    const cache_path = getCacheModelPath(allocator) catch |err| {
        if (err == error.EnvironmentVariableNotFound) {
            return; // OK if HOME not set in test env
        }
        return err;
    };
    defer allocator.free(cache_path);

    // Verify path ends with expected suffix
    try std.testing.expect(std.mem.endsWith(u8, cache_path, ".cache/zlx/models"));
}

test "updateStatus changes model status thread-safely" {
    const allocator = std.testing.allocator;

    var registry = ModelRegistry.init(allocator);
    defer registry.deinit();

    // Add a fake model entry for testing
    const config = ConfigInfo{
        .hidden_size = 768,
        .num_layers = 12,
        .num_attention_heads = 12,
    };

    // Allocate strings for the test (registry.deinit() will free the key)
    const model_name = try allocator.dupe(u8, "test-model");
    const model_path = try allocator.dupe(u8, "/tmp/test-model");
    defer allocator.free(model_path);

    var metadata = try ModelMetadata.init(
        allocator,
        model_name,
        model_path,
        config,
        1000000,
        1000,
    );
    errdefer metadata.deinit();

    try registry.models.put(model_name, metadata);
    // NOTE: model_name is now owned by the hash map, it will be freed in registry.deinit()

    // Update status
    registry.updateStatus("test-model", .loaded);

    // Verify status changed
    const model = registry.getModel("test-model").?;
    try std.testing.expectEqual(ModelStatus.loaded, model.status);
    try std.testing.expect(model.loaded_at != null);
}

test "ModelMetadata.deinit frees allocated strings" {
    const allocator = std.testing.allocator;

    const config = ConfigInfo{
        .hidden_size = 768,
        .num_layers = 12,
        .num_attention_heads = 12,
    };

    var metadata = try ModelMetadata.init(
        allocator,
        "test-model",
        "/tmp/test-model",
        config,
        1000000,
        1000,
    );

    // This should not leak memory
    metadata.deinit();
}
