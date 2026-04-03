//! memory_constraints.zig - Small machine memory management
//!
//! Detects system memory and automatically configures TurboQuant and context limits
//! to ensure models fit within available RAM (8-16GB machines).

const std = @import("std");
const mlx = @import("mlx.zig/src/mlx.zig");

/// System memory information
pub const SystemInfo = struct {
    total_ram_mb: u64,
    available_ram_mb: u64,
    has_gpu: bool,
    gpu_memory_mb: u64,
    os_name: []const u8,

    /// Get total RAM in GB (for display)
    pub fn totalRamGB(self: SystemInfo) f64 {
        return @as(f64, @floatFromInt(self.total_ram_mb)) / 1024.0;
    }

    /// Get available RAM in GB (for display)
    pub fn availableRamGB(self: SystemInfo) f64 {
        return @as(f64, @floatFromInt(self.available_ram_mb)) / 1024.0;
    }
};

/// Model memory requirements
pub const ModelRequirements = struct {
    model_id: []const u8,
    weights_mb: f64,
    kv_per_1k_tokens_mb: f64, // KV cache MB per 1024 tokens
    min_context: usize = 512,
    max_context: usize = 131072,
    min_ram_mb: u64 = 8192, // Minimum RAM to run at all
    recommended_ram_mb: u64 = 16384, // Recommended RAM for full functionality

    /// Calculate total memory needed for a given context
    pub fn calculateMemory(self: ModelRequirements, context_len: usize, use_turboquant: bool, compression_ratio: f64) f64 {
        const kv_cache_mb = self.kv_per_1k_tokens_mb * (@as(f64, @floatFromInt(context_len)) / 1024.0);
        const compressed_kv = if (use_turboquant) kv_cache_mb / compression_ratio else kv_cache_mb;
        return self.weights_mb + compressed_kv;
    }
};

/// Known model configurations
pub const KNOWN_MODELS = [_]ModelRequirements{
    .{
        .model_id = "Qwen2.5-Coder-1.5B-Instruct-4bit",
        .weights_mb = 900.0,
        .kv_per_1k_tokens_mb = 50.0,
        .max_context = 8192,
        .min_ram_mb = 4096,
        .recommended_ram_mb = 8192,
    },
    .{
        .model_id = "DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx",
        .weights_mb = 8300.0,
        .kv_per_1k_tokens_mb = 180.0,
        .max_context = 128000,
        .min_ram_mb = 12288, // 12GB minimum (with TurboQuant)
        .recommended_ram_mb = 16384,
    },
    .{
        .model_id = "gpt-oss-20b-MXFP4-Q4",
        .weights_mb = 11200.0,
        .kv_per_1k_tokens_mb = 250.0,
        .max_context = 131072,
        .min_ram_mb = 14336, // 14GB minimum (with TurboQuant)
        .recommended_ram_mb = 20480,
    },
};

/// Optimal configuration determined by memory analysis
pub const OptimalConfig = struct {
    use_turboquant: bool,
    max_context: usize,
    warning: ?[]const u8,
    fits_in_memory: bool,
};

/// Get system memory information (macOS-specific)
pub fn getSystemInfo(allocator: std.mem.Allocator) !SystemInfo {
    // Get total RAM using sysctl on macOS
    var total_ram: u64 = 0;
    var len: usize = @sizeOf(u64);

    // Use sysctlbyname for hw.memsize
    const rc = std.c.sysctlbyname("hw.memsize", &total_ram, &len, null, 0);
    if (rc != 0) {
        // Fallback: assume 16GB if we can't detect
        std.log.warn("Failed to detect system memory, assuming 16GB", .{});
        total_ram = 16 * 1024 * 1024 * 1024; // 16GB in bytes
    }

    // Convert to MB
    const total_mb = total_ram / (1024 * 1024);

    // Estimate available memory (conservative: reserve 4GB for OS)
    const available_mb = if (total_mb > 4096) total_mb - 4096 else total_mb / 2;

    // Detect GPU (Metal on Apple Silicon)
    const has_gpu = true; // All Apple Silicon has GPU

    // Get OS name
    const os_name = try allocator.dupe(u8, "macOS");
    errdefer allocator.free(os_name);

    return .{
        .total_ram_mb = total_mb,
        .available_ram_mb = available_mb,
        .has_gpu = has_gpu,
        .gpu_memory_mb = total_mb, // Unified memory on Apple Silicon
        .os_name = os_name,
    };
}

/// Find model requirements by ID
pub fn getModelRequirements(model_id: []const u8) ?ModelRequirements {
    for (KNOWN_MODELS) |model| {
        if (std.mem.indexOf(u8, model.model_id, model_id) != null or
            std.mem.indexOf(u8, model_id, model.model_id) != null)
        {
            return model;
        }
    }
    return null;
}

/// Determine optimal configuration for a model on this system
pub fn determineOptimalConfig(
    model: ModelRequirements,
    system: SystemInfo,
    user_requested_context: ?usize,
    compression_ratio: f64,
) OptimalConfig {
    // Check if model can run at all
    if (system.available_ram_mb < model.min_ram_mb) {
        return .{
            .use_turboquant = true,
            .max_context = model.min_context,
            .warning = "Model may not fit in available memory. Close other applications or use a smaller model.",
            .fits_in_memory = false,
        };
    }

    // Calculate target context
    const target_context = user_requested_context orelse model.max_context;

    // Check if it fits without TurboQuant
    const memory_without_tq = model.calculateMemory(target_context, false, compression_ratio);
    const available_mb_f = @as(f64, @floatFromInt(system.available_ram_mb));

    if (memory_without_tq <= available_mb_f) {
        // Model fits without compression
        return .{
            .use_turboquant = false,
            .max_context = @min(target_context, model.max_context),
            .warning = null,
            .fits_in_memory = true,
        };
    }

    // Try with TurboQuant
    const memory_with_tq = model.calculateMemory(target_context, true, compression_ratio);

    if (memory_with_tq <= available_mb_f) {
        // Model fits with TurboQuant
        const warning = if (user_requested_context != null and target_context > 16384)
            "TurboQuant auto-enabled for memory efficiency. Slight quality reduction possible."
        else
            null;

        return .{
            .use_turboquant = true,
            .max_context = @min(target_context, model.max_context),
            .warning = warning,
            .fits_in_memory = true,
        };
    }

    // Calculate maximum possible context with TurboQuant
    const max_kv_mb = available_mb_f - model.weights_mb;
    const max_context_float = (max_kv_mb / model.kv_per_1k_tokens_mb) * 1024.0 * compression_ratio;
    const max_context: usize = @intFromFloat(@max(max_context_float, @as(f64, @floatFromInt(model.min_context))));
    const clamped_context = @min(max_context, model.max_context);

    var warning_buf: [256]u8 = undefined;
    const warning = std.fmt.bufPrint(&warning_buf, "Context limited to {d}K tokens to fit available memory ({d:.0}GB)", .{
        clamped_context / 1024,
        system.availableRamGB(),
    }) catch "Context limited to fit available memory";

    return .{
        .use_turboquant = true,
        .max_context = clamped_context,
        .warning = warning,
        .fits_in_memory = true,
    };
}

/// Print memory analysis and recommendations
pub fn printMemoryAnalysis(
    model: ModelRequirements,
    system: SystemInfo,
    config: OptimalConfig,
) void {
    const stdout = std.io.getStdOut().writer();

    stdout.print("\n╔════════════════════════════════════════════════════════════════╗\n", .{}) catch {};
    stdout.print("║           Model Memory Analysis                                ║\n", .{}) catch {};
    stdout.print("╚════════════════════════════════════════════════════════════════╝\n", .{}) catch {};

    stdout.print("\nSystem Information:\n", .{}) catch {};
    stdout.print("  Total RAM: {d:.1} GB\n", .{system.totalRamGB()}) catch {};
    stdout.print("  Available for Model: {d:.1} GB (reserved 4GB for OS)\n", .{system.availableRamGB()}) catch {};
    stdout.print("  GPU: {s}\n", .{if (system.has_gpu) "Yes (Metal)" else "No"}) catch {};

    stdout.print("\nModel: {s}\n", .{model.model_id}) catch {};
    stdout.print("  Weights: {d:.1} GB\n", .{model.weights_mb / 1024.0}) catch {};
    stdout.print("  KV Cache Rate: {d:.1} MB per 1K tokens\n", .{model.kv_per_1k_tokens_mb}) catch {};
    stdout.print("  Max Context: {d}K tokens\n", .{model.max_context / 1024}) catch {};

    stdout.print("\nConfiguration:\n", .{}) catch {};
    stdout.print("  TurboQuant: {s}\n", .{if (config.use_turboquant) "ENABLED (auto)" else "Not needed"}) catch {};
    stdout.print("  Context Limit: {d}K tokens\n", .{config.max_context / 1024}) catch {};

    if (config.warning) |w| {
        stdout.print("\n⚠️  Warning: {s}\n", .{w}) catch {};
    }

    // Print compatibility
    stdout.print("\nCompatibility:\n", .{}) catch {};
    if (config.fits_in_memory) {
        stdout.print("  ✅ Model fits in available memory\n", .{}) catch {};
    } else {
        stdout.print("  ❌ Model may not fit - consider using a smaller model\n", .{}) catch {};
    }

    if (config.use_turboquant) {
        stdout.print("  💾 TurboQuant compression active (~4.6x reduction)\n", .{}) catch {};
    }

    stdout.print("\n", .{}) catch {};
}

/// Get model recommendation based on system specs
pub fn getModelRecommendation(system: SystemInfo) []const u8 {
    if (system.total_ram_mb <= 8192) {
        return "Qwen2.5-Coder-1.5B-Instruct-4bit (recommended for 8GB)";
    } else if (system.total_ram_mb <= 16384) {
        return "Any model with TurboQuant auto-enabled. DeepSeek/GPT-OSS limited to ~16K context.";
    } else {
        return "All models supported with full context.";
    }
}

/// Check if model can load and return detailed error if not
pub fn checkModelCompatibility(
    model: ModelRequirements,
    system: SystemInfo,
    requested_context: ?usize,
) !OptimalConfig {
    const config = determineOptimalConfig(model, system, requested_context, 4.6);

    if (!config.fits_in_memory) {
        std.log.err("\n╔════════════════════════════════════════════════════════════════╗", .{});
        std.log.err("║  ERROR: Model Too Large for Available Memory                    ║", .{});
        std.log.err("╚════════════════════════════════════════════════════════════════╝", .{});
        std.log.err("\nModel: {s}", .{model.model_id});
        std.log.err("Requires: {d:.1}GB minimum", .{@as(f64, @floatFromInt(model.min_ram_mb)) / 1024.0});
        std.log.err("Available: {d:.1}GB", .{system.availableRamGB()});
        std.log.err("\nRecommendations:", .{});
        std.log.err("  1. Use Qwen2.5-Coder-1.5B (fits in 8GB)", .{});
        std.log.err("  2. Close other applications to free memory", .{});
        std.log.err("  3. Upgrade to 16GB+ RAM for larger models", .{});
        std.log.err("  4. Use --turboquant flag to force compression", .{});
        return error.InsufficientMemory;
    }

    return config;
}

// ============================================================================
// Tests
// ============================================================================

test "ModelRequirements.calculateMemory" {
    const model = ModelRequirements{
        .model_id = "test",
        .weights_mb = 1000.0,
        .kv_per_1k_tokens_mb = 100.0,
    };

    // Without compression: 1000 + (100 * 8) = 1800 MB for 8K context
    const uncomp = model.calculateMemory(8192, false, 4.6);
    try std.testing.expectApproxEqAbs(@as(f64, 1800.0), uncomp, 10.0);

    // With compression: 1000 + (100 * 8 / 4.6) ≈ 1174 MB for 8K context
    const comp = model.calculateMemory(8192, true, 4.6);
    try std.testing.expectApproxEqAbs(@as(f64, 1174.0), comp, 10.0);
}

test "determineOptimalConfig fits without turboquant" {
    const model = ModelRequirements{
        .model_id = "test",
        .weights_mb = 1000.0,
        .kv_per_1k_tokens_mb = 50.0,
        .max_context = 8192,
    };

    const system = SystemInfo{
        .total_ram_mb = 16384,
        .available_ram_mb = 12000,
        .has_gpu = true,
        .gpu_memory_mb = 16384,
        .os_name = "macOS",
    };

    const config = determineOptimalConfig(model, system, 4096, 4.6);

    try std.testing.expect(!config.use_turboquant); // Should fit without
    try std.testing.expect(config.max_context == 4096);
    try std.testing.expect(config.warning == null);
    try std.testing.expect(config.fits_in_memory);
}

test "determineOptimalConfig needs turboquant" {
    const model = ModelRequirements{
        .model_id = "test",
        .weights_mb = 8000.0,
        .kv_per_1k_tokens_mb = 200.0,
        .max_context = 32768,
    };

    const system = SystemInfo{
        .total_ram_mb = 16384,
        .available_ram_mb = 12000,
        .has_gpu = true,
        .gpu_memory_mb = 16384,
        .os_name = "macOS",
    };

    // Request 16K context with large model on 16GB machine
    const config = determineOptimalConfig(model, system, 16384, 4.6);

    try std.testing.expect(config.use_turboquant); // Should need compression
    try std.testing.expect(config.fits_in_memory);
}

test "determineOptimalConfig context limiting" {
    const model = ModelRequirements{
        .model_id = "test",
        .weights_mb = 8000.0,
        .kv_per_1k_tokens_mb = 200.0,
        .max_context = 131072,
    };

    const system = SystemInfo{
        .total_ram_mb = 8192,
        .available_ram_mb = 4096, // Only 4GB available!
        .has_gpu = true,
        .gpu_memory_mb = 8192,
        .os_name = "macOS",
    };

    // Request 128K context but system can't handle it
    const config = determineOptimalConfig(model, system, 131072, 4.6);

    try std.testing.expect(config.use_turboquant);
    try std.testing.expect(config.max_context < 131072); // Should be limited
    try std.testing.expect(config.warning != null); // Should warn
}

test "getModelRequirements finds known models" {
    const qwen = getModelRequirements("Qwen2.5-Coder");
    try std.testing.expect(qwen != null);
    try std.testing.expectEqualStrings("Qwen2.5-Coder-1.5B-Instruct-4bit", qwen.?.model_id);

    const deepseek = getModelRequirements("DeepSeek-Coder-V2-Lite");
    try std.testing.expect(deepseek != null);

    const gptoss = getModelRequirements("gpt-oss-20b");
    try std.testing.expect(gptoss != null);

    const unknown = getModelRequirements("unknown-model");
    try std.testing.expect(unknown == null);
}

test "getModelRecommendation for 8GB" {
    const system = SystemInfo{
        .total_ram_mb = 8192,
        .available_ram_mb = 4096,
        .has_gpu = true,
        .gpu_memory_mb = 8192,
        .os_name = "macOS",
    };

    const rec = getModelRecommendation(system);
    try std.testing.expect(std.mem.indexOf(u8, rec, "Qwen2.5-Coder") != null);
}

test "getModelRecommendation for 16GB" {
    const system = SystemInfo{
        .total_ram_mb = 16384,
        .available_ram_mb = 12000,
        .has_gpu = true,
        .gpu_memory_mb = 16384,
        .os_name = "macOS",
    };

    const rec = getModelRecommendation(system);
    try std.testing.expect(std.mem.indexOf(u8, rec, "TurboQuant") != null);
}
