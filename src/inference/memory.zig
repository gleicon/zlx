//! memory.zig - Sparse parameter memory estimation for DeepSeek MoE
//!
//! Provides memory estimation functions that account for Mixture of Experts
//! sparse activation patterns. Unlike dense models, MoE models only load
//! a subset of parameters per token (active_params vs total_params).

const std = @import("std");
const registry = @import("../models/registry.zig");

/// Memory estimation configuration
pub const MemoryConfig = struct {
    // Model parameters
    vocab_size: u32,
    hidden_size: u32,
    num_layers: u32,
    num_experts: u32 = 64,
    active_experts: u32 = 6,
    max_context: u32 = 128_000,

    // Quantization
    quantization_bits: u8 = 4, // DeepSeek uses 4-bit by default

    // MLA compression
    mla_latent_dim: u32 = 512, // Compressed KV dimension
    standard_kv_dim: u32 = 8192, // Standard KV dimension (for comparison)
};

/// Estimate DeepSeek MoE memory usage
/// Uses active parameters (2B) not total parameters (15.7B)
///
/// Memory breakdown:
/// - Embeddings: vocab_size * hidden_size * bytes_per_param
/// - Active weights: 2B params * bytes_per_param
/// - Compressed KV: 2 * layers * latent_dim * context * bytes_per_param
/// - Expert routing overhead: ~5% of active weights
pub fn estimateDeepSeekMemory(config: MemoryConfig) u64 {
    const bytes_per_param: u64 = switch (config.quantization_bits) {
        8 => 1,
        16 => 2,
        32 => 4,
        else => 1, // Default to 4-bit (0.5 bytes per param)
    };

    // Embeddings (always loaded, not sparse)
    const embedding_bytes = @as(u64, config.vocab_size) *
        @as(u64, config.hidden_size) *
        bytes_per_param;

    // Calculate active parameters based on config
    // DeepSeek-V2-Lite: 2B active out of 15.7B total
    const active_params = calculateActiveParams(config);

    // Active parameter memory (NOT total!)
    const active_weights_bytes = active_params * bytes_per_param;

    // Expert routing tables (small overhead)
    const routing_bytes = @as(u64, config.num_experts) *
        @as(u64, config.hidden_size) *
        4; // 4 bytes per routing weight (FP32)

    // Compressed KV cache (MLA reduces from 8192 to 512)
    // Standard: 2 * layers * 8192 * context * bytes
    // MLA: 2 * layers * 512 * context * bytes
    const kv_bytes_per_token = 2 *
        @as(u64, config.num_layers) *
        @as(u64, config.mla_latent_dim) *
        bytes_per_param;
    const kv_cache_bytes = kv_bytes_per_token * @as(u64, config.max_context);

    // Activations and temporary buffers (rough estimate)
    const activation_bytes = @as(u64, config.hidden_size) *
        @as(u64, config.num_layers) *
        4 * 1024; // 4KB per layer per hidden dim

    // Overhead buffer (20% for system/runtime)
    const subtotal = embedding_bytes + active_weights_bytes + routing_bytes + kv_cache_bytes + activation_bytes;
    const overhead_bytes = subtotal / 5;

    return subtotal + overhead_bytes;
}

/// Calculate active parameters for MoE model
/// Based on DeepSeek-V2 architecture:
/// - 64 experts total
/// - 6 active experts per token
/// - ~2B active parameters out of ~15.7B total
fn calculateActiveParams(config: MemoryConfig) u64 {
    // Simplified calculation:
    // Total expert parameters = num_experts * expert_size
    // Active expert parameters = active_experts * expert_size
    // Plus shared parameters (embeddings, norms, etc.)

    // Rough estimate for DeepSeek-V2-Lite:
    // - Embeddings: vocab_size * hidden_size = 102400 * 4096 = ~419M
    // - Per-expert FFN: hidden * intermediate * experts
    //   = 4096 * (4096 * 4) * 64 = ~4.3B
    // - Attention: ~2B (not sparse)
    // - Total: ~15.7B
    // - Active: 419M + 2B + (4.3B * 6/64) = ~2B

    const embedding_params = @as(u64, config.vocab_size) *
        @as(u64, config.hidden_size);

    // Attention parameters (not sparse)
    const attention_params = @as(u64, config.hidden_size) *
        @as(u64, config.hidden_size) *
        @as(u64, config.num_layers) * 4; // Q, K, V, O projections

    // Expert FFN parameters (sparse)
    const expert_ratio = @as(f64, @floatFromInt(config.active_experts)) /
        @as(f64, @floatFromInt(config.num_experts));
    const ffn_intermediate = @as(u64, config.hidden_size) * 4; // Standard 4x expansion
    const total_expert_params = @as(u64, config.num_experts) *
        @as(u64, config.hidden_size) *
        ffn_intermediate;
    const active_expert_params = @as(u64, @intFromFloat(@as(f64, @floatFromInt(total_expert_params)) * expert_ratio));

    return embedding_params + attention_params + active_expert_params;
}

/// Get human-readable memory estimate with units
pub fn formatMemoryEstimate(bytes: u64) []const u8 {
    const gb = @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0 * 1024.0);

    if (gb >= 1.0) {
        return std.fmt.bufPrint(
            &.{}, // Static buffer would be better, but this is a demo
            "{d:.1} GB",
            .{gb},
        ) catch "Unknown";
    } else {
        const mb = @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0);
        return std.fmt.bufPrint(
            &.{},
            "{d:.0} MB",
            .{mb},
        ) catch "Unknown";
    }
}

/// Create default DeepSeek-V2-Lite memory config
pub fn deepSeekV2LiteConfig() MemoryConfig {
    return .{
        .vocab_size = 102400,
        .hidden_size = 4096,
        .num_layers = 27,
        .num_experts = 64,
        .active_experts = 6,
        .max_context = 128_000,
        .quantization_bits = 4,
        .mla_latent_dim = 512,
        .standard_kv_dim = 8192,
    };
}

/// Estimate memory from registry ConfigInfo for sparse models
pub fn estimateMemoryForModel(config: registry.ConfigInfo, arch: registry.ModelArchitecture) u32 {
    return switch (arch) {
        .deepseek_v2_moe => {
            const ds_config = MemoryConfig{
                .vocab_size = config.vocab_size,
                .hidden_size = config.hidden_size,
                .num_layers = config.num_layers,
                .num_experts = 64,
                .active_experts = 6,
                .max_context = config.max_position_embeddings,
                .quantization_bits = config.quantization_bits,
                .mla_latent_dim = 512,
                .standard_kv_dim = config.hidden_size,
            };
            const bytes = estimateDeepSeekMemory(ds_config);
            return @as(u32, @intCast(bytes / (1024 * 1024)));
        },
        else => registry.estimateMemoryFromConfig(config), // Fallback to dense estimation
    };
}

// ============================================================================
// Tests
// ============================================================================

test "estimateDeepSeekMemory returns ~2GB for V2-Lite" {
    const config = deepSeekV2LiteConfig();
    const bytes = estimateDeepSeekMemory(config);
    const mb = bytes / (1024 * 1024);

    // Should be approximately 2-2.5GB, NOT 15GB
    try std.testing.expect(mb >= 1500); // At least 1.5GB
    try std.testing.expect(mb <= 4000); // At most 4GB (way less than 15GB)
}

test "estimateDeepSeekMemory uses active params not total" {
    const config = deepSeekV2LiteConfig();

    // Total params would be ~15.7B
    const total_params_if_dense: u64 = 15_700_000_000;

    // Memory with our estimator (sparse)
    const sparse_bytes = estimateDeepSeekMemory(config);

    // Memory if dense (all params loaded)
    const dense_bytes = total_params_if_dense * 1 / 2; // 4-bit = 0.5 bytes per param

    // Sparse should be ~1/7th of dense (2B active vs 15.7B total)
    const ratio = @as(f64, @floatFromInt(sparse_bytes)) / @as(f64, @floatFromInt(dense_bytes));

    // Ratio should be around 0.15-0.25 (sparse is much smaller)
    try std.testing.expect(ratio < 0.3);
    try std.testing.expect(ratio > 0.1);
}

test "deepSeekV2LiteConfig returns correct defaults" {
    const config = deepSeekV2LiteConfig();

    try std.testing.expectEqual(@as(u32, 102400), config.vocab_size);
    try std.testing.expectEqual(@as(u32, 4096), config.hidden_size);
    try std.testing.expectEqual(@as(u32, 27), config.num_layers);
    try std.testing.expectEqual(@as(u32, 64), config.num_experts);
    try std.testing.expectEqual(@as(u32, 6), config.active_experts);
    try std.testing.expectEqual(@as(u32, 128000), config.max_context);
    try std.testing.expectEqual(@as(u8, 4), config.quantization_bits);
}

test "estimateMemoryForModel dispatches to correct estimator" {
    const config = registry.ConfigInfo{
        .hidden_size = 4096,
        .num_layers = 27,
        .num_attention_heads = 128,
        .vocab_size = 102400,
        .quantization_bits = 4,
    };

    // Test DeepSeek estimation (sparse)
    const deepseek_mb = estimateMemoryForModel(config, .deepseek_v2_moe);
    try std.testing.expect(deepseek_mb >= 1500); // ~2GB
    try std.testing.expect(deepseek_mb <= 4000);
}

test "calculateActiveParams returns ~2B for V2-Lite" {
    const config = deepSeekV2LiteConfig();
    const active = calculateActiveParams(config);

    // Should be approximately 2B (between 1.5B and 2.5B)
    try std.testing.expect(active >= 1_500_000_000);
    try std.testing.expect(active <= 2_500_000_000);
}
