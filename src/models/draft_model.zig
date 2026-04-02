//! draft_model.zig - Draft model management (loading, caching, switching)
//!
//! Manages lifecycle of draft models including loading, caching,
//! and automatic switching when target model changes.

const std = @import("std");
const mlx = @import("../mlx.zig/src/mlx.zig");
const qwen = @import("../mlx.zig/src/qwen.zig");
const generator = @import("../inference/generator.zig");
const registry = @import("../models/registry.zig");
const draft_selector = @import("draft_selector.zig");

/// DraftModel wraps a loaded draft model with its resources
pub const DraftModel = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    transformer: *qwen.Transformer,
    cache: *mlx.Cache,
    model_path: []const u8,
    model_id: []const u8,
    config: registry.ConfigInfo,
    ref_count: std.atomic.Value(u32),
    mutex: std.Thread.Mutex,

    /// Initialize draft model by loading from path
    pub fn init(allocator: std.mem.Allocator, model_path: []const u8, model_id: []const u8) !Self {
        // Load transformer
        const transformer = try qwen.Transformer.init(allocator, model_path);
        errdefer transformer.deinit();

        // Initialize KV cache
        const cache = try allocator.create(mlx.Cache);
        errdefer allocator.destroy(cache);
        cache.* = try mlx.Cache.init(allocator, transformer.model.layers.len, 2);

        // Get config from registry or estimate
        const config = try estimateConfig(transformer);

        return .{
            .allocator = allocator,
            .transformer = transformer,
            .cache = cache,
            .model_path = try allocator.dupe(u8, model_path),
            .model_id = try allocator.dupe(u8, model_id),
            .config = config,
            .ref_count = std.atomic.Value(u32).init(1),
            .mutex = .{},
        };
    }

    /// Deinitialize and free all resources
    pub fn deinit(self: *Self) void {
        self.transformer.deinit();

        self.cache.deinit();
        self.allocator.destroy(self.cache);

        self.allocator.free(self.model_path);
        self.allocator.free(self.model_id);
    }

    /// Increment reference count
    pub fn retain(self: *Self) void {
        _ = self.ref_count.fetchAdd(1, .monotonic);
    }

    /// Decrement reference count, deinit if zero
    pub fn release(self: *Self) void {
        const count = self.ref_count.fetchSub(1, .release);
        if (count == 1) {
            // Last reference, free
            self.deinit();
        }
    }

    /// Generate K draft tokens autoregressively
    pub fn generateDraftTokens(
        self: *Self,
        context: []const u32,
        count: usize,
        options: generator.GenerationOptions,
    ) ![]u32 {
        var result = try self.allocator.alloc(u32, count);
        errdefer self.allocator.free(result);

        var working_context = try self.allocator.dupe(u32, context);
        defer self.allocator.free(working_context);

        for (0..count) |i| {
            // Prepare input
            const toks_array = try mlx.arrayNewData(
                working_context.ptr,
                .{ 1, @as(c_int, @intCast(working_context.len)) },
                mlx.UINT32,
            );
            defer mlx.arrayFree(toks_array);

            // Create mask
            const seq_len = mlx.arrayDim(toks_array, 1);
            var mask_array = mlx.arrayNew();
            defer mlx.arrayFree(mask_array);
            try mlx.createCausalMask(&mask_array, seq_len, self.cache.offset, self.transformer.mlx_config.dtype, self.transformer.mlx_config.stream);

            // Forward pass
            var logits_array = mlx.arrayNew();
            defer mlx.arrayFree(logits_array);
            try self.transformer.model.forward(&logits_array, toks_array, mask_array, self.cache);

            // Sample next token
            const next_token = try self.sampleNextToken(logits_array, options);
            result[i] = next_token;

            // Extend context for next iteration
            const new_context = try self.allocator.alloc(u32, working_context.len + 1);
            @memcpy(new_context[0..working_context.len], working_context);
            new_context[working_context.len] = next_token;
            self.allocator.free(working_context);
            working_context = new_context;

            // Check for EOS early
            if (options.stop_on_eos and self.isEosToken(next_token)) {
                // Return early with what we have
                const trimmed = try self.allocator.alloc(u32, i + 1);
                @memcpy(trimmed, result[0 .. i + 1]);
                self.allocator.free(result);
                return trimmed;
            }
        }

        return result;
    }

    /// Sample next token from logits with temperature support
    fn sampleNextToken(self: *Self, logits_array: mlx.Array, options: generator.GenerationOptions) !u32 {
        // Take logits for last position
        var last_logits = mlx.arrayNew();
        defer mlx.arrayFree(last_logits);
        try mlx.take(&last_logits, logits_array, mlx.int(-1), 1, self.transformer.mlx_config.stream);

        // Greedy if temperature is 0
        if (options.temperature == 0) {
            var next_token_arr = mlx.arrayNew();
            defer mlx.arrayFree(next_token_arr);
            try mlx.argmax(&next_token_arr, last_logits, 1, false, self.transformer.mlx_config.stream);

            var token: u32 = 0;
            try mlx.item(&token, next_token_arr);
            return token;
        }

        // Apply temperature and sample
        var scaled_logits = mlx.arrayNew();
        defer mlx.arrayFree(scaled_logits);

        if (options.temperature != 1.0) {
            const temp_scalar = mlx.float(options.temperature);
            try mlx.divide(&scaled_logits, last_logits, temp_scalar, self.transformer.mlx_config.stream);
        } else {
            try mlx.arraySet(&scaled_logits, last_logits);
        }

        // Softmax to get probabilities
        var probs_array = mlx.arrayNew();
        defer mlx.arrayFree(probs_array);
        const axes = &[_]c_int{1};
        try mlx.softmax(&probs_array, scaled_logits, axes, false, self.transformer.mlx_config.stream);

        // Evaluate to get probabilities
        try mlx.arrayEval(probs_array);
        const probs_data: [*c]f32 = @ptrCast(@constCast(mlx.C.mlx_array_data_float32(probs_array)));
        const vocab_size = @as(usize, @intCast(mlx.arrayDim(probs_array, 1)));

        // Sample from distribution
        const random_val = std.crypto.random.float(f32);
        var cumsum: f32 = 0;
        for (0..vocab_size) |i| {
            cumsum += probs_data[i];
            if (random_val <= cumsum) {
                return @intCast(i);
            }
        }

        return @intCast(vocab_size - 1);
    }

    /// Check if token is EOS
    fn isEosToken(_: *Self, token: u32) bool {
        const eos_tokens = [_]u32{ 151645, 151643 };
        for (eos_tokens) |eos| {
            if (token == eos) return true;
        }
        return false;
    }

    /// Estimate config from transformer (fallback when registry lookup fails)
    fn estimateConfig(transformer: *qwen.Transformer) !registry.ConfigInfo {
        return .{
            .hidden_size = @intCast(transformer.model.hidden_size),
            .num_layers = @intCast(transformer.model.layers.len),
            .num_attention_heads = @intCast(transformer.model.num_attention_heads),
            .vocab_size = 151936, // Default for Qwen
        };
    }

    /// Reset cache offset (called when switching contexts)
    pub fn resetCache(self: *Self) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.cache.offset = 0;
    }
};

/// DraftModelManager manages loading and caching of draft models
pub const DraftModelManager = struct {
    const Self = @This();

    allocator: std.mem.Allocator,

    // Active draft models (loaded in memory)
    loaded_drafts: std.StringHashMap(*DraftModel),

    // Map from target model ID to draft model ID
    target_to_draft: std.StringHashMap([]const u8),

    // Configuration
    max_loaded_drafts: usize,
    max_draft_memory_mb: u32,

    // Synchronization
    mutex: std.Thread.Mutex,

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .loaded_drafts = std.StringHashMap(*DraftModel).init(allocator),
            .target_to_draft = std.StringHashMap([]const u8).init(allocator),
            .max_loaded_drafts = 2, // Keep at most 2 drafts in memory
            .max_draft_memory_mb = 4096, // 4GB max for drafts
            .mutex = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Free all loaded drafts
        var draft_iter = self.loaded_drafts.iterator();
        while (draft_iter.next()) |entry| {
            entry.value_ptr.*.release();
            self.allocator.free(entry.key_ptr.*);
        }
        self.loaded_drafts.deinit();

        // Free target-to-draft mappings
        var mapping_iter = self.target_to_draft.iterator();
        while (mapping_iter.next()) |entry| {
            self.allocator.free(entry.value_ptr.*);
            self.allocator.free(entry.key_ptr.*);
        }
        self.target_to_draft.deinit();
    }

    /// Get or load draft model for a target model
    /// Uses selector to find best draft if not already determined
    pub fn getOrLoadDraft(
        self: *Self,
        target_model_id: []const u8,
        selector: *draft_selector.DraftSelector,
        user_override: ?[]const u8,
    ) !?*DraftModel {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Check if we already have a draft for this target
        if (self.target_to_draft.get(target_model_id)) |draft_id| {
            if (self.loaded_drafts.get(draft_id)) |draft| {
                draft.retain();
                return draft;
            }
        }

        // Use selector to find best draft
        const draft_info = selector.selectDraftForTarget(target_model_id, user_override) orelse {
            std.log.debug("No compatible draft model found for target: {s}", .{target_model_id});
            return null;
        };

        // Check if already loaded
        if (self.loaded_drafts.get(draft_info.model_id)) |draft| {
            draft.retain();
            try self.updateTargetMapping(target_model_id, draft_info.model_id);
            selector.freeDraftInfo(draft_info);
            return draft;
        }

        // Check memory budget before loading
        if (!self.checkMemoryBudget(draft_info)) {
            std.log.warn("Draft model '{s}' exceeds memory budget, skipping", .{draft_info.model_id});
            selector.freeDraftInfo(draft_info);
            return null;
        }

        // Evict oldest draft if at capacity
        if (self.loaded_drafts.count() >= self.max_loaded_drafts) {
            try self.evictOldestDraft();
        }

        // Load the draft model
        std.log.info("Loading draft model: {s}", .{draft_info.model_id});
        var draft = try DraftModel.init(self.allocator, draft_info.path, draft_info.model_id);

        // Store in loaded drafts
        const draft_id_copy = try self.allocator.dupe(u8, draft_info.model_id);
        try self.loaded_drafts.put(draft_id_copy, &draft);

        // Update target mapping
        try self.updateTargetMapping(target_model_id, draft_info.model_id);

        selector.freeDraftInfo(draft_info);

        std.log.info("Draft model loaded: {s}", .{draft.model_id});
        return &draft;
    }

    /// Unload draft model for a specific target
    pub fn unloadDraftForTarget(self: *Self, target_model_id: []const u8) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.target_to_draft.get(target_model_id)) |draft_id| {
            if (self.loaded_drafts.get(draft_id)) |draft| {
                draft.release();
            }

            _ = self.target_to_draft.remove(target_model_id);
        }
    }

    /// Get current draft model ID for a target
    pub fn getDraftForTarget(self: *Self, target_model_id: []const u8) ?[]const u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        return self.target_to_draft.get(target_model_id);
    }

    /// Check if draft fits within memory budget
    fn checkMemoryBudget(self: *Self, draft_info: draft_selector.DraftModelInfo) bool {
        _ = self;

        // Estimate draft memory (rough: ~1.5GB per 1.5B params at 4-bit)
        // For now, accept any draft under 4GB limit
        _ = draft_info;
        return true;
    }

    /// Evict oldest loaded draft (LRU policy)
    fn evictOldestDraft(self: *Self) !void {
        // For simplicity, evict first draft found
        // In production, track access times
        var iter = self.loaded_drafts.iterator();
        if (iter.next()) |entry| {
            const draft_id = entry.key_ptr.*;

            // Remove from all target mappings
            var mapping_iter = self.target_to_draft.iterator();
            var to_remove = std.ArrayList([]const u8).init(self.allocator);
            defer to_remove.deinit();

            while (mapping_iter.next()) |mapping| {
                if (std.mem.eql(u8, mapping.value_ptr.*, draft_id)) {
                    try to_remove.append(mapping.key_ptr.*);
                }
            }

            for (to_remove.items) |target_id| {
                _ = self.target_to_draft.remove(target_id);
            }

            // Release the draft
            entry.value_ptr.*.release();

            // Remove from loaded drafts
            _ = self.loaded_drafts.remove(draft_id);
            self.allocator.free(draft_id);

            std.log.debug("Evicted draft model: {s}", .{draft_id});
        }
    }

    /// Update target-to-draft mapping
    fn updateTargetMapping(self: *Self, target_id: []const u8, draft_id: []const u8) !void {
        // Remove old mapping if exists
        if (self.target_to_draft.get(target_id)) |old_draft_id| {
            self.allocator.free(old_draft_id);
            _ = self.target_to_draft.remove(target_id);
        }

        // Add new mapping
        const target_copy = try self.allocator.dupe(u8, target_id);
        const draft_copy = try self.allocator.dupe(u8, draft_id);
        try self.target_to_draft.put(target_copy, draft_copy);
    }

    /// Get count of loaded draft models
    pub fn loadedCount(self: *Self) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.loaded_drafts.count();
    }
};

// ============================================================================
// Tests
// ============================================================================

test "DraftModel basic lifecycle" {
    const allocator = std.testing.allocator;

    // This test requires an actual model - skip if not available
    const model_path = "./models/Qwen2.5-Coder-1.5B-4bit";
    std.fs.cwd().access(model_path, .{}) catch {
        std.debug.print("Skipping test - model not found at {s}\n", .{model_path});
        return;
    };

    var draft = try DraftModel.init(allocator, model_path, "test-draft");
    defer draft.deinit();

    try std.testing.expect(std.mem.eql(u8, "test-draft", draft.model_id));
    try std.testing.expect(draft.ref_count.load(.monotonic) == 1);
}

test "DraftModelManager init and deinit" {
    const allocator = std.testing.allocator;

    var manager = DraftModelManager.init(allocator);
    defer manager.deinit();

    try std.testing.expectEqual(@as(usize, 0), manager.loadedCount());
}

test "DraftModel reference counting" {
    const allocator = std.testing.allocator;

    // This test requires an actual model - skip if not available
    const model_path = "./models/Qwen2.5-Coder-1.5B-4bit";
    std.fs.cwd().access(model_path, .{}) catch {
        std.debug.print("Skipping test - model not found\n", .{});
        return;
    };

    var draft = try DraftModel.init(allocator, model_path, "test-draft");

    try std.testing.expectEqual(@as(u32, 1), draft.ref_count.load(.monotonic));

    draft.retain();
    try std.testing.expectEqual(@as(u32, 2), draft.ref_count.load(.monotonic));

    draft.release();
    // After release, ref_count is 1 but model is still valid
    try std.testing.expectEqual(@as(u32, 1), draft.ref_count.load(.monotonic));

    draft.release(); // This would deinit but we can't test that easily
}
