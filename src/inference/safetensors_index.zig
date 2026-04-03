//! safetensors_index.zig - Parse safetensors index files for weight loading
//!
//! Handles model.safetensors.index.json parsing and weight mapping for MoE models.

const std = @import("std");
const mlx = @import("../mlx.zig/src/mlx.zig");

/// Weight index entry mapping weight name to file path
pub const WeightIndex = struct {
    weight_name: []const u8,
    file_path: []const u8,

    pub fn deinit(self: *WeightIndex, allocator: std.mem.Allocator) void {
        allocator.free(self.weight_name);
        allocator.free(self.file_path);
    }
};

/// Parsed safetensors index with weight mappings
pub const SafetensorsIndex = struct {
    allocator: std.mem.Allocator,
    weight_map: std.StringHashMap([]const u8), // weight_name -> file_path

    pub fn init(allocator: std.mem.Allocator) SafetensorsIndex {
        return .{
            .allocator = allocator,
            .weight_map = std.StringHashMap([]const u8).init(allocator),
        };
    }

    pub fn deinit(self: *SafetensorsIndex) void {
        var iter = self.weight_map.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.weight_map.deinit();
    }

    /// Get file path for a specific weight
    pub fn getFileForWeight(self: SafetensorsIndex, weight_name: []const u8) ?[]const u8 {
        return self.weight_map.get(weight_name);
    }

    /// Check if weight exists in index
    pub fn hasWeight(self: SafetensorsIndex, weight_name: []const u8) bool {
        return self.weight_map.contains(weight_name);
    }

    /// Get total number of weights
    pub fn weightCount(self: SafetensorsIndex) usize {
        return self.weight_map.count();
    }
};

/// Parse safetensors index file
pub fn parseSafetensorsIndex(
    allocator: std.mem.Allocator,
    model_path: []const u8,
) !SafetensorsIndex {
    // Construct index file path
    var index_path_buf: [1024]u8 = undefined;
    const index_path = try std.fmt.bufPrint(&index_path_buf, "{s}/model.safetensors.index.json", .{model_path});

    // Check if index file exists
    std.fs.cwd().access(index_path, .{}) catch {
        // No index file - model has single safetensors file
        // Create simple index with all weights in model.safetensors
        return createSimpleIndex(allocator, model_path);
    };

    // Read index file
    const file = try std.fs.cwd().openFile(index_path, .{});
    defer file.close();

    const content = try file.readToEndAlloc(allocator, 10 * 1024 * 1024); // Max 10MB
    defer allocator.free(content);

    // Parse JSON manually (simple parser for index format)
    var index = SafetensorsIndex.init(allocator);
    errdefer index.deinit();

    try parseIndexJson(allocator, content, model_path, &index);

    return index;
}

/// Create simple index for models without index.json (single file)
fn createSimpleIndex(allocator: std.mem.Allocator, model_path: []const u8) !SafetensorsIndex {
    var index = SafetensorsIndex.init(allocator);
    errdefer index.deinit();

    // Look for model.safetensors file
    var dir = try std.fs.cwd().openDir(model_path, .{ .iterate = true });
    defer dir.close();

    var iter = dir.iterate();
    while (try iter.next()) |entry| {
        if (std.mem.endsWith(u8, entry.name, ".safetensors") and
            !std.mem.containsAtLeast(u8, entry.name, 1, "index"))
        {
            // Found a safetensors file
            const file_path = try std.fs.path.join(allocator, &.{ model_path, entry.name });

            // For simple case, we'd need to load the file to get weight names
            // For now, just store the file path pattern
            const marker_key = try allocator.dupe(u8, "__single_file__");
            const marker_value = try allocator.dupe(u8, file_path);
            try index.weight_map.put(marker_key, marker_value);

            break;
        }
    }

    return index;
}

/// Parse index.json content
fn parseIndexJson(
    allocator: std.mem.Allocator,
    content: []const u8,
    model_path: []const u8,
    index: *SafetensorsIndex,
) !void {
    // Simple JSON parsing - look for "weight_map" section
    // Format: {"weight_map": {"weight.name": "model-00001-of-00002.safetensors", ...}}

    const weight_map_key = "\"weight_map\"";
    const weight_map_start = std.mem.indexOf(u8, content, weight_map_key) orelse {
        // Try alternative format
        return try parseAlternativeFormat(allocator, content, model_path, index);
    };

    // Find opening brace after weight_map
    const brace_start = std.mem.indexOfPos(u8, content, weight_map_start, "{") orelse {
        std.log.err("Invalid index.json: no opening brace after weight_map", .{});
        return error.InvalidIndexFormat;
    };

    // Parse key-value pairs
    var pos = brace_start + 1;
    while (pos < content.len) {
        // Skip whitespace
        while (pos < content.len and std.ascii.isWhitespace(content[pos])) pos += 1;
        if (pos >= content.len) break;

        // Check for closing brace
        if (content[pos] == '}') break;

        // Expect quote for key
        if (content[pos] != '"') {
            pos += 1;
            continue;
        }

        // Parse key (weight name)
        const key_start = pos + 1;
        const key_end = std.mem.indexOfPos(u8, content, key_start, "\"") orelse break;
        const weight_name = content[key_start..key_end];

        // Move past key
        pos = key_end + 1;

        // Skip to colon
        while (pos < content.len and content[pos] != ':') pos += 1;
        if (pos >= content.len) break;
        pos += 1; // Skip colon

        // Skip whitespace
        while (pos < content.len and std.ascii.isWhitespace(content[pos])) pos += 1;

        // Expect quote for value
        if (pos >= content.len or content[pos] != '"') continue;

        // Parse value (file name)
        const val_start = pos + 1;
        const val_end = std.mem.indexOfPos(u8, content, val_start, "\"") orelse break;
        const file_name = content[val_start..val_end];

        // Store mapping
        const owned_name = try allocator.dupe(u8, weight_name);
        const file_path = try std.fs.path.join(allocator, &.{ model_path, file_name });

        try index.weight_map.put(owned_name, file_path);

        // Move to next entry
        pos = val_end + 1;

        // Skip comma if present
        while (pos < content.len and std.ascii.isWhitespace(content[pos])) pos += 1;
        if (pos < content.len and content[pos] == ',') pos += 1;
    }
}

/// Parse alternative index formats
fn parseAlternativeFormat(
    allocator: std.mem.Allocator,
    content: []const u8,
    model_path: []const u8,
    index: *SafetensorsIndex,
) !void {
    // Some models use different formats - try to find any mapping
    _ = allocator;
    _ = content;
    _ = model_path;
    _ = index;

    std.log.warn("Unknown index.json format, attempting fallback parsing", .{});
    return error.UnsupportedIndexFormat;
}

/// Get list of unique shard files from index
pub fn getShardFiles(
    allocator: std.mem.Allocator,
    index: SafetensorsIndex,
) !std.ArrayList([]const u8) {
    var shards = std.ArrayList([]const u8).init(allocator);
    errdefer {
        for (shards.items) |path| allocator.free(path);
        shards.deinit();
    }

    var seen = std.StringHashMap(void).init(allocator);
    defer seen.deinit();

    var iter = index.weight_map.iterator();
    while (iter.next()) |entry| {
        const file_path = entry.value_ptr.*;

        // Check if we've seen this file
        if (seen.contains(file_path)) continue;

        // Add to unique list
        const path_copy = try allocator.dupe(u8, file_path);
        try shards.append(path_copy);
        try seen.put(file_path, {});
    }

    return shards;
}

/// Format weight name for DeepSeek model
pub fn formatDeepSeekWeightName(
    buf: []u8,
    layer_idx: usize,
    component: []const u8,
    weight_type: []const u8,
) ![]const u8 {
    return try std.fmt.bufPrint(buf, "model.layers.{d}.{s}.{s}.weight", .{
        layer_idx, component, weight_type,
    });
}

/// Format expert weight name
pub fn formatExpertWeightName(
    buf: []u8,
    layer_idx: usize,
    expert_idx: usize,
    proj_type: []const u8,
) ![]const u8 {
    return try std.fmt.bufPrint(buf, "model.layers.{d}.mlp.experts.{d}.{s}_proj.weight", .{
        layer_idx, expert_idx, proj_type,
    });
}

// ============================================================================
// Tests
// ============================================================================

test "parseSafetensorsIndex - simple format" {
    const allocator = std.testing.allocator;

    // Create temporary test directory
    const test_dir = "/tmp/test_safetensors_index";
    try std.fs.cwd().makePath(test_dir);
    defer std.fs.cwd().deleteTree(test_dir) catch {};

    // Create test index file
    const index_content =
        \\{"weight_map": {
        \\"model.embed_tokens.weight\": \"model-00001-of-00002.safetensors\",
        \\"model.layers.0.input_layernorm.weight\": \"model-00001-of-00002.safetensors\",
        \\"model.layers.0.self_attn.q_proj.weight\": \"model-00001-of-00002.safetensors\",
        \\"model.norm.weight\": \"model-00002-of-00002.safetensors\",
        \\"lm_head.weight\": \"model-00002-of-00002.safetensors\"
        \\}}
    ;

    const index_path = try std.fs.path.join(allocator, &.{ test_dir, "model.safetensors.index.json" });
    defer allocator.free(index_path);

    const file = try std.fs.cwd().createFile(index_path, .{});
    defer file.close();
    try file.writeAll(index_content);

    // Parse index
    var index = try parseSafetensorsIndex(allocator, test_dir);
    defer index.deinit();

    // Verify
    try std.testing.expectEqual(@as(usize, 5), index.weightCount());
    try std.testing.expect(index.hasWeight("model.embed_tokens.weight"));
    try std.testing.expect(index.hasWeight("model.layers.0.input_layernorm.weight"));
    try std.testing.expect(index.hasWeight("lm_head.weight"));

    // Check file mapping
    const embed_file = index.getFileForWeight("model.embed_tokens.weight");
    try std.testing.expect(embed_file != null);
    try std.testing.expect(std.mem.containsAtLeast(u8, embed_file.?, 1, "model-00001-of-00002"));
}

test "formatDeepSeekWeightName" {
    var buf: [256]u8 = undefined;

    const name = try formatDeepSeekWeightName(&buf, 5, "self_attn", "q_proj");
    try std.testing.expectEqualStrings("model.layers.5.self_attn.q_proj.weight", name);
}

test "formatExpertWeightName" {
    var buf: [256]u8 = undefined;

    const name = try formatExpertWeightName(&buf, 3, 7, "gate");
    try std.testing.expectEqualStrings("model.layers.3.mlp.experts.7.gate_proj.weight", name);
}

test "getShardFiles" {
    const allocator = std.testing.allocator;

    var index = SafetensorsIndex.init(allocator);
    defer index.deinit();

    // Add test entries
    const key1 = try allocator.dupe(u8, "weight1");
    const val1 = try allocator.dupe(u8, "/models/model-00001.safetensors");
    try index.weight_map.put(key1, val1);

    const key2 = try allocator.dupe(u8, "weight2");
    const val2 = try allocator.dupe(u8, "/models/model-00001.safetensors"); // Same file
    try index.weight_map.put(key2, val2);

    const key3 = try allocator.dupe(u8, "weight3");
    const val3 = try allocator.dupe(u8, "/models/model-00002.safetensors"); // Different file
    try index.weight_map.put(key3, val3);

    const shards = try getShardFiles(allocator, index);
    defer {
        for (shards.items) |path| allocator.free(path);
        shards.deinit();
    }

    // Should have 2 unique files
    try std.testing.expectEqual(@as(usize, 2), shards.items.len);
}
