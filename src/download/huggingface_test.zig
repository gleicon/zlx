//! huggingface_test.zig - Unit tests for HuggingFace API client

const std = @import("std");
const huggingface = @import("huggingface.zig");
const HuggingFaceId = huggingface.HuggingFaceId;

// Test HuggingFaceId parsing
test "HuggingFaceId.parse valid org/repo format" {
    const id = try HuggingFaceId.parse("mlx-community/Qwen2.5-Coder-1.5B");
    try std.testing.expectEqualStrings("mlx-community", id.org);
    try std.testing.expectEqualStrings("Qwen2.5-Coder-1.5B", id.repo);
}

test "HuggingFaceId.parse handles hyphens and numbers" {
    const id = try HuggingFaceId.parse("organization-123/model-name-456");
    try std.testing.expectEqualStrings("organization-123", id.org);
    try std.testing.expectEqualStrings("model-name-456", id.repo);
}

test "HuggingFaceId.parse rejects missing slash" {
    const result = HuggingFaceId.parse("invalid-no-slash");
    try std.testing.expectError(error.InvalidModelId, result);
}

test "HuggingFaceId.parse rejects leading slash" {
    const result = HuggingFaceId.parse("/leading-slash");
    try std.testing.expectError(error.InvalidModelId, result);
}

test "HuggingFaceId.parse rejects trailing slash" {
    const result = HuggingFaceId.parse("trailing-slash/");
    try std.testing.expectError(error.InvalidModelId, result);
}

test "HuggingFaceId.parse rejects multiple slashes" {
    const result = HuggingFaceId.parse("org/repo/extra");
    try std.testing.expectError(error.InvalidModelId, result);
}

test "HuggingFaceId.parse rejects empty org" {
    const result = HuggingFaceId.parse("/repo-only");
    try std.testing.expectError(error.InvalidModelId, result);
}

test "HuggingFaceId.parse rejects empty repo" {
    const result = HuggingFaceId.parse("org-only/");
    try std.testing.expectError(error.InvalidModelId, result);
}

// Test URL building
test "HuggingFaceId.buildDownloadUrl constructs correct URL" {
    const allocator = std.testing.allocator;
    const id = HuggingFaceId{ .org = "mlx-community", .repo = "Qwen2.5-Coder-1.5B" };

    const url = try id.buildDownloadUrl("model.safetensors", allocator);
    defer allocator.free(url);

    try std.testing.expectEqualStrings("https://huggingface.co/mlx-community/Qwen2.5-Coder-1.5B/resolve/main/model.safetensors", url);
}

test "HuggingFaceId.buildDownloadUrl handles special characters" {
    const allocator = std.testing.allocator;
    const id = HuggingFaceId{ .org = "test-org", .repo = "test-repo" };

    const url = try id.buildDownloadUrl("file with spaces.json", allocator);
    defer allocator.free(url);

    // URL should contain the filename (though ideally it would be URL-encoded)
    try std.testing.expect(std.mem.indexOf(u8, url, "file with spaces.json") != null);
}

test "HuggingFaceId.buildApiUrl constructs correct API URL" {
    const allocator = std.testing.allocator;
    const id = HuggingFaceId{ .org = "mlx-community", .repo = "Qwen2.5-Coder-1.5B" };

    const url = try id.buildApiUrl(allocator);
    defer allocator.free(url);

    try std.testing.expectEqualStrings("https://huggingface.co/api/models/mlx-community/Qwen2.5-Coder-1.5B/tree/main", url);
}

// Test FileInfo structure (if accessible)
test "FileInfo parsing from JSON" {
    const allocator = std.testing.allocator;
    const json =
        \\{
        \\  "path": "model.safetensors",
        \\  "size": 1000000,
        \\  "type": "file"
        \\}
    ;

    const parsed = try std.json.parseFromSlice(huggingface.FileInfo, allocator, json, .{});
    defer parsed.deinit();

    try std.testing.expectEqualStrings("model.safetensors", parsed.value.path);
    try std.testing.expectEqual(@as(u64, 1000000), parsed.value.size);
    try std.testing.expectEqual(huggingface.FileType.file, parsed.value.file_type);
}

test "FileInfo detects directory type" {
    const allocator = std.testing.allocator;
    const json =
        \\{
        \\  "path": "directory",
        \\  "size": 0,
        \\  "type": "directory"
        \\}
    ;

    const parsed = try std.json.parseFromSlice(huggingface.FileInfo, allocator, json, .{});
    defer parsed.deinit();

    try std.testing.expectEqual(huggingface.FileType.directory, parsed.value.file_type);
}
