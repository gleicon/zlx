//! safetensors_test.zig - Tests for safetensors file format parser

const std = @import("std");
const testing = std.testing;
const safetensors = @import("safetensors.zig");

const SafetensorsReader = safetensors.SafetensorsReader;
const TensorInfo = safetensors.TensorInfo;
const Dtype = safetensors.Dtype;
const Header = safetensors.Header;

// ============================================================================
// Dtype tests
// ============================================================================

test "Dtype.fromString BF16" {
    const dtype = Dtype.fromString("BF16");
    try testing.expect(dtype != null);
    try testing.expectEqual(Dtype.bfloat16, dtype.?);
}

test "Dtype.fromString F32" {
    const dtype = Dtype.fromString("F32");
    try testing.expect(dtype != null);
    try testing.expectEqual(Dtype.float32, dtype.?);
}

test "Dtype.fromString F4_E2M1 (MXFP4)" {
    const dtype = Dtype.fromString("F4_E2M1");
    try testing.expect(dtype != null);
    try testing.expectEqual(Dtype.f4_e2m1, dtype.?);
}

test "Dtype.fromString F16" {
    const dtype = Dtype.fromString("F16");
    try testing.expect(dtype != null);
    try testing.expectEqual(Dtype.float16, dtype.?);
}

test "Dtype.fromString unknown returns null" {
    const dtype = Dtype.fromString("UNKNOWN_DTYPE");
    try testing.expect(dtype == null);
}

test "Dtype.fromString all supported types" {
    const cases = [_]struct { str: []const u8, expected: Dtype }{
        .{ .str = "BOOL", .expected = .bool },
        .{ .str = "I8", .expected = .int8 },
        .{ .str = "U8", .expected = .uint8 },
        .{ .str = "I16", .expected = .int16 },
        .{ .str = "U16", .expected = .uint16 },
        .{ .str = "I32", .expected = .int32 },
        .{ .str = "U32", .expected = .uint32 },
        .{ .str = "I64", .expected = .int64 },
        .{ .str = "U64", .expected = .uint64 },
        .{ .str = "F16", .expected = .float16 },
        .{ .str = "F32", .expected = .float32 },
        .{ .str = "F64", .expected = .float64 },
        .{ .str = "BF16", .expected = .bfloat16 },
        .{ .str = "F4_E2M1", .expected = .f4_e2m1 },
        .{ .str = "F8_E4M3", .expected = .f8_e4m3 },
        .{ .str = "F8_E5M2", .expected = .f8_e5m2 },
    };
    for (cases) |case| {
        const dtype = Dtype.fromString(case.str);
        try testing.expect(dtype != null);
        try testing.expectEqual(case.expected, dtype.?);
    }
}

// ============================================================================
// Dtype.sizeInBytes tests
// ============================================================================

test "Dtype.sizeInBytes BF16 is 2" {
    try testing.expectEqual(@as(usize, 2), Dtype.bfloat16.sizeInBytes());
}

test "Dtype.sizeInBytes F32 is 4" {
    try testing.expectEqual(@as(usize, 4), Dtype.float32.sizeInBytes());
}

test "Dtype.sizeInBytes F16 is 2" {
    try testing.expectEqual(@as(usize, 2), Dtype.float16.sizeInBytes());
}

test "Dtype.sizeInBytes F4_E2M1 is 1 (2 per byte)" {
    // F4_E2M1 stores 2 values per byte, sizeInBytes returns 1
    try testing.expectEqual(@as(usize, 1), Dtype.f4_e2m1.sizeInBytes());
}

test "Dtype.sizeInBytes I8 is 1" {
    try testing.expectEqual(@as(usize, 1), Dtype.int8.sizeInBytes());
}

test "Dtype.sizeInBytes I64 is 8" {
    try testing.expectEqual(@as(usize, 8), Dtype.int64.sizeInBytes());
}

// ============================================================================
// TensorInfo tests
// ============================================================================

test "TensorInfo.numElements scalar" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const shape = try allocator.alloc(i64, 0);
    var info = TensorInfo{
        .dtype = .float32,
        .shape = shape,
        .data_offsets = .{ 0, 4 },
    };
    defer info.deinit(allocator);

    // Scalar: product of empty shape = 1
    try testing.expectEqual(@as(usize, 1), info.numElements());
}

test "TensorInfo.numElements 1D tensor" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const shape = try allocator.alloc(i64, 1);
    shape[0] = 1024;
    var info = TensorInfo{
        .dtype = .float32,
        .shape = shape,
        .data_offsets = .{ 0, 4096 },
    };
    defer info.deinit(allocator);

    try testing.expectEqual(@as(usize, 1024), info.numElements());
}

test "TensorInfo.numElements 2D tensor" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const shape = try allocator.alloc(i64, 2);
    shape[0] = 4096;
    shape[1] = 5120;
    var info = TensorInfo{
        .dtype = .bfloat16,
        .shape = shape,
        .data_offsets = .{ 0, 4096 * 5120 * 2 },
    };
    defer info.deinit(allocator);

    try testing.expectEqual(@as(usize, 4096 * 5120), info.numElements());
}

test "TensorInfo.dataSize BF16 2D tensor" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const shape = try allocator.alloc(i64, 2);
    shape[0] = 128;
    shape[1] = 64;
    var info = TensorInfo{
        .dtype = .bfloat16,
        .shape = shape,
        .data_offsets = .{ 0, 128 * 64 * 2 },
    };
    defer info.deinit(allocator);

    // 128 * 64 elements * 2 bytes per BF16
    try testing.expectEqual(@as(usize, 128 * 64 * 2), info.dataSize());
}

test "TensorInfo.dataSize F4_E2M1 packs 2 values per byte" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const shape = try allocator.alloc(i64, 1);
    shape[0] = 64; // 64 elements
    var info = TensorInfo{
        .dtype = .f4_e2m1,
        .shape = shape,
        .data_offsets = .{ 0, 32 },
    };
    defer info.deinit(allocator);

    // 64 elements / 2 = 32 bytes
    try testing.expectEqual(@as(usize, 32), info.dataSize());
}

test "TensorInfo.dataSize F4_E2M1 odd number rounds up" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const shape = try allocator.alloc(i64, 1);
    shape[0] = 65; // 65 elements (odd)
    var info = TensorInfo{
        .dtype = .f4_e2m1,
        .shape = shape,
        .data_offsets = .{ 0, 33 },
    };
    defer info.deinit(allocator);

    // (65 + 1) / 2 = 33 bytes
    try testing.expectEqual(@as(usize, 33), info.dataSize());
}

// ============================================================================
// Header tests
// ============================================================================

test "Header.init creates empty header" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var header = Header.init(allocator);
    defer header.deinit();

    try testing.expect(header.metadata == null);
    try testing.expectEqual(@as(usize, 0), header.tensors.count());
}

// ============================================================================
// SafetensorsReader tests with mock file
// ============================================================================

/// Create a minimal valid safetensors file in memory for testing
fn createMockSafetensors(allocator: std.mem.Allocator) ![]u8 {
    // Header JSON: one BF16 tensor with shape [2, 4], 8 elements = 16 bytes
    const header_json =
        \\{"weight":{"dtype":"BF16","shape":[2,4],"data_offsets":[0,16]}}
    ;

    // Mock BF16 data: 8 values of BF16, 16 bytes total (all zeros)
    const data_size = 16;

    // File format: 8-byte header length + header JSON + tensor data
    const header_len: u64 = header_json.len;
    const total_size = 8 + header_len + data_size;

    const buf = try allocator.alloc(u8, total_size);

    // Write header length as little-endian u64
    std.mem.writeInt(u64, buf[0..8], header_len, .little);

    // Write header JSON
    @memcpy(buf[8..][0..header_json.len], header_json);

    // Zero-fill data section
    @memset(buf[8 + header_json.len ..], 0);

    return buf;
}

test "SafetensorsReader can open and parse mock file" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create mock safetensors file
    const mock_data = try createMockSafetensors(allocator);
    defer allocator.free(mock_data);

    // Write to temp file
    const tmp_path = "/tmp/zlx_safetensors_test.safetensors";
    {
        const f = try std.fs.cwd().createFile(tmp_path, .{});
        defer f.close();
        try f.writeAll(mock_data);
    }
    defer std.fs.cwd().deleteFile(tmp_path) catch {};

    // Parse with SafetensorsReader
    var reader = SafetensorsReader.init(allocator);
    defer reader.deinit();

    try reader.open(tmp_path);

    // Verify header is parsed
    try testing.expect(reader.header != null);

    const header = reader.header.?;
    try testing.expectEqual(@as(usize, 1), header.tensors.count());
}

test "SafetensorsReader.getTensorNames returns tensor names" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const mock_data = try createMockSafetensors(allocator);
    defer allocator.free(mock_data);

    const tmp_path = "/tmp/zlx_safetensors_names_test.safetensors";
    {
        const f = try std.fs.cwd().createFile(tmp_path, .{});
        defer f.close();
        try f.writeAll(mock_data);
    }
    defer std.fs.cwd().deleteFile(tmp_path) catch {};

    var reader = SafetensorsReader.init(allocator);
    defer reader.deinit();
    try reader.open(tmp_path);

    const names = try reader.getTensorNames(allocator);
    defer {
        for (names) |n| allocator.free(n);
        allocator.free(names);
    }

    try testing.expectEqual(@as(usize, 1), names.len);
    try testing.expectEqualStrings("weight", names[0]);
}

test "SafetensorsReader.getTensorInfo returns correct info" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const mock_data = try createMockSafetensors(allocator);
    defer allocator.free(mock_data);

    const tmp_path = "/tmp/zlx_safetensors_info_test.safetensors";
    {
        const f = try std.fs.cwd().createFile(tmp_path, .{});
        defer f.close();
        try f.writeAll(mock_data);
    }
    defer std.fs.cwd().deleteFile(tmp_path) catch {};

    var reader = SafetensorsReader.init(allocator);
    defer reader.deinit();
    try reader.open(tmp_path);

    const info = reader.getTensorInfo("weight");
    try testing.expect(info != null);
    try testing.expectEqual(Dtype.bfloat16, info.?.dtype);
    try testing.expectEqual(@as(usize, 2), info.?.shape.len);
    try testing.expectEqual(@as(i64, 2), info.?.shape[0]);
    try testing.expectEqual(@as(i64, 4), info.?.shape[1]);
}

test "SafetensorsReader.readTensor returns correct data size" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const mock_data = try createMockSafetensors(allocator);
    defer allocator.free(mock_data);

    const tmp_path = "/tmp/zlx_safetensors_read_test.safetensors";
    {
        const f = try std.fs.cwd().createFile(tmp_path, .{});
        defer f.close();
        try f.writeAll(mock_data);
    }
    defer std.fs.cwd().deleteFile(tmp_path) catch {};

    var reader = SafetensorsReader.init(allocator);
    defer reader.deinit();
    try reader.open(tmp_path);

    const data = try reader.readTensor("weight");
    defer allocator.free(data);

    // BF16 tensor [2, 4] = 8 elements * 2 bytes = 16 bytes
    try testing.expectEqual(@as(usize, 16), data.len);
}

test "SafetensorsReader.getTensorInfo returns null for unknown tensor" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const mock_data = try createMockSafetensors(allocator);
    defer allocator.free(mock_data);

    const tmp_path = "/tmp/zlx_safetensors_unknown_test.safetensors";
    {
        const f = try std.fs.cwd().createFile(tmp_path, .{});
        defer f.close();
        try f.writeAll(mock_data);
    }
    defer std.fs.cwd().deleteFile(tmp_path) catch {};

    var reader = SafetensorsReader.init(allocator);
    defer reader.deinit();
    try reader.open(tmp_path);

    const info = reader.getTensorInfo("nonexistent_tensor");
    try testing.expect(info == null);
}

test "SafetensorsReader.readTensor error for unknown tensor" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const mock_data = try createMockSafetensors(allocator);
    defer allocator.free(mock_data);

    const tmp_path = "/tmp/zlx_safetensors_err_test.safetensors";
    {
        const f = try std.fs.cwd().createFile(tmp_path, .{});
        defer f.close();
        try f.writeAll(mock_data);
    }
    defer std.fs.cwd().deleteFile(tmp_path) catch {};

    var reader = SafetensorsReader.init(allocator);
    defer reader.deinit();
    try reader.open(tmp_path);

    const result = reader.readTensor("no_such_tensor");
    try testing.expectError(error.TensorNotFound, result);
}

test "SafetensorsReader multi-tensor file" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create multi-tensor file: weight (BF16 [4]) + bias (F32 [2])
    // BF16 [4] = 8 bytes, F32 [2] = 8 bytes
    const header_json =
        \\{"weight":{"dtype":"BF16","shape":[4],"data_offsets":[0,8]},"bias":{"dtype":"F32","shape":[2],"data_offsets":[8,16]}}
    ;
    const header_len: u64 = header_json.len;
    const total_size = 8 + header_len + 16;

    const buf = try allocator.alloc(u8, total_size);
    defer allocator.free(buf);

    std.mem.writeInt(u64, buf[0..8], header_len, .little);
    @memcpy(buf[8..][0..header_json.len], header_json);
    @memset(buf[8 + header_json.len ..], 0);

    const tmp_path = "/tmp/zlx_safetensors_multi_test.safetensors";
    {
        const f = try std.fs.cwd().createFile(tmp_path, .{});
        defer f.close();
        try f.writeAll(buf);
    }
    defer std.fs.cwd().deleteFile(tmp_path) catch {};

    var reader = SafetensorsReader.init(allocator);
    defer reader.deinit();
    try reader.open(tmp_path);

    const header = reader.header.?;
    try testing.expectEqual(@as(usize, 2), header.tensors.count());

    // Verify weight tensor
    const weight_info = reader.getTensorInfo("weight");
    try testing.expect(weight_info != null);
    try testing.expectEqual(Dtype.bfloat16, weight_info.?.dtype);

    // Verify bias tensor
    const bias_info = reader.getTensorInfo("bias");
    try testing.expect(bias_info != null);
    try testing.expectEqual(Dtype.float32, bias_info.?.dtype);
}

test "SafetensorsReader.data_start is correct after header" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const mock_data = try createMockSafetensors(allocator);
    defer allocator.free(mock_data);

    const tmp_path = "/tmp/zlx_safetensors_offset_test.safetensors";
    {
        const f = try std.fs.cwd().createFile(tmp_path, .{});
        defer f.close();
        try f.writeAll(mock_data);
    }
    defer std.fs.cwd().deleteFile(tmp_path) catch {};

    var reader = SafetensorsReader.init(allocator);
    defer reader.deinit();
    try reader.open(tmp_path);

    // Header JSON length
    const header_json =
        \\{"weight":{"dtype":"BF16","shape":[2,4],"data_offsets":[0,16]}}
    ;
    // data_start = 8 (header_len bytes) + len(header_json)
    const expected_data_start: u64 = 8 + @as(u64, header_json.len);
    try testing.expectEqual(expected_data_start, reader.data_start);
}
