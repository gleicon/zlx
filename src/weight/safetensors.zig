//! safetensors.zig - Safetensors file format parser

const std = @import("std");

/// Tensor data type
pub const Dtype = enum {
    bool,
    int8,
    uint8,
    int16,
    uint16,
    int32,
    uint32,
    int64,
    uint64,
    float16,
    float32,
    float64,
    bfloat16,
    // MXFP4 variants
    f4_e2m1, // 4-bit FP with 2 exponent, 1 mantissa
    f8_e4m3, // 8-bit FP
    f8_e5m2, // 8-bit FP

    pub fn fromString(str: []const u8) ?Dtype {
        const map = std.StaticStringMap(Dtype).initComptime(&.{
            .{ "BOOL", .bool },
            .{ "I8", .int8 },
            .{ "U8", .uint8 },
            .{ "I16", .int16 },
            .{ "U16", .uint16 },
            .{ "I32", .int32 },
            .{ "U32", .uint32 },
            .{ "I64", .int64 },
            .{ "U64", .uint64 },
            .{ "F16", .float16 },
            .{ "F32", .float32 },
            .{ "F64", .float64 },
            .{ "BF16", .bfloat16 },
            .{ "F4_E2M1", .f4_e2m1 },
            .{ "F8_E4M3", .f8_e4m3 },
            .{ "F8_E5M2", .f8_e5m2 },
        });
        return map.get(str);
    }

    pub fn sizeInBytes(self: Dtype) usize {
        return switch (self) {
            .bool, .int8, .uint8 => 1,
            .int16, .uint16, .float16, .bfloat16 => 2,
            .f8_e4m3, .f8_e5m2 => 1,
            .int32, .uint32, .float32 => 4,
            .int64, .uint64, .float64 => 8,
            .f4_e2m1 => 1, // 2 values per byte
        };
    }
};

/// Tensor information from header
pub const TensorInfo = struct {
    dtype: Dtype,
    shape: []i64,
    data_offsets: [2]u64, // [start, end] in file

    pub fn deinit(self: *TensorInfo, allocator: std.mem.Allocator) void {
        allocator.free(self.shape);
    }

    /// Calculate total number of elements
    pub fn numElements(self: *const TensorInfo) usize {
        var count: usize = 1;
        for (self.shape) |dim| {
            count *= @intCast(dim);
        }
        return count;
    }

    /// Calculate data size in bytes
    pub fn dataSize(self: *const TensorInfo) usize {
        const elements = self.numElements();
        if (self.dtype == .f4_e2m1) {
            return (elements + 1) / 2; // 2 values per byte
        }
        return elements * self.dtype.sizeInBytes();
    }
};

/// Safetensors header
pub const Header = struct {
    metadata: ?std.json.Value = null,
    tensors: std.StringHashMap(TensorInfo),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Header {
        return .{
            .tensors = std.StringHashMap(TensorInfo).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Header) void {
        // Free both keys (owned copies) and values (shapes)
        var key_iter = self.tensors.keyIterator();
        while (key_iter.next()) |key| {
            self.allocator.free(key.*);
        }
        var val_iter = self.tensors.valueIterator();
        while (val_iter.next()) |info| {
            info.deinit(self.allocator);
        }
        self.tensors.deinit();
        // metadata is not stored (std.json.Value not owned after parsing)
    }
};

/// Safetensors file reader
pub const SafetensorsReader = struct {
    allocator: std.mem.Allocator,
    file: ?std.fs.File,
    header: ?Header,
    data_start: u64,
    data_len: u64,

    pub fn init(allocator: std.mem.Allocator) SafetensorsReader {
        return .{
            .allocator = allocator,
            .file = null,
            .header = null,
            .data_start = 0,
            .data_len = 0,
        };
    }

    pub fn deinit(self: *SafetensorsReader) void {
        if (self.file) |*f| f.close();
        if (self.header) |*h| h.deinit();
    }

    /// Open and parse a safetensors file
    pub fn open(self: *SafetensorsReader, file_path: []const u8) !void {
        // Open file
        self.file = try std.fs.cwd().openFile(file_path, .{ .mode = .read_only });

        // Read header length (first 8 bytes, little-endian u64)
        var header_len_bytes: [8]u8 = undefined;
        _ = try self.file.?.read(&header_len_bytes);
        const header_len = std.mem.readInt(u64, &header_len_bytes, .little);

        // Validate header length
        if (header_len > 100_000_000) { // 100MB max header
            return error.HeaderTooLarge;
        }

        // Read header
        const header_buf = try self.allocator.alloc(u8, header_len);
        defer self.allocator.free(header_buf);

        _ = try self.file.?.read(header_buf);

        // Parse header JSON
        self.header = try self.parseHeader(header_buf);

        // Data starts after header
        self.data_start = 8 + header_len;
        self.data_len = (try self.file.?.stat()).size - self.data_start;
    }

    /// Parse header JSON
    fn parseHeader(self: *SafetensorsReader, data: []const u8) !Header {
        var header = Header.init(self.allocator);
        errdefer header.deinit();

        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            data,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        // Extract metadata if present (stored as std.json.Value)
        if (parsed.value.object.get("__metadata__")) |meta| {
            _ = meta; // Metadata present but not stored (std.json.Value not owned after parsed.deinit)
        }

        // Extract tensor information
        var iter = parsed.value.object.iterator();
        while (iter.next()) |entry| {
            if (std.mem.eql(u8, entry.key_ptr.*, "__metadata__")) continue;

            const tensor_data = entry.value_ptr.*;
            const dtype_str = tensor_data.object.get("dtype").?.string;
            const dtype = Dtype.fromString(dtype_str) orelse {
                std.log.err("Unknown dtype: {s}", .{dtype_str});
                continue;
            };

            // Parse shape
            const shape_json = tensor_data.object.get("shape").?.array;
            var shape = try self.allocator.alloc(i64, shape_json.items.len);
            for (shape_json.items, 0..) |item, i| {
                shape[i] = item.integer;
            }

            // Parse data offsets
            const offsets = tensor_data.object.get("data_offsets").?.array;
            const data_offsets = [2]u64{
                @intCast(offsets.items[0].integer),
                @intCast(offsets.items[1].integer),
            };

            // Dupe the key so it survives parsed.deinit()
            const key_owned = try self.allocator.dupe(u8, entry.key_ptr.*);
            errdefer self.allocator.free(key_owned);
            try header.tensors.put(key_owned, .{
                .dtype = dtype,
                .shape = shape,
                .data_offsets = data_offsets,
            });
        }

        return header;
    }

    /// Read tensor data
    pub fn readTensor(
        self: *SafetensorsReader,
        name: []const u8,
    ) ![]const u8 {
        const header = self.header orelse return error.NoHeader;
        const info = header.tensors.get(name) orelse return error.TensorNotFound;

        // Calculate absolute offsets
        const abs_start = self.data_start + info.data_offsets[0];
        const abs_end = self.data_start + info.data_offsets[1];
        const size = abs_end - abs_start;

        // Seek to data
        try self.file.?.seekTo(abs_start);

        // Read data
        const data = try self.allocator.alloc(u8, size);
        _ = try self.file.?.read(data);

        return data;
    }

    /// Get list of tensor names
    pub fn getTensorNames(self: *const SafetensorsReader, allocator: std.mem.Allocator) ![][]const u8 {
        const header = self.header orelse return error.NoHeader;

        var names = std.ArrayList([]const u8).empty;
        errdefer names.deinit(allocator);

        var iter = header.tensors.keyIterator();
        while (iter.next()) |key| {
            try names.append(allocator, try allocator.dupe(u8, key.*));
        }

        return names.toOwnedSlice(allocator);
    }

    /// Get tensor info
    pub fn getTensorInfo(self: *const SafetensorsReader, name: []const u8) ?*const TensorInfo {
        const header = self.header orelse return null;
        return header.tensors.getPtr(name);
    }
};
