//! gptoss_loader.zig - GPT-OSS weight loading from safetensors

const std = @import("std");
const mlx = @import("mlx.zig/src/mlx.zig");
const safetensors = @import("safetensors.zig");
const mxfp4 = @import("../mxfp4.zig");

const SafetensorsReader = safetensors.SafetensorsReader;
const TensorInfo = safetensors.TensorInfo;
const Dtype = safetensors.Dtype;

/// GPT-OSS model configuration
pub const GPTOSSWeightConfig = struct {
    hidden_size: usize,
    num_layers: usize,
    num_experts: usize,
    num_shared_experts: usize,
    top_k: usize,
    vocab_size: usize,
    intermediate_size: usize,

    pub fn from20B() GPTOSSWeightConfig {
        return .{
            .hidden_size = 5120,
            .num_layers = 40,
            .num_experts = 32,
            .num_shared_experts = 2,
            .top_k = 4,
            .vocab_size = 151936,
            .intermediate_size = 9216,
        };
    }

    pub fn from120B() GPTOSSWeightConfig {
        return .{
            .hidden_size = 6656,
            .num_layers = 56,
            .num_experts = 64,
            .num_shared_experts = 2,
            .top_k = 6,
            .vocab_size = 151936,
            .intermediate_size = 10240,
        };
    }
};

/// Weight loading progress callback
pub const ProgressCallback = *const fn (current: usize, total: usize, tensor_name: []const u8) void;

/// GPT-OSS weight loader
pub const GPTOSSWeightLoader = struct {
    allocator: std.mem.Allocator,
    checkpoint_path: []const u8,
    config: GPTOSSWeightConfig,
    tensors: std.StringHashMap(mlx.Array),
    loaded: bool,

    pub fn init(
        allocator: std.mem.Allocator,
        checkpoint_path: []const u8,
        config: GPTOSSWeightConfig,
    ) GPTOSSWeightLoader {
        return .{
            .allocator = allocator,
            .checkpoint_path = checkpoint_path,
            .config = config,
            .tensors = std.StringHashMap(mlx.Array).init(allocator),
            .loaded = false,
        };
    }

    pub fn deinit(self: *GPTOSSWeightLoader) void {
        var iter = self.tensors.valueIterator();
        while (iter.next()) |arr| {
            mlx.arrayFree(arr.*);
        }
        self.tensors.deinit();
    }

    /// Load weights from checkpoint directory
    pub fn load(
        self: *GPTOSSWeightLoader,
        progress_callback: ?ProgressCallback,
    ) !void {
        // Check if it's a single file or directory
        const stat = std.fs.cwd().statFile(self.checkpoint_path) catch |err| switch (err) {
            error.IsDir => {
                // Load from directory
                try self.loadFromDirectory(progress_callback);
                return;
            },
            else => return err,
        };

        // Single file
        if (std.mem.endsWith(u8, self.checkpoint_path, ".safetensors")) {
            try self.loadFromSafetensors(self.checkpoint_path, progress_callback);
        } else if (std.mem.endsWith(u8, self.checkpoint_path, ".gguf")) {
            // GGUF format not supported in this loader
            return error.UnsupportedFormat;
        }

        _ = stat;
    }

    /// Load from directory containing safetensors files
    fn loadFromDirectory(
        self: *GPTOSSWeightLoader,
        progress_callback: ?ProgressCallback,
    ) !void {
        var dir = try std.fs.cwd().openDir(self.checkpoint_path, .{ .iterate = true });
        defer dir.close();

        var files = std.ArrayList([]const u8).init(self.allocator);
        defer {
            for (files.items) |f| self.allocator.free(f);
            files.deinit();
        }

        // Find all safetensors files
        var iter = dir.iterate();
        while (try iter.next()) |entry| {
            if (std.mem.endsWith(u8, entry.name, ".safetensors")) {
                const path = try std.fs.path.join(self.allocator, &.{ self.checkpoint_path, entry.name });
                try files.append(path);
            }
        }

        // Sort files to load in order
        std.sort.block([]const u8, files.items, {}, stringLessThan);

        // Load each file
        for (files.items, 0..) |file_path, i| {
            if (progress_callback) |cb| {
                cb(i, files.items.len, file_path);
            }
            try self.loadFromSafetensors(file_path, null);
        }

        self.loaded = true;
    }

    /// Load from single safetensors file
    fn loadFromSafetensors(
        self: *GPTOSSWeightLoader,
        file_path: []const u8,
        progress_callback: ?ProgressCallback,
    ) !void {
        var reader = SafetensorsReader.init(self.allocator);
        defer reader.deinit();

        try reader.open(file_path);

        const header = reader.header orelse return error.NoHeader;

        var tensor_names = try reader.getTensorNames(self.allocator);
        defer {
            for (tensor_names) |n| self.allocator.free(n);
            self.allocator.free(tensor_names);
        }

        for (tensor_names, 0..) |name, i| {
            if (progress_callback) |cb| {
                cb(i, tensor_names.len, name);
            }

            const info = header.tensors.get(name) orelse continue;

            // Read raw tensor data
            const raw_data = try reader.readTensor(name);
            defer self.allocator.free(raw_data);

            // Convert to MLX array based on dtype
            const mlx_array = try self.tensorToMLX(info, raw_data);

            try self.tensors.put(try self.allocator.dupe(u8, name), mlx_array);
        }
    }

    /// Convert tensor to MLX array
    fn tensorToMLX(
        self: *GPTOSSWeightLoader,
        info: TensorInfo,
        data: []const u8,
    ) !mlx.Array {
        return switch (info.dtype) {
            .bfloat16, .float16, .float32 => try self.loadFloatTensor(info, data),
            .f4_e2m1 => try self.loadMXFP4Tensor(info, data),
            else => {
                std.log.err("Unsupported dtype: {any}", .{info.dtype});
                return error.UnsupportedDtype;
            },
        };
    }

    /// Load float tensor (BF16/FP16/FP32)
    fn loadFloatTensor(
        self: *GPTOSSWeightLoader,
        info: TensorInfo,
        data: []const u8,
    ) !mlx.Array {
        // Convert shape to MLX format
        const mlx_shape = try self.allocator.alloc(i32, info.shape.len);
        defer self.allocator.free(mlx_shape);

        for (info.shape, 0..) |dim, i| {
            mlx_shape[i] = @intCast(dim);
        }

        // Determine MLX dtype
        const mlx_dtype: mlx.Dtype = switch (info.dtype) {
            .float32 => mlx.Float32,
            .float16 => mlx.Float16,
            .bfloat16 => mlx.BFloat16,
            else => unreachable,
        };

        return mlx.arrayFromData(
            data.ptr,
            @intCast(data.len / info.dtype.sizeInBytes()),
            @intCast(info.shape.len),
            mlx_shape.ptr,
            mlx_dtype,
        );
    }

    /// Load MXFP4 tensor with dequantization
    fn loadMXFP4Tensor(
        self: *GPTOSSWeightLoader,
        info: TensorInfo,
        data: []const u8,
    ) !mlx.Array {
        var shape_usize = try self.allocator.alloc(usize, info.shape.len);
        defer self.allocator.free(shape_usize);

        for (info.shape, 0..) |dim, i| {
            shape_usize[i] = @intCast(dim);
        }

        return try mxfp4.safetensorsToMLX(
            self.allocator,
            data,
            shape_usize,
            true, // Use GPU
        );
    }

    /// Get loaded tensor
    pub fn getTensor(self: *const GPTOSSWeightLoader, name: []const u8) ?mlx.Array {
        return self.tensors.get(name);
    }

    /// Load all transformer weights into GPTOSSTransformer
    pub fn loadIntoTransformer(
        self: *GPTOSSWeightLoader,
        transformer: anytype,
    ) !void {
        _ = transformer;
        _ = self;
        // TODO: Map tensor names to transformer layers
    }

    /// Check if weights are loaded
    pub fn isLoaded(self: *const GPTOSSWeightLoader) bool {
        return self.loaded;
    }

    /// Get memory usage of loaded weights
    pub fn getMemoryUsage(self: *const GPTOSSWeightLoader) usize {
        var total: usize = 0;
        var iter = self.tensors.valueIterator();
        while (iter.next()) |arr| {
            total += @intCast(mlx.arrayNbytes(arr.*));
        }
        return total;
    }
};

fn stringLessThan(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.lessThan(u8, a, b);
}
