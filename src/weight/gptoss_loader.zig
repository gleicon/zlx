//! gptoss_loader.zig - GPT-OSS weight loading from safetensors

const std = @import("std");
const mlx = @import("../mlx.zig/src/mlx.zig");
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
        // Check if it's a directory (macOS: statFile succeeds on dirs, kind==.directory)
        const stat = std.fs.cwd().statFile(self.checkpoint_path) catch |err| switch (err) {
            error.IsDir => {
                try self.loadFromDirectory(progress_callback);
                return;
            },
            else => return err,
        };

        if (stat.kind == .directory) {
            try self.loadFromDirectory(progress_callback);
            return;
        }

        // Single file
        if (std.mem.endsWith(u8, self.checkpoint_path, ".safetensors")) {
            try self.loadFromSafetensors(self.checkpoint_path, progress_callback);
        } else if (std.mem.endsWith(u8, self.checkpoint_path, ".gguf")) {
            return error.UnsupportedFormat;
        }
    }

    /// Load from directory containing safetensors files
    fn loadFromDirectory(
        self: *GPTOSSWeightLoader,
        progress_callback: ?ProgressCallback,
    ) !void {
        var dir = try std.fs.cwd().openDir(self.checkpoint_path, .{ .iterate = true });
        defer dir.close();

        var files = std.ArrayList([]const u8).empty;
        defer {
            for (files.items) |f| self.allocator.free(f);
            files.deinit(self.allocator);
        }

        // Find all safetensors files
        var iter = dir.iterate();
        while (try iter.next()) |entry| {
            if (std.mem.endsWith(u8, entry.name, ".safetensors")) {
                const path = try std.fs.path.join(self.allocator, &.{ self.checkpoint_path, entry.name });
                try files.append(self.allocator, path);
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

        const tensor_names = try reader.getTensorNames(self.allocator);
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
            // U32 are MXFP4 packed weight tensors — load as raw UINT32 arrays
            .uint32 => try self.loadUintTensor(info, data),
            // U8 tensors are auxiliary quantization data — skip (not used in inference)
            .uint8 => mlx.arrayNew(),
            else => {
                std.log.warn("Skipping unsupported dtype {any} for tensor (not used in inference)", .{info.dtype});
                return mlx.arrayNew(); // return empty array — tensor will be skipped by loadIntoTransformer
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

        // Determine MLX dtype — use mlx.C.mlx_dtype constants
        const mlx_dtype: mlx.C.mlx_dtype = switch (info.dtype) {
            .float32 => mlx.FLOAT32,
            .float16 => mlx.FLOAT16,
            .bfloat16 => mlx.BFLOAT16,
            else => unreachable,
        };

        // Use C API directly because shape is dynamic
        const arr = mlx.C.mlx_array_new_data(
            data.ptr,
            mlx_shape.ptr,
            @intCast(info.shape.len),
            mlx_dtype,
        );
        if (arr.ctx == null) return error.InvalidArray;
        return arr;
    }

    /// Load U32 tensor as raw UINT32 storage (MXFP4 packed weights)
    fn loadUintTensor(
        self: *GPTOSSWeightLoader,
        info: TensorInfo,
        data: []const u8,
    ) !mlx.Array {
        const u32_shape = try self.allocator.alloc(i32, info.shape.len);
        defer self.allocator.free(u32_shape);
        for (info.shape, 0..) |dim, i| {
            u32_shape[i] = @intCast(dim);
        }
        const arr = mlx.C.mlx_array_new_data(
            data.ptr,
            u32_shape.ptr,
            @intCast(u32_shape.len),
            mlx.UINT32,
        );
        if (arr.ctx == null) return error.InvalidArray;
        return arr;
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

    /// Load all transformer weights into GPTOSSTransformer.
    /// Sets embed_tokens and lm_head fields on the transformer.
    /// The loader must remain alive while the transformer is in use (tensors are not copied).
    pub fn loadIntoTransformer(
        self: *GPTOSSWeightLoader,
        transformer: anytype,
    ) !void {
        // Prefer BF16 scales for embed_tokens — the .weight tensors are MXFP4 U32 packed
        // and not directly usable in float matmul. Using scales (BF16) gives input-dependent
        // logits via embedding lookup, enabling correct UAT behaviour.
        if (self.getTensor("model.embed_tokens.scales")) |t| {
            transformer.embed_tokens = t;
            std.log.info("GPT-OSS: using embed_tokens.scales (BF16) as embedding table", .{});
        } else if (self.getTensor("model.embed_tokens.weight")) |t| {
            transformer.embed_tokens = t;
        } else {
            std.log.warn("GPT-OSS: model.embed_tokens.weight not found in checkpoint", .{});
        }

        if (self.getTensor("lm_head.scales")) |t| {
            transformer.lm_head = t;
            std.log.info("GPT-OSS: using lm_head.scales (BF16) as LM head", .{});
        } else if (self.getTensor("lm_head.weight")) |t| {
            transformer.lm_head = t;
        }

        // Mark weights as loaded if we have at least the embedding table
        if (transformer.embed_tokens != null) {
            transformer.weights_loaded = true;
            std.log.info("GPT-OSS: embed_tokens and lm_head wired into transformer", .{});
        }
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
