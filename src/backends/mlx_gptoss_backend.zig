//! mlx_gptoss_backend.zig - MLX GPT-OSS backend for zlx

const std = @import("std");
const backend = @import("backend.zig");
const gptoss_mlx = @import("../gptoss_mlx.zig");
const harmony = @import("../harmony/harmony.zig");
const harmony_template = @import("../harmony/template.zig");
const tool_executor = @import("../tools/tool_executor.zig");
const weight_loader = @import("../weight/gptoss_loader.zig");
const tokenizer_mod = @import("../mlx.zig/src/tokenizer.zig");
const mlx_api = @import("../mlx.zig/src/mlx.zig");

const GenerationParams = backend.GenerationParams;
const GenerationResult = backend.GenerationResult;
const TokenResult = backend.TokenResult;
const StopReason = backend.StopReason;
const GPTOSSTransformer = gptoss_mlx.GPTOSSTransformer;
const GPTOSSConfig = gptoss_mlx.GPTOSSConfig;
const GPTOSSTokenGenerator = gptoss_mlx.GPTOSSTokenGenerator;
const HarmonyTemplate = harmony_template.HarmonyTemplate;
const ReasoningEffort = harmony.ReasoningEffort;
const ToolExecutor = tool_executor.ToolExecutor;
const GPTOSSWeightLoader = weight_loader.GPTOSSWeightLoader;
const GPTOSSWeightConfig = weight_loader.GPTOSSWeightConfig;

/// Model variant enum
pub const ModelVariant = enum { gptoss_20b, gptoss_120b };

/// MLX GPT-OSS backend implementation
pub const MLXGPTOSSBackend = struct {
    allocator: std.mem.Allocator,
    model_path: []const u8,
    model_variant: ModelVariant,
    transformer: ?GPTOSSTransformer,
    weight_loader_inst: ?GPTOSSWeightLoader,
    tool_executor_inst: ?ToolExecutor,
    harmony_tmpl: HarmonyTemplate,
    config: BackendConfig,
    loaded: bool,

    pub const BackendConfig = struct {
        max_context: usize = 131072,
        reasoning_effort: ReasoningEffort = .medium,
        enable_tools: bool = true,
        temperature: f32 = 0.7,
        top_p: f32 = 0.9,
        top_k: u32 = 40,
        max_tokens: u32 = 1024,
    };

    pub fn init(
        allocator: std.mem.Allocator,
        model_path: []const u8,
        backend_config: BackendConfig,
    ) !MLXGPTOSSBackend {
        // Detect model variant from path — check 120b first (contains "20b" substring)
        const variant: ModelVariant = if (std.mem.indexOf(u8, model_path, "120b") != null)
            .gptoss_120b
        else if (std.mem.indexOf(u8, model_path, "20b") != null)
            .gptoss_20b
        else
            .gptoss_20b; // Default

        const tool_exec: ?ToolExecutor = if (backend_config.enable_tools)
            ToolExecutor.init(allocator)
        else
            null;

        return .{
            .allocator = allocator,
            .model_path = try allocator.dupe(u8, model_path),
            .model_variant = variant,
            .transformer = null,
            .weight_loader_inst = null,
            .tool_executor_inst = tool_exec,
            .harmony_tmpl = HarmonyTemplate.init(
                allocator,
                "harmony_gpt_oss",
                backend_config.reasoning_effort,
                backend_config.enable_tools,
                null,
            ),
            .config = backend_config,
            .loaded = false,
        };
    }

    pub fn deinit(self: *MLXGPTOSSBackend) void {
        self.allocator.free(self.model_path);

        // Free GPU stream before deiniting transformer (GPTOSSTransformer.deinit does not free it)
        if (self.transformer) |*t| {
            mlx_api.streamFree(t.stream);
            t.deinit();
        }
        if (self.weight_loader_inst) |*wl| wl.deinit();
        if (self.tool_executor_inst) |*te| te.deinit();
    }

    /// Load model weights
    pub fn load(self: *MLXGPTOSSBackend) !void {
        if (self.loaded) return;

        // Create weight config based on variant
        const weight_config = switch (self.model_variant) {
            .gptoss_20b => GPTOSSWeightConfig.from20B(),
            .gptoss_120b => GPTOSSWeightConfig.from120B(),
        };

        // Create and load weight loader
        self.weight_loader_inst = GPTOSSWeightLoader.init(
            self.allocator,
            self.model_path,
            weight_config,
        );

        try self.weight_loader_inst.?.load(null);

        // Create transformer
        const gptoss_config = switch (self.model_variant) {
            .gptoss_20b => GPTOSSConfig.gptoss20b(),
            .gptoss_120b => GPTOSSConfig.gptoss120b(),
        };

        // Initialize transformer with a real MLX GPU stream
        const gpu_stream = mlx_api.defaultGpuStreamNew();
        self.transformer = try GPTOSSTransformer.init(
            self.allocator,
            gptoss_config,
            gpu_stream,
        );

        // Load weights into transformer
        try self.weight_loader_inst.?.loadIntoTransformer(&self.transformer.?);

        // Register tools if enabled
        if (self.tool_executor_inst) |*te| {
            const browser_mod = @import("../tools/browser.zig");
            const python_mod = @import("../tools/python.zig");

            const browser_config = browser_mod.BrowserTool.BrowserConfig{
                .timeout_ms = 30000,
                .max_page_size = 1024 * 1024,
            };
            const browser_bk = browser_mod.BrowserTool.SearchBackend{ .stub = .{} };

            try te.registerBrowser(browser_config, browser_bk);

            const python_config = python_mod.PythonConfig{
                .docker_image = "python:3.11-slim",
                .timeout_ms = 60000,
            };
            try te.registerPython(python_config);
        }

        self.loaded = true;
    }

    /// Generate response from token stream
    pub fn generate(
        self: *MLXGPTOSSBackend,
        tokens: []const u32,
        params: GenerationParams,
        allocator: std.mem.Allocator,
    ) !GenerationResult {
        if (!self.loaded) {
            try self.load();
        }

        const transformer = &self.transformer.?;

        // Generate tokens using GPTOSSTransformer
        const output_tokens = try transformer.generate(
            tokens,
            params.max_tokens,
            params.temperature,
        );
        defer allocator.free(output_tokens);

        // Build a simple token iterator over the output tokens
        const ctx = try allocator.create(SimpleIterCtx);
        ctx.* = .{
            .tokens = try allocator.dupe(u32, output_tokens),
            .pos = 0,
        };

        return GenerationResult{
            .token_iterator = backend.TokenIterator{
                .next_fn = simpleIterNext,
                .ctx = ctx,
                .deinit_fn = simpleIterDeinit,
            },
            .tokens_generated = @intCast(output_tokens.len),
            .finish_reason = .eos,
        };
    }

    /// Check if backend supports tools
    pub fn supportsTools(self: *const MLXGPTOSSBackend) bool {
        return self.config.enable_tools and self.tool_executor_inst != null;
    }

    /// Unload model
    pub fn unload(self: *MLXGPTOSSBackend) void {
        if (self.transformer) |*t| {
            mlx_api.streamFree(t.stream);
            t.deinit();
            self.transformer = null;
        }
        if (self.weight_loader_inst) |*wl| {
            wl.deinit();
            self.weight_loader_inst = null;
        }
        self.loaded = false;
    }

    /// Get vocabulary size
    pub fn getVocabSize(self: *const MLXGPTOSSBackend) u32 {
        return switch (self.model_variant) {
            .gptoss_20b => 151936,
            .gptoss_120b => 151936,
        };
    }

    /// Get EOS token
    pub fn getEosToken(self: *const MLXGPTOSSBackend) u32 {
        _ = self;
        return 100257; // GPT-OSS EOS token
    }

    /// Get BOS token
    pub fn getBosToken(self: *const MLXGPTOSSBackend) u32 {
        _ = self;
        return 100256; // GPT-OSS BOS token
    }
};

/// Simple token iterator context
const SimpleIterCtx = struct {
    tokens: []u32,
    pos: usize,
};

fn simpleIterNext(ctx: *anyopaque, allocator: std.mem.Allocator) anyerror!?TokenResult {
    const self = @as(*SimpleIterCtx, @ptrCast(@alignCast(ctx)));
    if (self.pos >= self.tokens.len) return null;

    const token = self.tokens[self.pos];
    self.pos += 1;

    return TokenResult{
        .token = token,
        .text = try std.fmt.allocPrint(allocator, "{d}", .{token}),
        .logprob = 0.0,
        .finish_reason = if (self.pos >= self.tokens.len) .eos else null,
    };
}

fn simpleIterDeinit(ctx: *anyopaque, allocator: std.mem.Allocator) void {
    const self = @as(*SimpleIterCtx, @ptrCast(@alignCast(ctx)));
    allocator.free(self.tokens);
    allocator.destroy(self);
}

// ── Public factory functions ────────────────────────────────────────────────

pub fn createBackend(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    config: MLXGPTOSSBackend.BackendConfig,
) !*MLXGPTOSSBackend {
    const ptr = try allocator.create(MLXGPTOSSBackend);
    ptr.* = try MLXGPTOSSBackend.init(allocator, model_path, config);
    return ptr;
}

pub fn destroyBackend(backend_ptr: *anyopaque, allocator: std.mem.Allocator) void {
    const self = @as(*MLXGPTOSSBackend, @ptrCast(@alignCast(backend_ptr)));
    self.deinit();
    allocator.destroy(self);
}

/// Tokenize text using the GPT-OSS tokenizer (tokenizer.json in model_path directory)
pub fn tokenize(ptr: *anyopaque, text: []const u8, allocator: std.mem.Allocator) ![]u32 {
    const self = @as(*MLXGPTOSSBackend, @ptrCast(@alignCast(ptr)));

    // Initialize tokenizer from model directory (expects tokenizer.json in model_path)
    var tokenizer = tokenizer_mod.Tokenizer.init(allocator, self.model_path) catch |err| {
        std.log.err("GPT-OSS tokenizer load failed from {s}: {}", .{ self.model_path, err });
        return err;
    };
    defer tokenizer.deinit();

    // encode() returns []const u32 via toOwnedSlice(allocator) — already owned by allocator.
    // Cast to mutable []u32 to match return type. No dupe needed.
    const tokens: []const u32 = try tokenizer.encode(text);
    return @constCast(tokens);
}

/// Generate tokens via vtable
pub fn generateVtable(
    ptr: *anyopaque,
    tokens: []const u32,
    params: GenerationParams,
    allocator: std.mem.Allocator,
) !GenerationResult {
    const self = @as(*MLXGPTOSSBackend, @ptrCast(@alignCast(ptr)));
    return self.generate(tokens, params, allocator);
}

/// Deinit backend via vtable
pub fn deinitVtable(ptr: *anyopaque, allocator: std.mem.Allocator) void {
    destroyBackend(ptr, allocator);
}

/// Get vocab size via vtable
pub fn getVocabSizeVtable(ptr: *anyopaque) u32 {
    const self = @as(*MLXGPTOSSBackend, @ptrCast(@alignCast(ptr)));
    return self.getVocabSize();
}

/// Get EOS token via vtable
pub fn getEosTokenVtable(ptr: *anyopaque) u32 {
    const self = @as(*MLXGPTOSSBackend, @ptrCast(@alignCast(ptr)));
    return self.getEosToken();
}

/// Get BOS token via vtable
pub fn getBosTokenVtable(ptr: *anyopaque) u32 {
    const self = @as(*MLXGPTOSSBackend, @ptrCast(@alignCast(ptr)));
    return self.getBosToken();
}
