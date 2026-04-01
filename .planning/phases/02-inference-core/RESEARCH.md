# Phase 2: Inference Core - Research

## MLX.zig Transformer API

### Initialization Pattern

**Source**: `src/mlx.zig/src/mlx.zig:881-957`

```zig
pub fn Transformer(comptime ModelType: type, comptime ConfigType: type) type {
    return struct {
        pub fn init(allocator: std.mem.Allocator, model_path: []const u8) !Self {
            // 1. Load config.json from model directory
            const model_config = try utils.loadConfigJson(ConfigType, allocator, model_path, true);
            
            // 2. Determine dtype from config (bfloat16 or float16)
            const mlx_dtype = if (std.mem.eql(u8, "bfloat16", model_config.value.torch_dtype)) BFLOAT16 else FLOAT16;
            
            // 3. Create MLXConfig with GPU stream
            var mlx_config = try MLXConfig.init(allocator, mlx_dtype);
            
            // 4. Initialize the model architecture
            const model = try ModelType.init(model_config.value, &mlx_config);
            
            // 5. Load weights from .safetensors files
            try loadModelSafetensors(&mlx_config.weights_hash, model_path, mlx_config.stream);
            
            return .{
                .mlx_config = mlx_config,
                .model = model,
                .eos_token_ids = model_config.value.eos_token_ids.?,
            };
        }
    };
}
```

### GPU Stream Setup

**Source**: `src/mlx.zig/src/mlx.zig:410-429`

```zig
pub const MLXConfig = struct {
    allocator: std.mem.Allocator,
    stream: Stream,              // GPU stream for Metal
    weights_hash: std.StringHashMap(*Array),
    dtype: C.mlx_dtype,

    pub fn init(allocator: std.mem.Allocator, mlx_dtype: C.mlx_dtype) !MLXConfig {
        return MLXConfig{
            .allocator = allocator,
            .stream = C.mlx_default_gpu_stream_new(),  // ← GPU stream for Metal
            .weights_hash = std.StringHashMap(*Array).init(allocator),
            .dtype = mlx_dtype,
        };
    }
};
```

### Generate Function

**Source**: `src/mlx.zig/src/mlx.zig:908-956`

**Signature**: `pub fn generate(self: *Self, initial_tokens: []const u32, num_tokens: usize) ![]u32`

The current implementation accumulates all tokens and returns them at the end. For streaming SSE responses, we'll need to modify this to yield tokens one at a time.

### Tokenizer API

**Source**: `src/mlx.zig/src/tokenizer.zig`

```zig
pub const Tokenizer = struct {
    pub fn init(allocator: std.mem.Allocator, path_json: []const u8) !Self
    pub fn encode(self: *Self, text: []const u8) ![]const u32
    pub fn encodeChat(self: *Self, chat_format: ?[]const u8, replacements: []const []const u8) ![]const u32
    pub fn decode(self: *Self, token_ids: []const u32) ![]const u8
};
```

### Complete Usage Example

**Source**: `src/mlx.zig/src/llm.zig` + `src/mlx.zig/src/qwen.zig`

```zig
const std = @import("std");
const mlx = @import("mlx.zig");
const Tokenizer = @import("tokenizer.zig").Tokenizer;

pub const QwenTransformer = mlx.Transformer(mlx.Model(QwenTransformerBlock, ModelConfig), ModelConfig);

pub fn generateCompletion(allocator: std.mem.Allocator, model_path: []const u8, prompt: []const u8) ![]const u8 {
    var tokenizer = try Tokenizer.init(allocator, model_path);
    defer tokenizer.deinit();
    
    const input_ids = try tokenizer.encode(prompt);
    defer allocator.free(input_ids);
    
    var transformer = try QwenTransformer.init(allocator, model_path);
    defer transformer.deinit();
    
    const max_tokens = 256;
    const output_ids = try transformer.generate(input_ids, max_tokens);
    defer allocator.free(output_ids);
    
    const output_text = try tokenizer.decode(output_ids);
    return output_text;
}
```

## Key Findings

### 1. Model Loading
- MLX.zig loads `config.json` for architecture parameters
- Weights loaded from `.safetensors` files via `loadModelSafetensors()`
- Dtype determined from config (`bfloat16` or `float16`)

### 2. Generation Loop
- Token-by-token generation with KV cache
- Auto-regressive: output token becomes next input
- EOS token detection to stop early

### 3. Streaming Requirement
The current `generate()` returns all tokens at once. For SSE streaming, we need:
- Modify loop to yield tokens via callback, OR
- Create iterator wrapper around generate()

### 4. Thread Safety
- Single model instance can be shared
- Need mutex around generate() calls for concurrent requests
- MLX streams are not thread-safe

### 5. Model Support
- Qwen, Llama, Phi architectures defined in MLX.zig
- Qwen2.5-Coder specifically supported
- Model-specific configs in respective files (qwen.zig, llama.zig, etc.)

## Confidence: HIGH

All API signatures verified from MLX.zig source code.
