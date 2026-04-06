//! llama_cpp.zig - llama.cpp backend implementation
//!
//! Implements the Backend interface using llama.cpp for DeepSeek and GPT-OSS models.

const std = @import("std");
const backend = @import("backend.zig");

// Import C bindings (when llama.cpp is available)
const llama_c = @import("llama_c.zig");

/// llama.cpp backend state
pub const LlamaBackend = struct {
    /// llama.cpp model handle
    model: *llama_c.llama_model,
    /// llama.cpp context handle
    ctx: *llama_c.llama_context,
    /// llama.cpp vocab handle (owned by model, do not free separately)
    vocab: *const llama_c.llama_vocab,
    /// Memory allocator
    allocator: std.mem.Allocator,
    /// Vocabulary size
    vocab_size: u32,
    /// EOS token ID
    eos_token: llama_c.llama_token,
    /// BOS token ID
    bos_token: llama_c.llama_token,
    /// Random seed
    seed: u64,

    /// Initialize llama.cpp backend with GGUF model
    pub fn init(allocator: std.mem.Allocator, model_path: []const u8) !LlamaBackend {
        std.log.info("Loading llama.cpp model from: {s}", .{model_path});

        // 1. Set up model parameters - offload all layers to GPU
        var model_params = llama_c.llama_model_default_params();
        model_params.n_gpu_layers = 1000; // All layers on Metal
        model_params.main_gpu = 0;
        model_params.split_mode = 1; // LLAMA_SPLIT_MODE_LAYER

        // 2. Load model from GGUF file
        const model = llama_c.llama_model_load_from_file(
            model_path.ptr,
            model_params,
        ) orelse {
            std.log.err("Failed to load model from: {s}", .{model_path});
            return error.ModelLoadFailed;
        };

        // 3. Set up context parameters
        var ctx_params = llama_c.llama_context_default_params();
        ctx_params.n_ctx = 8192; // Default 8K context
        ctx_params.n_batch = 512;
        ctx_params.n_threads = @intCast(std.Thread.getCpuCount() catch 4);
        ctx_params.n_threads_batch = ctx_params.n_threads;
        // Note: seed was removed from llama_context_params in newer llama.cpp; use llama_sampler_init_dist(seed) instead

        // 4. Create context
        const ctx = llama_c.llama_new_context_with_model(model, ctx_params) orelse {
            std.log.err("Failed to create llama.cpp context", .{});
            llama_c.llama_model_free(model);
            return error.ContextCreationFailed;
        };

        // 5. Get metadata via vocab API (llama_n_vocab/llama_token_eos/llama_token_bos now take llama_vocab*)
        const vocab = llama_c.llama_model_get_vocab(model);
        const vocab_size = @as(u32, @intCast(llama_c.llama_vocab_n_tokens(vocab)));
        const eos_token = llama_c.llama_vocab_eos(vocab);
        const bos_token = llama_c.llama_vocab_bos(vocab);

        std.log.info("llama.cpp model loaded: vocab={d}, eos={d}, bos={d}", .{
            vocab_size,
            eos_token,
            bos_token,
        });

        return LlamaBackend{
            .model = model,
            .ctx = ctx,
            .vocab = vocab.?,
            .allocator = allocator,
            .vocab_size = vocab_size,
            .eos_token = eos_token,
            .bos_token = bos_token,
            .seed = 42, // default seed (llama_context_params.seed removed in newer llama.cpp)
        };
    }

    /// Free llama.cpp resources
    pub fn deinit(self: *LlamaBackend) void {
        std.log.info("Freeing llama.cpp resources", .{});
        llama_c.llama_free(self.ctx);
        llama_c.llama_model_free(self.model);
    }

    /// Tokenize text into token IDs
    pub fn tokenize(self: *LlamaBackend, text: []const u8) ![]llama_c.llama_token {
        // Allocate buffer for tokens (text.len is upper bound)
        const max_tokens = text.len + 8; // Extra for BOS/EOS
        var tokens = try self.allocator.alloc(llama_c.llama_token, max_tokens);
        errdefer self.allocator.free(tokens);

        // Tokenize with BOS (llama_tokenize now takes llama_vocab* not llama_model*)
        const n_tokens = llama_c.llama_tokenize(
            self.vocab,
            text.ptr,
            @intCast(text.len),
            tokens.ptr,
            @intCast(max_tokens),
            true, // add_special (BOS)
            false, // parse_special
        );

        if (n_tokens < 0) {
            return error.TokenizationFailed;
        }

        // Resize to actual token count
        const actual_count = @as(usize, @intCast(n_tokens));
        if (actual_count < max_tokens) {
            // Realloc not strictly necessary but good practice
            tokens = try self.allocator.realloc(tokens, actual_count);
        }

        return tokens;
    }

    /// Decode tokens and generate next token
    pub fn decode(self: *LlamaBackend, tokens: []const llama_c.llama_token) !void {
        if (tokens.len == 0) return;

        // Create batch
        var batch = llama_c.llama_batch_init(@intCast(tokens.len), 0, 1);
        defer llama_c.llama_batch_free(batch);

        // Fill batch
        for (tokens, 0..) |token, i| {
            batch.token[i] = token;
            batch.pos[i] = @intCast(i);
            // seq_id and other fields use defaults
        }
        batch.n_tokens = @intCast(tokens.len);

        // Decode
        const result = llama_c.llama_decode(self.ctx, batch);
        if (result != 0) {
            return error.DecodeFailed;
        }
    }

    /// Sample next token using configured sampler
    /// Uses llama_sampler_sample (new API; llama_sample_token was removed)
    pub fn sample(
        self: *LlamaBackend,
        sampler: *llama_c.llama_sampler,
    ) llama_c.llama_token {
        return llama_c.llama_sampler_sample(sampler, self.ctx, -1);
    }

    /// Get vocabulary size
    pub fn vocabSize(self: LlamaBackend) u32 {
        return self.vocab_size;
    }

    /// Get EOS token
    pub fn eosToken(self: LlamaBackend) llama_c.llama_token {
        return self.eos_token;
    }

    /// Get BOS token
    pub fn bosToken(self: LlamaBackend) llama_c.llama_token {
        return self.bos_token;
    }
};

/// Create sampler chain for generation parameters
pub fn createSampler(
    allocator: std.mem.Allocator,
    params: backend.GenerationParams,
) !*llama_c.llama_sampler {
    _ = allocator;

    // Initialize sampler chain
    const chain_params = llama_c.llama_sampler_chain_default_params();
    const chain = llama_c.llama_sampler_chain_init(chain_params);

    // Add samplers in order: top_k -> top_p -> temperature -> greedy
    if (params.top_k > 0) {
        llama_c.llama_sampler_chain_add(
            chain,
            llama_c.llama_sampler_init_top_k(@intCast(params.top_k)),
        );
    }

    if (params.top_p < 1.0 and params.top_p > 0.0) {
        llama_c.llama_sampler_chain_add(
            chain,
            llama_c.llama_sampler_init_top_p(params.top_p, 1),
        );
    }

    if (params.temperature > 0.0) {
        llama_c.llama_sampler_chain_add(
            chain,
            llama_c.llama_sampler_init_temp(params.temperature),
        );
    } else {
        // Greedy sampling when temperature is 0
        llama_c.llama_sampler_chain_add(
            chain,
            llama_c.llama_sampler_init_greedy(),
        );
    }

    return chain;
}

/// Free sampler
pub fn freeSampler(sampler: *llama_c.llama_sampler) void {
    llama_c.llama_sampler_free(sampler);
}

/// Tokenize using llama.cpp tokenizer
pub fn llamaTokenize(ptr: *anyopaque, text: []const u8, allocator: std.mem.Allocator) anyerror![]u32 {
    const backend_ptr = @as(*LlamaBackend, @ptrCast(@alignCast(ptr)));

    const llama_tokens = try backend_ptr.tokenize(text);
    defer backend_ptr.allocator.free(llama_tokens);

    // Convert llama_token (i32) to u32
    var u32_tokens = try allocator.alloc(u32, llama_tokens.len);
    for (llama_tokens, 0..) |t, i| {
        u32_tokens[i] = @intCast(t);
    }

    return u32_tokens;
}

/// Generate using llama.cpp
pub fn llamaGenerate(
    ptr: *anyopaque,
    tokens: []const u32,
    params: backend.GenerationParams,
    allocator: std.mem.Allocator,
) anyerror!backend.GenerationResult {
    const backend_ptr = @as(*LlamaBackend, @ptrCast(@alignCast(ptr)));

    // Convert u32 tokens to llama_token
    var llama_tokens = try allocator.alloc(llama_c.llama_token, tokens.len);
    defer allocator.free(llama_tokens);
    for (tokens, 0..) |t, i| {
        llama_tokens[i] = @intCast(t);
    }

    // Create sampler
    const sampler = try createSampler(allocator, params);
    defer freeSampler(sampler);

    // Initial decode of prompt tokens
    try backend_ptr.decode(llama_tokens);

    // Create token iterator for streaming
    const IteratorContext = struct {
        backend: *LlamaBackend,
        sampler: *llama_c.llama_sampler,
        max_tokens: u32,
        generated: u32,
        eos_hit: bool,
    };

    const ctx = try allocator.create(IteratorContext);
    ctx.* = .{
        .backend = backend_ptr,
        .sampler = sampler,
        .max_tokens = params.max_tokens,
        .generated = 0,
        .eos_hit = false,
    };

    const iterator = backend.TokenIterator{
        .next_fn = struct {
            fn next(it_ctx: *anyopaque, it_allocator: std.mem.Allocator) anyerror!?backend.TokenResult {
                const it = @as(*IteratorContext, @ptrCast(@alignCast(it_ctx)));

                if (it.eos_hit or it.generated >= it.max_tokens) {
                    return null;
                }

                // Sample next token
                const token = it.backend.sample(it.sampler);

                // Check for EOS
                if (token == it.backend.eos_token) {
                    it.eos_hit = true;
                    return null;
                }

                // Get token text (llama_token_get_text now takes llama_vocab* not llama_model*)
                const token_text = llama_c.llama_token_get_text(it.backend.vocab, token);
                const text_copy = try it_allocator.dupe(u8, std.mem.sliceTo(token_text, 0));

                // Decode for next iteration
                var single_token = [_]llama_c.llama_token{token};
                try it.backend.decode(&single_token);

                it.generated += 1;

                return backend.TokenResult{
                    .token = @intCast(token),
                    .text = text_copy,
                    .logprob = 0.0, // TODO: Get actual logprob
                    .finish_reason = if (it.generated >= it.max_tokens) .length else null,
                };
            }
        }.next,
        .ctx = ctx,
        .deinit_fn = struct {
            fn deinit(it_ctx: *anyopaque, it_allocator: std.mem.Allocator) void {
                const it = @as(*IteratorContext, @ptrCast(@alignCast(it_ctx)));
                it_allocator.destroy(it);
            }
        }.deinit,
    };

    return backend.GenerationResult{
        .token_iterator = iterator,
        .tokens_generated = 0,
        .finish_reason = .length,
    };
}

/// Deinit llama.cpp backend
pub fn llamaDeinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
    const backend_ptr = @as(*LlamaBackend, @ptrCast(@alignCast(ptr)));
    backend_ptr.deinit();
    allocator.destroy(backend_ptr);
}

/// Get vocabulary size
pub fn llamaGetVocabSize(ptr: *anyopaque) u32 {
    const backend_ptr = @as(*LlamaBackend, @ptrCast(@alignCast(ptr)));
    return backend_ptr.vocabSize();
}

/// Get EOS token
pub fn llamaEosToken(ptr: *anyopaque) u32 {
    const backend_ptr = @as(*LlamaBackend, @ptrCast(@alignCast(ptr)));
    return @intCast(backend_ptr.eosToken());
}

/// Get BOS token
pub fn llamaBosToken(ptr: *anyopaque) u32 {
    const backend_ptr = @as(*LlamaBackend, @ptrCast(@alignCast(ptr)));
    return @intCast(backend_ptr.bosToken());
}

/// Get KV cache handle
pub fn llamaGetKvCache(ptr: *anyopaque) ?backend.KvCacheHandle {
    const backend_ptr = @as(*LlamaBackend, @ptrCast(@alignCast(ptr)));
    // llama.cpp stores KV cache in the context
    // Return context pointer as opaque handle
    return backend.KvCacheHandle{
        .ptr = @ptrCast(backend_ptr.ctx),
        .backend_type = .llama_cpp,
    };
}

/// Apply compression to llama.cpp KV cache
pub fn llamaApplyCompression(ptr: *anyopaque, params: backend.CompressionParams) anyerror!void {
    const backend_ptr = @as(*LlamaBackend, @ptrCast(@alignCast(ptr)));
    _ = backend_ptr;

    std.log.info("llama.cpp TurboQuant compression: bits={d}, adaptive_layers={d}", .{
        params.bits,
        params.adaptive_layers,
    });

    // TODO: Implement TurboQuant compression for llama.cpp KV cache
    // llama.cpp has built-in quantization, but TurboQuant provides better compression
    // This would involve accessing raw KV tensors and applying rotation + quantization
}
