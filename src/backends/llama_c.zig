//! llama_c.zig - C API bindings for llama.cpp
//!
//! Provides @cImport bindings to llama.cpp C API for use in Zig code.

const std = @import("std");

// C API import for llama.cpp
// Note: This will only compile when llama.cpp headers are available
pub const c = @cImport({
    @cDefine("LLAMA_USE_METAL", "1");
    @cInclude("llama.h");
});

// Re-export commonly used types for convenience
pub const llama_model = c.llama_model;
pub const llama_context = c.llama_context;
pub const llama_token = c.llama_token;
pub const llama_model_params = c.llama_model_params;
pub const llama_context_params = c.llama_context_params;
pub const llama_batch = c.llama_batch;
pub const llama_sampler = c.llama_sampler;
pub const llama_vocab = c.llama_vocab;

// Model loading functions
pub const llama_model_load_from_file = c.llama_model_load_from_file;
pub const llama_model_free = c.llama_model_free;
pub const llama_model_default_params = c.llama_model_default_params;
pub const llama_model_n_params = c.llama_model_n_params;
pub const llama_model_size = c.llama_model_size;

// Context functions
pub const llama_new_context_with_model = c.llama_new_context_with_model;
pub const llama_free = c.llama_free;
pub const llama_context_default_params = c.llama_context_default_params;
pub const llama_n_ctx = c.llama_n_ctx;

// Tokenization functions
pub const llama_tokenize = c.llama_tokenize;
pub const llama_token_get_text = c.llama_token_get_text;
pub const llama_token_eos = c.llama_token_eos;
pub const llama_token_bos = c.llama_token_bos;
pub const llama_token_nl = c.llama_token_nl;
pub const llama_n_vocab = c.llama_n_vocab;

// KV cache functions
pub const llama_get_kv_cache_token_count = c.llama_get_kv_cache_token_count;
pub const llama_kv_cache_update = c.llama_kv_cache_update;
pub const llama_kv_cache_clear = c.llama_kv_cache_clear;

// Decode and sampling functions
pub const llama_decode = c.llama_decode;
pub const llama_get_logits = c.llama_get_logits;
pub const llama_sample_token_greedy = c.llama_sample_token_greedy;
pub const llama_sample_token = c.llama_sample_token;

// Sampler chain functions
pub const llama_sampler_chain_init = c.llama_sampler_chain_init;
pub const llama_sampler_chain_add = c.llama_sampler_chain_add;
pub const llama_sampler_chain_get = c.llama_sampler_chain_get;
pub const llama_sampler_free = c.llama_sampler_free;
pub const llama_sampler_chain_default_params = c.llama_sampler_chain_default_params;

// Individual samplers
pub const llama_sampler_init_greedy = c.llama_sampler_init_greedy;
pub const llama_sampler_init_temp = c.llama_sampler_init_temp;
pub const llama_sampler_init_top_p = c.llama_sampler_init_top_p;
pub const llama_sampler_init_top_k = c.llama_sampler_init_top_k;
pub const llama_sampler_init_min_p = c.llama_sampler_init_min_p;

// Batch functions
pub const llama_batch_init = c.llama_batch_init;
pub const llama_batch_free = c.llama_batch_free;

// Performance timing
pub const llama_time_us = c.llama_time_us;
pub const llama_perf_context = c.llama_perf_context;

/// Helper to check if llama.cpp is available at compile time
pub const LLAMA_AVAILABLE = true;

/// Log callback type for llama.cpp messages
pub const llama_log_callback = c.llama_log_callback;
pub const llama_log_set = c.llama_log_set;

/// Backend (CPU/GPU) information
pub const llama_numa_init = c.llama_numa_init;
