//! test_models_gptoss.zig - GPT-OSS integration tests
//!
//! End-to-end tests for GPT-OSS inference pipeline:
//! - Basic generation
//! - Harmony format round-trip
//! - Tool execution flow
//! - Streaming generation
//! - Performance benchmark

const std = @import("std");
const gptoss_mlx = @import("gptoss_mlx.zig");
const harmony = @import("harmony/harmony.zig");
const harmony_template = @import("harmony/template.zig");
const harmony_parser = @import("harmony/parser.zig");
const backend_mod = @import("backends/backend.zig");
const mlx_gptoss_backend = @import("backends/mlx_gptoss_backend.zig");
const model_manager = @import("model/gptoss_manager.zig");

const GPTOSSTransformer = gptoss_mlx.GPTOSSTransformer;
const GPTOSSConfig = gptoss_mlx.GPTOSSConfig;
const GPTOSSTokenGenerator = gptoss_mlx.GPTOSSTokenGenerator;
const HarmonyTemplate = harmony_template.HarmonyTemplate;
const HarmonyParser = harmony_parser.HarmonyParser;
const OpenAIMessage = harmony.OpenAIMessage;
const MLXGPTOSSBackend = mlx_gptoss_backend.MLXGPTOSSBackend;
const GPTOSSModelManager = model_manager.GPTOSSModelManager;
const GenerationParams = backend_mod.GenerationParams;

// ── Infrastructure tests ──────────────────────────────────────────────────────

test "GPT-OSS config 20B has correct parameters" {
    const config = GPTOSSConfig.gptoss20b();
    try std.testing.expectEqual(@as(usize, 151936), config.vocab_size);
    try std.testing.expectEqual(@as(usize, 5120), config.hidden_size);
    try std.testing.expectEqual(@as(usize, 40), config.num_hidden_layers);
    try std.testing.expectEqual(@as(usize, 64), config.num_attention_heads);
    try std.testing.expectEqual(@as(usize, 8), config.num_key_value_heads);
    try std.testing.expectEqual(@as(usize, 32), config.num_experts);
    try std.testing.expectEqual(@as(usize, 4096), config.sliding_window);
    try std.testing.expectEqual(@as(usize, 131072), config.max_position_embeddings);
}

test "GPT-OSS config 120B has correct parameters" {
    const config = GPTOSSConfig.gptoss120b();
    try std.testing.expectEqual(@as(usize, 151936), config.vocab_size);
    try std.testing.expectEqual(@as(usize, 6656), config.hidden_size);
    try std.testing.expectEqual(@as(usize, 56), config.num_hidden_layers);
    try std.testing.expectEqual(@as(usize, 64), config.num_experts);
    try std.testing.expectEqual(@as(usize, 6), config.top_k);
}

// ── testGptossInference ──────────────────────────────────────────────────────

/// Test basic GPT-OSS inference with the stub transformer
pub fn testGptossInference(allocator: std.mem.Allocator) !void {
    // Initialize stub transformer (no real weights needed for structural test)
    const config = GPTOSSConfig.gptoss20b();
    var transformer = try GPTOSSTransformer.init(allocator, config, undefined);
    defer transformer.deinit();

    // Generate a short sequence
    const prompt = [_]u32{ 100256, 1, 2, 3 }; // BOS + some tokens
    const output = try transformer.generate(&prompt, 10, 0.7);
    defer allocator.free(output);

    try std.testing.expect(output.len > 0);
    try std.testing.expect(output.len <= 10);
}

test "GPT-OSS basic generation (stub transformer)" {
    const allocator = std.testing.allocator;
    try testGptossInference(allocator);
}

// ── Harmony format round-trip ────────────────────────────────────────────────

/// Test that messages can be formatted to Harmony and parsed back
pub fn testGptossHarmonyRoundTrip(allocator: std.mem.Allocator) !void {
    // Format messages to Harmony
    var tmpl = HarmonyTemplate.init(allocator, "harmony_gpt_oss", .medium, false, null);

    const messages = [_]OpenAIMessage{
        .{ .role = "user", .content = "Hello, GPT-OSS!" },
    };

    const formatted = try tmpl.formatHarmonyChat(&messages, null);
    defer allocator.free(formatted);

    // Verify it contains Harmony markers
    try std.testing.expect(std.mem.indexOf(u8, formatted, "<|startoftext|>") != null);
    try std.testing.expect(std.mem.indexOf(u8, formatted, "<|user|>") != null);
    try std.testing.expect(std.mem.indexOf(u8, formatted, "Hello, GPT-OSS!") != null);
    try std.testing.expect(std.mem.indexOf(u8, formatted, "<|assistant|>") != null);

    // Parse back
    var parser = HarmonyParser.init(allocator);
    var conversation = try parser.parse(formatted);
    defer conversation.deinit();

    // Should have at least one message
    try std.testing.expect(conversation.messages.items.len > 0);
}

test "GPT-OSS Harmony format round-trip" {
    const allocator = std.testing.allocator;
    try testGptossHarmonyRoundTrip(allocator);
}

// ── Tool calls ───────────────────────────────────────────────────────────────

/// Test tool call extraction from Harmony-formatted output
pub fn testGptossTools(allocator: std.mem.Allocator) !void {
    // Simulate model output containing a tool call
    const model_output =
        "<|startoftext|>\n" ++
        "<|recipient|>assistant<|/recipient|>\n" ++
        "<|assistant|>\n" ++
        "I need to search for that.\n" ++
        "<|recipient|>browser<|/recipient|>\n" ++
        "<|tool_call|>{\"action\": \"search\", \"query\": \"weather SF\"}<|/tool_call|>\n";

    var parser = HarmonyParser.init(allocator);
    const tool_calls = try parser.extractToolCalls(model_output);
    defer {
        for (tool_calls) |*tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
            if (tc.id) |id| allocator.free(id);
        }
        allocator.free(tool_calls);
    }

    // Should have extracted one tool call
    try std.testing.expectEqual(@as(usize, 1), tool_calls.len);
    try std.testing.expectEqualStrings("browser", tool_calls[0].name);
    try std.testing.expect(std.mem.indexOf(u8, tool_calls[0].arguments, "weather SF") != null);
}

test "GPT-OSS browser tool execution (parse tool call from Harmony)" {
    const allocator = std.testing.allocator;
    try testGptossTools(allocator);
}

// ── Streaming generation ──────────────────────────────────────────────────────

/// Test streaming generation via token iterator
pub fn testGptossStreaming(allocator: std.mem.Allocator) !void {
    const config = GPTOSSConfig.gptoss20b();
    var transformer = try GPTOSSTransformer.init(allocator, config, undefined);
    defer transformer.deinit();

    // Create a stub backend to test streaming via token iterator
    var bknd = try MLXGPTOSSBackend.init(allocator, "gpt-oss-20b", .{
        .enable_tools = false,
        .max_tokens = 5,
    });
    defer bknd.deinit();

    // Inject pre-loaded transformer to avoid weight loading
    bknd.transformer = transformer;
    bknd.loaded = true;

    const prompt_tokens = [_]u32{ 100256, 1 };
    var result = try bknd.generate(&prompt_tokens, .{ .max_tokens = 5, .temperature = 0.0 }, allocator);
    defer result.deinit(allocator);

    // Collect tokens via iterator
    var count: u32 = 0;
    while (try result.token_iterator.next(allocator)) |tok| {
        defer allocator.free(tok.text);
        count += 1;
    }

    try std.testing.expect(count > 0);
    try std.testing.expect(count <= 5);
}

test "GPT-OSS streaming generation via token iterator" {
    const allocator = std.testing.allocator;
    try testGptossStreaming(allocator);
}

// ── Performance benchmark ─────────────────────────────────────────────────────

/// Benchmark: generate N tokens and assert throughput > threshold.
/// In stub mode (no real weights) this just validates the loop overhead.
pub fn testGptossPerformanceBenchmark(allocator: std.mem.Allocator) !void {
    const config = GPTOSSConfig.gptoss20b();
    var transformer = try GPTOSSTransformer.init(allocator, config, undefined);
    defer transformer.deinit();

    const N: usize = 100;
    const prompt = [_]u32{100256};

    const start_ns = std.time.nanoTimestamp();
    const output = try transformer.generate(&prompt, N, 0.7);
    defer allocator.free(output);
    const elapsed_ns = std.time.nanoTimestamp() - start_ns;

    const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / 1e9;
    const tokens_per_sec = @as(f64, @floatFromInt(output.len)) / elapsed_s;

    // Stub transformer is CPU-only and trivial, so threshold is low here.
    // With real MLX weights on M-series, target is 30+ tokens/sec.
    // We assert > 0.001 tokens/sec to ensure the loop isn't deadlocking.
    try std.testing.expect(tokens_per_sec > 0.001);
}

test "GPT-OSS performance benchmark (stub transformer)" {
    const allocator = std.testing.allocator;
    try testGptossPerformanceBenchmark(allocator);
}

// ── Model manager tests ───────────────────────────────────────────────────────

test "GPT-OSS model manager lifecycle" {
    const allocator = std.testing.allocator;
    var mgr = GPTOSSModelManager.init(allocator, .{});
    defer mgr.deinit();

    // Register model
    try mgr.registerModel("gpt-oss-20b", "/tmp/gptoss-test");
    try std.testing.expect(mgr.registry.contains("gpt-oss-20b"));

    // Not yet loaded
    try std.testing.expect(!mgr.isLoaded("gpt-oss-20b"));
    try std.testing.expectEqual(@as(usize, 0), mgr.currentMemoryUsage());
}

test "GPT-OSS backend variant detection" {
    const allocator = std.testing.allocator;

    // 20B detection
    const bknd20 = try MLXGPTOSSBackend.init(allocator, "/models/gpt-oss-20b", .{
        .enable_tools = false,
    });
    var bknd20_mut = bknd20;
    defer bknd20_mut.deinit();
    try std.testing.expectEqual(mlx_gptoss_backend.ModelVariant.gptoss_20b, bknd20.model_variant);

    // 120B detection
    const bknd120 = try MLXGPTOSSBackend.init(allocator, "/models/gpt-oss-120b", .{
        .enable_tools = false,
    });
    var bknd120_mut = bknd120;
    defer bknd120_mut.deinit();
    try std.testing.expectEqual(mlx_gptoss_backend.ModelVariant.gptoss_120b, bknd120.model_variant);
}

test "Backend selectBackend routing" {
    const select = backend_mod.selectBackend;
    try std.testing.expectEqual(backend_mod.BackendType.mlx_gptoss, select("gpt-oss-20b"));
    try std.testing.expectEqual(backend_mod.BackendType.mlx_gptoss, select("gptoss-120b"));
    try std.testing.expectEqual(backend_mod.BackendType.llama_cpp, select("deepseek-coder-v2"));
    try std.testing.expectEqual(backend_mod.BackendType.llama_cpp, select("qwen2.5-coder"));
}
