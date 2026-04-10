// gemma4_test.zig - Integration tests for Gemma 4 E4B handler
//
// Run with: zig build test -- src/api/gemma4_test.zig
// Or: zig test src/api/gemma4_test.zig

const std = @import("std");
const chat_gemma4 = @import("chat_gemma4.zig");

// Test chat prompt formatting
test "Gemma 4 prompt formatting - single user message" {
    const allocator = std.testing.allocator;

    // Single user message
    const messages = &[_]chat_gemma4.Message{
        .{
            .role = "user",
            .content = "Hello",
        },
    };

    const prompt = try chat_gemma4.buildGemma4Prompt(allocator, messages, true, false);
    defer allocator.free(prompt);

    // Expected: <|turn|>user\nHello<turn|>\n<|turn|>model\n<|channel>thought\n<channel|>
    try std.testing.expect(std.mem.indexOf(u8, prompt, "<|turn|>user") != null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "Hello") != null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "<|turn|>model") != null);
}

// Test system + user prompt
test "Gemma 4 prompt formatting - system and user" {
    const allocator = std.testing.allocator;

    const messages = &[_]chat_gemma4.Message{
        .{
            .role = "system",
            .content = "You are a coding assistant.",
        },
        .{
            .role = "user",
            .content = "Write Python code",
        },
    };

    const prompt = try chat_gemma4.buildGemma4Prompt(allocator, messages, true, false);
    defer allocator.free(prompt);

    try std.testing.expect(std.mem.indexOf(u8, prompt, "system") != null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "coding assistant") != null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "Write Python code") != null);
}

// Test assistant role mapping
test "Gemma 4 prompt formatting - assistant role maps to model" {
    const allocator = std.testing.allocator;

    const messages = &[_]chat_gemma4.Message{
        .{
            .role = "user",
            .content = "Hi",
        },
        .{
            .role = "assistant",
            .content = "Hello!",
        },
    };

    const prompt = try chat_gemma4.buildGemma4Prompt(allocator, messages, false, false);
    defer allocator.free(prompt);

    // OpenAI "assistant" should be mapped to Gemma "model"
    try std.testing.expect(std.mem.indexOf(u8, prompt, "<|turn|>model") != null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "Hello!") != null);
}

// Test no-think suffix
test "Gemma 4 prompt formatting - no-think mode" {
    const allocator = std.testing.allocator;

    const messages = &[_]chat_gemma4.Message{
        .{ .role = "user", .content = "Test" },
    };

    // enable_thinking = false should add no-think suffix
    const prompt = try chat_gemma4.buildGemma4Prompt(allocator, messages, true, false);
    defer allocator.free(prompt);

    try std.testing.expect(std.mem.indexOf(u8, prompt, "<|channel>thought") != null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "<channel|>") != null);
}

// Test with thinking enabled
test "Gemma 4 prompt formatting - thinking mode" {
    const allocator = std.testing.allocator;

    const messages = &[_]chat_gemma4.Message{
        .{ .role = "user", .content = "Test" },
    };

    // enable_thinking = true should NOT add no-think suffix
    const prompt = try chat_gemma4.buildGemma4Prompt(allocator, messages, true, true);
    defer allocator.free(prompt);

    try std.testing.expect(std.mem.indexOf(u8, prompt, "<|channel>thought") == null);
}

// Test model detection
test "isGemma4Model detection" {
    try std.testing.expect(chat_gemma4.isGemma4Model("gemma4-e4b"));
    try std.testing.expect(chat_gemma4.isGemma4Model("gemma-4-e4b-it"));
    try std.testing.expect(chat_gemma4.isGemma4Model("gemma4-e4b-it-UD-MLX-4bit"));

    // Should NOT match
    try std.testing.expect(!chat_gemma4.isGemma4Model("qwen2.5-coder"));
    try std.testing.expect(!chat_gemma4.isGemma4Model("gpt-oss-20b"));
    try std.testing.expect(!chat_gemma4.isGemma4Model("deepseek-coder"));
}

// Integration test - requires GGUF model
// Skipped if model not available
test "Gemma 4 handler integration" {
    const model_path = "./models/gemma4-e4b-gguf";

    // Skip if model not available
    std.fs.cwd().access(model_path, .{}) catch {
        std.debug.print("Skipping integration test - model not found at {s}\n", .{model_path});
        return;
    };

    _ = chat_gemma4; // Reference to avoid unused import

    // Note: Full integration test would require server setup
    // Run manually: ./zig-out/bin/zlx --model gemma4-e4b

    std.debug.print("Gemma 4 handler test placeholder\n", .{});
}

// Test streaming response format
test "Gemma 4 streaming format" {
    const allocator = std.testing.allocator;

    // Build a test prompt
    const messages = &[_]chat_gemma4.Message{
        .{ .role = "user", .content = "Say hi" },
    };

    const prompt = try chat_gemma4.buildGemma4Prompt(allocator, messages, true, false);
    defer allocator.free(prompt);

    // Verify the prompt structure for streaming
    // Streaming uses the same prompt format as non-streaming
    try std.testing.expect(prompt.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "<|turn|>") != null);
}

// Performance benchmark placeholder
// Run this with: zig test -O ReleaseFast src/api/gemma4_test.zig
test "Gemma 4 generation speed benchmark" {
    const model_path = "./models/gemma4-e4b-gguf";

    // Skip if model not available
    std.fs.cwd().access(model_path, .{}) catch {
        std.debug.print("Skipping benchmark - model not found\n", .{});
        return;
    };

    // This would measure:
    // - Time to first token (TTFT)
    // - Tokens per second
    // - Memory usage

    std.debug.print("Benchmark placeholder - run with actual server for real metrics\n", .{});
}

// Export test runner
pub fn main() !void {
    std.debug.print("\n=== Gemma 4 Test Suite ===\n\n", .{});

    // Run all tests manually (for debugging)
    std.debug.print("Note: Run with 'zig build test' for proper test execution\n", .{});

    // Check model availability
    const model_path = "./models/gemma4-e4b-gguf";
    if (std.fs.cwd().access(model_path, .{})) {
        std.debug.print("✓ Model found at {s}\n", .{model_path});
    } else |_| {
        std.debug.print("✗ Model not found at {s}\n", .{model_path});
        std.debug.print("  Download from: https://huggingface.co/bartowski/gemma-4-e4b-it-GGUF\n", .{});
    }
}
