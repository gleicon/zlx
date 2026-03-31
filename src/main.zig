// src/main.zig
// Phase 2: Inference Core with CLI harness
// CLI harness loads model and streams tokens via Metal GPU

const std = @import("std");
const inference = @import("inference/mod.zig");
const qwen = @import("mlx.zig/src/qwen.zig");
const mlx = @import("mlx.zig/src/mlx.zig");

const USAGE =
    "Usage: zlx [OPTIONS] [PROMPT]\n" ++
    "\n" ++
    "Options:\n" ++
    "  --model <NAME>          Model name or path (default: qwen2.5-coder-1.5b)\n" ++
    "  --port <PORT>           Server port (default: 8080)\n" ++
    "  --max-kv-size <SIZE>    Maximum KV cache size (default: 2048)\n" ++
    "  --max-tokens <N>        Maximum tokens to generate (default: 256)\n" ++
    "  --temperature <T>       Sampling temperature (default: 0.7)\n" ++
    "  --help                  Show this help message\n" ++
    "\n" ++
    "Examples:\n" ++
    "  zlx --model qwen2.5-coder-1.5b \"Write a hello world in Python\"\n" ++
    "  zlx --model ./models/my-model --port 9000\n" ++
    "\n";

const Config = struct {
    model_name: []const u8 = "qwen2.5-coder-1.5b",
    model_path: ?[]const u8 = null,
    port: u16 = 8080,
    max_kv_size: usize = 2048,
    max_tokens: usize = 256,
    temperature: f32 = 0.7,
    prompt: ?[]const u8 = null,
};

fn printUsage() void {
    std.debug.print("{s}", .{USAGE});
}

fn parseArgs(allocator: std.mem.Allocator) !Config {
    var config = Config{};
    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    // Skip program name
    _ = args.next();

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--help")) {
            printUsage();
            std.process.exit(0);
        } else if (std.mem.eql(u8, arg, "--model")) {
            const value = args.next() orelse {
                std.log.err("Expected value after --model", .{});
                return error.MissingArgument;
            };
            config.model_name = value;
        } else if (std.mem.eql(u8, arg, "--port")) {
            const value = args.next() orelse {
                std.log.err("Expected value after --port", .{});
                return error.MissingArgument;
            };
            config.port = try std.fmt.parseInt(u16, value, 10);
        } else if (std.mem.eql(u8, arg, "--max-kv-size")) {
            const value = args.next() orelse {
                std.log.err("Expected value after --max-kv-size", .{});
                return error.MissingArgument;
            };
            config.max_kv_size = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, arg, "--max-tokens")) {
            const value = args.next() orelse {
                std.log.err("Expected value after --max-tokens", .{});
                return error.MissingArgument;
            };
            config.max_tokens = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, arg, "--temperature")) {
            const value = args.next() orelse {
                std.log.err("Expected value after --temperature", .{});
                return error.MissingArgument;
            };
            config.temperature = try std.fmt.parseFloat(f32, value);
        } else if (!std.mem.startsWith(u8, arg, "--")) {
            // This is the prompt (or first word of it)
            if (config.prompt == null) {
                config.prompt = arg;
            }
        }
    }

    // Build model path from name if not explicitly set
    if (config.model_path == null) {
        // Try to find model in ./models/{name} or use as-is if it contains /
        if (std.mem.indexOf(u8, config.model_name, "/")) |_| {
            config.model_path = config.model_name;
        } else {
            config.model_path = try std.fmt.allocPrint(allocator, "./models/{s}", .{config.model_name});
        }
    }

    return config;
}

fn resolveModelPath(allocator: std.mem.Allocator, name_or_path: []const u8) ![]const u8 {
    // If it contains a slash, treat as explicit path
    if (std.mem.indexOf(u8, name_or_path, "/")) |_| {
        return allocator.dupe(u8, name_or_path);
    }

    // Otherwise, look in ./models/
    return try std.fmt.allocPrint(allocator, "./models/{s}", .{name_or_path});
}

/// Run interactive generation mode
fn runInteractiveMode(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    config: Config,
) !void {
    std.debug.print("\\n=== zlx Interactive Mode ===\\n", .{});
    std.debug.print("Model: {s}\\n", .{model_path});
    std.debug.print("Type 'quit' to exit\\n\\n", .{});

    // Initialize transformer
    std.debug.print("Loading model...\\n", .{});
    var transformer = try qwen.Transformer.init(allocator, model_path);
    defer transformer.deinit();
    std.debug.print("Model loaded successfully!\\n\\n", .{});

    // Initialize tokenizer
    var tokenizer = try inference.Tokenizer.init(allocator, model_path);
    defer tokenizer.deinit();

    // Interactive loop - using debug output for simplicity
    // In production, use std.fs.File with buffers
    var stdin_buf: [4096]u8 = undefined;

    while (true) {
        std.debug.print("> ", .{});

        const stdin_file = std.fs.File.stdin();
        const input_len = try stdin_file.read(&stdin_buf);
        if (input_len == 0) break;

        const input = std.mem.trimRight(u8, stdin_buf[0..input_len], "\r\n");
        if (input.len == 0) continue;

        if (std.mem.eql(u8, input, "quit") or std.mem.eql(u8, input, "exit")) {
            break;
        }

        // Encode prompt
        const input_tokens = try tokenizer.encode(input);
        defer allocator.free(input_tokens);

        std.debug.print("\\n[Generating...]\\n", .{});

        // Generate with streaming
        const start_time = std.time.milliTimestamp();
        var first_token_time: ?i64 = null;

        const options = inference.GenerationOptions{
            .max_tokens = config.max_tokens,
            .temperature = config.temperature,
            .stop_on_eos = true,
        };

        var state = try inference.GenerationState.init(
            allocator,
            &transformer,
            input_tokens,
            transformer.eos_token_ids,
            options,
        );
        defer state.deinit();

        var generated_count: usize = 0;

        // Stream tokens one at a time
        while (try state.next()) |token| {
            if (first_token_time == null) {
                first_token_time = std.time.milliTimestamp();
            }

            // Decode single token and print immediately
            const token_slice = [_]u32{token};
            const text = try tokenizer.decode(&token_slice);
            defer allocator.free(text);

            std.debug.print("{s}", .{text});

            generated_count += 1;
        }

        const end_time = std.time.milliTimestamp();

        std.debug.print("\n\n", .{});

        // Print stats
        if (first_token_time) |ftt| {
            const prompt_time_ms = @as(f64, @floatFromInt(ftt - start_time));
            const prompt_tps = @as(f64, @floatFromInt(input_tokens.len)) / (prompt_time_ms / 1000.0);

            const gen_time_ms = @as(f64, @floatFromInt(end_time - ftt));
            const gen_tps = @as(f64, @floatFromInt(generated_count)) / (gen_time_ms / 1000.0);

            std.debug.print("[Stats: {d} tokens, prompt: {d:.1} tps, gen: {d:.1} tps]\\n", .{
                generated_count,
                prompt_tps,
                gen_tps,
            });
        }
    }

    std.debug.print("\nGoodbye!\n", .{});
}

/// Run single-shot generation from command line prompt
fn runSingleShot(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    prompt: []const u8,
    config: Config,
) !void {
    std.debug.print("Loading model from: {s}\\n", .{model_path});

    // Initialize transformer
    var transformer = try qwen.Transformer.init(allocator, model_path);
    defer transformer.deinit();

    // Initialize tokenizer
    var tokenizer = try inference.Tokenizer.init(allocator, model_path);
    defer tokenizer.deinit();

    // Encode prompt
    const input_tokens = try tokenizer.encode(prompt);
    defer allocator.free(input_tokens);

    std.debug.print("\\nPrompt: {s}\\n", .{prompt});
    std.debug.print("Input tokens: {d}\\n\\n", .{input_tokens.len});
    std.debug.print("[Generating...]\\n", .{});

    // Generate with streaming
    const start_time = std.time.milliTimestamp();

    const options = inference.GenerationOptions{
        .max_tokens = config.max_tokens,
        .temperature = config.temperature,
        .stop_on_eos = true,
    };

    var state = try inference.GenerationState.init(
        allocator,
        &transformer,
        input_tokens,
        transformer.eos_token_ids,
        options,
    );
    defer state.deinit();

    var generated_count: usize = 0;

    // Stream tokens one at a time
    while (try state.next()) |token| {
        // Decode single token and print immediately
        const token_slice = [_]u32{token};
        const text = try tokenizer.decode(&token_slice);
        defer allocator.free(text);

        std.debug.print("{s}", .{text});

        generated_count += 1;
    }

    const end_time = std.time.milliTimestamp();
    const total_time_ms = @as(f64, @floatFromInt(end_time - start_time));
    const tps = @as(f64, @floatFromInt(generated_count)) / (total_time_ms / 1000.0);

    std.debug.print("\n", .{});
    std.debug.print("[Generated {d} tokens in {d:.1}ms ({d:.1} tps)]\\n", .{
        generated_count,
        total_time_ms,
        tps,
    });
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const config = parseArgs(allocator) catch |err| {
        if (err == error.MissingArgument) {
            printUsage();
            std.process.exit(1);
        }
        return err;
    };

    // Ensure model_path is owned by allocator
    const model_path = try resolveModelPath(allocator, config.model_name);
    defer allocator.free(model_path);

    // Validate model path
    std.fs.cwd().access(model_path, .{}) catch {
        std.log.err("Model not found at: {s}", .{model_path});
        std.log.err("Make sure the model is downloaded to ./models/", .{});
        std.process.exit(1);
    };

    // Run in appropriate mode
    if (config.prompt) |prompt| {
        try runSingleShot(allocator, model_path, prompt, config);
    } else {
        try runInteractiveMode(allocator, model_path, config);
    }
}
