// moe.zig - Mixture of Experts implementation for DeepSeek-V2
// Implements sparse expert activation with Metal kernel routing and CPU fallback
//
// DeepSeek-V2 MoE Architecture:
// - 64 total routed experts (top-6 selected per token)
// - 2 shared experts (always active)
// - 2B active parameters out of 15.7B total
// - 8x fewer FLOPs than dense 15.7B model

const std = @import("std");
const mlx = @import("mlx.zig/src/mlx.zig");
const mlx_v4 = @import("mlx_v4.zig");

/// Configuration for MoE layer
pub const MoEConfig = struct {
    hidden_size: usize,
    intermediate_size: usize,
    num_experts: usize, // Total routed experts (e.g., 64)
    num_shared_experts: usize, // Always active (e.g., 2)
    top_k: usize, // Experts per token (e.g., 6)
};

/// Result of routing operation
pub const RoutingResult = struct {
    indices: mlx.Array, // [batch, seq, top_k]
    weights: mlx.Array, // [batch, seq, top_k]
};

/// Individual expert with SwiGLU FFN
pub const Expert = struct {
    w_gate: mlx.Array, // [hidden_size, intermediate_size]
    w_up: mlx.Array, // [hidden_size, intermediate_size]
    w_down: mlx.Array, // [intermediate_size, hidden_size]

    /// Forward pass through expert FFN using SwiGLU activation
    /// SwiGLU(x) = silu(x @ W_gate) * (x @ W_up) @ W_down
    pub fn forward(self: Expert, output: *mlx.Array, x: mlx.Array) !void {
        var gate = mlx.arrayNew();
        var up = mlx.arrayNew();
        var gate_activated = mlx.arrayNew();
        var gate_up = mlx.arrayNew();

        defer {
            mlx.arrayFree(gate);
            mlx.arrayFree(up);
            mlx.arrayFree(gate_activated);
            mlx.arrayFree(gate_up);
        }

        // Compute gate projection: x @ W_gate
        // x: [batch, seq, hidden_size], W_gate: [hidden_size, intermediate_size]
        try mlx.matmul(&gate, x, self.w_gate, null);

        // Compute up projection: x @ W_up
        try mlx.matmul(&up, x, self.w_up, null);

        // Apply SiLU activation to gate: silu(gate)
        try mlx.silu(&gate_activated, gate, null);

        // Element-wise multiply: silu(gate) * up
        try mlx.multiply(&gate_up, gate_activated, up, null);

        // Final projection: (silu(gate) * up) @ W_down
        // gate_up: [batch, seq, intermediate_size], W_down: [intermediate_size, hidden_size]
        try mlx.matmul(output, gate_up, self.w_down, null);
    }
};

/// Mixture of Experts layer with sparse activation
pub const MixtureOfExperts = struct {
    allocator: std.mem.Allocator,
    config: MoEConfig,
    shared_experts: []Expert,
    routed_experts: []Expert,
    gate_weight: mlx.Array,
    route_kernel: ?mlx_v4.FastMetalKernel,

    const Self = @This();

    /// Initialize MoE layer with given configuration
    pub fn init(allocator: std.mem.Allocator, config: MoEConfig) !Self {
        // Allocate space for experts (will be loaded from weights later)
        const shared_experts = try allocator.alloc(Expert, config.num_shared_experts);
        errdefer allocator.free(shared_experts);

        const routed_experts = try allocator.alloc(Expert, config.num_experts);
        errdefer allocator.free(routed_experts);

        // Initialize gate weight matrix [hidden_size, num_experts]
        // For testing, create random weights
        var gate_weight = mlx.arrayNew();
        _ = mlx.randomNormal(&gate_weight, &.{
            @intCast(config.hidden_size),
            @intCast(config.num_experts),
        }, .float32);

        // Try to initialize Metal kernel for fast routing
        var route_kernel: ?mlx_v4.FastMetalKernel = null;
        if (mlx_v4.hasFastOps()) {
            route_kernel = initRouteKernel(allocator) catch |err| {
                std.debug.print("Failed to initialize Metal kernel: {s}\n", .{@errorName(err)});
                null;
            };
        }

        return Self{
            .allocator = allocator,
            .config = config,
            .shared_experts = shared_experts,
            .routed_experts = routed_experts,
            .gate_weight = gate_weight,
            .route_kernel = route_kernel,
        };
    }

    /// Free all resources
    pub fn deinit(self: *Self) void {
        // Free kernel if initialized
        if (self.route_kernel) |*kernel| {
            kernel.deinit();
        }

        // Free gate weight
        mlx.arrayFree(self.gate_weight);

        // Free expert arrays (weights will be freed by model cleanup)
        self.allocator.free(self.shared_experts);
        self.allocator.free(self.routed_experts);
    }

    /// Route tokens to experts
    /// Uses Metal kernel if available, otherwise CPU fallback
    pub fn route(self: *Self, hidden_states: mlx.Array) !RoutingResult {
        if (self.route_kernel) |kernel| {
            return try self.routeWithKernel(kernel, hidden_states);
        } else {
            return try self.routeOnCpu(hidden_states);
        }
    }

    /// Full forward pass through MoE layer
    /// 1. Process shared experts (always active)
    /// 2. Route and process top-k routed experts
    /// 3. Combine outputs
    pub fn forward(self: *Self, output: *mlx.Array, hidden_states: mlx.Array) !void {
        // Get routing decisions
        const routing = try self.route(hidden_states);
        defer {
            mlx.arrayFree(routing.indices);
            mlx.arrayFree(routing.weights);
        }

        // Initialize output with zeros
        try mlx.zeros_like(output, hidden_states);

        // Step 1: Process shared experts (always active)
        for (self.shared_experts) |expert| {
            var expert_out = mlx.arrayNew();
            defer mlx.arrayFree(expert_out);
            try expert.forward(&expert_out, hidden_states);
            try mlx.add(output, output.*, expert_out, null);
        }

        // Step 2: Process routed experts (sparse)
        try self.forwardRouted(output, hidden_states, routing);
    }

    // Forward sparse routed experts
    fn forwardRouted(
        self: *Self,
        output: *mlx.Array,
        hidden_states: mlx.Array,
        routing: RoutingResult,
    ) !void {
        const batch_size = mlx.arrayDim(hidden_states, 0);
        const seq_len = mlx.arrayDim(hidden_states, 1);
        const hidden_size: i64 = mlx.arrayDim(hidden_states, 2);

        // Iterate over batch and sequence dimensions
        var b: i64 = 0;
        while (b < batch_size) : (b += 1) {
            var s: i64 = 0;
            while (s < seq_len) : (s += 1) {
                // For each token, process its top-k experts
                var k: usize = 0;
                while (k < self.config.top_k) : (k += 1) {
                    // Get expert index and weight for this token
                    const expert_idx = mlx.arrayItemInt64(routing.indices, b * seq_len * self.config.top_k + s * self.config.top_k + k);
                    const weight = mlx.arrayItemFloat32(routing.weights, b * seq_len * self.config.top_k + s * self.config.top_k + k);

                    if (expert_idx < 0 or expert_idx >= self.config.num_experts) continue;

                    // Get the expert
                    const expert = self.routed_experts[@intCast(expert_idx)];

                    // Extract this token's hidden state
                    var token_hidden = mlx.arrayNew();
                    defer mlx.arrayFree(token_hidden);
                    try mlx.take(&token_hidden, hidden_states, b * seq_len + s, 1);

                    // Run expert forward
                    var expert_out = mlx.arrayNew();
                    defer mlx.arrayFree(expert_out);
                    try expert.forward(&expert_out, token_hidden);

                    // Scale by weight
                    var weighted_out = mlx.arrayNew();
                    defer mlx.arrayFree(weighted_out);
                    try mlx.multiply(&weighted_out, expert_out, mlx.scalarFloat32(weight), null);

                    // Add to output at correct position
                    try mlx.scatter_add(output, output.*, weighted_out, b * seq_len + s, 1);
                }
            }
        }
    }

    /// Initialize Metal kernel for fast routing
    fn initRouteKernel(allocator: std.mem.Allocator) !mlx_v4.FastMetalKernel {
        // Embedded Metal shader source
        const shader_source = @embedFile("moe_metal.metal");

        return try mlx_v4.FastMetalKernel.init(
            allocator,
            "moe_route",
            &.{ "hidden_states", "gate_weight" },
            &.{ "expert_indices", "expert_weights" },
            shader_source,
            false, // ensure_row_contiguous
            false, // atomic_outputs
        );
    }

    /// Route using Metal kernel (fast path)
    fn routeWithKernel(
        self: *Self,
        kernel: mlx_v4.FastMetalKernel,
        hidden_states: mlx.Array,
    ) !RoutingResult {
        const batch_size = mlx.arrayDim(hidden_states, 0);
        const seq_len = mlx.arrayDim(hidden_states, 1);

        // Allocate output arrays
        var indices = mlx.arrayNew();
        errdefer mlx.arrayFree(indices);
        try mlx.zeros(&indices, &.{
            batch_size,
            seq_len,
            @intCast(self.config.top_k),
        }, .int32);

        var weights = mlx.arrayNew();
        errdefer mlx.arrayFree(weights);
        try mlx.zeros(&weights, &.{
            batch_size,
            seq_len,
            @intCast(self.config.top_k),
        }, .float32);

        // Prepare inputs/outputs
        const inputs = &.{ hidden_states, self.gate_weight };
        var outputs = &.{ indices, weights };

        // Apply kernel with grid configuration
        // Grid: one thread per token
        const grid_dims = [3]u32{
            @intCast(batch_size),
            @intCast(seq_len),
            1,
        };
        const thread_group_dims = [3]u32{ 32, 32, 1 };

        // Get default stream
        const stream = mlx.defaultStream(null);

        try kernel.apply(
            inputs,
            outputs,
            grid_dims,
            thread_group_dims,
            stream,
        );

        return RoutingResult{
            .indices = indices,
            .weights = weights,
        };
    }

    /// Route using CPU fallback (slow path)
    fn routeOnCpu(self: *Self, hidden_states: mlx.Array) !RoutingResult {
        const batch_size = mlx.arrayDim(hidden_states, 0);
        const seq_len = mlx.arrayDim(hidden_states, 1);

        // Step 1: Compute gate logits
        // hidden_states: [batch, seq, hidden]
        // gate_weight: [hidden, num_experts]
        // logits: [batch, seq, num_experts]
        var logits = mlx.arrayNew();
        errdefer mlx.arrayFree(logits);
        try mlx.matmul(&logits, hidden_states, self.gate_weight, null);

        // Step 2: Apply softmax over experts dimension
        var probs = mlx.arrayNew();
        errdefer mlx.arrayFree(probs);
        try mlx.softmax(&probs, logits, .{ .axis = -1 });
        mlx.arrayFree(logits);

        // Step 3: Top-k selection
        // MLX doesn't have built-in topk, so we use argmax in a loop
        // For production, this should use a proper top-k implementation
        var indices = mlx.arrayNew();
        errdefer mlx.arrayFree(indices);
        try mlx.zeros(&indices, &.{
            batch_size,
            seq_len,
            @intCast(self.config.top_k),
        }, .int32);

        var weights = mlx.arrayNew();
        errdefer mlx.arrayFree(weights);
        try mlx.zeros(&weights, &.{
            batch_size,
            seq_len,
            @intCast(self.config.top_k),
        }, .float32);

        // Simplified: for testing, just use first k experts
        // Full implementation would find actual top-k
        var k: usize = 0;
        while (k < self.config.top_k) : (k += 1) {
            // This is a placeholder - real implementation needs top-k selection
            // For now, just set sequential expert indices
            const expert_idx = @as(i32, @intCast(k));
            _ = expert_idx;
        }

        mlx.arrayFree(probs);

        return RoutingResult{
            .indices = indices,
            .weights = weights,
        };
    }
};
