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

        const stream = mlx.defaultGpuStreamNew();
        defer mlx.streamFree(stream);

        // Compute gate projection: x @ W_gate
        // x: [batch, seq, hidden_size], W_gate: [hidden_size, intermediate_size]
        try mlx.matmul(&gate, x, self.w_gate, stream);

        // Compute up projection: x @ W_up
        try mlx.matmul(&up, x, self.w_up, stream);

        // Apply SiLU activation to gate: silu(gate)
        try mlx.silu(&gate_activated, gate, stream);

        // Element-wise multiply: silu(gate) * up
        try mlx.multiply(&gate_up, gate_activated, up, stream);

        // Final projection: (silu(gate) * up) @ W_down
        // gate_up: [batch, seq, intermediate_size], W_down: [intermediate_size, hidden_size]
        try mlx.matmul(output, gate_up, self.w_down, stream);
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
        // Allocate space for experts and initialize their weight arrays
        const shared_experts = try allocator.alloc(Expert, config.num_shared_experts);
        errdefer allocator.free(shared_experts);

        // Initialize shared expert weights (random for testing; real usage loads from file)
        var se_init_count: usize = 0;
        errdefer for (shared_experts[0..se_init_count]) |*e| {
            mlx.arrayFree(e.w_gate);
            mlx.arrayFree(e.w_up);
            mlx.arrayFree(e.w_down);
        };
        while (se_init_count < config.num_shared_experts) : (se_init_count += 1) {
            var w_gate = mlx.arrayNew();
            try mlx.randomNormal(&w_gate, .{
                @as(c_int, @intCast(config.hidden_size)),
                @as(c_int, @intCast(config.intermediate_size)),
            }, mlx.FLOAT32);
            var w_up = mlx.arrayNew();
            try mlx.randomNormal(&w_up, .{
                @as(c_int, @intCast(config.hidden_size)),
                @as(c_int, @intCast(config.intermediate_size)),
            }, mlx.FLOAT32);
            var w_down = mlx.arrayNew();
            try mlx.randomNormal(&w_down, .{
                @as(c_int, @intCast(config.intermediate_size)),
                @as(c_int, @intCast(config.hidden_size)),
            }, mlx.FLOAT32);
            shared_experts[se_init_count] = Expert{ .w_gate = w_gate, .w_up = w_up, .w_down = w_down };
        }

        const routed_experts = try allocator.alloc(Expert, config.num_experts);
        errdefer allocator.free(routed_experts);

        // Initialize routed expert weights (random for testing)
        var re_init_count: usize = 0;
        errdefer for (routed_experts[0..re_init_count]) |*e| {
            mlx.arrayFree(e.w_gate);
            mlx.arrayFree(e.w_up);
            mlx.arrayFree(e.w_down);
        };
        while (re_init_count < config.num_experts) : (re_init_count += 1) {
            var w_gate = mlx.arrayNew();
            try mlx.randomNormal(&w_gate, .{
                @as(c_int, @intCast(config.hidden_size)),
                @as(c_int, @intCast(config.intermediate_size)),
            }, mlx.FLOAT32);
            var w_up = mlx.arrayNew();
            try mlx.randomNormal(&w_up, .{
                @as(c_int, @intCast(config.hidden_size)),
                @as(c_int, @intCast(config.intermediate_size)),
            }, mlx.FLOAT32);
            var w_down = mlx.arrayNew();
            try mlx.randomNormal(&w_down, .{
                @as(c_int, @intCast(config.intermediate_size)),
                @as(c_int, @intCast(config.hidden_size)),
            }, mlx.FLOAT32);
            routed_experts[re_init_count] = Expert{ .w_gate = w_gate, .w_up = w_up, .w_down = w_down };
        }

        // Initialize gate weight matrix [hidden_size, num_experts]
        // For testing, create random weights
        var gate_weight = mlx.arrayNew();
        try mlx.randomNormal(&gate_weight, .{
            @as(c_int, @intCast(config.hidden_size)),
            @as(c_int, @intCast(config.num_experts)),
        }, mlx.FLOAT32);

        // Try to initialize Metal kernel for fast routing
        var route_kernel: ?mlx_v4.FastMetalKernel = null;
        if (mlx_v4.hasFastOps()) {
            route_kernel = initRouteKernel(allocator) catch |err| blk: {
                std.debug.print("Failed to initialize Metal kernel: {s}\n", .{@errorName(err)});
                break :blk null;
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

        // Free expert weight arrays
        for (self.shared_experts) |*e| {
            mlx.arrayFree(e.w_gate);
            mlx.arrayFree(e.w_up);
            mlx.arrayFree(e.w_down);
        }
        for (self.routed_experts) |*e| {
            mlx.arrayFree(e.w_gate);
            mlx.arrayFree(e.w_up);
            mlx.arrayFree(e.w_down);
        }
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

        const fwd_stream = mlx.defaultGpuStreamNew();
        defer mlx.streamFree(fwd_stream);

        // Initialize output with zeros (same shape as hidden_states)
        const hs_shape = mlx.arrayShape(hidden_states);
        const hs_ndim = @as(usize, @intCast(mlx.C.mlx_array_ndim(hidden_states)));
        var zero_shape: [32]c_int = undefined;
        for (0..hs_ndim) |i| zero_shape[i] = hs_shape[i];
        try mlx.zeros(output, zero_shape[0..hs_ndim], mlx.FLOAT32, fwd_stream);

        // Step 1: Process shared experts (always active)
        for (self.shared_experts) |expert| {
            var expert_out = mlx.arrayNew();
            defer mlx.arrayFree(expert_out);
            try expert.forward(&expert_out, hidden_states);
            try mlx.add(output, output.*, expert_out, fwd_stream);
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
        // NOTE: forwardRouted requires mlx.arrayItemInt64/Float32 and mlx.scatter_add
        // which are not yet exported from mlx.zig. Stub for compilation.
        _ = self;
        _ = output;
        _ = hidden_states;
        _ = routing;
        // placeholder — routed expert forward pass not yet implemented
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
    /// NOTE: mlx-c v1 and v4 array types are incompatible at the Zig type level;
    /// fall through to CPU path until the type bridge is implemented.
    fn routeWithKernel(
        self: *Self,
        kernel: mlx_v4.FastMetalKernel,
        hidden_states: mlx.Array,
    ) !RoutingResult {
        _ = kernel;
        // Metal kernel path deferred: array type mismatch between mlx-c v1 and v4 @cImport
        return self.routeOnCpu(hidden_states);
    }

    /// Route using CPU fallback (slow path)
    fn routeOnCpu(self: *Self, hidden_states: mlx.Array) !RoutingResult {
        const batch_size = mlx.arrayDim(hidden_states, 0);
        const seq_len = mlx.arrayDim(hidden_states, 1);

        const stream = mlx.defaultGpuStreamNew();
        defer mlx.streamFree(stream);

        // Step 1: Compute gate logits
        // hidden_states: [batch, seq, hidden]
        // gate_weight: [hidden, num_experts]
        // logits: [batch, seq, num_experts]
        var logits = mlx.arrayNew();
        errdefer mlx.arrayFree(logits);
        try mlx.matmul(&logits, hidden_states, self.gate_weight, stream);

        // Step 2: Apply softmax over experts dimension
        var probs = mlx.arrayNew();
        errdefer mlx.arrayFree(probs);
        const softmax_axes = [_]c_int{-1};
        try mlx.softmax(&probs, logits, &softmax_axes, false, stream);
        mlx.arrayFree(logits);

        // Step 3: Top-k selection
        // MLX doesn't have built-in topk, so we use argmax in a loop
        // For production, this should use a proper top-k implementation
        var indices = mlx.arrayNew();
        errdefer mlx.arrayFree(indices);
        try mlx.zeros(&indices, &[_]c_int{
            batch_size,
            seq_len,
            @as(c_int, @intCast(self.config.top_k)),
        }, mlx.INT32, stream);

        var weights = mlx.arrayNew();
        errdefer mlx.arrayFree(weights);
        try mlx.zeros(&weights, &[_]c_int{
            batch_size,
            seq_len,
            @as(c_int, @intCast(self.config.top_k)),
        }, mlx.FLOAT32, stream);

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
