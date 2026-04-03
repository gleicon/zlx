const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{
        .default_target = .{ .cpu_arch = .aarch64, .os_tag = .macos },
    });
    const optimize = b.standardOptimizeOption(.{});

    // Inline dependency setup from src/mlx.zig/build.zig (Approach B).
    // MLX.zig has no Zig module export, so we do not use b.dependency("mlx").
    const deps = try setupDependencies(b, target, optimize);

    const exe = b.addExecutable(.{
        .name = "zlx",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    // Wire httpz (BUILD-03)
    const httpz_dep = b.dependency("httpz", .{
        .target = target,
        .optimize = optimize,
    });
    exe.root_module.addImport("httpz", httpz_dep.module("httpz"));

    // Wire turboquant via git submodule (PHASE-07-02)
    // Source is in nested turboquant/ directory
    const turboquant_mod = b.createModule(.{
        .root_source_file = b.path("deps/turboquant/turboquant/src/turboquant.zig"),
        .target = target,
        .optimize = optimize,
    });
    // Add internal dependencies that turboquant expects
    turboquant_mod.addImport("matrix", b.createModule(.{
        .root_source_file = b.path("deps/turboquant/turboquant/src/matrix.zig"),
        .target = target,
        .optimize = optimize,
    }));
    turboquant_mod.addImport("polar", b.createModule(.{
        .root_source_file = b.path("deps/turboquant/turboquant/src/polar.zig"),
        .target = target,
        .optimize = optimize,
    }));
    turboquant_mod.addImport("qjl", b.createModule(.{
        .root_source_file = b.path("deps/turboquant/turboquant/src/qjl.zig"),
        .target = target,
        .optimize = optimize,
    }));
    turboquant_mod.addImport("format", b.createModule(.{
        .root_source_file = b.path("deps/turboquant/turboquant/src/format.zig"),
        .target = target,
        .optimize = optimize,
    }));
    turboquant_mod.addImport("rotation", b.createModule(.{
        .root_source_file = b.path("deps/turboquant/turboquant/src/rotation.zig"),
        .target = target,
        .optimize = optimize,
    }));
    turboquant_mod.addImport("math", b.createModule(.{
        .root_source_file = b.path("deps/turboquant/turboquant/src/math.zig"),
        .target = target,
        .optimize = optimize,
    }));
    exe.root_module.addImport("turboquant", turboquant_mod);

    // Wire MLX-C + frameworks + pcre2 (BUILD-02)
    configureExecutable(exe, b, deps);

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    if (b.args) |args| run_cmd.addArgs(args);
    const run_step = b.step("run", "Run zlx inference server");
    run_step.dependOn(&run_cmd.step);

    // Add test step for registry and other modules
    const test_step = b.step("test", "Run unit tests");

    // Test registry module
    const registry_test = b.addTest(.{
        .name = "registry_test",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/models/registry.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const run_registry_test = b.addRunArtifact(registry_test);
    test_step.dependOn(&run_registry_test.step);

    // Test models module
    const models_test = b.addTest(.{
        .name = "models_test",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/models/mod.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const run_models_test = b.addRunArtifact(models_test);
    test_step.dependOn(&run_models_test.step);

    // Test memory module
    const memory_test = b.addTest(.{
        .name = "memory_test",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/models/memory.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const run_memory_test = b.addRunArtifact(memory_test);
    test_step.dependOn(&run_memory_test.step);

    // Test prompt cache module
    const cache_test = b.addTest(.{
        .name = "cache_test",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cache/prompt_cache.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const run_cache_test = b.addRunArtifact(cache_test);
    test_step.dependOn(&run_cache_test.step);

    // Test turboquant integration (PHASE-07-02)
    const turboquant_test_mod = b.createModule(.{
        .root_source_file = b.path("src/turboquant_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    turboquant_test_mod.addImport("turboquant", turboquant_mod);

    const turboquant_test = b.addTest(.{
        .name = "turboquant_test",
        .root_module = turboquant_test_mod,
    });

    const run_turboquant_test = b.addRunArtifact(turboquant_test);
    test_step.dependOn(&run_turboquant_test.step);

    // Test mlx_bridge (PHASE-07-02)
    const mlx_bridge_test_mod = b.createModule(.{
        .root_source_file = b.path("src/mlx_bridge.zig"),
        .target = target,
        .optimize = optimize,
    });
    mlx_bridge_test_mod.addImport("mlx.zig", b.createModule(.{
        .root_source_file = b.path("src/mlx.zig/src/mlx.zig"),
        .target = target,
        .optimize = optimize,
    }));

    const c_mod = b.createModule(.{
        .root_source_file = b.path("src/c.zig"),
        .target = target,
        .optimize = optimize,
    });
    mlx_bridge_test_mod.addImport("c.zig", c_mod);

    const mlx_bridge_test = b.addTest(.{
        .name = "mlx_bridge_test",
        .root_module = mlx_bridge_test_mod,
    });

    const run_mlx_bridge_test = b.addRunArtifact(mlx_bridge_test);
    test_step.dependOn(&run_mlx_bridge_test.step);

    // Test turboquant_engine (PHASE-07-02)
    const turboquant_engine_test_mod = b.createModule(.{
        .root_source_file = b.path("src/compression/turboquant_engine.zig"),
        .target = target,
        .optimize = optimize,
    });
    turboquant_engine_test_mod.addImport("turboquant", turboquant_mod);

    const turboquant_engine_test = b.addTest(.{
        .name = "turboquant_engine_test",
        .root_module = turboquant_engine_test_mod,
    });

    const run_turboquant_engine_test = b.addRunArtifact(turboquant_engine_test);
    test_step.dependOn(&run_turboquant_engine_test.step);

    // Note: kv_compressor is tested via main executable build
    // (complex dependencies make isolated testing difficult)

    // Test config module (PHASE-09-01)
    const config_test = b.addTest(.{
        .name = "config_test",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/config_test.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const run_config_test = b.addRunArtifact(config_test);
    test_step.dependOn(&run_config_test.step);

    // Test mlx_v4 module (PHASE-11-01)
    const mlx_v4_test_mod = b.createModule(.{
        .root_source_file = b.path("src/mlx_v4_test.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add v0.4.x include paths for test compilation (use absolute path)
    const mlx_c_v4_absolute = b.pathJoin(&.{ "/Users/gleicon/code/zig/zlx", deps.mlx_c_v4_path });
    mlx_v4_test_mod.addIncludePath(.{ .cwd_relative = mlx_c_v4_absolute });
    mlx_v4_test_mod.addObjectFile(.{ .cwd_relative = deps.mlx_c_v4_lib_path });
    mlx_v4_test_mod.addObjectFile(.{ .cwd_relative = b.pathJoin(&.{ deps.mlx_c_v4_build_path, "_deps/mlx-build/libmlx.a" }) });

    const mlx_v4_test = b.addTest(.{
        .name = "mlx_v4_test",
        .root_module = mlx_v4_test_mod,
    });
    mlx_v4_test.addIncludePath(.{ .cwd_relative = mlx_c_v4_absolute });
    mlx_v4_test.addObjectFile(.{ .cwd_relative = deps.mlx_c_v4_lib_path });
    mlx_v4_test.addObjectFile(.{ .cwd_relative = b.pathJoin(&.{ deps.mlx_c_v4_build_path, "_deps/mlx-build/libmlx.a" }) });
    mlx_v4_test.addLibraryPath(.{ .cwd_relative = "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/lib" });
    mlx_v4_test.addFrameworkPath(.{ .cwd_relative = "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks" });
    mlx_v4_test.linkLibCpp();
    mlx_v4_test.linkFramework("Metal");
    mlx_v4_test.linkFramework("Foundation");
    mlx_v4_test.linkFramework("QuartzCore");
    mlx_v4_test.linkFramework("Accelerate");

    const run_mlx_v4_test = b.addRunArtifact(mlx_v4_test);
    test_step.dependOn(&run_mlx_v4_test.step);

    // Test MoE module (PHASE-11-03)
    const moe_test_mod = b.createModule(.{
        .root_source_file = b.path("src/moe_test.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add dependencies for moe tests
    moe_test_mod.addImport("mlx.zig/src/mlx.zig", b.createModule(.{
        .root_source_file = b.path("src/mlx.zig/src/mlx.zig"),
        .target = target,
        .optimize = optimize,
    }));
    moe_test_mod.addImport("moe.zig", b.createModule(.{
        .root_source_file = b.path("src/moe.zig"),
        .target = target,
        .optimize = optimize,
    }));
    moe_test_mod.addImport("mlx_v4.zig", b.createModule(.{
        .root_source_file = b.path("src/mlx_v4.zig"),
        .target = target,
        .optimize = optimize,
    }));

    const moe_test = b.addTest(.{
        .name = "moe_test",
        .root_module = moe_test_mod,
    });

    const run_moe_test = b.addRunArtifact(moe_test);
    test_step.dependOn(&run_moe_test.step);

    // Test DeepSeek module (PHASE-11-04)
    const deepseek_test_mod = b.createModule(.{
        .root_source_file = b.path("src/deepseek_test.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add dependencies for deepseek tests
    deepseek_test_mod.addImport("mlx.zig/src/mlx.zig", b.createModule(.{
        .root_source_file = b.path("src/mlx.zig/src/mlx.zig"),
        .target = target,
        .optimize = optimize,
    }));
    deepseek_test_mod.addImport("mlx.zig/src/mla.zig", b.createModule(.{
        .root_source_file = b.path("src/mlx.zig/src/mla.zig"),
        .target = target,
        .optimize = optimize,
    }));
    deepseek_test_mod.addImport("moe.zig", b.createModule(.{
        .root_source_file = b.path("src/moe.zig"),
        .target = target,
        .optimize = optimize,
    }));
    deepseek_test_mod.addImport("deepseek.zig", b.createModule(.{
        .root_source_file = b.path("src/deepseek.zig"),
        .target = target,
        .optimize = optimize,
    }));
    deepseek_test_mod.addImport("inference/mod.zig", b.createModule(.{
        .root_source_file = b.path("src/inference/mod.zig"),
        .target = target,
        .optimize = optimize,
    }));
    deepseek_test_mod.addImport("inference/loader.zig", b.createModule(.{
        .root_source_file = b.path("src/inference/loader.zig"),
        .target = target,
        .optimize = optimize,
    }));

    const deepseek_test = b.addTest(.{
        .name = "deepseek_test",
        .root_module = deepseek_test_mod,
    });

    const run_deepseek_test = b.addRunArtifact(deepseek_test);
    test_step.dependOn(&run_deepseek_test.step);

    // Note: manager.zig tests are compiled as part of main build
    // due to cross-module dependencies
    // due to cross-module dependencies
}

// ── Inlined from src/mlx.zig/build.zig ────────────────────────────────────────

const Dependencies = struct {
    mlx_c_path: []const u8,
    mlx_c_build_path: []const u8,
    mlx_c_lib_path: []const u8,
    mlx_c_v4_path: []const u8,
    mlx_c_v4_build_path: []const u8,
    mlx_c_v4_lib_path: []const u8,
    install_step: *std.Build.Step,
    install_v4_step: *std.Build.Step,
};

fn setupDependencies(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) !Dependencies {
    _ = target;
    _ = optimize;

    // v0.1.2 paths
    const mlx_c_path = b.pathJoin(&.{ b.cache_root.path.?, "mlx-c" });
    const mlx_c_build_path = b.pathJoin(&.{ mlx_c_path, "build" });
    const mlx_c_lib_path = b.pathJoin(&.{ mlx_c_build_path, "libmlxc.a" });

    // v0.4.1 paths
    const mlx_c_v4_path = b.pathJoin(&.{ b.cache_root.path.?, "mlx-c-v4" });
    const mlx_c_v4_build_path = b.pathJoin(&.{ mlx_c_v4_path, "build" });
    const mlx_c_v4_lib_path = b.pathJoin(&.{ mlx_c_v4_build_path, "libmlxc-v4.a" });

    const install_step = b.step("install-mlx-c", "Download and build libmlxc.a if not cached");
    const needs_install = !doesFileExist(mlx_c_lib_path);

    if (needs_install) {
        // Download mlx-c v0.1.2 tarball and extract to cache
        const clone_cmd = b.addSystemCommand(&[_][]const u8{
            "sh", "-c",
            b.fmt(
                "if [ ! -d {s} ]; then mkdir -p $(dirname {s}) && " ++
                    "curl -L https://github.com/ml-explore/mlx-c/archive/refs/tags/v0.1.2.tar.gz " ++
                    "| tar xz -C $(dirname {s}) && mv $(dirname {s})/mlx-c-0.1.2 {s}; fi",
                .{ mlx_c_path, mlx_c_path, mlx_c_path, mlx_c_path, mlx_c_path },
            ),
        });
        const mkdir_cmd = b.addSystemCommand(&[_][]const u8{ "mkdir", "-p", mlx_c_build_path });
        mkdir_cmd.step.dependOn(&clone_cmd.step);
        const cmake_cmd = b.addSystemCommand(&[_][]const u8{
            "cmake",                      "..",
            "-DCMAKE_BUILD_TYPE=Release", "-DMLX_BUILD_METAL=ON",
            "-DCMAKE_CXX_FLAGS=-w",
        });
        cmake_cmd.setCwd(.{ .cwd_relative = mlx_c_build_path });
        cmake_cmd.step.dependOn(&mkdir_cmd.step);
        const make_cmd = b.addSystemCommand(&[_][]const u8{ "make", "-j" });
        make_cmd.setCwd(.{ .cwd_relative = mlx_c_build_path });
        make_cmd.step.dependOn(&cmake_cmd.step);
        install_step.dependOn(&make_cmd.step);
    }

    // v0.4.1 install step
    const install_v4_step = b.step("install-mlx-c-v4", "Download and build libmlxc-v4.a if not cached");
    const needs_v4_install = !doesFileExist(mlx_c_v4_lib_path);

    if (needs_v4_install) {
        // Download mlx-c v0.4.1 tarball and extract to cache
        const clone_v4_cmd = b.addSystemCommand(&[_][]const u8{
            "sh", "-c",
            b.fmt(
                "if [ ! -d {s} ]; then mkdir -p $(dirname {s}) && " ++
                    "curl -L https://github.com/ml-explore/mlx-c/archive/refs/tags/v0.4.1.tar.gz " ++
                    "| tar xz -C $(dirname {s}) && mv $(dirname {s})/mlx-c-0.4.1 {s}; fi",
                .{ mlx_c_v4_path, mlx_c_v4_path, mlx_c_v4_path, mlx_c_v4_path, mlx_c_v4_path },
            ),
        });
        const mkdir_v4_cmd = b.addSystemCommand(&[_][]const u8{ "mkdir", "-p", mlx_c_v4_build_path });
        mkdir_v4_cmd.step.dependOn(&clone_v4_cmd.step);
        const cmake_v4_cmd = b.addSystemCommand(&[_][]const u8{
            "cmake",                      "..",
            "-DCMAKE_BUILD_TYPE=Release", "-DMLX_BUILD_METAL=ON",
            "-DCMAKE_CXX_FLAGS=-w",
        });
        cmake_v4_cmd.setCwd(.{ .cwd_relative = mlx_c_v4_build_path });
        cmake_v4_cmd.step.dependOn(&mkdir_v4_cmd.step);
        const make_v4_cmd = b.addSystemCommand(&[_][]const u8{ "make", "-j" });
        make_v4_cmd.setCwd(.{ .cwd_relative = mlx_c_v4_build_path });
        make_v4_cmd.step.dependOn(&cmake_v4_cmd.step);
        // Rename libmlxc.a to libmlxc-v4.a to avoid symbol conflicts
        const rename_v4_cmd = b.addSystemCommand(&[_][]const u8{
            "sh", "-c",
            b.fmt(
                "if [ -f {s}/libmlxc.a ] && [ ! -f {s} ]; then mv {s}/libmlxc.a {s}; fi",
                .{ mlx_c_v4_build_path, mlx_c_v4_lib_path, mlx_c_v4_build_path, mlx_c_v4_lib_path },
            ),
        });
        rename_v4_cmd.step.dependOn(&make_v4_cmd.step);
        install_v4_step.dependOn(&rename_v4_cmd.step);
    }

    // Copy mlx.metallib to install dir if present
    if (doesFileExist(b.pathJoin(&.{ mlx_c_build_path, "_deps/mlx-build/mlx.metallib" }))) {
        const dest_dir = b.pathJoin(&.{ b.install_path, "lib", "metal" });
        const mkdir_cmd = b.addSystemCommand(&.{ "mkdir", "-p", dest_dir });
        const copy_cmd = b.addSystemCommand(&.{
            "cp",
            b.pathJoin(&.{ mlx_c_build_path, "_deps/mlx-build/mlx.metallib" }),
            b.pathJoin(&.{ dest_dir, "mlx.metallib" }),
        });
        copy_cmd.step.dependOn(&mkdir_cmd.step);
        b.getInstallStep().dependOn(&copy_cmd.step);
    }

    return Dependencies{
        .mlx_c_path = mlx_c_path,
        .mlx_c_build_path = mlx_c_build_path,
        .mlx_c_lib_path = mlx_c_lib_path,
        .mlx_c_v4_path = mlx_c_v4_path,
        .mlx_c_v4_build_path = mlx_c_v4_build_path,
        .mlx_c_v4_lib_path = mlx_c_v4_lib_path,
        .install_step = install_step,
        .install_v4_step = install_v4_step,
    };
}

fn configureExecutable(
    exe: *std.Build.Step.Compile,
    b: *std.Build,
    deps: Dependencies,
) void {
    exe.step.dependOn(deps.install_step);
    exe.step.dependOn(deps.install_v4_step);

    // Get absolute paths for C imports
    const cwd = std.fs.cwd();
    const mlx_c_absolute = cwd.realpathAlloc(b.allocator, deps.mlx_c_path) catch deps.mlx_c_path;
    const mlx_c_v4_absolute = cwd.realpathAlloc(b.allocator, deps.mlx_c_v4_path) catch deps.mlx_c_v4_path;

    // macOS SDK framework path — required on macOS 26 / Xcode 21 where Zig doesn't auto-detect it
    exe.addFrameworkPath(.{ .cwd_relative = "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks" });
    // macOS SDK library path — needed for libobjc and other system libraries
    exe.addLibraryPath(.{ .cwd_relative = "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/lib" });

    // v0.1.2 includes and library
    exe.addIncludePath(.{ .cwd_relative = mlx_c_absolute });
    exe.addObjectFile(.{ .cwd_relative = b.pathJoin(&.{ deps.mlx_c_build_path, "libmlxc.a" }) });
    exe.addObjectFile(.{ .cwd_relative = b.pathJoin(&.{ deps.mlx_c_build_path, "_deps/mlx-build/libmlx.a" }) });

    // v0.4.x includes and library (separate to avoid conflicts)
    exe.addIncludePath(.{ .cwd_relative = mlx_c_v4_absolute });
    exe.addLibraryPath(.{ .cwd_relative = deps.mlx_c_v4_build_path });
    exe.linkSystemLibrary("mlxc-v4");

    exe.linkFramework("Metal");
    exe.linkFramework("Foundation");
    exe.linkFramework("QuartzCore");
    exe.linkFramework("Accelerate");
    exe.linkLibCpp();
    // pcre2: link arm64 static lib directly, bypassing pkg-config which finds an
    // old x86_64-only installation at /usr/local/. The homebrew arm64 build is at
    // /opt/homebrew/opt/pcre2/. The upstream pcre2 tarball's build.zig is also
    // incompatible with Zig 0.15.2 (Zig 0.14-era API), so we don't use b.dependency.
    exe.addIncludePath(.{ .cwd_relative = "/opt/homebrew/opt/pcre2/include" });
    exe.addObjectFile(.{ .cwd_relative = "/opt/homebrew/opt/pcre2/lib/libpcre2-8.a" });
}

fn doesFileExist(path: []const u8) bool {
    var file = std.fs.cwd().openFile(path, .{}) catch return false;
    file.close();
    return true;
}
