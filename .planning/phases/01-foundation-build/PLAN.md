# Plan: Phase 1 — Foundation & Build

## Goal

`zig build` produces a working `zig-out/bin/zlx` binary on macOS aarch64 with Zig 0.15.2, with MLX.zig and httpz correctly integrated and all C interop behind a single boundary.

## Requirements

BUILD-01, BUILD-02, BUILD-03, BUILD-04

## Context

- Zig 0.15.2 is installed (`/opt/homebrew/Cellar/zig/0.15.2_1/bin/zig`). All build artifacts must target this version.
- MLX.zig lives at `src/mlx.zig/` as a git submodule (commit bfdb46f, heads/main). It has no Zig module export — `b.dependency("mlx").module("mlx")` panics at build time.
- The build strategy (Approach B) is: do NOT include MLX.zig as a `build.zig.zon` package dependency. Instead, inline `setupDependencies` and `configureExecutable` from `src/mlx.zig/build.zig` directly into our `build.zig`, referencing MLX.zig source files via `b.path(...)`. Because Zig never resolves MLX.zig as a package, the submodule's own `build.zig.zon` (which uses the old string-name format) is never parsed — no patch needed.
- Phase 1 only needs the binary to compile, link, and exit cleanly. No MLX inference calls are made until Phase 2.

---

## Task 1: Rewrite `build.zig.zon` — BUILD-01, BUILD-03

**File:** `build.zig.zon`

**Action:** Rewrite (full replacement)

**Why:** The current file uses `.name = "zlx"` (a string), which is invalid in Zig 0.15 — the compiler requires an enum literal. It also has a placeholder `"..."` hash for httpz and includes an `.mlx` path dependency that we are removing (Approach B). pcre2 must be added here because `configureExecutable` will call `b.dependency("pcre2", ...)`.

**Hash notes:**

The hashes below were verified against Zig 0.15.2's `zig fetch` during the research phase. httpz master is a rolling branch — if `zig build` reports a hash mismatch, run `zig fetch --save https://github.com/karlseguin/http.zig/archive/refs/heads/master.tar.gz` to get the current hash and update this file.

**Full file content:**

```zig
.{
    .name = .zlx,
    .version = "0.0.1",
    .minimum_zig_version = "0.15.2",
    .dependencies = .{
        .httpz = .{
            .url = "https://github.com/karlseguin/http.zig/archive/refs/heads/master.tar.gz",
            .hash = "httpz-0.0.0-PNVzrMpOBwCS42DpyNnY8WTlwcuzannBeDLuqkUV8H7-",
        },
        .pcre2 = .{
            .url = "https://github.com/PCRE2Project/pcre2/archive/refs/tags/pcre2-10.45.tar.gz",
            .hash = "N-V-__8AAPh79wDJJ3MaSzH16_i08z8O_m-hVhzc8CbznL7l",
        },
    },
    .paths = .{
        "build.zig",
        "build.zig.zon",
        "src",
    },
}
```

**If `zig build` complains about a missing `.fingerprint` field**, add:
```zig
    .fingerprint = 0x0000000000000000,
```
after `.minimum_zig_version`. The fingerprint is optional for root packages but some Zig 0.15 patch releases enforce it. Any 64-bit hex literal is valid for a local root package.

---

## Task 2: Rewrite `build.zig` — BUILD-01, BUILD-02, BUILD-03

**File:** `build.zig`

**Action:** Rewrite (full replacement)

**Why:** The current file calls `b.dependency("mlx", .{}).module("mlx")` which panics because MLX.zig exports no Zig module. We remove that call entirely. Instead, we inline `setupDependencies`, `configureExecutable`, and `doesFileExist` from `src/mlx.zig/build.zig`. These functions download and build `libmlxc.a` (via curl + cmake), then link it, `libmlx.a`, and the required macOS frameworks into the executable. httpz is wired normally via `b.dependency("httpz")`.

**Notes for the executor:**

- `setCwd(.{ .cwd_relative = ... })` signature is unchanged in 0.15.2 — the research phase confirmed `b.cache_root.path.?` still works.
- If any `addSystemCommand` or path API has changed, fix compilation errors iteratively with `zig build 2>&1` — the errors will be descriptive.
- The `src/mlx.zig/` source files are referenced only for the include path (`mlx/c/mlx.h` lives inside the mlx-c download, not the submodule). We do not `addCSourceFiles` from the submodule yet — Phase 2 handles that.

**Full file content:**

```zig
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
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Wire httpz (BUILD-03)
    const httpz_dep = b.dependency("httpz", .{
        .target = target,
        .optimize = optimize,
    });
    exe.root_module.addImport("httpz", httpz_dep.module("httpz"));

    // Wire MLX-C + frameworks + pcre2 (BUILD-02)
    configureExecutable(exe, b, deps);

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    if (b.args) |args| run_cmd.addArgs(args);
    const run_step = b.step("run", "Run zlx inference server");
    run_step.dependOn(&run_cmd.step);
}

// ── Inlined from src/mlx.zig/build.zig ────────────────────────────────────────

const Dependencies = struct {
    mlx_c_path: []const u8,
    mlx_c_build_path: []const u8,
    mlx_c_lib_path: []const u8,
    install_step: *std.Build.Step,
    pcre2_dep: *std.Build.Dependency,
};

fn setupDependencies(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) !Dependencies {
    const mlx_c_path = b.pathJoin(&.{ b.cache_root.path.?, "mlx-c" });
    const mlx_c_build_path = b.pathJoin(&.{ mlx_c_path, "build" });
    const mlx_c_lib_path = b.pathJoin(&.{ mlx_c_build_path, "libmlxc.a" });

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
        const cmake_cmd = b.addSystemCommand(&[_][]const u8{ "cmake", "..", "-DCMAKE_BUILD_TYPE=Release" });
        cmake_cmd.setCwd(.{ .cwd_relative = mlx_c_build_path });
        cmake_cmd.step.dependOn(&mkdir_cmd.step);
        const make_cmd = b.addSystemCommand(&[_][]const u8{ "make", "-j" });
        make_cmd.setCwd(.{ .cwd_relative = mlx_c_build_path });
        make_cmd.step.dependOn(&cmake_cmd.step);
        install_step.dependOn(&make_cmd.step);
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

    const pcre2_dep = b.dependency("pcre2", .{
        .target = target,
        .optimize = optimize,
    });

    return Dependencies{
        .mlx_c_path = mlx_c_path,
        .mlx_c_build_path = mlx_c_build_path,
        .mlx_c_lib_path = mlx_c_lib_path,
        .install_step = install_step,
        .pcre2_dep = pcre2_dep,
    };
}

fn configureExecutable(
    exe: *std.Build.Step.Compile,
    b: *std.Build,
    deps: Dependencies,
) void {
    exe.step.dependOn(deps.install_step);
    exe.addIncludePath(.{ .cwd_relative = deps.mlx_c_path });
    exe.addObjectFile(.{ .cwd_relative = b.pathJoin(&.{ deps.mlx_c_build_path, "libmlxc.a" }) });
    exe.addObjectFile(.{ .cwd_relative = b.pathJoin(&.{ deps.mlx_c_build_path, "_deps/mlx-build/libmlx.a" }) });
    exe.linkFramework("Metal");
    exe.linkFramework("Foundation");
    exe.linkFramework("QuartzCore");
    exe.linkFramework("Accelerate");
    exe.linkLibCpp();
    exe.linkLibrary(deps.pcre2_dep.artifact("pcre2-8"));
}

fn doesFileExist(path: []const u8) bool {
    var file = std.fs.cwd().openFile(path, .{}) catch return false;
    file.close();
    return true;
}
```

---

## Task 3: Create `src/c.zig` and rewrite `src/main.zig` — BUILD-04, BUILD-01

### 3a — Create `src/c.zig`

**File:** `src/c.zig`

**Action:** Create (new file)

**Why (BUILD-04):** A single `@cImport` boundary prevents the "duplicate type" linker error that occurs when two Zig files both `@cImport` the same C header — the compiler generates distinct anonymous types per call site, which are not interchangeable. All C types (`mlx_array`, `mlx_stream`, etc.) flow through this one file. Phase 2 and Phase 3 code imports `@import("c.zig")` rather than doing their own `@cImport`.

**Full file content:**

```zig
// src/c.zig
// Single @cImport boundary for all mlx-c C interop (BUILD-04).
// Import this file everywhere MLX C types are needed:
//   const c = @import("c.zig");
//   _ = c.mlx.mlx_array_new();
pub const mlx = @cImport({
    @cInclude("mlx/c/mlx.h");
});
```

### 3b — Rewrite `src/main.zig`

**File:** `src/main.zig`

**Action:** Rewrite (full replacement)

**Why (BUILD-01):** The current `main.zig` calls `@import("zlx")` (the `zig init` scaffold's self-import) and `zlx.bufferedPrint()`, neither of which exist. This makes compilation fail immediately. Phase 1 only requires the binary to start and exit cleanly — no MLX inference is invoked until Phase 2. The stub imports `c.zig` (proving that mlx-c header resolution works at compile time) but does not call any mlx-c functions at runtime.

Note: Zig does not emit warnings for unused file-scope `const` declarations, so there is no need for `_ = c;` — the import alone forces the header to be processed by the compiler.

**Full file content:**

```zig
// src/main.zig
// Phase 1 stub — proves the build compiles and links.
// No MLX inference calls yet (Phase 2).
const std = @import("std");

// Import the C interop boundary to verify mlx-c header resolution at compile time.
// Unused at runtime in Phase 1; Phase 2 will call c.mlx.* functions.
const c = @import("c.zig");

pub fn main() !void {
    // Reference c to confirm the header resolved. This is a compile-time proof only.
    _ = c;
    const stdout = std.io.getStdOut().writer();
    try stdout.print("zlx: starting (Phase 1 stub — inference not yet wired)\n", .{});
    // Phase 2 will replace this with: model load, GenerationState init, HTTP server start.
}
```

**Note on `src/root.zig`:** The `zig init` scaffold created this file. Nothing in our build references it — it is safe to leave it as-is (it will not be compiled). Do not delete it unless it causes confusion; it is simply unreferenced.

---

## Verification Steps

Run these in order after completing all three tasks. Each command and its expected result are listed.

```sh
# 1. Confirm Zig version matches expected toolchain
zig version
# Expected: 0.15.2

# 2. Verify build.zig.zon parses cleanly (no "expected enum literal" error)
zig build --help 2>&1 | head -5
# Expected: help text, no parse errors

# 3. Full build (first run will download + cmake mlx-c — takes ~5 min)
zig build 2>&1
# Expected: exits 0, no errors

# 4. Confirm binary was produced
ls -lh zig-out/bin/zlx
# Expected: file exists, size > 0

# 5. Confirm binary architecture
file zig-out/bin/zlx
# Expected: "Mach-O 64-bit executable arm64"

# 6. Run the binary and confirm clean exit
./zig-out/bin/zlx
# Expected: prints "zlx: starting ..." and exits 0

# 7. Confirm exit code is 0
echo $?
# Expected: 0
```

**If hash mismatch on httpz:** Run `zig fetch --save https://github.com/karlseguin/http.zig/archive/refs/heads/master.tar.gz` and replace the hash in `build.zig.zon` with the value printed to stdout.

**If hash mismatch on pcre2:** Run `zig fetch --save https://github.com/PCRE2Project/pcre2/archive/refs/tags/pcre2-10.45.tar.gz` and replace accordingly.

**If cmake is not on PATH:** Install via `brew install cmake`. Required for the mlx-c build step.

**If mlx-c cmake step fails:** Check that Xcode command-line tools are installed (`xcode-select --install`) — Metal.framework is required for the mlx-c cmake build.
