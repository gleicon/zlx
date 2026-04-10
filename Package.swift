// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "ZLXServer",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.20.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", from: "2.30.0"),
        .package(url: "https://github.com/hummingbird-project/hummingbird", from: "2.0.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.5.0"),
    ],
    targets: [
        .executableTarget(
            name: "ZLXServer",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            resources: [
                // Include Metal libraries from mlx-swift
                .process("Resources")
            ],
            swiftSettings: [
                .unsafeFlags(["-enable-experimental-feature", "StrictConcurrency"]),
                .define("SWIFT_PACKAGE"),
            ]
        )
    ]
)
