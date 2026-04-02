//! Compression module for KV cache compression
//!
//! Provides pluggable compression backends for KV cache optimization.
//! Currently stubbed - TurboQuant integration requires significant porting effort.
//!
//! Architecture:
//! - KvCompressor: Generic interface for compression backends
//! - CompressionType: Enum of available compression methods
//! - CompressionConfig: Configuration for compression parameters

pub const KvCompressor = @import("kv_compressor.zig").KvCompressor;
pub const CompressionType = @import("kv_compressor.zig").CompressionType;
pub const CompressionConfig = @import("kv_compressor.zig").CompressionConfig;
pub const CompressionResult = @import("kv_compressor.zig").CompressionResult;
pub const TurboQuantCompressor = @import("turboquant_stub.zig").TurboQuantCompressor;
