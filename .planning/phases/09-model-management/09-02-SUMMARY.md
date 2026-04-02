---
phase: 09
plan: 02
name: Model Auto-Download
subsystem: download
status: complete
dependencies: [09-01]
tech-stack:
  added:
    - src/download/huggingface.zig - HF API client
    - src/download/manager.zig - Download manager
    - src/download/mod.zig - Public API
  patterns:
    - HTTP Range requests for resume support
    - Thread-based background downloads
    - Atomic flags for cancellation
    - Global manager instance
key-files:
  created:
    - src/download/huggingface.zig
    - src/download/manager.zig
    - src/download/mod.zig
decisions:
  - Blocking download for MVP simplicity (non-blocking architecture ready)
  - Cache location ~/.cache/zlx/models/{org}/{repo}/
  - Filter files: config.json, tokenizer.json, *.safetensors
  - Skip SHA256 verification for MVP (file size check only)
  - HuggingFace API v2 (tree endpoint) for file listing
metrics:
  duration: 1h 15m
  completed: 2026-04-02
---

# Phase 09 Plan 02: Model Auto-Download Summary

## Overview

Implemented automatic model downloading from HuggingFace. The download system supports resume of interrupted downloads via HTTP Range requests, shows progress indicators, and has architecture for background downloads.

## What Was Built

### 1. HuggingFace Client (`src/download/huggingface.zig`)

**HuggingFaceId** - Parser for "org/repo" format:
- `parse()` - Validates and splits model IDs
- `buildDownloadUrl()` - Constructs download URLs
- `buildApiUrl()` - Constructs API endpoint URLs

**File Download**:
- `downloadFile()` - Downloads with HTTP Range support
  - Checks existing file size for resume
  - Handles 206 Partial Content (resume)
  - Handles 200 OK (full download)
  - Handles 416 Range Not Satisfiable (retry without range)
  - Progress callback support

**API Integration**:
- `getFileList()` - Queries HF API for model files
  - Filters to relevant files only
  - Parses JSON response
  - Returns FileInfo array with path, size

**Checksum Verification**:
- `verifyChecksum()` - Stub for SHA256 verification
  - Currently checks file exists and is non-empty
  - Full SHA256 implementation deferred

### 2. Download Manager (`src/download/manager.zig`)

**DownloadTask**:
- Model ID, HF ID, destination directory
- Status tracking (queued, downloading, completed, failed, cancelled)
- Progress tracking (files, bytes, timestamps)
- Thread handle and cancellation atomic flag
- Error message storage

**DownloadManager**:
- Queue management (ArrayList of tasks)
- Active download tracking
- Thread-safe access with mutex
- Integration with ModelRegistry for status updates

**Key Methods**:
- `queueDownload()` - Add model to download queue
- `startDownload()` - Spawn download thread
- `getProgress()` - Get current download progress
- `cancelDownload()` - Cancel active or queued download

**Background Download Architecture**:
- `downloadWorker()` runs in separate thread
- Streams files with progress updates
- Checks cancellation flag periodically
- Updates registry status (loading → available/failed)
- Processes next queued download on completion

**Global Instance**:
- `initGlobalManager()` - Initialize global manager
- `getGlobalManager()` - Access global instance
- `deinitGlobalManager()` - Cleanup

### 3. Public API (`src/download/mod.zig`)

**Convenience Functions**:
- `downloadModelIfNeeded()` - Check cache, return true if download needed
- `downloadModelBlocking()` - Blocking download with progress logging

**Re-exports**:
- All key types (HuggingFaceId, DownloadManager, DownloadStatus, etc.)
- Global manager functions

## Download Architecture

```
User requests model: "mlx-community/Qwen2.5-Coder-7B"
  ↓
DownloadManager.queueDownload()
  ↓
Parse HF ID → Check cache → Create DownloadTask
  ↓
Spawn thread: downloadWorker()
  ↓
Get file list from HF API
  ↓
For each file:
  - Build download URL
  - Check existing size (resume)
  - HTTP GET with Range header
  - Stream to file with progress
  ↓
Update registry status → Process next queued
```

## Cache Structure

```
~/.cache/zlx/models/
  └── {org}/
      └── {repo}/
          ├── config.json
          ├── tokenizer.json
          └── model.safetensors
```

## API Usage

```zig
// Initialize
const download_mod = @import("download/mod.zig");
try download_mod.initGlobalManager(allocator, registry);

// Check if download needed
if (try download_mod.downloadModelIfNeeded(allocator, "mlx-community/model")) {
    // Download model (blocking)
    try download_mod.downloadModelBlocking(allocator, "mlx-community/model");
}
```

## Verification

```bash
# Build succeeds
zig build

# All tests pass
zig build test
```

## Deviations from Plan

### Architectural Decisions

1. **Blocking downloads for MVP**: While the architecture supports background downloads with threads, the current `downloadModelBlocking()` waits for completion. True non-blocking (start server while downloading) can be added later.

2. **Simplified checksum verification**: Full SHA256 verification would require adding a SHA256 implementation. For MVP, we verify file exists and is non-empty.

3. **No persistent download queue**: Downloads are lost on restart. Production would save queue state.

4. **No download retry logic**: Failed downloads need manual retry. Production would auto-retry with backoff.

## Known Limitations

- Download checksums not fully verified (SHA256)
- No automatic retry on network failure
- Download queue not persisted across restarts
- HuggingFace API requires internet connection
- No proxy support configured
- Large models (>10GB) may timeout without progress updates

## Next Steps

Phase 09 Plan 03 will add:
1. Background model loading during active inference
2. Load management endpoints (/v1/models/load, /v1/models/load-status)
3. Enhanced CORS for Open WebUI integration
4. Checkpoint for human verification of full Phase 09
