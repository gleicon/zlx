# ZLX macOS Menubar App

A proper macOS menubar application for the ZLX inference server.

## Features

- **Menu Bar Icon**: Always-available status indicator in the system tray
- **Server Controls**: Start/stop the inference server with one click
- **Model Selection**: Choose from Qwen, DeepSeek, and Gemma models
- **Port Configuration**: Customize the server port
- **Real-time Logs**: View server output directly in the app
- **Web UI Access**: Single-click to open the API endpoint in browser

## Project Structure

```
Sources/
├── ZLXServer/        # CLI server (headless)
│   ├── main.swift
│   ├── ModelRegistry.swift
│   ├── ModelLoader.swift
│   ├── ModelContainer.swift
│   └── Gemma4PLEBlock.swift
└── ZLXApp/           # macOS menubar app
    ├── ZLXApp.swift
    ├── StatusBarController.swift
    └── MenuView.swift
```

## Building

### Build Server
```bash
swift build --target ZLXServer -c release
```

### Build App
```bash
swift build --target ZLXApp -c release
```

### Create App Bundle
```bash
./scripts/build-app-bundle.sh
```

## Usage

### Development
```bash
# Run server directly
./run.sh --model qwen2.5-coder-1.5b

# Run app (menubar)
swift run ZLXApp
```

### Production
```bash
# Open the built app
open ZLX.app
```

## Architecture

The app uses a subprocess model:
1. ZLXApp (SwiftUI menubar app) manages the UI
2. ZLXServer (CLI) runs as a subprocess
3. Pipe-based communication for log streaming
4. NSWorkspace for opening the web UI

This approach:
- Keeps the server logic separate and testable
- Allows the CLI to work standalone
- Simplifies the app (no MLX integration needed in UI layer)

## Configuration

The app looks for the ZLXServer binary in:
1. Bundle resources (production)
2. `.build/debug/ZLXServer` (development)
3. `.build/release/ZLXServer` (release builds)

## Troubleshooting

### App doesn't show in menubar
- Check that LSUIElement is set in Info.plist
- Look for crash logs in Console.app

### Server won't start
- Verify metallib is in Resources/
- Check Server logs in the UI
- Ensure port is not in use: `lsof -i :8080`

### Model not found
- Models are downloaded from HuggingFace on first use
- Requires internet connection
- Check logs for download errors

## GSD Documentation

See `.planning/phases/20-macos-menubar-app/20-PLAN.md` for full implementation details.
