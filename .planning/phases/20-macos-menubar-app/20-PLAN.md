# Phase 20: macOS Menubar App

## Goal
Transform the CLI-based ZLX server into a proper macOS menubar application with a native SwiftUI interface, allowing users to control the server from the system tray.

## Success Criteria
- [ ] Menu bar icon with status indicator
- [ ] Popover UI with server controls (start/stop, model selection, port configuration)
- [ ] Real-time log viewer in the UI
- [ ] Single-click access to web UI
- [ ] Auto-start server on app launch (optional)
- [ ] Proper .app bundle generation
- [ ] Code signing preparation (optional)

## Current State

### Completed
1. ✅ Fixed model loading (synchronous registry, no race conditions)
2. ✅ Removed emojis from all log output
3. ✅ Created macOS app target structure (Sources/ZLXApp/)
4. ✅ Implemented StatusBarController for menubar integration
5. ✅ Created SwiftUI MenuView with server controls
6. ✅ Implemented ServerManager with Process wrapper
7. ✅ Updated Package.swift with dual targets (ZLXServer + ZLXApp)

### Files Created
```
Sources/ZLXApp/
├── ZLXApp.swift              # App entry point
├── StatusBarController.swift # Menubar controller
└── MenuView.swift            # SwiftUI interface
```

## Architecture

### App Structure
```
ZLX.app/
├── Contents/
│   ├── MacOS/
│   │   ├── ZLXApp          # Main app binary
│   │   └── ZLXServer       # Embedded server binary
│   ├── Resources/
│   │   └── default.metallib # MLX Metal library
│   └── Info.plist
```

### Communication Flow
1. User clicks menubar icon → Popover appears
2. User clicks "Start Server" → ServerManager.launchProcess()
3. ServerManager spawns ZLXServer subprocess
4. Logs captured via Pipe and displayed in UI
5. User clicks "Open Web UI" → NSWorkspace.openURL()

## Implementation Details

### ServerManager (Sources/ZLXApp/MenuView.swift)
- @MainActor class for UI thread safety
- Process wrapper for ZLXServer subprocess
- Pipe-based log capture with real-time UI updates
- Model selection and port configuration

### StatusBarController (Sources/ZLXApp/StatusBarController.swift)
- NSStatusBar integration
- NSPopover for dropdown UI
- NSHostingController for SwiftUI embedding

### MenuView (Sources/ZLXApp/MenuView.swift)
- Model picker with available models
- Port configuration text field
- Start/Stop button with status indicator
- Scrollable log viewer (last 100 lines)
- Open Web UI and Quit buttons

## Build Instructions

### Build Server Binary
```bash
swift build --target ZLXServer -c release
```

### Build App
```bash
swift build --target ZLXApp -c release
```

### Create .app Bundle
```bash
# Create bundle structure
mkdir -p ZLX.app/Contents/MacOS
mkdir -p ZLX.app/Contents/Resources

# Copy binaries
cp .build/release/ZLXApp ZLX.app/Contents/MacOS/
cp .build/release/ZLXServer ZLX.app/Contents/MacOS/

# Copy metallib
cp Sources/ZLXServer/Resources/default.metallib ZLX.app/Contents/Resources/

# Create Info.plist
cat > ZLX.app/Contents/Info.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>ZLXApp</string>
    <key>CFBundleIdentifier</key>
    <string>com.yourcompany.zlx</string>
    <key>CFBundleName</key>
    <string>ZLX</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSUIElement</key>
    <true/>
</dict>
</plist>
EOF

# Run
open ZLX.app
```

## Testing Checklist

- [ ] App launches without crashing
- [ ] Menubar icon appears
- [ ] Clicking icon shows popover UI
- [ ] Model selection dropdown works
- [ ] Port configuration works
- [ ] Start Server button launches subprocess
- [ ] Logs appear in real-time
- [ ] Stop Server button terminates subprocess
- [ ] Open Web UI opens browser
- [ ] Quit button terminates server and exits app

## Future Enhancements

1. **Preferences Window**: Settings for default model, auto-start
2. **Keyboard Shortcuts**: Global hotkey to toggle server
3. **Notifications**: macOS notifications for server events
4. **Auto-updater**: Sparkle framework integration
5. **Code Signing**: Prepare for distribution outside App Store

## Notes for Other AI Agents

### If Continuing This Work
1. The ServerManager needs proper path resolution for the ZLXServer binary
2. Current implementation assumes the binary is in .build/debug/ during development
3. For production, the binary should be bundled in the app resources
4. The metallib loading needs to work from within the app bundle

### If Debugging
- Check Console.app for crash logs
- Use `log stream --predicate 'process == "ZLXApp"'` for real-time logging
- The popover may not show if the statusItem.button is nil

### If Extending
- Add more models to `availableModels` array
- Extend MenuView with additional controls (temperature, max tokens)
- Add preference pane using Settings scene
