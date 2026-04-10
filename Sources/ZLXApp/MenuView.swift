import SwiftUI
import Combine

@MainActor
class ServerManager: ObservableObject {
    @Published var isRunning = false
    @Published var status = "Stopped"
    @Published var port = 8080
    @Published var selectedModel = "qwen2.5-coder-1.5b"
    @Published var logs: [String] = []

    private var task: Process?
    private var logBuffer = ""

    let availableModels = [
        "qwen2.5-coder-1.5b",
        "qwen2.5-coder-3b",
        "qwen2.5-coder-7b",
        "deepseek-coder-v2-lite",
        "gemma-4-e4b"
    ]

    func start() {
        guard !isRunning else { return }

        let process = Process()
        process.executableURL = Bundle.main.url(forAuxiliaryExecutable: "ZLXServer")
            ?? URL(fileURLWithPath: ".build/debug/ZLXServer")

        process.arguments = [
            "--model", selectedModel,
            "--port", String(port),
            "--host", "127.0.0.1"
        ]

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe

        let handle = pipe.fileHandleForReading
        handle.readabilityHandler = { [weak self] fileHandle in
            guard let data = try? fileHandle.read(upToCount: 1024),
                  let str = String(data: data, encoding: .utf8) else { return }

            Task { @MainActor [weak self] in
                self?.processLogOutput(str)
            }
        }

        do {
            try process.run()
            task = process
            isRunning = true
            status = "Running on port \(port)"
            addLog("[ZLX] Server started on port \(port)")
        } catch {
            status = "Failed to start: \(error.localizedDescription)"
            addLog("[ZLX] Error: \(error)")
        }
    }

    func stop() {
        task?.terminate()
        task = nil
        isRunning = false
        status = "Stopped"
        addLog("[ZLX] Server stopped")
    }

    func restart() {
        stop()
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
            self?.start()
        }
    }

    private func processLogOutput(_ str: String) {
        logBuffer.append(str)
        let lines = logBuffer.components(separatedBy: .newlines)
        if lines.count > 1 {
            logs.append(contentsOf: lines.dropLast())
            logBuffer = lines.last ?? ""
            if logs.count > 100 {
                logs.removeFirst(logs.count - 100)
            }
        }
    }

    private func addLog(_ message: String) {
        logs.append(message)
        if logs.count > 100 {
            logs.removeFirst(logs.count - 100)
        }
    }
}

struct MenuView: View {
    @EnvironmentObject var serverManager: ServerManager

    var body: some View {
        VStack(spacing: 16) {
            // Header
            HStack {
                Image(systemName: "cpu")
                    .font(.title2)
                Text("ZLX Server")
                    .font(.headline)
                Spacer()
                StatusIndicator(isRunning: serverManager.isRunning)
            }

            Divider()

            // Controls
            VStack(alignment: .leading, spacing: 12) {
                // Model selector
                HStack {
                    Text("Model:")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    Spacer()
                    Picker("", selection: $serverManager.selectedModel) {
                        ForEach(serverManager.availableModels, id: \.self) { model in
                            Text(model).tag(model)
                        }
                    }
                    .pickerStyle(MenuPickerStyle())
                    .frame(width: 180)
                    .disabled(serverManager.isRunning)
                }

                // Port selector
                HStack {
                    Text("Port:")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    Spacer()
                    TextField("Port", value: $serverManager.port, formatter: NumberFormatter())
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                        .frame(width: 80)
                        .disabled(serverManager.isRunning)
                }

                // Start/Stop button
                Button(action: {
                    if serverManager.isRunning {
                        serverManager.stop()
                    } else {
                        serverManager.start()
                    }
                }) {
                    HStack {
                        Image(systemName: serverManager.isRunning ? "stop.fill" : "play.fill")
                        Text(serverManager.isRunning ? "Stop Server" : "Start Server")
                    }
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(ControlButtonStyle(isRunning: serverManager.isRunning))
            }

            Divider()

            // Status
            HStack {
                Text("Status:")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                Spacer()
                Text(serverManager.status)
                    .font(.subheadline)
                    .foregroundColor(serverManager.isRunning ? .green : .secondary)
            }

            // Logs
            VStack(alignment: .leading, spacing: 4) {
                Text("Recent Logs")
                    .font(.caption)
                    .foregroundColor(.secondary)

                ScrollView {
                    VStack(alignment: .leading, spacing: 2) {
                        ForEach(serverManager.logs.suffix(10), id: \.self) { log in
                            Text(log)
                                .font(.system(.caption, design: .monospaced))
                                .foregroundColor(.secondary)
                                .lineLimit(1)
                        }
                    }
                }
                .frame(height: 100)
                .background(Color(NSColor.textBackgroundColor))
                .cornerRadius(4)
            }

            Divider()

            // Links
            HStack {
                Button("Open Web UI") {
                    if let url = URL(string: "http://localhost:\(serverManager.port)") {
                        NSWorkspace.shared.open(url)
                    }
                }
                .disabled(!serverManager.isRunning)
                .buttonStyle(LinkButtonStyle())

                Spacer()

                Button("Quit") {
                    serverManager.stop()
                    NSApplication.shared.terminate(nil)
                }
                .buttonStyle(LinkButtonStyle())
            }
        }
        .padding()
        .frame(width: 340)
    }
}

struct StatusIndicator: View {
    let isRunning: Bool

    var body: some View {
        Circle()
            .fill(isRunning ? Color.green : Color.gray)
            .frame(width: 10, height: 10)
    }
}

struct ControlButtonStyle: ButtonStyle {
    let isRunning: Bool

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .padding(.vertical, 8)
            .background(isRunning ? Color.red.opacity(0.2) : Color.blue.opacity(0.2))
            .foregroundColor(isRunning ? .red : .blue)
            .cornerRadius(6)
            .scaleEffect(configuration.isPressed ? 0.95 : 1.0)
    }
}

struct LinkButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundColor(.blue)
            .font(.subheadline)
    }
}
