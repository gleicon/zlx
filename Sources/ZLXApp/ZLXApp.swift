import SwiftUI
import AppKit

@main
struct ZLXApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        Settings {
            EmptyView()
        }
    }
}

@MainActor
class AppDelegate: NSObject, NSApplicationDelegate {
    var statusBar: StatusBarController?
    var serverManager = ServerManager()

    func applicationDidFinishLaunching(_ notification: Notification) {
        statusBar = StatusBarController(serverManager: serverManager)
    }

    func applicationWillTerminate(_ notification: Notification) {
        serverManager.stop()
    }
}
