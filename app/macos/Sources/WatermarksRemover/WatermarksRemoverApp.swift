import AppKit
import SwiftUI

@main
@MainActor
struct WatermarksRemoverApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @StateObject private var settings: AppSettings
    @StateObject private var service: ServiceController
    @StateObject private var model: AppModel

    init() {
        let settings = AppSettings()
        let service = ServiceController()
        _settings = StateObject(wrappedValue: settings)
        _service = StateObject(wrappedValue: service)
        _model = StateObject(wrappedValue: AppModel(settings: settings, service: service))
    }

    var body: some Scene {
        WindowGroup {
            ContentView(service: service)
                .environmentObject(model)
                .frame(minWidth: 900, minHeight: 560)
        }
        .windowToolbarStyle(.unified)
        .commands { commands }

        Settings {
            SettingsView(settings: settings, service: service)
        }
    }

    @CommandsBuilder
    private var commands: some Commands {
        CommandGroup(replacing: .newItem) {}
        CommandGroup(after: .newItem) {
            Button("Add Files…") { addFiles() }
                .keyboardShortcut("o")
            Divider()
            Button("Export Report…") { model.exportReport() }
                .keyboardShortcut("e")
                .disabled(model.items.isEmpty)
        }
        CommandMenu("Service") {
            Button("Inspect All") { Task { await model.inspectAll() } }
                .keyboardShortcut("i", modifiers: [.command, .shift])
            Button("Clean All") { Task { await model.cleanAll() } }
                .keyboardShortcut("k", modifiers: [.command, .shift])
            Divider()
            Button("Restart Service") {
                Task { await service.restart(settings: settings) }
            }
            Button("Stop Service") { service.stop() }
                .disabled(!service.state.isRunning)
        }
        CommandGroup(replacing: .help) {
            Link(
                "watermarks-remover on GitHub",
                destination: URL(string: "https://github.com/guillaumemeyer/watermarks-remover")!)
            Link(
                "Responsible use",
                destination: URL(
                    string:
                        "https://github.com/guillaumemeyer/watermarks-remover#responsible-use")!)
        }
    }

    private func addFiles() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = true
        panel.canChooseFiles = true
        if panel.runModal() == .OK {
            model.add(urls: panel.urls)
        }
    }
}

/// Makes the app quit cleanly: the Python child process is terminated with the
/// last window, so a crashed or force-quit app never leaves a service listening.
final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    func applicationWillTerminate(_ notification: Notification) {
        ProcessRegistry.shared.terminateAll()
    }
}
