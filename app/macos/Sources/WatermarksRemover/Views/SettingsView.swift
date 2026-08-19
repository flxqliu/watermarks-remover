import AppKit
import SwiftUI

struct SettingsView: View {
    @ObservedObject var settings: AppSettings
    @ObservedObject var service: ServiceController
    @State private var probeResult: String?

    var body: some View {
        TabView {
            general.tabItem { Label("General", systemImage: "gearshape") }
            serviceTab.tabItem { Label("Service", systemImage: "server.rack") }
        }
        .frame(width: 480, height: 380)
    }

    private var general: some View {
        Form {
            Section("Queue") {
                Toggle("Inspect files as soon as they are added", isOn: $settings.autoInspectOnAdd)
                Toggle("Run watermark detectors when inspecting", isOn: $settings.runDetectorsOnInspect)
                Text("Some detectors send text to a vendor API. Leave this off for private drafts.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Section("Output") {
                Picker("Write cleaned files", selection: $settings.outputMode) {
                    ForEach(OutputMode.allCases) { mode in
                        Text(mode.label).tag(mode)
                    }
                }
                Text(settings.outputMode.detail)
                    .font(.caption)
                    .foregroundStyle(settings.outputMode == .inPlace ? .orange : .secondary)
                if settings.outputMode == .folder {
                    HStack {
                        Text(settings.outputFolderPath.isEmpty ? "No folder chosen" : settings.outputFolderPath)
                            .lineLimit(1)
                            .truncationMode(.head)
                        Spacer()
                        Button("Choose…") { chooseFolder() }
                    }
                }
            }
        }
        .formStyle(.grouped)
    }

    private var serviceTab: some View {
        Form {
            Section("Local service") {
                HStack {
                    TextField("Python path", text: $settings.pythonPath, prompt: Text("Auto-detect"))
                    Button("Test") { probe() }
                }
                if let probeResult {
                    Text(probeResult)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Text(
                    "The app starts service/scripts/server.py on a private loopback port with a "
                        + "one-off bearer token. Nothing leaves your Mac unless you enable a "
                        + "vendor detector."
                )
                .font(.caption)
                .foregroundStyle(.secondary)
            }
            Section("Use a service I already run") {
                Toggle("Connect to an existing service", isOn: $settings.useExternalService)
                TextField("URL", text: $settings.externalServiceURL)
                    .disabled(!settings.useExternalService)
                SecureField("API key (optional)", text: $settings.externalAPIKey)
                    .disabled(!settings.useExternalService)
            }
            Section {
                HStack {
                    Text(service.state.label)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button("Restart service") {
                        Task { await service.restart(settings: settings) }
                    }
                }
            }
        }
        .formStyle(.grouped)
    }

    private func probe() {
        let path = settings.pythonPath.isEmpty ? nil : settings.pythonPath
        if let interpreter = PythonLocator.discover(preferred: path) {
            probeResult = "Found Python \(interpreter.version) at \(interpreter.url.path)"
        } else {
            probeResult = "No usable Python 3.10+ found. Try `brew install python`."
        }
    }

    private func chooseFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        if panel.runModal() == .OK, let url = panel.url {
            settings.outputFolderPath = url.path
        }
    }
}
