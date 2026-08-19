import SwiftUI

/// The window's bottom bar: service state, capabilities, progress.
struct StatusFooter: View {
    @EnvironmentObject private var model: AppModel
    @ObservedObject var service: ServiceController
    @State private var showingCapabilities = false
    @State private var showingLog = false

    var body: some View {
        HStack(spacing: 10) {
            Circle()
                .fill(stateColor)
                .frame(width: 8, height: 8)
            Text(statusText)
                .font(.callout)
                .lineLimit(1)

            if case .failed(let message) = service.state {
                Button {
                    showingLog = true
                } label: {
                    Image(systemName: "info.circle")
                }
                .buttonStyle(.borderless)
                .help(message)
            }

            if let progress = model.progress {
                ProgressView(value: Double(progress.done), total: Double(progress.total))
                    .progressViewStyle(.linear)
                    .frame(width: 130)
                Text("\(progress.done)/\(progress.total)")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }

            Spacer()

            if !model.items.isEmpty {
                Text("\(model.items.count) file\(model.items.count == 1 ? "" : "s")")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                if model.suspiciousCount > 0 {
                    Chip(text: "\(model.suspiciousCount) with marks", tint: .orange)
                }
            }

            Button {
                showingCapabilities = true
            } label: {
                Label("Capabilities", systemImage: "puzzlepiece.extension")
            }
            .buttonStyle(.borderless)
            .popover(isPresented: $showingCapabilities, arrowEdge: .bottom) {
                CapabilitiesPopover(capabilities: service.capabilities, service: service)
            }

            switch service.state {
            case .running:
                Button("Stop") { service.stop() }
                    .buttonStyle(.borderless)
            case .starting:
                ProgressView().controlSize(.small)
            default:
                Button("Start") {
                    Task { await service.start(settings: model.settings) }
                }
                .buttonStyle(.borderless)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(.bar)
        .overlay(alignment: .top) { Divider() }
        .sheet(isPresented: $showingLog) {
            ServiceLogSheet(service: service)
        }
    }

    private var statusText: String {
        switch service.state {
        case .running:
            let version = service.capabilities?.version ?? ""
            let interpreter = service.interpreter.map { "Python \($0.version)" } ?? "external service"
            return "Service ready" + (version.isEmpty ? "" : " · \(version)") + " · \(interpreter)"
        case .failed(let message):
            return message.split(separator: "\n").first.map(String.init) ?? "Service unavailable"
        default:
            return service.state.label
        }
    }

    private var stateColor: Color {
        switch service.state {
        case .running: return .green
        case .starting: return .yellow
        case .failed: return .red
        case .idle: return .secondary
        }
    }
}

struct CapabilitiesPopover: View {
    let capabilities: Capabilities?
    @ObservedObject var service: ServiceController

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Service capabilities")
                .font(.headline)
            if let capabilities {
                section("System tools", capabilities.tools)
                section("Backends and scorers", capabilities.backends)
                section("Text detectors", capabilities.detectors)
                Text(
                    "Missing tools are optional. exiftool and qpdf matter most: without "
                        + "them PDF cleaning falls back to a partial strip."
                )
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(maxWidth: 320, alignment: .leading)
            } else {
                Text("Start the service to see what it can do.")
                    .foregroundStyle(.secondary)
            }
            HStack {
                Spacer()
                Button("Refresh") { Task { await service.refreshCapabilities() } }
                    .disabled(!service.state.isRunning)
            }
        }
        .padding(16)
        .frame(width: 360)
    }

    @ViewBuilder
    private func section(_ title: String, _ items: [Capabilities.Item]) -> some View {
        if !items.isEmpty {
            VStack(alignment: .leading, spacing: 5) {
                Text(title)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                ForEach(items) { item in
                    HStack(spacing: 7) {
                        Image(systemName: item.available ? "checkmark.circle.fill" : "circle.dashed")
                            .foregroundStyle(item.available ? Color.green : Color.secondary)
                        Text(item.name)
                        Spacer()
                        Text(item.hint)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                }
            }
        }
    }
}

struct ServiceLogSheet: View {
    @ObservedObject var service: ServiceController
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Service log")
                .font(.headline)
            if case .failed(let message) = service.state {
                Text(message)
                    .foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
            }
            ScrollView {
                Text(service.log.isEmpty ? "No output captured." : service.log.joined(separator: "\n"))
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(width: 560, height: 260)
            .background(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(Color(nsColor: .textBackgroundColor)))
            HStack {
                Spacer()
                Button("Done") { dismiss() }
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding(18)
    }
}
