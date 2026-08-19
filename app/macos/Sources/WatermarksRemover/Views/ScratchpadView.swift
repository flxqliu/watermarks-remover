import AppKit
import SwiftUI

/// Paste-in text, inspected and cleaned entirely in memory.
struct ScratchpadView: View {
    @EnvironmentObject private var model: AppModel
    @State private var input = ""
    @State private var cleaned = ""
    @State private var inspection: InspectSummary?
    @State private var cleanSummary: CleanSummary?
    @State private var busy = false
    @State private var errorMessage: String?

    var body: some View {
        HSplitView {
            editorPane
            resultPane
        }
    }

    private var editorPane: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Draft")
                    .font(.headline)
                Spacer()
                Text("\(input.count) characters")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(12)

            TextEditor(text: $input)
                .font(.system(.body, design: .monospaced))
                .padding(6)

            Divider()

            HStack(spacing: 10) {
                Group {
                    Button {
                        Task { await inspect() }
                    } label: {
                        Label("Inspect", systemImage: "magnifyingglass")
                    }
                    Button {
                        Task { await clean() }
                    } label: {
                        Label("Clean", systemImage: "sparkles")
                    }
                    .buttonStyle(.borderedProminent)
                }
                .disabled(busy || input.isEmpty)
                if busy { ProgressView().controlSize(.small) }
                Spacer()
                Button("Paste") { paste() }
                    .disabled(busy)
                Button("Clear") { reset() }
                    .disabled(busy || input.isEmpty)
            }
            .padding(12)
        }
        .frame(minWidth: 320)
    }

    private var resultPane: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if let errorMessage {
                    Card(title: "Failed", symbol: "exclamationmark.triangle") {
                        Text(errorMessage).foregroundStyle(.red)
                    }
                }
                if let inspection {
                    InspectionCards(inspection: inspection)
                }
                if let cleanSummary {
                    Card(title: "Cleaned text", symbol: "text.badge.checkmark",
                         accessory: cleanSummary.headline) {
                        VStack(alignment: .leading, spacing: 10) {
                            HStack(spacing: 10) {
                                StatTile(
                                    value: "\(cleanSummary.removedCount)", caption: "Removed",
                                    tint: .blue)
                                StatTile(
                                    value: "\(cleanSummary.replacedCount)", caption: "Replaced",
                                    tint: .blue)
                            }
                            if !cleanSummary.removedLabels.isEmpty {
                                FlowText(items: cleanSummary.removedLabels.map { "\($0.0) ×\($0.1)" })
                            }
                            ScrollView {
                                Text(cleaned)
                                    .font(.system(.body, design: .monospaced))
                                    .textSelection(.enabled)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .padding(8)
                            }
                            .frame(maxHeight: 240)
                            .background(
                                RoundedRectangle(cornerRadius: 8, style: .continuous)
                                    .fill(Color(nsColor: .textBackgroundColor)))
                            HStack {
                                Button {
                                    copyCleaned()
                                } label: {
                                    Label("Copy cleaned text", systemImage: "doc.on.doc")
                                }
                                Button("Replace draft") { input = cleaned }
                            }
                        }
                    }
                }
                if inspection == nil && cleanSummary == nil && errorMessage == nil {
                    VStack(spacing: 8) {
                        Image(systemName: "text.magnifyingglass")
                            .font(.system(size: 34, weight: .light))
                            .foregroundStyle(.tint)
                        Text("Nothing inspected yet")
                            .font(.headline)
                        Text("Paste a draft on the left, then Inspect or Clean.")
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, minHeight: 260)
                }
            }
            .padding(18)
        }
        .frame(minWidth: 380)
    }

    // MARK: - Actions

    private func inspect() async {
        guard let client = model.service.client else {
            errorMessage = "The service is not running."
            return
        }
        busy = true
        defer { busy = false }
        errorMessage = nil
        do {
            let result = try await client.inspect(
                data: Data(input.utf8), name: "scratchpad.txt",
                detect: model.settings.runDetectorsOnInspect)
            inspection = InspectSummary(
                kind: result.kind, suspicious: result.suspicious, report: result.report)
        } catch {
            errorMessage = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
        }
    }

    private func clean() async {
        guard let client = model.service.client else {
            errorMessage = "The service is not running."
            return
        }
        busy = true
        defer { busy = false }
        errorMessage = nil
        do {
            let result = try await client.clean(
                data: Data(input.utf8), name: "scratchpad.txt", options: model.options)
            cleaned = String(decoding: result.cleaned, as: UTF8.self)
            cleanSummary = CleanSummary(kind: result.kind, report: result.report)
        } catch {
            errorMessage = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
        }
    }

    private func paste() {
        if let text = NSPasteboard.general.string(forType: .string) {
            input = text
        }
    }

    private func copyCleaned() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(cleaned, forType: .string)
    }

    private func reset() {
        input = ""
        cleaned = ""
        inspection = nil
        cleanSummary = nil
        errorMessage = nil
    }
}
