import AppKit
import SwiftUI

/// The clean-options popover hanging off the toolbar.
struct OptionsPopover: View {
    @EnvironmentObject private var model: AppModel
    @ObservedObject var settings: AppSettings
    let capabilities: Capabilities?

    var body: some View {
        Form {
            Section("Text") {
                Toggle("Unicode NFKC normalization", isOn: $model.options.nfkc)
                    .help("Folds compatibility forms. Changes more than invisible characters.")
                Toggle("Fold look-alike letters", isOn: $model.options.aggressiveHomoglyphs)
                    .help("Aggressive homoglyph mapping — can alter legitimate non-Latin text.")
                Toggle("Scrub text bodies inside documents", isOn: $model.options.alsoLayerAText)
                    .help("Runs the Layer A pass on Markdown and HTML bodies too.")
            }

            Section("Metadata") {
                Toggle("Keep non-AI metadata", isOn: $model.options.keepNonAIMetadata)
                    .help("Preserves camera and authoring metadata; strips only AI provenance.")
            }

            Section("Images") {
                Picker("Pixel watermark removal", selection: $model.options.pixelRemover) {
                    ForEach(PixelRemover.allCases) { remover in
                        Text(remover.label).tag(remover)
                    }
                }
                .disabled(capabilities?.pixelBackendAvailable != true)
                if capabilities?.pixelBackendAvailable != true {
                    Text("No pixel backend configured. See the repo's Docker profiles.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            Section("Detectors") {
                Toggle("Run detectors when inspecting", isOn: $settings.runDetectorsOnInspect)
                Toggle("Detect before cleaning", isOn: $model.options.detectBefore)
                Toggle("Detect after cleaning", isOn: $model.options.detectAfter)
                Text(
                    "Detectors are opt-in because some of them send your text to a vendor API."
                )
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
                        Text(
                            settings.outputFolderPath.isEmpty
                                ? "No folder chosen" : settings.outputFolderPath
                        )
                        .font(.caption)
                        .lineLimit(1)
                        .truncationMode(.head)
                        Spacer()
                        Button("Choose…") { chooseFolder() }
                    }
                }
            }
        }
        .formStyle(.grouped)
        .frame(width: 380, height: 480)
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
