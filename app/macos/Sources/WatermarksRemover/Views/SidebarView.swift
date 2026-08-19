import AppKit
import SwiftUI

struct SidebarView: View {
    @EnvironmentObject private var model: AppModel
    @Binding var mode: WorkspaceMode

    var body: some View {
        VStack(spacing: 0) {
            Picker("", selection: $mode) {
                ForEach(WorkspaceMode.allCases) { value in
                    Text(value.label).tag(value)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .padding(10)

            if mode == .files {
                fileList
            } else {
                scratchpadHint
            }
        }
        .frame(minWidth: 260)
    }

    @ViewBuilder
    private var fileList: some View {
        if model.items.isEmpty {
            DropZone()
                .padding(12)
        } else {
            List(selection: $model.selection) {
                ForEach(model.items) { item in
                    FileRow(item: item)
                        .tag(item.id)
                        .contextMenu {
                            Button("Reveal in Finder") { model.reveal(item.url) }
                            if let output = item.outputURL {
                                Button("Reveal Cleaned File") { model.reveal(output) }
                            }
                            Divider()
                            Button("Remove from List") { model.remove(ids: [item.id]) }
                        }
                }
            }
            .listStyle(.sidebar)
            .onDeleteCommand { model.remove(ids: model.selection) }
            .overlay(alignment: .bottom) {
                DropHintBar()
            }
        }
    }

    private var scratchpadHint: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Text scratchpad")
                .font(.headline)
            Text(
                "Paste anything you are about to publish. The scratchpad runs the "
                    + "same Layer A pass as the file queue, without touching disk."
            )
            .font(.callout)
            .foregroundStyle(.secondary)
            Spacer()
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

enum WorkspaceMode: String, CaseIterable, Identifiable {
    case files
    case text

    var id: String { rawValue }

    var label: String {
        switch self {
        case .files: return "Files"
        case .text: return "Text"
        }
    }
}

struct FileRow: View {
    let item: FileItem

    var body: some View {
        HStack(spacing: 9) {
            StatusDot(item: item)
            VStack(alignment: .leading, spacing: 2) {
                Text(item.name)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text(item.subtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
            Spacer(minLength: 4)
            if item.status.isBusy {
                ProgressView()
                    .controlSize(.small)
            } else if let inspection = item.inspection {
                Image(systemName: inspection.kind.symbol)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 3)
    }
}

/// The always-available "drop more files here" strip under a populated list.
private struct DropHintBar: View {
    @EnvironmentObject private var model: AppModel
    @State private var targeted = false

    var body: some View {
        Text(targeted ? "Release to add" : "Drop more files here")
            .font(.caption)
            .foregroundStyle(.secondary)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 7)
            .background(.ultraThinMaterial)
            .overlay(alignment: .top) { Divider() }
            .dropDestination(for: URL.self) { urls, _ in
                model.add(urls: urls)
                return true
            } isTargeted: { targeted = $0 }
    }
}

/// The empty-state target: the app's front door.
struct DropZone: View {
    @EnvironmentObject private var model: AppModel
    @State private var targeted = false

    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: "square.and.arrow.down.on.square")
                .font(.system(size: 34, weight: .light))
                .foregroundStyle(targeted ? Color.accentColor : .secondary)
            VStack(spacing: 4) {
                Text("Drop files or folders")
                    .font(.headline)
                Text("Text, images, documents, audio and video")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
            Button("Choose Files…") { chooseFiles() }
                .buttonStyle(.borderedProminent)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(20)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Color.accentColor.opacity(targeted ? 0.12 : 0.04))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(
                    targeted ? Color.accentColor : Color.secondary.opacity(0.35),
                    style: StrokeStyle(lineWidth: 1.5, dash: [6, 5]))
        )
        .dropDestination(for: URL.self) { urls, _ in
            model.add(urls: urls)
            return true
        } isTargeted: { targeted = $0 }
    }

    private func chooseFiles() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = true
        panel.canChooseFiles = true
        panel.message = "Select the files you want to inspect or clean."
        if panel.runModal() == .OK {
            model.add(urls: panel.urls)
        }
    }
}
