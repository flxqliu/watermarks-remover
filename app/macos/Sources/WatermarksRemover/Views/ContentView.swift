import AppKit
import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var model: AppModel
    @ObservedObject var service: ServiceController
    @State private var mode: WorkspaceMode = .files
    @State private var showingOptions = false
    @State private var confirmingInPlace = false

    var body: some View {
        NavigationSplitView {
            SidebarView(mode: $mode)
        } detail: {
            Group {
                switch mode {
                case .files: DetailView()
                case .text: ScratchpadView()
                }
            }
            .frame(minWidth: 460, minHeight: 380)
        }
        .navigationTitle("Watermarks Remover")
        .toolbar { toolbarContent }
        .safeAreaInset(edge: .bottom, spacing: 0) {
            StatusFooter(service: service)
        }
        .alert(
            model.alert?.title ?? "",
            isPresented: Binding(
                get: { model.alert != nil },
                set: { if !$0 { model.alert = nil } }),
            presenting: model.alert
        ) { _ in
            Button("OK", role: .cancel) {}
        } message: { alert in
            Text(alert.message)
        }
        .confirmationDialog(
            "Replace the original files?",
            isPresented: $confirmingInPlace,
            titleVisibility: .visible
        ) {
            Button("Replace \(model.actionTargets.count) file(s)", role: .destructive) {
                Task { await model.clean(ids: model.actionTargets) }
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Cleaned bytes overwrite the source files. This cannot be undone.")
        }
        .task {
            if !service.state.isRunning {
                await service.start(settings: model.settings)
            }
        }
    }

    @ToolbarContentBuilder
    private var toolbarContent: some ToolbarContent {
        ToolbarItemGroup(placement: .navigation) {
            Button {
                addFiles()
            } label: {
                Label("Add Files", systemImage: "plus")
            }
            .help("Add files or folders to the queue")
            .disabled(mode == .text)
        }

        ToolbarItemGroup(placement: .primaryAction) {
            Button {
                Task { await model.inspectAll() }
            } label: {
                Label("Inspect", systemImage: "magnifyingglass")
            }
            .disabled(!canRun)

            Button {
                startClean()
            } label: {
                Label("Clean", systemImage: "sparkles")
            }
            .disabled(!canRun)

            Button {
                showingOptions = true
            } label: {
                Label("Options", systemImage: "slider.horizontal.3")
            }
            .popover(isPresented: $showingOptions, arrowEdge: .bottom) {
                OptionsPopover(settings: model.settings, capabilities: service.capabilities)
                    .environmentObject(model)
            }
        }
    }

    private var canRun: Bool {
        mode == .files && !model.items.isEmpty && !model.isProcessing && service.state.isRunning
    }

    private func startClean() {
        if model.settings.outputMode == .inPlace {
            confirmingInPlace = true
        } else {
            Task { await model.clean(ids: model.actionTargets) }
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
