import AppKit
import Foundation
import UniformTypeIdentifiers

/// Queue orchestration: turns dropped URLs into service calls and collects the
/// results the views render.
@MainActor
final class AppModel: ObservableObject {
    @Published var items: [FileItem] = []
    @Published var selection: Set<FileItem.ID> = []
    @Published var options = CleanOptions()
    @Published var isProcessing = false
    @Published var progress: (done: Int, total: Int)?
    @Published var alert: AppAlert?

    let settings: AppSettings
    let service: ServiceController

    /// Directories are walked, but never without a ceiling: a stray drop of a
    /// home folder should not enqueue a hundred thousand files.
    static let maxFilesPerDrop = 500

    init(settings: AppSettings, service: ServiceController) {
        self.settings = settings
        self.service = service
    }

    /// The detail pane follows the first selected row; multi-selection exists
    /// for batch actions, not for a split detail view.
    var selectedItem: FileItem? {
        items.first { selection.contains($0.id) }
    }

    var actionTargets: Set<FileItem.ID> {
        selection.isEmpty ? Set(items.map(\.id)) : selection
    }

    var suspiciousCount: Int { items.filter(\.isSuspicious).count }

    // MARK: - Queue

    func add(urls: [URL]) {
        let expanded = expand(urls: urls)
        var added = 0
        for url in expanded {
            let standardized = url.standardizedFileURL
            guard !items.contains(where: { $0.url.standardizedFileURL == standardized }) else {
                continue
            }
            let attributes = try? FileManager.default.attributesOfItem(atPath: standardized.path)
            let size = (attributes?[.size] as? NSNumber)?.intValue ?? 0
            items.append(FileItem(url: standardized, byteCount: size))
            added += 1
        }
        if selection.isEmpty, let first = items.first { selection = [first.id] }
        guard added > 0 else { return }
        if settings.autoInspectOnAdd {
            Task { await inspectPending() }
        }
    }

    func remove(ids: Set<FileItem.ID>) {
        items.removeAll { ids.contains($0.id) }
        selection.subtract(ids)
        if selection.isEmpty, let first = items.first { selection = [first.id] }
    }

    func clearAll() {
        items.removeAll()
        selection = []
    }

    private func expand(urls: [URL]) -> [URL] {
        var result: [URL] = []
        let manager = FileManager.default
        for url in urls {
            var isDirectory: ObjCBool = false
            guard manager.fileExists(atPath: url.path, isDirectory: &isDirectory) else { continue }
            if !isDirectory.boolValue {
                result.append(url)
                continue
            }
            let enumerator = manager.enumerator(
                at: url,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles, .skipsPackageDescendants])
            while let next = enumerator?.nextObject() as? URL {
                guard result.count < Self.maxFilesPerDrop else { break }
                let values = try? next.resourceValues(forKeys: [.isRegularFileKey])
                if values?.isRegularFile == true { result.append(next) }
            }
        }
        if result.count > Self.maxFilesPerDrop {
            alert = AppAlert(
                title: "Too many files",
                message: "Only the first \(Self.maxFilesPerDrop) files were added.")
            result = Array(result.prefix(Self.maxFilesPerDrop))
        }
        return result
    }

    // MARK: - Actions

    func inspectPending() async {
        await run(over: items.indices.filter { items[$0].inspection == nil }) { index, client in
            try await self.inspect(index: index, client: client)
        }
    }

    func inspectAll() async {
        await run(over: Array(items.indices)) { index, client in
            try await self.inspect(index: index, client: client)
        }
    }

    func cleanAll() async {
        await run(over: Array(items.indices)) { index, client in
            try await self.clean(index: index, client: client)
        }
    }

    func clean(ids: Set<FileItem.ID>) async {
        let targets = items.indices.filter { ids.contains(items[$0].id) }
        await run(over: targets) { index, client in
            try await self.clean(index: index, client: client)
        }
    }

    /// Shared driver: resolves the client once, walks the targets in order and
    /// records per-file failures without stopping the run.
    private func run(
        over targets: [Int],
        body: @escaping (Int, ServiceClient) async throws -> Void
    ) async {
        guard !targets.isEmpty else { return }
        guard let client = service.client else {
            alert = AppAlert(
                title: "Service not running",
                message: "Start the local service from the status bar, then try again.")
            return
        }
        isProcessing = true
        progress = (0, targets.count)
        defer {
            isProcessing = false
            progress = nil
        }
        for (offset, index) in targets.enumerated() {
            guard items.indices.contains(index) else { continue }
            do {
                try await body(index, client)
            } catch {
                if items.indices.contains(index) {
                    items[index].status = .failed(message(for: error))
                }
            }
            progress = (offset + 1, targets.count)
        }
    }

    private func inspect(index: Int, client: ServiceClient) async throws {
        let item = items[index]
        items[index].status = .working("Inspecting…")
        let data = try Data(contentsOf: item.url)
        items[index].byteCount = data.count
        let result = try await client.inspect(
            data: data, name: item.name, detect: settings.runDetectorsOnInspect)
        guard let current = items.firstIndex(where: { $0.id == item.id }) else { return }
        items[current].inspection = InspectSummary(
            kind: result.kind, suspicious: result.suspicious, report: result.report)
        items[current].status = .inspected
    }

    private func clean(index: Int, client: ServiceClient) async throws {
        let item = items[index]
        items[index].status = .working("Cleaning…")
        let data = try Data(contentsOf: item.url)
        let result = try await client.clean(data: data, name: item.name, options: options)
        let destination = try outputURL(for: item.url)
        try write(result.cleaned, to: destination)
        guard let current = items.firstIndex(where: { $0.id == item.id }) else { return }
        items[current].cleanResult = CleanSummary(kind: result.kind, report: result.report)
        items[current].outputURL = destination
        items[current].status = .cleaned
    }

    // MARK: - Output

    func outputURL(for source: URL) throws -> URL {
        switch settings.outputMode {
        case .inPlace:
            return source
        case .sibling:
            return uniqueURL(cleanedName(for: source, in: source.deletingLastPathComponent()))
        case .folder:
            guard let folder = settings.outputFolder else {
                throw AppError.noOutputFolder
            }
            try FileManager.default.createDirectory(
                at: folder, withIntermediateDirectories: true)
            return uniqueURL(cleanedName(for: source, in: folder))
        }
    }

    private func cleanedName(for source: URL, in folder: URL) -> URL {
        let base = source.deletingPathExtension().lastPathComponent
        let ext = source.pathExtension
        let name = ext.isEmpty ? "\(base).cleaned" : "\(base).cleaned.\(ext)"
        return folder.appendingPathComponent(name)
    }

    private func uniqueURL(_ candidate: URL) -> URL {
        var result = candidate
        var counter = 2
        let base = candidate.deletingPathExtension().lastPathComponent
        let ext = candidate.pathExtension
        let folder = candidate.deletingLastPathComponent()
        while FileManager.default.fileExists(atPath: result.path) {
            let name = ext.isEmpty ? "\(base) \(counter)" : "\(base) \(counter).\(ext)"
            result = folder.appendingPathComponent(name)
            counter += 1
        }
        return result
    }

    /// Writes through a temporary file in the destination directory so an
    /// interrupted write can never leave a half-cleaned file behind.
    private func write(_ data: Data, to destination: URL) throws {
        let folder = destination.deletingLastPathComponent()
        let temporary = folder.appendingPathComponent(".\(UUID().uuidString).tmp")
        try data.write(to: temporary, options: .atomic)
        if FileManager.default.fileExists(atPath: destination.path) {
            _ = try FileManager.default.replaceItemAt(destination, withItemAt: temporary)
        } else {
            try FileManager.default.moveItem(at: temporary, to: destination)
        }
    }

    // MARK: - Reveal / export

    func reveal(_ url: URL) {
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }

    func exportReport() {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "watermarks-report.json"
        panel.allowedContentTypes = [.json]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        let payload = JSONValue.array(
            items.map { item in
                var entry: [String: JSONValue] = [
                    "name": .string(item.name),
                    "path": .string(item.url.path),
                ]
                if let inspection = item.inspection {
                    entry["kind"] = .string(inspection.kind.rawValue)
                    entry["suspicious"] = .bool(inspection.suspicious)
                    entry["inspect_report"] = inspection.raw
                }
                if let cleanResult = item.cleanResult {
                    entry["clean_report"] = cleanResult.raw
                }
                if let output = item.outputURL {
                    entry["output"] = .string(output.path)
                }
                return .object(entry)
            })
        do {
            try Data(payload.prettyPrinted.utf8).write(to: url, options: .atomic)
        } catch {
            alert = AppAlert(title: "Could not save the report", message: error.localizedDescription)
        }
    }

    private func message(for error: Error) -> String {
        (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
    }
}

enum AppError: LocalizedError {
    case noOutputFolder

    var errorDescription: String? {
        switch self {
        case .noOutputFolder:
            return "Choose an output folder in Settings first."
        }
    }
}

struct AppAlert: Identifiable {
    let id = UUID()
    let title: String
    let message: String
}
