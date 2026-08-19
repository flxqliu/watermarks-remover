import Foundation

/// One file in the queue, with whatever the service has said about it so far.
struct FileItem: Identifiable {
    enum Status: Equatable {
        case queued
        case working(String)
        case inspected
        case cleaned
        case failed(String)

        var isBusy: Bool {
            if case .working = self { return true }
            return false
        }
    }

    let id = UUID()
    let url: URL
    var byteCount: Int
    var status: Status = .queued
    var inspection: InspectSummary?
    var cleanResult: CleanSummary?
    var outputURL: URL?

    var name: String { url.lastPathComponent }

    var kind: FileKind { inspection?.kind ?? FileKind(raw: nil) }

    /// The row's short status line.
    var subtitle: String {
        switch status {
        case .queued: return ByteCountFormatter.string(fromByteCount: Int64(byteCount), countStyle: .file)
        case .working(let phase): return phase
        case .inspected: return inspection?.headline ?? "Inspected"
        case .cleaned: return cleanResult?.headline ?? "Cleaned"
        case .failed(let message): return message
        }
    }

    var isSuspicious: Bool { inspection?.suspicious ?? false }
}
