import Foundation

/// Where cleaned files are written.
enum OutputMode: String, CaseIterable, Identifiable {
    case sibling
    case folder
    case inPlace

    var id: String { rawValue }

    var label: String {
        switch self {
        case .sibling: return "Next to the original"
        case .folder: return "Into a folder"
        case .inPlace: return "Replace the original"
        }
    }

    var detail: String {
        switch self {
        case .sibling: return "Writes name.cleaned.ext beside each source file."
        case .folder: return "Writes every cleaned file into one chosen folder."
        case .inPlace: return "Overwrites the source file. There is no undo."
        }
    }
}

/// Which pixel-level remover to run, when a heavy backend is configured.
enum PixelRemover: String, CaseIterable, Identifiable {
    case none
    case ctrlregen
    case diffusion

    var id: String { rawValue }

    var label: String {
        switch self {
        case .none: return "Off"
        case .ctrlregen: return "CtrlRegen"
        case .diffusion: return "Diffusion purification"
        }
    }

    var wireValue: String? { self == .none ? nil : rawValue }
}

/// The `options` object of a `/clean` request, plus the app-side choices that
/// never reach the service.
struct CleanOptions: Equatable {
    var nfkc = false
    var aggressiveHomoglyphs = false
    var keepNonAIMetadata = false
    var alsoLayerAText = true
    var pixelRemover: PixelRemover = .none
    var detectBefore = false
    var detectAfter = false

    /// Only the keys the server accepts, and only the ones that differ from its
    /// own defaults, so requests stay minimal and forward-compatible.
    var wireOptions: [String: JSONValue] {
        var options: [String: JSONValue] = [
            "nfkc": .bool(nfkc),
            "aggressive_homoglyphs": .bool(aggressiveHomoglyphs),
            "keep_non_ai_metadata": .bool(keepNonAIMetadata),
            "also_layer_a_text": .bool(alsoLayerAText),
        ]
        if let remover = pixelRemover.wireValue {
            options["remove_pixel"] = .string(remover)
        }
        if detectBefore { options["detect_before"] = .bool(true) }
        if detectAfter { options["detect_after"] = .bool(true) }
        return options
    }
}
