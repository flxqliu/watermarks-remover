import Foundation

/// The service's `/capabilities` answer, reduced to what the UI surfaces.
struct Capabilities {
    struct Item: Identifiable {
        let id = UUID()
        let name: String
        let available: Bool
        let hint: String
    }

    let version: String
    let tools: [Item]
    let backends: [Item]
    let detectors: [Item]

    init(json: JSONValue) {
        self.version = json["version"]?.stringValue ?? "unknown"

        let toolHints = [
            "c2patool": "Reads C2PA manifests",
            "exiftool": "Strips residual metadata (needed for PDF)",
            "qpdf": "Structural PDF rebuild",
        ]
        self.tools = (json["tools"]?.objectValue ?? [:])
            .map { key, value in
                Item(
                    name: key, available: value.boolValue ?? false,
                    hint: toolHints[key] ?? "Optional system tool")
            }
            .sorted { $0.name < $1.name }

        var backends: [Item] = []
        for (key, value) in json["pixel_backends"]?.objectValue ?? [:] {
            backends.append(
                Item(
                    name: key, available: value.boolValue ?? false,
                    hint: "Pixel-level watermark removal"))
        }
        for (key, value) in json["scorers"]?.objectValue ?? [:] {
            backends.append(
                Item(
                    name: key, available: value.boolValue ?? false, hint: "Watermark scorer"))
        }
        self.backends = backends.sorted { $0.name < $1.name }

        self.detectors = (json["text_detectors"]?.objectValue ?? [:])
            .map { key, value in
                Item(
                    name: key, available: value.boolValue ?? false,
                    hint: "Text watermark detector")
            }
            .sorted { $0.name < $1.name }
    }

    var pixelBackendAvailable: Bool {
        backends.contains { ($0.name == "ctrlregen" || $0.name == "diffusion") && $0.available }
    }

    var anyDetectorAvailable: Bool {
        detectors.contains { $0.available }
    }
}
