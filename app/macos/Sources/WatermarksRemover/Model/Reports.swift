import Foundation

/// Which pipeline the service routed a file through.
enum FileKind: String, Codable {
    case text
    case image
    case container
    case av
    case unknown

    var label: String {
        switch self {
        case .text: return "Text"
        case .image: return "Image"
        case .container: return "Document"
        case .av: return "Audio / Video"
        case .unknown: return "Unrecognized"
        }
    }

    var symbol: String {
        switch self {
        case .text: return "doc.plaintext"
        case .image: return "photo"
        case .container: return "doc.richtext"
        case .av: return "waveform"
        case .unknown: return "questionmark.square.dashed"
        }
    }

    init(raw: String?) {
        self = FileKind(rawValue: raw ?? "") ?? .unknown
    }
}

/// How strongly a finding implicates AI provenance. Mirrors the service's own
/// `findings_confidence` / per-hit `confidence` strings.
enum Confidence: String {
    case high
    case medium
    case low
    case unknown

    init(raw: String?) {
        self = Confidence(rawValue: (raw ?? "").lowercased()) ?? .unknown
    }

    var label: String {
        switch self {
        case .high: return "High"
        case .medium: return "Medium"
        case .low: return "Low"
        case .unknown: return "—"
        }
    }
}

struct LayerAHit: Identifiable {
    let id = UUID()
    let codepoint: String
    let label: String
    let count: Int
    let kind: String
    let confidence: Confidence

    init?(json: JSONValue) {
        guard let label = json["label"]?.stringValue else { return nil }
        self.codepoint = json["codepoint"]?.stringValue ?? ""
        self.label = label
        self.count = json["count"]?.intValue ?? 0
        self.kind = json["kind"]?.stringValue ?? ""
        self.confidence = Confidence(raw: json["confidence"]?.stringValue)
    }
}

struct Finding: Identifiable {
    let id = UUID()
    let text: String
    let confidence: Confidence
}

struct StylometrySummary {
    let score: Double
    let confidenceLevel: String
    let status: String
    let wordCount: Int
    let burstiness: Double?
    let lexicalDiversity: Double
    let markers: [String]

    init?(json: JSONValue?) {
        guard let json, let score = json["score"]?.doubleValue else { return nil }
        self.score = score
        self.confidenceLevel = json["confidence_level"]?.stringValue ?? "unknown"
        self.status = json["status"]?.stringValue ?? ""
        self.wordCount = json["word_count"]?.intValue ?? 0
        self.burstiness = json["burstiness_cv"]?.doubleValue
        self.lexicalDiversity = json["lexical_diversity"]?.doubleValue ?? 0
        self.markers = (json["matched_markers"]?.arrayValue ?? []).compactMap {
            $0["phrase"]?.stringValue ?? $0.stringValue
        }
    }
}

struct DetectorSummary: Identifiable {
    let id = UUID()
    let name: String
    let available: Bool
    let watermarked: Bool?
    let detail: String?

    init(json: JSONValue) {
        self.name = json["detector"]?.stringValue ?? "detector"
        self.available = json["available"]?.boolValue ?? false
        self.watermarked = json["is_watermarked"]?.boolValue
        if let error = json["error"]?.stringValue {
            self.detail = error
        } else if let score = json["score"]?.doubleValue {
            self.detail = String(format: "score %.3f", score)
        } else if let pvalue = json["p_value"]?.doubleValue {
            self.detail = String(format: "p = %.4f", pvalue)
        } else {
            self.detail = json["status"]?.stringValue
        }
    }
}

/// Everything the detail pane renders, distilled from one `/inspect` report.
struct InspectSummary {
    let kind: FileKind
    let suspicious: Bool
    let layerATotal: Int
    let hits: [LayerAHit]
    let findings: [Finding]
    let notes: [String]
    let hasC2PA: Bool
    let hasAIMetadata: Bool
    let format: String?
    let stylometry: StylometrySummary?
    let detectors: [DetectorSummary]
    let raw: JSONValue

    init(kind: FileKind, suspicious: Bool, report: JSONValue) {
        self.kind = kind
        self.suspicious = suspicious
        self.raw = report
        self.layerATotal = report["suspicious_total"]?.intValue ?? 0
        self.hasC2PA = report["has_c2pa"]?.boolValue ?? false
        self.hasAIMetadata = report["has_ai_metadata"]?.boolValue ?? false
        self.format = report["format"]?.stringValue
        self.notes = report["notes"]?.stringArray ?? []
        self.stylometry = StylometrySummary(json: report["stylometry"])

        // Layer-A hits live under "hits" for text and "layer_a_hits" for
        // containers that also carry a text body.
        let hitSource = report["hits"] ?? report["layer_a_hits"]
        self.hits = (hitSource?.arrayValue ?? []).compactMap(LayerAHit.init(json:))

        let findingTexts = report["findings"]?.stringArray ?? []
        let confidences = report["findings_confidence"]?.stringArray ?? []
        self.findings = findingTexts.enumerated().map { index, text in
            Finding(
                text: text,
                confidence: Confidence(raw: index < confidences.count ? confidences[index] : nil))
        }

        let detectorSource = report["text_detectors"]?.arrayValue ?? []
        self.detectors = detectorSource.map(DetectorSummary.init(json:))
    }

    /// One-line verdict for the file row.
    var headline: String {
        if !suspicious { return "No AI provenance marks found" }
        var parts: [String] = []
        if layerATotal > 0 { parts.append("\(layerATotal) hidden character\(layerATotal == 1 ? "" : "s")") }
        if hasC2PA { parts.append("C2PA manifest") }
        if hasAIMetadata { parts.append("AI metadata") }
        if let stylometry, stylometry.score >= 0.65 {
            parts.append(String(format: "stylometry %.2f", stylometry.score))
        }
        if detectors.contains(where: { $0.available && $0.watermarked == true }) {
            parts.append("detector hit")
        }
        if parts.isEmpty { parts.append("suspicious") }
        return parts.joined(separator: " · ")
    }
}

/// Everything the detail pane renders after a `/clean` round-trip.
struct CleanSummary {
    let kind: FileKind
    let actions: [String]
    let removedCount: Int
    let replacedCount: Int
    let removedLabels: [(String, Int)]
    let bytesIn: Int?
    let bytesOut: Int?
    let stillHasC2PA: Bool
    let stillHasAIMetadata: Bool
    let raw: JSONValue

    init(kind: FileKind, report: JSONValue) {
        self.kind = kind
        self.raw = report
        self.actions = report["actions"]?.stringArray ?? []
        self.bytesIn = report["bytes_in"]?.intValue
        self.bytesOut = report["bytes_out"]?.intValue
        self.stillHasC2PA = report["still_has_c2pa"]?.boolValue ?? false
        self.stillHasAIMetadata = report["still_has_ai_metadata"]?.boolValue ?? false

        let stats = report["stats"]
        self.removedCount = stats?["removed_count"]?.intValue ?? 0
        self.replacedCount = stats?["replaced_count"]?.intValue ?? 0
        let removed = stats?["removed"]?.objectValue ?? [:]
        let replaced = stats?["replaced"]?.objectValue ?? [:]
        self.removedLabels = (removed.merging(replaced) { lhs, _ in lhs })
            .compactMap { key, value in value.intValue.map { (key, $0) } }
            .sorted { lhs, rhs in lhs.1 == rhs.1 ? lhs.0 < rhs.0 : lhs.1 > rhs.1 }
    }

    var headline: String {
        var parts: [String] = []
        if removedCount > 0 { parts.append("\(removedCount) removed") }
        if replacedCount > 0 { parts.append("\(replacedCount) replaced") }
        if !actions.isEmpty { parts.append("\(actions.count) action\(actions.count == 1 ? "" : "s")") }
        if parts.isEmpty { parts.append("Nothing to change") }
        if stillHasC2PA || stillHasAIMetadata { parts.append("residue remains") }
        return parts.joined(separator: " · ")
    }
}
