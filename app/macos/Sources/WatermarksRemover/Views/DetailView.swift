import SwiftUI

/// The right-hand pane: what the service found, and what it changed.
struct DetailView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        Group {
            if let item = model.selectedItem {
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        header(item)
                        if model.selection.count > 1 {
                            multiSelectionNotice
                        }
                        if case .failed(let message) = item.status {
                            failureCard(message)
                        }
                        if let inspection = item.inspection {
                            InspectionCards(inspection: inspection)
                        } else if !item.status.isBusy {
                            Card(title: "Not inspected yet", symbol: "clock") {
                                Text("Run Inspect to see what this file carries.")
                                    .foregroundStyle(.secondary)
                            }
                        }
                        if let cleanResult = item.cleanResult {
                            CleanCards(result: cleanResult, outputURL: item.outputURL)
                        }
                    }
                    .padding(18)
                }
            } else {
                emptyState
            }
        }
        .frame(minWidth: 420)
    }

    private func header(_ item: FileItem) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Image(systemName: item.kind.symbol)
                    .font(.title2)
                    .foregroundStyle(.tint)
                Text(item.name)
                    .font(.title2)
                    .fontWeight(.semibold)
                    .lineLimit(1)
                    .truncationMode(.middle)
                if item.status.isBusy { ProgressView().controlSize(.small) }
            }
            HStack(spacing: 8) {
                Chip(text: item.kind.label)
                Chip(
                    text: ByteCountFormatter.string(
                        fromByteCount: Int64(item.byteCount), countStyle: .file))
                if let inspection = item.inspection {
                    Chip(
                        text: inspection.suspicious ? "Marks found" : "Clean",
                        tint: inspection.suspicious ? .orange : .green)
                }
                Spacer()
                Button {
                    model.reveal(item.outputURL ?? item.url)
                } label: {
                    Label("Reveal", systemImage: "folder")
                }
                .buttonStyle(.borderless)
            }
            Text(item.url.deletingLastPathComponent().path)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.middle)
        }
    }

    private var multiSelectionNotice: some View {
        Label(
            "\(model.selection.count) files selected — actions apply to all of them.",
            systemImage: "square.stack.3d.up"
        )
        .font(.callout)
        .foregroundStyle(.secondary)
    }

    private func failureCard(_ message: String) -> some View {
        Card(title: "Failed", symbol: "exclamationmark.triangle") {
            Text(message)
                .foregroundStyle(.red)
                .textSelection(.enabled)
        }
    }

    private var emptyState: some View {
        VStack(spacing: 10) {
            Image(systemName: "shield.lefthalf.filled")
                .font(.system(size: 42, weight: .light))
                .foregroundStyle(.tint)
            Text("Nothing selected")
                .font(.title3)
                .fontWeight(.medium)
            Text("Add files on the left, or switch to the text scratchpad.")
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

/// Inspection results, one card per layer of the pipeline.
struct InspectionCards: View {
    let inspection: InspectSummary

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            summary
            if !inspection.hits.isEmpty { hitsCard }
            if !inspection.findings.isEmpty { findingsCard }
            if let stylometry = inspection.stylometry { stylometryCard(stylometry) }
            if !inspection.detectors.isEmpty { detectorsCard }
            if !inspection.notes.isEmpty { notesCard }
            Card(title: "Raw report", symbol: "curlybraces") {
                RawReportView(title: "Show the service's JSON", value: inspection.raw)
            }
        }
    }

    private var summary: some View {
        HStack(spacing: 10) {
            StatTile(
                value: "\(inspection.layerATotal)",
                caption: "Hidden characters",
                tint: inspection.layerATotal > 0 ? .orange : .green)
            StatTile(
                value: inspection.hasC2PA ? "Yes" : "No",
                caption: "C2PA manifest",
                tint: inspection.hasC2PA ? .orange : .green)
            StatTile(
                value: inspection.hasAIMetadata ? "Yes" : "No",
                caption: "AI metadata",
                tint: inspection.hasAIMetadata ? .orange : .green)
            if let stylometry = inspection.stylometry {
                StatTile(
                    value: String(format: "%.2f", stylometry.score),
                    caption: "Stylometry",
                    tint: stylometry.score >= 0.65 ? .orange : .green)
            }
        }
    }

    private var hitsCard: some View {
        Card(
            title: "Layer A — invisible characters", symbol: "eye.slash",
            accessory: "\(inspection.hits.count) kinds"
        ) {
            VStack(spacing: 0) {
                ForEach(inspection.hits) { hit in
                    HStack(spacing: 10) {
                        Text(hit.codepoint)
                            .font(.system(.caption, design: .monospaced))
                            .frame(width: 66, alignment: .leading)
                        Text(hit.label)
                            .lineLimit(1)
                        Spacer(minLength: 8)
                        Chip(text: hit.kind, tint: .secondary)
                        Chip(text: hit.confidence.label, tint: hit.confidence.tint)
                        Text("×\(hit.count)")
                            .font(.callout.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                    .padding(.vertical, 5)
                    if hit.id != inspection.hits.last?.id { Divider() }
                }
            }
        }
    }

    private var findingsCard: some View {
        Card(
            title: "Metadata findings", symbol: "tag",
            accessory: inspection.format.map { $0.uppercased() }
        ) {
            VStack(alignment: .leading, spacing: 7) {
                ForEach(inspection.findings) { finding in
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Chip(text: finding.confidence.label, tint: finding.confidence.tint)
                        Text(finding.text)
                            .textSelection(.enabled)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
        }
    }

    private func stylometryCard(_ stylometry: StylometrySummary) -> some View {
        Card(
            title: "Layer B — stylometry", symbol: "waveform.path.ecg",
            accessory: stylometry.status
        ) {
            VStack(alignment: .leading, spacing: 10) {
                Gauge(value: min(max(stylometry.score, 0), 1)) {
                    Text("LLM-likeness")
                } currentValueLabel: {
                    Text(String(format: "%.2f", stylometry.score))
                }
                .gaugeStyle(.accessoryLinearCapacity)
                .tint(stylometry.score >= 0.65 ? .orange : .green)

                HStack(spacing: 14) {
                    LabeledContent("Confidence", value: stylometry.confidenceLevel)
                    LabeledContent("Words", value: "\(stylometry.wordCount)")
                    LabeledContent(
                        "Burstiness",
                        value: stylometry.burstiness.map { String(format: "%.2f", $0) }
                            ?? "unmeasurable")
                    LabeledContent(
                        "Diversity", value: String(format: "%.2f", stylometry.lexicalDiversity))
                }
                .font(.callout)

                if !stylometry.markers.isEmpty {
                    Text("Marker phrases")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    FlowText(items: Array(stylometry.markers.prefix(24)))
                }
                Text(
                    "Stylometry is a heuristic on writing style, not a watermark detector. "
                        + "A high score is a prompt to rewrite (Layer B), never proof."
                )
                .font(.caption)
                .foregroundStyle(.secondary)
            }
        }
    }

    private var detectorsCard: some View {
        Card(title: "Watermark detectors", symbol: "antenna.radiowaves.left.and.right") {
            VStack(spacing: 0) {
                ForEach(inspection.detectors) { detector in
                    HStack {
                        Text(detector.name)
                        Spacer()
                        if !detector.available {
                            Chip(text: "not configured")
                        } else if let watermarked = detector.watermarked {
                            Chip(
                                text: watermarked ? "watermarked" : "no mark",
                                tint: watermarked ? .red : .green)
                        }
                        if let detail = detector.detail {
                            Text(detail)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                    .padding(.vertical, 4)
                    if detector.id != inspection.detectors.last?.id { Divider() }
                }
            }
        }
    }

    private var notesCard: some View {
        Card(title: "Notes", symbol: "info.circle") {
            VStack(alignment: .leading, spacing: 5) {
                ForEach(inspection.notes, id: \.self) { note in
                    Text("• \(note)")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }
}

/// What the clean pass changed.
struct CleanCards: View {
    let result: CleanSummary
    let outputURL: URL?
    @EnvironmentObject private var model: AppModel

    var body: some View {
        Card(title: "Cleaned", symbol: "sparkles", accessory: result.headline) {
            VStack(alignment: .leading, spacing: 12) {
                if result.removedCount > 0 || result.replacedCount > 0 {
                    HStack(spacing: 10) {
                        StatTile(value: "\(result.removedCount)", caption: "Removed", tint: .blue)
                        StatTile(value: "\(result.replacedCount)", caption: "Replaced", tint: .blue)
                        if let bytesIn = result.bytesIn, let bytesOut = result.bytesOut {
                            StatTile(
                                value: "\(bytesIn - bytesOut) B",
                                caption: "Size delta", tint: .blue)
                        }
                    }
                }
                if !result.removedLabels.isEmpty {
                    FlowText(items: result.removedLabels.map { "\($0.0) ×\($0.1)" })
                }
                if !result.actions.isEmpty {
                    VStack(alignment: .leading, spacing: 5) {
                        ForEach(result.actions, id: \.self) { action in
                            Label(action, systemImage: "checkmark.circle")
                                .font(.callout)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                }
                if result.stillHasC2PA || result.stillHasAIMetadata {
                    Label(
                        "The service still sees provenance data after cleaning. "
                            + "Install exiftool and qpdf for a full strip.",
                        systemImage: "exclamationmark.triangle"
                    )
                    .foregroundStyle(.orange)
                    .font(.callout)
                }
                if let outputURL {
                    HStack {
                        Text(outputURL.lastPathComponent)
                            .font(.callout)
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Spacer()
                        Button("Reveal") { model.reveal(outputURL) }
                            .buttonStyle(.borderless)
                    }
                }
                RawReportView(title: "Show the clean report", value: result.raw)
            }
        }
    }
}

/// Wrapping row of small labels. `Layout` keeps it to one pass and avoids the
/// nested-GeometryReader tricks these usually need.
struct FlowText: View {
    let items: [String]

    var body: some View {
        FlowLayout(spacing: 6) {
            ForEach(items, id: \.self) { item in
                Chip(text: item)
            }
        }
    }
}

struct FlowLayout: Layout {
    var spacing: CGFloat = 6

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let width = proposal.width ?? 400
        var rowWidth: CGFloat = 0
        var rowHeight: CGFloat = 0
        var total: CGFloat = 0
        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if rowWidth > 0, rowWidth + spacing + size.width > width {
                total += rowHeight + spacing
                rowWidth = size.width
                rowHeight = size.height
            } else {
                rowWidth += (rowWidth > 0 ? spacing : 0) + size.width
                rowHeight = max(rowHeight, size.height)
            }
        }
        return CGSize(width: width, height: total + rowHeight)
    }

    func placeSubviews(
        in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()
    ) {
        var x = bounds.minX
        var y = bounds.minY
        var rowHeight: CGFloat = 0
        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x > bounds.minX, x + size.width > bounds.maxX {
                x = bounds.minX
                y += rowHeight + spacing
                rowHeight = 0
            }
            subview.place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(size))
            x += size.width + spacing
            rowHeight = max(rowHeight, size.height)
        }
    }
}
