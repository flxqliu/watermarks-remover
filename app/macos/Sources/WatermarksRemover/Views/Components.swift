import SwiftUI

/// A titled panel used throughout the detail pane.
struct Card<Content: View>: View {
    let title: String
    var symbol: String?
    var accessory: String?
    @ViewBuilder let content: () -> Content

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 6) {
                if let symbol {
                    Image(systemName: symbol)
                        .foregroundStyle(.secondary)
                }
                Text(title)
                    .font(.headline)
                Spacer()
                if let accessory {
                    Text(accessory)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            content()
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color(nsColor: .controlBackgroundColor))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(Color.primary.opacity(0.07))
        )
    }
}

/// Small pill for a count or a label.
struct Chip: View {
    let text: String
    var tint: Color = .secondary

    var body: some View {
        Text(text)
            .font(.caption)
            .fontWeight(.medium)
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(tint.opacity(0.14), in: Capsule())
            .foregroundStyle(tint)
    }
}

extension Confidence {
    var tint: Color {
        switch self {
        case .high: return .red
        case .medium: return .orange
        case .low: return .yellow
        case .unknown: return .secondary
        }
    }
}

/// Headline number with a caption, used in the summary row.
struct StatTile: View {
    let value: String
    let caption: String
    var tint: Color = .primary

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(value)
                .font(.system(size: 22, weight: .semibold, design: .rounded))
                .foregroundStyle(tint)
            Text(caption)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.vertical, 10)
        .padding(.horizontal, 12)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(tint.opacity(0.08))
        )
    }
}

/// Collapsible pretty-printed JSON, so nothing the service reports is hidden.
struct RawReportView: View {
    let title: String
    let value: JSONValue
    @State private var expanded = false

    var body: some View {
        DisclosureGroup(isExpanded: $expanded) {
            ScrollView([.horizontal, .vertical]) {
                Text(value.prettyPrinted)
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)
                    .padding(8)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: 260)
            .background(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(Color(nsColor: .textBackgroundColor))
            )
        } label: {
            Text(title)
                .font(.subheadline)
        }
    }
}

/// A dot that reads at a glance in the file list.
struct StatusDot: View {
    let item: FileItem

    var body: some View {
        Circle()
            .fill(tint)
            .frame(width: 8, height: 8)
            .overlay(Circle().strokeBorder(Color.black.opacity(0.08)))
    }

    private var tint: Color {
        switch item.status {
        case .queued: return .secondary.opacity(0.5)
        case .working: return .blue
        case .failed: return .red
        case .inspected: return item.isSuspicious ? .orange : .green
        case .cleaned: return .green
        }
    }
}
