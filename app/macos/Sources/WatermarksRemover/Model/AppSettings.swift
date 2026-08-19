import Foundation

/// User preferences, persisted in `UserDefaults`.
@MainActor
final class AppSettings: ObservableObject {
    private let defaults: UserDefaults

    @Published var pythonPath: String { didSet { store("pythonPath", pythonPath) } }
    @Published var useExternalService: Bool { didSet { store("useExternalService", useExternalService) } }
    @Published var externalServiceURL: String { didSet { store("externalServiceURL", externalServiceURL) } }
    @Published var externalAPIKey: String { didSet { store("externalAPIKey", externalAPIKey) } }
    @Published var outputMode: OutputMode { didSet { store("outputMode", outputMode.rawValue) } }
    @Published var outputFolderPath: String { didSet { store("outputFolderPath", outputFolderPath) } }
    @Published var runDetectorsOnInspect: Bool { didSet { store("runDetectorsOnInspect", runDetectorsOnInspect) } }
    @Published var autoInspectOnAdd: Bool { didSet { store("autoInspectOnAdd", autoInspectOnAdd) } }

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        self.pythonPath = defaults.string(forKey: "pythonPath") ?? ""
        self.useExternalService = defaults.bool(forKey: "useExternalService")
        self.externalServiceURL =
            defaults.string(forKey: "externalServiceURL") ?? "http://127.0.0.1:8765"
        self.externalAPIKey = defaults.string(forKey: "externalAPIKey") ?? ""
        self.outputMode =
            OutputMode(rawValue: defaults.string(forKey: "outputMode") ?? "") ?? .sibling
        self.outputFolderPath = defaults.string(forKey: "outputFolderPath") ?? ""
        self.runDetectorsOnInspect = defaults.bool(forKey: "runDetectorsOnInspect")
        self.autoInspectOnAdd = defaults.object(forKey: "autoInspectOnAdd") as? Bool ?? true
    }

    var outputFolder: URL? {
        outputFolderPath.isEmpty ? nil : URL(fileURLWithPath: outputFolderPath)
    }

    private func store(_ key: String, _ value: Any) {
        defaults.set(value, forKey: key)
    }
}
