import Foundation

struct ServiceEndpoint: Equatable {
    let baseURL: URL
    let apiKey: String?
}

enum ServiceState: Equatable {
    case idle
    case starting
    case running(ServiceEndpoint)
    case failed(String)

    var endpoint: ServiceEndpoint? {
        if case .running(let endpoint) = self { return endpoint }
        return nil
    }

    var isRunning: Bool { endpoint != nil }

    var label: String {
        switch self {
        case .idle: return "Stopped"
        case .starting: return "Starting…"
        case .running: return "Ready"
        case .failed: return "Unavailable"
        }
    }
}

/// Owns the local `service/scripts/server.py` process.
///
/// The app never shells out to the CLI scripts one file at a time: it starts
/// the same HTTP service the agent skill talks to, on a private loopback port
/// with a per-launch bearer token, and speaks the documented API to it. That
/// keeps the app a thin client over the pipeline the repo already tests.
@MainActor
final class ServiceController: ObservableObject {
    @Published private(set) var state: ServiceState = .idle
    @Published private(set) var interpreter: PythonInterpreter?
    @Published private(set) var capabilities: Capabilities?
    @Published private(set) var log: [String] = []

    private var process: Process?
    private let session: URLSession

    /// Where the service scripts live inside the app bundle, with a fallback to
    /// the checkout for `swift run` during development.
    static var bundledScriptURL: URL? {
        if let resource = Bundle.main.resourceURL {
            let bundled = resource.appendingPathComponent("service/scripts/server.py")
            if FileManager.default.fileExists(atPath: bundled.path) { return bundled }
        }
        // swift run leaves the binary in .build/<config>/; walk up to the repo.
        var directory = URL(fileURLWithPath: CommandLine.arguments[0])
            .resolvingSymlinksInPath()
            .deletingLastPathComponent()
        for _ in 0..<8 {
            let candidate = directory.appendingPathComponent("service/scripts/server.py")
            if FileManager.default.fileExists(atPath: candidate.path) { return candidate }
            directory = directory.deletingLastPathComponent()
        }
        return nil
    }

    init() {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 600
        configuration.timeoutIntervalForResource = 1800
        configuration.waitsForConnectivity = false
        self.session = URLSession(configuration: configuration)
    }

    var client: ServiceClient? {
        state.endpoint.map { ServiceClient(endpoint: $0, session: session) }
    }

    // MARK: - Lifecycle

    func start(settings: AppSettings) async {
        guard !state.isRunning, state != .starting else { return }
        state = .starting
        log.removeAll()

        if settings.useExternalService {
            await attachToExternalService(settings: settings)
            return
        }

        guard let script = Self.bundledScriptURL else {
            state = .failed(
                "The bundled service scripts are missing. Rebuild the app with "
                    + "app/macos/Scripts/build-app.sh.")
            return
        }
        guard let python = PythonLocator.discover(preferred: settings.pythonPath) else {
            state = .failed(
                "No Python 3.10+ interpreter found. Install one with `brew install python` "
                    + "or `xcode-select --install`, then set its path in Settings.")
            return
        }
        interpreter = python

        guard let port = PortFinder.freeLoopbackPort() else {
            state = .failed("Could not reserve a loopback port.")
            return
        }
        let apiKey = Self.makeAPIKey()

        let process = Process()
        process.executableURL = python.url
        // --host/--port are safe in argv; the API key goes through the
        // environment instead, so it never shows up in `ps` output.
        process.arguments = [script.path, "--host", "127.0.0.1", "--port", String(port)]
        var environment = ProcessInfo.processInfo.environment
        environment["WATERMARKS_SERVER_API_KEY"] = apiKey
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        // A GUI app inherits a bare PATH, so exiftool/qpdf/c2patool installed by
        // Homebrew would look absent to the capability probe without this.
        let extraPaths = ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin"]
        let existing = environment["PATH"].map { $0.split(separator: ":").map(String.init) } ?? []
        environment["PATH"] = (existing + extraPaths.filter { !existing.contains($0) })
            .joined(separator: ":")
        process.environment = environment

        let errorPipe = Pipe()
        process.standardError = errorPipe
        // Nothing reads stdout, and an undrained pipe would eventually block
        // the child; stderr is the stream the service actually logs to.
        process.standardOutput = FileHandle.nullDevice
        errorPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            Task { @MainActor in self?.appendLog(text) }
        }

        do {
            try process.run()
        } catch {
            state = .failed("Could not launch the service: \(error.localizedDescription)")
            return
        }
        self.process = process
        ProcessRegistry.shared.register(process)
        process.terminationHandler = { finished in
            Task { @MainActor [weak self] in
                self?.serviceDidExit(finished)
            }
        }

        let endpoint = ServiceEndpoint(
            baseURL: URL(string: "http://127.0.0.1:\(port)")!, apiKey: apiKey)
        if await waitForHealth(endpoint: endpoint, process: process) {
            state = .running(endpoint)
            await refreshCapabilities()
        } else {
            let tail = log.suffix(6).joined(separator: "\n")
            stop()
            state = .failed(
                "The service did not become healthy."
                    + (tail.isEmpty ? "" : "\n\n\(tail)"))
        }
    }

    private func attachToExternalService(settings: AppSettings) async {
        guard let url = URL(string: settings.externalServiceURL),
            url.scheme != nil, url.host != nil
        else {
            state = .failed("\(settings.externalServiceURL) is not a valid service URL.")
            return
        }
        let key = settings.externalAPIKey.isEmpty ? nil : settings.externalAPIKey
        let endpoint = ServiceEndpoint(baseURL: url, apiKey: key)
        let client = ServiceClient(endpoint: endpoint, session: session)
        do {
            _ = try await client.health()
            state = .running(endpoint)
            await refreshCapabilities()
        } catch {
            state = .failed("Could not reach \(url.absoluteString): \(error.localizedDescription)")
        }
    }

    func stop() {
        if let process {
            (process.standardError as? Pipe)?.fileHandleForReading.readabilityHandler = nil
            if process.isRunning { process.terminate() }
            ProcessRegistry.shared.forget(process)
        }
        process = nil
        capabilities = nil
        state = .idle
    }

    /// Called when the interpreter exits on its own — a crash, a port grab, a
    /// user killing it from Activity Monitor. Without this the UI would keep
    /// claiming the service is ready.
    private func serviceDidExit(_ finished: Process) {
        ProcessRegistry.shared.forget(finished)
        guard process === finished else { return }
        process = nil
        capabilities = nil
        guard state.isRunning else { return }
        let tail = log.suffix(4).joined(separator: "\n")
        state = .failed(
            "The service stopped unexpectedly (exit \(finished.terminationStatus))."
                + (tail.isEmpty ? "" : "\n\n\(tail)"))
    }

    func restart(settings: AppSettings) async {
        stop()
        await start(settings: settings)
    }

    func refreshCapabilities() async {
        guard let client else { return }
        capabilities = try? await client.capabilities()
    }

    // MARK: - Helpers

    private func waitForHealth(endpoint: ServiceEndpoint, process: Process) async -> Bool {
        let client = ServiceClient(endpoint: endpoint, session: session)
        let deadline = Date().addingTimeInterval(20)
        while Date() < deadline {
            if !process.isRunning { return false }
            if (try? await client.health()) != nil { return true }
            try? await Task.sleep(nanoseconds: 200_000_000)
        }
        return false
    }

    private func appendLog(_ text: String) {
        let lines = text.split(separator: "\n", omittingEmptySubsequences: true).map(String.init)
        log.append(contentsOf: lines)
        if log.count > 200 { log.removeFirst(log.count - 200) }
    }

    private static func makeAPIKey() -> String {
        var generator = SystemRandomNumberGenerator()
        return (0..<4)
            .map { _ in String(UInt64.random(in: UInt64.min...UInt64.max, using: &generator), radix: 16) }
            .joined()
    }
}
