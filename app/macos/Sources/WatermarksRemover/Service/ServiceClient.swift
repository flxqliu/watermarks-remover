import Foundation

enum ServiceError: LocalizedError {
    case tooLarge(Int)
    case badResponse
    case server(String)
    case http(Int)

    var errorDescription: String? {
        switch self {
        case .tooLarge(let limit):
            return "The file is larger than the service's \(limit / (1 << 20)) MB limit."
        case .badResponse:
            return "The service returned a response the app could not read."
        case .server(let message):
            return message
        case .http(let status):
            return "The service answered with HTTP \(status)."
        }
    }
}

struct InspectResult {
    let kind: FileKind
    let suspicious: Bool
    let report: JSONValue
}

struct CleanResult {
    let kind: FileKind
    let cleaned: Data
    let report: JSONValue
}

struct DetectResult {
    let kind: FileKind
    let detections: [DetectorSummary]
    let raw: JSONValue
}

/// A typed client for the endpoints documented in `service/scripts/server.py`.
struct ServiceClient {
    /// Mirrors `common.MAX_INPUT_BYTES`; requests above it are rejected before
    /// the app spends time base64-encoding a payload the service will refuse.
    static let maxInputBytes = 256 << 20

    let endpoint: ServiceEndpoint
    let session: URLSession

    // MARK: - GET

    func health() async throws -> String {
        let payload = try await get("/health")
        return payload["version"]?.stringValue ?? "unknown"
    }

    func capabilities() async throws -> Capabilities {
        Capabilities(json: try await get("/capabilities"))
    }

    // MARK: - POST

    func inspect(data: Data, name: String, detect: Bool) async throws -> InspectResult {
        var body = try envelope(data: data, name: name)
        if detect { body["detect"] = .bool(true) }
        let payload = try await post("/inspect", body: body)
        return InspectResult(
            kind: FileKind(raw: payload["kind"]?.stringValue),
            suspicious: payload["suspicious"]?.boolValue ?? false,
            report: payload["report"] ?? .object([:]))
    }

    func clean(data: Data, name: String, options: CleanOptions) async throws -> CleanResult {
        var body = try envelope(data: data, name: name)
        body["options"] = .object(options.wireOptions)
        let payload = try await post("/clean", body: body)
        guard let encoded = payload["cleaned"]?.stringValue,
            let cleaned = Data(base64Encoded: encoded)
        else { throw ServiceError.badResponse }
        return CleanResult(
            kind: FileKind(raw: payload["kind"]?.stringValue),
            cleaned: cleaned,
            report: payload["report"] ?? .object([:]))
    }

    func detect(data: Data, name: String) async throws -> DetectResult {
        let payload = try await post("/detect", body: try envelope(data: data, name: name))
        let detections = (payload["detections"]?.arrayValue ?? []).map(DetectorSummary.init(json:))
        return DetectResult(
            kind: FileKind(raw: payload["kind"]?.stringValue),
            detections: detections,
            raw: payload)
    }

    // MARK: - Plumbing

    private func envelope(data: Data, name: String) throws -> [String: JSONValue] {
        guard data.count <= Self.maxInputBytes else {
            throw ServiceError.tooLarge(Self.maxInputBytes)
        }
        return [
            "file": .string(data.base64EncodedString()),
            "name": .string(name),
        ]
    }

    /// Joins `path` onto the base URL by string, because
    /// `appendingPathComponent` treats a multi-segment path as one component.
    private func url(for path: String) -> URL {
        var base = endpoint.baseURL.absoluteString
        while base.hasSuffix("/") { base.removeLast() }
        return URL(string: base + path) ?? endpoint.baseURL
    }

    private func request(_ path: String) -> URLRequest {
        var request = URLRequest(url: url(for: path))
        if let apiKey = endpoint.apiKey {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        return request
    }

    private func get(_ path: String) async throws -> JSONValue {
        try await send(request(path))
    }

    private func post(_ path: String, body: [String: JSONValue]) async throws -> JSONValue {
        var request = self.request(path)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(JSONValue.object(body))
        return try await send(request)
    }

    private func send(_ request: URLRequest) async throws -> JSONValue {
        let (data, response) = try await session.data(for: request)
        let payload = try? JSONDecoder().decode(JSONValue.self, from: data)
        if let payload, payload["ok"]?.boolValue == false {
            throw ServiceError.server(payload["error"]?.stringValue ?? "the service refused the request")
        }
        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            throw ServiceError.http(http.statusCode)
        }
        guard let payload else { throw ServiceError.badResponse }
        return payload
    }
}
