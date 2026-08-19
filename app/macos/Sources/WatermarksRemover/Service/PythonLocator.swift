import Foundation

struct PythonInterpreter: Equatable {
    let url: URL
    let version: String
}

/// Finds a Python 3.10+ interpreter for the bundled service.
///
/// The service is stdlib-only, so any modern interpreter works. Homebrew comes
/// first because `/usr/bin/python3` is a Command Line Tools stub on a machine
/// where they were never installed — running it pops an installer dialog and
/// exits non-zero, which the probe treats as "not available".
enum PythonLocator {
    static let searchPaths = [
        "/opt/homebrew/bin/python3",
        "/usr/local/bin/python3",
        "/opt/homebrew/bin/python3.13",
        "/opt/homebrew/bin/python3.12",
        "/opt/homebrew/bin/python3.11",
        "/usr/bin/python3",
    ]

    static func discover(preferred: String?) -> PythonInterpreter? {
        var candidates: [String] = []
        if let preferred, !preferred.trimmingCharacters(in: .whitespaces).isEmpty {
            candidates.append(preferred)
        }
        candidates.append(contentsOf: searchPaths)
        for candidate in candidates {
            if let interpreter = probe(path: candidate) { return interpreter }
        }
        return nil
    }

    /// Runs the candidate and keeps it only if it reports 3.10 or newer.
    static func probe(path: String) -> PythonInterpreter? {
        let url = URL(fileURLWithPath: path)
        guard FileManager.default.isExecutableFile(atPath: url.path) else { return nil }

        let process = Process()
        process.executableURL = url
        process.arguments = [
            "-c", "import sys; print('%d.%d.%d' % sys.version_info[:3])",
        ]
        let output = Pipe()
        process.standardOutput = output
        process.standardError = Pipe()
        do {
            try process.run()
        } catch {
            return nil
        }
        let data = output.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0,
            let text = String(data: data, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines)
        else { return nil }

        let parts = text.split(separator: ".").compactMap { Int($0) }
        guard parts.count >= 2, parts[0] == 3, parts[1] >= 10 else { return nil }
        return PythonInterpreter(url: url, version: text)
    }
}
