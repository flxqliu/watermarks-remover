import Foundation

/// Holds the running service process so it can be killed from anywhere,
/// including `applicationWillTerminate`, without hopping to the main actor.
///
/// A leaked interpreter would keep listening on a loopback port after the app
/// is gone, which is exactly the kind of surprise a privacy tool should not
/// leave behind.
final class ProcessRegistry: @unchecked Sendable {
    static let shared = ProcessRegistry()

    private let lock = NSLock()
    private var processes: [Process] = []

    func register(_ process: Process) {
        lock.lock()
        defer { lock.unlock() }
        processes.removeAll { !$0.isRunning }
        processes.append(process)
    }

    func forget(_ process: Process) {
        lock.lock()
        defer { lock.unlock() }
        processes.removeAll { $0 === process }
    }

    func terminateAll() {
        lock.lock()
        let running = processes
        processes.removeAll()
        lock.unlock()
        for process in running where process.isRunning {
            process.terminate()
        }
    }
}
