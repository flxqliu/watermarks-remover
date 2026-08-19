import Darwin
import Foundation

/// Asks the kernel for a free loopback port.
///
/// The port is released before the service is spawned, so there is a small
/// window where something else could claim it. `ServiceController` treats a
/// service that never becomes healthy as a start failure and retries with a
/// fresh port, which covers that race.
enum PortFinder {
    static func freeLoopbackPort() -> Int? {
        let descriptor = socket(AF_INET, SOCK_STREAM, 0)
        guard descriptor >= 0 else { return nil }
        defer { Darwin.close(descriptor) }

        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = 0
        address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))

        let bound = withUnsafePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { generic in
                Darwin.bind(descriptor, generic, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bound == 0 else { return nil }

        var assigned = sockaddr_in()
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        let named = withUnsafeMutablePointer(to: &assigned) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { generic in
                getsockname(descriptor, generic, &length)
            }
        }
        guard named == 0 else { return nil }

        let port = Int(UInt16(bigEndian: assigned.sin_port))
        return port > 0 ? port : nil
    }
}
