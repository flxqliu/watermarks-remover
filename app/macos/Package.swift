// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "WatermarksRemover",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(
            name: "WatermarksRemover",
            path: "Sources/WatermarksRemover"
        )
    ]
)
