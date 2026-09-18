// swift-tools-version:5.9
// Swift Package Manager manifest for SciRS2

import PackageDescription

let package = Package(
    name: "SciRS2",
    platforms: [
        .iOS(.v13),
        .macOS(.v11)
    ],
    products: [
        .library(
            name: "SciRS2",
            targets: ["SciRS2"]
        ),
    ],
    dependencies: [],
    targets: [
        // Swift wrapper
        .target(
            name: "SciRS2",
            dependencies: ["SciRS2Native"],
            path: "mobile/ios",
            sources: ["SciRS2.swift"]
        ),
        // Pre-built XCFramework
        .binaryTarget(
            name: "SciRS2Native",
            path: "target/ios/SciRS2.xcframework"
        ),
        // Tests
        .testTarget(
            name: "SciRS2Tests",
            dependencies: ["SciRS2"],
            path: "mobile/ios/Tests"
        ),
    ]
)
