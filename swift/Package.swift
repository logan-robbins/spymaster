// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MarketParticlePhysics",
    platforms: [.macOS(.v13)],
    products: [
        .executable(name: "MarketParticlePhysicsApp", targets: ["MarketParticlePhysicsApp"])
    ],
    targets: [
        .executableTarget(
            name: "MarketParticlePhysicsApp",
            resources: [
                .process("Metal/Shaders.metal")
            ]
        )
    ]
)
