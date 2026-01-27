import Foundation

typealias TimestampNs = Int64

struct GridConfig {
    let deltaTicks: Float
    let halfWidth: Int

    var count: Int { (halfWidth * 2) + 1 }
    var maxTicks: Float { Float(halfWidth) * deltaTicks }
}

struct MarketFrame {
    let timestampNs: TimestampNs
    let spotTicks: Float
    let tickSize: Float
    let directionalPush: Float
    let viscosity: Float
    let wallFeature: WallFeature
    let profilePush: [Float]?
}

struct WallFeature {
    let centerOffsetTicks: Float
    let strength: Float
    let width: Float
}

struct FieldGrid {
    var potential: [Float]
    var current: [Float]
    var damping: [Float]
    var obstacle: [Float]
}

struct ParticleState {
    var x: Float
    var v: Float
}

struct ForceContribs {
    let fU: Float
    let fC: Float
    let fG: Float
    let fO: Float
    let bias: Float
}

struct PredictionRecord {
    let originTimestampNs: TimestampNs
    let targetTimestampNs: TimestampNs
    let horizon: Int
    let predictedTicks: Float
    let forces: ForceContribs
}

struct ResolvedPrediction {
    let horizon: Int
    let errorTicks: Float
    let predictedTicks: Float
    let actualTicks: Float
    let forces: ForceContribs
}

struct ResolvedOriginFrame {
    let originTimestampNs: TimestampNs
    let originSpotTicks: Float
    let resolved: [ResolvedPrediction]
}

struct ChartPoint: Identifiable {
    let id = UUID()
    let timestamp: Date
    let value: Double
}

struct WeightSummary {
    let name: String
    let value: Double
}

struct EngineSnapshot {
    let frame: MarketFrame
    let fields: FieldGrid
    let predictedOffsets: [Float]
    let resolvedOrigin: ResolvedOriginFrame?
    let particleVelocity: Float
    let particleAcceleration: Float
    let weights: Weights
}
