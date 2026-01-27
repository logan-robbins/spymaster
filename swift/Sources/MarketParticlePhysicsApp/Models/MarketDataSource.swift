import Foundation

final class MarketDataSource {
    private let grid: GridConfig
    private let tickSize: Float
    private let startTimestampNs: TimestampNs
    private let endTimestampNs: TimestampNs
    private let dtNs: TimestampNs
    private var currentTimestampNs: TimestampNs
    private var spotTicks: Float
    private var rng: DeterministicRNG

    init(grid: GridConfig, seed: UInt64) {
        self.grid = grid
        self.tickSize = 0.25
        self.dtNs = 500_000_000
        self.startTimestampNs = MarketDataSource.makeTimestamp(year: 2026, month: 1, day: 6, hour: 9, minute: 30)
        self.endTimestampNs = MarketDataSource.makeTimestamp(year: 2026, month: 1, day: 6, hour: 10, minute: 30)
        self.currentTimestampNs = startTimestampNs
        self.spotTicks = 4800.0 / tickSize
        self.rng = DeterministicRNG(seed: seed)
    }

    func reset(seed: UInt64) {
        self.currentTimestampNs = startTimestampNs
        self.spotTicks = 4800.0 / tickSize
        self.rng = DeterministicRNG(seed: seed)
    }

    func nextFrame() -> MarketFrame? {
        guard currentTimestampNs <= endTimestampNs else {
            return nil
        }

        let directionalPush = Float(rng.nextSignedUnit()) * 2.0
        let noise = Float(rng.nextSignedUnit()) * 0.6
        let drift = directionalPush * 0.4

        spotTicks += drift + noise

        let wallCenter = Float(rng.nextSignedUnit()) * grid.maxTicks * 0.3
        let wallStrength = abs(Float(rng.nextSignedUnit())) * 6.0
        let wallWidth = 12.0 + abs(Float(rng.nextSignedUnit())) * 18.0

        let profile = makeProfilePush(centerOffset: wallCenter, strength: wallStrength * 0.2, width: wallWidth)

        let frame = MarketFrame(
            timestampNs: currentTimestampNs,
            spotTicks: spotTicks,
            tickSize: tickSize,
            directionalPush: directionalPush,
            viscosity: 0.9,
            wallFeature: WallFeature(centerOffsetTicks: wallCenter, strength: wallStrength, width: wallWidth),
            profilePush: profile
        )

        currentTimestampNs += dtNs
        return frame
    }

    private func makeProfilePush(centerOffset: Float, strength: Float, width: Float) -> [Float] {
        var profile = Array(repeating: Float(0), count: grid.count)
        for i in 0..<grid.count {
            let offset = Float(i - grid.halfWidth) * grid.deltaTicks
            let distance = (offset - centerOffset) / max(width, 1)
            let value = strength * exp(-0.5 * distance * distance)
            profile[i] = value
        }
        return profile
    }

    private static func makeTimestamp(year: Int, month: Int, day: Int, hour: Int, minute: Int) -> TimestampNs {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "America/New_York") ?? .current
        var components = DateComponents()
        components.year = year
        components.month = month
        components.day = day
        components.hour = hour
        components.minute = minute
        components.second = 0
        guard let date = calendar.date(from: components) else {
            return 0
        }
        return TimestampNs(date.timeIntervalSince1970 * 1_000_000_000)
    }
}

struct DeterministicRNG {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed == 0 ? 0xCAFEBABE : seed
    }

    mutating func nextUInt() -> UInt64 {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return state
    }

    mutating func nextDouble() -> Double {
        let value = nextUInt() >> 11
        return Double(value) / Double(1 << 53)
    }

    mutating func nextSignedUnit() -> Double {
        (nextDouble() * 2.0) - 1.0
    }
}
