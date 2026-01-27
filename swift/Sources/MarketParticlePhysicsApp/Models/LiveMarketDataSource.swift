import Foundation

/// Protocol for market data sources (synthetic or live).
protocol MarketDataSourceProtocol {
    func nextFrame() -> MarketFrame?
    func reset(seed: UInt64)
}

/// Adapts MarketDataSource to protocol.
extension MarketDataSource: MarketDataSourceProtocol {}

/// Live market data source that transforms HUD WebSocket batches into MarketFrames.
final class LiveMarketDataSource: MarketDataSourceProtocol, HudWebSocketClientDelegate {
    private let grid: GridConfig
    private let tickSize: Float = 0.25

    private var client: HudWebSocketClient?
    private var pendingBatches: [HudBatch] = []
    private let batchLock = NSLock()
    private var isConnected = false

    init(grid: GridConfig, symbol: String = "ESH6", dt: String = "2026-01-06", speed: Double = 1.0) {
        self.grid = grid
        self.client = HudWebSocketClient(symbol: symbol, dt: dt, speed: speed)
        self.client?.delegate = self
    }

    func connect() {
        client?.connect()
    }

    func disconnect() {
        client?.disconnect()
        isConnected = false
    }

    func reset(seed: UInt64) {
        batchLock.lock()
        pendingBatches.removeAll()
        batchLock.unlock()
    }

    func nextFrame() -> MarketFrame? {
        batchLock.lock()
        defer { batchLock.unlock() }

        guard !pendingBatches.isEmpty else {
            return nil
        }

        let batch = pendingBatches.removeFirst()
        return convertBatchToFrame(batch)
    }

    /// Check if frames are available without blocking.
    var hasFrames: Bool {
        batchLock.lock()
        defer { batchLock.unlock() }
        return !pendingBatches.isEmpty
    }

    /// Number of pending frames.
    var pendingCount: Int {
        batchLock.lock()
        defer { batchLock.unlock() }
        return pendingBatches.count
    }

    private func convertBatchToFrame(_ batch: HudBatch) -> MarketFrame {
        // Extract spot price
        let spotTicks: Float
        if let snap = batch.snap {
            spotTicks = Float(snap.midPrice) / tickSize
        } else {
            spotTicks = 0
        }

        // Compute wall feature from wall data
        let wallFeature = computeWallFeature(from: batch.wall, spotTicks: spotTicks)

        // Compute directional push from physics data
        let directionalPush = computeDirectionalPush(from: batch.physics)

        // Build profile push from wall depths
        let profilePush = computeProfilePush(from: batch.wall, grid: grid)

        return MarketFrame(
            timestampNs: batch.windowEndTsNs,
            spotTicks: spotTicks,
            tickSize: tickSize,
            directionalPush: directionalPush,
            viscosity: 0.9,
            wallFeature: wallFeature,
            profilePush: profilePush
        )
    }

    private func computeWallFeature(from wall: [ParsedWallRow], spotTicks: Float) -> WallFeature {
        // Find the dominant wall (largest depth concentration)
        guard !wall.isEmpty else {
            return WallFeature(centerOffsetTicks: 0, strength: 0, width: 10)
        }

        // Compute weighted center of mass for depth
        var totalDepth: Double = 0
        var weightedSum: Double = 0
        var maxDepth: Double = 0

        for row in wall {
            let depth = row.depthQtyRest
            if depth > 0 {
                totalDepth += depth
                weightedSum += Double(row.relTicks) * depth
                maxDepth = max(maxDepth, depth)
            }
        }

        let centerOffset: Float
        if totalDepth > 0 {
            centerOffset = Float(weightedSum / totalDepth)
        } else {
            centerOffset = 0
        }

        // Strength based on max depth (normalized)
        let strength = Float(min(maxDepth / 100.0, 10.0))

        // Width based on spread of significant depth
        let significantRows = wall.filter { $0.depthQtyRest > maxDepth * 0.1 }
        let width: Float
        if significantRows.count > 1 {
            let ticks = significantRows.map { Float($0.relTicks) }
            let minTick = ticks.min() ?? 0
            let maxTick = ticks.max() ?? 0
            width = max(maxTick - minTick, 10)
        } else {
            width = 10
        }

        return WallFeature(centerOffsetTicks: centerOffset, strength: strength, width: width)
    }

    private func computeDirectionalPush(from physics: [ParsedPhysicsRow]) -> Float {
        // Compute net directional bias from physics scores
        guard !physics.isEmpty else {
            return 0
        }

        var sumSigned: Double = 0
        for row in physics {
            sumSigned += row.physicsScoreSigned
        }

        // Normalize to reasonable range [-2, 2]
        let avg = sumSigned / Double(physics.count)
        return Float(avg * 4.0).clamped(to: -2...2)
    }

    private func computeProfilePush(from wall: [ParsedWallRow], grid: GridConfig) -> [Float] {
        // Build profile array matching grid dimensions
        var profile = [Float](repeating: 0, count: grid.count)

        for row in wall {
            let relTicks = Float(row.relTicks)
            let gridIndex = Int((relTicks / grid.deltaTicks) + Float(grid.halfWidth))

            if gridIndex >= 0 && gridIndex < grid.count {
                // Depth creates resistance (positive value = push away)
                let pushValue = Float(log1p(row.depthQtyRest)) * 0.1
                profile[gridIndex] += pushValue
            }
        }

        return profile
    }

    // MARK: - HudWebSocketClientDelegate

    func hudWebSocketClient(_ client: HudWebSocketClient, didReceiveBatch batch: HudBatch) {
        batchLock.lock()
        pendingBatches.append(batch)

        // Keep buffer bounded to prevent memory growth
        while pendingBatches.count > 100 {
            pendingBatches.removeFirst()
        }
        batchLock.unlock()
    }

    func hudWebSocketClient(_ client: HudWebSocketClient, didConnect url: URL) {
        isConnected = true
        print("[LiveDataSource] Connected to \(url)")
    }

    func hudWebSocketClient(_ client: HudWebSocketClient, didDisconnectWithError error: Error?) {
        isConnected = false
        if let error = error {
            print("[LiveDataSource] Disconnected with error: \(error)")
        } else {
            print("[LiveDataSource] Disconnected")
        }
    }
}

// MARK: - Helpers

extension Float {
    func clamped(to range: ClosedRange<Float>) -> Float {
        return min(max(self, range.lowerBound), range.upperBound)
    }
}
