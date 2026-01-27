import Foundation
import SwiftUI

@MainActor
final class EngineViewModel: ObservableObject {
    @Published private(set) var actualSeries: [ChartPoint] = []
    @Published private(set) var predictionSeries: [Int: [ChartPoint]] = [:]
    @Published var isRunning: Bool = false
    @Published var isLiveMode: Bool = true
    @Published var isFrozen: Bool = false
    @Published var showPredictions: Bool = true
    @Published var zoomFactor: Double = 1.0
    @Published var scrubIndex: Double = 0
    @Published var hoverReadout: String = ""
    @Published var debugOverlayText: String = ""
    @Published var showDebugOverlay: Bool = false
    @Published private(set) var statusLine: String = ""
    @Published private(set) var weightSummaries: [WeightSummary] = []
    @Published private(set) var resolvedOrigin: ResolvedOriginFrame?

    let renderer: MetalRenderer

    private let grid = GridConfig(deltaTicks: 2, halfWidth: 200)
    private let horizon = 4
    private let dtSeconds: Float = 0.5
    private let ledger = PredictionLedger()
    private let initialWallState = WallState(centerOffsetTicks: 0, strength: 3, width: 20, decay: 0.92)
    private lazy var wallMemory = WallMemory(initial: initialWallState)
    private let initialWeights = Weights(wU: 0.35, wC: 0.45, wG: 0.25, wO: 0.35, bias: 0.0)
    private lazy var learner = OnlineLearner(
        initialWeights: initialWeights,
        learningRate: 0.002,
        huberDelta: 4.0,
        maxAbsWeight: 2.5
    )

    private lazy var engine = PhysicsEngine(grid: grid, horizon: horizon, dtSeconds: dtSeconds, ledger: ledger, wallMemory: wallMemory, learner: learner)
    private lazy var dataSource = MarketDataSource(grid: grid, seed: 42)

    private var timer: Timer?
    private var history: [EngineSnapshot] = []
    private var currentSnapshot: EngineSnapshot?
    private var hoverNormalizedX: Float?

    init() {
        guard let renderer = MetalRenderer() else {
            fatalError("Metal is required for this app")
        }
        self.renderer = renderer
    }

    func startIfNeeded() {
        if actualSeries.isEmpty {
            resume()
        }
    }

    func toggleRun() {
        if isRunning {
            stop()
        } else {
            resume()
        }
    }

    func replay() {
        stop()
        resetState(seed: 42)
        resume()
    }

    func freeze() {
        isFrozen = true
        stopTimer()
    }

    func stepOnce() {
        if !isRunning {
            tick()
        }
    }

    func resume() {
        isFrozen = false
        startTimer()
    }

    private func startTimer() {
        guard timer == nil else { return }
        isRunning = true
        timer = Timer.scheduledTimer(withTimeInterval: TimeInterval(dtSeconds), repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.tick()
            }
        }
    }

    private func stopTimer() {
        timer?.invalidate()
        timer = nil
        isRunning = false
    }

    private func stop() {
        stopTimer()
    }

    private func resetState(seed: UInt64) {
        actualSeries.removeAll()
        predictionSeries.removeAll()
        resolvedOrigin = nil
        ledger.reset()
        engine.reset()
        wallMemory.reset(to: initialWallState)
        learner.reset()
        dataSource.reset(seed: seed)
        history.removeAll()
        currentSnapshot = nil
        scrubIndex = 0
    }

    private func tick() {
        if !isLiveMode {
            return
        }

        guard let frame = dataSource.nextFrame() else {
            stop()
            return
        }

        let result = engine.step(frame: frame)
        let snapshot = EngineSnapshot(
            frame: frame,
            fields: engine.currentFields(),
            predictedOffsets: result.predictedOffsets,
            resolvedOrigin: result.resolvedOrigin ?? resolvedOrigin,
            particleVelocity: result.particleVelocity,
            particleAcceleration: result.particleAcceleration,
            weights: result.weights
        )
        history.append(snapshot)
        currentSnapshot = snapshot

        resolvedOrigin = result.resolvedOrigin ?? resolvedOrigin
        updateSeries(frame: frame, result: result)
        updateRenderer(snapshot: snapshot)
        updateStatus(frame: frame, result: result)
        updateDebugOverlay(snapshot: snapshot)

        scrubIndex = Double(max(history.count - 1, 0))
    }

    private func updateSeries(frame: MarketFrame, result: EngineStepResult) {
        let actualPoint = ChartPoint(timestamp: dateFrom(timestampNs: frame.timestampNs), value: Double(frame.spotTicks * frame.tickSize))
        actualSeries.append(actualPoint)
        trimSeries(&actualSeries)

        for record in result.predictedTargets {
            let date = dateFrom(timestampNs: record.targetTimestampNs)
            let value = Double(record.predictedTicks * frame.tickSize)
            let point = ChartPoint(timestamp: date, value: value)
            predictionSeries[record.horizon, default: []].append(point)
            trimSeries(&predictionSeries[record.horizon, default: []])
        }
    }

    private func updateRenderer(snapshot: EngineSnapshot) {
        renderer.update(
            snapshot: snapshot,
            grid: grid,
            wallState: wallMemory.state,
            showPredictions: showPredictions,
            resolvedOrigin: resolvedOrigin,
            hoverNormalizedX: hoverNormalizedX,
            zoomFactor: Float(zoomFactor)
        )
    }

    private func updateStatus(frame: MarketFrame, result: EngineStepResult) {
        weightSummaries = [
            WeightSummary(name: "wU", value: Double(result.weights.wU)),
            WeightSummary(name: "wC", value: Double(result.weights.wC)),
            WeightSummary(name: "wG", value: Double(result.weights.wG)),
            WeightSummary(name: "wO", value: Double(result.weights.wO)),
            WeightSummary(name: "bias", value: Double(result.weights.bias))
        ]

        let timestamp = dateFrom(timestampNs: frame.timestampNs).formatted(date: .omitted, time: .standard)
        statusLine = "t=\(timestamp) spot=\(String(format: "%.2f", frame.spotTicks * frame.tickSize)) pred=\(result.predictedOffsets.count)"
    }

    private func trimSeries(_ series: inout [ChartPoint], limit: Int = 600) {
        if series.count > limit {
            series.removeFirst(series.count - limit)
        }
    }

    func horizonColor(_ horizon: Int) -> Color {
        switch horizon {
        case 1: return .orange
        case 2: return .yellow
        case 3: return .green
        default: return .purple
        }
    }

    func updateHover(normalizedX: Float?) {
        hoverNormalizedX = normalizedX
        guard let snapshot = currentSnapshot else {
            hoverReadout = ""
            renderer.updateHover(normalizedX: normalizedX)
            return
        }

        renderer.updateHover(normalizedX: normalizedX)

        guard let normalizedX else {
            hoverReadout = ""
            return
        }

        let visibleMaxTicks = grid.maxTicks / Float(max(zoomFactor, 1e-3))
        let ticks = normalizedX * visibleMaxTicks

        let sample = FieldSampler.sample(fields: snapshot.fields, weights: snapshot.weights, grid: grid, ticks: ticks)
        hoverReadout = String(
            format: "x=%.1f U=%.2f O=%.2f C=%.2f Γ=%.2f dV/dx=%.3f",
            ticks,
            sample.potential,
            sample.obstacle,
            sample.current,
            sample.damping,
            sample.dVdx
        )
    }

    func scrub(to index: Double) {
        guard !history.isEmpty else { return }
        let clamped = Int(max(0, min(Double(history.count - 1), index)))
        scrubIndex = Double(clamped)
        let snapshot = history[clamped]
        currentSnapshot = snapshot
        resolvedOrigin = snapshot.resolvedOrigin
        updateRenderer(snapshot: snapshot)
        updateDebugOverlay(snapshot: snapshot)

        let timestamp = dateFrom(timestampNs: snapshot.frame.timestampNs).formatted(date: .omitted, time: .standard)
        statusLine = "SCRUB t=\(timestamp) spot=\(String(format: "%.2f", snapshot.frame.spotTicks * snapshot.frame.tickSize))"
    }

    func setLiveMode(_ enabled: Bool) {
        isLiveMode = enabled
        if enabled {
            resume()
        } else {
            freeze()
            scrub(to: scrubIndex)
        }
    }

    var historyCount: Int {
        history.count
    }

    var scrubLabel: String {
        guard !history.isEmpty else { return "--:--:--" }
        let index = Int(max(0, min(Double(history.count - 1), scrubIndex)))
        let snapshot = history[index]
        return dateFrom(timestampNs: snapshot.frame.timestampNs).formatted(date: .omitted, time: .standard)
    }

    var horizonCount: Int {
        horizon
    }

    var dtSecondsValue: Double {
        Double(dtSeconds)
    }

    func currentSnapshotForDisplay() -> EngineSnapshot? {
        currentSnapshot
    }

    func refreshRenderer(snapshot: EngineSnapshot) {
        updateRenderer(snapshot: snapshot)
    }

    func mapTicksToX(_ ticks: Float, width: CGFloat) -> CGFloat {
        let visibleMaxTicks = grid.maxTicks / Float(max(zoomFactor, 1e-3))
        let normalized = max(-1.0, min(1.0, ticks / visibleMaxTicks))
        return CGFloat((normalized + 1.0) * 0.5) * width
    }

    private func updateDebugOverlay(snapshot: EngineSnapshot) {
        debugOverlayText = String(
            format: "v=%.2f a=%.2f wU=%.2f wC=%.2f wG=%.2f wO=%.2f",
            snapshot.particleVelocity,
            snapshot.particleAcceleration,
            snapshot.weights.wU,
            snapshot.weights.wC,
            snapshot.weights.wG,
            snapshot.weights.wO
        )
    }

    private func dateFrom(timestampNs: TimestampNs) -> Date {
        Date(timeIntervalSince1970: TimeInterval(timestampNs) / 1_000_000_000)
    }
}

private struct FieldSample {
    let potential: Float
    let obstacle: Float
    let current: Float
    let damping: Float
    let dVdx: Float
}

private enum FieldSampler {
    static func sample(fields: FieldGrid, weights: Weights, grid: GridConfig, ticks: Float) -> FieldSample {
        let u = sample(field: fields.potential, grid: grid, ticks: ticks)
        let o = sample(field: fields.obstacle, grid: grid, ticks: ticks)
        let c = sample(field: fields.current, grid: grid, ticks: ticks)
        let g = sample(field: fields.damping, grid: grid, ticks: ticks)
        let v0 = (weights.wU * u) + (weights.wO * o)
        let grad = gradient(field: fields.potential, grid: grid, ticks: ticks) * weights.wU
            + gradient(field: fields.obstacle, grid: grid, ticks: ticks) * weights.wO
        return FieldSample(potential: u, obstacle: o, current: c, damping: g, dVdx: grad)
    }

    private static func sample(field: [Float], grid: GridConfig, ticks: Float) -> Float {
        guard !field.isEmpty else { return 0 }
        let indexFloat = (ticks / grid.deltaTicks) + Float(grid.halfWidth)
        let idx0 = Int(floor(indexFloat))
        let idx1 = idx0 + 1
        let t = indexFloat - Float(idx0)
        let v0 = field[clampIndex(idx0, max: field.count - 1)]
        let v1 = field[clampIndex(idx1, max: field.count - 1)]
        return v0 + (v1 - v0) * t
    }

    private static func gradient(field: [Float], grid: GridConfig, ticks: Float) -> Float {
        guard !field.isEmpty else { return 0 }
        let indexFloat = (ticks / grid.deltaTicks) + Float(grid.halfWidth)
        let idx = Int(round(indexFloat))
        let idxPrev = clampIndex(idx - 1, max: field.count - 1)
        let idxNext = clampIndex(idx + 1, max: field.count - 1)
        return (field[idxNext] - field[idxPrev]) / (2 * grid.deltaTicks)
    }

    private static func clampIndex(_ index: Int, max: Int) -> Int {
        if index < 0 { return 0 }
        if index > max { return max }
        return index
    }
}
