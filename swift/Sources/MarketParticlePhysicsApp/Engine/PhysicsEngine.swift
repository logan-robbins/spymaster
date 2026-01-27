import Foundation

struct EngineStepResult {
    let predictedOffsets: [Float]
    let predictedTargets: [PredictionRecord]
    let resolvedErrors: [ResolvedPrediction]
    let resolvedOrigin: ResolvedOriginFrame?
    let particleVelocity: Float
    let particleAcceleration: Float
    let weights: Weights
}

final class PhysicsEngine {
    private let grid: GridConfig
    private var fields: FieldGrid
    private let wallMemory: WallMemory
    private let learner: OnlineLearner
    private let ledger: PredictionLedger
    private let horizon: Int
    private let dtSeconds: Float
    private let dtNs: TimestampNs
    private var lastSpotTicks: Float?
    private var lastTimestampNs: TimestampNs?

    init(grid: GridConfig, horizon: Int, dtSeconds: Float, ledger: PredictionLedger, wallMemory: WallMemory, learner: OnlineLearner) {
        self.grid = grid
        self.fields = FieldGrid(potential: [], current: [], damping: [], obstacle: [])
        self.wallMemory = wallMemory
        self.learner = learner
        self.ledger = ledger
        self.horizon = horizon
        self.dtSeconds = dtSeconds
        self.dtNs = TimestampNs(dtSeconds * 1_000_000_000)
    }

    func step(frame: MarketFrame) -> EngineStepResult {
        let resolution = ledger.resolve(timestampNs: frame.timestampNs, actualTicks: frame.spotTicks)
        if !resolution.resolved.isEmpty {
            learner.update(with: resolution.resolved)
        }

        wallMemory.update(with: frame.wallFeature)
        FeatureComposer.compose(fields: &fields, grid: grid, frame: frame, wallMemory: wallMemory)

        let v0 = estimateVelocity(currentTicks: frame.spotTicks, timestampNs: frame.timestampNs)
        var state = ParticleState(x: 0, v: v0)
        var records: [PredictionRecord] = []
        var offsets: [Float] = []

        for h in 1...horizon {
            let forces = sampleForces(at: state)
            let accel = combine(forces: forces, weights: learner.weights)
            let cappedAccel = clamp(accel, min: -50, max: 50)
            state.v = clamp(state.v + cappedAccel * dtSeconds, min: -200, max: 200)
            state.x = state.x + state.v * dtSeconds

            let targetTimestamp = frame.timestampNs + TimestampNs(h) * dtNs
            let predictedTicks = frame.spotTicks + state.x
            records.append(PredictionRecord(originTimestampNs: frame.timestampNs, targetTimestampNs: targetTimestamp, horizon: h, predictedTicks: predictedTicks, forces: forces))
            offsets.append(state.x)
        }

        ledger.store(originTimestampNs: frame.timestampNs, originSpotTicks: frame.spotTicks, horizonCount: horizon, records: records)

        lastSpotTicks = frame.spotTicks
        lastTimestampNs = frame.timestampNs

        let initialForces = sampleForces(at: ParticleState(x: 0, v: v0))
        let initialAccel = combine(forces: initialForces, weights: learner.weights)

        return EngineStepResult(
            predictedOffsets: offsets,
            predictedTargets: records,
            resolvedErrors: resolution.resolved,
            resolvedOrigin: resolution.resolvedOrigin,
            particleVelocity: v0,
            particleAcceleration: initialAccel,
            weights: learner.weights
        )
    }

    func currentFields() -> FieldGrid {
        fields
    }

    func reset() {
        lastSpotTicks = nil
        lastTimestampNs = nil
        ledger.reset()
    }

    private func estimateVelocity(currentTicks: Float, timestampNs: TimestampNs) -> Float {
        guard let lastTicks = lastSpotTicks, let lastTs = lastTimestampNs else {
            return 0
        }
        let dt = Float(timestampNs - lastTs) / 1_000_000_000
        if dt <= 0 {
            return 0
        }
        return (currentTicks - lastTicks) / dt
    }

    private func sampleForces(at state: ParticleState) -> ForceContribs {
        let fU = -gradient(of: fields.potential, at: state.x)
        let current = sample(field: fields.current, at: state.x)
        let damping = sample(field: fields.damping, at: state.x)
        let fC = current - state.v
        let fG = -damping * state.v
        let fO = -gradient(of: fields.obstacle, at: state.x)
        return ForceContribs(fU: fU, fC: fC, fG: fG, fO: fO, bias: 1.0)
    }

    private func combine(forces: ForceContribs, weights: Weights) -> Float {
        (weights.wU * forces.fU) + (weights.wC * forces.fC) + (weights.wG * forces.fG) + (weights.wO * forces.fO) + (weights.bias * forces.bias)
    }

    private func sample(field: [Float], at x: Float) -> Float {
        guard !field.isEmpty else { return 0 }
        let indexFloat = (x / grid.deltaTicks) + Float(grid.halfWidth)
        let idx0 = Int(floor(indexFloat))
        let idx1 = idx0 + 1
        let t = indexFloat - Float(idx0)
        let v0 = field[clampIndex(idx0, max: field.count - 1)]
        let v1 = field[clampIndex(idx1, max: field.count - 1)]
        return v0 + (v1 - v0) * t
    }

    private func gradient(of field: [Float], at x: Float) -> Float {
        guard !field.isEmpty else { return 0 }
        let indexFloat = (x / grid.deltaTicks) + Float(grid.halfWidth)
        let idx = Int(round(indexFloat))
        let idxPrev = clampIndex(idx - 1, max: field.count - 1)
        let idxNext = clampIndex(idx + 1, max: field.count - 1)
        let dy = field[idxNext] - field[idxPrev]
        return dy / (2 * grid.deltaTicks)
    }

    private func clamp(_ value: Float, min: Float, max: Float) -> Float {
        Swift.max(min, Swift.min(max, value))
    }

    private func clampIndex(_ index: Int, max: Int) -> Int {
        if index < 0 { return 0 }
        if index > max { return max }
        return index
    }
}
