import Foundation

struct Weights {
    var wU: Float
    var wC: Float
    var wG: Float
    var wO: Float
    var bias: Float
}

final class OnlineLearner {
    private(set) var weights: Weights
    private let learningRate: Float
    private let huberDelta: Float
    private let maxAbsWeight: Float
    private let initialWeights: Weights

    init(initialWeights: Weights, learningRate: Float, huberDelta: Float, maxAbsWeight: Float) {
        self.weights = initialWeights
        self.initialWeights = initialWeights
        self.learningRate = learningRate
        self.huberDelta = huberDelta
        self.maxAbsWeight = maxAbsWeight
    }

    func update(with predictions: [ResolvedPrediction]) {
        for prediction in predictions {
            let alpha = 1.0 + (Float(prediction.horizon - 1) * 0.25)
            let grad = huberGradient(prediction.errorTicks) * alpha
            let forces = prediction.forces

            weights.wU += learningRate * grad * forces.fU
            weights.wC += learningRate * grad * forces.fC
            weights.wG += learningRate * grad * forces.fG
            weights.wO += learningRate * grad * forces.fO
            weights.bias += learningRate * grad * forces.bias

            enforceBounds()
        }
    }

    private func huberGradient(_ error: Float) -> Float {
        let absError = abs(error)
        if absError <= huberDelta {
            return error
        }
        return huberDelta * (error / absError)
    }

    private func enforceBounds() {
        weights.wU = clamp(weights.wU)
        weights.wC = clamp(weights.wC)
        weights.wG = max(0.0, clamp(weights.wG))
        weights.wO = clamp(weights.wO)
        weights.bias = clamp(weights.bias)
    }

    private func clamp(_ value: Float) -> Float {
        Swift.max(-maxAbsWeight, Swift.min(maxAbsWeight, value))
    }

    func reset() {
        weights = initialWeights
    }
}
