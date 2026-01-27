import Foundation

struct WallState {
    var centerOffsetTicks: Float
    var strength: Float
    var width: Float
    var decay: Float
}

final class WallMemory {
    private(set) var state: WallState

    init(initial: WallState) {
        self.state = initial
    }

    func update(with feature: WallFeature) {
        let blendedCenter = (state.centerOffsetTicks * 0.8) + (feature.centerOffsetTicks * 0.2)
        state.centerOffsetTicks = blendedCenter
        state.width = max(1.0, feature.width)
        state.strength = (state.strength * state.decay) + feature.strength
    }

    func reset(to initial: WallState) {
        state = initial
    }
}
