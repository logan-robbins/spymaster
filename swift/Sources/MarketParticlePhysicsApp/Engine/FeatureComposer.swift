import Foundation

struct FeatureComposer {
    static func compose(fields: inout FieldGrid, grid: GridConfig, frame: MarketFrame, wallMemory: WallMemory) {
        let count = grid.count
        if fields.potential.count != count {
            fields.potential = Array(repeating: 0, count: count)
            fields.current = Array(repeating: 0, count: count)
            fields.damping = Array(repeating: 0, count: count)
            fields.obstacle = Array(repeating: 0, count: count)
        }

        let wall = wallMemory.state
        let profile = frame.profilePush
        let maxTicks = grid.maxTicks

        for i in 0..<count {
            let offset = Float(i - grid.halfWidth) * grid.deltaTicks
            let normalized = maxTicks > 0 ? offset / maxTicks : 0
            var potential: Float = 0
            var current: Float = frame.directionalPush

            if let profile, profile.count == count {
                potential += profile[i]
                current += profile[i] * 0.2
            } else {
                potential += -frame.directionalPush * normalized
            }

            let wallDistance = (offset - wall.centerOffsetTicks) / max(wall.width, 1)
            let wallInfluence = wall.strength * exp(-0.5 * wallDistance * wallDistance)

            fields.potential[i] = clamp(potential, min: -5, max: 5)
            fields.current[i] = clamp(current, min: -10, max: 10)
            fields.damping[i] = max(0.0, frame.viscosity)
            fields.obstacle[i] = clamp(wallInfluence, min: -10, max: 10)
        }
    }

    private static func clamp(_ value: Float, min: Float, max: Float) -> Float {
        Swift.max(min, Swift.min(max, value))
    }
}
