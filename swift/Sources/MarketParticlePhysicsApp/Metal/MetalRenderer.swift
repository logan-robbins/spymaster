import Foundation
import Metal
import MetalKit
import simd

final class MetalRenderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private var pipelineState: MTLRenderPipelineState?

    private var triangleBuffer: MTLBuffer?
    private var lineBuffer: MTLBuffer?
    private var triangleCount: Int = 0
    private var lineCount: Int = 0

    private var input: RenderInput?
    private var hoverNormalizedX: Float?
    private var lastWallStrength: Float?

    init?(device: MTLDevice? = MTLCreateSystemDefaultDevice()) {
        guard let device else { return nil }
        guard let queue = device.makeCommandQueue() else { return nil }
        self.device = device
        self.commandQueue = queue
        super.init()
        buildPipeline()
    }

    func attach(view: MTKView) {
        view.delegate = self
    }

    func update(snapshot: EngineSnapshot,
                grid: GridConfig,
                wallState: WallState,
                showPredictions: Bool,
                resolvedOrigin: ResolvedOriginFrame?,
                hoverNormalizedX: Float?,
                zoomFactor: Float) {
        input = RenderInput(
            snapshot: snapshot,
            grid: grid,
            wallState: wallState,
            showPredictions: showPredictions,
            resolvedOrigin: resolvedOrigin,
            zoomFactor: zoomFactor
        )
        self.hoverNormalizedX = hoverNormalizedX
        rebuildGeometry()
    }

    func updateHover(normalizedX: Float?) {
        hoverNormalizedX = normalizedX
        rebuildGeometry()
    }

    func invalidate() {
        // MTKView will continuously draw.
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        // No-op
    }

    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let renderPass = view.currentRenderPassDescriptor,
              let pipelineState else {
            return
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            return
        }

        if let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPass) {
            encoder.setRenderPipelineState(pipelineState)

            if let triangleBuffer, triangleCount > 0 {
                encoder.setVertexBuffer(triangleBuffer, offset: 0, index: 0)
                encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: triangleCount)
            }

            if let lineBuffer, lineCount > 0 {
                encoder.setVertexBuffer(lineBuffer, offset: 0, index: 0)
                encoder.drawPrimitives(type: .line, vertexStart: 0, vertexCount: lineCount)
            }

            encoder.endEncoding()
        }

        commandBuffer.present(drawable)
        commandBuffer.commit()
    }

    private func buildPipeline() {
        guard let library = loadLibrary() else { return }
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = library.makeFunction(name: "color_vertex")
        descriptor.fragmentFunction = library.makeFunction(name: "color_fragment")
        descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        descriptor.colorAttachments[0].isBlendingEnabled = true
        descriptor.colorAttachments[0].rgbBlendOperation = .add
        descriptor.colorAttachments[0].alphaBlendOperation = .add
        descriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        descriptor.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
        descriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        descriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha

        do {
            pipelineState = try device.makeRenderPipelineState(descriptor: descriptor)
        } catch {
            pipelineState = nil
        }
    }

    private func loadLibrary() -> MTLLibrary? {
        guard let url = Bundle.module.url(forResource: "Shaders", withExtension: "metal") else {
            return nil
        }
        do {
            let source = try String(contentsOf: url)
            return try device.makeLibrary(source: source, options: nil)
        } catch {
            return nil
        }
    }

    private func rebuildGeometry() {
        guard let input else {
            triangleCount = 0
            lineCount = 0
            return
        }

        var triangleVertices: [ColoredVertex] = []
        var lineVertices: [ColoredVertex] = []

        let layout = BandLayout()
        let visibleMaxTicks = input.grid.maxTicks / max(input.zoomFactor, 0.1)

        let normalizedX: (Float) -> Float = { ticks in
            let value = ticks / visibleMaxTicks
            return max(-1.0, min(1.0, value))
        }

        let bandA = layout.bandA
        let bandB = layout.bandB
        let bandC = layout.bandC

        addVerticalGridLines(grid: input.grid, top: bandA.top, bottom: bandC.bottom, normalizedX: normalizedX, lineVertices: &lineVertices)
        addSpotMarkers(band: bandA, normalizedX: normalizedX, lineVertices: &lineVertices)

        let samples = FieldSamples(fields: input.snapshot.fields, weights: input.snapshot.weights)
        let terrain = samples.terrain
        let terrainScale = (bandA.top - bandA.bottom) * 0.4
        let midline = (bandA.top + bandA.bottom) * 0.5

        addTerrainFill(terrain: terrain, grid: input.grid, band: bandA, midline: midline, scale: terrainScale, normalizedX: normalizedX, triangleVertices: &triangleVertices)
        addTerrainContours(terrain: terrain, grid: input.grid, band: bandA, midline: midline, scale: terrainScale, normalizedX: normalizedX, lineVertices: &lineVertices)

        addObstacleGate(wallState: input.wallState, band: bandA, normalizedX: normalizedX, hoverNormalizedX: hoverNormalizedX, triangleVertices: &triangleVertices, lineVertices: &lineVertices)
        addViscosityHaze(damping: samples.damping, grid: input.grid, band: bandA, normalizedX: normalizedX, triangleVertices: &triangleVertices, lineVertices: &lineVertices)
        addCurrentArrows(current: samples.current, grid: input.grid, band: bandA, normalizedX: normalizedX, lineVertices: &lineVertices)
        addParticle(snapshot: input.snapshot, band: bandA, lineVertices: &lineVertices)

        addForecastLadder(predictedOffsets: input.showPredictions ? input.snapshot.predictedOffsets : [], band: bandB, normalizedX: normalizedX, lineVertices: &lineVertices)
        addResolvedLadder(resolvedOrigin: input.resolvedOrigin, band: bandC, normalizedX: normalizedX, lineVertices: &lineVertices)

        if let hoverNormalizedX {
            addCrosshair(x: hoverNormalizedX, band: bandA, bandB: bandB, bandC: bandC, lineVertices: &lineVertices)
        }

        writeBuffers(triangles: triangleVertices, lines: lineVertices)
        lastWallStrength = input.wallState.strength
    }

    private func addVerticalGridLines(grid: GridConfig,
                                      top: Float,
                                      bottom: Float,
                                      normalizedX: (Float) -> Float,
                                      lineVertices: inout [ColoredVertex]) {
        let minorColor = SIMD4<Float>(0.25, 0.28, 0.32, 0.25)
        let majorColor = SIMD4<Float>(0.4, 0.45, 0.5, 0.45)

        for bucketIndex in -grid.halfWidth...grid.halfWidth {
            let ticks = Float(bucketIndex) * grid.deltaTicks
            let x = normalizedX(ticks)
            let isMajor = bucketIndex % 10 == 0
            let color = isMajor ? majorColor : minorColor
            addLine(from: SIMD2<Float>(x, bottom), to: SIMD2<Float>(x, top), color: color, lineVertices: &lineVertices)
        }
    }

    private func addSpotMarkers(band: Band,
                                normalizedX: (Float) -> Float,
                                lineVertices: inout [ColoredVertex]) {
        let centerColor = SIMD4<Float>(0.7, 0.8, 0.9, 0.8)
        let nearColor = SIMD4<Float>(0.5, 0.6, 0.7, 0.5)
        let xCenter: Float = 0

        addLine(from: SIMD2<Float>(xCenter, band.bottom), to: SIMD2<Float>(xCenter, band.top), color: centerColor, lineVertices: &lineVertices)

        let nearTicks: Float = 2
        let xPos = normalizedX(nearTicks)
        let xNeg = normalizedX(-nearTicks)
        addDashedLine(x: xPos, band: band, color: nearColor, lineVertices: &lineVertices)
        addDashedLine(x: xNeg, band: band, color: nearColor, lineVertices: &lineVertices)
    }

    private func addDashedLine(x: Float, band: Band, color: SIMD4<Float>, lineVertices: inout [ColoredVertex]) {
        let segments = 10
        let segmentHeight = (band.top - band.bottom) / Float(segments)
        for i in 0..<segments {
            if i % 2 == 0 {
                let y0 = band.bottom + Float(i) * segmentHeight
                let y1 = y0 + segmentHeight * 0.6
                addLine(from: SIMD2<Float>(x, y0), to: SIMD2<Float>(x, y1), color: color, lineVertices: &lineVertices)
            }
        }
    }

    private func addTerrainFill(terrain: [Float],
                                grid: GridConfig,
                                band: Band,
                                midline: Float,
                                scale: Float,
                                normalizedX: (Float) -> Float,
                                triangleVertices: inout [ColoredVertex]) {
        let color = SIMD4<Float>(0.2, 0.55, 0.9, 0.18)
        let baseline = band.bottom
        for i in 0..<(terrain.count - 1) {
            let ticks0 = Float(i - grid.halfWidth) * grid.deltaTicks
            let ticks1 = Float(i + 1 - grid.halfWidth) * grid.deltaTicks
            let x0 = normalizedX(ticks0)
            let x1 = normalizedX(ticks1)
            let y0 = midline - terrain[i] * scale
            let y1 = midline - terrain[i + 1] * scale

            addTriangleStrip(x0: x0, x1: x1, y0: baseline, y1: y0, y2: y1, color: color, triangleVertices: &triangleVertices)
        }
    }

    private func addTriangleStrip(x0: Float, x1: Float, y0: Float, y1: Float, y2: Float, color: SIMD4<Float>, triangleVertices: inout [ColoredVertex]) {
        let p0 = ColoredVertex(position: SIMD2<Float>(x0, y0), color: color)
        let p1 = ColoredVertex(position: SIMD2<Float>(x1, y0), color: color)
        let p2 = ColoredVertex(position: SIMD2<Float>(x0, y1), color: color)
        let p3 = ColoredVertex(position: SIMD2<Float>(x1, y2), color: color)

        triangleVertices.append(contentsOf: [p0, p1, p2])
        triangleVertices.append(contentsOf: [p1, p3, p2])
    }

    private func addTerrainContours(terrain: [Float],
                                    grid: GridConfig,
                                    band: Band,
                                    midline: Float,
                                    scale: Float,
                                    normalizedX: (Float) -> Float,
                                    lineVertices: inout [ColoredVertex]) {
        let contourColor = SIMD4<Float>(0.35, 0.75, 0.95, 0.35)
        let contourLevels: [Float] = [0.3, 0.6, 0.9]
        for level in contourLevels {
            for i in 0..<(terrain.count - 1) {
                let ticks0 = Float(i - grid.halfWidth) * grid.deltaTicks
                let ticks1 = Float(i + 1 - grid.halfWidth) * grid.deltaTicks
                let x0 = normalizedX(ticks0)
                let x1 = normalizedX(ticks1)
                let y0 = midline - terrain[i] * scale * level
                let y1 = midline - terrain[i + 1] * scale * level
                addLine(from: SIMD2<Float>(x0, y0), to: SIMD2<Float>(x1, y1), color: contourColor, lineVertices: &lineVertices)
            }
        }
    }

    private func addObstacleGate(wallState: WallState,
                                 band: Band,
                                 normalizedX: (Float) -> Float,
                                 hoverNormalizedX: Float?,
                                 triangleVertices: inout [ColoredVertex],
                                 lineVertices: inout [ColoredVertex]) {
        let widthTicks = max(wallState.width, 1)
        let x0 = normalizedX(wallState.centerOffsetTicks - widthTicks)
        let x1 = normalizedX(wallState.centerOffsetTicks + widthTicks)
        let strengthNorm = clamp(wallState.strength / 10.0, min: 0, max: 1)
        let highlighted = hoverNormalizedX.map { abs($0 - normalizedX(wallState.centerOffsetTicks)) <= abs(x1 - x0) * 0.6 } ?? false
        let baseColor = highlighted ? SIMD4<Float>(0.95, 0.5, 0.2, 0.35) : SIMD4<Float>(0.8, 0.35, 0.2, 0.25)

        addRect(x0: x0, x1: x1, y0: band.bottom, y1: band.top, color: baseColor, triangleVertices: &triangleVertices)

        let deltaStrength = abs((lastWallStrength ?? wallState.strength) - wallState.strength)
        let edgeAlpha: Float = deltaStrength > 0.5 ? 0.9 : 0.65
        let edgeColor = SIMD4<Float>(0.9, 0.6, 0.4, edgeAlpha)
        addLine(from: SIMD2<Float>(x0, band.bottom), to: SIMD2<Float>(x0, band.top), color: edgeColor, lineVertices: &lineVertices)
        addLine(from: SIMD2<Float>(x1, band.bottom), to: SIMD2<Float>(x1, band.top), color: edgeColor, lineVertices: &lineVertices)

        let wearHeight = (band.top - band.bottom) * 0.08
        let wearBottom = band.top - wearHeight * 1.2
        let wearTop = wearBottom + wearHeight * strengthNorm
        let inset = (x1 - x0) * 0.2
        let wearColor = SIMD4<Float>(1.0, 0.7, 0.4, 0.7)
        let wearBg = SIMD4<Float>(0.25, 0.15, 0.1, 0.6)
        addRect(x0: x0 + inset, x1: x1 - inset, y0: wearBottom, y1: wearBottom + wearHeight, color: wearBg, triangleVertices: &triangleVertices)
        addRect(x0: x0 + inset, x1: x1 - inset, y0: wearBottom, y1: wearTop, color: wearColor, triangleVertices: &triangleVertices)
    }

    private func addViscosityHaze(damping: [Float],
                                  grid: GridConfig,
                                  band: Band,
                                  normalizedX: (Float) -> Float,
                                  triangleVertices: inout [ColoredVertex],
                                  lineVertices: inout [ColoredVertex]) {
        guard !damping.isEmpty else { return }
        let maxDamping = max(1.0, damping.max() ?? 1.0)
        var mean: Float = 0
        for i in 0..<damping.count {
            let ticks0 = Float(i - grid.halfWidth) * grid.deltaTicks
            let ticks1 = Float(i + 1 - grid.halfWidth) * grid.deltaTicks
            let x0 = normalizedX(ticks0)
            let x1 = normalizedX(ticks1)
            let alpha = clamp(damping[i] / maxDamping * 0.12, min: 0.02, max: 0.12)
            let color = SIMD4<Float>(0.2, 0.25, 0.32, alpha)
            addRect(x0: x0, x1: x1, y0: band.bottom, y1: band.top, color: color, triangleVertices: &triangleVertices)
            mean += damping[i]
        }
        mean /= Float(damping.count)

        let stripeCount = 6
        let stripeAlpha = clamp((mean / maxDamping) * 0.25, min: 0.05, max: 0.18)
        let stripeColor = SIMD4<Float>(0.35, 0.4, 0.48, stripeAlpha)
        for i in 1...stripeCount {
            let y = band.bottom + (band.top - band.bottom) * (Float(i) / Float(stripeCount + 1))
            addLine(from: SIMD2<Float>(-1.0, y), to: SIMD2<Float>(1.0, y), color: stripeColor, lineVertices: &lineVertices)
        }
    }

    private func addCurrentArrows(current: [Float],
                                  grid: GridConfig,
                                  band: Band,
                                  normalizedX: (Float) -> Float,
                                  lineVertices: inout [ColoredVertex]) {
        guard !current.isEmpty else { return }
        let maxCurrent = max(1.0, current.map { abs($0) }.max() ?? 1.0)
        let baseline = band.bottom + (band.top - band.bottom) * 0.18
        let arrowColor = SIMD4<Float>(0.7, 0.85, 0.95, 0.6)

        for i in stride(from: 0, to: current.count, by: 2) {
            let ticks = Float(i - grid.halfWidth) * grid.deltaTicks
            let x = normalizedX(ticks)
            let direction = current[i]
            let length = clamp(direction / maxCurrent * 0.08, min: -0.08, max: 0.08)
            let endX = x + length
            addArrow(from: SIMD2<Float>(x, baseline), to: SIMD2<Float>(endX, baseline), color: arrowColor, lineVertices: &lineVertices)
        }
    }

    private func addParticle(snapshot: EngineSnapshot,
                             band: Band,
                             lineVertices: inout [ColoredVertex]) {
        let center = SIMD2<Float>(0, (band.top + band.bottom) * 0.5)
        let circleColor = SIMD4<Float>(0.2, 1.0, 0.7, 0.9)
        addCircle(center: center, radius: 0.02, segments: 12, color: circleColor, lineVertices: &lineVertices)

        let velocityScale: Float = 0.0008
        let vLength = clamp(snapshot.particleVelocity * velocityScale, min: -0.15, max: 0.15)
        let velocityColor = SIMD4<Float>(0.9, 0.9, 0.4, 0.8)
        addArrow(from: center, to: SIMD2<Float>(center.x + vLength, center.y), color: velocityColor, lineVertices: &lineVertices)

        let accelScale: Float = 0.0012
        let aLength = clamp(snapshot.particleAcceleration * accelScale, min: -0.12, max: 0.12)
        let accelColor = SIMD4<Float>(1.0, 0.4, 0.3, 0.8)
        let accelStart = SIMD2<Float>(center.x, center.y - 0.05)
        addArrow(from: accelStart, to: SIMD2<Float>(accelStart.x + aLength, accelStart.y), color: accelColor, lineVertices: &lineVertices)
    }

    private func addForecastLadder(predictedOffsets: [Float],
                                   band: Band,
                                   normalizedX: (Float) -> Float,
                                   lineVertices: inout [ColoredVertex]) {
        guard !predictedOffsets.isEmpty else { return }
        let rows = predictedOffsets.count
        let rowHeight = (band.top - band.bottom) / Float(rows)
        let baseColor = SIMD4<Float>(1.0, 0.65, 0.2, 0.8)

        var previousPoint: SIMD2<Float>?
        for (index, offset) in predictedOffsets.enumerated() {
            let rowCenter = band.top - rowHeight * (Float(index) + 0.5)
            let x = normalizedX(offset)
            let opacity = 1.0 - (Float(index) / Float(rows)) * 0.5
            let color = SIMD4<Float>(baseColor.x, baseColor.y, baseColor.z, opacity)

            addLine(from: SIMD2<Float>(-1.0, rowCenter), to: SIMD2<Float>(1.0, rowCenter), color: SIMD4<Float>(0.25, 0.3, 0.35, 0.4), lineVertices: &lineVertices)
            addLine(from: SIMD2<Float>(x, rowCenter - rowHeight * 0.2), to: SIMD2<Float>(x, rowCenter + rowHeight * 0.2), color: color, lineVertices: &lineVertices)
            addLine(from: SIMD2<Float>(x, rowCenter), to: SIMD2<Float>(x, band.bottom), color: SIMD4<Float>(color.x, color.y, color.z, 0.25), lineVertices: &lineVertices)

            let point = SIMD2<Float>(x, rowCenter)
            if let previousPoint {
                addLine(from: previousPoint, to: point, color: color, lineVertices: &lineVertices)
            }
            previousPoint = point
        }
    }

    private func addResolvedLadder(resolvedOrigin: ResolvedOriginFrame?,
                                   band: Band,
                                   normalizedX: (Float) -> Float,
                                   lineVertices: inout [ColoredVertex]) {
        guard let resolvedOrigin else { return }
        let rows = resolvedOrigin.resolved.count
        guard rows > 0 else { return }
        let rowHeight = (band.top - band.bottom) / Float(rows)

        for (index, resolved) in resolvedOrigin.resolved.enumerated() {
            let rowCenter = band.top - rowHeight * (Float(index) + 0.5)
            let predictedOffset = resolved.predictedTicks - resolvedOrigin.originSpotTicks
            let actualOffset = resolved.actualTicks - resolvedOrigin.originSpotTicks
            let xPred = normalizedX(predictedOffset)
            let xActual = normalizedX(actualOffset)
            let predColor = SIMD4<Float>(0.9, 0.55, 0.25, 0.8)
            let actualColor = SIMD4<Float>(0.9, 0.9, 0.9, 0.85)
            let errorColor = SIMD4<Float>(0.8, 0.2, 0.2, 0.7)

            addLine(from: SIMD2<Float>(-1.0, rowCenter), to: SIMD2<Float>(1.0, rowCenter), color: SIMD4<Float>(0.22, 0.26, 0.3, 0.35), lineVertices: &lineVertices)
            addLine(from: SIMD2<Float>(xPred, rowCenter - rowHeight * 0.2), to: SIMD2<Float>(xPred, rowCenter + rowHeight * 0.2), color: predColor, lineVertices: &lineVertices)
            addLine(from: SIMD2<Float>(xActual, rowCenter - rowHeight * 0.2), to: SIMD2<Float>(xActual, rowCenter + rowHeight * 0.2), color: actualColor, lineVertices: &lineVertices)
            addLine(from: SIMD2<Float>(xPred, rowCenter), to: SIMD2<Float>(xActual, rowCenter), color: errorColor, lineVertices: &lineVertices)
        }
    }

    private func addCrosshair(x: Float, band: Band, bandB: Band, bandC: Band, lineVertices: inout [ColoredVertex]) {
        let color = SIMD4<Float>(0.8, 0.85, 0.9, 0.5)
        addLine(from: SIMD2<Float>(x, bandC.bottom), to: SIMD2<Float>(x, band.top), color: color, lineVertices: &lineVertices)
    }

    private func addRect(x0: Float, x1: Float, y0: Float, y1: Float, color: SIMD4<Float>, triangleVertices: inout [ColoredVertex]) {
        let p0 = ColoredVertex(position: SIMD2<Float>(x0, y0), color: color)
        let p1 = ColoredVertex(position: SIMD2<Float>(x1, y0), color: color)
        let p2 = ColoredVertex(position: SIMD2<Float>(x0, y1), color: color)
        let p3 = ColoredVertex(position: SIMD2<Float>(x1, y1), color: color)
        triangleVertices.append(contentsOf: [p0, p1, p2, p1, p3, p2])
    }

    private func addLine(from: SIMD2<Float>, to: SIMD2<Float>, color: SIMD4<Float>, lineVertices: inout [ColoredVertex]) {
        lineVertices.append(ColoredVertex(position: from, color: color))
        lineVertices.append(ColoredVertex(position: to, color: color))
    }

    private func addArrow(from: SIMD2<Float>, to: SIMD2<Float>, color: SIMD4<Float>, lineVertices: inout [ColoredVertex]) {
        addLine(from: from, to: to, color: color, lineVertices: &lineVertices)
        let delta = to - from
        let length = simd_length(delta)
        if length < 0.0001 {
            return
        }
        let direction = delta / length
        let arrowSize: Float = 0.015
        let angle: Float = .pi * 0.75
        let left = SIMD2<Float>(
            cos(angle) * direction.x - sin(angle) * direction.y,
            sin(angle) * direction.x + cos(angle) * direction.y
        ) * arrowSize
        let right = SIMD2<Float>(
            cos(-angle) * direction.x - sin(-angle) * direction.y,
            sin(-angle) * direction.x + cos(-angle) * direction.y
        ) * arrowSize
        addLine(from: to, to: to + left, color: color, lineVertices: &lineVertices)
        addLine(from: to, to: to + right, color: color, lineVertices: &lineVertices)
    }

    private func addCircle(center: SIMD2<Float>, radius: Float, segments: Int, color: SIMD4<Float>, lineVertices: inout [ColoredVertex]) {
        guard segments > 2 else { return }
        let step = 2 * Float.pi / Float(segments)
        var prevPoint = center + SIMD2<Float>(radius, 0)
        for i in 1...segments {
            let angle = Float(i) * step
            let point = center + SIMD2<Float>(cos(angle) * radius, sin(angle) * radius)
            addLine(from: prevPoint, to: point, color: color, lineVertices: &lineVertices)
            prevPoint = point
        }
    }

    private func writeBuffers(triangles: [ColoredVertex], lines: [ColoredVertex]) {
        let triangleByteLength = triangles.count * MemoryLayout<ColoredVertex>.stride
        if triangleBuffer == nil || triangleBuffer?.length != triangleByteLength {
            triangleBuffer = device.makeBuffer(length: triangleByteLength, options: .storageModeShared)
        }
        if let triangleBuffer, triangleByteLength > 0 {
            _ = triangles.withUnsafeBytes { bytes in
                memcpy(triangleBuffer.contents(), bytes.baseAddress, triangleByteLength)
            }
        }
        triangleCount = triangles.count

        let lineByteLength = lines.count * MemoryLayout<ColoredVertex>.stride
        if lineBuffer == nil || lineBuffer?.length != lineByteLength {
            lineBuffer = device.makeBuffer(length: lineByteLength, options: .storageModeShared)
        }
        if let lineBuffer, lineByteLength > 0 {
            _ = lines.withUnsafeBytes { bytes in
                memcpy(lineBuffer.contents(), bytes.baseAddress, lineByteLength)
            }
        }
        lineCount = lines.count
    }

    private func clamp(_ value: Float, min: Float, max: Float) -> Float {
        Swift.max(min, Swift.min(max, value))
    }
}

private struct Band {
    let top: Float
    let bottom: Float
}

private struct BandLayout {
    let bandA: Band
    let bandB: Band
    let bandC: Band

    init() {
        let totalHeight: Float = 2.0
        let bandAHeight = totalHeight * 0.65
        let bandBHeight = totalHeight * 0.20

        let top: Float = 1.0
        let bandABottom = top - bandAHeight
        let bandBBottom = bandABottom - bandBHeight

        bandA = Band(top: top, bottom: bandABottom)
        bandB = Band(top: bandABottom, bottom: bandBBottom)
        bandC = Band(top: bandBBottom, bottom: -1.0)
    }
}

private struct RenderInput {
    let snapshot: EngineSnapshot
    let grid: GridConfig
    let wallState: WallState
    let showPredictions: Bool
    let resolvedOrigin: ResolvedOriginFrame?
    let zoomFactor: Float
}

private struct FieldSamples {
    let terrain: [Float]
    let current: [Float]
    let damping: [Float]

    init(fields: FieldGrid, weights: Weights) {
        let count = fields.potential.count
        var combined = Array(repeating: Float(0), count: count)
        var maxAbs: Float = 1
        for i in 0..<count {
            combined[i] = (weights.wU * fields.potential[i]) + (weights.wO * fields.obstacle[i])
            maxAbs = max(maxAbs, abs(combined[i]))
        }
        terrain = combined.map { $0 / maxAbs }
        current = fields.current
        damping = fields.damping
    }
}

private struct ColoredVertex {
    var position: SIMD2<Float>
    var color: SIMD4<Float>
}
