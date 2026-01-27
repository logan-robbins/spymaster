import Foundation
import Metal
import MetalKit
import simd

final class MetalRenderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private var fieldPipeline: MTLRenderPipelineState?
    private var markerPipeline: MTLRenderPipelineState?
    private var computePipeline: MTLComputePipelineState?

    private var potentialBuffer: MTLBuffer?
    private var obstacleBuffer: MTLBuffer?
    private var combinedBuffer: MTLBuffer?
    private var uniformsBuffer: MTLBuffer?

    private var markerBuffer: MTLBuffer?
    private var markerVertexCount: Int = 0

    private var gridCount: Int = 0
    private var maxTicks: Float = 1
    private var fieldScale: Float = 0.15

    init?(device: MTLDevice? = MTLCreateSystemDefaultDevice()) {
        guard let device else { return nil }
        guard let queue = device.makeCommandQueue() else { return nil }
        self.device = device
        self.commandQueue = queue
        super.init()
        buildPipelines()
    }

    func attach(view: MTKView) {
        view.delegate = self
    }

    func update(fields: FieldGrid, grid: GridConfig, predictedOffsets: [Float], showPredictions: Bool) {
        gridCount = grid.count
        maxTicks = grid.maxTicks

        ensureBuffer(&potentialBuffer, length: fields.potential.count)
        ensureBuffer(&obstacleBuffer, length: fields.obstacle.count)
        ensureBuffer(&combinedBuffer, length: fields.potential.count)

        if let buffer = potentialBuffer {
            copy(array: fields.potential, to: buffer)
        }
        if let buffer = obstacleBuffer {
            copy(array: fields.obstacle, to: buffer)
        }

        var uniforms = FieldUniforms(count: UInt32(gridCount), fieldScale: fieldScale, maxTicks: maxTicks, padding: 0)
        if uniformsBuffer == nil {
            uniformsBuffer = device.makeBuffer(length: MemoryLayout<FieldUniforms>.stride, options: .storageModeShared)
        }
        if let buffer = uniformsBuffer {
            memcpy(buffer.contents(), &uniforms, MemoryLayout<FieldUniforms>.stride)
        }

        if showPredictions {
            updateMarkers(predictedOffsets: predictedOffsets, grid: grid)
        } else {
            markerVertexCount = 0
        }
    }

    func invalidate() {
        // No-op: MTKView drives rendering.
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        // No-op
    }

    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let renderPass = view.currentRenderPassDescriptor else {
            return
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            return
        }

        if let computePipeline, let combinedBuffer, let potentialBuffer, let obstacleBuffer, let uniformsBuffer {
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(computePipeline)
                encoder.setBuffer(potentialBuffer, offset: 0, index: 0)
                encoder.setBuffer(obstacleBuffer, offset: 0, index: 1)
                encoder.setBuffer(combinedBuffer, offset: 0, index: 2)
                encoder.setBuffer(uniformsBuffer, offset: 0, index: 3)

                let count = max(gridCount, 1)
                let threadsPerGroup = MTLSize(width: min(256, count), height: 1, depth: 1)
                let groups = MTLSize(width: (count + threadsPerGroup.width - 1) / threadsPerGroup.width, height: 1, depth: 1)
                encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threadsPerGroup)
                encoder.endEncoding()
            }
        }

        if let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPass) {
            if let pipeline = fieldPipeline, let combinedBuffer, let uniformsBuffer {
                encoder.setRenderPipelineState(pipeline)
                encoder.setVertexBuffer(combinedBuffer, offset: 0, index: 0)
                encoder.setVertexBuffer(uniformsBuffer, offset: 0, index: 1)
                encoder.drawPrimitives(type: .lineStrip, vertexStart: 0, vertexCount: gridCount)
            }

            if markerVertexCount > 0, let pipeline = markerPipeline, let markerBuffer {
                encoder.setRenderPipelineState(pipeline)
                encoder.setVertexBuffer(markerBuffer, offset: 0, index: 0)
                encoder.drawPrimitives(type: .line, vertexStart: 0, vertexCount: markerVertexCount)
            }
            encoder.endEncoding()
        }

        commandBuffer.present(drawable)
        commandBuffer.commit()
    }

    private func buildPipelines() {
        let library = loadLibrary()
        let fieldDescriptor = MTLRenderPipelineDescriptor()
        fieldDescriptor.vertexFunction = library.makeFunction(name: "field_vertex")
        fieldDescriptor.fragmentFunction = library.makeFunction(name: "field_fragment")
        fieldDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm

        let markerDescriptor = MTLRenderPipelineDescriptor()
        markerDescriptor.vertexFunction = library.makeFunction(name: "marker_vertex")
        markerDescriptor.fragmentFunction = library.makeFunction(name: "marker_fragment")
        markerDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm

        do {
            fieldPipeline = try device.makeRenderPipelineState(descriptor: fieldDescriptor)
            markerPipeline = try device.makeRenderPipelineState(descriptor: markerDescriptor)
        } catch {
            fieldPipeline = nil
            markerPipeline = nil
        }

        if let computeFunction = library.makeFunction(name: "combine_fields") {
            computePipeline = try? device.makeComputePipelineState(function: computeFunction)
        }
    }

    private func loadLibrary() -> MTLLibrary {
        guard let url = Bundle.module.url(forResource: "Shaders", withExtension: "metal") else {
            fatalError("Missing Shaders.metal resource")
        }
        do {
            let source = try String(contentsOf: url)
            return try device.makeLibrary(source: source, options: nil)
        } catch {
            fatalError("Failed to compile Shaders.metal: \\(error)")
        }
    }

    private func ensureBuffer(_ buffer: inout MTLBuffer?, length: Int) {
        let byteLength = length * MemoryLayout<Float>.stride
        if buffer == nil || buffer?.length != byteLength {
            buffer = device.makeBuffer(length: byteLength, options: .storageModeShared)
        }
    }

    private func copy(array: [Float], to buffer: MTLBuffer) {
        let byteLength = array.count * MemoryLayout<Float>.stride
        _ = array.withUnsafeBytes { bytes in
            memcpy(buffer.contents(), bytes.baseAddress, byteLength)
        }
    }

    private func updateMarkers(predictedOffsets: [Float], grid: GridConfig) {
        var vertices: [MarkerVertex] = []
        vertices.reserveCapacity((predictedOffsets.count + 1) * 2)

        for offset in predictedOffsets {
            let x = normalize(offset, maxTicks: grid.maxTicks)
            vertices.append(MarkerVertex(position: SIMD2<Float>(x, -0.05), color: SIMD4<Float>(1, 0.6, 0.2, 1)))
            vertices.append(MarkerVertex(position: SIMD2<Float>(x, 0.05), color: SIMD4<Float>(1, 0.6, 0.2, 1)))
        }

        let particleX: Float = 0
        vertices.append(MarkerVertex(position: SIMD2<Float>(particleX, -0.08), color: SIMD4<Float>(0.2, 1, 0.7, 1)))
        vertices.append(MarkerVertex(position: SIMD2<Float>(particleX, 0.08), color: SIMD4<Float>(0.2, 1, 0.7, 1)))

        let byteLength = vertices.count * MemoryLayout<MarkerVertex>.stride
        if markerBuffer == nil || markerBuffer?.length != byteLength {
            markerBuffer = device.makeBuffer(length: byteLength, options: .storageModeShared)
        }
        if let buffer = markerBuffer {
            _ = vertices.withUnsafeBytes { bytes in
                memcpy(buffer.contents(), bytes.baseAddress, byteLength)
            }
            markerVertexCount = vertices.count
        }
    }

    private func normalize(_ value: Float, maxTicks: Float) -> Float {
        guard maxTicks > 0 else { return 0 }
        return clamp(value / maxTicks, min: -1, max: 1)
    }

    private func clamp(_ value: Float, min: Float, max: Float) -> Float {
        Swift.max(min, Swift.min(max, value))
    }
}

private struct FieldUniforms {
    var count: UInt32
    var fieldScale: Float
    var maxTicks: Float
    var padding: Float
}

private struct MarkerVertex {
    var position: SIMD2<Float>
    var color: SIMD4<Float>
}
