import MetalKit
import SwiftUI

struct MetalView: NSViewRepresentable {
    let renderer: MetalRenderer

    func makeNSView(context: Context) -> MTKView {
        let view = MTKView()
        view.device = renderer.device
        view.clearColor = MTLClearColorMake(0.05, 0.05, 0.08, 1.0)
        view.colorPixelFormat = .bgra8Unorm
        view.preferredFramesPerSecond = 60
        view.enableSetNeedsDisplay = false
        view.isPaused = false
        renderer.attach(view: view)
        return view
    }

    func updateNSView(_ nsView: MTKView, context: Context) {
        renderer.invalidate()
    }
}
