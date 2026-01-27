import MetalKit
import SwiftUI

struct MetalView: NSViewRepresentable {
    let renderer: MetalRenderer
    let onHover: (Float?) -> Void

    func makeNSView(context: Context) -> TrackingMTKView {
        let view = TrackingMTKView()
        view.device = renderer.device
        view.clearColor = MTLClearColorMake(0.05, 0.05, 0.08, 1.0)
        view.colorPixelFormat = .bgra8Unorm
        view.preferredFramesPerSecond = 60
        view.enableSetNeedsDisplay = false
        view.isPaused = false
        view.onHover = { location in
            let normalized = view.normalizedX(for: location)
            onHover(normalized)
        }
        view.onExit = {
            onHover(nil)
        }
        renderer.attach(view: view)
        return view
    }

    func updateNSView(_ nsView: TrackingMTKView, context: Context) {
        renderer.invalidate()
    }
}

final class TrackingMTKView: MTKView {
    var onHover: ((CGPoint) -> Void)?
    var onExit: (() -> Void)?
    private var trackingArea: NSTrackingArea?

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        if let trackingArea {
            removeTrackingArea(trackingArea)
        }
        let options: NSTrackingArea.Options = [.mouseMoved, .activeAlways, .inVisibleRect, .mouseEnteredAndExited]
        let area = NSTrackingArea(rect: bounds, options: options, owner: self, userInfo: nil)
        addTrackingArea(area)
        trackingArea = area
    }

    override func mouseMoved(with event: NSEvent) {
        let location = convert(event.locationInWindow, from: nil)
        onHover?(location)
    }

    override func mouseExited(with event: NSEvent) {
        onExit?()
    }

    func normalizedX(for location: CGPoint) -> Float {
        guard bounds.width > 0 else { return 0 }
        let ratio = location.x / bounds.width
        let normalized = (ratio * 2.0) - 1.0
        return Float(max(-1.0, min(1.0, normalized)))
    }
}
