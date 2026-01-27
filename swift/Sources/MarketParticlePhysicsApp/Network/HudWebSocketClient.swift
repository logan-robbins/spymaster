import Foundation

/// Batch of parsed surface data for a single window.
struct HudBatch {
    let windowEndTsNs: Int64
    var snap: ParsedSnap?
    var wall: [ParsedWallRow] = []
    var vacuum: [ParsedVacuumRow] = []
    var physics: [ParsedPhysicsRow] = []
    var gex: [ParsedGexRow] = []
}

/// Delegate protocol for HUD WebSocket events.
protocol HudWebSocketClientDelegate: AnyObject {
    func hudWebSocketClient(_ client: HudWebSocketClient, didReceiveBatch batch: HudBatch)
    func hudWebSocketClient(_ client: HudWebSocketClient, didConnect url: URL)
    func hudWebSocketClient(_ client: HudWebSocketClient, didDisconnectWithError error: Error?)
}

/// WebSocket client for HUD streaming data.
/// Connects to backend, parses JSON control frames and Arrow IPC binary frames.
final class HudWebSocketClient: NSObject {
    weak var delegate: HudWebSocketClientDelegate?

    private let symbol: String
    private let dt: String
    private let speed: Double

    private var urlSession: URLSession?
    private var webSocketTask: URLSessionWebSocketTask?

    private var pendingSurface: String?
    private var currentBatch: HudBatch?
    private var expectedSurfaces: Set<String> = []
    private var receivedSurfaces: Set<String> = []

    init(symbol: String = "ESH6", dt: String = "2026-01-06", speed: Double = 1.0) {
        self.symbol = symbol
        self.dt = dt
        self.speed = speed
        super.init()
    }

    func connect() {
        let urlString = "ws://localhost:8000/v1/hud/stream?symbol=\(symbol)&dt=\(dt)&speed=\(speed)"
        guard let url = URL(string: urlString) else {
            print("[HudWS] Invalid URL: \(urlString)")
            return
        }

        let config = URLSessionConfiguration.default
        config.waitsForConnectivity = true
        urlSession = URLSession(configuration: config, delegate: self, delegateQueue: .main)
        webSocketTask = urlSession?.webSocketTask(with: url)
        webSocketTask?.resume()

        print("[HudWS] Connecting to \(urlString)")
        receiveNextMessage()
    }

    func disconnect() {
        webSocketTask?.cancel(with: .goingAway, reason: nil)
        webSocketTask = nil
        urlSession?.invalidateAndCancel()
        urlSession = nil
    }

    private func receiveNextMessage() {
        webSocketTask?.receive { [weak self] result in
            guard let self = self else { return }

            switch result {
            case .success(let message):
                self.handleMessage(message)
                self.receiveNextMessage()

            case .failure(let error):
                print("[HudWS] Receive error: \(error)")
                self.delegate?.hudWebSocketClient(self, didDisconnectWithError: error)
            }
        }
    }

    private func handleMessage(_ message: URLSessionWebSocketTask.Message) {
        switch message {
        case .string(let text):
            handleJSONFrame(text)
        case .data(let data):
            handleBinaryFrame(data)
        @unknown default:
            break
        }
    }

    private func handleJSONFrame(_ text: String) {
        guard let jsonData = text.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any],
              let type = json["type"] as? String else {
            return
        }

        switch type {
        case "batch_start":
            // Start new batch
            if let windowEndStr = json["window_end_ts_ns"] as? String,
               let windowEndTsNs = Int64(windowEndStr) {

                // Emit previous batch if complete
                if let batch = currentBatch {
                    delegate?.hudWebSocketClient(self, didReceiveBatch: batch)
                }

                currentBatch = HudBatch(windowEndTsNs: windowEndTsNs)

                if let surfaces = json["surfaces"] as? [String] {
                    expectedSurfaces = Set(surfaces)
                    receivedSurfaces = []
                }
            }

        case "surface_header":
            if let surface = json["surface"] as? String {
                pendingSurface = surface
            }

        default:
            break
        }
    }

    private func handleBinaryFrame(_ data: Data) {
        guard let surface = pendingSurface else {
            print("[HudWS] Received binary without surface header")
            return
        }

        pendingSurface = nil
        receivedSurfaces.insert(surface)

        // Parse Arrow IPC based on surface type
        switch surface {
        case "snap":
            if let snap = ArrowSurfaceParser.parseSnap(data: data) {
                currentBatch?.snap = snap
            }

        case "wall":
            let rows = ArrowSurfaceParser.parseWall(data: data)
            currentBatch?.wall = rows

        case "vacuum":
            let rows = ArrowSurfaceParser.parseVacuum(data: data)
            currentBatch?.vacuum = rows

        case "physics":
            let rows = ArrowSurfaceParser.parsePhysics(data: data)
            currentBatch?.physics = rows

        case "gex":
            let rows = ArrowSurfaceParser.parseGex(data: data)
            currentBatch?.gex = rows

        default:
            // Skip unknown surfaces (radar, bucket_radar, etc.)
            break
        }

        // Check if batch is complete
        checkBatchComplete()
    }

    private func checkBatchComplete() {
        // We consider a batch complete when we've received all expected surfaces
        // For simplicity, we emit on snap since it's always last or we've received all
        guard let batch = currentBatch else { return }

        // Emit if we have snap (primary data)
        if batch.snap != nil {
            // Batch will be emitted on next batch_start
        }
    }
}

// MARK: - URLSessionWebSocketDelegate

extension HudWebSocketClient: URLSessionWebSocketDelegate {
    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask, didOpenWithProtocol protocol: String?) {
        print("[HudWS] Connected")
        if let url = webSocketTask.originalRequest?.url {
            delegate?.hudWebSocketClient(self, didConnect: url)
        }
    }

    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask, didCloseWith closeCode: URLSessionWebSocketTask.CloseCode, reason: Data?) {
        let reasonStr = reason.flatMap { String(data: $0, encoding: .utf8) } ?? "unknown"
        print("[HudWS] Disconnected: \(closeCode) - \(reasonStr)")
        delegate?.hudWebSocketClient(self, didDisconnectWithError: nil)
    }
}
