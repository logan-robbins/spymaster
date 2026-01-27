import SwiftUI

@main
struct MarketParticlePhysicsApp: App {
    @StateObject private var viewModel = EngineViewModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(viewModel)
                .frame(minWidth: 1200, minHeight: 720)
        }
    }
}
