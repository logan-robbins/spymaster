import SwiftUI
import Charts

struct ContentView: View {
    @EnvironmentObject private var viewModel: EngineViewModel

    var body: some View {
        VStack(spacing: 12) {
            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Trajectory")
                        .font(.headline)
                    Chart {
                        ForEach(viewModel.actualSeries) { point in
                            LineMark(
                                x: .value("Time", point.timestamp),
                                y: .value("Actual", point.value)
                            )
                            .foregroundStyle(.cyan)
                        }
                        ForEach(viewModel.predictionSeries.keys.sorted(), id: \ .self) { horizon in
                            if let series = viewModel.predictionSeries[horizon] {
                                ForEach(series) { point in
                                    LineMark(
                                        x: .value("Time", point.timestamp),
                                        y: .value("Predicted", point.value)
                                    )
                                    .foregroundStyle(viewModel.horizonColor(horizon))
                                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 4]))
                                }
                            }
                        }
                    }
                    .chartXAxis {
                        AxisMarks(values: .automatic(desiredCount: 6))
                    }
                    .chartYAxis {
                        AxisMarks(values: .automatic(desiredCount: 5))
                    }
                    .frame(minHeight: 280)
                }
                .frame(maxWidth: .infinity)

                VStack(alignment: .leading, spacing: 8) {
                    Text("Physics View")
                        .font(.headline)
                    MetalView(renderer: viewModel.renderer)
                        .frame(minWidth: 500, minHeight: 280)
                        .background(Color.black.opacity(0.9))
                        .cornerRadius(8)
                }
                .frame(maxWidth: .infinity)
            }

            ControlsPanel()
                .padding(.top, 8)
        }
        .padding(16)
        .onAppear {
            viewModel.startIfNeeded()
        }
    }
}

struct ControlsPanel: View {
    @EnvironmentObject private var viewModel: EngineViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 16) {
                Button(viewModel.isRunning ? "Stop" : "Start") {
                    viewModel.toggleRun()
                }
                Button("Replay") {
                    viewModel.replay()
                }
                Toggle("Live Mode", isOn: $viewModel.isLiveMode)
                    .toggleStyle(.switch)
                Toggle("Show Predictions", isOn: $viewModel.showPredictions)
                    .toggleStyle(.switch)
            }

            HStack(spacing: 16) {
                Text("Weights")
                    .font(.headline)
                ForEach(viewModel.weightSummaries, id: \ .name) { item in
                    VStack(alignment: .leading) {
                        Text(item.name)
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text(String(format: "%.3f", item.value))
                            .font(.system(.body, design: .monospaced))
                    }
                }
            }

            HStack(spacing: 16) {
                Text("Diagnostics")
                    .font(.headline)
                Text(viewModel.statusLine)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundColor(.secondary)
            }
        }
        .padding(12)
        .background(Color.gray.opacity(0.08))
        .cornerRadius(8)
    }
}

#Preview {
    ContentView()
        .environmentObject(EngineViewModel())
}
