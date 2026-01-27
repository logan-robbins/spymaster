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
                        ForEach(viewModel.predictionSeries.keys.sorted(), id: \.self) { horizon in
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
                    ZStack {
                        MetalView(renderer: viewModel.renderer) { normalizedX in
                            viewModel.updateHover(normalizedX: normalizedX)
                        }
                        PhysicsOverlayView()
                    }
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
        .onChange(of: viewModel.zoomFactor) { _ in
            if let snapshot = viewModel.currentSnapshotForDisplay() {
                viewModel.refreshRenderer(snapshot: snapshot)
            }
        }
        .onChange(of: viewModel.showPredictions) { _ in
            if let snapshot = viewModel.currentSnapshotForDisplay() {
                viewModel.refreshRenderer(snapshot: snapshot)
            }
        }
    }
}

struct ControlsPanel: View {
    @EnvironmentObject private var viewModel: EngineViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 16) {
                Picker("Source", selection: Binding(
                    get: { viewModel.dataSourceMode },
                    set: { viewModel.setDataSourceMode($0) }
                )) {
                    ForEach(DataSourceMode.allCases, id: \.self) { mode in
                        Text(mode.rawValue).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 140)

                Button(viewModel.isRunning ? "Stop" : "Start") {
                    viewModel.toggleRun()
                }
                Button(viewModel.isFrozen ? "Resume" : "Freeze") {
                    if viewModel.isFrozen {
                        viewModel.resume()
                    } else {
                        viewModel.freeze()
                    }
                }
                Button("Step") {
                    viewModel.stepOnce()
                }
                Button("Replay") {
                    viewModel.replay()
                }
                Toggle("Live Mode", isOn: Binding(get: { viewModel.isLiveMode }, set: { viewModel.setLiveMode($0) }))
                    .toggleStyle(.switch)
                Toggle("Show Predictions", isOn: $viewModel.showPredictions)
                    .toggleStyle(.switch)
                Toggle("Debug", isOn: $viewModel.showDebugOverlay)
                    .toggleStyle(.switch)
            }

            HStack(spacing: 16) {
                Text("Weights")
                    .font(.headline)
                ForEach(viewModel.weightSummaries, id: \.name) { item in
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
                VStack(alignment: .leading) {
                    Text(viewModel.statusLine)
                        .font(.system(.caption, design: .monospaced))
                        .foregroundColor(.secondary)
                    if !viewModel.hoverReadout.isEmpty {
                        Text(viewModel.hoverReadout)
                            .font(.system(.caption2, design: .monospaced))
                            .foregroundColor(.secondary)
                    }
                }
            }

            HStack(spacing: 16) {
                Text("Scrub")
                    .font(.headline)
                Slider(
                    value: $viewModel.scrubIndex,
                    in: 0...Double(max(viewModel.historyCount - 1, 1)),
                    step: 1
                )
                .disabled(viewModel.isLiveMode || viewModel.historyCount < 2)
                .onChange(of: viewModel.scrubIndex) { newValue in
                    viewModel.scrub(to: newValue)
                }
                Text(viewModel.scrubLabel)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundColor(.secondary)
            }

            HStack(spacing: 16) {
                Text("Zoom X")
                    .font(.headline)
                Slider(value: $viewModel.zoomFactor, in: 1...4, step: 0.1)
                Text(String(format: "%.1fx", viewModel.zoomFactor))
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

struct PhysicsOverlayView: View {
    @EnvironmentObject private var viewModel: EngineViewModel

    var body: some View {
        GeometryReader { geo in
            let width = geo.size.width
            let height = geo.size.height
            let bandAHeight = height * 0.65
            let bandBHeight = height * 0.20
            let bandCHeight = height * 0.15
            let bandBTop = bandAHeight
            let bandCTop = bandAHeight + bandBHeight
            let rowHeightB = bandBHeight / CGFloat(viewModel.horizonCount)
            let rowHeightC = bandCHeight / CGFloat(viewModel.horizonCount)

            ZStack(alignment: .topLeading) {
                Text("Spot (t0)")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundColor(.white.opacity(0.6))
                    .position(x: width * 0.5, y: 12)

                ForEach(Array(1...viewModel.horizonCount), id: \.self) { horizon in
                    let label = String(format: "+%.1fs", Double(horizon) * viewModel.dtSecondsValue)
                    Text(label)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundColor(.white.opacity(0.6))
                        .position(x: 34, y: bandBTop + rowHeightB * (CGFloat(horizon) - 0.5))
                }

                if let resolved = viewModel.resolvedOrigin {
                    ForEach(resolved.resolved, id: \.horizon) { item in
                        let error = item.errorTicks
                        if abs(error) >= 2 {
                            let offset = item.predictedTicks - resolved.originSpotTicks
                            let actualOffset = item.actualTicks - resolved.originSpotTicks
                            let center = (offset + actualOffset) * 0.5
                            let x = viewModel.mapTicksToX(center, width: width)
                            let y = bandCTop + rowHeightC * (CGFloat(item.horizon) - 0.5)
                            Text(String(format: "%.1f", error))
                                .font(.system(size: 10, design: .monospaced))
                                .foregroundColor(.white)
                                .position(x: x, y: y)
                        }
                    }
                }

                if viewModel.showDebugOverlay {
                    Text(viewModel.debugOverlayText)
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundColor(.white.opacity(0.8))
                        .padding(6)
                }
            }
            .frame(width: width, height: height, alignment: .topLeading)
        }
        .allowsHitTesting(false)
    }
}
