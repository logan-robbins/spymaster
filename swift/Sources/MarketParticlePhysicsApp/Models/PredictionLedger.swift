import Foundation

final class PredictionLedger {
    private struct TargetLink {
        let originTimestampNs: TimestampNs
        let horizon: Int
    }

    private struct OriginEntry {
        let originTimestampNs: TimestampNs
        let originSpotTicks: Float
        let horizonCount: Int
        var records: [Int: PredictionRecord]
        var resolved: [Int: ResolvedPrediction]
    }

    private var pendingByOrigin: [TimestampNs: OriginEntry] = [:]
    private var pendingByTarget: [TimestampNs: [TargetLink]] = [:]
    private(set) var latestResolvedOrigin: ResolvedOriginFrame?

    func store(originTimestampNs: TimestampNs, originSpotTicks: Float, horizonCount: Int, records: [PredictionRecord]) {
        var entry = OriginEntry(
            originTimestampNs: originTimestampNs,
            originSpotTicks: originSpotTicks,
            horizonCount: horizonCount,
            records: [:],
            resolved: [:]
        )
        for record in records {
            entry.records[record.horizon] = record
            pendingByTarget[record.targetTimestampNs, default: []].append(TargetLink(originTimestampNs: originTimestampNs, horizon: record.horizon))
        }
        pendingByOrigin[originTimestampNs] = entry
    }

    func resolve(timestampNs: TimestampNs, actualTicks: Float) -> (resolved: [ResolvedPrediction], resolvedOrigin: ResolvedOriginFrame?) {
        guard let links = pendingByTarget.removeValue(forKey: timestampNs) else {
            return ([], nil)
        }

        var resolvedPredictions: [ResolvedPrediction] = []
        var completedOrigin: ResolvedOriginFrame?

        for link in links {
            guard var entry = pendingByOrigin[link.originTimestampNs],
                  let record = entry.records[link.horizon] else {
                continue
            }

            let error = actualTicks - record.predictedTicks
            let resolved = ResolvedPrediction(
                horizon: record.horizon,
                errorTicks: error,
                predictedTicks: record.predictedTicks,
                actualTicks: actualTicks,
                forces: record.forces
            )
            entry.resolved[record.horizon] = resolved
            pendingByOrigin[link.originTimestampNs] = entry
            resolvedPredictions.append(resolved)

            if entry.resolved.count == entry.horizonCount {
                let resolvedList = entry.resolved.values.sorted { $0.horizon < $1.horizon }
                let originFrame = ResolvedOriginFrame(
                    originTimestampNs: entry.originTimestampNs,
                    originSpotTicks: entry.originSpotTicks,
                    resolved: resolvedList
                )
                latestResolvedOrigin = originFrame
                completedOrigin = originFrame
                pendingByOrigin.removeValue(forKey: entry.originTimestampNs)
            }
        }

        return (resolvedPredictions, completedOrigin)
    }

    func reset() {
        pendingByOrigin.removeAll()
        pendingByTarget.removeAll()
        latestResolvedOrigin = nil
    }
}
