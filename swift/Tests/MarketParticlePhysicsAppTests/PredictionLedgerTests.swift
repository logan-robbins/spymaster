import XCTest
@testable import MarketParticlePhysicsApp

final class PredictionLedgerTests: XCTestCase {
    func testLedgerResolvesOriginWhenAllHorizonsArrive() {
        let ledger = PredictionLedger()
        let originTs: TimestampNs = 1_000
        let dt: TimestampNs = 500
        let horizonCount = 3

        let records = (1...horizonCount).map { h -> PredictionRecord in
            PredictionRecord(
                originTimestampNs: originTs,
                targetTimestampNs: originTs + (TimestampNs(h) * dt),
                horizon: h,
                predictedTicks: Float(h) * 2.0,
                forces: ForceContribs(fU: 1, fC: 0, fG: 0, fO: 0, bias: 1)
            )
        }

        ledger.store(originTimestampNs: originTs, originSpotTicks: 100, horizonCount: horizonCount, records: records)

        var resolvedOrigin: ResolvedOriginFrame?
        for h in 1...horizonCount {
            let resolution = ledger.resolve(timestampNs: originTs + TimestampNs(h) * dt, actualTicks: Float(h) * 2.5)
            resolvedOrigin = resolution.resolvedOrigin ?? resolvedOrigin
        }

        XCTAssertNotNil(resolvedOrigin)
        XCTAssertEqual(resolvedOrigin?.resolved.count, horizonCount)
        XCTAssertEqual(resolvedOrigin?.originTimestampNs, originTs)
    }
}
