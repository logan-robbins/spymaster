import XCTest
@testable import MarketParticlePhysicsApp

final class ArrowParserTests: XCTestCase {

    func testParseSnapFromSampleFile() throws {
        // Load sample Arrow IPC file captured from backend
        let samplePath = "/tmp/snap_sample.arrow"
        guard FileManager.default.fileExists(atPath: samplePath) else {
            print("Skipping test - sample file not found at \(samplePath)")
            return
        }

        let data = try Data(contentsOf: URL(fileURLWithPath: samplePath))
        XCTAssertGreaterThan(data.count, 0, "Sample data should not be empty")

        let snap = ArrowSurfaceParser.parseSnap(data: data)
        XCTAssertNotNil(snap, "Should parse snap successfully")

        guard let snap = snap else { return }

        // Verify expected values from 2026-01-06 ESH6 data
        XCTAssertEqual(snap.windowEndTsNs, 1767709800000000000, "Window timestamp should match")
        XCTAssertEqual(snap.midPrice, 6946.625, accuracy: 0.001, "Mid price should be ~6946.625")
        XCTAssertEqual(snap.spotRefPriceInt, 6945000000000, "Spot ref price int should match")
        XCTAssertTrue(snap.bookValid, "Book should be valid")

        print("Parsed snap: midPrice=\(snap.midPrice), spotRefPriceInt=\(snap.spotRefPriceInt)")
    }

    func testParseWallFromSampleFile() throws {
        let samplePath = "/tmp/wall_sample.arrow"
        guard FileManager.default.fileExists(atPath: samplePath) else {
            print("Skipping test - sample file not found at \(samplePath)")
            return
        }

        let data = try Data(contentsOf: URL(fileURLWithPath: samplePath))
        XCTAssertGreaterThan(data.count, 0, "Sample data should not be empty")

        let rows = ArrowSurfaceParser.parseWall(data: data)
        XCTAssertGreaterThan(rows.count, 0, "Should parse wall rows")

        // Expected: 61 rows from sample
        XCTAssertEqual(rows.count, 61, "Should have 61 rows")

        // Check first row
        let first = rows[0]
        XCTAssertEqual(first.relTicks, -44, "First row rel_ticks should be -44")
        XCTAssertEqual(first.side, "B", "First row side should be B (bid)")

        print("Parsed \(rows.count) wall rows")
        print("Sample rows:")
        for row in rows.prefix(5) {
            print("  relTicks=\(row.relTicks) side=\(row.side) depth=\(row.depthQtyRest)")
        }
    }

    func testArrowIPCReaderFindsRecordBatch() throws {
        let samplePath = "/tmp/snap_sample.arrow"
        guard FileManager.default.fileExists(atPath: samplePath) else {
            print("Skipping test - sample file not found")
            return
        }

        let data = try Data(contentsOf: URL(fileURLWithPath: samplePath))
        let reader = ArrowIPCReader(data: data)

        let result = reader.findRecordBatch()
        XCTAssertNotNil(result, "Should find RecordBatch")

        guard let (body, numRows, bodyLength) = result else { return }

        XCTAssertEqual(numRows, 1, "Snap should have 1 row")
        XCTAssertEqual(bodyLength, 40, "Body should be 40 bytes")
        XCTAssertEqual(body.count, 40, "Body data should be 40 bytes")

        print("RecordBatch: numRows=\(numRows), bodyLength=\(bodyLength)")
    }
}
