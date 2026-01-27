import Foundation

// MARK: - Parsed Data Types

struct ParsedSnap {
    let windowEndTsNs: Int64
    let midPrice: Double
    let spotRefPriceInt: Int64
    let bookValid: Bool
}

struct ParsedWallRow {
    let windowEndTsNs: Int64
    let relTicks: Int64
    let side: String
    let depthQtyRest: Double
}

struct ParsedVacuumRow {
    let windowEndTsNs: Int64
    let relTicks: Int64
    let vacuumScore: Double
}

struct ParsedPhysicsRow {
    let windowEndTsNs: Int64
    let relTicks: Int64
    let physicsScore: Double
    let physicsScoreSigned: Double
}

struct ParsedGexRow {
    let windowEndTsNs: Int64
    let strikePoints: Double
    let underlyingSpotRef: Double
    let spotRefPriceInt: Int64
    let relTicks: Int64
    let gexAbs: Double
    let gex: Double
    let gexImbalanceRatio: Double
}

// MARK: - Surface Parsers

enum ArrowSurfaceParser {

    /// Parse snap surface (1 row per window).
    /// Schema: window_end_ts_ns (i64), mid_price (f64), spot_ref_price_int (i64), book_valid (bool), __index (i64)
    static func parseSnap(data: Data) -> ParsedSnap? {
        let reader = ArrowIPCReader(data: data)
        guard let (body, numRows, _) = reader.findRecordBatch(), numRows >= 1 else {
            return nil
        }

        var colReader = ArrowColumnReader(body: body, numRows: Int(numRows))

        // Column 0: window_end_ts_ns
        let windowEndTsNs = colReader.readInt64Column()

        // Column 1: mid_price
        let midPrice = colReader.readDoubleColumn()

        // Column 2: spot_ref_price_int
        let spotRefPriceInt = colReader.readInt64Column()

        // Column 3: book_valid
        let bookValid = colReader.readBoolColumn()

        // Column 4: __index_level_0__ (skip)

        guard !windowEndTsNs.isEmpty else { return nil }

        return ParsedSnap(
            windowEndTsNs: windowEndTsNs[0],
            midPrice: midPrice[0],
            spotRefPriceInt: spotRefPriceInt[0],
            bookValid: bookValid[0]
        )
    }

    /// Parse wall surface (variable rows per window).
    /// Schema: window_end_ts_ns (i64), rel_ticks (i64), side (str), depth_qty_rest (f64), __index (i64)
    static func parseWall(data: Data) -> [ParsedWallRow] {
        let reader = ArrowIPCReader(data: data)
        guard let (body, numRows, _) = reader.findRecordBatch(), numRows >= 1 else {
            return []
        }

        var colReader = ArrowColumnReader(body: body, numRows: Int(numRows))

        // Column 0: window_end_ts_ns
        let windowEndTsNs = colReader.readInt64Column()

        // Column 1: rel_ticks
        let relTicks = colReader.readInt64Column()

        // Column 2: side
        let side = colReader.readStringColumn()

        // Column 3: depth_qty_rest
        let depthQtyRest = colReader.readDoubleColumn()

        // Column 4: __index_level_0__ (skip)

        var rows: [ParsedWallRow] = []
        rows.reserveCapacity(Int(numRows))

        for i in 0..<Int(numRows) {
            rows.append(ParsedWallRow(
                windowEndTsNs: windowEndTsNs[i],
                relTicks: relTicks[i],
                side: side[i],
                depthQtyRest: depthQtyRest[i]
            ))
        }

        return rows
    }

    /// Parse vacuum surface (variable rows per window).
    /// Schema: window_end_ts_ns (i64), rel_ticks (i64), vacuum_score (f64), __index (i64)
    static func parseVacuum(data: Data) -> [ParsedVacuumRow] {
        let reader = ArrowIPCReader(data: data)
        guard let (body, numRows, _) = reader.findRecordBatch(), numRows >= 1 else {
            return []
        }

        var colReader = ArrowColumnReader(body: body, numRows: Int(numRows))

        // Column 0: window_end_ts_ns
        let windowEndTsNs = colReader.readInt64Column()

        // Column 1: rel_ticks
        let relTicks = colReader.readInt64Column()

        // Column 2: vacuum_score
        let vacuumScore = colReader.readDoubleColumn()

        // Column 3: __index_level_0__ (skip)

        var rows: [ParsedVacuumRow] = []
        rows.reserveCapacity(Int(numRows))

        for i in 0..<Int(numRows) {
            rows.append(ParsedVacuumRow(
                windowEndTsNs: windowEndTsNs[i],
                relTicks: relTicks[i],
                vacuumScore: vacuumScore[i]
            ))
        }

        return rows
    }

    /// Parse physics surface (variable rows per window).
    /// Schema: window_end_ts_ns (i64), rel_ticks (i64), physics_score (f64), physics_score_signed (f64), __index (i64)
    static func parsePhysics(data: Data) -> [ParsedPhysicsRow] {
        let reader = ArrowIPCReader(data: data)
        guard let (body, numRows, _) = reader.findRecordBatch(), numRows >= 1 else {
            return []
        }

        var colReader = ArrowColumnReader(body: body, numRows: Int(numRows))

        // Column 0: window_end_ts_ns
        let windowEndTsNs = colReader.readInt64Column()

        // Column 1: rel_ticks
        let relTicks = colReader.readInt64Column()

        // Column 2: physics_score
        let physicsScore = colReader.readDoubleColumn()

        // Column 3: physics_score_signed
        let physicsScoreSigned = colReader.readDoubleColumn()

        // Column 4: __index_level_0__ (skip)

        var rows: [ParsedPhysicsRow] = []
        rows.reserveCapacity(Int(numRows))

        for i in 0..<Int(numRows) {
            rows.append(ParsedPhysicsRow(
                windowEndTsNs: windowEndTsNs[i],
                relTicks: relTicks[i],
                physicsScore: physicsScore[i],
                physicsScoreSigned: physicsScoreSigned[i]
            ))
        }

        return rows
    }

    /// Parse GEX surface (variable rows per window).
    /// Schema: window_end_ts_ns (i64), strike_points (f64), underlying_spot_ref (f64),
    ///         spot_ref_price_int (i64), rel_ticks (i64), gex_abs (f64), gex (f64),
    ///         gex_imbalance_ratio (f64), __index (i64)
    static func parseGex(data: Data) -> [ParsedGexRow] {
        let reader = ArrowIPCReader(data: data)
        guard let (body, numRows, _) = reader.findRecordBatch(), numRows >= 1 else {
            return []
        }

        var colReader = ArrowColumnReader(body: body, numRows: Int(numRows))

        // Column 0: window_end_ts_ns
        let windowEndTsNs = colReader.readInt64Column()

        // Column 1: strike_points
        let strikePoints = colReader.readDoubleColumn()

        // Column 2: underlying_spot_ref
        let underlyingSpotRef = colReader.readDoubleColumn()

        // Column 3: spot_ref_price_int
        let spotRefPriceInt = colReader.readInt64Column()

        // Column 4: rel_ticks
        let relTicks = colReader.readInt64Column()

        // Column 5: gex_abs
        let gexAbs = colReader.readDoubleColumn()

        // Column 6: gex
        let gex = colReader.readDoubleColumn()

        // Column 7: gex_imbalance_ratio
        let gexImbalanceRatio = colReader.readDoubleColumn()

        // Column 8: __index_level_0__ (skip)

        var rows: [ParsedGexRow] = []
        rows.reserveCapacity(Int(numRows))

        for i in 0..<Int(numRows) {
            rows.append(ParsedGexRow(
                windowEndTsNs: windowEndTsNs[i],
                strikePoints: strikePoints[i],
                underlyingSpotRef: underlyingSpotRef[i],
                spotRefPriceInt: spotRefPriceInt[i],
                relTicks: relTicks[i],
                gexAbs: gexAbs[i],
                gex: gex[i],
                gexImbalanceRatio: gexImbalanceRatio[i]
            ))
        }

        return rows
    }
}
