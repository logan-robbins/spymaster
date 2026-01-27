import Foundation

/// Low-level Arrow IPC stream reader.
/// Parses continuation markers, metadata, and body buffers.
struct ArrowIPCReader {
    private let data: Data

    init(data: Data) {
        self.data = data
    }

    /// Find the RecordBatch message and return body data with row count.
    func findRecordBatch() -> (body: Data, numRows: Int64, bodyLength: Int64)? {
        var offset = 0

        while offset < data.count - 8 {
            // Check continuation marker (0xFFFFFFFF)
            let continuation = readUInt32(at: offset)
            guard continuation == 0xFFFFFFFF else {
                return nil
            }
            offset += 4

            // Metadata length
            let metadataLength = Int(readInt32(at: offset))
            offset += 4

            guard metadataLength > 0, offset + metadataLength <= data.count else {
                return nil
            }

            let metadataStart = offset
            let metadataEnd = offset + metadataLength

            // Parse FlatBuffers to get message type
            let (messageType, bodyLength, numRows) = parseMessageMetadata(
                metadataStart: metadataStart,
                metadataLength: metadataLength
            )

            // Align to 8 bytes
            offset = metadataEnd
            let padding = (8 - (offset % 8)) % 8
            offset += padding

            // If this is a RecordBatch (type 3), extract body
            if messageType == 3 {
                let bodyStart = offset
                let bodyEnd = min(bodyStart + Int(bodyLength), data.count)
                let body = data.subdata(in: bodyStart..<bodyEnd)
                return (body: body, numRows: numRows, bodyLength: bodyLength)
            }

            // Skip body if present (for Schema messages, bodyLength is 0)
            offset += Int(bodyLength)
        }

        return nil
    }

    private func parseMessageMetadata(metadataStart: Int, metadataLength: Int) -> (type: UInt8, bodyLength: Int64, numRows: Int64) {
        // FlatBuffers structure for Arrow Message:
        // Root table has: version, header_type, header (union), bodyLength

        // Read root table offset
        let rootOffset = Int(readInt32(at: metadataStart))
        let rootPos = metadataStart + rootOffset

        // Read vtable offset (negative from root)
        let vtableOffset = Int(readInt32(at: rootPos))
        let vtableStart = rootPos - vtableOffset

        // vtable: [vtable_size (u16), object_size (u16), field_offsets (u16)...]
        let vtableSize = Int(readUInt16(at: vtableStart))

        var messageType: UInt8 = 0
        var bodyLength: Int64 = 0
        var numRows: Int64 = 0

        // Field 1 (index 0): version (skip)
        // Field 2 (index 1): header_type
        if vtableSize >= 8 {
            let headerTypeFieldOffset = Int(readUInt16(at: vtableStart + 6))
            if headerTypeFieldOffset > 0 {
                messageType = data[rootPos + headerTypeFieldOffset]
            }
        }

        // Field 4 (index 3): bodyLength
        if vtableSize >= 12 {
            let bodyLengthFieldOffset = Int(readUInt16(at: vtableStart + 10))
            if bodyLengthFieldOffset > 0 {
                bodyLength = readInt64(at: rootPos + bodyLengthFieldOffset)
            }
        }

        // For RecordBatch, parse the header union to get row count
        if messageType == 3 && vtableSize >= 10 {
            let headerFieldOffset = Int(readUInt16(at: vtableStart + 8))
            if headerFieldOffset > 0 {
                // header is a union offset to RecordBatch table
                let headerOffset = Int(readInt32(at: rootPos + headerFieldOffset))
                let recordBatchPos = rootPos + headerFieldOffset + headerOffset

                // RecordBatch table: length (int64), nodes (vector), buffers (vector)
                let rbVtableOffset = Int(readInt32(at: recordBatchPos))
                let rbVtableStart = recordBatchPos - rbVtableOffset
                let rbVtableSize = Int(readUInt16(at: rbVtableStart))

                // Field 0: length (row count)
                if rbVtableSize >= 6 {
                    let lengthFieldOffset = Int(readUInt16(at: rbVtableStart + 4))
                    if lengthFieldOffset > 0 {
                        numRows = readInt64(at: recordBatchPos + lengthFieldOffset)
                    }
                }
            }
        }

        return (type: messageType, bodyLength: bodyLength, numRows: numRows)
    }

    // MARK: - Primitive readers

    private func readUInt16(at offset: Int) -> UInt16 {
        guard offset + 2 <= data.count else { return 0 }
        return data.withUnsafeBytes { ptr in
            ptr.load(fromByteOffset: offset, as: UInt16.self)
        }
    }

    private func readInt32(at offset: Int) -> Int32 {
        guard offset + 4 <= data.count else { return 0 }
        return data.withUnsafeBytes { ptr in
            ptr.load(fromByteOffset: offset, as: Int32.self)
        }
    }

    private func readUInt32(at offset: Int) -> UInt32 {
        guard offset + 4 <= data.count else { return 0 }
        return data.withUnsafeBytes { ptr in
            ptr.load(fromByteOffset: offset, as: UInt32.self)
        }
    }

    private func readInt64(at offset: Int) -> Int64 {
        guard offset + 8 <= data.count else { return 0 }
        return data.withUnsafeBytes { ptr in
            ptr.load(fromByteOffset: offset, as: Int64.self)
        }
    }
}

/// Column buffer reader for Arrow RecordBatch body data.
struct ArrowColumnReader {
    let body: Data
    let numRows: Int
    private(set) var offset: Int = 0

    init(body: Data, numRows: Int) {
        self.body = body
        self.numRows = numRows
    }

    /// Read a column of Int64 values.
    mutating func readInt64Column() -> [Int64] {
        var values: [Int64] = []
        values.reserveCapacity(numRows)

        for _ in 0..<numRows {
            let value = body.withUnsafeBytes { ptr in
                ptr.load(fromByteOffset: offset, as: Int64.self)
            }
            values.append(value)
            offset += 8
        }

        return values
    }

    /// Read a column of Double values.
    mutating func readDoubleColumn() -> [Double] {
        var values: [Double] = []
        values.reserveCapacity(numRows)

        for _ in 0..<numRows {
            let value = body.withUnsafeBytes { ptr in
                ptr.load(fromByteOffset: offset, as: Double.self)
            }
            values.append(value)
            offset += 8
        }

        return values
    }

    /// Read a column of Bool values (packed bits).
    mutating func readBoolColumn() -> [Bool] {
        var values: [Bool] = []
        values.reserveCapacity(numRows)

        let byteCount = (numRows + 7) / 8
        for i in 0..<numRows {
            let byteIndex = i / 8
            let bitIndex = i % 8
            let byte = body[offset + byteIndex]
            let value = (byte >> bitIndex) & 1 == 1
            values.append(value)
        }

        // Advance offset with 8-byte alignment
        offset += byteCount
        let padding = (8 - (offset % 8)) % 8
        offset += padding

        return values
    }

    /// Read a column of UTF-8 strings (offset + data format).
    mutating func readStringColumn() -> [String] {
        // Read offsets array (N+1 int32s)
        var offsets: [Int32] = []
        offsets.reserveCapacity(numRows + 1)

        for _ in 0..<(numRows + 1) {
            let value = body.withUnsafeBytes { ptr in
                ptr.load(fromByteOffset: offset, as: Int32.self)
            }
            offsets.append(value)
            offset += 4
        }

        // Align to 8 bytes
        let padding1 = (8 - (offset % 8)) % 8
        offset += padding1

        // Read string data
        let stringDataLength = Int(offsets.last ?? 0)
        let stringData = body.subdata(in: offset..<(offset + stringDataLength))
        offset += stringDataLength

        // Align to 8 bytes
        let padding2 = (8 - (offset % 8)) % 8
        offset += padding2

        // Decode strings
        var strings: [String] = []
        strings.reserveCapacity(numRows)

        for i in 0..<numRows {
            let start = Int(offsets[i])
            let end = Int(offsets[i + 1])
            let slice = stringData.subdata(in: start..<end)
            let str = String(data: slice, encoding: .utf8) ?? ""
            strings.append(str)
        }

        return strings
    }

    /// Skip a column (for columns we don't need like __index_level_0__).
    mutating func skipInt64Column() {
        offset += numRows * 8
    }
}
