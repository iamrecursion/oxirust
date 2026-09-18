// SciRS2 Swift bindings for iOS
// Copyright: COOLJAPAN OU (Team KitaSan)
// Version: 0.2.0

import Foundation

/// SciRS2 Error codes
public enum SciError: Int32 {
    case success = 0
    case invalidArgument = 1
    case nullPointer = 2
    case allocationFailed = 3
    case computationFailed = 4
    case unsupportedOperation = 5
}

/// Battery optimization mode
public enum BatteryMode: Int32 {
    case performance = 0
    case balanced = 1
    case powerSaver = 2
}

/// Thermal state
public enum ThermalState: Int32 {
    case normal = 0
    case warm = 1
    case hot = 2
    case critical = 3
}

/// SciRS2 Vector wrapper
public class SciVector {
    private var handle: UnsafeMutablePointer<Float>?
    public let count: Int

    public init(capacity: Int) {
        self.count = capacity
        self.handle = scirs2_vector_create(capacity)?.pointee.data
    }

    deinit {
        if let h = handle {
            // Free vector
            _ = scirs2_vector_destroy(UnsafeMutablePointer(h))
        }
    }

    /// Access element at index
    public subscript(index: Int) -> Float {
        get {
            guard let h = handle, index < count else { return 0.0 }
            return h[index]
        }
        set {
            guard let h = handle, index < count else { return }
            h[index] = newValue
        }
    }

    /// Convert to Array
    public func toArray() -> [Float] {
        guard let h = handle else { return [] }
        return Array(UnsafeBufferPointer(start: h, count: count))
    }
}

/// SciRS2 Matrix wrapper
public class SciMatrix {
    private var handle: UnsafeMutablePointer<Float>?
    public let rows: Int
    public let cols: Int

    public init(rows: Int, cols: Int) {
        self.rows = rows
        self.cols = cols
        self.handle = scirs2_matrix_create(rows, cols)?.pointee.data
    }

    deinit {
        if let h = handle {
            _ = scirs2_matrix_destroy(UnsafeMutablePointer(h))
        }
    }

    /// Access element at row, column
    public subscript(row: Int, col: Int) -> Float {
        get {
            guard let h = handle, row < rows, col < cols else { return 0.0 }
            return h[row * cols + col]
        }
        set {
            guard let h = handle, row < rows, col < cols else { return }
            h[row * cols + col] = newValue
        }
    }
}

/// Main SciRS2 interface
public class SciRS2 {
    private static var initialized = false

    /// Initialize SciRS2 library
    public static func initialize() throws {
        guard !initialized else { return }
        let result = scirs2_init()
        guard result == SciFfiError.success else {
            throw NSError(domain: "SciRS2", code: Int(result.rawValue))
        }
        initialized = true
    }

    /// Shutdown SciRS2 library
    public static func shutdown() {
        guard initialized else { return }
        _ = scirs2_shutdown()
        initialized = false
    }

    /// Get library version
    public static var version: String {
        guard let cStr = scirs2_version() else { return "unknown" }
        return String(cString: cStr)
    }

    /// Check if NEON is available
    public static var hasNEON: Bool {
        return scirs2_has_neon() != 0
    }

    /// Set battery optimization mode
    public static func setBatteryMode(_ mode: BatteryMode) {
        _ = scirs2_set_battery_mode(SciBatteryMode(rawValue: mode.rawValue)!)
    }

    /// Set thermal state
    public static func setThermalState(_ state: ThermalState) {
        _ = scirs2_set_thermal_state(SciThermalState(rawValue: state.rawValue)!)
    }

    /// Vector dot product
    public static func dot(_ a: [Float], _ b: [Float]) -> Float {
        var result: Float = 0.0
        a.withUnsafeBufferPointer { aPtr in
            b.withUnsafeBufferPointer { bPtr in
                _ = scirs2_vector_dot(
                    aPtr.baseAddress,
                    bPtr.baseAddress,
                    min(a.count, b.count),
                    &result
                )
            }
        }
        return result
    }

    /// Vector addition
    public static func add(_ a: [Float], _ b: [Float]) -> [Float] {
        let count = min(a.count, b.count)
        var result = [Float](repeating: 0.0, count: count)
        a.withUnsafeBufferPointer { aPtr in
            b.withUnsafeBufferPointer { bPtr in
                result.withUnsafeMutableBufferPointer { outPtr in
                    _ = scirs2_vector_add(
                        aPtr.baseAddress,
                        bPtr.baseAddress,
                        outPtr.baseAddress,
                        count
                    )
                }
            }
        }
        return result
    }

    /// ReLU activation
    public static func relu(_ x: [Float]) -> [Float] {
        var result = [Float](repeating: 0.0, count: x.count)
        x.withUnsafeBufferPointer { xPtr in
            result.withUnsafeMutableBufferPointer { outPtr in
                _ = scirs2_relu(
                    xPtr.baseAddress,
                    outPtr.baseAddress,
                    x.count
                )
            }
        }
        return result
    }
}

// Example usage:
/*
 do {
     try SciRS2.initialize()

     print("SciRS2 version: \(SciRS2.version)")
     print("NEON available: \(SciRS2.hasNEON)")

     // Set battery mode
     SciRS2.setBatteryMode(.balanced)

     // Vector operations
     let a: [Float] = [1.0, 2.0, 3.0, 4.0]
     let b: [Float] = [1.0, 1.0, 1.0, 1.0]

     let dot = SciRS2.dot(a, b)
     print("Dot product: \(dot)") // 10.0

     let sum = SciRS2.add(a, b)
     print("Sum: \(sum)") // [2.0, 3.0, 4.0, 5.0]

     // Neural network operations
     let activations = SciRS2.relu([-1.0, 0.0, 1.0, 2.0])
     print("ReLU: \(activations)") // [0.0, 0.0, 1.0, 2.0]

     SciRS2.shutdown()
 } catch {
     print("Error: \(error)")
 }
 */
