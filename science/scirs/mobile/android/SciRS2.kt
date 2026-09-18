// SciRS2 Kotlin/JNI bindings for Android
// Copyright: COOLJAPAN OU (Team KitaSan)
// Version: 0.2.0

package com.cooljapan.scirs2

import androidx.annotation.Keep

/**
 * SciRS2 error codes
 */
enum class SciError(val code: Int) {
    SUCCESS(0),
    INVALID_ARGUMENT(1),
    NULL_POINTER(2),
    ALLOCATION_FAILED(3),
    COMPUTATION_FAILED(4),
    UNSUPPORTED_OPERATION(5);

    companion object {
        fun fromCode(code: Int): SciError = values().find { it.code == code } ?: UNSUPPORTED_OPERATION
    }
}

/**
 * Battery optimization mode
 */
enum class BatteryMode(val code: Int) {
    PERFORMANCE(0),
    BALANCED(1),
    POWER_SAVER(2)
}

/**
 * Thermal state
 */
enum class ThermalState(val code: Int) {
    NORMAL(0),
    WARM(1),
    HOT(2),
    CRITICAL(3)
}

/**
 * SciRS2 exception
 */
class SciException(val error: SciError, message: String? = null) :
    Exception(message ?: "SciRS2 error: ${error.name}")

/**
 * Main SciRS2 interface for Android
 */
@Keep
object SciRS2 {
    private var initialized = false

    init {
        System.loadLibrary("scirs2_core")
    }

    /**
     * Initialize SciRS2 library
     */
    @JvmStatic
    fun initialize() {
        if (initialized) return
        val result = nativeInit()
        if (result != 0) {
            throw SciException(SciError.fromCode(result), "Failed to initialize SciRS2")
        }
        initialized = true
    }

    /**
     * Shutdown SciRS2 library
     */
    @JvmStatic
    fun shutdown() {
        if (!initialized) return
        nativeShutdown()
        initialized = false
    }

    /**
     * Get library version
     */
    @JvmStatic
    val version: String
        get() = nativeVersion() ?: "unknown"

    /**
     * Check if NEON is available
     */
    @JvmStatic
    val hasNEON: Boolean
        get() = nativeHasNeon() != 0

    /**
     * Set battery optimization mode
     */
    @JvmStatic
    fun setBatteryMode(mode: BatteryMode) {
        nativeSetBatteryMode(mode.code)
    }

    /**
     * Set thermal state
     */
    @JvmStatic
    fun setThermalState(state: ThermalState) {
        nativeSetThermalState(state.code)
    }

    /**
     * Vector dot product
     */
    @JvmStatic
    fun dot(a: FloatArray, b: FloatArray): Float {
        require(a.size == b.size) { "Vector sizes must match" }
        val result = FloatArray(1)
        val error = nativeVectorDot(a, b, a.size, result)
        if (error != 0) {
            throw SciException(SciError.fromCode(error), "Dot product failed")
        }
        return result[0]
    }

    /**
     * Vector addition
     */
    @JvmStatic
    fun add(a: FloatArray, b: FloatArray): FloatArray {
        val size = minOf(a.size, b.size)
        val result = FloatArray(size)
        val error = nativeVectorAdd(a, b, result, size)
        if (error != 0) {
            throw SciException(SciError.fromCode(error), "Vector addition failed")
        }
        return result
    }

    /**
     * Vector multiplication (element-wise)
     */
    @JvmStatic
    fun multiply(a: FloatArray, b: FloatArray): FloatArray {
        val size = minOf(a.size, b.size)
        val result = FloatArray(size)
        val error = nativeVectorMul(a, b, result, size)
        if (error != 0) {
            throw SciException(SciError.fromCode(error), "Vector multiplication failed")
        }
        return result
    }

    /**
     * ReLU activation
     */
    @JvmStatic
    fun relu(x: FloatArray): FloatArray {
        val result = FloatArray(x.size)
        val error = nativeRelu(x, result, x.size)
        if (error != 0) {
            throw SciException(SciError.fromCode(error), "ReLU activation failed")
        }
        return result
    }

    /**
     * Sigmoid activation
     */
    @JvmStatic
    fun sigmoid(x: FloatArray): FloatArray {
        val result = FloatArray(x.size)
        val error = nativeSigmoid(x, result, x.size)
        if (error != 0) {
            throw SciException(SciError.fromCode(error), "Sigmoid activation failed")
        }
        return result
    }

    /**
     * Battery-optimized dot product
     */
    @JvmStatic
    fun dotBatteryOptimized(a: FloatArray, b: FloatArray): Float {
        require(a.size == b.size) { "Vector sizes must match" }
        val result = FloatArray(1)
        val error = nativeVectorDotBatteryOptimized(a, b, a.size, result)
        if (error != 0) {
            throw SciException(SciError.fromCode(error), "Battery-optimized dot product failed")
        }
        return result[0]
    }

    // Native method declarations
    @JvmStatic
    private external fun nativeInit(): Int

    @JvmStatic
    private external fun nativeShutdown(): Int

    @JvmStatic
    private external fun nativeVersion(): String?

    @JvmStatic
    private external fun nativeHasNeon(): Int

    @JvmStatic
    private external fun nativeSetBatteryMode(mode: Int): Int

    @JvmStatic
    private external fun nativeSetThermalState(state: Int): Int

    @JvmStatic
    private external fun nativeVectorDot(
        a: FloatArray,
        b: FloatArray,
        len: Int,
        result: FloatArray
    ): Int

    @JvmStatic
    private external fun nativeVectorAdd(
        a: FloatArray,
        b: FloatArray,
        out: FloatArray,
        len: Int
    ): Int

    @JvmStatic
    private external fun nativeVectorMul(
        a: FloatArray,
        b: FloatArray,
        out: FloatArray,
        len: Int
    ): Int

    @JvmStatic
    private external fun nativeRelu(
        x: FloatArray,
        out: FloatArray,
        len: Int
    ): Int

    @JvmStatic
    private external fun nativeSigmoid(
        x: FloatArray,
        out: FloatArray,
        len: Int
    ): Int

    @JvmStatic
    private external fun nativeVectorDotBatteryOptimized(
        a: FloatArray,
        b: FloatArray,
        len: Int,
        result: FloatArray
    ): Int
}

/**
 * Matrix class for 2D arrays
 */
@Keep
class SciMatrix(val rows: Int, val cols: Int) {
    private var data: FloatArray = FloatArray(rows * cols)

    /**
     * Get element at [row, col]
     */
    operator fun get(row: Int, col: Int): Float {
        require(row in 0 until rows && col in 0 until cols) { "Index out of bounds" }
        return data[row * cols + col]
    }

    /**
     * Set element at [row, col]
     */
    operator fun set(row: Int, col: Int, value: Float) {
        require(row in 0 until rows && col in 0 until cols) { "Index out of bounds" }
        data[row * cols + col] = value
    }

    /**
     * Matrix-vector multiplication
     */
    fun multiply(vector: FloatArray): FloatArray {
        require(vector.size == cols) { "Vector size must match number of columns" }
        val result = FloatArray(rows)
        // TODO: Call native JNI function
        return result
    }

    companion object {
        /**
         * Create matrix from 2D array
         */
        fun fromArray(array: Array<FloatArray>): SciMatrix {
            val rows = array.size
            val cols = array[0].size
            val matrix = SciMatrix(rows, cols)
            for (i in 0 until rows) {
                for (j in 0 until cols) {
                    matrix[i, j] = array[i][j]
                }
            }
            return matrix
        }
    }
}

// Example usage:
/*
fun main() {
    SciRS2.initialize()

    println("SciRS2 version: ${SciRS2.version}")
    println("NEON available: ${SciRS2.hasNEON}")

    // Set battery mode based on device state
    SciRS2.setBatteryMode(BatteryMode.BALANCED)

    // Vector operations
    val a = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f)
    val b = floatArrayOf(1.0f, 1.0f, 1.0f, 1.0f)

    val dot = SciRS2.dot(a, b)
    println("Dot product: $dot") // 10.0

    val sum = SciRS2.add(a, b)
    println("Sum: ${sum.contentToString()}") // [2.0, 3.0, 4.0, 5.0]

    // Neural network operations
    val activations = SciRS2.relu(floatArrayOf(-1.0f, 0.0f, 1.0f, 2.0f))
    println("ReLU: ${activations.contentToString()}") // [0.0, 0.0, 1.0, 2.0]

    // Monitor thermal state and adjust
    SciRS2.setThermalState(ThermalState.HOT)
    val batteryDot = SciRS2.dotBatteryOptimized(a, b)
    println("Battery-optimized dot: $batteryDot")

    SciRS2.shutdown()
}
*/
