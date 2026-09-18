# SciRS2 Mobile Platform Quick Start Guide

## Quick Setup

### Prerequisites

**For iOS:**
- macOS with Xcode 14+
- Rust toolchain
- CocoaPods or Swift Package Manager

**For Android:**
- Android Studio
- Android NDK r25+
- Rust toolchain

### Installation

```bash
# Clone repository
git clone https://github.com/cool-japan/scirs.git
cd scirs

# Setup mobile toolchains
./scripts/setup_mobile_toolchains.sh
```

## iOS Integration (5 minutes)

### Step 1: Build Framework

```bash
./scripts/build_ios.sh --crate scirs2-core --features "mobile,ios"
```

This creates: `target/ios/SciRS2.xcframework`

### Step 2: Add to Xcode Project

**Option A: CocoaPods**

```ruby
# Podfile
pod 'SciRS2', :path => 'path/to/scirs'
```

```bash
pod install
```

**Option B: Manual**

1. Drag `SciRS2.xcframework` into your Xcode project
2. Add to "Frameworks, Libraries, and Embedded Content"
3. Copy `mobile/ios/SciRS2.swift` to your project

### Step 3: Use in Swift

```swift
import SciRS2

// Initialize
try SciRS2.initialize()

// Perform operations
let a: [Float] = [1.0, 2.0, 3.0, 4.0]
let b: [Float] = [1.0, 1.0, 1.0, 1.0]
let result = SciRS2.dot(a, b) // 10.0

// Cleanup
SciRS2.shutdown()
```

## Android Integration (5 minutes)

### Step 1: Build Libraries

```bash
./scripts/build_android.sh --crate scirs2-core --features "mobile,android"
```

This creates: `target/android/jniLibs/`

### Step 2: Add to Android Project

1. Copy `target/android/jniLibs` to `app/src/main/jniLibs`
2. Copy `mobile/android/SciRS2.kt` to your project

### Step 3: Configure Gradle

```gradle
// app/build.gradle
android {
    defaultConfig {
        ndk {
            abiFilters 'arm64-v8a', 'armeabi-v7a'
        }
    }
}
```

### Step 4: Use in Kotlin

```kotlin
import com.cooljapan.scirs2.SciRS2

// Initialize
SciRS2.initialize()

// Perform operations
val a = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f)
val b = floatArrayOf(1.0f, 1.0f, 1.0f, 1.0f)
val result = SciRS2.dot(a, b) // 10.0

// Cleanup
SciRS2.shutdown()
```

## Common Use Cases

### 1. Neural Network Inference

**iOS:**
```swift
// Load model weights
let weights = loadWeights()
let input = preprocessImage(image)

// Set battery mode for inference
SciRS2.setBatteryMode(.balanced)

// Forward pass
let hidden = SciRS2.dot(input, weights.layer1)
let activated = SciRS2.relu(hidden)
let output = SciRS2.sigmoid(SciRS2.dot(activated, weights.layer2))
```

**Android:**
```kotlin
// Load model weights
val weights = loadWeights()
val input = preprocessImage(image)

// Monitor battery and adjust
SciRS2.setBatteryMode(BatteryMode.BALANCED)

// Forward pass
val hidden = SciRS2.dot(input, weights.layer1)
val activated = SciRS2.relu(hidden)
val output = SciRS2.sigmoid(SciRS2.dot(activated, weights.layer2))
```

### 2. Signal Processing

**iOS:**
```swift
// Process audio samples
let samples = audioBuffer.samples
let filtered = applyFilter(samples)
let spectrum = SciRS2.fft(filtered) // Note: FFT not yet in mobile API
```

**Android:**
```kotlin
// Process sensor data
val accelerometer = sensorData.accelerometer
val filtered = applyFilter(accelerometer)
```

### 3. Battery-Aware Computing

**iOS:**
```swift
import UIKit

class ComputeManager {
    func performComputation() {
        // Monitor battery state
        UIDevice.current.isBatteryMonitoringEnabled = true
        let batteryLevel = UIDevice.current.batteryLevel
        let batteryState = UIDevice.current.batteryState

        // Adjust mode based on battery
        if batteryState == .unplugged && batteryLevel < 0.2 {
            SciRS2.setBatteryMode(.powerSaver)
        } else if batteryState == .charging {
            SciRS2.setBatteryMode(.performance)
        } else {
            SciRS2.setBatteryMode(.balanced)
        }

        // Perform computation
        let result = SciRS2.dotBatteryOptimized(data.a, data.b)
    }
}
```

**Android:**
```kotlin
import android.os.PowerManager

class ComputeManager(context: Context) {
    private val powerManager = context.getSystemService(Context.POWER_SERVICE) as PowerManager

    fun performComputation() {
        // Adjust based on power save mode
        val mode = when {
            powerManager.isPowerSaveMode -> BatteryMode.POWER_SAVER
            powerManager.isInteractive -> BatteryMode.BALANCED
            else -> BatteryMode.PERFORMANCE
        }
        SciRS2.setBatteryMode(mode)

        // Perform computation
        val result = SciRS2.dotBatteryOptimized(data.a, data.b)
    }
}
```

### 4. Thermal Management

**iOS:**
```swift
import Foundation

class ThermalMonitor {
    init() {
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(thermalStateChanged),
            name: ProcessInfo.thermalStateDidChangeNotification,
            object: nil
        )
    }

    @objc func thermalStateChanged() {
        let state = ProcessInfo.processInfo.thermalState
        switch state {
        case .nominal:
            SciRS2.setThermalState(.normal)
        case .fair:
            SciRS2.setThermalState(.warm)
        case .serious:
            SciRS2.setThermalState(.hot)
        case .critical:
            SciRS2.setThermalState(.critical)
        @unknown default:
            break
        }
    }
}
```

**Android:**
```kotlin
import android.os.PowerManager

class ThermalMonitor(context: Context) {
    private val powerManager = context.getSystemService(Context.POWER_SERVICE) as PowerManager

    fun monitorThermalState() {
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.Q) {
            val thermalStatus = powerManager.currentThermalStatus
            val state = when (thermalStatus) {
                PowerManager.THERMAL_STATUS_NONE -> ThermalState.NORMAL
                PowerManager.THERMAL_STATUS_LIGHT -> ThermalState.WARM
                PowerManager.THERMAL_STATUS_MODERATE -> ThermalState.HOT
                else -> ThermalState.CRITICAL
            }
            SciRS2.setThermalState(state)
        }
    }
}
```

## Performance Tips

### 1. Choose the Right Battery Mode

- **Performance**: Real-time applications, charging
- **Balanced**: Default for most apps
- **PowerSaver**: Background processing, low battery

### 2. Monitor Thermal State

Always monitor thermal state to prevent device overheating:

```swift
// iOS
ProcessInfo.processInfo.thermalState

// Android (API 29+)
powerManager.currentThermalStatus
```

### 3. Batch Operations

Process data in batches rather than one-by-one:

```swift
// Good
let results = SciRS2.dot(allInputs, weights)

// Bad (slower)
for input in inputs {
    let result = SciRS2.dot(input, weights)
}
```

### 4. Reuse Buffers

Allocate buffers once and reuse them:

```swift
let buffer = SciVector(capacity: 1000)
// Reuse buffer for multiple operations
```

## Troubleshooting

### iOS: Library Not Found

```
ld: library not found for -lscirs2_core
```

**Solution:**
- Verify XCFramework is in project
- Check "Embed & Sign" setting
- Clean build folder (Cmd+Shift+K)

### Android: UnsatisfiedLinkError

```
java.lang.UnsatisfiedLinkError: dlopen failed: library "libscirs2_core.so" not found
```

**Solution:**
- Verify jniLibs directory contains .so files
- Check build.gradle ABI filters
- Clean and rebuild project

### NEON Not Available

```
// Check if NEON is available
if !SciRS2.hasNEON {
    print("WARNING: NEON not available, using scalar fallback")
}
```

## Next Steps

- Read [MOBILE_PLATFORM_SUPPORT.md](MOBILE_PLATFORM_SUPPORT.md) for detailed documentation
- Check [examples/](../examples/) for complete sample apps
- Join our [Discord](https://discord.gg/scirs2) for support

## Support

- GitHub Issues: https://github.com/cool-japan/scirs/issues
- Documentation: https://docs.rs/scirs2
- Discord: https://discord.gg/scirs2

---

**Happy Computing! 🚀**
