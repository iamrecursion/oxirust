//! Embedded Example: ARM Cortex-M Fixed-Point Signal Processing
//!
//! This example demonstrates SciRS2's embedded systems support for ARM Cortex-M
//! microcontrollers without standard library or floating-point unit.
//!
//! # Target Platform
//! - ARM Cortex-M4F (STM32F4, nRF52, etc.)
//! - No std, no heap allocation
//! - Real-time signal processing
//!
//! # Build Instructions
//! ```bash
//! # Install ARM toolchain
//! rustup target add thumbv7em-none-eabihf
//!
//! # Build for ARM Cortex-M4F
//! cargo build --target thumbv7em-none-eabihf \
//!   --no-default-features \
//!   --features embedded,fixed-point \
//!   --example cortex_m_example
//! ```
//!
//! # Memory Usage
//! - Stack: ~4KB
//! - Flash: ~16KB
//! - No heap allocation

#![no_std]
#![no_main]

// Panic handler for embedded systems
use core::panic::PanicInfo;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

// Entry point (would be provided by cortex-m-rt in real hardware)
#[no_mangle]
pub extern "C" fn main() -> ! {
    // Example: Real-time FIR filter using fixed-point arithmetic
    example_fir_filter();

    // Example: Stack-based array operations
    example_stack_arrays();

    // Example: Fixed-point trigonometry
    example_fixed_point_math();

    // Embedded systems typically run in an infinite loop
    loop {
        // In real hardware, this would be:
        // - Read sensor data
        // - Process with fixed-point math
        // - Output results
        // - Sleep or wait for interrupt
    }
}

/// Example 1: Fixed-point FIR filter for real-time signal processing
fn example_fir_filter() {
    use scirs2_core::fixed_point::{Fixed32, signal::FirFilter};

    // Low-pass filter coefficients (Q15.16 format)
    // Normalized cutoff at 0.2 * Nyquist
    let coeffs = [
        Fixed32::<16>::from_float(0.054),
        Fixed32::<16>::from_float(0.244),
        Fixed32::<16>::from_float(0.402),
        Fixed32::<16>::from_float(0.244),
        Fixed32::<16>::from_float(0.054),
    ];

    let mut filter = FirFilter::new(&coeffs);

    // Simulate ADC input samples
    let samples = [
        Fixed32::<16>::from_float(0.5),
        Fixed32::<16>::from_float(0.8),
        Fixed32::<16>::from_float(1.0),
        Fixed32::<16>::from_float(0.8),
        Fixed32::<16>::from_float(0.5),
    ];

    // Process samples (real-time, deterministic)
    for sample in &samples {
        let filtered = filter.process(*sample);
        // In real hardware: send to DAC or UART
        let _ = filtered.to_float(); // Convert for output
    }
}

/// Example 2: Stack-based array operations (no heap)
fn example_stack_arrays() {
    use scirs2_core::embedded::{StackArray, stack_add, stack_mean};

    // Fixed-size arrays allocated on stack
    let mut sensor_data = StackArray::<f32, 16>::new();
    let mut baseline = StackArray::<f32, 16>::new();

    // Initialize with sensor readings
    for i in 0..16 {
        sensor_data[i] = i as f32 * 0.1;
        baseline[i] = 0.5;
    }

    // Compute difference (calibration)
    let calibrated = stack_add(&sensor_data, &baseline);

    // Calculate mean (statistics)
    let average = stack_mean(&calibrated);

    // Use result (e.g., threshold detection)
    let threshold = 5.0;
    let _ = average > threshold;
}

/// Example 3: Fixed-point trigonometry for sensor fusion
fn example_fixed_point_math() {
    use scirs2_core::fixed_point::{Fixed32, math};

    // IMU sensor fusion: calculate orientation
    // Angles in radians (Q15.16 format)
    let pitch = Fixed32::<16>::from_float(0.1); // ~5.7 degrees
    let roll = Fixed32::<16>::from_float(0.2);  // ~11.5 degrees

    // Compute trigonometric functions
    let sin_pitch = math::sin(pitch);
    let cos_pitch = math::cos(pitch);
    let sin_roll = math::sin(roll);
    let cos_roll = math::cos(roll);

    // Rotation matrix elements (simplified)
    let r11 = cos_pitch * cos_roll;
    let r12 = sin_pitch;

    // Use for coordinate transformation
    let _ = r11.to_float();
    let _ = r12.to_float();
}

/// Example 4: Circular buffer for streaming data (real-time)
fn example_circular_buffer() {
    use scirs2_core::embedded::FixedSizeBuffer;

    let mut buffer = FixedSizeBuffer::<i16, 32>::new();

    // Simulate ADC interrupt handler
    for _ in 0..100 {
        // Receive sample from ADC
        let adc_value: i16 = 2048; // 12-bit ADC reading

        // Push to buffer (overwrite oldest if full)
        if buffer.is_full() {
            let _ = buffer.pop(); // Remove oldest
        }
        let _ = buffer.push(adc_value);

        // Process when buffer is full
        if buffer.is_full() {
            // Run FFT or other processing
            process_buffer(&buffer);
            buffer.clear();
        }
    }
}

fn process_buffer<const N: usize>(_buffer: &scirs2_core::embedded::FixedSizeBuffer<i16, N>) {
    // Would perform FFT, filtering, etc.
}

/// Memory requirements estimation
fn estimate_memory_usage() {
    use scirs2_core::embedded::memory_estimation::{MemoryRequirement, estimate_stack_usage};

    // Estimate for FIR filter
    let fir_stack = estimate_stack_usage::<f32>(64); // 64 taps
    assert!(fir_stack == 256); // 64 * 4 bytes

    // Estimate for signal processing
    let req = MemoryRequirement::signal_processing();
    assert!(req.stack_bytes >= 4096);
    assert!(req.heap_bytes == 0); // No heap in embedded mode
}

// Configuration for different Cortex-M variants
#[cfg(target_arch = "arm")]
mod platform {
    // Cortex-M0/M0+: No FPU, limited RAM
    #[cfg(target_feature = "thumb-mode")]
    pub const MAX_FILTER_SIZE: usize = 32;

    // Cortex-M4F: Has FPU, more RAM
    #[cfg(target_feature = "fp")]
    pub const MAX_FILTER_SIZE: usize = 128;

    // Cortex-M7: High performance, lots of RAM
    #[cfg(all(target_feature = "fp", target_feature = "d32"))]
    pub const MAX_FILTER_SIZE: usize = 256;
}

// Default for non-ARM targets
#[cfg(not(target_arch = "arm"))]
mod platform {
    pub const MAX_FILTER_SIZE: usize = 64;
}
