//! Mobile-specific optimizations for ARM NEON
//!
//! This module provides battery-aware and thermal-aware implementations
//! specifically designed for mobile platforms (iOS/Android).
//!
//! ## Features
//!
//! - Battery-optimized algorithms that reduce power consumption
//! - Thermal-aware processing that prevents device overheating
//! - Memory-efficient chunking for constrained devices
//! - Background task support with reduced CPU usage

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Battery optimization mode for mobile devices
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatteryMode {
    /// Maximum performance, highest power consumption
    Performance,
    /// Balanced performance and power consumption
    Balanced,
    /// Maximum battery life, reduced performance
    PowerSaver,
}

/// Thermal state of the device
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ThermalState {
    /// Normal operating temperature
    Normal,
    /// Slightly elevated temperature
    Warm,
    /// High temperature, should reduce workload
    Hot,
    /// Critical temperature, must throttle immediately
    Critical,
}

/// Mobile-specific optimizer configuration
#[derive(Debug, Clone)]
pub struct MobileOptimizer {
    /// Current battery mode
    pub battery_mode: BatteryMode,
    /// Current thermal state
    pub thermal_state: ThermalState,
    /// Maximum chunk size for processing (bytes)
    pub max_chunk_size: usize,
    /// Enable background processing optimizations
    pub background_mode: bool,
}

impl Default for MobileOptimizer {
    fn default() -> Self {
        Self {
            battery_mode: BatteryMode::Balanced,
            thermal_state: ThermalState::Normal,
            max_chunk_size: 1024 * 1024, // 1MB default
            background_mode: false,
        }
    }
}

impl MobileOptimizer {
    /// Create a new mobile optimizer with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set battery mode
    pub fn with_battery_mode(mut self, mode: BatteryMode) -> Self {
        self.battery_mode = mode;
        self
    }

    /// Set thermal state
    pub fn with_thermal_state(mut self, state: ThermalState) -> Self {
        self.thermal_state = state;
        self
    }

    /// Enable background processing mode
    pub fn with_background_mode(mut self, enabled: bool) -> Self {
        self.background_mode = enabled;
        self
    }

    /// Get optimal chunk size based on current settings
    pub fn get_optimal_chunk_size(&self) -> usize {
        let base_size = match self.battery_mode {
            BatteryMode::Performance => 4096 * 1024, // 4MB
            BatteryMode::Balanced => 1024 * 1024,    // 1MB
            BatteryMode::PowerSaver => 256 * 1024,   // 256KB
        };

        // Reduce chunk size if thermal state is elevated
        let thermal_multiplier = match self.thermal_state {
            ThermalState::Normal => 1.0,
            ThermalState::Warm => 0.75,
            ThermalState::Hot => 0.5,
            ThermalState::Critical => 0.25,
        };

        // Further reduce if in background mode
        let background_multiplier = if self.background_mode { 0.5 } else { 1.0 };

        (base_size as f32 * thermal_multiplier * background_multiplier) as usize
    }

    /// Get delay between chunks (microseconds) for thermal management
    pub fn get_chunk_delay_us(&self) -> u64 {
        match (self.thermal_state, self.battery_mode) {
            (ThermalState::Critical, _) => 10000, // 10ms delay
            (ThermalState::Hot, _) => 5000,       // 5ms delay
            (ThermalState::Warm, BatteryMode::PowerSaver) => 2000, // 2ms delay
            (ThermalState::Warm, _) => 1000,      // 1ms delay
            _ => 0,                               // No delay
        }
    }

    /// Check if processing should be paused due to thermal conditions
    pub fn should_throttle(&self) -> bool {
        self.thermal_state >= ThermalState::Hot
    }
}

/// Battery-optimized dot product for f32
///
/// Adapts processing based on battery mode:
/// - Performance: No optimization, maximum speed
/// - Balanced: Chunked processing with moderate delays
/// - PowerSaver: Small chunks with sleep intervals
pub fn neon_dot_battery_optimized(a: &[f32], b: &[f32], optimizer: &MobileOptimizer) -> f32 {
    let len = a.len().min(b.len());
    let chunk_size = optimizer.get_optimal_chunk_size() / std::mem::size_of::<f32>();
    let delay_us = optimizer.get_chunk_delay_us();

    let mut result = 0.0;
    let mut offset = 0;

    while offset < len {
        let chunk_end = (offset + chunk_size).min(len);
        let chunk_a = &a[offset..chunk_end];
        let chunk_b = &b[offset..chunk_end];

        // Process chunk
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                result += unsafe { neon_dot_chunk_f32(chunk_a, chunk_b) };
            } else {
                result += scalar_dot_f32(chunk_a, chunk_b);
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            result += scalar_dot_f32(chunk_a, chunk_b);
        }

        offset = chunk_end;

        // Thermal management: insert delay between chunks if needed
        if delay_us > 0 && offset < len {
            std::thread::sleep(std::time::Duration::from_micros(delay_us));
        }
    }

    result
}

/// NEON-optimized chunk processing for dot product
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_dot_chunk_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut i = 0;
    let mut sum = vdupq_n_f32(0.0);

    while i + 4 <= len {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        sum = vfmaq_f32(sum, va, vb);
        i += 4;
    }

    let mut result = vaddvq_f32(sum);

    while i < len {
        result += a[i] * b[i];
        i += 1;
    }

    result
}

/// Scalar fallback for dot product
fn scalar_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut sum = 0.0;
    for i in 0..len {
        sum += a[i] * b[i];
    }
    sum
}

/// Battery-optimized GEMM for mobile devices
pub fn neon_gemm_battery_optimized(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
    optimizer: &MobileOptimizer,
) {
    // Adjust block sizes based on battery mode
    let (mc, nc, kc) = match optimizer.battery_mode {
        BatteryMode::Performance => (128, 512, 256),
        BatteryMode::Balanced => (64, 256, 128),
        BatteryMode::PowerSaver => (32, 128, 64),
    };

    let delay_us = optimizer.get_chunk_delay_us();

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { neon_gemm_chunked(m, n, k, alpha, a, b, beta, c, mc, nc, kc, delay_us) }
        } else {
            fallback_gemm_chunked(m, n, k, alpha, a, b, beta, c, mc, nc, kc, delay_us)
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        fallback_gemm_chunked(m, n, k, alpha, a, b, beta, c, mc, nc, kc, delay_us)
    }
}

/// NEON-optimized chunked GEMM with thermal management
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_gemm_chunked(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
    mc: usize,
    nc: usize,
    kc: usize,
    delay_us: u64,
) {
    for jc in (0..n).step_by(nc) {
        let nc_actual = (jc + nc).min(n) - jc;

        for pc in (0..k).step_by(kc) {
            let kc_actual = (pc + kc).min(k) - pc;

            for ic in (0..m).step_by(mc) {
                let mc_actual = (ic + mc).min(m) - ic;

                // Process micro-kernel
                for i in 0..mc_actual {
                    for j in 0..nc_actual {
                        let mut sum = vdupq_n_f32(0.0);
                        let mut p = 0;

                        while p + 4 <= kc_actual {
                            let va = vld1q_f32(a.as_ptr().add((ic + i) * k + pc + p));
                            let vb = vld1q_f32(b.as_ptr().add((pc + p) * n + jc + j));
                            sum = vfmaq_f32(sum, va, vb);
                            p += 4;
                        }

                        let mut dot = vaddvq_f32(sum);

                        while p < kc_actual {
                            dot += a[(ic + i) * k + pc + p] * b[(pc + p) * n + jc + j];
                            p += 1;
                        }

                        let c_idx = (ic + i) * n + jc + j;
                        if pc == 0 {
                            c[c_idx] = alpha * dot + beta * c[c_idx];
                        } else {
                            c[c_idx] += alpha * dot;
                        }
                    }
                }

                // Insert delay for thermal management if needed
                if delay_us > 0 {
                    std::thread::sleep(std::time::Duration::from_micros(delay_us));
                }
            }
        }
    }
}

/// Thermal-aware GEMM that automatically throttles based on temperature
pub fn neon_gemm_thermal_aware(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
    thermal_state: ThermalState,
) {
    let optimizer = MobileOptimizer::default().with_thermal_state(thermal_state);
    neon_gemm_battery_optimized(m, n, k, alpha, a, b, beta, c, &optimizer);
}

// Fallback implementation
fn fallback_gemm_chunked(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
    mc: usize,
    nc: usize,
    kc: usize,
    delay_us: u64,
) {
    for jc in (0..n).step_by(nc) {
        let nc_actual = (jc + nc).min(n) - jc;

        for pc in (0..k).step_by(kc) {
            let kc_actual = (pc + kc).min(k) - pc;

            for ic in (0..m).step_by(mc) {
                let mc_actual = (ic + mc).min(m) - ic;

                for i in 0..mc_actual {
                    for j in 0..nc_actual {
                        let mut sum = 0.0;

                        for p in 0..kc_actual {
                            sum += a[(ic + i) * k + pc + p] * b[(pc + p) * n + jc + j];
                        }

                        let c_idx = (ic + i) * n + jc + j;
                        if pc == 0 {
                            c[c_idx] = alpha * sum + beta * c[c_idx];
                        } else {
                            c[c_idx] += alpha * sum;
                        }
                    }
                }

                if delay_us > 0 {
                    std::thread::sleep(std::time::Duration::from_micros(delay_us));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mobile_optimizer_defaults() {
        let opt = MobileOptimizer::new();
        assert_eq!(opt.battery_mode, BatteryMode::Balanced);
        assert_eq!(opt.thermal_state, ThermalState::Normal);
        assert!(!opt.background_mode);
    }

    #[test]
    fn test_battery_mode_chunk_sizes() {
        let perf = MobileOptimizer::new().with_battery_mode(BatteryMode::Performance);
        let balanced = MobileOptimizer::new().with_battery_mode(BatteryMode::Balanced);
        let saver = MobileOptimizer::new().with_battery_mode(BatteryMode::PowerSaver);

        assert!(perf.get_optimal_chunk_size() > balanced.get_optimal_chunk_size());
        assert!(balanced.get_optimal_chunk_size() > saver.get_optimal_chunk_size());
    }

    #[test]
    fn test_thermal_throttling() {
        let normal = MobileOptimizer::new().with_thermal_state(ThermalState::Normal);
        let hot = MobileOptimizer::new().with_thermal_state(ThermalState::Hot);
        let critical = MobileOptimizer::new().with_thermal_state(ThermalState::Critical);

        assert!(!normal.should_throttle());
        assert!(hot.should_throttle());
        assert!(critical.should_throttle());

        assert!(critical.get_chunk_delay_us() > hot.get_chunk_delay_us());
        assert!(hot.get_chunk_delay_us() > normal.get_chunk_delay_us());
    }

    #[test]
    fn test_battery_optimized_dot() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];
        let opt = MobileOptimizer::new();

        let result = neon_dot_battery_optimized(&a, &b, &opt);

        assert!((result - 10.0).abs() < 1e-6);
    }
}
