//! GPU profiling functionality.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// GPU profiling statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuStats {
    /// GPU utilization percentage (0.0-100.0).
    pub utilization: f64,

    /// Memory used in bytes.
    pub memory_used: u64,

    /// Total memory in bytes.
    pub total_memory: u64,

    /// Number of draw calls.
    pub draw_calls: u64,

    /// Number of compute dispatches.
    pub compute_dispatches: u64,

    /// Average frame time.
    pub avg_frame_time: Duration,

    /// GPU temperature in Celsius.
    pub temperature: Option<f64>,

    /// GPU clock speed in MHz.
    pub clock_speed: Option<u32>,
}

/// GPU operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GpuOperation {
    /// Draw call.
    Draw,

    /// Compute dispatch.
    Compute,

    /// Memory transfer.
    Transfer,

    /// Synchronization.
    Sync,
}

/// GPU profiler.
#[derive(Debug)]
pub struct GpuProfiler {
    running: bool,
    start_time: Option<Instant>,
    operations: HashMap<GpuOperation, u64>,
    operation_times: HashMap<GpuOperation, Duration>,
    frame_times: Vec<Duration>,
}

impl GpuProfiler {
    /// Create a new GPU profiler.
    pub fn new() -> Self {
        Self {
            running: false,
            start_time: None,
            operations: HashMap::new(),
            operation_times: HashMap::new(),
            frame_times: Vec::new(),
        }
    }

    /// Start profiling.
    pub fn start(&mut self) {
        self.running = true;
        self.start_time = Some(Instant::now());
        self.operations.clear();
        self.operation_times.clear();
        self.frame_times.clear();
    }

    /// Stop profiling.
    pub fn stop(&mut self) {
        self.running = false;
    }

    /// Record a GPU operation.
    pub fn record_operation(&mut self, op: GpuOperation, duration: Duration) {
        if !self.running {
            return;
        }

        *self.operations.entry(op).or_insert(0) += 1;
        *self.operation_times.entry(op).or_insert(Duration::ZERO) += duration;
    }

    /// Record frame time.
    pub fn record_frame_time(&mut self, duration: Duration) {
        if !self.running {
            return;
        }

        self.frame_times.push(duration);
    }

    /// Get GPU statistics.
    pub fn stats(&self) -> GpuStats {
        let draw_calls = self
            .operations
            .get(&GpuOperation::Draw)
            .copied()
            .unwrap_or(0);
        let compute_dispatches = self
            .operations
            .get(&GpuOperation::Compute)
            .copied()
            .unwrap_or(0);

        let avg_frame_time = if !self.frame_times.is_empty() {
            let total: Duration = self.frame_times.iter().sum();
            total / self.frame_times.len() as u32
        } else {
            Duration::ZERO
        };

        // Collect real GPU metrics from the system.
        let (utilization, memory_used, total_memory, temperature, clock_speed) =
            Self::read_system_gpu_stats();

        GpuStats {
            utilization,
            memory_used,
            total_memory,
            draw_calls,
            compute_dispatches,
            avg_frame_time,
            temperature,
            clock_speed,
        }
    }

    /// Check if profiler is running.
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// Get elapsed time.
    pub fn elapsed(&self) -> Option<Duration> {
        self.start_time.map(|start| start.elapsed())
    }

    /// Generate a summary report.
    pub fn summary(&self) -> String {
        let stats = self.stats();
        let mut report = String::new();

        report.push_str(&format!("  Draw Calls: {}\n", stats.draw_calls));
        report.push_str(&format!(
            "  Compute Dispatches: {}\n",
            stats.compute_dispatches
        ));
        report.push_str(&format!("  Avg Frame Time: {:?}\n", stats.avg_frame_time));

        if let Some(temp) = stats.temperature {
            report.push_str(&format!("  Temperature: {:.1}°C\n", temp));
        }

        if let Some(clock) = stats.clock_speed {
            report.push_str(&format!("  Clock Speed: {} MHz\n", clock));
        }

        report
    }

    /// Read live GPU statistics from the system.
    ///
    /// Tries NVIDIA sysfs first, then falls back to DRM/Intel sysfs paths.
    /// Returns `(utilization %, memory_used bytes, total_memory bytes, temperature °C, clock MHz)`.
    #[allow(clippy::type_complexity)]
    fn read_system_gpu_stats() -> (f64, u64, u64, Option<f64>, Option<u32>) {
        // --- Try NVIDIA ---
        if let Some(stats) = Self::read_nvidia_stats() {
            return stats;
        }

        // --- Try DRM sysfs (Intel / AMD) ---
        if let Some(stats) = Self::read_drm_stats() {
            return stats;
        }

        (0.0, 0, 0, None, None)
    }

    /// Attempt to read NVIDIA GPU stats from `/proc/driver/nvidia/gpus/`.
    #[allow(clippy::type_complexity)]
    fn read_nvidia_stats() -> Option<(f64, u64, u64, Option<f64>, Option<u32>)> {
        use std::fs;

        let gpus_dir = fs::read_dir("/proc/driver/nvidia/gpus/").ok()?;
        for entry in gpus_dir.flatten() {
            let info_path = entry.path().join("information");
            let content = match fs::read_to_string(&info_path) {
                Ok(c) => c,
                Err(_) => continue,
            };

            let mut total_memory: u64 = 0;
            let mut clock_speed: Option<u32> = None;

            for line in content.lines() {
                let line = line.trim();
                if line.starts_with("Video Memory:") {
                    // e.g. "Video Memory:  8192 MB"
                    if let Some(val) = line.split_whitespace().nth(2) {
                        if let Ok(mb) = val.parse::<u64>() {
                            total_memory = mb * 1024 * 1024;
                        }
                    }
                }
                if line.starts_with("GPU Clock:") {
                    // e.g. "GPU Clock:    1500 MHz"
                    if let Some(val) = line.split_whitespace().nth(2) {
                        clock_speed = val.parse::<u32>().ok();
                    }
                }
            }

            // Read utilization from /proc/driver/nvidia/gpus/<id>/utilization if present.
            let util_path = entry.path().join("utilization");
            let utilization = fs::read_to_string(util_path)
                .ok()
                .and_then(|s| {
                    s.split_whitespace()
                        .find_map(|w| w.trim_end_matches('%').parse::<f64>().ok())
                })
                .unwrap_or(0.0);

            // Estimate used memory as half of total (we lack a reliable per-process view here).
            let memory_used = total_memory / 2;

            return Some((utilization, memory_used, total_memory, None, clock_speed));
        }
        None
    }

    /// Attempt to read GPU stats from DRM sysfs (`/sys/class/drm/card*`).
    #[allow(clippy::type_complexity)]
    fn read_drm_stats() -> Option<(f64, u64, u64, Option<f64>, Option<u32>)> {
        use std::fs;

        // Find the first DRM card directory.
        let drm_dir = fs::read_dir("/sys/class/drm/").ok()?;
        let card_path = drm_dir
            .flatten()
            .find(|e| {
                let name = e.file_name();
                let s = name.to_string_lossy();
                // Match "card0", "card1", … but not "card0-DP-1" etc.
                s.starts_with("card") && s.chars().skip(4).all(|c| c.is_ascii_digit())
            })
            .map(|e| e.path())?;

        // Read current clock speed (MHz).
        let clock_speed: Option<u32> = fs::read_to_string(card_path.join("gt_cur_freq_mhz"))
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok());

        // Estimate GPU utilization: compare current vs boost frequency.
        let cur_freq = clock_speed.unwrap_or(0) as f64;
        let boost_freq = fs::read_to_string(card_path.join("gt_boost_freq_mhz"))
            .ok()
            .and_then(|s| s.trim().parse::<f64>().ok())
            .unwrap_or(1.0);
        let utilization = if boost_freq > 0.0 {
            (cur_freq / boost_freq * 100.0).min(100.0)
        } else {
            0.0
        };

        // Read GPU memory from /proc/meminfo MemAvailable as a rough total estimate
        // (no GPU-specific API without driver headers).
        let (memory_used, total_memory) = Self::read_gpu_memory_from_proc();

        Some((utilization, memory_used, total_memory, None, clock_speed))
    }

    /// Parse `/proc/meminfo` and use MemTotal/MemAvailable as a conservative GPU memory proxy.
    /// On integrated GPUs the GPU shares system RAM, so this is a meaningful approximation.
    fn read_gpu_memory_from_proc() -> (u64, u64) {
        use std::fs;

        let content = match fs::read_to_string("/proc/meminfo") {
            Ok(c) => c,
            Err(_) => return (0, 0),
        };

        let mut total_kb: u64 = 0;
        let mut available_kb: u64 = 0;

        for line in content.lines() {
            if line.starts_with("MemTotal:") {
                total_kb = line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0);
            } else if line.starts_with("MemAvailable:") {
                available_kb = line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0);
            }
            if total_kb > 0 && available_kb > 0 {
                break;
            }
        }

        let total = total_kb * 1024;
        let used = total.saturating_sub(available_kb * 1024);
        (used, total)
    }
}
impl Default for GpuProfiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_profiler_creation() {
        let profiler = GpuProfiler::new();
        assert!(!profiler.is_running());
    }

    #[test]
    fn test_gpu_profiler_start_stop() {
        let mut profiler = GpuProfiler::new();
        profiler.start();
        assert!(profiler.is_running());
        profiler.stop();
        assert!(!profiler.is_running());
    }

    #[test]
    fn test_record_operation() {
        let mut profiler = GpuProfiler::new();
        profiler.start();
        profiler.record_operation(GpuOperation::Draw, Duration::from_millis(1));
        profiler.record_operation(GpuOperation::Draw, Duration::from_millis(1));
        profiler.record_operation(GpuOperation::Compute, Duration::from_millis(2));
        profiler.stop();

        let stats = profiler.stats();
        assert_eq!(stats.draw_calls, 2);
        assert_eq!(stats.compute_dispatches, 1);
    }

    #[test]
    fn test_frame_time() {
        let mut profiler = GpuProfiler::new();
        profiler.start();
        profiler.record_frame_time(Duration::from_millis(16));
        profiler.record_frame_time(Duration::from_millis(17));
        profiler.stop();

        let stats = profiler.stats();
        assert!(stats.avg_frame_time > Duration::ZERO);
    }

    #[test]
    fn test_gpu_stats() {
        let stats = GpuStats {
            utilization: 75.5,
            memory_used: 1_000_000_000,
            total_memory: 4_000_000_000,
            draw_calls: 1000,
            compute_dispatches: 50,
            avg_frame_time: Duration::from_millis(16),
            temperature: Some(65.0),
            clock_speed: Some(1500),
        };

        assert_eq!(stats.utilization, 75.5);
        assert_eq!(stats.draw_calls, 1000);
        assert_eq!(stats.temperature, Some(65.0));
    }
}
