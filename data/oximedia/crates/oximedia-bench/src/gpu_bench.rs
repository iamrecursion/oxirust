//! GPU benchmarking (simulation) tools.
//!
//! This module provides deterministic simulation of GPU-accelerated media
//! processing tasks, enabling reproducible performance comparisons between
//! hypothetical GPU configurations without requiring real hardware.

/// The type of GPU-accelerated task being benchmarked.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[allow(dead_code)]
pub enum GpuBenchTask {
    /// Hardware video decoding.
    VideoDecoding,
    /// Hardware video encoding.
    VideoEncoding,
    /// Color space and pixel format conversion.
    ColorConversion,
    /// AI/GPU-accelerated video denoising.
    Denoising,
    /// Scene-change detection.
    SceneDetection,
}

impl GpuBenchTask {
    /// Human-readable label for the task.
    #[must_use]
    pub fn label(&self) -> &str {
        match self {
            Self::VideoDecoding => "Video Decoding",
            Self::VideoEncoding => "Video Encoding",
            Self::ColorConversion => "Color Conversion",
            Self::Denoising => "Denoising",
            Self::SceneDetection => "Scene Detection",
        }
    }

    /// Base throughput for a 1920×1080 input in frames per second.
    #[must_use]
    fn base_fps_hd(&self) -> f32 {
        match self {
            Self::VideoDecoding => 240.0,
            Self::VideoEncoding => 120.0,
            Self::ColorConversion => 480.0,
            Self::Denoising => 60.0,
            Self::SceneDetection => 720.0,
        }
    }

    /// Simulated GPU utilisation percentage at HD resolution.
    #[must_use]
    fn base_utilization_hd(&self) -> f32 {
        match self {
            Self::VideoDecoding => 65.0,
            Self::VideoEncoding => 90.0,
            Self::ColorConversion => 40.0,
            Self::Denoising => 95.0,
            Self::SceneDetection => 30.0,
        }
    }

    /// Simulated VRAM usage in MB at HD resolution.
    #[must_use]
    fn base_vram_mb_hd(&self) -> u32 {
        match self {
            Self::VideoDecoding => 512,
            Self::VideoEncoding => 1024,
            Self::ColorConversion => 256,
            Self::Denoising => 2048,
            Self::SceneDetection => 128,
        }
    }

    /// Simulated power draw in watts at HD resolution.
    #[must_use]
    fn base_power_watts_hd(&self) -> f32 {
        match self {
            Self::VideoDecoding => 50.0,
            Self::VideoEncoding => 120.0,
            Self::ColorConversion => 30.0,
            Self::Denoising => 150.0,
            Self::SceneDetection => 20.0,
        }
    }
}

/// Result of a single GPU benchmark run.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct GpuBenchResult {
    /// The task that was benchmarked.
    pub task: GpuBenchTask,
    /// Achieved throughput in frames per second.
    pub throughput_fps: f32,
    /// GPU utilisation percentage (0–100).
    pub gpu_utilization_pct: f32,
    /// VRAM used in megabytes.
    pub vram_used_mb: u32,
    /// Power draw in watts.
    pub power_watts: f32,
}

impl GpuBenchResult {
    /// Frames per second per watt (efficiency metric).
    #[must_use]
    pub fn efficiency_fps_per_watt(&self) -> f32 {
        if self.power_watts.abs() < f32::EPSILON {
            return 0.0;
        }
        self.throughput_fps / self.power_watts
    }
}

/// Deterministic GPU simulator.
pub struct GpuSimulator;

impl GpuSimulator {
    /// Run a deterministic simulated GPU task.
    ///
    /// The simulation scales base HD performance by a resolution factor and
    /// adds a small deterministic noise component derived from `seed` via a
    /// linear congruential generator (LCG).
    #[must_use]
    pub fn run_task(task: GpuBenchTask, resolution: (u32, u32), seed: u64) -> GpuBenchResult {
        let (w, h) = resolution;
        // Resolution factor relative to HD (1920×1080 = ~2 Mpix).
        let hd_pixels = 1920.0_f32 * 1080.0;
        let input_pixels = w as f32 * h as f32;
        let resolution_factor = (hd_pixels / input_pixels.max(1.0)).sqrt();

        // LCG noise: value in [0.85, 1.15]
        let lcg = lcg_noise(seed);
        let noise = 0.85 + lcg * 0.30;

        let fps = task.base_fps_hd() * resolution_factor * noise as f32;
        let util = (task.base_utilization_hd() * noise as f32).clamp(0.0, 100.0);
        let vram = (task.base_vram_mb_hd() as f32 * (1.0 / resolution_factor)).round() as u32;
        let power = task.base_power_watts_hd() * noise as f32;

        GpuBenchResult {
            task,
            throughput_fps: fps,
            gpu_utilization_pct: util,
            vram_used_mb: vram,
            power_watts: power,
        }
    }
}

/// LCG pseudo-random noise normalised to [0.0, 1.0].
fn lcg_noise(seed: u64) -> f64 {
    // Standard LCG constants from Numerical Recipes.
    let state = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (state >> 33) as f64 / (u32::MAX as f64)
}

/// Comparison between two GPU configurations across multiple tasks.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct GpuComparison {
    /// Name of the first GPU.
    pub gpu_a: String,
    /// Name of the second GPU.
    pub gpu_b: String,
    /// Results for `gpu_a`.
    pub results_a: Vec<GpuBenchResult>,
    /// Results for `gpu_b`.
    pub results_b: Vec<GpuBenchResult>,
}

impl GpuComparison {
    /// Return the name of the GPU that won on a specific task.
    ///
    /// Returns the name of the GPU with the higher `throughput_fps` for that
    /// task.  Returns `"tie"` when neither GPU has a result for the task or the
    /// values are equal.
    #[must_use]
    pub fn winner_by_task(&self, task: &GpuBenchTask) -> &str {
        let fps_a = self
            .results_a
            .iter()
            .find(|r| &r.task == task)
            .map(|r| r.throughput_fps);
        let fps_b = self
            .results_b
            .iter()
            .find(|r| &r.task == task)
            .map(|r| r.throughput_fps);

        match (fps_a, fps_b) {
            (Some(a), Some(b)) => {
                if a > b {
                    &self.gpu_a
                } else if b > a {
                    &self.gpu_b
                } else {
                    "tie"
                }
            }
            (Some(_), None) => &self.gpu_a,
            (None, Some(_)) => &self.gpu_b,
            (None, None) => "tie",
        }
    }

    /// Mean throughput advantage of `gpu_a` over `gpu_b` across all common tasks.
    ///
    /// A positive value means `gpu_a` is faster on average.
    #[must_use]
    pub fn mean_fps_advantage_a(&self) -> f32 {
        let mut diffs = Vec::new();
        for res_a in &self.results_a {
            if let Some(res_b) = self.results_b.iter().find(|r| r.task == res_a.task) {
                diffs.push(res_a.throughput_fps - res_b.throughput_fps);
            }
        }
        if diffs.is_empty() {
            return 0.0;
        }
        diffs.iter().sum::<f32>() / diffs.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_label() {
        assert_eq!(GpuBenchTask::VideoDecoding.label(), "Video Decoding");
        assert_eq!(GpuBenchTask::Denoising.label(), "Denoising");
    }

    #[test]
    fn test_simulator_hd_decoding() {
        let result = GpuSimulator::run_task(GpuBenchTask::VideoDecoding, (1920, 1080), 42);
        assert!(result.throughput_fps > 0.0);
        assert!(result.gpu_utilization_pct > 0.0);
        assert!(result.gpu_utilization_pct <= 100.0);
        assert!(result.vram_used_mb > 0);
        assert!(result.power_watts > 0.0);
    }

    #[test]
    fn test_simulator_4k_lower_fps_than_hd() {
        let hd = GpuSimulator::run_task(GpuBenchTask::VideoEncoding, (1920, 1080), 1);
        let uhd = GpuSimulator::run_task(GpuBenchTask::VideoEncoding, (3840, 2160), 1);
        // At 4K the resolution factor is smaller so throughput should be lower.
        assert!(uhd.throughput_fps < hd.throughput_fps);
    }

    #[test]
    fn test_simulator_deterministic() {
        let r1 = GpuSimulator::run_task(GpuBenchTask::ColorConversion, (1280, 720), 99);
        let r2 = GpuSimulator::run_task(GpuBenchTask::ColorConversion, (1280, 720), 99);
        assert!((r1.throughput_fps - r2.throughput_fps).abs() < f32::EPSILON);
    }

    #[test]
    fn test_efficiency_fps_per_watt() {
        let result = GpuBenchResult {
            task: GpuBenchTask::VideoDecoding,
            throughput_fps: 200.0,
            gpu_utilization_pct: 60.0,
            vram_used_mb: 512,
            power_watts: 100.0,
        };
        assert!((result.efficiency_fps_per_watt() - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_efficiency_zero_power() {
        let result = GpuBenchResult {
            task: GpuBenchTask::SceneDetection,
            throughput_fps: 300.0,
            gpu_utilization_pct: 25.0,
            vram_used_mb: 128,
            power_watts: 0.0,
        };
        assert_eq!(result.efficiency_fps_per_watt(), 0.0);
    }

    #[test]
    fn test_comparison_winner() {
        let r_a = GpuBenchResult {
            task: GpuBenchTask::VideoDecoding,
            throughput_fps: 300.0,
            gpu_utilization_pct: 70.0,
            vram_used_mb: 512,
            power_watts: 80.0,
        };
        let r_b = GpuBenchResult {
            task: GpuBenchTask::VideoDecoding,
            throughput_fps: 200.0,
            gpu_utilization_pct: 65.0,
            vram_used_mb: 512,
            power_watts: 70.0,
        };
        let cmp = GpuComparison {
            gpu_a: "RTX 4090".to_string(),
            gpu_b: "RTX 3080".to_string(),
            results_a: vec![r_a],
            results_b: vec![r_b],
        };
        assert_eq!(cmp.winner_by_task(&GpuBenchTask::VideoDecoding), "RTX 4090");
    }

    #[test]
    fn test_comparison_winner_no_results() {
        let cmp = GpuComparison {
            gpu_a: "A".to_string(),
            gpu_b: "B".to_string(),
            results_a: vec![],
            results_b: vec![],
        };
        assert_eq!(cmp.winner_by_task(&GpuBenchTask::Denoising), "tie");
    }

    #[test]
    fn test_comparison_mean_advantage() {
        let tasks = vec![GpuBenchTask::VideoDecoding, GpuBenchTask::VideoEncoding];
        let results_a: Vec<GpuBenchResult> = tasks
            .iter()
            .map(|t| GpuSimulator::run_task(t.clone(), (1920, 1080), 7))
            .collect();
        let results_b: Vec<GpuBenchResult> = tasks
            .iter()
            .map(|t| GpuSimulator::run_task(t.clone(), (1920, 1080), 13))
            .collect();
        let cmp = GpuComparison {
            gpu_a: "A".to_string(),
            gpu_b: "B".to_string(),
            results_a,
            results_b,
        };
        // Just ensure it computes without panic.
        let _ = cmp.mean_fps_advantage_a();
    }

    #[test]
    fn test_all_tasks_simulate() {
        let tasks = [
            GpuBenchTask::VideoDecoding,
            GpuBenchTask::VideoEncoding,
            GpuBenchTask::ColorConversion,
            GpuBenchTask::Denoising,
            GpuBenchTask::SceneDetection,
        ];
        for task in &tasks {
            let r = GpuSimulator::run_task(task.clone(), (1920, 1080), 0);
            assert!(
                r.throughput_fps > 0.0,
                "{} should produce positive fps",
                task.label()
            );
        }
    }

    #[test]
    fn test_lcg_noise_range() {
        for seed in [0u64, 1, 42, u64::MAX] {
            let n = super::lcg_noise(seed);
            assert!(
                (0.0..=1.0).contains(&n),
                "LCG noise out of range for seed {seed}"
            );
        }
    }
}
