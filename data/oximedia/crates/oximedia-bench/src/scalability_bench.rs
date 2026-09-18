#![allow(dead_code)]
//! Scalability benchmarking across thread counts and resolutions.
//!
//! This module measures how encoding/decoding performance scales with the
//! number of worker threads, input resolution, and frame count, producing
//! efficiency curves and Amdahl's-law estimates.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Resolution preset for scalability tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResolutionPreset {
    /// 640x360 (360p).
    Res360p,
    /// 1280x720 (720p).
    Res720p,
    /// 1920x1080 (1080p).
    Res1080p,
    /// 2560x1440 (1440p / 2K).
    Res1440p,
    /// 3840x2160 (4K UHD).
    Res4k,
    /// 7680x4320 (8K UHD).
    Res8k,
}

impl ResolutionPreset {
    /// Width in pixels.
    pub fn width(self) -> u32 {
        match self {
            Self::Res360p => 640,
            Self::Res720p => 1280,
            Self::Res1080p => 1920,
            Self::Res1440p => 2560,
            Self::Res4k => 3840,
            Self::Res8k => 7680,
        }
    }

    /// Height in pixels.
    pub fn height(self) -> u32 {
        match self {
            Self::Res360p => 360,
            Self::Res720p => 720,
            Self::Res1080p => 1080,
            Self::Res1440p => 1440,
            Self::Res4k => 2160,
            Self::Res8k => 4320,
        }
    }

    /// Total pixel count.
    pub fn pixel_count(self) -> u64 {
        u64::from(self.width()) * u64::from(self.height())
    }

    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Res360p => "360p",
            Self::Res720p => "720p",
            Self::Res1080p => "1080p",
            Self::Res1440p => "1440p",
            Self::Res4k => "4K",
            Self::Res8k => "8K",
        }
    }
}

/// Configuration for a scalability benchmark run.
#[derive(Debug, Clone)]
pub struct ScalabilityConfig {
    /// Thread counts to test.
    pub thread_counts: Vec<usize>,
    /// Resolutions to test.
    pub resolutions: Vec<ResolutionPreset>,
    /// Number of frames per test.
    pub frame_count: usize,
    /// Number of measurement iterations per (threads, resolution) combination.
    pub iterations: u32,
    /// Label / name for this benchmark run.
    pub label: String,
}

impl Default for ScalabilityConfig {
    fn default() -> Self {
        Self {
            thread_counts: vec![1, 2, 4, 8],
            resolutions: vec![
                ResolutionPreset::Res720p,
                ResolutionPreset::Res1080p,
                ResolutionPreset::Res4k,
            ],
            frame_count: 100,
            iterations: 3,
            label: String::from("default"),
        }
    }
}

impl ScalabilityConfig {
    /// Create a new default config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set thread counts.
    pub fn with_thread_counts(mut self, tc: Vec<usize>) -> Self {
        self.thread_counts = tc;
        self
    }

    /// Builder: set resolutions.
    pub fn with_resolutions(mut self, res: Vec<ResolutionPreset>) -> Self {
        self.resolutions = res;
        self
    }

    /// Builder: set frame count.
    pub fn with_frame_count(mut self, n: usize) -> Self {
        self.frame_count = n.max(1);
        self
    }

    /// Builder: set iterations.
    pub fn with_iterations(mut self, n: u32) -> Self {
        self.iterations = n.max(1);
        self
    }

    /// Builder: set label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    /// Total number of test combinations.
    pub fn combination_count(&self) -> usize {
        self.thread_counts.len() * self.resolutions.len()
    }

    /// Validate.
    pub fn validate(&self) -> Result<(), String> {
        if self.thread_counts.is_empty() {
            return Err("thread_counts must not be empty".into());
        }
        if self.resolutions.is_empty() {
            return Err("resolutions must not be empty".into());
        }
        if self.thread_counts.iter().any(|&t| t == 0) {
            return Err("thread count must be > 0".into());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

/// A single measurement point.
#[derive(Debug, Clone)]
pub struct ScalabilityPoint {
    /// Thread count used.
    pub threads: usize,
    /// Resolution used.
    pub resolution: ResolutionPreset,
    /// Measured FPS (mean over iterations).
    pub fps: f64,
    /// Standard deviation of FPS across iterations.
    pub fps_stddev: f64,
    /// Throughput in megapixels per second.
    pub megapixels_per_sec: f64,
}

impl ScalabilityPoint {
    /// Compute speed-up relative to a single-thread baseline FPS.
    pub fn speedup(&self, single_thread_fps: f64) -> f64 {
        if single_thread_fps <= 0.0 {
            return 0.0;
        }
        self.fps / single_thread_fps
    }

    /// Compute parallel efficiency (speedup / threads).
    #[allow(clippy::cast_precision_loss)]
    pub fn efficiency(&self, single_thread_fps: f64) -> f64 {
        if self.threads == 0 {
            return 0.0;
        }
        self.speedup(single_thread_fps) / self.threads as f64
    }
}

/// Aggregated scalability results for one resolution across all thread counts.
#[derive(Debug, Clone)]
pub struct ResolutionScalability {
    /// Resolution tested.
    pub resolution: ResolutionPreset,
    /// Points ordered by thread count.
    pub points: Vec<ScalabilityPoint>,
    /// Estimated serial fraction from Amdahl's law.
    pub amdahl_serial_fraction: f64,
}

impl ResolutionScalability {
    /// Compute from a set of points (must be for the same resolution).
    #[allow(clippy::cast_precision_loss)]
    pub fn from_points(resolution: ResolutionPreset, points: Vec<ScalabilityPoint>) -> Self {
        let serial_fraction = Self::estimate_amdahl(&points);
        Self {
            resolution,
            points,
            amdahl_serial_fraction: serial_fraction,
        }
    }

    /// Estimate the serial fraction using Amdahl's law from 1-thread and max-thread data.
    #[allow(clippy::cast_precision_loss)]
    fn estimate_amdahl(points: &[ScalabilityPoint]) -> f64 {
        if points.len() < 2 {
            return 1.0;
        }
        let single = points.iter().find(|p| p.threads == 1);
        let max_pt = points.iter().max_by_key(|p| p.threads);
        match (single, max_pt) {
            (Some(s), Some(m)) if m.threads > 1 && s.fps > 0.0 => {
                let speedup = m.fps / s.fps;
                let n = m.threads as f64;
                // Amdahl: S = 1 / (f + (1-f)/n)  =>  f = (1/S - 1/n) / (1 - 1/n)
                let inv_s = 1.0 / speedup;
                let inv_n = 1.0 / n;
                let denom = 1.0 - inv_n;
                if denom.abs() < 1e-15 {
                    return 1.0;
                }
                ((inv_s - inv_n) / denom).clamp(0.0, 1.0)
            }
            _ => 1.0,
        }
    }

    /// Maximum speed-up observed.
    pub fn max_speedup(&self) -> f64 {
        let base = self
            .points
            .iter()
            .find(|p| p.threads == 1)
            .map(|p| p.fps)
            .unwrap_or(1.0);
        self.points
            .iter()
            .map(|p| p.speedup(base))
            .fold(0.0_f64, f64::max)
    }
}

/// Complete scalability benchmark results.
#[derive(Debug, Clone)]
pub struct ScalabilityResult {
    /// Configuration used.
    pub config: ScalabilityConfig,
    /// Results per resolution.
    pub per_resolution: Vec<ResolutionScalability>,
    /// Flat map of all points for easy lookup.
    pub all_points: HashMap<(usize, String), ScalabilityPoint>,
}

impl ScalabilityResult {
    /// Number of data points.
    pub fn point_count(&self) -> usize {
        self.all_points.len()
    }

    /// Produce a human-readable summary.
    pub fn summary(&self) -> String {
        let mut s = format!("Scalability Results ({})\n", self.config.label);
        for rs in &self.per_resolution {
            s.push_str(&format!(
                "  {}: max speedup={:.2}x, serial fraction={:.2}%\n",
                rs.resolution.label(),
                rs.max_speedup(),
                rs.amdahl_serial_fraction * 100.0,
            ));
        }
        s
    }
}

// ---------------------------------------------------------------------------
// Runner (simulated)
// ---------------------------------------------------------------------------

/// Simulated scalability benchmark runner.
#[derive(Debug)]
pub struct ScalabilityRunner {
    /// Config.
    config: ScalabilityConfig,
}

impl ScalabilityRunner {
    /// Create a new runner.
    pub fn new(config: ScalabilityConfig) -> Self {
        Self { config }
    }

    /// Run the benchmark (simulated: models diminishing returns).
    #[allow(clippy::cast_precision_loss)]
    pub fn run(&self) -> ScalabilityResult {
        let mut per_resolution = Vec::new();
        let mut all_points = HashMap::new();

        for &res in &self.config.resolutions {
            let mut points = Vec::new();
            let base_fps = 100.0 / (res.pixel_count() as f64 / 1_000_000.0);
            let serial_frac = 0.1; // simulated 10% serial

            for &threads in &self.config.thread_counts {
                // Amdahl model: fps = base * 1/(f + (1-f)/n)
                let n = threads as f64;
                let speedup = 1.0 / (serial_frac + (1.0 - serial_frac) / n);
                let fps = base_fps * speedup;
                let megapixels_per_sec = fps * res.pixel_count() as f64 / 1_000_000.0;

                let pt = ScalabilityPoint {
                    threads,
                    resolution: res,
                    fps,
                    fps_stddev: fps * 0.02, // simulated 2% noise
                    megapixels_per_sec,
                };
                all_points.insert((threads, res.label().to_string()), pt.clone());
                points.push(pt);
            }
            per_resolution.push(ResolutionScalability::from_points(res, points));
        }

        ScalabilityResult {
            config: self.config.clone(),
            per_resolution,
            all_points,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolution_preset_dimensions() {
        assert_eq!(ResolutionPreset::Res1080p.width(), 1920);
        assert_eq!(ResolutionPreset::Res1080p.height(), 1080);
        assert_eq!(ResolutionPreset::Res1080p.pixel_count(), 1920 * 1080);
    }

    #[test]
    fn test_resolution_preset_label() {
        assert_eq!(ResolutionPreset::Res4k.label(), "4K");
        assert_eq!(ResolutionPreset::Res720p.label(), "720p");
    }

    #[test]
    fn test_config_default() {
        let cfg = ScalabilityConfig::default();
        assert!(!cfg.thread_counts.is_empty());
        assert!(!cfg.resolutions.is_empty());
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_builder() {
        let cfg = ScalabilityConfig::new()
            .with_thread_counts(vec![1, 2, 4])
            .with_resolutions(vec![ResolutionPreset::Res1080p])
            .with_frame_count(50)
            .with_iterations(5)
            .with_label("test");
        assert_eq!(cfg.thread_counts, vec![1, 2, 4]);
        assert_eq!(cfg.frame_count, 50);
        assert_eq!(cfg.label, "test");
    }

    #[test]
    fn test_config_combination_count() {
        let cfg = ScalabilityConfig::new()
            .with_thread_counts(vec![1, 2, 4])
            .with_resolutions(vec![ResolutionPreset::Res720p, ResolutionPreset::Res1080p]);
        assert_eq!(cfg.combination_count(), 6);
    }

    #[test]
    fn test_config_validate_empty_threads() {
        let cfg = ScalabilityConfig::new().with_thread_counts(vec![]);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_zero_thread() {
        let cfg = ScalabilityConfig::new().with_thread_counts(vec![0, 1]);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_scalability_point_speedup() {
        let pt = ScalabilityPoint {
            threads: 4,
            resolution: ResolutionPreset::Res1080p,
            fps: 200.0,
            fps_stddev: 5.0,
            megapixels_per_sec: 100.0,
        };
        assert!((pt.speedup(100.0) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_scalability_point_efficiency() {
        let pt = ScalabilityPoint {
            threads: 4,
            resolution: ResolutionPreset::Res1080p,
            fps: 200.0,
            fps_stddev: 5.0,
            megapixels_per_sec: 100.0,
        };
        assert!((pt.efficiency(100.0) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_resolution_scalability_amdahl() {
        let points = vec![
            ScalabilityPoint {
                threads: 1,
                resolution: ResolutionPreset::Res1080p,
                fps: 100.0,
                fps_stddev: 1.0,
                megapixels_per_sec: 50.0,
            },
            ScalabilityPoint {
                threads: 8,
                resolution: ResolutionPreset::Res1080p,
                fps: 500.0,
                fps_stddev: 5.0,
                megapixels_per_sec: 250.0,
            },
        ];
        let rs = ResolutionScalability::from_points(ResolutionPreset::Res1080p, points);
        // serial fraction should be between 0 and 1
        assert!(rs.amdahl_serial_fraction >= 0.0 && rs.amdahl_serial_fraction <= 1.0);
    }

    #[test]
    fn test_resolution_scalability_max_speedup() {
        let points = vec![
            ScalabilityPoint {
                threads: 1,
                resolution: ResolutionPreset::Res720p,
                fps: 100.0,
                fps_stddev: 1.0,
                megapixels_per_sec: 50.0,
            },
            ScalabilityPoint {
                threads: 4,
                resolution: ResolutionPreset::Res720p,
                fps: 350.0,
                fps_stddev: 3.0,
                megapixels_per_sec: 175.0,
            },
        ];
        let rs = ResolutionScalability::from_points(ResolutionPreset::Res720p, points);
        assert!(rs.max_speedup() > 3.0);
    }

    #[test]
    fn test_runner_simulated() {
        let cfg = ScalabilityConfig::new()
            .with_thread_counts(vec![1, 2, 4])
            .with_resolutions(vec![ResolutionPreset::Res1080p]);
        let runner = ScalabilityRunner::new(cfg);
        let result = runner.run();
        assert_eq!(result.per_resolution.len(), 1);
        assert_eq!(result.point_count(), 3);
    }

    #[test]
    fn test_result_summary() {
        let cfg = ScalabilityConfig::new()
            .with_thread_counts(vec![1, 4])
            .with_resolutions(vec![ResolutionPreset::Res720p])
            .with_label("test_run");
        let runner = ScalabilityRunner::new(cfg);
        let result = runner.run();
        let summary = result.summary();
        assert!(summary.contains("test_run"));
        assert!(summary.contains("720p"));
    }

    #[test]
    fn test_speedup_zero_baseline() {
        let pt = ScalabilityPoint {
            threads: 4,
            resolution: ResolutionPreset::Res1080p,
            fps: 100.0,
            fps_stddev: 1.0,
            megapixels_per_sec: 50.0,
        };
        assert!((pt.speedup(0.0) - 0.0).abs() < 1e-9);
    }
}
