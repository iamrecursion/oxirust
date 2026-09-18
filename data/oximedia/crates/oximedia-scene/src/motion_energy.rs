//! Motion energy module for quantifying overall motion intensity per scene segment.
//!
//! Computes per-frame and per-segment motion energy using inter-frame difference
//! analysis. Useful for detecting high-action segments, selecting representative
//! frames (low-motion), and pacing analysis.

use crate::error::{SceneError, SceneResult};
use serde::{Deserialize, Serialize};

/// Motion energy measurement for a single frame pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameMotionEnergy {
    /// Frame index.
    pub frame_index: u64,
    /// Overall motion energy (0.0-1.0).
    pub energy: f32,
    /// Motion energy per spatial quadrant [top-left, top-right, bottom-left, bottom-right].
    pub quadrant_energy: [f32; 4],
    /// Percentage of pixels with significant motion (0.0-1.0).
    pub motion_coverage: f32,
}

/// Motion energy summary for a scene segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentMotionEnergy {
    /// Start frame index.
    pub start_frame: u64,
    /// End frame index.
    pub end_frame: u64,
    /// Mean motion energy across the segment.
    pub mean_energy: f32,
    /// Peak motion energy within the segment.
    pub peak_energy: f32,
    /// Frame index of the peak.
    pub peak_frame: u64,
    /// Standard deviation of motion energy.
    pub energy_stddev: f32,
    /// Motion intensity classification.
    pub intensity: MotionIntensity,
    /// Per-frame energies.
    pub frame_energies: Vec<FrameMotionEnergy>,
}

/// Motion intensity classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MotionIntensity {
    /// Very low motion (static or near-static).
    VeryLow,
    /// Low motion (slow camera moves, subtle changes).
    Low,
    /// Moderate motion (typical dialogue scenes).
    Moderate,
    /// High motion (action sequences, fast camera moves).
    High,
    /// Very high motion (extreme action, rapid cuts).
    VeryHigh,
}

impl MotionIntensity {
    /// Human-readable label.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::VeryLow => "very_low",
            Self::Low => "low",
            Self::Moderate => "moderate",
            Self::High => "high",
            Self::VeryHigh => "very_high",
        }
    }

    /// Classify from a mean energy value.
    #[must_use]
    pub fn from_energy(energy: f32) -> Self {
        if energy < 0.02 {
            Self::VeryLow
        } else if energy < 0.08 {
            Self::Low
        } else if energy < 0.2 {
            Self::Moderate
        } else if energy < 0.5 {
            Self::High
        } else {
            Self::VeryHigh
        }
    }
}

/// Configuration for motion energy computation.
#[derive(Debug, Clone)]
pub struct MotionEnergyConfig {
    /// Pixel difference threshold to count as "motion" (0-255 scale, normalized to 0-1).
    pub motion_threshold: f32,
    /// Whether to compute per-quadrant energy.
    pub compute_quadrants: bool,
    /// Downscale factor for faster computation (1 = full resolution).
    pub downscale: usize,
}

impl Default for MotionEnergyConfig {
    fn default() -> Self {
        Self {
            motion_threshold: 0.03,
            compute_quadrants: true,
            downscale: 1,
        }
    }
}

/// Motion energy analyzer.
pub struct MotionEnergyAnalyzer {
    config: MotionEnergyConfig,
    prev_gray: Option<Vec<f32>>,
    prev_width: usize,
    prev_height: usize,
    frame_energies: Vec<FrameMotionEnergy>,
    frame_counter: u64,
}

impl MotionEnergyAnalyzer {
    /// Create a new motion energy analyzer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: MotionEnergyConfig::default(),
            prev_gray: None,
            prev_width: 0,
            prev_height: 0,
            frame_energies: Vec::new(),
            frame_counter: 0,
        }
    }

    /// Create with custom configuration.
    #[must_use]
    pub fn with_config(config: MotionEnergyConfig) -> Self {
        Self {
            config,
            prev_gray: None,
            prev_width: 0,
            prev_height: 0,
            frame_energies: Vec::new(),
            frame_counter: 0,
        }
    }

    /// Process a frame and compute motion energy.
    ///
    /// # Errors
    ///
    /// Returns error if dimensions are invalid.
    pub fn process_frame(
        &mut self,
        rgb_data: &[u8],
        width: usize,
        height: usize,
    ) -> SceneResult<FrameMotionEnergy> {
        if rgb_data.len() != width * height * 3 {
            return Err(SceneError::InvalidDimensions(
                "RGB data size mismatch".to_string(),
            ));
        }

        let gray = rgb_to_gray_f32(rgb_data);
        let frame_index = self.frame_counter;
        self.frame_counter += 1;

        let energy = if let Some(ref prev) = self.prev_gray {
            if prev.len() == gray.len() && self.prev_width == width && self.prev_height == height {
                self.compute_motion_energy(prev, &gray, width, height, frame_index)
            } else {
                FrameMotionEnergy {
                    frame_index,
                    energy: 0.0,
                    quadrant_energy: [0.0; 4],
                    motion_coverage: 0.0,
                }
            }
        } else {
            FrameMotionEnergy {
                frame_index,
                energy: 0.0,
                quadrant_energy: [0.0; 4],
                motion_coverage: 0.0,
            }
        };

        self.prev_gray = Some(gray);
        self.prev_width = width;
        self.prev_height = height;
        self.frame_energies.push(energy.clone());

        Ok(energy)
    }

    /// Get the accumulated segment motion energy summary.
    #[must_use]
    pub fn summarize(&self) -> SegmentMotionEnergy {
        if self.frame_energies.is_empty() {
            return SegmentMotionEnergy {
                start_frame: 0,
                end_frame: 0,
                mean_energy: 0.0,
                peak_energy: 0.0,
                peak_frame: 0,
                energy_stddev: 0.0,
                intensity: MotionIntensity::VeryLow,
                frame_energies: Vec::new(),
            };
        }

        let start_frame = self.frame_energies.first().map_or(0, |f| f.frame_index);
        let end_frame = self.frame_energies.last().map_or(0, |f| f.frame_index);

        let energies: Vec<f32> = self.frame_energies.iter().map(|f| f.energy).collect();
        let mean_energy = energies.iter().sum::<f32>() / energies.len() as f32;

        let (peak_energy, peak_frame) =
            self.frame_energies
                .iter()
                .fold((0.0_f32, 0_u64), |(max_e, max_f), f| {
                    if f.energy > max_e {
                        (f.energy, f.frame_index)
                    } else {
                        (max_e, max_f)
                    }
                });

        let variance = energies
            .iter()
            .map(|e| (e - mean_energy).powi(2))
            .sum::<f32>()
            / energies.len() as f32;
        let energy_stddev = variance.sqrt();

        let intensity = MotionIntensity::from_energy(mean_energy);

        SegmentMotionEnergy {
            start_frame,
            end_frame,
            mean_energy,
            peak_energy,
            peak_frame,
            energy_stddev,
            intensity,
            frame_energies: self.frame_energies.clone(),
        }
    }

    /// Reset the analyzer state.
    pub fn reset(&mut self) {
        self.prev_gray = None;
        self.frame_energies.clear();
        self.frame_counter = 0;
    }

    /// Compute motion energy between two grayscale frames.
    fn compute_motion_energy(
        &self,
        prev: &[f32],
        curr: &[f32],
        width: usize,
        height: usize,
        frame_index: u64,
    ) -> FrameMotionEnergy {
        let step = self.config.downscale.max(1);
        let mut total_diff = 0.0_f64;
        let mut motion_pixels = 0_u64;
        let mut total_pixels = 0_u64;
        let mut quadrant_diff = [0.0_f64; 4];
        let mut quadrant_count = [0_u64; 4];

        let mid_x = width / 2;
        let mid_y = height / 2;

        for y in (0..height).step_by(step) {
            for x in (0..width).step_by(step) {
                let idx = y * width + x;
                let diff = (curr[idx] - prev[idx]).abs();
                total_diff += diff as f64;
                total_pixels += 1;

                if diff > self.config.motion_threshold {
                    motion_pixels += 1;
                }

                if self.config.compute_quadrants {
                    let q = if x < mid_x {
                        if y < mid_y {
                            0
                        } else {
                            2
                        }
                    } else if y < mid_y {
                        1
                    } else {
                        3
                    };
                    quadrant_diff[q] += diff as f64;
                    quadrant_count[q] += 1;
                }
            }
        }

        let energy = if total_pixels > 0 {
            (total_diff / total_pixels as f64) as f32
        } else {
            0.0
        };

        let motion_coverage = if total_pixels > 0 {
            motion_pixels as f32 / total_pixels as f32
        } else {
            0.0
        };

        let quadrant_energy = if self.config.compute_quadrants {
            [
                if quadrant_count[0] > 0 {
                    (quadrant_diff[0] / quadrant_count[0] as f64) as f32
                } else {
                    0.0
                },
                if quadrant_count[1] > 0 {
                    (quadrant_diff[1] / quadrant_count[1] as f64) as f32
                } else {
                    0.0
                },
                if quadrant_count[2] > 0 {
                    (quadrant_diff[2] / quadrant_count[2] as f64) as f32
                } else {
                    0.0
                },
                if quadrant_count[3] > 0 {
                    (quadrant_diff[3] / quadrant_count[3] as f64) as f32
                } else {
                    0.0
                },
            ]
        } else {
            [0.0; 4]
        };

        FrameMotionEnergy {
            frame_index,
            energy,
            quadrant_energy,
            motion_coverage,
        }
    }
}

impl Default for MotionEnergyAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert RGB data to grayscale (0.0-1.0).
fn rgb_to_gray_f32(rgb: &[u8]) -> Vec<f32> {
    let mut gray = Vec::with_capacity(rgb.len() / 3);
    for chunk in rgb.chunks_exact(3) {
        gray.push(
            (0.299 * chunk[0] as f32 + 0.587 * chunk[1] as f32 + 0.114 * chunk[2] as f32) / 255.0,
        );
    }
    gray
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motion_intensity_classification() {
        assert_eq!(MotionIntensity::from_energy(0.01), MotionIntensity::VeryLow);
        assert_eq!(MotionIntensity::from_energy(0.05), MotionIntensity::Low);
        assert_eq!(
            MotionIntensity::from_energy(0.15),
            MotionIntensity::Moderate
        );
        assert_eq!(MotionIntensity::from_energy(0.3), MotionIntensity::High);
        assert_eq!(MotionIntensity::from_energy(0.6), MotionIntensity::VeryHigh);
    }

    #[test]
    fn test_motion_intensity_labels() {
        assert_eq!(MotionIntensity::VeryLow.label(), "very_low");
        assert_eq!(MotionIntensity::High.label(), "high");
    }

    #[test]
    fn test_single_frame_no_motion() {
        let mut analyzer = MotionEnergyAnalyzer::new();
        let width = 50;
        let height = 50;
        let frame = vec![128u8; width * height * 3];

        let result = analyzer.process_frame(&frame, width, height);
        assert!(result.is_ok());
        let energy = result.expect("should succeed");
        assert!((energy.energy - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_two_identical_frames() {
        let mut analyzer = MotionEnergyAnalyzer::new();
        let width = 50;
        let height = 50;
        let frame = vec![128u8; width * height * 3];

        let _ = analyzer.process_frame(&frame, width, height);
        let result = analyzer.process_frame(&frame, width, height);
        assert!(result.is_ok());
        let energy = result.expect("should succeed");
        assert!((energy.energy - 0.0).abs() < f32::EPSILON);
        assert!((energy.motion_coverage - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_two_different_frames() {
        let mut analyzer = MotionEnergyAnalyzer::new();
        let width = 50;
        let height = 50;
        let frame1 = vec![50u8; width * height * 3];
        let frame2 = vec![200u8; width * height * 3];

        let _ = analyzer.process_frame(&frame1, width, height);
        let result = analyzer.process_frame(&frame2, width, height);
        assert!(result.is_ok());
        let energy = result.expect("should succeed");
        assert!(energy.energy > 0.0);
        assert!(energy.motion_coverage > 0.0);
    }

    #[test]
    fn test_summarize_empty() {
        let analyzer = MotionEnergyAnalyzer::new();
        let summary = analyzer.summarize();
        assert!((summary.mean_energy - 0.0).abs() < f32::EPSILON);
        assert_eq!(summary.intensity, MotionIntensity::VeryLow);
    }

    #[test]
    fn test_summarize_with_frames() {
        let mut analyzer = MotionEnergyAnalyzer::new();
        let width = 50;
        let height = 50;

        for i in 0..5 {
            let val = (i * 50) as u8;
            let frame = vec![val; width * height * 3];
            let _ = analyzer.process_frame(&frame, width, height);
        }

        let summary = analyzer.summarize();
        assert_eq!(summary.start_frame, 0);
        assert_eq!(summary.end_frame, 4);
        assert!(summary.frame_energies.len() == 5);
    }

    #[test]
    fn test_reset() {
        let mut analyzer = MotionEnergyAnalyzer::new();
        let frame = vec![128u8; 50 * 50 * 3];
        let _ = analyzer.process_frame(&frame, 50, 50);
        analyzer.reset();
        let summary = analyzer.summarize();
        assert!(summary.frame_energies.is_empty());
    }

    #[test]
    fn test_invalid_dimensions() {
        let mut analyzer = MotionEnergyAnalyzer::new();
        let result = analyzer.process_frame(&[0u8; 10], 100, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_quadrant_energy() {
        let mut analyzer = MotionEnergyAnalyzer::new();
        let width = 100;
        let height = 100;

        let frame1 = vec![100u8; width * height * 3];
        let mut frame2 = vec![100u8; width * height * 3];
        // Add motion only in top-left quadrant
        for y in 0..50 {
            for x in 0..50 {
                let idx = (y * width + x) * 3;
                frame2[idx] = 200;
                frame2[idx + 1] = 200;
                frame2[idx + 2] = 200;
            }
        }

        let _ = analyzer.process_frame(&frame1, width, height);
        let result = analyzer.process_frame(&frame2, width, height);
        assert!(result.is_ok());
        let energy = result.expect("should succeed");
        // Top-left should have more energy than bottom-right
        assert!(energy.quadrant_energy[0] > energy.quadrant_energy[3]);
    }

    #[test]
    fn test_custom_config() {
        let config = MotionEnergyConfig {
            motion_threshold: 0.1,
            compute_quadrants: false,
            downscale: 2,
        };
        let mut analyzer = MotionEnergyAnalyzer::with_config(config);
        let frame = vec![128u8; 100 * 100 * 3];
        let result = analyzer.process_frame(&frame, 100, 100);
        assert!(result.is_ok());
    }

    #[test]
    fn test_peak_frame_detection() {
        let mut analyzer = MotionEnergyAnalyzer::new();
        let width = 50;
        let height = 50;

        // Frame 0: baseline
        let _ = analyzer.process_frame(&vec![100u8; width * height * 3], width, height);
        // Frame 1: small change
        let _ = analyzer.process_frame(&vec![110u8; width * height * 3], width, height);
        // Frame 2: big change
        let _ = analyzer.process_frame(&vec![250u8; width * height * 3], width, height);
        // Frame 3: small change
        let _ = analyzer.process_frame(&vec![240u8; width * height * 3], width, height);

        let summary = analyzer.summarize();
        assert_eq!(summary.peak_frame, 2);
    }
}
