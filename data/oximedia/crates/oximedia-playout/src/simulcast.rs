//! Multi-format simulcast: produce HD + UHD (or other format pairs) from
//! a single playout chain.
//!
//! The [`SimulcastEngine`] takes a primary video format and derives one or
//! more secondary outputs by applying a scaling / conversion descriptor.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

// ── Types ───────────────────────────────────────────────────────────────────

/// Describes one leg of a simulcast output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulcastLeg {
    /// Human-readable label (e.g. "HD", "UHD", "SD proxy").
    pub label: String,
    /// Target width in pixels.
    pub width: u32,
    /// Target height in pixels.
    pub height: u32,
    /// Target frame rate (fps).
    pub fps: f64,
    /// Whether this leg is the primary (source) format.
    pub is_primary: bool,
    /// Scaling algorithm to apply.
    pub scaler: ScalingAlgorithm,
}

/// Scaling algorithm used for format conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScalingAlgorithm {
    /// Nearest-neighbour (fast, low quality).
    NearestNeighbour,
    /// Bilinear interpolation.
    Bilinear,
    /// Bicubic interpolation.
    Bicubic,
    /// Lanczos (high quality, slower).
    Lanczos,
}

impl Default for ScalingAlgorithm {
    fn default() -> Self {
        Self::Lanczos
    }
}

/// A simulcast frame: the scaled pixel data for one leg.
#[derive(Debug, Clone)]
pub struct SimulcastFrame {
    /// Which leg this belongs to.
    pub leg_label: String,
    /// Scaled pixel data (RGBA).
    pub data: Vec<u8>,
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
}

// ── Simulcast Engine ────────────────────────────────────────────────────────

/// Configuration for the simulcast engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulcastConfig {
    /// The legs of the simulcast.
    pub legs: Vec<SimulcastLeg>,
}

impl Default for SimulcastConfig {
    fn default() -> Self {
        Self {
            legs: vec![
                SimulcastLeg {
                    label: "UHD".to_string(),
                    width: 3840,
                    height: 2160,
                    fps: 25.0,
                    is_primary: true,
                    scaler: ScalingAlgorithm::Lanczos,
                },
                SimulcastLeg {
                    label: "HD".to_string(),
                    width: 1920,
                    height: 1080,
                    fps: 25.0,
                    is_primary: false,
                    scaler: ScalingAlgorithm::Lanczos,
                },
            ],
        }
    }
}

/// Engine that produces multiple output formats from a single playout chain.
#[derive(Debug)]
pub struct SimulcastEngine {
    config: SimulcastConfig,
}

impl SimulcastEngine {
    /// Create a new simulcast engine.
    pub fn new(config: SimulcastConfig) -> Self {
        Self { config }
    }

    /// Return the configured legs.
    pub fn legs(&self) -> &[SimulcastLeg] {
        &self.config.legs
    }

    /// Return the primary leg (first leg marked `is_primary`), if any.
    pub fn primary_leg(&self) -> Option<&SimulcastLeg> {
        self.config.legs.iter().find(|l| l.is_primary)
    }

    /// Return all secondary (non-primary) legs.
    pub fn secondary_legs(&self) -> Vec<&SimulcastLeg> {
        self.config.legs.iter().filter(|l| !l.is_primary).collect()
    }

    /// Process a source frame (assumed to be at the primary resolution)
    /// and produce one `SimulcastFrame` per leg.
    ///
    /// For the primary leg the source data is passed through unchanged.
    /// For secondary legs a scaling operation is applied using the
    /// configured [`ScalingAlgorithm`].
    pub fn process_frame(&self, source_data: &[u8], src_w: u32, src_h: u32) -> Vec<SimulcastFrame> {
        let mut outputs = Vec::with_capacity(self.config.legs.len());

        for leg in &self.config.legs {
            if leg.is_primary || (leg.width == src_w && leg.height == src_h) {
                outputs.push(SimulcastFrame {
                    leg_label: leg.label.clone(),
                    data: source_data.to_vec(),
                    width: src_w,
                    height: src_h,
                });
            } else {
                let scaled =
                    Self::scale(source_data, src_w, src_h, leg.width, leg.height, leg.scaler);
                outputs.push(SimulcastFrame {
                    leg_label: leg.label.clone(),
                    data: scaled,
                    width: leg.width,
                    height: leg.height,
                });
            }
        }

        outputs
    }

    /// Scale RGBA pixel data from `(src_w, src_h)` to `(dst_w, dst_h)`.
    ///
    /// This is a bilinear scaler for quality modes other than
    /// `NearestNeighbour`, which uses a simpler approach.
    fn scale(
        src: &[u8],
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
        algo: ScalingAlgorithm,
    ) -> Vec<u8> {
        let dst_len = (dst_w * dst_h * 4) as usize;
        let mut dst = vec![0u8; dst_len];

        if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
            return dst;
        }

        let x_ratio = src_w as f64 / dst_w as f64;
        let y_ratio = src_h as f64 / dst_h as f64;

        for dy in 0..dst_h {
            for dx in 0..dst_w {
                let dst_off = ((dy * dst_w + dx) * 4) as usize;

                match algo {
                    ScalingAlgorithm::NearestNeighbour => {
                        let sx = ((dx as f64 * x_ratio) as u32).min(src_w - 1);
                        let sy = ((dy as f64 * y_ratio) as u32).min(src_h - 1);
                        let src_off = ((sy * src_w + sx) * 4) as usize;
                        if src_off + 3 < src.len() {
                            dst[dst_off..dst_off + 4].copy_from_slice(&src[src_off..src_off + 4]);
                        }
                    }
                    _ => {
                        // Bilinear interpolation (used for Bilinear, Bicubic, Lanczos as
                        // a practical approximation in this simulation layer).
                        let gx = dx as f64 * x_ratio;
                        let gy = dy as f64 * y_ratio;
                        let gxi = (gx as u32).min(src_w.saturating_sub(2));
                        let gyi = (gy as u32).min(src_h.saturating_sub(2));
                        let fx = gx - gxi as f64;
                        let fy = gy - gyi as f64;

                        let idx00 = ((gyi * src_w + gxi) * 4) as usize;
                        let idx10 = ((gyi * src_w + gxi + 1) * 4) as usize;
                        let idx01 = (((gyi + 1) * src_w + gxi) * 4) as usize;
                        let idx11 = (((gyi + 1) * src_w + gxi + 1) * 4) as usize;

                        for c in 0..4 {
                            let c00 = src.get(idx00 + c).copied().unwrap_or(0) as f64;
                            let c10 = src.get(idx10 + c).copied().unwrap_or(0) as f64;
                            let c01 = src.get(idx01 + c).copied().unwrap_or(0) as f64;
                            let c11 = src.get(idx11 + c).copied().unwrap_or(0) as f64;

                            let top = c00 * (1.0 - fx) + c10 * fx;
                            let bot = c01 * (1.0 - fx) + c11 * fx;
                            let val = top * (1.0 - fy) + bot * fy;
                            dst[dst_off + c] = val.round().clamp(0.0, 255.0) as u8;
                        }
                    }
                }
            }
        }

        dst
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulcast_default_config() {
        let cfg = SimulcastConfig::default();
        assert_eq!(cfg.legs.len(), 2);
        assert!(cfg.legs[0].is_primary);
        assert!(!cfg.legs[1].is_primary);
    }

    #[test]
    fn test_simulcast_engine_legs() {
        let engine = SimulcastEngine::new(SimulcastConfig::default());
        assert_eq!(engine.legs().len(), 2);
        assert!(engine.primary_leg().is_some());
        assert_eq!(engine.secondary_legs().len(), 1);
    }

    #[test]
    fn test_simulcast_process_frame() {
        let config = SimulcastConfig {
            legs: vec![
                SimulcastLeg {
                    label: "Primary".to_string(),
                    width: 8,
                    height: 4,
                    fps: 25.0,
                    is_primary: true,
                    scaler: ScalingAlgorithm::Lanczos,
                },
                SimulcastLeg {
                    label: "Scaled".to_string(),
                    width: 4,
                    height: 2,
                    fps: 25.0,
                    is_primary: false,
                    scaler: ScalingAlgorithm::Bilinear,
                },
            ],
        };
        let engine = SimulcastEngine::new(config);

        // Create a small test frame (8x4 RGBA = 128 bytes)
        let source = vec![128u8; 8 * 4 * 4];
        let outputs = engine.process_frame(&source, 8, 4);

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].leg_label, "Primary");
        assert_eq!(outputs[0].width, 8);
        assert_eq!(outputs[0].data.len(), 128);

        assert_eq!(outputs[1].leg_label, "Scaled");
        assert_eq!(outputs[1].width, 4);
        assert_eq!(outputs[1].height, 2);
        assert_eq!(outputs[1].data.len(), 4 * 2 * 4);
    }

    #[test]
    fn test_simulcast_nearest_neighbour() {
        let config = SimulcastConfig {
            legs: vec![
                SimulcastLeg {
                    label: "Src".to_string(),
                    width: 4,
                    height: 4,
                    fps: 25.0,
                    is_primary: true,
                    scaler: ScalingAlgorithm::default(),
                },
                SimulcastLeg {
                    label: "NN".to_string(),
                    width: 2,
                    height: 2,
                    fps: 25.0,
                    is_primary: false,
                    scaler: ScalingAlgorithm::NearestNeighbour,
                },
            ],
        };
        let engine = SimulcastEngine::new(config);
        let source = vec![200u8; 4 * 4 * 4];
        let outputs = engine.process_frame(&source, 4, 4);
        assert_eq!(outputs[1].data.len(), 2 * 2 * 4);
        // All source pixels are the same, so NN should produce the same value.
        assert!(outputs[1].data.iter().all(|&b| b == 200));
    }

    #[test]
    fn test_simulcast_same_resolution_passthrough() {
        let config = SimulcastConfig {
            legs: vec![SimulcastLeg {
                label: "Same".to_string(),
                width: 4,
                height: 4,
                fps: 25.0,
                is_primary: false,
                scaler: ScalingAlgorithm::Bilinear,
            }],
        };
        let engine = SimulcastEngine::new(config);
        let source = vec![50u8; 4 * 4 * 4];
        let outputs = engine.process_frame(&source, 4, 4);
        // Same resolution should passthrough.
        assert_eq!(outputs[0].data, source);
    }

    #[test]
    fn test_scaling_algorithm_default() {
        assert_eq!(ScalingAlgorithm::default(), ScalingAlgorithm::Lanczos);
    }

    #[test]
    fn test_simulcast_zero_size_no_panic() {
        let scaled = SimulcastEngine::scale(&[], 0, 0, 0, 0, ScalingAlgorithm::Bilinear);
        assert!(scaled.is_empty());
    }
}
