//! Vorbis floor curves (spectral envelope estimation).
//!
//! Vorbis I specifies two floor types:
//!
//! - **Floor 0**: log-spectrum amplitude (LSP) model — used in older encoders.
//! - **Floor 1**: piece-wise linear interpolated floor — used in modern encoders.
//!
//! This module implements Floor 1 (the commonly used type).

#![forbid(unsafe_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]

/// Maximum number of Floor 1 partition classes.
const MAX_FLOOR1_CLASSES: usize = 16;
/// Maximum number of Floor 1 multiplier values.
const MAX_FLOOR1_VALUES: usize = 256;

/// Floor 1 partition class descriptor.
#[derive(Clone, Copy, Debug, Default)]
pub struct Floor1Class {
    /// Number of dimensions in this class.
    pub dimensions: u8,
    /// Sub-book bits (0 = no sub-books).
    pub sub_books: u8,
    /// Amplitude bits for this class (per dimension).
    pub master_book: u8,
}

/// Vorbis Floor 1 configuration.
#[derive(Clone, Debug)]
pub struct Floor1Config {
    /// Floor multiplier: 1, 2, 3, or 4.
    pub multiplier: u8,
    /// Number of floor partitions.
    pub partitions: u8,
    /// Partition class assignments.
    pub partition_class_list: Vec<u8>,
    /// Class descriptors.
    pub classes: Vec<Floor1Class>,
    /// X-position list (sorted ascending).
    pub x_list: Vec<u16>,
}

impl Default for Floor1Config {
    fn default() -> Self {
        // Minimal default: 2 points at x=0 and x=255
        Self {
            multiplier: 1,
            partitions: 0,
            partition_class_list: Vec::new(),
            classes: Vec::new(),
            x_list: vec![0, 255],
        }
    }
}

/// One decoded Floor 1 curve (list of amplitude values at x-positions).
#[derive(Clone, Debug)]
pub struct Floor1Curve {
    /// Amplitude values at each x position (in multiplier-scaled units).
    pub amplitudes: Vec<i16>,
    /// Associated x-positions.
    pub x_list: Vec<u16>,
    /// Whether the floor is "unused" (all-zero → zeroing the residue).
    pub unused: bool,
}

impl Floor1Curve {
    /// Interpolate the floor curve at an arbitrary x position using linear interpolation.
    #[must_use]
    pub fn interpolate_at(&self, x: f64, n: usize) -> f64 {
        if self.unused || self.x_list.is_empty() {
            return 0.0;
        }

        let xs = &self.x_list;
        let ys = &self.amplitudes;
        let m = self.x_list.len();
        let scale = n as f64; // normalise x into [0, n)

        // Find surrounding control points
        let mut lo_idx = 0;
        let mut hi_idx = m.saturating_sub(1);

        for i in 1..m {
            if f64::from(xs[i]) / scale <= x / scale {
                lo_idx = i;
            }
        }
        if lo_idx + 1 < m {
            hi_idx = lo_idx + 1;
        }

        let x0 = f64::from(xs[lo_idx]);
        let x1 = f64::from(xs[hi_idx]);
        let y0 = f64::from(ys[lo_idx]);
        let y1 = f64::from(ys[hi_idx]);

        if (x1 - x0).abs() < 1e-10 {
            return y0;
        }

        y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    }

    /// Convert amplitude values to linear gain (dB-scaled floor → linear magnitude).
    ///
    /// Vorbis uses `floor = exp( amplitude * log(10) / 28.0 )`.
    #[must_use]
    pub fn to_linear(&self, multiplier: u8) -> Vec<f64> {
        let scale = f64::from(multiplier) * std::f64::consts::LN_10 / 20.0;
        self.amplitudes
            .iter()
            .map(|&a| (f64::from(a) * scale).exp())
            .collect()
    }
}

/// Encode a power spectrum envelope as a Floor 1 curve.
///
/// Fits piece-wise linear segments to `log_spectrum` (log₁₀ magnitude at each bin).
/// Returns the quantised amplitude values at each x-position from `x_list`.
pub fn encode_floor1(log_spectrum: &[f64], x_list: &[u16], multiplier: u8) -> Vec<i16> {
    let m = x_list.len();
    let n = log_spectrum.len();
    let inv_scale = 20.0 / (f64::from(multiplier) * std::f64::consts::LN_10);

    (0..m)
        .map(|i| {
            let x = x_list[i] as usize;
            let x_clamped = x.min(n.saturating_sub(1));
            let db_val = log_spectrum[x_clamped] * inv_scale;
            // Clamp and round to [0, 255]
            db_val.round().clamp(0.0, 255.0) as i16
        })
        .collect()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_floor1_config_default() {
        let cfg = Floor1Config::default();
        assert_eq!(cfg.x_list, vec![0, 255]);
        assert_eq!(cfg.multiplier, 1);
    }

    #[test]
    fn test_floor1_curve_unused() {
        let curve = Floor1Curve {
            amplitudes: vec![0; 2],
            x_list: vec![0, 255],
            unused: true,
        };
        let v = curve.interpolate_at(128.0, 256);
        assert_eq!(v, 0.0);
    }

    #[test]
    fn test_floor1_curve_interpolate_midpoint() {
        let curve = Floor1Curve {
            amplitudes: vec![0, 100],
            x_list: vec![0, 256],
            unused: false,
        };
        let v = curve.interpolate_at(128.0, 256);
        // Midpoint between 0 and 100 = 50
        assert!((v - 50.0).abs() < 1.0, "Expected ~50, got {v}");
    }

    #[test]
    fn test_floor1_to_linear_amplitude_zero() {
        let curve = Floor1Curve {
            amplitudes: vec![0],
            x_list: vec![128],
            unused: false,
        };
        let lin = curve.to_linear(1);
        // exp(0) = 1.0
        assert!((lin[0] - 1.0).abs() < 1e-9, "exp(0) should be 1.0");
    }

    #[test]
    fn test_encode_floor1_flat_spectrum() {
        let n = 256;
        // Flat spectrum at 20 dB
        let log_spec = vec![1.0f64; n]; // log10(spectrum) ≈ 1 → 20 dB
        let x_list: Vec<u16> = vec![0, 128, 255];
        let amps = encode_floor1(&log_spec, &x_list, 1);
        assert_eq!(amps.len(), 3);
        // All amplitudes should be the same (flat)
        assert_eq!(amps[0], amps[1]);
        assert_eq!(amps[1], amps[2]);
    }

    #[test]
    fn test_encode_floor1_clamped_to_u8_range() {
        let log_spec = vec![100.0f64]; // very loud → should clamp to 255
        let amps = encode_floor1(&log_spec, &[0], 1);
        assert_eq!(amps[0], 255);
    }
}
