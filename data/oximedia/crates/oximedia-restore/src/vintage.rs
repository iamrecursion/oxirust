//! Vintage film restoration utilities.
//!
//! Provides scratch detection and repair, vignette correction, and grain removal
//! for archival film footage.

/// Type of film scratch artefact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScratchType {
    /// Vertical streak across the frame.
    Vertical,
    /// Horizontal streak across the frame.
    Horizontal,
    /// Diagonal streak.
    Diagonal,
    /// Irregular blob-shaped damage.
    Blob,
}

impl ScratchType {
    /// Returns `true` if this scratch type is linear (not a blob).
    #[must_use]
    pub fn is_linear(self) -> bool {
        matches!(
            self,
            ScratchType::Vertical | ScratchType::Horizontal | ScratchType::Diagonal
        )
    }
}

/// A film scratch or damage region.
#[derive(Debug, Clone)]
pub struct FilmScratch {
    /// Left edge of the scratch in pixels.
    pub x: u32,
    /// Top edge of the scratch in pixels.
    pub y: u32,
    /// Width of the scratch in pixels.
    pub width: u32,
    /// Height of the scratch in pixels.
    pub height: u32,
    /// Brightness intensity of the scratch (0.0 = black, 1.0 = white).
    pub intensity: f32,
    /// Scratch type classification.
    pub scratch_type: ScratchType,
}

impl FilmScratch {
    /// Total number of pixels covered by this scratch region.
    #[must_use]
    pub fn pixel_count(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }
}

/// Detect bright vertical streaks in a grayscale frame.
///
/// `frame` must be a flat `width * height` byte slice (8-bit luma).
/// Returns scratches whose column average intensity exceeds the frame mean by 20%.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn detect_scratches(frame: &[u8], width: usize, height: usize) -> Vec<FilmScratch> {
    if frame.is_empty() || width == 0 || height == 0 {
        return Vec::new();
    }
    if frame.len() < width * height {
        return Vec::new();
    }

    // Compute overall frame mean.
    let frame_mean: f32 = frame[..width * height]
        .iter()
        .map(|&b| f32::from(b))
        .sum::<f32>()
        / (width * height) as f32;

    let threshold = frame_mean * 1.2;
    let mut scratches = Vec::new();

    for col in 0..width {
        let col_sum: f32 = (0..height)
            .map(|row| f32::from(frame[row * width + col]))
            .sum();
        let col_mean = col_sum / height as f32;

        if col_mean > threshold {
            let intensity = (col_mean / 255.0).clamp(0.0, 1.0);
            scratches.push(FilmScratch {
                x: col as u32,
                y: 0,
                width: 1,
                height: height as u32,
                intensity,
                scratch_type: ScratchType::Vertical,
            });
        }
    }
    scratches
}

/// Repair a scratch by replacing it with the average of adjacent columns.
///
/// `frame` must be a flat `width * height` byte slice (8-bit luma).
#[allow(clippy::cast_possible_truncation)]
pub fn repair_scratch(frame: &mut [u8], scratch: &FilmScratch, width: usize) {
    if width == 0 || frame.is_empty() {
        return;
    }
    let height = frame.len() / width;

    for col in scratch.x..scratch.x + scratch.width {
        let col = col as usize;
        if col >= width {
            continue;
        }
        // Get adjacent column values.
        let left_col = if col > 0 { col - 1 } else { col + 1 };
        let right_col = if col + 1 < width {
            col + 1
        } else if col > 0 {
            col - 1
        } else {
            col
        };

        for row in scratch.y as usize..(scratch.y + scratch.height) as usize {
            if row >= height {
                break;
            }
            let left_val = f32::from(frame[row * width + left_col]);
            let right_val = f32::from(frame[row * width + right_col]);
            frame[row * width + col] = ((left_val + right_val) / 2.0) as u8;
        }
    }
}

/// Vignette correction model based on corner samples.
#[derive(Debug, Clone)]
pub struct VignetteRemover {
    /// Gain factors at the four image corners: `[top-left, top-right, bottom-left, bottom-right]`.
    pub corners: [f32; 4],
}

impl VignetteRemover {
    /// Compute the correction gain at pixel `(x, y)` for an image of size `(w, h)`.
    ///
    /// Uses bilinear interpolation of the corner gains.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn correction_factor(&self, x: u32, y: u32, w: u32, h: u32) -> f32 {
        if w == 0 || h == 0 {
            return 1.0;
        }
        let tx = x as f32 / (w - 1).max(1) as f32;
        let ty = y as f32 / (h - 1).max(1) as f32;
        let top = self.corners[0] * (1.0 - tx) + self.corners[1] * tx;
        let bot = self.corners[2] * (1.0 - tx) + self.corners[3] * tx;
        top * (1.0 - ty) + bot * ty
    }
}

/// Film grain remover via outlier replacement.
#[derive(Debug, Clone)]
pub struct GrainRemover {
    /// Intensity threshold: pixels deviating more than this from the local average are replaced.
    pub threshold: f32,
}

impl GrainRemover {
    /// Create a new `GrainRemover` with the given threshold.
    #[must_use]
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }

    /// Remove grain from an 8-bit luma frame in-place.
    ///
    /// Pixels that deviate more than `threshold` (in 0–255 range) from the
    /// 4-neighbour average are replaced by that average.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    pub fn remove(&self, pixels: &mut [u8], width: usize, height: usize) {
        if pixels.is_empty() || width == 0 || height == 0 {
            return;
        }
        let original = pixels.to_vec();
        for row in 0..height {
            for col in 0..width {
                let idx = row * width + col;
                let val = f32::from(original[idx]);
                let mut sum = 0.0_f32;
                let mut count = 0u32;
                if col > 0 {
                    sum += f32::from(original[idx - 1]);
                    count += 1;
                }
                if col + 1 < width {
                    sum += f32::from(original[idx + 1]);
                    count += 1;
                }
                if row > 0 {
                    sum += f32::from(original[idx - width]);
                    count += 1;
                }
                if row + 1 < height {
                    sum += f32::from(original[idx + width]);
                    count += 1;
                }
                if count == 0 {
                    continue;
                }
                let avg = sum / count as f32;
                if (val - avg).abs() > self.threshold {
                    pixels[idx] = avg as u8;
                }
            }
        }
    }
}

/// Vintage noise characteristics (for audio simulations).
#[derive(Debug, Clone, PartialEq)]
pub struct VintageNoiseProfile {
    /// RMS level of continuous hiss (0.0 = none, 1.0 = full-scale).
    pub hiss_level: f64,
    /// RMS level of mains hum (0.0 = none, 1.0 = full-scale).
    pub hum_level: f64,
    /// Wow modulation frequency in Hz.
    pub wow_hz: f64,
    /// Flutter modulation frequency in Hz.
    pub flutter_hz: f64,
    /// Probability per sample of a vinyl crackle event.
    pub crackle_rate: f64,
}

impl VintageNoiseProfile {
    /// Profile for vinyl records.
    #[must_use]
    pub fn vinyl() -> Self {
        Self {
            hiss_level: 0.04,
            hum_level: 0.005,
            wow_hz: 0.5,
            flutter_hz: 6.0,
            crackle_rate: 0.002,
        }
    }

    /// Profile for analogue tape machines.
    #[must_use]
    pub fn tape() -> Self {
        Self {
            hiss_level: 0.025,
            hum_level: 0.010,
            wow_hz: 1.2,
            flutter_hz: 12.0,
            crackle_rate: 0.0002,
        }
    }

    /// Profile for AM radio broadcasts.
    #[must_use]
    pub fn radio() -> Self {
        Self {
            hiss_level: 0.08,
            hum_level: 0.020,
            wow_hz: 0.0,
            flutter_hz: 0.0,
            crackle_rate: 0.0,
        }
    }

    /// A clean baseline profile with no noise artefacts.
    #[must_use]
    pub fn clean() -> Self {
        Self {
            hiss_level: 0.0,
            hum_level: 0.0,
            wow_hz: 0.0,
            flutter_hz: 0.0,
            crackle_rate: 0.0,
        }
    }
}

/// Restorer that analyses and reduces vintage noise.
#[derive(Debug, Clone)]
pub struct VintageRestorer {
    /// Noise profile used for restoration.
    pub profile: VintageNoiseProfile,
    /// Sample rate of the audio being processed, in Hz.
    pub sample_rate: f64,
}

impl VintageRestorer {
    /// Create a new restorer.
    #[must_use]
    pub fn new(sample_rate: f64, profile: VintageNoiseProfile) -> Self {
        Self {
            profile,
            sample_rate,
        }
    }

    /// Estimate a `VintageNoiseProfile` from a sample buffer.
    #[must_use]
    pub fn analyze_noise(samples: &[f64]) -> VintageNoiseProfile {
        let hiss_level = Self::estimate_hiss_level(samples);
        let hum_level = Self::estimate_hum_level(samples, 44100.0);
        VintageNoiseProfile {
            hiss_level,
            hum_level,
            wow_hz: 0.5,
            flutter_hz: 6.0,
            crackle_rate: 0.001,
        }
    }

    /// Estimate hiss as RMS of first-order differences.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn estimate_hiss_level(samples: &[f64]) -> f64 {
        if samples.len() < 2 {
            return 0.0;
        }
        let sum_sq: f64 = samples
            .windows(2)
            .map(|w| {
                let d = w[1] - w[0];
                d * d
            })
            .sum();
        (sum_sq / (samples.len() - 1) as f64).sqrt()
    }

    /// Estimate hum via single-pole low-pass filter.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn estimate_hum_level(samples: &[f64], sample_rate: f64) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }
        let fc = 80.0_f64;
        let alpha = (2.0 * std::f64::consts::PI * fc / sample_rate).min(1.0);
        let mut y = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        for &x in samples {
            y += alpha * (x - y);
            sum_sq += y * y;
        }
        (sum_sq / samples.len() as f64).sqrt()
    }

    /// Compute the restoration gain (0.0–1.0) to reduce noise.
    #[must_use]
    pub fn restoration_gain(&self) -> f64 {
        let noise_power = self.profile.hiss_level.powi(2) + self.profile.hum_level.powi(2);
        (1.0 - (noise_power * 100.0).min(1.0)).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scratch_type_vertical_is_linear() {
        assert!(ScratchType::Vertical.is_linear());
    }

    #[test]
    fn test_scratch_type_horizontal_is_linear() {
        assert!(ScratchType::Horizontal.is_linear());
    }

    #[test]
    fn test_scratch_type_diagonal_is_linear() {
        assert!(ScratchType::Diagonal.is_linear());
    }

    #[test]
    fn test_scratch_type_blob_is_not_linear() {
        assert!(!ScratchType::Blob.is_linear());
    }

    #[test]
    fn test_film_scratch_pixel_count() {
        let s = FilmScratch {
            x: 0,
            y: 0,
            width: 3,
            height: 5,
            intensity: 1.0,
            scratch_type: ScratchType::Vertical,
        };
        assert_eq!(s.pixel_count(), 15);
    }

    #[test]
    fn test_detect_scratches_empty_frame() {
        let scratches = detect_scratches(&[], 0, 0);
        assert!(scratches.is_empty());
    }

    #[test]
    fn test_detect_scratches_flat_frame_no_scratches() {
        // Uniform frame → no column stands out
        let frame = vec![100u8; 10 * 10];
        let scratches = detect_scratches(&frame, 10, 10);
        assert!(scratches.is_empty());
    }

    #[test]
    fn test_detect_scratches_bright_column() {
        let mut frame = vec![100u8; 10 * 10];
        // Make column 5 very bright
        for row in 0..10usize {
            frame[row * 10 + 5] = 255;
        }
        let scratches = detect_scratches(&frame, 10, 10);
        assert!(!scratches.is_empty());
        assert!(scratches.iter().any(|s| s.x == 5));
    }

    #[test]
    fn test_repair_scratch_replaces_pixels() {
        let mut frame = vec![0u8; 10 * 10];
        // Make column 5 white (255)
        for row in 0..10usize {
            frame[row * 10 + 5] = 255;
        }
        // Set neighbours to 128
        for row in 0..10usize {
            frame[row * 10 + 4] = 128;
            frame[row * 10 + 6] = 128;
        }
        let scratch = FilmScratch {
            x: 5,
            y: 0,
            width: 1,
            height: 10,
            intensity: 1.0,
            scratch_type: ScratchType::Vertical,
        };
        repair_scratch(&mut frame, &scratch, 10);
        // Column 5 should now be ~128
        for row in 0..10usize {
            assert_eq!(frame[row * 10 + 5], 128);
        }
    }

    #[test]
    fn test_vignette_remover_center_gain() {
        let vr = VignetteRemover {
            corners: [1.0, 1.0, 1.0, 1.0],
        };
        // Center pixel (5,5) in a 10×10 image
        let g = vr.correction_factor(5, 5, 10, 10);
        assert!((g - 1.0).abs() < 1e-5, "g={g}");
    }

    #[test]
    fn test_vignette_remover_corner_gain() {
        let vr = VignetteRemover {
            corners: [2.0, 1.0, 1.0, 1.0],
        };
        let g = vr.correction_factor(0, 0, 10, 10);
        assert!((g - 2.0).abs() < 1e-5, "g={g}");
    }

    #[test]
    fn test_grain_remover_no_change_when_uniform() {
        let mut pixels = vec![128u8; 5 * 5];
        let remover = GrainRemover::new(10.0);
        remover.remove(&mut pixels, 5, 5);
        // All pixels stay at 128
        assert!(pixels.iter().all(|&p| p == 128));
    }

    #[test]
    fn test_grain_remover_replaces_outlier() {
        let mut pixels = vec![128u8; 5 * 5];
        // Set centre pixel (2,2) to 255 → outlier
        pixels[2 * 5 + 2] = 255;
        let remover = GrainRemover::new(50.0);
        remover.remove(&mut pixels, 5, 5);
        // Centre should now be close to 128
        assert!(pixels[2 * 5 + 2] < 200, "center={}", pixels[2 * 5 + 2]);
    }
}
