//! Film grain pattern analysis.
//!
//! Analyzes video frames to identify and characterize film grain patterns,
//! distinguishing them from digital noise.

use crate::{DenoiseError, DenoiseResult};
use oximedia_codec::VideoFrame;

/// Grain characteristics map.
#[derive(Clone, Debug)]
pub struct GrainMap {
    /// Width of the grain map.
    pub width: usize,
    /// Height of the grain map.
    pub height: usize,
    /// Grain strength at each pixel (0.0 = no grain, 1.0 = strong grain).
    pub strength: Vec<f32>,
    /// Grain size characteristic.
    pub average_grain_size: f32,
    /// Grain distribution pattern.
    pub pattern_type: GrainPattern,
}

/// Type of grain pattern detected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrainPattern {
    /// No significant grain detected.
    None,
    /// Fine grain (typical of high-speed film).
    Fine,
    /// Medium grain (typical of standard film).
    Medium,
    /// Coarse grain (typical of low-speed or pushed film).
    Coarse,
    /// Digital noise (not film grain).
    DigitalNoise,
}

/// Frequency-domain grain characterization result.
///
/// Describes how grain energy is distributed across spatial frequency bands
/// (low, mid, high), enabling frequency-aware grain-preserving denoising.
#[derive(Clone, Debug)]
pub struct FrequencyGrainProfile {
    /// Fraction of grain energy in low-frequency band (0.0–1.0).
    pub low_freq_energy: f32,
    /// Fraction of grain energy in mid-frequency band (0.0–1.0).
    pub mid_freq_energy: f32,
    /// Fraction of grain energy in high-frequency band (0.0–1.0).
    pub high_freq_energy: f32,
    /// Dominant spatial frequency in cycles/pixel (0.0–0.5).
    pub dominant_frequency: f32,
    /// Spectral tilt: ratio of high-freq to low-freq energy (> 1 = more texture).
    pub spectral_tilt: f32,
}

impl FrequencyGrainProfile {
    /// Classify whether this profile resembles film grain (spectrally flat/white)
    /// or structured texture (coloured noise).
    ///
    /// Film grain has roughly equal energy across bands (spectral_tilt near 1).
    /// Digital noise has more concentrated high-frequency energy.
    #[must_use]
    pub fn resembles_film_grain(&self) -> bool {
        // Film grain is spectrally flat; structured digital noise peaks sharply
        self.spectral_tilt < 3.0 && self.high_freq_energy < 0.8
    }
}

/// Analyze film grain in a video frame.
///
/// Detects and characterizes grain patterns to enable grain-preserving
/// denoising.
///
/// # Arguments
/// * `frame` - Input video frame
///
/// # Returns
/// Grain characteristics map
pub fn analyze_grain(frame: &VideoFrame) -> DenoiseResult<GrainMap> {
    if frame.planes.is_empty() {
        return Err(DenoiseError::ProcessingError(
            "Frame has no planes".to_string(),
        ));
    }

    let plane = &frame.planes[0]; // Use luma plane
    let (width, height) = frame.plane_dimensions(0);

    // Compute high-frequency component (potential grain)
    let high_freq = extract_high_frequency(
        plane.data.as_ref(),
        width as usize,
        height as usize,
        plane.stride,
    );

    // Analyze grain characteristics
    let average_grain_size = estimate_grain_size(&high_freq, width as usize, height as usize);
    let pattern_type = classify_grain_pattern(&high_freq, average_grain_size);

    // Create grain strength map
    let strength =
        compute_grain_strength_map(&high_freq, width as usize, height as usize, pattern_type);

    Ok(GrainMap {
        width: width as usize,
        height: height as usize,
        strength,
        average_grain_size,
        pattern_type,
    })
}

/// Characterize grain energy distribution across spatial frequency bands.
///
/// Uses a multi-band approach: divides the Laplacian response into frequency
/// sub-bands via progressively blurred residuals, then measures relative
/// energy in each band.
///
/// # Arguments
/// * `frame` - Input video frame (luma plane is used)
///
/// # Returns
/// [`FrequencyGrainProfile`] describing energy distribution across bands.
pub fn characterize_grain_frequency(frame: &VideoFrame) -> DenoiseResult<FrequencyGrainProfile> {
    if frame.planes.is_empty() {
        return Err(DenoiseError::ProcessingError(
            "Frame has no planes".to_string(),
        ));
    }

    let plane = &frame.planes[0];
    let (width, height) = frame.plane_dimensions(0);
    let w = width as usize;
    let h = height as usize;

    // Extract the pixel data into a flat row-major f32 buffer (no stride padding)
    let pixels: Vec<f32> = (0..h)
        .flat_map(|y| (0..w).map(move |x| f32::from(plane.data[y * plane.stride + x])))
        .collect();

    // Compute three frequency bands using box-blur approximations
    // Band 0 (high): pixels – box3(pixels)
    // Band 1 (mid):  box3(pixels) – box9(pixels)
    // Band 2 (low):  box9(pixels)
    let blurred3 = box_blur(&pixels, w, h, 3);
    let blurred9 = box_blur(&pixels, w, h, 9);

    let high_band: Vec<f32> = pixels
        .iter()
        .zip(blurred3.iter())
        .map(|(&p, &b)| (p - b).abs())
        .collect();
    let mid_band: Vec<f32> = blurred3
        .iter()
        .zip(blurred9.iter())
        .map(|(&b3, &b9)| (b3 - b9).abs())
        .collect();
    let low_band: Vec<f32> = blurred9.iter().map(|&v| v.abs()).collect();

    let energy_high: f32 = high_band.iter().map(|&v| v * v).sum::<f32>() / (w * h) as f32;
    let energy_mid: f32 = mid_band.iter().map(|&v| v * v).sum::<f32>() / (w * h) as f32;
    let energy_low: f32 = low_band.iter().map(|&v| v * v).sum::<f32>() / (w * h) as f32;

    let raw_total = energy_high + energy_mid + energy_low;

    // When the frame is perfectly flat (all bands carry zero energy), return a
    // uniform distribution so that the normalised fractions always sum to 1.0.
    if raw_total < 1e-12 {
        return Ok(FrequencyGrainProfile {
            low_freq_energy: 1.0 / 3.0,
            mid_freq_energy: 1.0 / 3.0,
            high_freq_energy: 1.0 / 3.0,
            dominant_frequency: (0.4 + 0.2 + 0.05) / 3.0,
            spectral_tilt: 1.0,
        });
    }

    let total_energy = raw_total;

    // Dominant frequency: weighted average of band centres (0.0=DC, 0.5=Nyquist)
    // High band ~ 0.4, Mid ~ 0.2, Low ~ 0.05 cycles/pixel
    let dominant_frequency =
        (0.4 * energy_high + 0.2 * energy_mid + 0.05 * energy_low) / total_energy;

    let spectral_tilt = (energy_high + 1e-9) / (energy_low + 1e-9);

    Ok(FrequencyGrainProfile {
        low_freq_energy: energy_low / total_energy,
        mid_freq_energy: energy_mid / total_energy,
        high_freq_energy: energy_high / total_energy,
        dominant_frequency,
        spectral_tilt,
    })
}

/// Fast separable box blur using sliding-window sums.
///
/// Returns a flat row-major `f32` buffer of size `width × height`.
fn box_blur(pixels: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    if radius == 0 || width == 0 || height == 0 {
        return pixels.to_vec();
    }

    let r = radius.min(width / 2).min(height / 2);

    // Horizontal pass
    let mut horiz = vec![0.0f32; width * height];
    for y in 0..height {
        let row_start = y * width;
        let mut sum: f32 = 0.0;
        let mut count: f32 = 0.0;
        // Initialise window centred at x=0
        for dx in 0..=r.min(width - 1) {
            sum += pixels[row_start + dx];
            count += 1.0;
        }
        for x in 0..width {
            horiz[row_start + x] = sum / count;
            // Advance window
            let add_x = x + r + 1;
            let rem_x = if x >= r { x - r } else { usize::MAX };
            if add_x < width {
                sum += pixels[row_start + add_x];
                count += 1.0;
            }
            if rem_x != usize::MAX {
                sum -= pixels[row_start + rem_x];
                count -= 1.0;
            }
        }
    }

    // Vertical pass on horiz result
    let mut vert = vec![0.0f32; width * height];
    for x in 0..width {
        let mut sum: f32 = 0.0;
        let mut count: f32 = 0.0;
        for dy in 0..=r.min(height - 1) {
            sum += horiz[dy * width + x];
            count += 1.0;
        }
        for y in 0..height {
            vert[y * width + x] = sum / count;
            let add_y = y + r + 1;
            let rem_y = if y >= r { y - r } else { usize::MAX };
            if add_y < height {
                sum += horiz[add_y * width + x];
                count += 1.0;
            }
            if rem_y != usize::MAX {
                sum -= horiz[rem_y * width + x];
                count -= 1.0;
            }
        }
    }

    vert
}

/// Extract high-frequency component using high-pass filter.
fn extract_high_frequency(data: &[u8], width: usize, height: usize, stride: usize) -> Vec<f32> {
    let mut high_freq = vec![0.0f32; width * height];

    // Apply Laplacian high-pass filter
    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let idx = y * stride + x;
            let center = f32::from(data[idx]);

            let laplacian = 4.0 * center
                - f32::from(data[idx - 1])
                - f32::from(data[idx + 1])
                - f32::from(data[idx - stride])
                - f32::from(data[idx + stride]);

            high_freq[y * width + x] = laplacian.abs();
        }
    }

    high_freq
}

/// Estimate average grain size from high-frequency component.
fn estimate_grain_size(high_freq: &[f32], width: usize, height: usize) -> f32 {
    // Compute autocorrelation to estimate grain size
    let mut autocorr_sum = 0.0f32;
    let mut count = 0;
    let max_lag = 5;

    for y in max_lag..(height - max_lag) {
        for x in max_lag..(width - max_lag) {
            let center = high_freq[y * width + x];

            for lag in 1..=max_lag {
                let neighbor = high_freq[y * width + (x + lag)];
                autocorr_sum += center * neighbor;
                count += 1;
            }
        }
    }

    let avg_autocorr = if count > 0 {
        autocorr_sum / count as f32
    } else {
        0.0
    };

    // Grain size inversely related to autocorrelation
    (1.0 / (avg_autocorr + 0.1)).clamp(1.0, 10.0)
}

/// Classify grain pattern type.
fn classify_grain_pattern(high_freq: &[f32], grain_size: f32) -> GrainPattern {
    // Compute statistics
    let sum: f32 = high_freq.iter().sum();
    let avg = sum / high_freq.len() as f32;

    let variance: f32 = high_freq
        .iter()
        .map(|&x| {
            let diff = x - avg;
            diff * diff
        })
        .sum::<f32>()
        / high_freq.len() as f32;

    let std_dev = variance.sqrt();

    // Classify based on grain size and variance
    if avg < 2.0 && std_dev < 3.0 {
        GrainPattern::None
    } else if grain_size < 2.0 {
        GrainPattern::Fine
    } else if grain_size < 5.0 {
        GrainPattern::Medium
    } else if std_dev > 10.0 {
        GrainPattern::DigitalNoise
    } else {
        GrainPattern::Coarse
    }
}

/// Compute grain strength map.
fn compute_grain_strength_map(
    high_freq: &[f32],
    width: usize,
    height: usize,
    pattern_type: GrainPattern,
) -> Vec<f32> {
    let mut strength_map = vec![0.0f32; width * height];

    // Normalize high-frequency component to strength values
    let max_hf = high_freq.iter().copied().fold(0.0f32, f32::max);

    if max_hf > 0.0 {
        for i in 0..strength_map.len() {
            let normalized = high_freq[i] / max_hf;

            // Adjust based on pattern type
            strength_map[i] = match pattern_type {
                GrainPattern::None => 0.0,
                GrainPattern::Fine => normalized * 0.3,
                GrainPattern::Medium => normalized * 0.5,
                GrainPattern::Coarse => normalized * 0.7,
                GrainPattern::DigitalNoise => 0.0, // Don't preserve digital noise
            };
        }
    }

    strength_map
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::PixelFormat;

    #[test]
    fn test_analyze_grain() {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        frame.allocate();

        let result = analyze_grain(&frame);
        assert!(result.is_ok());

        let grain_map = result.expect("grain_map should be valid");
        assert_eq!(grain_map.width, 64);
        assert_eq!(grain_map.height, 64);
        assert_eq!(grain_map.strength.len(), 64 * 64);
    }

    #[test]
    fn test_grain_pattern_classification() {
        let high_freq = vec![0.5f32; 100];
        let pattern = classify_grain_pattern(&high_freq, 3.0);
        assert!(matches!(
            pattern,
            GrainPattern::None
                | GrainPattern::Fine
                | GrainPattern::Medium
                | GrainPattern::Coarse
                | GrainPattern::DigitalNoise
        ));
    }

    #[test]
    fn test_grain_size_estimation() {
        let high_freq = vec![1.0f32; 64 * 64];
        let size = estimate_grain_size(&high_freq, 64, 64);
        assert!(size > 0.0);
    }

    #[test]
    fn test_high_frequency_extraction() {
        let data = vec![128u8; 64 * 64];
        let high_freq = extract_high_frequency(&data, 64, 64, 64);
        assert_eq!(high_freq.len(), 64 * 64);
    }

    // -------------------------------------------------------------------
    // Frequency-domain grain characterization tests
    // -------------------------------------------------------------------

    #[test]
    fn test_characterize_grain_frequency_flat_frame() {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        frame.allocate();
        // All zeros → energy in all bands should be near zero / dominated by low
        let result = characterize_grain_frequency(&frame);
        assert!(result.is_ok());
        let profile = result.expect("profile should be valid");
        // energies should sum to ~1.0
        let total = profile.low_freq_energy + profile.mid_freq_energy + profile.high_freq_energy;
        assert!(
            (total - 1.0).abs() < 0.01,
            "band energies should sum to 1, got {total}"
        );
    }

    #[test]
    fn test_characterize_grain_frequency_noisy_frame() {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        frame.allocate();
        // Inject checkerboard noise (maximum high-frequency content)
        {
            let stride = frame.planes[0].stride;
            for y in 0..64usize {
                for x in 0..64usize {
                    frame.planes[0].data[y * stride + x] = if (x + y) % 2 == 0 { 0 } else { 255 };
                }
            }
        }
        let result = characterize_grain_frequency(&frame);
        assert!(result.is_ok());
        let profile = result.expect("profile should be valid");
        // Checkerboard pattern is entirely high-frequency
        assert!(
            profile.high_freq_energy > 0.3,
            "checkerboard should have significant high-freq energy: {}",
            profile.high_freq_energy
        );
    }

    #[test]
    fn test_characterize_grain_frequency_dominant_freq_range() {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        frame.allocate();
        let result = characterize_grain_frequency(&frame).expect("should succeed");
        assert!(
            (0.0..=0.5).contains(&result.dominant_frequency),
            "dominant_frequency should be in [0, 0.5]: {}",
            result.dominant_frequency
        );
    }

    #[test]
    fn test_characterize_grain_frequency_spectral_tilt_positive() {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        frame.allocate();
        let result = characterize_grain_frequency(&frame).expect("should succeed");
        assert!(
            result.spectral_tilt >= 0.0,
            "spectral_tilt must be non-negative"
        );
    }

    #[test]
    fn test_resembles_film_grain_flat_content() {
        // Flat frame has near-zero energy; spectrally flat → resembles film grain
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 32, 32);
        frame.allocate();
        let profile = characterize_grain_frequency(&frame).expect("should succeed");
        // We can't assert a specific value for trivially flat content, but method should not panic
        let _ = profile.resembles_film_grain();
    }

    #[test]
    fn test_box_blur_uniform() {
        // Blurring a uniform image should return the same values
        let pixels = vec![100.0f32; 8 * 8];
        let blurred = box_blur(&pixels, 8, 8, 3);
        for &v in &blurred {
            assert!(
                (v - 100.0).abs() < 1e-3,
                "uniform blur should be unchanged: {v}"
            );
        }
    }

    #[test]
    fn test_box_blur_zero_radius() {
        let pixels: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let result = box_blur(&pixels, 4, 4, 0);
        assert_eq!(result, pixels);
    }

    #[test]
    fn test_characterize_grain_empty_planes() {
        // Not allocated → empty planes → error
        let frame = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        let result = characterize_grain_frequency(&frame);
        // Empty planes means planes vec is empty → error
        assert!(result.is_err());
    }
}
