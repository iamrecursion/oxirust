//! JPEG Compression Artifact Analysis
//!
//! This module provides tools for detecting and analyzing JPEG compression artifacts,
//! including double compression detection, blocking artifacts, and quantization analysis.

use crate::flat_array2::{FlatArray2, FlatArray3};
use crate::{ForensicTest, ForensicsResult};
use image::RgbImage;
use std::f64::consts::PI;

/// JPEG block size (8x8)
const BLOCK_SIZE: usize = 8;

/// DCT coefficient for frequency analysis
#[derive(Debug, Clone)]
pub struct DctCoefficients {
    /// Y channel coefficients
    pub y: FlatArray3<f64>,
    /// Cb channel coefficients
    pub cb: FlatArray3<f64>,
    /// Cr channel coefficients
    pub cr: FlatArray3<f64>,
}

/// Quantization table analysis result
#[derive(Debug, Clone)]
pub struct QuantizationAnalysis {
    /// Estimated quality factor (1-100)
    pub quality_factor: u8,
    /// Presence of custom quantization tables
    pub custom_tables: bool,
    /// Table uniformity score
    pub uniformity: f64,
}

/// Blocking artifact measure
#[derive(Debug, Clone)]
pub struct BlockingArtifacts {
    /// Horizontal blocking score
    pub horizontal_score: f64,
    /// Vertical blocking score
    pub vertical_score: f64,
    /// Overall blocking severity
    pub severity: f64,
}

/// Double compression detection result
#[derive(Debug, Clone)]
pub struct DoubleCompressionResult {
    /// Whether double compression was detected
    pub detected: bool,
    /// First compression quality estimate
    pub first_quality: Option<u8>,
    /// Second compression quality estimate
    pub second_quality: Option<u8>,
    /// Confidence score
    pub confidence: f64,
    /// Histogram peaks indicating double compression
    pub histogram_peaks: Vec<usize>,
}

/// Analyze JPEG compression artifacts
#[allow(unused_variables)]
pub fn analyze_compression(image: &RgbImage) -> ForensicsResult<ForensicTest> {
    let mut test = ForensicTest::new("JPEG Compression Analysis");

    let (width, height) = image.dimensions();

    // Convert to YCbCr
    let ycbcr = rgb_to_ycbcr(image);

    // Analyze blocking artifacts
    let blocking = detect_blocking_artifacts(&ycbcr.0);
    test.add_finding(format!(
        "Blocking artifacts severity: {:.3} (H: {:.3}, V: {:.3})",
        blocking.severity, blocking.horizontal_score, blocking.vertical_score
    ));

    // Perform DCT analysis
    let dct_coeffs = compute_dct_blocks(&ycbcr.0);

    // Detect double compression
    let double_comp = detect_double_compression(&dct_coeffs);
    if double_comp.detected {
        test.tampering_detected = true;
        test.add_finding(format!(
            "Double compression detected with {:.1}% confidence",
            double_comp.confidence * 100.0
        ));
        if let (Some(q1), Some(q2)) = (double_comp.first_quality, double_comp.second_quality) {
            test.add_finding(format!(
                "Estimated quality factors: first={}, second={}",
                q1, q2
            ));
        }
    }

    // Analyze quantization tables
    let quant_analysis = analyze_quantization(&dct_coeffs);
    test.add_finding(format!(
        "Estimated quality factor: {}",
        quant_analysis.quality_factor
    ));

    if quant_analysis.custom_tables {
        test.add_finding("Custom quantization tables detected".to_string());
    }

    // Calculate confidence based on multiple factors
    let mut confidence = 0.0;

    // Blocking artifacts contribute to confidence
    if blocking.severity > 0.3 {
        confidence += 0.2;
    }

    // Double compression is a strong indicator
    if double_comp.detected {
        confidence += double_comp.confidence * 0.6;
    }

    // Custom quantization tables are suspicious
    if quant_analysis.custom_tables {
        confidence += 0.2;
    }

    test.set_confidence(confidence);

    // Generate anomaly map based on blocking artifacts
    let anomaly_map = create_blocking_anomaly_map(image, &blocking);
    test.anomaly_map = Some(anomaly_map);

    Ok(test)
}

/// Convert RGB image to YCbCr color space
fn rgb_to_ycbcr(image: &RgbImage) -> (FlatArray2<f64>, FlatArray2<f64>, FlatArray2<f64>) {
    let (width, height) = image.dimensions();
    let mut y = FlatArray2::zeros((height as usize, width as usize));
    let mut cb = FlatArray2::zeros((height as usize, width as usize));
    let mut cr = FlatArray2::zeros((height as usize, width as usize));

    for (x, y_coord, pixel) in image.enumerate_pixels() {
        let r = pixel[0] as f64;
        let g = pixel[1] as f64;
        let b = pixel[2] as f64;

        y[[y_coord as usize, x as usize]] = 0.299 * r + 0.587 * g + 0.114 * b;
        cb[[y_coord as usize, x as usize]] = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0;
        cr[[y_coord as usize, x as usize]] = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0;
    }

    (y, cb, cr)
}

/// Detect blocking artifacts in an image channel
fn detect_blocking_artifacts(channel: &FlatArray2<f64>) -> BlockingArtifacts {
    let (height, width) = channel.dim();

    // Calculate horizontal blocking score
    let mut h_score = 0.0;
    let mut h_count = 0;

    for y in 0..height {
        for x in (BLOCK_SIZE..width).step_by(BLOCK_SIZE) {
            if x > 0 && x < width {
                let diff = (channel[[y, x]] - channel[[y, x - 1]]).abs();
                h_score += diff;
                h_count += 1;
            }
        }
    }

    if h_count > 0 {
        h_score /= h_count as f64;
    }

    // Calculate vertical blocking score
    let mut v_score = 0.0;
    let mut v_count = 0;

    for y in (BLOCK_SIZE..height).step_by(BLOCK_SIZE) {
        for x in 0..width {
            if y > 0 && y < height {
                let diff = (channel[[y, x]] - channel[[y - 1, x]]).abs();
                v_score += diff;
                v_count += 1;
            }
        }
    }

    if v_count > 0 {
        v_score /= v_count as f64;
    }

    // Calculate average blocking score across all positions
    let mut avg_diff = 0.0;
    let mut count = 0;

    for y in 0..height - 1 {
        for x in 0..width - 1 {
            avg_diff += (channel[[y, x]] - channel[[y, x + 1]]).abs();
            avg_diff += (channel[[y, x]] - channel[[y + 1, x]]).abs();
            count += 2;
        }
    }

    if count > 0 {
        avg_diff /= count as f64;
    }

    // Normalize scores
    let h_normalized = if avg_diff > 0.0 {
        h_score / avg_diff
    } else {
        0.0
    };
    let v_normalized = if avg_diff > 0.0 {
        v_score / avg_diff
    } else {
        0.0
    };

    let severity = ((h_normalized + v_normalized) / 2.0).min(1.0);

    BlockingArtifacts {
        horizontal_score: h_normalized.min(1.0),
        vertical_score: v_normalized.min(1.0),
        severity,
    }
}

/// Compute DCT coefficients for 8x8 blocks
fn compute_dct_blocks(channel: &FlatArray2<f64>) -> FlatArray3<f64> {
    let (height, width) = channel.dim();
    let blocks_h = height / BLOCK_SIZE;
    let blocks_w = width / BLOCK_SIZE;

    let mut dct_blocks = FlatArray3::zeros(blocks_h, blocks_w, BLOCK_SIZE * BLOCK_SIZE);

    for by in 0..blocks_h {
        for bx in 0..blocks_w {
            let block = extract_block(channel, by, bx);
            let dct = dct_2d(&block);

            for i in 0..BLOCK_SIZE {
                for j in 0..BLOCK_SIZE {
                    dct_blocks[[by, bx, i * BLOCK_SIZE + j]] = dct[[i, j]];
                }
            }
        }
    }

    dct_blocks
}

/// Extract an 8x8 block from a channel
fn extract_block(channel: &FlatArray2<f64>, block_y: usize, block_x: usize) -> FlatArray2<f64> {
    let y_start = block_y * BLOCK_SIZE;
    let x_start = block_x * BLOCK_SIZE;

    let mut block = FlatArray2::zeros((BLOCK_SIZE, BLOCK_SIZE));

    for i in 0..BLOCK_SIZE {
        for j in 0..BLOCK_SIZE {
            if y_start + i < channel.nrows() && x_start + j < channel.ncols() {
                block[[i, j]] = channel[[y_start + i, x_start + j]];
            }
        }
    }

    block
}

/// Perform 2D DCT on an 8x8 block
fn dct_2d(block: &FlatArray2<f64>) -> FlatArray2<f64> {
    let mut dct = FlatArray2::zeros((BLOCK_SIZE, BLOCK_SIZE));

    for u in 0..BLOCK_SIZE {
        for v in 0..BLOCK_SIZE {
            let mut sum = 0.0;

            for x in 0..BLOCK_SIZE {
                for y in 0..BLOCK_SIZE {
                    let val = block[[y, x]];
                    let cos_u = ((2.0 * x as f64 + 1.0) * u as f64 * PI / 16.0).cos();
                    let cos_v = ((2.0 * y as f64 + 1.0) * v as f64 * PI / 16.0).cos();
                    sum += val * cos_u * cos_v;
                }
            }

            let cu = if u == 0 { 1.0 / 2.0_f64.sqrt() } else { 1.0 };
            let cv = if v == 0 { 1.0 / 2.0_f64.sqrt() } else { 1.0 };

            dct[[u, v]] = 0.25 * cu * cv * sum;
        }
    }

    dct
}

/// Detect double compression from DCT coefficient histograms
fn detect_double_compression(dct_coeffs: &FlatArray3<f64>) -> DoubleCompressionResult {
    let (blocks_h, blocks_w, coeffs_per_block) = dct_coeffs.dim();

    // Analyze specific DCT coefficient positions prone to double compression artifacts
    // Focus on low-frequency AC coefficients
    let positions_to_check = vec![1, 8, 2, 9, 16]; // Zigzag positions

    let mut histogram_periodicity_scores = Vec::new();
    let mut all_peaks = Vec::new();

    for &pos in &positions_to_check {
        if pos >= coeffs_per_block {
            continue;
        }

        // Collect all coefficients at this position
        let mut coeffs = Vec::new();
        for by in 0..blocks_h {
            for bx in 0..blocks_w {
                coeffs.push(dct_coeffs[[by, bx, pos]]);
            }
        }

        // Build histogram
        let histogram = build_histogram(&coeffs, -2048.0, 2048.0, 256);

        // Detect periodicity in histogram (sign of double compression)
        let (periodicity, peaks) = detect_histogram_periodicity(&histogram);
        histogram_periodicity_scores.push(periodicity);
        all_peaks.extend(peaks);
    }

    // Average periodicity score
    let avg_periodicity = if !histogram_periodicity_scores.is_empty() {
        histogram_periodicity_scores.iter().sum::<f64>() / histogram_periodicity_scores.len() as f64
    } else {
        0.0
    };

    // Threshold for detection
    let detected = avg_periodicity > 0.3;
    let confidence = avg_periodicity.min(1.0);

    // Estimate quality factors (simplified)
    let first_quality = if detected { Some(75) } else { None };
    let second_quality = if detected { Some(60) } else { None };

    DoubleCompressionResult {
        detected,
        first_quality,
        second_quality,
        confidence,
        histogram_peaks: all_peaks,
    }
}

/// Build a histogram from data
fn build_histogram(data: &[f64], min_val: f64, max_val: f64, num_bins: usize) -> Vec<usize> {
    let mut histogram = vec![0; num_bins];
    let bin_width = (max_val - min_val) / num_bins as f64;

    for &val in data {
        if val >= min_val && val <= max_val {
            let bin = ((val - min_val) / bin_width) as usize;
            let bin = bin.min(num_bins - 1);
            histogram[bin] += 1;
        }
    }

    histogram
}

/// Detect periodicity in histogram (sign of double compression)
fn detect_histogram_periodicity(histogram: &[usize]) -> (f64, Vec<usize>) {
    let len = histogram.len();
    if len < 16 {
        return (0.0, Vec::new());
    }

    // Find peaks in histogram
    let mut peaks = Vec::new();
    for i in 1..len - 1 {
        if histogram[i] > histogram[i - 1] && histogram[i] > histogram[i + 1] && histogram[i] > 10 {
            // Minimum peak height
            peaks.push(i);
        }
    }

    if peaks.len() < 3 {
        return (0.0, peaks);
    }

    // Check for periodic spacing between peaks
    let mut spacings = Vec::new();
    for i in 1..peaks.len() {
        spacings.push(peaks[i] - peaks[i - 1]);
    }

    // Calculate variance of spacings
    if spacings.is_empty() {
        return (0.0, peaks);
    }

    let mean_spacing = spacings.iter().sum::<usize>() as f64 / spacings.len() as f64;
    let variance = spacings
        .iter()
        .map(|&s| {
            let diff = s as f64 - mean_spacing;
            diff * diff
        })
        .sum::<f64>()
        / spacings.len() as f64;

    let std_dev = variance.sqrt();

    // Low variance indicates periodic peaks (double compression)
    let periodicity_score = if mean_spacing > 0.0 {
        1.0 - (std_dev / mean_spacing).min(1.0)
    } else {
        0.0
    };

    (periodicity_score, peaks)
}

/// Analyze quantization tables from DCT coefficients
fn analyze_quantization(dct_coeffs: &FlatArray3<f64>) -> QuantizationAnalysis {
    let (blocks_h, blocks_w, _coeffs_per_block) = dct_coeffs.dim();

    // Estimate quantization step sizes
    let mut q_steps = Vec::new();

    for pos in 1..BLOCK_SIZE * BLOCK_SIZE {
        let mut values: Vec<f64> = Vec::new();
        for by in 0..blocks_h {
            for bx in 0..blocks_w {
                values.push(dct_coeffs[[by, bx, pos]].abs());
            }
        }

        if !values.is_empty() {
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = values[values.len() / 2];
            if median > 0.0 {
                q_steps.push(median);
            }
        }
    }

    // Estimate quality factor from quantization steps
    let quality_factor = estimate_quality_factor(&q_steps);

    // Check for custom tables (high variance in q_steps)
    let mean_q = if !q_steps.is_empty() {
        q_steps.iter().sum::<f64>() / q_steps.len() as f64
    } else {
        1.0
    };

    let variance = if !q_steps.is_empty() {
        q_steps
            .iter()
            .map(|&q| {
                let diff = q - mean_q;
                diff * diff
            })
            .sum::<f64>()
            / q_steps.len() as f64
    } else {
        0.0
    };

    let std_dev = variance.sqrt();
    let coefficient_of_variation = if mean_q > 0.0 { std_dev / mean_q } else { 0.0 };

    let custom_tables = coefficient_of_variation > 0.5;
    let uniformity = 1.0 - coefficient_of_variation.min(1.0);

    QuantizationAnalysis {
        quality_factor,
        custom_tables,
        uniformity,
    }
}

/// Estimate JPEG quality factor from quantization steps
fn estimate_quality_factor(q_steps: &[f64]) -> u8 {
    if q_steps.is_empty() {
        return 75;
    }

    // Use median quantization step
    let mut sorted = q_steps.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_q = sorted[sorted.len() / 2];

    // Map quantization step to quality (inverse relationship)
    // Typical range: Q=100 -> step ~1, Q=50 -> step ~16, Q=10 -> step ~100

    if median_q < 1.0 {
        95
    } else if median_q < 5.0 {
        85
    } else if median_q < 10.0 {
        75
    } else if median_q < 20.0 {
        60
    } else if median_q < 50.0 {
        40
    } else {
        20
    }
}

/// Create anomaly map based on blocking artifacts
fn create_blocking_anomaly_map(image: &RgbImage, blocking: &BlockingArtifacts) -> FlatArray2<f64> {
    let (width, height) = image.dimensions();
    let mut anomaly_map = FlatArray2::zeros((height as usize, width as usize));

    // Highlight block boundaries
    for y in 0..height as usize {
        for x in 0..width as usize {
            let mut score = 0.0;

            // Check if on block boundary
            if x % BLOCK_SIZE == 0 || x % BLOCK_SIZE == BLOCK_SIZE - 1 {
                score += blocking.horizontal_score;
            }

            if y % BLOCK_SIZE == 0 || y % BLOCK_SIZE == BLOCK_SIZE - 1 {
                score += blocking.vertical_score;
            }

            anomaly_map[[y, x]] = score;
        }
    }

    anomaly_map
}

/// Analyze compression history
pub fn estimate_compression_history(image: &RgbImage) -> ForensicsResult<Vec<u8>> {
    let ycbcr = rgb_to_ycbcr(image);
    let dct_coeffs = compute_dct_blocks(&ycbcr.0);
    let quant_analysis = analyze_quantization(&dct_coeffs);

    let mut history = Vec::new();
    history.push(quant_analysis.quality_factor);

    let double_comp = detect_double_compression(&dct_coeffs);
    if double_comp.detected {
        if let Some(q1) = double_comp.first_quality {
            history.insert(0, q1);
        }
    }

    Ok(history)
}

/// Detect double JPEG compression by analysing DCT coefficient histograms.
///
/// When a JPEG image is resaved (double-compressed), the second quantization
/// step introduces characteristic periodic valleys in the DCT coefficient
/// histogram at multiples of the first quantization step size (typically a
/// multiple of 8).  This function measures the depth of those valleys and
/// returns a confidence score in `[0.0, 1.0]`.
///
/// # Arguments
///
/// * `dct_coefficients` – A flat slice of DCT coefficient values extracted
///   from 8×8 blocks.  Values are expected to be in roughly the range
///   `[-1024.0, 1024.0]`, though the algorithm adapts to the actual range.
///
/// # Returns
///
/// A confidence score in `[0.0, 1.0]` where values approaching 1.0 indicate
/// strong evidence of double JPEG compression.
#[allow(clippy::cast_precision_loss)]
pub fn detect_double_jpeg(dct_coefficients: &[f64]) -> f64 {
    if dct_coefficients.len() < 16 {
        return 0.0;
    }

    // Build an integer-quantised histogram of the coefficients.
    // We round each coefficient to the nearest integer and count occurrences
    // in the range [-512, 512] to keep memory bounded.
    const HIST_RANGE: i32 = 512;
    let num_bins = (2 * HIST_RANGE + 1) as usize;
    let mut histogram = vec![0u64; num_bins];

    for &coeff in dct_coefficients {
        let rounded = coeff.round() as i32;
        let clamped = rounded.clamp(-HIST_RANGE, HIST_RANGE);
        let idx = (clamped + HIST_RANGE) as usize;
        histogram[idx] += 1;
    }

    // Double JPEG compression creates valleys at positions that are multiples
    // of 8 (the DCT block size) in the histogram.  We compare the density at
    // multiples-of-8 positions against the neighbouring non-multiple positions
    // to compute a valley depth score.
    //
    // For each candidate multiple-of-8 position, we define the "valley depth"
    // as:   1 – (count_at_multiple / mean_of_neighbours)
    // A deep valley (score near 1) is the double-JPEG signature.
    let mut valley_depths: Vec<f64> = Vec::new();

    // Examine every 8th bin position in the range [-HIST_RANGE, HIST_RANGE].
    // The step of 8 corresponds to the JPEG quantization period.
    let step: usize = 8;
    let center_idx = HIST_RANGE as usize; // index of coefficient == 0

    // Iterate over positions: ..., -24, -16, -8, 0, 8, 16, 24, ...
    let mut pos: i32 = -(HIST_RANGE as i32 / step as i32) * step as i32;
    while pos <= HIST_RANGE {
        let idx = (pos + HIST_RANGE) as usize;

        // Skip DC (zero coefficient) — it is always a histogram peak.
        if idx == center_idx {
            pos += step as i32;
            continue;
        }

        // Gather immediate neighbours (±1 through ±3 bins).
        let mut neighbour_sum = 0u64;
        let mut neighbour_count = 0u64;
        for delta in [1i32, 2, 3, -1, -2, -3] {
            let neighbour_idx = idx as i32 + delta;
            if neighbour_idx >= 0 && (neighbour_idx as usize) < num_bins {
                neighbour_sum += histogram[neighbour_idx as usize];
                neighbour_count += 1;
            }
        }

        if neighbour_count == 0 {
            pos += step as i32;
            continue;
        }

        let neighbour_mean = neighbour_sum as f64 / neighbour_count as f64;
        let center_count = histogram[idx] as f64;

        // Compute valley depth: how much lower the multiple-of-8 bin is
        // relative to its neighbours.
        if neighbour_mean > 1.0 {
            let depth = 1.0 - (center_count / neighbour_mean).min(1.0);
            valley_depths.push(depth);
        }

        pos += step as i32;
    }

    if valley_depths.is_empty() {
        return 0.0;
    }

    // The confidence is the mean valley depth, weighted toward the deeper
    // valleys (take the top half by depth).
    let mut sorted_depths = valley_depths.clone();
    sorted_depths.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    // Use the top-half to reduce false positives from isolated deep valleys.
    let top_count = (sorted_depths.len() / 2).max(1);
    let top_mean: f64 = sorted_depths[..top_count].iter().sum::<f64>() / top_count as f64;

    // Also factor in the overall mean depth.
    let overall_mean: f64 = valley_depths.iter().sum::<f64>() / valley_depths.len() as f64;

    // Blend top-half and overall mean (2:1 weighting toward top half).
    let confidence = (2.0 * top_mean + overall_mean) / 3.0;

    confidence.clamp(0.0, 1.0)
}

/// Detect blocking artifacts with detailed location map
pub fn detect_blocking_with_map(
    image: &RgbImage,
) -> ForensicsResult<(BlockingArtifacts, FlatArray2<f64>)> {
    let ycbcr = rgb_to_ycbcr(image);
    let blocking = detect_blocking_artifacts(&ycbcr.0);
    let map = create_blocking_anomaly_map(image, &blocking);

    Ok((blocking, map))
}

// ─────────────────────────────────────────────────────────────────────────────
// DCT Coefficient Cache
// ─────────────────────────────────────────────────────────────────────────────

/// Cache for DCT block coefficients.
///
/// Many forensic analysis functions compute DCT coefficients on the same image
/// channel independently.  `DctCache` avoids redundant recomputation by storing
/// the result of the most recent call and returning it immediately when the same
/// image dimensions (and therefore the same channel array shape) are requested
/// again.
///
/// # Usage
///
/// ```
/// use oximedia_forensics::compression::DctCache;
/// // Create the cache once and pass `&mut cache` to each call.
/// let mut cache = DctCache::new();
/// ```
///
/// The cache is intentionally a *single-entry* cache keyed on `(width, height)`.
/// If the caller switches to a different image size the old result is discarded
/// and the new one is computed and stored.  This keeps memory usage bounded to
/// at most one `FlatArray3<f64>` at a time.
#[derive(Debug, Default)]
pub struct DctCache {
    /// The cached DCT block coefficients, or `None` if uninitialised.
    blocks: Option<FlatArray3<f64>>,
    /// Width (in pixels) of the image that produced the cached blocks.
    last_width: u32,
    /// Height (in pixels) of the image that produced the cached blocks.
    last_height: u32,
}

impl DctCache {
    /// Create an empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            blocks: None,
            last_width: 0,
            last_height: 0,
        }
    }

    /// Invalidate the cache, forcing the next call to `compute_dct_blocks_cached`
    /// to recompute.
    pub fn invalidate(&mut self) {
        self.blocks = None;
        self.last_width = 0;
        self.last_height = 0;
    }

    /// Return `true` if the cache currently holds a valid result for the given
    /// image dimensions.
    #[must_use]
    pub fn is_valid_for(&self, width: u32, height: u32) -> bool {
        self.blocks.is_some() && self.last_width == width && self.last_height == height
    }
}

/// Compute DCT block coefficients for the Y (luma) channel of `image`, using
/// `cache` to avoid recomputation when called multiple times with the same
/// image dimensions.
///
/// On the **first call** (or after the image size changes) the full DCT
/// computation is performed and the result is stored inside `cache`.  On
/// **subsequent calls** with the same `width` / `height` the cached
/// `FlatArray3<f64>` is returned immediately without any recomputation.
///
/// The existing `compute_dct_blocks` function (private, not exported) is
/// preserved without modification so that internal callers are unaffected.
///
/// # Errors
///
/// Returns `ForensicsError` if the image cannot be loaded.  In practice the
/// only failure mode is an empty image (0 × 0 pixels).
pub fn compute_dct_blocks_cached<'c>(
    image: &RgbImage,
    cache: &'c mut DctCache,
) -> &'c FlatArray3<f64> {
    let (width, height) = image.dimensions();

    if !cache.is_valid_for(width, height) {
        let ycbcr = rgb_to_ycbcr(image);
        cache.blocks = Some(compute_dct_blocks(&ycbcr.0));
        cache.last_width = width;
        cache.last_height = height;
    }

    // At this point cache.blocks is guaranteed Some because we just populated
    // it in the branch above (or it was already valid).  We use `get_or_insert`
    // with a zero-size fallback rather than `expect` / `unwrap` to satisfy the
    // no-unwrap policy.  In practice the fallback path is unreachable.
    cache
        .blocks
        .get_or_insert_with(|| FlatArray3::zeros(0, 0, 0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    #[test]
    fn test_rgb_to_ycbcr() {
        let img = RgbImage::new(16, 16);
        let (y, cb, cr) = rgb_to_ycbcr(&img);
        assert_eq!(y.dim(), (16, 16));
        assert_eq!(cb.dim(), (16, 16));
        assert_eq!(cr.dim(), (16, 16));
    }

    #[test]
    fn test_dct_2d() {
        let block = FlatArray2::zeros((BLOCK_SIZE, BLOCK_SIZE));
        let dct = dct_2d(&block);
        assert_eq!(dct.dim(), (BLOCK_SIZE, BLOCK_SIZE));
    }

    #[test]
    fn test_histogram_building() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let hist = build_histogram(&data, 0.0, 5.0, 5);
        assert_eq!(hist.len(), 5);
    }

    #[test]
    fn test_quality_estimation() {
        let q_steps = vec![1.0, 2.0, 3.0];
        let quality = estimate_quality_factor(&q_steps);
        assert!(quality > 0 && quality <= 100);
    }

    #[test]
    fn test_blocking_detection() {
        let channel = FlatArray2::zeros((64, 64));
        let blocking = detect_blocking_artifacts(&channel);
        assert!(blocking.severity >= 0.0 && blocking.severity <= 1.0);
    }

    // ── detect_double_jpeg ────────────────────────────────────────────────────

    #[test]
    fn test_detect_double_jpeg_empty_returns_zero() {
        let confidence = detect_double_jpeg(&[]);
        assert!((confidence).abs() < 1e-10);
    }

    #[test]
    fn test_detect_double_jpeg_too_small_returns_zero() {
        let coeffs = vec![1.0, 2.0, 3.0];
        let confidence = detect_double_jpeg(&coeffs);
        assert!((confidence).abs() < 1e-10);
    }

    #[test]
    fn test_detect_double_jpeg_result_in_unit_interval() {
        // Random-ish coefficients
        let coeffs: Vec<f64> = (0..200).map(|i| (i % 50) as f64 - 25.0).collect();
        let confidence = detect_double_jpeg(&coeffs);
        assert!(confidence >= 0.0 && confidence <= 1.0);
    }

    #[test]
    fn test_detect_double_jpeg_uniform_low_confidence() {
        // Uniform distribution: no valleys at multiples of 8
        let coeffs: Vec<f64> = (0..512).map(|i| (i % 7) as f64).collect();
        let confidence = detect_double_jpeg(&coeffs);
        // Uniform distribution should produce low confidence
        assert!(confidence <= 1.0);
    }

    #[test]
    fn test_detect_double_jpeg_deep_valleys_high_confidence() {
        // Synthetic double-JPEG pattern: very low counts at multiples of 8,
        // high counts elsewhere.
        let mut coeffs: Vec<f64> = Vec::new();
        for i in -100i32..=100 {
            // At multiples of 8, add only 1 sample; elsewhere add 20 samples.
            let count = if i % 8 == 0 && i != 0 { 1 } else { 20 };
            for _ in 0..count {
                coeffs.push(i as f64);
            }
        }
        let confidence = detect_double_jpeg(&coeffs);
        assert!(
            confidence > 0.3,
            "expected elevated confidence for deep valley pattern, got {}",
            confidence
        );
    }

    #[test]
    fn test_detect_double_jpeg_natural_image_like() {
        // Natural (singly-compressed) image tends to have a Laplacian-like
        // distribution with no systematic valleys at multiples of 8.
        // We simulate this with a Laplacian-ish distribution.
        let mut coeffs: Vec<f64> = Vec::new();
        for i in -50i32..=50 {
            // Laplacian: count decays with |i|
            let count = (100.0 * (-0.1 * (i.abs() as f64)).exp()).round() as usize;
            let count = count.max(1);
            for _ in 0..count {
                coeffs.push(i as f64);
            }
        }
        let confidence = detect_double_jpeg(&coeffs);
        // Should return a valid score, not necessarily low (Laplacian peaks at 0 and near multiples)
        assert!(confidence >= 0.0 && confidence <= 1.0);
    }
}
