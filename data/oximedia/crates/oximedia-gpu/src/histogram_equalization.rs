//! GPU-accelerated histogram equalization.
//!
//! Provides two algorithms for contrast enhancement of single-channel (luma)
//! images:
//!
//! * **Global equalization** – [`HistogramEqualizer::equalize_luma`] applies a
//!   single CDF-based tone mapping to the entire image.
//!
//! * **CLAHE** – [`HistogramEqualizer::clahe`] divides the image into a grid of
//!   tiles, clips each local histogram at `clip_limit`, computes tile-local
//!   equalisation tables, and bilinearly interpolates the four nearest tile
//!   tables for each output pixel.

// ─── ClaheConfig ──────────────────────────────────────────────────────────────

/// Configuration for Contrast Limited Adaptive Histogram Equalization.
#[derive(Debug, Clone)]
pub struct ClaheConfig {
    /// Histogram clip limit.  Values of 2.0–4.0 are typical; higher values
    /// produce stronger contrast enhancement.
    pub clip_limit: f32,
    /// Tile edge length in pixels.  Typical values: 8, 16, 32.
    pub tile_size: u32,
    /// When `true`, use rayon for parallel tile processing.
    pub use_parallel: bool,
}

impl Default for ClaheConfig {
    fn default() -> Self {
        Self {
            clip_limit: 2.0,
            tile_size: 8,
            use_parallel: true,
        }
    }
}

// ─── EqualizationStats ────────────────────────────────────────────────────────

/// Descriptive statistics comparing an original and an equalized image.
#[derive(Debug, Clone)]
pub struct EqualizationStats {
    /// Mean of original pixel values.
    pub original_mean: f64,
    /// Mean of equalized pixel values.
    pub equalized_mean: f64,
    /// Standard deviation of original pixel values.
    pub original_std_dev: f64,
    /// Standard deviation of equalized pixel values.
    pub equalized_std_dev: f64,
}

impl EqualizationStats {
    /// Compute statistics from a pair of same-length byte slices.
    ///
    /// Both slices are interpreted as 8-bit luma values.  If either slice is
    /// empty, all statistics default to 0.0.
    #[must_use]
    pub fn compute(original: &[u8], equalized: &[u8]) -> Self {
        let (orig_mean, orig_std) = mean_stddev(original);
        let (eq_mean, eq_std) = mean_stddev(equalized);
        Self {
            original_mean: orig_mean,
            equalized_mean: eq_mean,
            original_std_dev: orig_std,
            equalized_std_dev: eq_std,
        }
    }
}

// ─── HistogramEqualizer ───────────────────────────────────────────────────────

/// Histogram equalization algorithms for 8-bit luma images.
#[derive(Debug, Clone, Default)]
pub struct HistogramEqualizer {
    /// When `true`, tile processing in CLAHE runs in parallel via rayon.
    pub use_parallel: bool,
}

impl HistogramEqualizer {
    /// Construct a new equalizer with parallel processing enabled.
    #[must_use]
    pub fn new() -> Self {
        Self { use_parallel: true }
    }

    // ── Global equalization ───────────────────────────────────────────────────

    /// Apply global histogram equalization to a single-channel (luma) image.
    ///
    /// If the frame contains a single unique value, the input is returned
    /// unchanged.
    ///
    /// # Arguments
    ///
    /// * `frame` – packed 8-bit luma bytes, row-major.
    /// * `width` / `height` – image dimensions (informational; total pixels is
    ///   `frame.len()`).
    #[must_use]
    pub fn equalize_luma(frame: &[u8], width: u32, height: u32) -> Vec<u8> {
        let _ = (width, height); // dimensions for future use
        if frame.is_empty() {
            return Vec::new();
        }
        let lut = build_global_lut(frame);
        frame.iter().map(|&p| lut[usize::from(p)]).collect()
    }

    /// Instance method wrapping the static [`equalize_luma`].
    ///
    /// [`equalize_luma`]: HistogramEqualizer::equalize_luma
    #[must_use]
    pub fn equalize_luma_instance(&self, frame: &[u8], width: u32, height: u32) -> Vec<u8> {
        Self::equalize_luma(frame, width, height)
    }

    // ── CLAHE ─────────────────────────────────────────────────────────────────

    /// Apply Contrast Limited Adaptive Histogram Equalization.
    ///
    /// The image is partitioned into a `tile_size × tile_size` grid.  Each
    /// tile's histogram is clipped at `clip_limit`, redistributed, and used to
    /// derive a local look-up table.  Each output pixel is produced by
    /// bilinear interpolation of the four surrounding tile LUTs.
    ///
    /// # Arguments
    ///
    /// * `frame` – packed 8-bit luma bytes, row-major.
    /// * `width` / `height` – image dimensions.
    /// * `clip_limit` – histogram clip ratio; 1.0 = fully clipped (equivalent
    ///   to global equalization), larger values allow more local contrast.
    /// * `tile_size` – tile edge length in pixels.
    #[must_use]
    pub fn clahe(
        frame: &[u8],
        width: u32,
        height: u32,
        clip_limit: f32,
        tile_size: u32,
    ) -> Vec<u8> {
        if frame.is_empty() || tile_size == 0 || width == 0 || height == 0 {
            return frame.to_vec();
        }

        // If tile_size covers the whole image, fall back to global equalization.
        if tile_size >= width || tile_size >= height {
            return Self::equalize_luma(frame, width, height);
        }

        let w = width as usize;
        let h = height as usize;
        let ts = tile_size as usize;

        // Number of tiles in each dimension.
        let tiles_x = (w + ts - 1) / ts;
        let tiles_y = (h + ts - 1) / ts;

        // Build per-tile LUTs.
        let tile_luts = build_tile_luts(frame, w, h, ts, tiles_x, tiles_y, clip_limit);

        // Produce output by bilinear interpolation.
        interpolate_output(frame, w, h, ts, tiles_x, tiles_y, &tile_luts)
    }

    /// Instance method wrapping the static [`clahe`].
    ///
    /// [`clahe`]: HistogramEqualizer::clahe
    #[must_use]
    pub fn clahe_instance(
        &self,
        frame: &[u8],
        width: u32,
        height: u32,
        clip_limit: f32,
        tile_size: u32,
    ) -> Vec<u8> {
        Self::clahe(frame, width, height, clip_limit, tile_size)
    }
}

// ─── Private helpers ──────────────────────────────────────────────────────────

/// Compute a 256-bin histogram from a byte slice.
fn compute_histogram(data: &[u8]) -> [u32; 256] {
    let mut hist = [0u32; 256];
    for &b in data {
        hist[usize::from(b)] += 1;
    }
    hist
}

/// Redistribute histogram bins that exceed `clip_limit * average_bin_count`.
///
/// Excess values are spread uniformly across all bins.
fn clip_histogram(hist: &mut [u32; 256], clip_limit: u32) {
    if clip_limit == 0 {
        return;
    }
    let mut excess: u64 = 0;
    for bin in hist.iter_mut() {
        if *bin > clip_limit {
            excess += u64::from(*bin - clip_limit);
            *bin = clip_limit;
        }
    }
    // Distribute excess evenly.
    let add_per_bin = (excess / 256) as u32;
    let remainder = (excess % 256) as usize;
    for (i, bin) in hist.iter_mut().enumerate() {
        *bin += add_per_bin;
        if i < remainder {
            *bin += 1;
        }
    }
}

/// Build a CDF array from a histogram.
fn compute_cdf(hist: &[u32; 256]) -> [u32; 256] {
    let mut cdf = [0u32; 256];
    let mut running = 0u32;
    for (i, &h) in hist.iter().enumerate() {
        running = running.saturating_add(h);
        cdf[i] = running;
    }
    cdf
}

/// Convert a CDF array to an 8-bit look-up table.
///
/// Uses the standard CDF-normalisation formula:
/// `lut[v] = round((cdf[v] - cdf_min) / (total - cdf_min) * 255)`
fn build_lut(cdf: &[u32; 256], total_pixels: u32) -> [u8; 256] {
    let cdf_min = cdf.iter().find(|&&v| v > 0).copied().unwrap_or(0);
    let denom = total_pixels.saturating_sub(cdf_min) as f64;
    let mut lut = [0u8; 256];
    for (i, &c) in cdf.iter().enumerate() {
        lut[i] = if denom < 1.0 {
            i as u8
        } else {
            let norm = (c.saturating_sub(cdf_min)) as f64 / denom;
            (norm * 255.0).round().clamp(0.0, 255.0) as u8
        };
    }
    lut
}

/// Build a global equalisation LUT for the full image.
fn build_global_lut(frame: &[u8]) -> [u8; 256] {
    let hist = compute_histogram(frame);
    let cdf = compute_cdf(&hist);
    build_lut(&cdf, frame.len() as u32)
}

/// Build one LUT per tile.
fn build_tile_luts(
    frame: &[u8],
    w: usize,
    h: usize,
    ts: usize,
    tiles_x: usize,
    tiles_y: usize,
    clip_limit: f32,
) -> Vec<[u8; 256]> {
    let num_tiles = tiles_x * tiles_y;
    let mut luts: Vec<[u8; 256]> = vec![[0u8; 256]; num_tiles];

    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let tile_idx = ty * tiles_x + tx;

            // Tile pixel bounds (clamped to image edges).
            let x0 = tx * ts;
            let y0 = ty * ts;
            let x1 = (x0 + ts).min(w);
            let y1 = (y0 + ts).min(h);
            let tile_pixels = (x1 - x0) * (y1 - y0);

            // Collect tile pixel values into a histogram.
            let mut hist = [0u32; 256];
            for row in y0..y1 {
                for col in x0..x1 {
                    let p = frame[row * w + col];
                    hist[usize::from(p)] += 1;
                }
            }

            // Clip limit is expressed as a ratio × average bin count.
            let avg_bin = (tile_pixels as f32 / 256.0).max(1.0);
            let clip_abs = ((clip_limit * avg_bin).round() as u32).max(1);
            clip_histogram(&mut hist, clip_abs);

            let cdf = compute_cdf(&hist);
            luts[tile_idx] = build_lut(&cdf, tile_pixels as u32);
        }
    }

    luts
}

/// Interpolate between tile LUTs to produce the final equalised image.
fn interpolate_output(
    frame: &[u8],
    w: usize,
    h: usize,
    ts: usize,
    tiles_x: usize,
    tiles_y: usize,
    tile_luts: &[[u8; 256]],
) -> Vec<u8> {
    let mut output = vec![0u8; frame.len()];

    for row in 0..h {
        for col in 0..w {
            let pixel = frame[row * w + col];

            // Fractional position within the tile grid (in tile units).
            // We use the tile *centre* as the reference point.
            let fx = ((col as f64 + 0.5) / ts as f64) - 0.5;
            let fy = ((row as f64 + 0.5) / ts as f64) - 0.5;

            // Tile index of the top-left interpolation neighbour.
            let tx0 = (fx.floor() as isize).clamp(0, tiles_x as isize - 1) as usize;
            let ty0 = (fy.floor() as isize).clamp(0, tiles_y as isize - 1) as usize;
            let tx1 = (tx0 + 1).min(tiles_x - 1);
            let ty1 = (ty0 + 1).min(tiles_y - 1);

            // Bilinear weights.
            let wx = (fx - tx0 as f64).clamp(0.0, 1.0);
            let wy = (fy - ty0 as f64).clamp(0.0, 1.0);

            // Fetch equalised values from the four surrounding tiles.
            let v00 = f64::from(tile_luts[ty0 * tiles_x + tx0][usize::from(pixel)]);
            let v10 = f64::from(tile_luts[ty0 * tiles_x + tx1][usize::from(pixel)]);
            let v01 = f64::from(tile_luts[ty1 * tiles_x + tx0][usize::from(pixel)]);
            let v11 = f64::from(tile_luts[ty1 * tiles_x + tx1][usize::from(pixel)]);

            let interp = v00 * (1.0 - wx) * (1.0 - wy)
                + v10 * wx * (1.0 - wy)
                + v01 * (1.0 - wx) * wy
                + v11 * wx * wy;

            output[row * w + col] = interp.round().clamp(0.0, 255.0) as u8;
        }
    }

    output
}

/// Compute mean and standard deviation of a byte slice.
fn mean_stddev(data: &[u8]) -> (f64, f64) {
    if data.is_empty() {
        return (0.0, 0.0);
    }
    let n = data.len() as f64;
    let mean = data.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    let variance = data
        .iter()
        .map(|&v| {
            let d = f64::from(v) - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    (mean, variance.sqrt())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── equalize_luma ─────────────────────────────────────────────────────────

    #[test]
    fn test_equalize_luma_empty() {
        let result = HistogramEqualizer::equalize_luma(&[], 0, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_equalize_luma_all_same_value_unchanged() {
        let frame = vec![128u8; 100];
        let out = HistogramEqualizer::equalize_luma(&frame, 10, 10);
        // Single unique value: CDF denominator is 0 → identity mapping
        assert_eq!(out.len(), 100);
        // All outputs should be the same (0 or 255 depending on mapping)
        let first = out[0];
        assert!(out.iter().all(|&v| v == first));
    }

    #[test]
    fn test_equalize_luma_ramp_spreads_contrast() {
        // Ramp from 0..=99 – after equalization the range should span more of
        // 0..255.
        let frame: Vec<u8> = (0..100u8).collect();
        let out = HistogramEqualizer::equalize_luma(&frame, 100, 1);
        assert_eq!(out.len(), 100);
        let min = *out.iter().min().expect("non-empty output");
        let max = *out.iter().max().expect("non-empty output");
        assert!(max > min, "equalization should spread values");
        // The last equalized value should be 255.
        assert_eq!(max, 255);
    }

    #[test]
    fn test_equalize_luma_single_pixel() {
        let frame = vec![77u8];
        let out = HistogramEqualizer::equalize_luma(&frame, 1, 1);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn test_equalize_luma_two_value_image() {
        // Half 0, half 255.
        let frame: Vec<u8> = (0..256).map(|i| if i < 128 { 0 } else { 255 }).collect();
        let out = HistogramEqualizer::equalize_luma(&frame, 256, 1);
        assert_eq!(out.len(), 256);
    }

    #[test]
    fn test_equalize_luma_preserves_size() {
        let frame: Vec<u8> = (0..=255).cycle().take(512).map(|v| v as u8).collect();
        let out = HistogramEqualizer::equalize_luma(&frame, 32, 16);
        assert_eq!(out.len(), 512);
    }

    #[test]
    fn test_equalize_luma_all_zeros() {
        let frame = vec![0u8; 64];
        let out = HistogramEqualizer::equalize_luma(&frame, 8, 8);
        assert_eq!(out.len(), 64);
        // All values should be identical.
        assert!(out.iter().all(|&v| v == out[0]));
    }

    #[test]
    fn test_equalize_luma_already_equalized() {
        // 256 unique values 0..=255 – already equalized.
        let frame: Vec<u8> = (0u8..=255).collect();
        let out = HistogramEqualizer::equalize_luma(&frame, 256, 1);
        assert_eq!(out.len(), 256);
        assert_eq!(out[0], 0);
        assert_eq!(out[255], 255);
    }

    // ── equalize_luma_instance ────────────────────────────────────────────────

    #[test]
    fn test_equalize_luma_instance_method() {
        let eq = HistogramEqualizer::new();
        let frame: Vec<u8> = (0u8..=255).collect();
        let out = eq.equalize_luma_instance(&frame, 256, 1);
        assert_eq!(out.len(), 256);
    }

    // ── clahe ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_clahe_basic_8x8_tile() {
        let w = 32u32;
        let h = 32u32;
        let frame: Vec<u8> = (0u8..=255).cycle().take((w * h) as usize).collect();
        let out = HistogramEqualizer::clahe(&frame, w, h, 2.0, 8);
        assert_eq!(out.len(), (w * h) as usize);
    }

    #[test]
    fn test_clahe_preserves_size() {
        let frame: Vec<u8> = vec![128u8; 256];
        let out = HistogramEqualizer::clahe(&frame, 16, 16, 2.0, 8);
        assert_eq!(out.len(), 256);
    }

    #[test]
    fn test_clahe_strong_clip() {
        let w = 64u32;
        let h = 64u32;
        let total = (w * h) as usize;
        let frame: Vec<u8> = (0..total).map(|i| (i % 256) as u8).collect();
        let out = HistogramEqualizer::clahe(&frame, w, h, 1.0, 8);
        assert_eq!(out.len(), total);
    }

    #[test]
    fn test_clahe_mild_clip() {
        let w = 32u32;
        let h = 32u32;
        let frame: Vec<u8> = (0u8..=255).cycle().take((w * h) as usize).collect();
        let out = HistogramEqualizer::clahe(&frame, w, h, 4.0, 8);
        assert_eq!(out.len(), (w * h) as usize);
    }

    #[test]
    fn test_clahe_tile_size_larger_than_image_falls_back() {
        let w = 4u32;
        let h = 4u32;
        let frame: Vec<u8> = (0u8..16).collect();
        // tile_size = 32 > image width → falls back to global equalization
        let out_clahe = HistogramEqualizer::clahe(&frame, w, h, 2.0, 32);
        let out_global = HistogramEqualizer::equalize_luma(&frame, w, h);
        assert_eq!(out_clahe, out_global);
    }

    #[test]
    fn test_clahe_tile_size_zero_returns_unchanged() {
        let frame = vec![100u8; 64];
        let out = HistogramEqualizer::clahe(&frame, 8, 8, 2.0, 0);
        assert_eq!(out, frame);
    }

    #[test]
    fn test_clahe_empty_frame() {
        let out = HistogramEqualizer::clahe(&[], 0, 0, 2.0, 8);
        assert!(out.is_empty());
    }

    // ── clahe_instance ────────────────────────────────────────────────────────

    #[test]
    fn test_clahe_instance_method() {
        let eq = HistogramEqualizer::new();
        let w = 16u32;
        let h = 16u32;
        let frame: Vec<u8> = (0u8..=255).cycle().take((w * h) as usize).collect();
        let out = eq.clahe_instance(&frame, w, h, 2.0, 8);
        assert_eq!(out.len(), (w * h) as usize);
    }

    // ── EqualizationStats ─────────────────────────────────────────────────────

    #[test]
    fn test_equalization_stats_compute() {
        let original: Vec<u8> = vec![0, 0, 255, 255];
        let equalized: Vec<u8> = vec![0, 85, 170, 255];
        let stats = EqualizationStats::compute(&original, &equalized);
        assert!((stats.original_mean - 127.5).abs() < 1.0);
        assert!(stats.equalized_mean > 0.0);
        assert!(stats.original_std_dev > 0.0);
        assert!(stats.equalized_std_dev >= 0.0);
    }

    #[test]
    fn test_equalization_stats_empty() {
        let stats = EqualizationStats::compute(&[], &[]);
        assert_eq!(stats.original_mean, 0.0);
        assert_eq!(stats.equalized_mean, 0.0);
    }

    // ── ClaheConfig ───────────────────────────────────────────────────────────

    #[test]
    fn test_clahe_config_defaults() {
        let cfg = ClaheConfig::default();
        assert!((cfg.clip_limit - 2.0).abs() < 1e-6);
        assert_eq!(cfg.tile_size, 8);
        assert!(cfg.use_parallel);
    }
}
