//! Texture analysis: GLCM (Gray Level Co-occurrence Matrix), Haralick features,
//! and Local Binary Pattern (LBP) histograms.
//!
//! All operations work on single-channel grayscale u8 images in row-major order.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// GLCM
// ---------------------------------------------------------------------------

/// Gray Level Co-occurrence Matrix.
///
/// A square `levels x levels` matrix where entry `(i, j)` counts how often
/// a pixel with gray level `i` co-occurs with a pixel with gray level `j`
/// at offset `(dx, dy)`.
#[derive(Debug, Clone)]
pub struct GlcmMatrix {
    /// Row-major matrix data, normalized to probabilities (sums to 1.0).
    pub data: Vec<f64>,
    /// Number of quantization levels (matrix side length).
    pub levels: u32,
}

impl GlcmMatrix {
    /// Create a zero-filled GLCM.
    #[must_use]
    pub fn new(levels: u32) -> Self {
        let n = (levels as usize) * (levels as usize);
        Self {
            data: vec![0.0; n],
            levels,
        }
    }

    /// Get the value at (row, col).
    #[must_use]
    pub fn get(&self, row: u32, col: u32) -> f64 {
        if row >= self.levels || col >= self.levels {
            return 0.0;
        }
        self.data[(row as usize) * (self.levels as usize) + (col as usize)]
    }

    /// Set the value at (row, col).
    pub fn set(&mut self, row: u32, col: u32, value: f64) {
        if row < self.levels && col < self.levels {
            self.data[(row as usize) * (self.levels as usize) + (col as usize)] = value;
        }
    }

    /// Sum of all entries (should be ~1.0 after normalization).
    #[must_use]
    pub fn sum(&self) -> f64 {
        self.data.iter().sum()
    }

    /// Create a symmetric GLCM: `(M + M^T) / 2`.
    #[must_use]
    pub fn symmetrize(&self) -> Self {
        let l = self.levels;
        let mut sym = Self::new(l);
        for i in 0..l {
            for j in 0..l {
                let val = (self.get(i, j) + self.get(j, i)) / 2.0;
                sym.set(i, j, val);
            }
        }
        sym
    }
}

/// Compute the GLCM for a grayscale image at a given offset `(dx, dy)`.
///
/// Gray levels are quantized to `levels` bins.  The resulting matrix is
/// normalized so that entries sum to 1.0.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn compute_glcm(
    image: &[u8],
    width: u32,
    height: u32,
    dx: i32,
    dy: i32,
    levels: u32,
) -> GlcmMatrix {
    let w = width as usize;
    let h = height as usize;
    let n = w * h;
    let levels = levels.max(2);
    let mut glcm = GlcmMatrix::new(levels);

    if n == 0 || image.len() < n {
        return glcm;
    }

    let scale = levels as f64 / 256.0;
    let mut total = 0u64;

    for y in 0..h {
        for x in 0..w {
            let nx = x as i64 + dx as i64;
            let ny = y as i64 + dy as i64;
            if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                continue;
            }

            let i_val = (image[y * w + x] as f64 * scale).floor() as u32;
            let j_val = (image[ny as usize * w + nx as usize] as f64 * scale).floor() as u32;

            let i_clamped = i_val.min(levels - 1);
            let j_clamped = j_val.min(levels - 1);

            let idx = (i_clamped as usize) * (levels as usize) + (j_clamped as usize);
            glcm.data[idx] += 1.0;
            total += 1;
        }
    }

    // Normalize
    if total > 0 {
        let inv = 1.0 / total as f64;
        for v in &mut glcm.data {
            *v *= inv;
        }
    }

    glcm
}

// ---------------------------------------------------------------------------
// Haralick texture features
// ---------------------------------------------------------------------------

/// Haralick texture features extracted from a GLCM.
#[derive(Debug, Clone, Copy, Default)]
pub struct TextureFeatures {
    /// Angular second moment (uniformity). Higher = more uniform texture.
    pub energy: f64,
    /// Contrast: weighted sum of squared differences.
    pub contrast: f64,
    /// Homogeneity (inverse difference moment): higher = more homogeneous.
    pub homogeneity: f64,
    /// Correlation: linear dependency of gray levels.
    pub correlation: f64,
    /// Entropy: randomness/disorder. Higher = more complex texture.
    pub entropy: f64,
}

impl TextureFeatures {
    /// Returns a summary vector `[energy, contrast, homogeneity, correlation, entropy]`.
    #[must_use]
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.energy,
            self.contrast,
            self.homogeneity,
            self.correlation,
            self.entropy,
        ]
    }

    /// Returns true if all features are finite.
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.energy.is_finite()
            && self.contrast.is_finite()
            && self.homogeneity.is_finite()
            && self.correlation.is_finite()
            && self.entropy.is_finite()
    }
}

/// Compute Haralick texture features from a GLCM.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn glcm_features(glcm: &GlcmMatrix) -> TextureFeatures {
    let l = glcm.levels;
    if l == 0 {
        return TextureFeatures::default();
    }

    let mut energy = 0.0_f64;
    let mut contrast = 0.0_f64;
    let mut homogeneity = 0.0_f64;
    let mut entropy = 0.0_f64;

    // Marginal means and standard deviations for correlation
    let mut mu_i = 0.0_f64;
    let mut mu_j = 0.0_f64;

    // Compute marginal means
    for i in 0..l {
        for j in 0..l {
            let p = glcm.get(i, j);
            mu_i += i as f64 * p;
            mu_j += j as f64 * p;
        }
    }

    // Compute marginal variances
    let mut sigma_i = 0.0_f64;
    let mut sigma_j = 0.0_f64;
    for i in 0..l {
        for j in 0..l {
            let p = glcm.get(i, j);
            sigma_i += (i as f64 - mu_i) * (i as f64 - mu_i) * p;
            sigma_j += (j as f64 - mu_j) * (j as f64 - mu_j) * p;
        }
    }

    let std_i = sigma_i.sqrt();
    let std_j = sigma_j.sqrt();

    // Compute all features in a single pass
    let mut correlation_num = 0.0_f64;
    for i in 0..l {
        for j in 0..l {
            let p = glcm.get(i, j);
            let diff = (i as i64 - j as i64).unsigned_abs() as f64;

            // Energy (Angular Second Moment)
            energy += p * p;

            // Contrast
            contrast += diff * diff * p;

            // Homogeneity (Inverse Difference Moment)
            homogeneity += p / (1.0 + diff * diff);

            // Entropy
            if p > 1e-15 {
                entropy -= p * p.ln();
            }

            // Correlation numerator
            correlation_num += (i as f64 - mu_i) * (j as f64 - mu_j) * p;
        }
    }

    let correlation = if std_i > 1e-12 && std_j > 1e-12 {
        correlation_num / (std_i * std_j)
    } else {
        0.0
    };

    TextureFeatures {
        energy,
        contrast,
        homogeneity,
        correlation,
        entropy,
    }
}

// ---------------------------------------------------------------------------
// Local Binary Pattern (LBP)
// ---------------------------------------------------------------------------

/// Compute the Local Binary Pattern histogram for a grayscale image.
///
/// For each pixel, samples `n_points` neighbors at distance `radius` in a
/// circular pattern.  Each neighbor is compared to the center: if >= center,
/// the corresponding bit is set.  The resulting binary code is used as a
/// histogram bin.
///
/// Returns a normalized histogram of length `2^n_points`.  `n_points` is
/// clamped to 1–16 to keep memory reasonable.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn lbp_features(image: &[u8], width: u32, height: u32, radius: u32, n_points: u32) -> Vec<f32> {
    let w = width as usize;
    let h = height as usize;
    let n = w * h;
    let np = n_points.clamp(1, 16) as usize;
    let r = radius.max(1) as f64;

    let num_bins = 1usize << np;
    let mut histogram = vec![0u64; num_bins];

    if n == 0 || image.len() < n {
        return vec![0.0_f32; num_bins];
    }

    // Precompute sample offsets
    let offsets: Vec<(f64, f64)> = (0..np)
        .map(|k| {
            let angle = 2.0 * std::f64::consts::PI * k as f64 / np as f64;
            (r * angle.cos(), -r * angle.sin())
        })
        .collect();

    let r_i = radius.max(1) as usize;

    for y in r_i..h.saturating_sub(r_i) {
        for x in r_i..w.saturating_sub(r_i) {
            let center = image[y * w + x];
            let mut code = 0u32;

            for (bit, &(dx, dy)) in offsets.iter().enumerate() {
                // Bilinear interpolation of the neighbor
                let fx = x as f64 + dx;
                let fy = y as f64 + dy;

                let x0 = fx.floor() as usize;
                let y0 = fy.floor() as usize;
                let x1 = (x0 + 1).min(w - 1);
                let y1 = (y0 + 1).min(h - 1);

                let tx = fx - fx.floor();
                let ty = fy - fy.floor();

                let v00 = image[y0 * w + x0] as f64;
                let v10 = image[y0 * w + x1] as f64;
                let v01 = image[y1 * w + x0] as f64;
                let v11 = image[y1 * w + x1] as f64;

                let val = v00 * (1.0 - tx) * (1.0 - ty)
                    + v10 * tx * (1.0 - ty)
                    + v01 * (1.0 - tx) * ty
                    + v11 * tx * ty;

                if val >= center as f64 {
                    code |= 1 << bit;
                }
            }

            if (code as usize) < num_bins {
                histogram[code as usize] += 1;
            }
        }
    }

    // Normalize
    let total: u64 = histogram.iter().sum();
    if total == 0 {
        return vec![0.0_f32; num_bins];
    }
    let inv = 1.0 / total as f64;
    histogram.iter().map(|&c| (c as f64 * inv) as f32).collect()
}

/// Compute uniform LBP pattern count.
///
/// A uniform LBP pattern has at most 2 bitwise transitions (0→1 or 1→0).
/// Returns the number of uniform patterns in the histogram indices.
#[must_use]
pub fn count_uniform_patterns(n_points: u32) -> u32 {
    let np = n_points.clamp(1, 16);
    let num_bins = 1u32 << np;
    let mut count = 0u32;

    for code in 0..num_bins {
        let transitions = count_transitions(code, np);
        if transitions <= 2 {
            count += 1;
        }
    }
    count
}

/// Count the number of 0→1 and 1→0 transitions in a circular binary pattern.
fn count_transitions(code: u32, n_bits: u32) -> u32 {
    let mut transitions = 0u32;
    for i in 0..n_bits {
        let bit_i = (code >> i) & 1;
        let bit_next = (code >> ((i + 1) % n_bits)) & 1;
        if bit_i != bit_next {
            transitions += 1;
        }
    }
    transitions
}

/// Compute the LBP uniformity measure for a histogram.
///
/// Returns the fraction of total weight concentrated in uniform bins.
#[must_use]
pub fn lbp_uniformity(histogram: &[f32], n_points: u32) -> f32 {
    let np = n_points.clamp(1, 16);
    let num_bins = 1usize << np;
    if histogram.len() != num_bins {
        return 0.0;
    }

    let mut uniform_weight = 0.0_f64;
    let mut total_weight = 0.0_f64;

    for (code, &w) in histogram.iter().enumerate() {
        total_weight += w as f64;
        if count_transitions(code as u32, np) <= 2 {
            uniform_weight += w as f64;
        }
    }

    if total_weight < 1e-15 {
        return 0.0;
    }
    (uniform_weight / total_weight) as f32
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- GlcmMatrix ---

    #[test]
    fn test_glcm_new() {
        let g = GlcmMatrix::new(4);
        assert_eq!(g.levels, 4);
        assert_eq!(g.data.len(), 16);
        assert!((g.sum()).abs() < 1e-12);
    }

    #[test]
    fn test_glcm_get_set() {
        let mut g = GlcmMatrix::new(4);
        g.set(1, 2, 0.5);
        assert!((g.get(1, 2) - 0.5).abs() < 1e-12);
        assert!(g.get(0, 0).abs() < 1e-12);
    }

    #[test]
    fn test_glcm_out_of_bounds() {
        let g = GlcmMatrix::new(2);
        assert!(g.get(5, 5).abs() < 1e-12);
    }

    #[test]
    fn test_glcm_symmetrize() {
        let mut g = GlcmMatrix::new(3);
        g.set(0, 1, 0.4);
        g.set(1, 0, 0.2);
        let s = g.symmetrize();
        assert!((s.get(0, 1) - 0.3).abs() < 1e-12);
        assert!((s.get(1, 0) - 0.3).abs() < 1e-12);
    }

    // --- compute_glcm ---

    #[test]
    fn test_compute_glcm_uniform() {
        let img = vec![128u8; 16];
        let g = compute_glcm(&img, 4, 4, 1, 0, 8);
        // All co-occurrences should land in one bin
        let bin = (128.0_f64 * 8.0 / 256.0).floor() as u32;
        assert!(g.get(bin, bin) > 0.9);
    }

    #[test]
    fn test_compute_glcm_normalized() {
        let img: Vec<u8> = (0..64).map(|i| (i * 4) as u8).collect();
        let g = compute_glcm(&img, 8, 8, 1, 0, 16);
        let sum = g.sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "GLCM should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn test_compute_glcm_empty() {
        let g = compute_glcm(&[], 0, 0, 1, 0, 8);
        assert!(g.sum().abs() < 1e-12);
    }

    #[test]
    fn test_compute_glcm_vertical() {
        let img = vec![0u8, 0, 255, 255]; // 2x2
        let g = compute_glcm(&img, 2, 2, 0, 1, 4);
        // Vertical offset: (0,0)→(0,255) and (0,0)→(0,255) transitions
        assert!(g.sum() > 0.0);
    }

    // --- TextureFeatures / glcm_features ---

    #[test]
    fn test_features_uniform_image() {
        let img = vec![100u8; 64];
        let g = compute_glcm(&img, 8, 8, 1, 0, 8);
        let f = glcm_features(&g);
        // Uniform image: high energy, zero contrast
        assert!(f.energy > 0.8, "Energy {}", f.energy);
        assert!(f.contrast < 0.01, "Contrast {}", f.contrast);
        assert!(f.is_finite());
    }

    #[test]
    fn test_features_to_vec() {
        let f = TextureFeatures {
            energy: 1.0,
            contrast: 2.0,
            homogeneity: 3.0,
            correlation: 4.0,
            entropy: 5.0,
        };
        assert_eq!(f.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_features_is_finite() {
        let f = TextureFeatures::default();
        assert!(f.is_finite());
    }

    #[test]
    fn test_features_high_contrast() {
        // Checkerboard pattern: high contrast
        let mut img = vec![0u8; 64];
        for i in 0..64 {
            let x = i % 8;
            let y = i / 8;
            img[i] = if (x + y) % 2 == 0 { 0 } else { 255 };
        }
        let g = compute_glcm(&img, 8, 8, 1, 0, 4);
        let f = glcm_features(&g);
        assert!(
            f.contrast > 0.5,
            "Expected high contrast, got {}",
            f.contrast
        );
    }

    #[test]
    fn test_features_correlation_uniform() {
        let img = vec![50u8; 64];
        let g = compute_glcm(&img, 8, 8, 1, 0, 8);
        let f = glcm_features(&g);
        // Uniform: sigma is 0, correlation should be 0
        assert!(f.correlation.abs() < 1e-6);
    }

    // --- LBP ---

    #[test]
    fn test_lbp_uniform_image() {
        let img = vec![128u8; 16 * 16];
        let hist = lbp_features(&img, 16, 16, 1, 8);
        assert_eq!(hist.len(), 256); // 2^8
                                     // Uniform image: all comparisons equal, single dominant bin
        let max_val = hist.iter().cloned().fold(0.0_f32, f32::max);
        assert!(max_val > 0.5, "Expected dominant bin, max = {max_val}");
    }

    #[test]
    fn test_lbp_histogram_normalized() {
        let img: Vec<u8> = (0..100).map(|i| (i * 2) as u8).collect();
        let hist = lbp_features(&img, 10, 10, 1, 8);
        let sum: f32 = hist.iter().sum();
        if sum > 0.0 {
            assert!(
                (sum - 1.0).abs() < 0.01,
                "LBP histogram should sum to ~1.0, got {sum}"
            );
        }
    }

    #[test]
    fn test_lbp_empty() {
        let hist = lbp_features(&[], 0, 0, 1, 8);
        assert_eq!(hist.len(), 256);
        assert!(hist.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_lbp_n_points_clamped() {
        let img = vec![100u8; 25];
        let hist = lbp_features(&img, 5, 5, 1, 20); // clamped to 16
        assert_eq!(hist.len(), 65536); // 2^16
    }

    // --- Uniform patterns ---

    #[test]
    fn test_count_uniform_patterns_8() {
        // For 8 points: 0 transitions (2: all-0 + all-1) + 2 transitions (8*7/... = 56)
        // Actually: uniform count for 8 bits = 58
        let count = count_uniform_patterns(8);
        assert_eq!(count, 58);
    }

    #[test]
    fn test_count_transitions() {
        assert_eq!(count_transitions(0b0000_0000, 8), 0); // all zeros
        assert_eq!(count_transitions(0b1111_1111, 8), 0); // all ones
        assert_eq!(count_transitions(0b0000_0001, 8), 2); // one bit set
        assert_eq!(count_transitions(0b0101_0101, 8), 8); // alternating
    }

    #[test]
    fn test_lbp_uniformity_uniform_image() {
        let img = vec![100u8; 64];
        let hist = lbp_features(&img, 8, 8, 1, 8);
        let u = lbp_uniformity(&hist, 8);
        // Uniform image should produce mostly uniform patterns
        assert!(u > 0.5, "Expected high uniformity, got {u}");
    }

    #[test]
    fn test_lbp_uniformity_empty() {
        let hist = vec![0.0_f32; 256];
        let u = lbp_uniformity(&hist, 8);
        assert_eq!(u, 0.0);
    }

    #[test]
    fn test_lbp_uniformity_wrong_size() {
        let hist = vec![1.0_f32; 10]; // wrong size for 8 points
        let u = lbp_uniformity(&hist, 8);
        assert_eq!(u, 0.0);
    }
}
