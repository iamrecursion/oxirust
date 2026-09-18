//! Image texture analysis using Gray-Level Co-occurrence Matrices (GLCM).
//!
//! Provides Haralick texture feature computation from co-occurrence statistics.

/// Texture descriptor computed from a GLCM.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TextureDescriptor {
    /// Energy (angular second moment): measures texture uniformity.
    pub energy: f32,
    /// Entropy: measures texture randomness / disorder.
    pub entropy: f32,
    /// Homogeneity (inverse difference moment): measures local texture uniformity.
    pub homogeneity: f32,
    /// Contrast: measures intensity differences between neighboring pixels.
    pub contrast: f32,
    /// Correlation: measures linear dependencies in the GLCM.
    pub correlation: f32,
}

impl TextureDescriptor {
    /// Returns `true` if the texture is considered uniform (energy above `threshold`).
    #[must_use]
    pub fn is_uniform(&self, threshold: f32) -> bool {
        self.energy >= threshold
    }

    /// Returns `true` if the texture is considered complex (high entropy, high contrast).
    #[must_use]
    pub fn is_complex(&self) -> bool {
        self.entropy > 3.0 && self.contrast > 10.0
    }
}

/// A Gray-Level Co-occurrence Matrix (GLCM).
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct GlcmMatrix {
    /// Raw co-occurrence counts stored as a flattened `levels × levels` matrix.
    pub matrix: Vec<Vec<u32>>,
    /// Number of gray levels (quantization bins).
    pub levels: usize,
}

impl GlcmMatrix {
    /// Creates a new zeroed GLCM with the given number of gray levels.
    #[must_use]
    pub fn new(levels: usize) -> Self {
        Self {
            matrix: vec![vec![0u32; levels]; levels],
            levels,
        }
    }

    /// Increments the co-occurrence count for the pair `(i, j)`.
    pub fn add(&mut self, i: usize, j: usize) {
        if i < self.levels && j < self.levels {
            self.matrix[i][j] = self.matrix[i][j].saturating_add(1);
        }
    }

    /// Returns a normalized version of the GLCM as probabilities.
    #[must_use]
    pub fn normalize(&self) -> Vec<Vec<f32>> {
        let total = self.total_count();
        if total == 0 {
            return vec![vec![0.0_f32; self.levels]; self.levels];
        }
        let total_f = total as f32;
        self.matrix
            .iter()
            .map(|row| row.iter().map(|&v| v as f32 / total_f).collect())
            .collect()
    }

    /// Returns the total number of co-occurrence counts in the matrix.
    #[must_use]
    pub fn total_count(&self) -> u32 {
        self.matrix.iter().flat_map(|row| row.iter()).sum()
    }
}

/// Computes a GLCM from a grayscale pixel buffer with the given spatial offset `(dx, dy)`.
///
/// Pixels are quantized into `levels` bins. Symmetric pairs are counted
/// (both `(i,j)` and `(j,i)` are incremented for each neighbor pair).
#[must_use]
pub fn compute_glcm(
    pixels: &[u8],
    width: usize,
    height: usize,
    dx: i32,
    dy: i32,
    levels: usize,
) -> GlcmMatrix {
    let mut glcm = GlcmMatrix::new(levels);
    if levels == 0 || width == 0 || height == 0 {
        return glcm;
    }
    let scale = levels as f32 / 256.0;

    for y in 0..height {
        for x in 0..width {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                continue;
            }
            let src = pixels[y * width + x];
            let dst = pixels[ny as usize * width + nx as usize];
            let i = ((src as f32 * scale) as usize).min(levels - 1);
            let j = ((dst as f32 * scale) as usize).min(levels - 1);
            glcm.add(i, j);
            glcm.add(j, i);
        }
    }
    glcm
}

/// Computes Haralick texture features from a GLCM.
///
/// Returns a [`TextureDescriptor`] with energy, entropy, homogeneity, contrast,
/// and correlation computed from the normalized co-occurrence probabilities.
#[must_use]
pub fn compute_texture_descriptor(glcm: &GlcmMatrix) -> TextureDescriptor {
    let norm = glcm.normalize();
    let n = glcm.levels;

    let mut energy = 0.0_f32;
    let mut entropy = 0.0_f32;
    let mut homogeneity = 0.0_f32;
    let mut contrast = 0.0_f32;
    let mut mean_i = 0.0_f32;
    let mut mean_j = 0.0_f32;
    let mut std_i = 0.0_f32;
    let mut std_j = 0.0_f32;
    let mut correlation = 0.0_f32;

    // First pass: energy, entropy, homogeneity, contrast, means
    for i in 0..n {
        for j in 0..n {
            let p = norm[i][j];
            if p == 0.0 {
                continue;
            }
            energy += p * p;
            entropy -= p * p.ln();
            let diff = (i as f32 - j as f32).abs();
            homogeneity += p / (1.0 + diff);
            contrast += diff * diff * p;
            mean_i += i as f32 * p;
            mean_j += j as f32 * p;
        }
    }

    // Second pass: standard deviations
    for i in 0..n {
        for j in 0..n {
            let p = norm[i][j];
            std_i += (i as f32 - mean_i).powi(2) * p;
            std_j += (j as f32 - mean_j).powi(2) * p;
        }
    }
    std_i = std_i.sqrt();
    std_j = std_j.sqrt();

    // Third pass: correlation
    if std_i > 1e-10 && std_j > 1e-10 {
        for i in 0..n {
            for j in 0..n {
                let p = norm[i][j];
                correlation += (i as f32 - mean_i) * (j as f32 - mean_j) * p / (std_i * std_j);
            }
        }
    }

    TextureDescriptor {
        energy,
        entropy,
        homogeneity,
        contrast,
        correlation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- TextureDescriptor tests ----

    #[test]
    fn test_is_uniform_true() {
        let td = TextureDescriptor {
            energy: 0.9,
            entropy: 0.1,
            homogeneity: 0.95,
            contrast: 0.5,
            correlation: 0.1,
        };
        assert!(td.is_uniform(0.8));
    }

    #[test]
    fn test_is_uniform_false() {
        let td = TextureDescriptor {
            energy: 0.3,
            entropy: 2.5,
            homogeneity: 0.6,
            contrast: 5.0,
            correlation: -0.2,
        };
        assert!(!td.is_uniform(0.5));
    }

    #[test]
    fn test_is_complex_true() {
        let td = TextureDescriptor {
            energy: 0.01,
            entropy: 5.0,
            homogeneity: 0.2,
            contrast: 50.0,
            correlation: 0.0,
        };
        assert!(td.is_complex());
    }

    #[test]
    fn test_is_complex_false_low_entropy() {
        let td = TextureDescriptor {
            energy: 0.8,
            entropy: 1.0,
            homogeneity: 0.9,
            contrast: 50.0,
            correlation: 0.8,
        };
        assert!(!td.is_complex());
    }

    #[test]
    fn test_is_complex_false_low_contrast() {
        let td = TextureDescriptor {
            energy: 0.01,
            entropy: 5.0,
            homogeneity: 0.2,
            contrast: 5.0,
            correlation: 0.0,
        };
        assert!(!td.is_complex());
    }

    // ---- GlcmMatrix tests ----

    #[test]
    fn test_glcm_new_zeroed() {
        let glcm = GlcmMatrix::new(4);
        assert_eq!(glcm.total_count(), 0);
        assert_eq!(glcm.levels, 4);
    }

    #[test]
    fn test_glcm_add_and_total() {
        let mut glcm = GlcmMatrix::new(4);
        glcm.add(0, 1);
        glcm.add(0, 1);
        glcm.add(2, 3);
        assert_eq!(glcm.total_count(), 3);
        assert_eq!(glcm.matrix[0][1], 2);
        assert_eq!(glcm.matrix[2][3], 1);
    }

    #[test]
    fn test_glcm_add_out_of_bounds_ignored() {
        let mut glcm = GlcmMatrix::new(4);
        glcm.add(10, 0); // out of bounds
        assert_eq!(glcm.total_count(), 0);
    }

    #[test]
    fn test_glcm_normalize_sums_to_one() {
        let mut glcm = GlcmMatrix::new(4);
        for i in 0..4 {
            for j in 0..4 {
                glcm.add(i, j);
            }
        }
        let norm = glcm.normalize();
        let sum: f32 = norm.iter().flat_map(|row| row.iter()).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_glcm_normalize_empty_returns_zeros() {
        let glcm = GlcmMatrix::new(3);
        let norm = glcm.normalize();
        let sum: f32 = norm.iter().flat_map(|row| row.iter()).sum();
        assert!(sum.abs() < 1e-10);
    }

    // ---- compute_glcm tests ----

    #[test]
    fn test_compute_glcm_uniform_image() {
        // A 4x4 uniform image (all pixels = 128)
        let pixels = vec![128u8; 16];
        let glcm = compute_glcm(&pixels, 4, 4, 1, 0, 8);
        // All pairs should land in the same bin; diagonal should dominate
        assert!(glcm.total_count() > 0);
    }

    #[test]
    fn test_compute_glcm_zero_levels_returns_empty() {
        let pixels = vec![128u8; 16];
        let glcm = compute_glcm(&pixels, 4, 4, 1, 0, 0);
        assert_eq!(glcm.total_count(), 0);
    }

    #[test]
    fn test_compute_glcm_symmetric_counts() {
        // Two-pixel image with different values -> symmetric pairs added
        let pixels = vec![0u8, 255u8, 0u8, 255u8]; // 2x2
        let glcm = compute_glcm(&pixels, 2, 2, 1, 0, 2);
        let l = glcm.levels;
        // Symmetry: matrix[i][j] should equal matrix[j][i]
        for i in 0..l {
            for j in 0..l {
                assert_eq!(glcm.matrix[i][j], glcm.matrix[j][i]);
            }
        }
    }

    // ---- compute_texture_descriptor tests ----

    #[test]
    fn test_texture_descriptor_uniform_image() {
        // Uniform image -> diagonal GLCM -> high energy, low entropy
        let pixels = vec![100u8; 100]; // 10x10
        let glcm = compute_glcm(&pixels, 10, 10, 1, 0, 8);
        let td = compute_texture_descriptor(&glcm);
        // Energy should be high for uniform texture
        assert!(td.energy > 0.5, "energy={}", td.energy);
        // Contrast should be low (all same gray level)
        assert!(td.contrast < 1.0, "contrast={}", td.contrast);
    }

    #[test]
    fn test_texture_descriptor_energy_in_range() {
        let pixels: Vec<u8> = (0..64).map(|i| (i * 4) as u8).collect();
        let glcm = compute_glcm(&pixels, 8, 8, 1, 0, 8);
        let td = compute_texture_descriptor(&glcm);
        assert!(td.energy >= 0.0 && td.energy <= 1.0);
    }

    #[test]
    fn test_texture_descriptor_homogeneity_positive() {
        let pixels = vec![50u8; 25]; // 5x5
        let glcm = compute_glcm(&pixels, 5, 5, 0, 1, 8);
        let td = compute_texture_descriptor(&glcm);
        assert!(td.homogeneity > 0.0);
    }

    #[test]
    fn test_texture_descriptor_empty_glcm() {
        let glcm = GlcmMatrix::new(8);
        let td = compute_texture_descriptor(&glcm);
        assert_eq!(td.energy, 0.0);
        assert_eq!(td.entropy, 0.0);
        assert_eq!(td.contrast, 0.0);
    }
}
