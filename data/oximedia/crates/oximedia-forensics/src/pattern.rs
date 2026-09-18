//! Pattern analysis for forensic detection of copy-paste, interpolation, and synthesis artefacts.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Classification of a detected pattern signature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatternType {
    /// Pattern typical of DCT compression block boundaries.
    Compression,
    /// Pattern consistent with copy-paste manipulation.
    CopyPaste,
    /// Pattern produced by interpolation / upscaling.
    Interpolation,
    /// Pattern generated synthetically (AI/procedural).
    Synthesis,
    /// Pattern that appears naturally captured.
    Natural,
}

/// A signature extracted from image data for pattern comparison.
#[derive(Debug, Clone)]
pub struct PatternSignature {
    /// Numeric signature vector.
    pub signature: Vec<f64>,
    /// Interpretation of the pattern.
    pub pattern_type: PatternType,
}

impl PatternSignature {
    /// Compute the cosine similarity between this signature and another.
    ///
    /// Returns a value in [−1, 1]; 1.0 means identical direction.
    pub fn similarity(&self, other: &Self) -> f64 {
        cosine_similarity(&self.signature, &other.signature)
    }
}

/// Extract a simplified DCT-like energy pattern from an 8×8 (or `block_size²`) luma block.
///
/// A proper 2-D DCT is replaced by column-then-row 1-D transforms so that the
/// function remains dependency-free.  The result is a `block_size²`-length
/// vector of frequency coefficients normalised to [−1, 1].
pub fn extract_dct_pattern(luma: &[u8], block_size: usize) -> Vec<f64> {
    if block_size == 0 || luma.len() < block_size * block_size {
        return Vec::new();
    }
    let n = block_size;
    // Build f64 block.
    let mut block: Vec<f64> = luma[..n * n]
        .iter()
        .map(|&v| v as f64 / 127.5 - 1.0) // centre on zero
        .collect();

    // 1-D DCT-II along rows.
    dct_rows(&mut block, n);
    // 1-D DCT-II along columns.
    dct_cols(&mut block, n);

    // Normalise each coefficient by the DC term magnitude (avoid div-by-zero).
    let dc = block[0].abs().max(1e-9);
    block.iter().map(|v| v / dc).collect()
}

/// Compute per-block uniformity scores for a luma plane.
///
/// Each score is the variance of pixel values within the block, normalised to
/// [0, 1] (0 = flat, 1 = max variance).
pub fn analyze_block_uniformity(
    luma: &[u8],
    width: usize,
    height: usize,
    block_size: usize,
) -> Vec<f64> {
    if block_size == 0 || luma.len() < width * height {
        return Vec::new();
    }
    let cols = width / block_size;
    let rows = height / block_size;
    let mut scores = Vec::with_capacity(cols * rows);

    for by in 0..rows {
        for bx in 0..cols {
            let mut vals: Vec<f64> = Vec::with_capacity(block_size * block_size);
            for dy in 0..block_size {
                for dx in 0..block_size {
                    let y = by * block_size + dy;
                    let x = bx * block_size + dx;
                    if y < height && x < width {
                        vals.push(luma[y * width + x] as f64);
                    }
                }
            }
            if vals.is_empty() {
                scores.push(0.0);
                continue;
            }
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            let variance = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
            // Max possible variance for u8 is 128^2 = 16384.
            scores.push((variance / 16384.0).min(1.0));
        }
    }
    scores
}

/// Detect regions that may have been copy-pasted within the same image.
///
/// Returns a list of `(x, y, w, h)` bounding boxes (in pixels) of suspect blocks.
///
/// The algorithm:
/// 1. Divide the image into non-overlapping `block_size` × `block_size` blocks.
/// 2. Compute a simple hash (sum of absolute values) for each block.
/// 3. Flag pairs of blocks whose hash values are within `threshold` of each other.
///    Only the first occurrence is reported to avoid duplicates.
pub fn detect_copy_paste_regions(
    luma: &[u8],
    width: usize,
    height: usize,
) -> Vec<(u32, u32, u32, u32)> {
    let block_size: usize = 8;
    let threshold: f64 = 4.0; // tolerance for near-identical blocks

    if luma.len() < width * height || width < block_size || height < block_size {
        return Vec::new();
    }
    let cols = width / block_size;
    let rows = height / block_size;

    // Build (block_hash, bx, by) list.
    let mut hashes: Vec<(f64, usize, usize)> = Vec::with_capacity(cols * rows);
    for by in 0..rows {
        for bx in 0..cols {
            let mut sum = 0.0_f64;
            for dy in 0..block_size {
                for dx in 0..block_size {
                    let y = by * block_size + dy;
                    let x = bx * block_size + dx;
                    sum += luma[y * width + x] as f64;
                }
            }
            hashes.push((sum, bx, by));
        }
    }

    // Sort by hash to make duplicate search O(n log n).
    hashes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut suspect: Vec<(u32, u32, u32, u32)> = Vec::new();
    let bs = block_size as u32;

    for pair in hashes.windows(2) {
        let (h1, bx1, by1) = pair[0];
        let (h2, bx2, by2) = pair[1];
        if (h2 - h1).abs() <= threshold && !(bx1 == bx2 && by1 == by2) {
            // Report the second block (the suspected copy destination).
            let rx = bx2 as u32 * bs;
            let ry = by2 as u32 * bs;
            suspect.push((rx, ry, bs, bs));
        }
    }

    suspect.dedup();
    suspect
}

/// Compute the cosine similarity between two equal-length vectors.
///
/// Returns 0.0 if either vector is zero or they differ in length.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mag_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if mag_a < 1e-12 || mag_b < 1e-12 {
        return 0.0;
    }
    (dot / (mag_a * mag_b)).clamp(-1.0, 1.0)
}

// ────────────────────────── internal DCT helpers ──────────────────────────

fn dct_1d(data: &mut [f64]) {
    let n = data.len();
    if n == 0 {
        return;
    }
    let mut out = vec![0.0_f64; n];
    let pi = std::f64::consts::PI;
    for k in 0..n {
        let mut sum = 0.0_f64;
        for (i, &v) in data.iter().enumerate() {
            sum += v * (pi * k as f64 * (2 * i + 1) as f64 / (2 * n) as f64).cos();
        }
        out[k] = sum;
    }
    data.copy_from_slice(&out);
}

fn dct_rows(block: &mut [f64], n: usize) {
    for row in 0..n {
        let start = row * n;
        dct_1d(&mut block[start..start + n]);
    }
}

fn dct_cols(block: &mut [f64], n: usize) {
    let mut col_buf = vec![0.0_f64; n];
    for c in 0..n {
        for (r, v) in col_buf.iter_mut().enumerate() {
            *v = block[r * n + c];
        }
        dct_1d(&mut col_buf);
        for (r, &v) in col_buf.iter().enumerate() {
            block[r * n + c] = v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn test_cosine_similarity_different_lengths() {
        assert_eq!(cosine_similarity(&[1.0, 2.0], &[1.0]), 0.0);
    }

    #[test]
    fn test_extract_dct_pattern_returns_correct_length() {
        let luma = vec![128u8; 64];
        let pattern = extract_dct_pattern(&luma, 8);
        assert_eq!(pattern.len(), 64);
    }

    #[test]
    fn test_extract_dct_pattern_empty_input() {
        let pattern = extract_dct_pattern(&[], 8);
        assert!(pattern.is_empty());
    }

    #[test]
    fn test_extract_dct_pattern_zero_block_size() {
        let luma = vec![100u8; 64];
        let pattern = extract_dct_pattern(&luma, 0);
        assert!(pattern.is_empty());
    }

    #[test]
    fn test_analyze_block_uniformity_flat_image() {
        // A perfectly flat image should have zero variance → score 0.
        let luma = vec![128u8; 64];
        let scores = analyze_block_uniformity(&luma, 8, 8, 4);
        assert!(scores.iter().all(|&s| s < 1e-9));
    }

    #[test]
    fn test_analyze_block_uniformity_scores_in_range() {
        let luma: Vec<u8> = (0..256).map(|i| (i % 256) as u8).collect();
        let scores = analyze_block_uniformity(&luma, 16, 16, 4);
        assert!(scores.iter().all(|&s| (0.0..=1.0).contains(&s)));
    }

    #[test]
    fn test_detect_copy_paste_no_copies_in_varied_image() {
        // Random-ish image — unlikely to have identical blocks.
        let luma: Vec<u8> = (0..512).map(|i| (i * 37 % 256) as u8).collect();
        // Should not crash.
        let regions = detect_copy_paste_regions(&luma, 16, 32);
        // Regions may or may not be found; just verify the return type is sane.
        for &(x, y, w, h) in &regions {
            assert!(x < 16);
            assert!(y < 32);
            assert!(w > 0);
            assert!(h > 0);
        }
    }

    #[test]
    fn test_detect_copy_paste_finds_duplicate_blocks() {
        // Build a 16×8 image whose left and right halves are identical.
        let half: Vec<u8> = (0..64).map(|_| 42u8).collect();
        let luma: Vec<u8> = [half.clone(), half].concat();
        let regions = detect_copy_paste_regions(&luma, 16, 8);
        // Both halves are identical → at least one suspect region.
        assert!(!regions.is_empty());
    }

    #[test]
    fn test_pattern_signature_similarity() {
        let sig = PatternSignature {
            signature: vec![1.0, 0.0, 0.0],
            pattern_type: PatternType::Natural,
        };
        assert!((sig.similarity(&sig) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_pattern_type_variants() {
        // Just verify the enum variants are usable.
        let types = [
            PatternType::Compression,
            PatternType::CopyPaste,
            PatternType::Interpolation,
            PatternType::Synthesis,
            PatternType::Natural,
        ];
        assert_eq!(types.len(), 5);
    }
}
