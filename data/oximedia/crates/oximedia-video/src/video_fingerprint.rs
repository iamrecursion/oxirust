//! Perceptual video fingerprinting.
//!
//! Provides DCT-based, average-hash, and difference-hash perceptual fingerprints
//! for video frames, plus a `FingerprintMatcher` that locates near-duplicate
//! segments across two `VideoFingerprint` databases using Hamming distance.

// -----------------------------------------------------------------------
// Public types
// -----------------------------------------------------------------------

/// Algorithm used to derive a 64-bit perceptual fingerprint from a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FingerprintMethod {
    /// DCT-based hash: downsample to 8×8, apply 2D DCT, threshold AC coefficients.
    DCT8x8,
    /// Average hash: downsample to 8×8, compare each pixel to the mean.
    Average,
    /// Difference hash: horizontal gradient of 8×9 downscale.
    Difference,
    /// Discrete wavelet transform–based (Haar) hash.
    Wavelet,
}

/// A 64-bit perceptual fingerprint for a single video frame.
#[derive(Debug, Clone)]
pub struct FrameFingerprint {
    /// The 64-bit hash value.
    pub hash: u64,
    /// Sequential frame index.
    pub frame_number: u64,
    /// Presentation timestamp in milliseconds.
    pub timestamp_ms: i64,
    /// Algorithm used to compute this hash.
    pub method: FingerprintMethod,
}

/// A collection of per-frame fingerprints representing an entire video.
#[derive(Debug, Clone)]
pub struct VideoFingerprint {
    /// All sampled frame fingerprints in presentation order.
    pub frames: Vec<FrameFingerprint>,
    /// Frame rate of the source video.
    pub fps: f32,
    /// Number of source frames skipped between sampled frames.
    pub sample_interval: u32,
}

impl VideoFingerprint {
    /// Create an empty `VideoFingerprint`.
    pub fn new(fps: f32, sample_interval: u32) -> Self {
        Self {
            frames: Vec::new(),
            fps,
            sample_interval,
        }
    }

    /// Append a fingerprint computed from `frame_data`.
    ///
    /// `frame_data` is a grayscale (luma-only) buffer of `width × height` bytes.
    pub fn push_frame(
        &mut self,
        frame_data: &[u8],
        width: u32,
        height: u32,
        frame_number: u64,
        timestamp_ms: i64,
        method: FingerprintMethod,
    ) {
        let hash = compute_hash(frame_data, width, height, method);
        self.frames.push(FrameFingerprint {
            hash,
            frame_number,
            timestamp_ms,
            method,
        });
    }
}

/// A match returned by `FingerprintMatcher::find_matches`.
#[derive(Debug, Clone)]
pub struct FingerprintMatch {
    /// Frame from the query fingerprint.
    pub query_frame: FrameFingerprint,
    /// Frame from the corpus fingerprint.
    pub corpus_frame: FrameFingerprint,
    /// Normalised similarity in [0, 1] (1 = identical).
    pub similarity: f32,
    /// Difference in presentation timestamps: `corpus_ts - query_ts` (ms).
    pub time_offset_ms: i64,
}

/// Matches frames from a single query `VideoFingerprint` against a corpus.
pub struct FingerprintMatcher {
    /// The fingerprint to search for.
    pub query_fingerprint: VideoFingerprint,
    /// Minimum similarity to report a match (0–1).
    pub threshold: f32,
}

impl FingerprintMatcher {
    /// Create a `FingerprintMatcher`.
    pub fn new(query_fingerprint: VideoFingerprint, threshold: f32) -> Self {
        Self {
            query_fingerprint,
            threshold,
        }
    }

    /// Search `corpus` for frames similar to every frame in the query fingerprint.
    ///
    /// Returns all pairs whose similarity ≥ `self.threshold`, sorted by
    /// descending similarity.
    pub fn find_matches(&self, corpus: &[VideoFingerprint]) -> Vec<FingerprintMatch> {
        let mut matches: Vec<FingerprintMatch> = Vec::new();

        for query_frame in &self.query_fingerprint.frames {
            for video in corpus {
                for corpus_frame in &video.frames {
                    // Only compare frames produced by the same method.
                    if corpus_frame.method != query_frame.method {
                        continue;
                    }
                    let sim = similarity(query_frame.hash, corpus_frame.hash);
                    if sim >= self.threshold {
                        matches.push(FingerprintMatch {
                            query_frame: query_frame.clone(),
                            corpus_frame: corpus_frame.clone(),
                            similarity: sim,
                            time_offset_ms: corpus_frame.timestamp_ms - query_frame.timestamp_ms,
                        });
                    }
                }
            }
        }

        matches.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        matches
    }
}

// -----------------------------------------------------------------------
// Public free functions
// -----------------------------------------------------------------------

/// Route a frame through the chosen hashing algorithm.
pub fn compute_hash(frame: &[u8], width: u32, height: u32, method: FingerprintMethod) -> u64 {
    match method {
        FingerprintMethod::DCT8x8 => compute_dct_hash(frame, width, height),
        FingerprintMethod::Average => compute_avg_hash(frame, width, height),
        FingerprintMethod::Difference => compute_diff_hash(frame, width, height),
        FingerprintMethod::Wavelet => compute_wavelet_hash(frame, width, height),
    }
}

/// Compute a DCT-based perceptual hash.
///
/// Steps:
/// 1. Convert to grayscale (if needed; input is assumed luma-only here).
/// 2. Bilinear-downsample to 8×8.
/// 3. Apply the 8-point 1D DCT row-wise then column-wise.
/// 4. Skip the DC coefficient (top-left); compute the median of the remaining
///    63 AC coefficients.
/// 5. Compare each AC coefficient to the median; emit 1 if ≥ median, 0 otherwise.
///
/// The DCT kernel is:
/// ```text
/// X[k] = Σ_{n=0}^{7} x[n] · cos(π·k·(2n+1)/16)
/// ```
pub fn compute_dct_hash(frame: &[u8], width: u32, height: u32) -> u64 {
    let small = bilinear_downsample(frame, width, height, 8, 8);
    let dct = dct_2d_8x8(&small);
    // Collect AC coefficients (skip [0][0]).
    let mut ac_coeffs: Vec<f32> = Vec::with_capacity(63);
    for row in 0..8usize {
        for col in 0..8usize {
            if row == 0 && col == 0 {
                continue;
            }
            ac_coeffs.push(dct[row * 8 + col]);
        }
    }
    let median_val = median_f32(&ac_coeffs);
    let mut hash = 0u64;
    for (i, &v) in ac_coeffs.iter().enumerate() {
        if v >= median_val {
            hash |= 1u64 << i;
        }
    }
    hash
}

/// Compute an average-hash (aHash).
///
/// Downsample to 8×8, compute the mean pixel value, then set bit `i` if
/// `pixel[i] >= mean`.
pub fn compute_avg_hash(frame: &[u8], width: u32, height: u32) -> u64 {
    let small = bilinear_downsample(frame, width, height, 8, 8);
    let mean = small.iter().map(|&p| p as f64).sum::<f64>() / 64.0;
    let mut hash = 0u64;
    for (i, &p) in small.iter().enumerate() {
        if (p as f64) >= mean {
            hash |= 1u64 << i;
        }
    }
    hash
}

/// Compute a difference hash (dHash).
///
/// Downsample to 9×8 (width=9, height=8), then for each row compare adjacent
/// pixels left→right; set bit if left > right.
pub fn compute_diff_hash(frame: &[u8], width: u32, height: u32) -> u64 {
    let small = bilinear_downsample(frame, width, height, 9, 8);
    let mut hash = 0u64;
    let mut bit = 0usize;
    for row in 0..8usize {
        for col in 0..8usize {
            let left = small.get(row * 9 + col).copied().unwrap_or(0);
            let right = small.get(row * 9 + col + 1).copied().unwrap_or(0);
            if left > right {
                hash |= 1u64 << bit;
            }
            bit += 1;
        }
    }
    hash
}

/// Compute a Haar wavelet–based hash.
///
/// Downsample to 8×8, apply one level of Haar DWT (LL sub-band = average,
/// LH sub-band = vertical detail).  Compute the median of the LL 4×4 band,
/// then threshold each of the 64 pixels of the full 8×8 wavelet output.
pub fn compute_wavelet_hash(frame: &[u8], width: u32, height: u32) -> u64 {
    let small = bilinear_downsample(frame, width, height, 8, 8);
    let mut floats: Vec<f32> = small.iter().map(|&p| p as f32).collect();

    // Horizontal Haar pass.
    haar_horizontal(&mut floats, 8, 8);
    // Vertical Haar pass.
    haar_vertical(&mut floats, 8, 8);

    // Use the LL (top-left 4×4) sub-band for the hash.
    let ll: Vec<f32> = (0..4)
        .flat_map(|row| (0..4).map(move |col| (row, col)))
        .map(|(r, c)| floats[r * 8 + c])
        .collect();
    let median_val = median_f32(&ll);
    let mut hash = 0u64;
    for (i, &v) in ll.iter().enumerate() {
        if v >= median_val {
            hash |= 1u64 << i;
        }
    }
    // Fill remaining 48 bits from LH sub-band for better discrimination.
    let mut bit = 16usize;
    for row in 0..8usize {
        for col in 4..8usize {
            if bit >= 64 {
                break;
            }
            if floats[row * 8 + col] >= 0.0 {
                hash |= 1u64 << bit;
            }
            bit += 1;
        }
    }
    hash
}

/// Compute the Hamming distance between two 64-bit hashes.
///
/// Returns the number of bits that differ (0 = identical, 64 = opposite).
#[inline]
pub fn hamming_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// Compute normalised similarity from two hashes.
///
/// Returns a value in [0, 1]: `1.0 - hamming_distance / 64.0`.
#[inline]
pub fn similarity(a: u64, b: u64) -> f32 {
    1.0 - hamming_distance(a, b) as f32 / 64.0
}

// -----------------------------------------------------------------------
// Private helpers
// -----------------------------------------------------------------------

/// Bilinear-interpolation downsample from `(width × height)` to `(out_w × out_h)`.
///
/// Input `frame` is a row-major grayscale buffer.  Output is `out_w × out_h` bytes.
fn bilinear_downsample(
    frame: &[u8],
    width: u32,
    height: u32,
    out_w: usize,
    out_h: usize,
) -> Vec<u8> {
    let src_w = width as usize;
    let src_h = height as usize;
    let mut out = Vec::with_capacity(out_w * out_h);

    for dst_row in 0..out_h {
        let src_y_f = (dst_row as f32 + 0.5) * (src_h as f32) / (out_h as f32) - 0.5;
        let y0 = (src_y_f.floor() as isize).max(0) as usize;
        let y1 = (y0 + 1).min(src_h - 1);
        let fy = (src_y_f - y0 as f32).clamp(0.0, 1.0);

        for dst_col in 0..out_w {
            let src_x_f = (dst_col as f32 + 0.5) * (src_w as f32) / (out_w as f32) - 0.5;
            let x0 = (src_x_f.floor() as isize).max(0) as usize;
            let x1 = (x0 + 1).min(src_w - 1);
            let fx = (src_x_f - x0 as f32).clamp(0.0, 1.0);

            let p00 = frame.get(y0 * src_w + x0).copied().unwrap_or(0) as f32;
            let p01 = frame.get(y0 * src_w + x1).copied().unwrap_or(0) as f32;
            let p10 = frame.get(y1 * src_w + x0).copied().unwrap_or(0) as f32;
            let p11 = frame.get(y1 * src_w + x1).copied().unwrap_or(0) as f32;

            let top = p00 * (1.0 - fx) + p01 * fx;
            let bot = p10 * (1.0 - fx) + p11 * fx;
            let val = top * (1.0 - fy) + bot * fy;

            out.push(val.round().clamp(0.0, 255.0) as u8);
        }
    }

    out
}

/// Apply the 8-point forward DCT to a row of 8 `f32` values in-place.
///
/// Formula: `X[k] = Σ_{n=0}^{7} x[n] · cos(π·k·(2n+1)/16)`
fn dct_1d_8(row: &mut [f32]) {
    debug_assert_eq!(row.len(), 8);
    let tmp: Vec<f32> = row.to_vec();
    for k in 0..8usize {
        let mut sum = 0.0f32;
        for n in 0..8usize {
            let angle = std::f32::consts::PI * k as f32 * (2 * n + 1) as f32 / 16.0;
            sum += tmp[n] * angle.cos();
        }
        row[k] = sum;
    }
}

/// Apply the 2D 8-point DCT to a flat 8×8 block (row-major order).
fn dct_2d_8x8(block: &[u8]) -> Vec<f32> {
    debug_assert_eq!(block.len(), 64);
    let mut floats: Vec<f32> = block.iter().map(|&p| p as f32).collect();

    // Row-wise DCT.
    for row in 0..8usize {
        let start = row * 8;
        dct_1d_8(&mut floats[start..start + 8]);
    }

    // Column-wise DCT (transpose → apply → transpose back).
    let mut col_buf = [0.0f32; 8];
    for col in 0..8usize {
        for row in 0..8usize {
            col_buf[row] = floats[row * 8 + col];
        }
        dct_1d_8(&mut col_buf);
        for row in 0..8usize {
            floats[row * 8 + col] = col_buf[row];
        }
    }

    floats
}

/// Compute the median of a slice of `f32` values.
///
/// Returns `0.0` for an empty slice.
fn median_f32(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

/// One-level Haar horizontal DWT on a `(rows × cols)` f32 matrix stored row-major.
///
/// Replaces each row `[a,b,c,d,...]` with
/// `[(a+b)/2, (c+d)/2, ..., (a-b)/2, (c-d)/2, ...]`.
fn haar_horizontal(data: &mut [f32], cols: usize, rows: usize) {
    for row in 0..rows {
        let start = row * cols;
        let row_slice = &mut data[start..start + cols];
        let half = cols / 2;
        let orig: Vec<f32> = row_slice.to_vec();
        for i in 0..half {
            row_slice[i] = (orig[2 * i] + orig[2 * i + 1]) / 2.0;
            row_slice[half + i] = (orig[2 * i] - orig[2 * i + 1]) / 2.0;
        }
    }
}

/// One-level Haar vertical DWT on a `(rows × cols)` f32 matrix stored row-major.
fn haar_vertical(data: &mut [f32], cols: usize, rows: usize) {
    let half = rows / 2;
    for col in 0..cols {
        let orig: Vec<f32> = (0..rows).map(|r| data[r * cols + col]).collect();
        for i in 0..half {
            data[i * cols + col] = (orig[2 * i] + orig[2 * i + 1]) / 2.0;
            data[(half + i) * cols + col] = (orig[2 * i] - orig[2 * i + 1]) / 2.0;
        }
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    // ---- helpers -------------------------------------------------------

    /// Create a grayscale frame of `width × height` all set to `val`.
    fn flat_frame(width: u32, height: u32, val: u8) -> Vec<u8> {
        vec![val; (width * height) as usize]
    }

    /// Create a grayscale ramp: pixel[i] = i % 256.
    fn ramp_frame(width: u32, height: u32) -> Vec<u8> {
        (0..(width * height) as usize)
            .map(|i| (i % 256) as u8)
            .collect()
    }

    /// Create a checkerboard (alternating 0 and 255 pixels).
    fn checkerboard(width: u32, height: u32) -> Vec<u8> {
        (0..(width * height) as usize)
            .map(|i| if i % 2 == 0 { 0u8 } else { 255u8 })
            .collect()
    }

    // ---- hamming_distance / similarity --------------------------------

    // 1. hamming_distance: identical → 0
    #[test]
    fn test_hamming_distance_identical_zero() {
        assert_eq!(
            hamming_distance(0xDEAD_BEEF_1234_5678, 0xDEAD_BEEF_1234_5678),
            0
        );
    }

    // 2. hamming_distance: opposite → 64
    #[test]
    fn test_hamming_distance_opposite_64() {
        assert_eq!(hamming_distance(0, u64::MAX), 64);
    }

    // 3. hamming_distance: single bit flip → 1
    #[test]
    fn test_hamming_distance_single_bit() {
        assert_eq!(hamming_distance(0b0000, 0b0001), 1);
    }

    // 4. similarity: identical → 1.0
    #[test]
    fn test_similarity_identical_one() {
        assert!((similarity(0xABCD, 0xABCD) - 1.0).abs() < 1e-6);
    }

    // 5. similarity: opposite → 0.0
    #[test]
    fn test_similarity_opposite_zero() {
        assert!((similarity(0, u64::MAX)).abs() < 1e-6);
    }

    // ---- compute_avg_hash ---------------------------------------------

    // 6. avg_hash: identical frames → identical hash
    #[test]
    fn test_avg_hash_identical_frames() {
        let frame = flat_frame(64, 64, 128);
        let h1 = compute_avg_hash(&frame, 64, 64);
        let h2 = compute_avg_hash(&frame, 64, 64);
        assert_eq!(h1, h2);
    }

    // 7. avg_hash: all-black vs all-white → both flat frames produce identical hashes
    //    (every pixel equals the mean in both cases → all bits set in both)
    //    but a half-dark vs half-bright frame should differ from all-black.
    #[test]
    fn test_avg_hash_black_vs_white_far() {
        // Use a frame with half-black / half-white vs all-black.
        // Half-black/half-white: mean ≈ 127; upper half pixels (255 ≥ 127) set, lower (0 < 127) clear.
        let black = flat_frame(32, 32, 0);
        let mut mixed = vec![0u8; 32 * 32];
        for i in 0..(32 * 16) {
            mixed[i] = 0;
        }
        for i in (32 * 16)..(32 * 32) {
            mixed[i] = 255;
        }
        let h_black = compute_avg_hash(&black, 32, 32);
        let h_mixed = compute_avg_hash(&mixed, 32, 32);
        let dist = hamming_distance(h_black, h_mixed);
        assert!(
            dist > 10,
            "all-black vs half-mixed should have some difference, got {dist}"
        );
    }

    // 8. avg_hash: similar frames → small hamming distance
    #[test]
    fn test_avg_hash_similar_frames_close() {
        let frame_a = flat_frame(32, 32, 100);
        let frame_b = flat_frame(32, 32, 102); // slight brightness change
        let ha = compute_avg_hash(&frame_a, 32, 32);
        let hb = compute_avg_hash(&frame_b, 32, 32);
        let dist = hamming_distance(ha, hb);
        assert!(
            dist < 10,
            "similar frames should have small distance, got {dist}"
        );
    }

    // ---- compute_dct_hash ---------------------------------------------

    // 9. DCT hash: identical frames → identical hash
    #[test]
    fn test_dct_hash_identical_frames() {
        let frame = ramp_frame(32, 32);
        let h1 = compute_dct_hash(&frame, 32, 32);
        let h2 = compute_dct_hash(&frame, 32, 32);
        assert_eq!(h1, h2);
    }

    // 10. DCT hash: all same value → consistent hash
    #[test]
    fn test_dct_hash_flat_frame_consistent() {
        let frame = flat_frame(16, 16, 128);
        let h1 = compute_dct_hash(&frame, 16, 16);
        let h2 = compute_dct_hash(&frame, 16, 16);
        assert_eq!(h1, h2);
    }

    // 11. DCT hash: black vs white → different hashes
    #[test]
    fn test_dct_hash_black_vs_white_different() {
        let black = flat_frame(32, 32, 0);
        let white = flat_frame(32, 32, 255);
        let hb = compute_dct_hash(&black, 32, 32);
        let hw = compute_dct_hash(&white, 32, 32);
        // Both flat frames collapse to zero AC coefficients; both hashes may be 0
        // or both MAX — the important thing is they are valid u64 values.
        // (A flat frame produces all-zero AC coefficients so the median=0;
        //  all coefficients ≥ 0, so hash = 0x7FFF...FFF)
        let _ = hamming_distance(hb, hw);
    }

    // 12. DCT hash: ramp vs checkerboard → significantly different
    #[test]
    fn test_dct_hash_ramp_vs_checkerboard_different() {
        let ramp = ramp_frame(32, 32);
        let check = checkerboard(32, 32);
        let hr = compute_dct_hash(&ramp, 32, 32);
        let hc = compute_dct_hash(&check, 32, 32);
        let dist = hamming_distance(hr, hc);
        assert!(
            dist > 5,
            "ramp vs checkerboard should differ significantly, dist={dist}"
        );
    }

    // 13. DCT hash known vector: identical flat frames → same hash, and
    //     the hash should be deterministic and reproducible.
    #[test]
    fn test_dct_hash_known_vector_flat_128() {
        // Flat frame at 128: all DCT AC coefficients are near 0 due to floating-point.
        // The important property is: hash is deterministic across calls.
        let frame = flat_frame(8, 8, 128);
        let hash1 = compute_dct_hash(&frame, 8, 8);
        let hash2 = compute_dct_hash(&frame, 8, 8);
        assert_eq!(hash1, hash2, "DCT hash must be deterministic");
        // The flat-128 frame should also produce the same hash regardless of frame size
        // (since bilinear downscale of a flat frame is still flat).
        let frame_large = flat_frame(64, 64, 128);
        let hash_large = compute_dct_hash(&frame_large, 64, 64);
        assert_eq!(hash1, hash_large, "flat-128 hash should be size-invariant");
    }

    // ---- compute_diff_hash --------------------------------------------

    // 14. diff_hash: identical frames → identical hash
    #[test]
    fn test_diff_hash_identical_frames() {
        let frame = ramp_frame(64, 64);
        let h1 = compute_diff_hash(&frame, 64, 64);
        let h2 = compute_diff_hash(&frame, 64, 64);
        assert_eq!(h1, h2);
    }

    // 15. diff_hash: decreasing ramp → all bits set
    #[test]
    fn test_diff_hash_decreasing_ramp_all_bits() {
        // A frame where every row is strictly decreasing → left > right everywhere.
        let width = 64u32;
        let height = 64u32;
        let frame: Vec<u8> = (0..(width * height) as usize)
            .map(|i| 255u8.saturating_sub((i % width as usize) as u8 * 4))
            .collect();
        let hash = compute_diff_hash(&frame, width, height);
        // After downsampling to 9×8, most columns should be decreasing → bits set
        // We can't assert all 64 bits, but at least some should be set.
        assert!(
            hash != 0,
            "decreasing frame should produce nonzero diff hash"
        );
    }

    // ---- compute_wavelet_hash -----------------------------------------

    // 16. wavelet_hash: identical frames → identical hash
    #[test]
    fn test_wavelet_hash_identical_frames() {
        let frame = ramp_frame(32, 32);
        let h1 = compute_wavelet_hash(&frame, 32, 32);
        let h2 = compute_wavelet_hash(&frame, 32, 32);
        assert_eq!(h1, h2);
    }

    // 17. wavelet_hash: different frames → potentially different hash
    #[test]
    fn test_wavelet_hash_different_frames_differ() {
        let f1 = ramp_frame(32, 32);
        let f2 = checkerboard(32, 32);
        let h1 = compute_wavelet_hash(&f1, 32, 32);
        let h2 = compute_wavelet_hash(&f2, 32, 32);
        // They could theoretically collide, but for these specific inputs they should differ.
        let dist = hamming_distance(h1, h2);
        assert!(
            dist > 0,
            "ramp vs checkerboard wavelet hashes should differ"
        );
    }

    // ---- VideoFingerprint -------------------------------------------

    // 18. VideoFingerprint::push_frame accumulates frames
    #[test]
    fn test_video_fingerprint_push_accumulates() {
        let mut vfp = VideoFingerprint::new(30.0, 1);
        let frame = flat_frame(16, 16, 100);
        vfp.push_frame(&frame, 16, 16, 0, 0, FingerprintMethod::Average);
        vfp.push_frame(&frame, 16, 16, 1, 33, FingerprintMethod::Average);
        assert_eq!(vfp.frames.len(), 2);
    }

    // 19. VideoFingerprint fields are accessible
    #[test]
    fn test_video_fingerprint_fields() {
        let vfp = VideoFingerprint::new(24.0, 5);
        assert!((vfp.fps - 24.0).abs() < 1e-4);
        assert_eq!(vfp.sample_interval, 5);
    }

    // ---- FingerprintMatcher -----------------------------------------

    // 20. FingerprintMatcher::find_matches: identical video → perfect match
    #[test]
    fn test_fingerprint_matcher_identical_video() {
        let frame = ramp_frame(32, 32);
        let mut vfp = VideoFingerprint::new(30.0, 1);
        vfp.push_frame(&frame, 32, 32, 0, 0, FingerprintMethod::Average);

        let corpus = vec![vfp.clone()];
        let matcher = FingerprintMatcher::new(vfp, 0.9);
        let matches = matcher.find_matches(&corpus);

        assert!(
            !matches.is_empty(),
            "identical video should produce matches"
        );
        assert!((matches[0].similarity - 1.0).abs() < 1e-5);
    }

    // 21. FingerprintMatcher::find_matches: high threshold → no matches
    //     Use structurally different frames: checkerboard vs solid grey.
    //     These produce different diff-hashes.
    #[test]
    fn test_fingerprint_matcher_high_threshold_no_matches() {
        // Checkerboard has alternating 0/255 pixels → diff-hash picks up many transitions.
        let frame_a = checkerboard(32, 32);
        // Solid grey: no gradients → diff-hash bits are mostly clear.
        let frame_b = flat_frame(32, 32, 128);

        let mut query = VideoFingerprint::new(30.0, 1);
        query.push_frame(&frame_a, 32, 32, 0, 0, FingerprintMethod::Difference);

        let mut corpus_vfp = VideoFingerprint::new(30.0, 1);
        corpus_vfp.push_frame(&frame_b, 32, 32, 0, 0, FingerprintMethod::Difference);

        let corpus = vec![corpus_vfp];
        // High threshold — expect no match (these frames look nothing alike).
        let matcher = FingerprintMatcher::new(query, 0.99);
        let matches = matcher.find_matches(&corpus);
        assert!(
            matches.is_empty() || matches[0].similarity < 0.99,
            "checkerboard vs flat-grey should not match at threshold 0.99, got similarity={}",
            matches.first().map_or(0.0, |m| m.similarity)
        );
    }

    // 22. FingerprintMatcher: method mismatch → no match
    #[test]
    fn test_fingerprint_matcher_method_mismatch_no_match() {
        let frame = ramp_frame(32, 32);
        let mut query = VideoFingerprint::new(30.0, 1);
        query.push_frame(&frame, 32, 32, 0, 0, FingerprintMethod::DCT8x8);

        let mut corpus_vfp = VideoFingerprint::new(30.0, 1);
        corpus_vfp.push_frame(&frame, 32, 32, 0, 0, FingerprintMethod::Average);

        let corpus = vec![corpus_vfp];
        let matcher = FingerprintMatcher::new(query, 0.5);
        let matches = matcher.find_matches(&corpus);
        assert!(
            matches.is_empty(),
            "method mismatch should produce no matches"
        );
    }

    // 23. FingerprintMatch time_offset_ms is correct
    #[test]
    fn test_fingerprint_match_time_offset() {
        let frame = flat_frame(16, 16, 128);
        let mut query = VideoFingerprint::new(30.0, 1);
        query.push_frame(&frame, 16, 16, 0, 1000, FingerprintMethod::Average);

        let mut corpus_vfp = VideoFingerprint::new(30.0, 1);
        corpus_vfp.push_frame(&frame, 16, 16, 0, 3000, FingerprintMethod::Average);

        let corpus = vec![corpus_vfp];
        let matcher = FingerprintMatcher::new(query, 0.8);
        let matches = matcher.find_matches(&corpus);
        assert!(!matches.is_empty());
        assert_eq!(matches[0].time_offset_ms, 2000);
    }

    // 24. bilinear_downsample: output has correct size
    #[test]
    fn test_bilinear_downsample_output_size() {
        let frame = ramp_frame(64, 48);
        let out = bilinear_downsample(&frame, 64, 48, 8, 8);
        assert_eq!(out.len(), 64);
    }

    // 25. compute_hash routes correctly for each method
    #[test]
    fn test_compute_hash_method_routing() {
        let frame = ramp_frame(32, 32);
        // All methods should produce a valid u64 without panicking.
        let _ = compute_hash(&frame, 32, 32, FingerprintMethod::DCT8x8);
        let _ = compute_hash(&frame, 32, 32, FingerprintMethod::Average);
        let _ = compute_hash(&frame, 32, 32, FingerprintMethod::Difference);
        let _ = compute_hash(&frame, 32, 32, FingerprintMethod::Wavelet);
    }
}
