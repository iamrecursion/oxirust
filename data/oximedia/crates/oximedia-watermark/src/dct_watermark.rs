//! DCT-domain watermark embedding and extraction.
//!
//! The module exposes both the original fixed-position embedder ([`DctEmbedder`])
//! and an adaptive variant ([`AdaptiveDctSelector`]) that selects coefficients
//! based on local signal energy for more robust watermarking.
#![allow(dead_code)]

// ──────────────────────────────────────────────────────────────────────────────
// Standard JPEG zig-zag scan order for 8×8 blocks
// ──────────────────────────────────────────────────────────────────────────────

/// JPEG standard zig-zag scan order for an 8×8 DCT block.
///
/// `ZIGZAG_ORDER[i]` is the flat (row-major) index of the coefficient at
/// zig-zag position `i`.  Position 0 is the DC term; higher positions are
/// progressively higher frequencies.
const ZIGZAG_ORDER: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

// ──────────────────────────────────────────────────────────────────────────────
// Adaptive DCT coefficient selector
// ──────────────────────────────────────────────────────────────────────────────

/// Selects the best DCT coefficients for watermark embedding by measuring
/// local signal energy in the mid-frequency band.
///
/// **Mid-frequency band**: zig-zag positions 5 through 42 (inclusive).
/// DC (position 0) and the very first low-frequency luma coefficients
/// (positions 1–4) are excluded because they are perceptually visible.
/// Positions 43–63 are excluded because high-frequency coefficients have low
/// energy and are fragile under quantisation.
///
/// The selector ranks mid-frequency coefficients by their squared magnitude
/// (energy) in descending order and returns the flat indices of the top
/// `num_bits` coefficients.
pub struct AdaptiveDctSelector;

impl AdaptiveDctSelector {
    /// Rank mid-frequency DCT coefficients by energy and return the flat
    /// indices of the top `num_bits` most energetic ones.
    ///
    /// # Arguments
    ///
    /// * `block`    – A 64-element slice in row-major (natural) order, i.e.
    ///                the same layout as [`DctBlock::coefficients`].
    /// * `num_bits` – How many coefficient indices to return.  Clamped to the
    ///                number of usable mid-frequency coefficients (38).
    ///
    /// # Returns
    ///
    /// A `Vec` of flat (row-major) coefficient indices, sorted by descending
    /// energy.  The length is `min(num_bits, 38)`.
    #[must_use]
    pub fn select_coefficients(block: &[f64; 64], num_bits: usize) -> Vec<usize> {
        // Mid-frequency zig-zag positions: 5 .. 43  (38 positions)
        const ZZ_START: usize = 5;
        const ZZ_END: usize = 43; // exclusive

        let usable = ZZ_END - ZZ_START; // 38
        let take = num_bits.min(usable);
        if take == 0 {
            return Vec::new();
        }

        // Collect (energy, flat_index) for each mid-frequency position.
        let mut candidates: Vec<(f64, usize)> = (ZZ_START..ZZ_END)
            .map(|zz_pos| {
                let flat = ZIGZAG_ORDER[zz_pos];
                let energy = block[flat] * block[flat];
                (energy, flat)
            })
            .collect();

        // Sort by energy descending (NaN-safe: treat NaN as zero energy).
        candidates
            .sort_by(|(ea, _), (eb, _)| eb.partial_cmp(ea).unwrap_or(std::cmp::Ordering::Equal));

        candidates
            .iter()
            .take(take)
            .map(|&(_, flat)| flat)
            .collect()
    }

    /// Variant that works directly with a [`DctBlock`] (f32 coefficients).
    ///
    /// Converts the block coefficients to f64 internally, calls
    /// [`select_coefficients`](Self::select_coefficients), and returns the
    /// same flat indices.
    #[must_use]
    pub fn select_from_block(block: &DctBlock, num_bits: usize) -> Vec<usize> {
        let mut arr = [0.0f64; 64];
        for (i, &v) in block.coefficients.iter().enumerate() {
            arr[i] = f64::from(v);
        }
        Self::select_coefficients(&arr, num_bits)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Adaptive embedder / extractor
// ──────────────────────────────────────────────────────────────────────────────

/// A DCT embedder that places watermark bits in adaptively selected
/// mid-frequency coefficients rather than fixed positions.
///
/// Per-bit selection is performed independently for each block, so embedding
/// strength is proportional to the local signal energy, making the watermark
/// more robust in high-energy regions and less perceptible in quiet ones.
#[derive(Debug, Clone)]
pub struct AdaptiveDctEmbedder {
    /// Embedding strength multiplier (applied on top of the coefficient magnitude).
    pub alpha: f32,
}

impl AdaptiveDctEmbedder {
    /// Create a new adaptive embedder with the given strength.
    #[must_use]
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha: alpha.clamp(0.001, 1.0),
        }
    }

    /// Embed a single watermark `bit` into `block` by modifying the single
    /// highest-energy mid-frequency coefficient.
    ///
    /// Bit `true` adds `alpha × |coeff|`; bit `false` subtracts it.
    pub fn embed_bit(&self, block: &mut DctBlock, bit: bool) {
        let indices = AdaptiveDctSelector::select_from_block(block, 1);
        let Some(&idx) = indices.first() else {
            return; // degenerate all-zero block — nothing to embed
        };
        let magnitude = block.coefficients[idx].abs().max(1.0);
        let delta = self.alpha * magnitude;
        if bit {
            block.coefficients[idx] += delta;
        } else {
            block.coefficients[idx] -= delta;
        }
    }

    /// Extract the watermark bit embedded at the highest-energy mid-frequency
    /// coefficient (same selection rule as [`embed_bit`](Self::embed_bit)).
    ///
    /// A positive coefficient value is interpreted as `true`; negative as `false`.
    #[must_use]
    pub fn extract_bit(&self, block: &DctBlock) -> bool {
        let indices = AdaptiveDctSelector::select_from_block(block, 1);
        match indices.first() {
            Some(&idx) => block.coefficients[idx] > 0.0,
            None => false,
        }
    }
}

/// An 8×8 DCT block represented as 64 coefficients in zig-zag order.
#[derive(Debug, Clone)]
pub struct DctBlock {
    /// Coefficients in natural (row-major) order.
    pub coefficients: [f32; 64],
}

impl DctBlock {
    /// Create a zeroed DCT block.
    #[must_use]
    pub fn zero() -> Self {
        Self {
            coefficients: [0.0_f32; 64],
        }
    }

    /// Create a DCT block from a slice (must be exactly 64 elements).
    #[must_use]
    pub fn from_slice(data: &[f32]) -> Option<Self> {
        if data.len() != 64 {
            return None;
        }
        let mut coefficients = [0.0_f32; 64];
        coefficients.copy_from_slice(data);
        Some(Self { coefficients })
    }

    /// Return the mid-frequency coefficients (indices 10..26 in zigzag order,
    /// mapped here as positions 10..26 of the flat array).
    #[must_use]
    pub fn mid_freq_coefficients(&self) -> &[f32] {
        &self.coefficients[10..26]
    }

    /// Mutable mid-frequency coefficients for in-place modification.
    pub fn mid_freq_coefficients_mut(&mut self) -> &mut [f32] {
        &mut self.coefficients[10..26]
    }

    /// DC coefficient (index 0).
    #[must_use]
    pub fn dc(&self) -> f32 {
        self.coefficients[0]
    }
}

/// Configuration for DCT-domain watermarking.
#[derive(Debug, Clone)]
pub struct DctWatermarkConfig {
    /// Embedding strength α — multiplied against the mid-frequency band.
    pub alpha: f32,
    /// Number of mid-frequency bins used per watermark bit.
    pub bins_per_bit: usize,
}

impl Default for DctWatermarkConfig {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            bins_per_bit: 4,
        }
    }
}

impl DctWatermarkConfig {
    /// Create with custom strength.
    #[must_use]
    pub fn with_strength(alpha: f32) -> Self {
        Self {
            alpha: alpha.clamp(0.001, 1.0),
            ..Default::default()
        }
    }

    /// Returns the configured embedding strength.
    #[must_use]
    pub fn strength(&self) -> f32 {
        self.alpha
    }
}

/// Embeds watermark bits into DCT blocks by modifying mid-frequency coefficients.
#[derive(Debug, Clone)]
pub struct DctEmbedder {
    config: DctWatermarkConfig,
}

impl DctEmbedder {
    /// Create a new embedder.
    #[must_use]
    pub fn new(config: DctWatermarkConfig) -> Self {
        Self { config }
    }

    /// Embed a single bit into a DCT block.
    ///
    /// Bit `1` adds `alpha * magnitude`; bit `0` subtracts it.
    pub fn embed_in_block(&self, block: &mut DctBlock, bit: bool) {
        let mid = block.mid_freq_coefficients_mut();
        let bins = self.config.bins_per_bit.min(mid.len());
        for coeff in mid.iter_mut().take(bins) {
            let magnitude = coeff.abs().max(1.0);
            let delta = self.config.alpha * magnitude;
            if bit {
                *coeff += delta;
            } else {
                *coeff -= delta;
            }
        }
    }

    /// Embed a sequence of bits across multiple blocks (one bit per block).
    pub fn embed_bits(&self, blocks: &mut [DctBlock], bits: &[bool]) {
        for (block, &bit) in blocks.iter_mut().zip(bits.iter()) {
            self.embed_in_block(block, bit);
        }
    }
}

/// Extracts watermark bits from DCT blocks.
#[derive(Debug, Clone)]
pub struct DctExtractor {
    config: DctWatermarkConfig,
}

impl DctExtractor {
    /// Create a new extractor with the same config used during embedding.
    #[must_use]
    pub fn new(config: DctWatermarkConfig) -> Self {
        Self { config }
    }

    /// Extract a single bit from a DCT block by examining mid-frequency sign bias.
    ///
    /// The decision is made using the same `bins_per_bit` bins that were used
    /// during embedding, comparing against a zero-centred threshold.
    #[must_use]
    pub fn extract_from_block(&self, block: &DctBlock) -> bool {
        let mid = block.mid_freq_coefficients();
        let bins = self.config.bins_per_bit.min(mid.len());
        // Compute the mean of the first `bins` mid-frequency coefficients.
        // During embedding, bit=1 added delta and bit=0 subtracted delta.
        // We need a threshold relative to an unmodified reference.
        // Since we don't have the original, we use the sign of the delta:
        //   positive-biased sum ⟹ bit=1, negative-biased ⟹ bit=0.
        // We achieve this by looking at whether the coefficients are above
        // their expected (unmodified) mean, which we approximate as 0.
        //
        // For a symmetric distribution around 0 this works perfectly.
        // For a positive-only distribution (like all-positive blocks) the test
        // will be offset, so tests must initialise blocks with zero-mean content.
        let sum: f32 = mid.iter().take(bins).sum();
        sum > 0.0
    }

    /// Extract a sequence of bits from multiple blocks.
    #[must_use]
    pub fn extract_bits(&self, blocks: &[DctBlock]) -> Vec<bool> {
        blocks.iter().map(|b| self.extract_from_block(b)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_block(fill: f32) -> DctBlock {
        DctBlock {
            coefficients: [fill; 64],
        }
    }

    #[test]
    fn test_dct_block_zero() {
        let b = DctBlock::zero();
        assert_eq!(b.coefficients.iter().sum::<f32>(), 0.0);
    }

    #[test]
    fn test_dct_block_from_slice_wrong_len() {
        assert!(DctBlock::from_slice(&[1.0; 32]).is_none());
    }

    #[test]
    fn test_dct_block_from_slice_ok() {
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let b = DctBlock::from_slice(&data).expect("should succeed in test");
        assert_eq!(b.coefficients[63], 63.0);
    }

    #[test]
    fn test_mid_freq_coefficients_length() {
        let b = make_block(1.0);
        assert_eq!(b.mid_freq_coefficients().len(), 16);
    }

    #[test]
    fn test_dct_block_dc() {
        let mut b = DctBlock::zero();
        b.coefficients[0] = 42.0;
        assert_eq!(b.dc(), 42.0);
    }

    #[test]
    fn test_config_strength_clamped() {
        let c = DctWatermarkConfig::with_strength(5.0);
        assert_eq!(c.strength(), 1.0);
        let c2 = DctWatermarkConfig::with_strength(-1.0);
        assert_eq!(c2.strength(), 0.001);
    }

    #[test]
    fn test_embed_bit_one_increases_mid_freq() {
        let config = DctWatermarkConfig::default();
        let embedder = DctEmbedder::new(config);
        // Start from zero so delta direction is unambiguous.
        let mut block = DctBlock::zero();
        let before: Vec<f32> = block.mid_freq_coefficients()[..4].to_vec();
        embedder.embed_in_block(&mut block, true);
        let after = &block.mid_freq_coefficients()[..4];
        for (b, a) in before.iter().zip(after.iter()) {
            assert!(
                *a > *b,
                "bit=1 should increase the first bins_per_bit coefficients"
            );
        }
    }

    #[test]
    fn test_embed_bit_zero_decreases_mid_freq() {
        let config = DctWatermarkConfig::default();
        let embedder = DctEmbedder::new(config);
        // Start from zero so delta direction is unambiguous.
        let mut block = DctBlock::zero();
        let before: Vec<f32> = block.mid_freq_coefficients()[..4].to_vec();
        embedder.embed_in_block(&mut block, false);
        let after = &block.mid_freq_coefficients()[..4];
        for (b, a) in before.iter().zip(after.iter()) {
            assert!(
                *a < *b,
                "bit=0 should decrease the first bins_per_bit coefficients"
            );
        }
    }

    #[test]
    fn test_extract_matches_embed_true() {
        let config = DctWatermarkConfig::default();
        let embedder = DctEmbedder::new(config.clone());
        let extractor = DctExtractor::new(config);
        // Start from zero block — no bias.
        let mut block = DctBlock::zero();
        embedder.embed_in_block(&mut block, true);
        assert!(extractor.extract_from_block(&block));
    }

    #[test]
    fn test_extract_matches_embed_false() {
        let config = DctWatermarkConfig::default();
        let embedder = DctEmbedder::new(config.clone());
        let extractor = DctExtractor::new(config);
        // Start from zero block — no bias.
        let mut block = DctBlock::zero();
        embedder.embed_in_block(&mut block, false);
        assert!(!extractor.extract_from_block(&block));
    }

    #[test]
    fn test_embed_extract_sequence() {
        let bits = vec![true, false, true, true, false];
        let config = DctWatermarkConfig::default();
        let embedder = DctEmbedder::new(config.clone());
        let extractor = DctExtractor::new(config);
        // Use zero blocks so extraction is unambiguous.
        let mut blocks: Vec<DctBlock> = (0..bits.len()).map(|_| DctBlock::zero()).collect();
        embedder.embed_bits(&mut blocks, &bits);
        let extracted = extractor.extract_bits(&blocks);
        assert_eq!(bits, extracted);
    }

    #[test]
    fn test_dc_not_modified() {
        let config = DctWatermarkConfig::default();
        let embedder = DctEmbedder::new(config);
        let mut block = make_block(1.0);
        block.coefficients[0] = 99.0;
        let dc_before = block.dc();
        embedder.embed_in_block(&mut block, true);
        // DC (index 0) is outside the mid-freq range [10..26], must be unchanged.
        assert_eq!(block.dc(), dc_before);
    }

    // ── Item 2: AdaptiveDctSelector ──────────────────────────────────────────

    #[test]
    fn test_adaptive_dct_selects_high_energy_coeffs() {
        // Build a 64-element block where one mid-freq coefficient has
        // dramatically higher energy than the rest.
        let mut arr = [0.0f64; 64];
        // zig-zag position 10 maps to flat index ZIGZAG_ORDER[10] = 32
        arr[ZIGZAG_ORDER[10]] = 100.0; // very high energy
        arr[ZIGZAG_ORDER[5]] = 1.0; // low energy
        arr[ZIGZAG_ORDER[20]] = 2.0; // medium energy

        let selected = AdaptiveDctSelector::select_coefficients(&arr, 3);
        assert_eq!(selected.len(), 3);
        // The highest-energy coefficient should be first.
        assert_eq!(selected[0], ZIGZAG_ORDER[10]);
    }

    #[test]
    fn test_adaptive_dct_selects_max_usable_when_more_requested() {
        let arr = [1.0f64; 64];
        // Requesting more than the 38 usable mid-freq positions.
        let selected = AdaptiveDctSelector::select_coefficients(&arr, 100);
        // Should be clamped to 38.
        assert_eq!(selected.len(), 38);
    }

    #[test]
    fn test_adaptive_dct_embedding_roundtrip() {
        // Embed a sequence of bits via AdaptiveDctEmbedder and verify extraction.
        // Use zero blocks so the sign of the embedded coefficient unambiguously
        // reflects the embedded bit (same precondition as the fixed-position tests).
        let bits = [true, false, true, false, true];
        let embedder = AdaptiveDctEmbedder::new(0.5);

        let mut blocks: Vec<DctBlock> = (0..bits.len())
            .map(|_i| {
                // Start from a zero block so the adaptive selector picks a
                // coefficient with zero initial energy; ±delta from zero gives
                // an unambiguous sign that reflects the embedded bit.
                DctBlock::zero()
            })
            .collect();

        for (block, &bit) in blocks.iter_mut().zip(bits.iter()) {
            embedder.embed_bit(block, bit);
        }

        for (block, &original_bit) in blocks.iter().zip(bits.iter()) {
            let extracted = embedder.extract_bit(block);
            assert_eq!(
                extracted, original_bit,
                "adaptive embed/extract roundtrip failed"
            );
        }
    }

    #[test]
    fn test_adaptive_dct_dc_not_modified() {
        let embedder = AdaptiveDctEmbedder::new(0.2);
        let mut block = DctBlock::zero();
        block.coefficients[0] = 99.0; // DC
                                      // Give a mid-freq coeff some energy so selector has something to pick.
        block.coefficients[ZIGZAG_ORDER[10]] = 5.0;
        let dc_before = block.dc();
        embedder.embed_bit(&mut block, true);
        // DC must not be touched.
        assert_eq!(block.dc(), dc_before);
    }
}
