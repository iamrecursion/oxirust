//! Dolby Vision metadata block parsing and generation.
//!
//! Provides frame-level PQ metadata blocks and extended display metadata for
//! Dolby Vision HDR streams.

#![allow(dead_code)]

// ── PQ conversion helper ──────────────────────────────────────────────────────

/// Convert a PQ code value (0–4095) to display nits.
///
/// Uses the simplified relationship: nits = 10 000 × (pq / 4095)^(1 / 0.1593).
#[allow(clippy::cast_precision_loss)]
fn pq_code_to_nits(pq: u16) -> f32 {
    let pq_norm = pq as f64 / 4095.0_f64;
    let nits = 10_000.0_f64 * pq_norm.powf(1.0 / 0.159_3_f64);
    nits as f32
}

// ── DvMetadataBlock ───────────────────────────────────────────────────────────

/// Frame-level Dolby Vision metadata block.
#[derive(Clone, Debug)]
pub struct DvMetadataBlock {
    /// Frame index (0-based) within the sequence.
    pub frame_number: u64,
    /// Source minimum PQ code value (0–4095).
    pub source_min_pq: u16,
    /// Source maximum PQ code value (0–4095).
    pub source_max_pq: u16,
    /// Target minimum PQ code value (0–4095).
    pub target_min_pq: u16,
    /// Target maximum PQ code value (0–4095).
    pub target_max_pq: u16,
}

impl DvMetadataBlock {
    /// Source minimum luminance in nits derived from the PQ code.
    #[must_use]
    pub fn source_nits(&self) -> f32 {
        pq_code_to_nits(self.source_min_pq)
    }

    /// Source maximum luminance in nits derived from the PQ code.
    #[must_use]
    pub fn target_nits(&self) -> f32 {
        pq_code_to_nits(self.target_max_pq)
    }
}

// ── ExtMetadata ───────────────────────────────────────────────────────────────

/// Extended display management metadata accompanying a Dolby Vision stream.
#[derive(Clone, Debug)]
pub struct ExtMetadata {
    /// Number of processing windows (typically 1).
    pub num_windows: u8,
    /// Source content peak luminance in nits.
    pub source_peak_nits: f32,
    /// Target display peak luminance capability in nits.
    pub target_display_peak_nits: f32,
}

impl ExtMetadata {
    /// Returns `true` when the source peak exceeds the target display capacity,
    /// indicating that tone mapping is necessary.
    #[must_use]
    pub fn needs_tone_mapping(&self) -> bool {
        self.source_peak_nits > self.target_display_peak_nits
    }
}

// ── MetadataBlockStream ───────────────────────────────────────────────────────

/// An ordered sequence of per-frame [`DvMetadataBlock`] entries.
#[derive(Default, Debug)]
pub struct MetadataBlockStream {
    /// All blocks in presentation order.
    pub blocks: Vec<DvMetadataBlock>,
}

impl MetadataBlockStream {
    /// Create an empty stream.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a metadata block to the stream.
    pub fn add(&mut self, block: DvMetadataBlock) {
        self.blocks.push(block);
    }

    /// Total number of frames (blocks) in the stream.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.blocks.len()
    }

    /// Arithmetic mean of `source_max_pq` converted to nits across all blocks.
    ///
    /// Returns `0.0` if the stream is empty.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn average_source_nits(&self) -> f32 {
        if self.blocks.is_empty() {
            return 0.0;
        }
        let sum: f32 = self
            .blocks
            .iter()
            .map(|b| pq_code_to_nits(b.source_max_pq))
            .sum();
        sum / self.blocks.len() as f32
    }

    /// Maximum `source_max_pq` value converted to nits across all blocks.
    ///
    /// Returns `0.0` if the stream is empty.
    #[must_use]
    pub fn peak_source_nits(&self) -> f32 {
        self.blocks
            .iter()
            .map(|b| pq_code_to_nits(b.source_max_pq))
            .fold(0.0_f32, f32::max)
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn block(
        frame: u64,
        src_min: u16,
        src_max: u16,
        tgt_min: u16,
        tgt_max: u16,
    ) -> DvMetadataBlock {
        DvMetadataBlock {
            frame_number: frame,
            source_min_pq: src_min,
            source_max_pq: src_max,
            target_min_pq: tgt_min,
            target_max_pq: tgt_max,
        }
    }

    #[test]
    fn test_pq_zero_is_zero_nits() {
        let b = block(0, 0, 0, 0, 0);
        assert_eq!(b.source_nits(), 0.0_f32);
    }

    #[test]
    fn test_pq_max_is_ten_thousand_nits() {
        let _b = block(0, 0, 4095, 0, 4095);
        let nits = pq_code_to_nits(4095);
        assert!((nits - 10_000.0).abs() < 0.1, "nits = {nits}");
    }

    #[test]
    fn test_source_nits_uses_source_min_pq() {
        let b = block(5, 100, 3000, 0, 2000);
        assert_eq!(b.source_nits(), pq_code_to_nits(100));
    }

    #[test]
    fn test_target_nits_uses_target_max_pq() {
        let b = block(5, 100, 3000, 0, 2000);
        assert_eq!(b.target_nits(), pq_code_to_nits(2000));
    }

    #[test]
    fn test_ext_metadata_needs_tone_mapping_true() {
        let ext = ExtMetadata {
            num_windows: 1,
            source_peak_nits: 4000.0,
            target_display_peak_nits: 1000.0,
        };
        assert!(ext.needs_tone_mapping());
    }

    #[test]
    fn test_ext_metadata_needs_tone_mapping_false() {
        let ext = ExtMetadata {
            num_windows: 1,
            source_peak_nits: 800.0,
            target_display_peak_nits: 1000.0,
        };
        assert!(!ext.needs_tone_mapping());
    }

    #[test]
    fn test_ext_metadata_equal_no_tone_mapping() {
        let ext = ExtMetadata {
            num_windows: 1,
            source_peak_nits: 1000.0,
            target_display_peak_nits: 1000.0,
        };
        assert!(!ext.needs_tone_mapping());
    }

    #[test]
    fn test_stream_empty_frame_count() {
        let s = MetadataBlockStream::new();
        assert_eq!(s.frame_count(), 0);
    }

    #[test]
    fn test_stream_add_and_count() {
        let mut s = MetadataBlockStream::new();
        s.add(block(0, 0, 2000, 0, 1500));
        s.add(block(1, 0, 3000, 0, 2000));
        assert_eq!(s.frame_count(), 2);
    }

    #[test]
    fn test_stream_average_source_nits_empty() {
        let s = MetadataBlockStream::new();
        assert_eq!(s.average_source_nits(), 0.0);
    }

    #[test]
    fn test_stream_average_source_nits_uniform() {
        let mut s = MetadataBlockStream::new();
        s.add(block(0, 0, 2048, 0, 2048));
        s.add(block(1, 0, 2048, 0, 2048));
        let avg = s.average_source_nits();
        let expected = pq_code_to_nits(2048);
        assert!(
            (avg - expected).abs() < 0.1,
            "avg={avg} expected={expected}"
        );
    }

    #[test]
    fn test_stream_peak_source_nits() {
        let mut s = MetadataBlockStream::new();
        s.add(block(0, 0, 1000, 0, 1000));
        s.add(block(1, 0, 3500, 0, 3000));
        s.add(block(2, 0, 2000, 0, 2000));
        let peak = s.peak_source_nits();
        let expected = pq_code_to_nits(3500);
        assert!(
            (peak - expected).abs() < 0.1,
            "peak={peak} expected={expected}"
        );
    }

    #[test]
    fn test_stream_peak_source_nits_empty() {
        let s = MetadataBlockStream::new();
        assert_eq!(s.peak_source_nits(), 0.0);
    }
}

// ── Spec-required types ───────────────────────────────────────────────────────

/// Raw Dolby Vision metadata block found in a bitstream.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct DvRawMetadataBlock {
    /// Byte offset from the start of the NAL unit where this block begins.
    pub block_start_offset: u32,
    /// Length of this block in bytes.
    pub block_length: u32,
    /// Block type identifier (0x19 = RPU NAL).
    pub block_type: u8,
}

impl DvRawMetadataBlock {
    /// Returns `true` when this block is an RPU NAL unit (type == 0x19).
    #[must_use]
    pub fn is_rpu(&self) -> bool {
        self.block_type == 0x19
    }
}

/// Dolby Vision Level 1 metadata (frame-level luminance statistics).
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct DvLevel1 {
    /// Minimum PQ code value (0–4095).
    pub min_pq: u16,
    /// Maximum PQ code value (0–4095).
    pub max_pq: u16,
    /// Average PQ code value (0–4095).
    pub avg_pq: u16,
}

impl DvLevel1 {
    /// Minimum luminance in nits derived from `min_pq`.
    #[must_use]
    pub fn min_nits(&self) -> f32 {
        pq_code_to_nits(self.min_pq)
    }

    /// Maximum luminance in nits derived from `max_pq`.
    #[must_use]
    pub fn max_nits(&self) -> f32 {
        pq_code_to_nits(self.max_pq)
    }
}

/// Dolby Vision Level 6 (HDR10 fallback) metadata.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct DvLevel6 {
    /// Maximum mastering display luminance (nits).
    pub max_display_mastering_luminance: u16,
    /// Minimum mastering display luminance (nits / 10 000).
    pub min_display_mastering_luminance: u16,
    /// Maximum content light level (nits).
    pub max_content_light_level: u16,
    /// Maximum frame average light level (nits).
    pub max_frame_average_light_level: u16,
}

impl DvLevel6 {
    /// Returns a human-readable HDR10 metadata summary string.
    #[must_use]
    pub fn hdr10_metadata_string(&self) -> String {
        format!(
            "MaxCLL={} MaxFALL={} MasteringDisplay={}-{}nits",
            self.max_content_light_level,
            self.max_frame_average_light_level,
            self.min_display_mastering_luminance,
            self.max_display_mastering_luminance,
        )
    }
}

/// Target display configuration for Dolby Vision rendering.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct DvDisplayConfig {
    /// Peak luminance the display can achieve (nits).
    pub peak_luminance_nits: f32,
    /// Minimum luminance the display can achieve (nits).
    pub min_luminance_nits: f32,
    /// Colour primaries description (e.g., "BT.2020").
    pub color_primaries: String,
}

impl DvDisplayConfig {
    /// Returns `true` if this is a typical consumer display (≤ 1 000 nits peak).
    #[must_use]
    pub fn is_consumer_display(&self) -> bool {
        self.peak_luminance_nits <= 1_000.0
    }
}

/// NAL-unit scanner that locates Dolby Vision RPU blocks (0x7C01 marker).
#[derive(Default, Debug)]
#[allow(dead_code)]
pub struct DvBlockParser {
    /// Detected blocks in scan order.
    pub blocks: Vec<DvRawMetadataBlock>,
}

impl DvBlockParser {
    /// Scan `data` for 0x7C01 RPU NAL markers and return a populated parser.
    #[must_use]
    pub fn parse_nal_units(data: &[u8]) -> Self {
        let mut blocks = Vec::new();
        let mut i = 0usize;
        while i + 1 < data.len() {
            if data[i] == 0x7C && data[i + 1] == 0x01 {
                blocks.push(DvRawMetadataBlock {
                    block_start_offset: i as u32,
                    block_length: 2,
                    block_type: 0x19,
                });
            }
            i += 1;
        }
        Self { blocks }
    }

    /// Total number of RPU blocks found.
    #[must_use]
    pub fn rpu_count(&self) -> usize {
        self.blocks.iter().filter(|b| b.is_rpu()).count()
    }
}

#[cfg(test)]
mod spec_tests {
    use super::*;

    #[test]
    fn test_dv_raw_block_is_rpu_true() {
        let b = DvRawMetadataBlock {
            block_start_offset: 0,
            block_length: 2,
            block_type: 0x19,
        };
        assert!(b.is_rpu());
    }

    #[test]
    fn test_dv_raw_block_is_rpu_false() {
        let b = DvRawMetadataBlock {
            block_start_offset: 0,
            block_length: 4,
            block_type: 0x01,
        };
        assert!(!b.is_rpu());
    }

    #[test]
    fn test_dv_level1_min_nits_zero() {
        let l1 = DvLevel1 {
            min_pq: 0,
            max_pq: 4095,
            avg_pq: 2000,
        };
        assert_eq!(l1.min_nits(), 0.0);
    }

    #[test]
    fn test_dv_level1_max_nits_full() {
        let l1 = DvLevel1 {
            min_pq: 0,
            max_pq: 4095,
            avg_pq: 2000,
        };
        assert!(
            (l1.max_nits() - 10_000.0).abs() < 1.0,
            "max_nits={}",
            l1.max_nits()
        );
    }

    #[test]
    fn test_dv_level6_hdr10_string_contains_maxcll() {
        let l6 = DvLevel6 {
            max_display_mastering_luminance: 4000,
            min_display_mastering_luminance: 50,
            max_content_light_level: 1200,
            max_frame_average_light_level: 400,
        };
        let s = l6.hdr10_metadata_string();
        assert!(s.contains("MaxCLL=1200"), "string={s}");
        assert!(s.contains("MaxFALL=400"), "string={s}");
    }

    #[test]
    fn test_dv_level6_hdr10_string_contains_mastering() {
        let l6 = DvLevel6 {
            max_display_mastering_luminance: 1000,
            min_display_mastering_luminance: 5,
            max_content_light_level: 800,
            max_frame_average_light_level: 200,
        };
        let s = l6.hdr10_metadata_string();
        assert!(s.contains("1000nits"), "string={s}");
    }

    #[test]
    fn test_dv_display_config_consumer_display_true() {
        let cfg = DvDisplayConfig {
            peak_luminance_nits: 1000.0,
            min_luminance_nits: 0.005,
            color_primaries: "BT.709".to_string(),
        };
        assert!(cfg.is_consumer_display());
    }

    #[test]
    fn test_dv_display_config_consumer_display_false() {
        let cfg = DvDisplayConfig {
            peak_luminance_nits: 4000.0,
            min_luminance_nits: 0.001,
            color_primaries: "BT.2020".to_string(),
        };
        assert!(!cfg.is_consumer_display());
    }

    #[test]
    fn test_dv_block_parser_no_markers() {
        let data = [0x00u8, 0x01, 0x02, 0x03];
        let p = DvBlockParser::parse_nal_units(&data);
        assert_eq!(p.rpu_count(), 0);
    }

    #[test]
    fn test_dv_block_parser_single_marker() {
        let data = [0x00u8, 0x7C, 0x01, 0xFF];
        let p = DvBlockParser::parse_nal_units(&data);
        assert_eq!(p.rpu_count(), 1);
        assert_eq!(p.blocks[0].block_start_offset, 1);
    }

    #[test]
    fn test_dv_block_parser_multiple_markers() {
        let data = [0x7Cu8, 0x01, 0x00, 0x7C, 0x01];
        let p = DvBlockParser::parse_nal_units(&data);
        assert_eq!(p.rpu_count(), 2);
    }

    #[test]
    fn test_dv_block_parser_empty_data() {
        let p = DvBlockParser::parse_nal_units(&[]);
        assert_eq!(p.rpu_count(), 0);
    }

    #[test]
    fn test_dv_level1_avg_pq_midrange() {
        let l1 = DvLevel1 {
            min_pq: 100,
            max_pq: 3000,
            avg_pq: 1500,
        };
        assert_eq!(l1.avg_pq, 1500);
    }
}
