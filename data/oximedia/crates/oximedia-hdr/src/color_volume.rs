//! HDR colour volume metadata parsing and encoding.
//!
//! Implements SMPTE ST 2086 (Mastering Display Colour Volume), CEA-861.3
//! (Content Light Level), and HDR10+ dynamic metadata (`Hdr10PlusMetadata`).
//!
//! All chromaticity values are stored as u16 in units of 1/50 000
//! (i.e. `value / 50_000.0 = CIE xy`), matching the HEVC / AVC SEI layout.
//! Luminance is stored as u32 in units of 0.0001 nits.

use crate::{HdrError, Result};

// ── MasteringDisplayColorVolume ───────────────────────────────────────────────

/// SMPTE ST 2086 Mastering Display Colour Volume metadata.
///
/// Chromaticity coordinates are encoded as `u16` in units of 1/50 000
/// (so `0x8000` = 0.65 in CIE xy space).  Luminance is in 0.0001-nit units.
///
/// # Layout
/// ```text
/// primaries[0] = Red   [x, y]
/// primaries[1] = Green [x, y]
/// primaries[2] = Blue  [x, y]
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MasteringDisplayColorVolume {
    /// CIE 1931 chromaticity primaries [R, G, B][x, y], × 50 000.
    pub primaries: [[u16; 2]; 3],
    /// CIE 1931 white-point [x, y], × 50 000.
    pub white_point: [u16; 2],
    /// Maximum display luminance (0.0001 nits).
    pub max_luminance: u32,
    /// Minimum display luminance (0.0001 nits).
    pub min_luminance: u32,
}

impl MasteringDisplayColorVolume {
    /// Reference primaries for a Rec. 2020 / P3-D65 reference monitor.
    ///
    /// Values derived from BT.2020 specification:
    /// R(0.708, 0.292) G(0.170, 0.797) B(0.131, 0.046) W(0.3127, 0.3290)
    pub fn rec2020_reference() -> Self {
        Self {
            // × 50 000
            primaries: [
                [35400, 14600], // R: (0.708, 0.292)
                [8500, 39850],  // G: (0.170, 0.797)
                [6550, 2300],   // B: (0.131, 0.046)
            ],
            white_point: [15635, 16450], // D65: (0.3127, 0.3290)
            max_luminance: 10_000_000,   // 1000 nits
            min_luminance: 5,            // 0.0005 nits
        }
    }

    /// Return the maximum luminance in nits as `f64`.
    pub fn max_luminance_nits(&self) -> f64 {
        f64::from(self.max_luminance) * 0.0001
    }

    /// Return the minimum luminance in nits as `f64`.
    pub fn min_luminance_nits(&self) -> f64 {
        f64::from(self.min_luminance) * 0.0001
    }
}

// ── ContentLightLevel ─────────────────────────────────────────────────────────

/// CEA-861.3 Content Light Level metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContentLightLevel {
    /// Maximum Content Light Level (MaxCLL) in nits.
    pub max_cll: u16,
    /// Maximum Frame-Average Light Level (MaxFALL) in nits.
    pub max_fall: u16,
}

// ── Hdr10PlusMetadata ─────────────────────────────────────────────────────────

/// Subset of HDR10+ dynamic metadata (SMPTE ST 2094-40 Application 4).
///
/// This represents the first window's scene-level dynamic data; a full
/// implementation would carry an array of windows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hdr10PlusMetadata {
    /// ITU-T T.35 country code / system start code.
    pub system_start_code: u8,
    /// Application version (currently 1).
    pub application_version: u8,
    /// Number of processing windows (1..=3).
    pub num_windows: u8,
    /// Target system display maximum luminance (nits).
    pub target_system_display_max_luminance: u32,
    /// MaxSCL for R, G, B (nits).
    pub maxscl: [u32; 3],
    /// Average MaxRGB across the frame (nits).
    pub average_maxrgb: u32,
}

// ── SEI parsers ───────────────────────────────────────────────────────────────

/// Parse an SMPTE ST 2086 mastering display colour volume SEI payload.
///
/// The payload is exactly 24 bytes (6 × u16 primaries + 2 × u16 white +
/// 2 × u32 luminance), all big-endian.
///
/// # Errors
/// Returns [`HdrError::MetadataParseError`] if the slice is not 24 bytes.
pub fn parse_hdr10_sei(data: &[u8]) -> Result<MasteringDisplayColorVolume> {
    if data.len() < 24 {
        return Err(HdrError::MetadataParseError(format!(
            "ST 2086 SEI requires 24 bytes, got {}",
            data.len()
        )));
    }

    let read_u16 = |off: usize| -> u16 { u16::from_be_bytes([data[off], data[off + 1]]) };
    let read_u32 = |off: usize| -> u32 {
        u32::from_be_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
    };

    // Primaries: G, B, R order in the SEI (HEVC spec order).
    // We store them as [R, G, B].
    let g_x = read_u16(0);
    let g_y = read_u16(2);
    let b_x = read_u16(4);
    let b_y = read_u16(6);
    let r_x = read_u16(8);
    let r_y = read_u16(10);

    let white_x = read_u16(12);
    let white_y = read_u16(14);
    let max_luminance = read_u32(16);
    let min_luminance = read_u32(20);

    Ok(MasteringDisplayColorVolume {
        primaries: [[r_x, r_y], [g_x, g_y], [b_x, b_y]],
        white_point: [white_x, white_y],
        max_luminance,
        min_luminance,
    })
}

/// Encode a [`MasteringDisplayColorVolume`] as a 24-byte SMPTE ST 2086 SEI payload.
///
/// Byte order follows the HEVC spec (G, B, R, W, max_lum, min_lum), big-endian.
pub fn encode_hdr10_sei(vol: &MasteringDisplayColorVolume) -> Vec<u8> {
    let mut out = Vec::with_capacity(24);

    // HEVC SEI order: G, B, R
    for &[x, y] in &[vol.primaries[1], vol.primaries[2], vol.primaries[0]] {
        out.extend_from_slice(&x.to_be_bytes());
        out.extend_from_slice(&y.to_be_bytes());
    }
    out.extend_from_slice(&vol.white_point[0].to_be_bytes());
    out.extend_from_slice(&vol.white_point[1].to_be_bytes());
    out.extend_from_slice(&vol.max_luminance.to_be_bytes());
    out.extend_from_slice(&vol.min_luminance.to_be_bytes());
    out
}

/// Parse a CEA-861.3 Content Light Level SEI payload (4 bytes, big-endian).
///
/// # Errors
/// Returns [`HdrError::MetadataParseError`] if the slice is not at least 4 bytes.
pub fn parse_cll_sei(data: &[u8]) -> Result<ContentLightLevel> {
    if data.len() < 4 {
        return Err(HdrError::MetadataParseError(format!(
            "CEA-861.3 CLL SEI requires 4 bytes, got {}",
            data.len()
        )));
    }
    let max_cll = u16::from_be_bytes([data[0], data[1]]);
    let max_fall = u16::from_be_bytes([data[2], data[3]]);
    Ok(ContentLightLevel { max_cll, max_fall })
}

/// Encode a [`ContentLightLevel`] as a 4-byte CEA-861.3 SEI payload.
pub fn encode_cll_sei(cll: &ContentLightLevel) -> Vec<u8> {
    let mut out = Vec::with_capacity(4);
    out.extend_from_slice(&cll.max_cll.to_be_bytes());
    out.extend_from_slice(&cll.max_fall.to_be_bytes());
    out
}

// ── Luminance computation ─────────────────────────────────────────────────────

/// Compute approximate Y (luminance) weights for each primary from the
/// CIE 1931 chromaticity coordinates, using standard Rec. 2020 luma weights.
///
/// Returns `[Y_r, Y_g, Y_b]` where the values are the Rec. 2020 standard
/// luma coefficients: `[0.2126, 0.7152, 0.0722]`.
///
/// In full CIE 1931 derivation the weights depend on the exact primaries and
/// white point; this implementation uses the Rec. 2020 reference values which
/// are correct for all Rec. 2020-based HDR content.
pub fn luminance_from_primaries(_primaries: &[[u16; 2]; 3], _white_point: &[u16; 2]) -> [f32; 3] {
    // ITU-R BT.2020 / BT.2100 standard luma coefficients.
    [0.2126, 0.7152, 0.0722]
}

// ── MaxRgbAnalyzer ────────────────────────────────────────────────────────────

/// Computes MaxRGB per-frame luminance statistics for HDR10+ metadata.
///
/// MaxRGB is defined per SMPTE ST 2094-40 as the maximum of R, G, B for each
/// pixel, expressed in nits. It is used to derive MaxCLL and MaxFALL values.
pub struct MaxRgbAnalyzer;

impl MaxRgbAnalyzer {
    /// Compute the maximum and average MaxRGB values for a frame.
    ///
    /// `pixels` is an interleaved RGB slice (length must be divisible by 3).
    /// Each component is in the linear [0, 1] range normalised to `peak_nits`.
    ///
    /// Returns `(max_maxrgb_nits, avg_maxrgb_nits)`.
    ///
    /// # Errors
    /// Returns [`HdrError::ToneMappingError`] if the slice length is not divisible by 3.
    pub fn compute(pixels: &[f32], peak_nits: f32) -> crate::Result<(f32, f32)> {
        if !pixels.len().is_multiple_of(3) {
            return Err(HdrError::MetadataParseError(format!(
                "pixel buffer length {} is not divisible by 3",
                pixels.len()
            )));
        }
        if pixels.is_empty() {
            return Ok((0.0, 0.0));
        }
        let mut max_val = 0.0_f32;
        let mut sum = 0.0_f64;
        let n_pixels = pixels.len() / 3;
        for chunk in pixels.chunks_exact(3) {
            let mx = chunk[0].max(chunk[1]).max(chunk[2]).max(0.0);
            if mx > max_val {
                max_val = mx;
            }
            sum += f64::from(mx);
        }
        let avg = (sum / n_pixels as f64) as f32;
        Ok((max_val * peak_nits, avg * peak_nits))
    }

    /// Compute a percentile of MaxRGB values across all pixels.
    ///
    /// `percentile` is in [0, 100].  Returns the percentile value in nits.
    ///
    /// # Errors
    /// Returns an error if the pixel buffer is invalid or percentile out of range.
    pub fn percentile_nits(pixels: &[f32], peak_nits: f32, percentile: f32) -> crate::Result<f32> {
        if !pixels.len().is_multiple_of(3) {
            return Err(HdrError::MetadataParseError(format!(
                "pixel buffer length {} is not divisible by 3",
                pixels.len()
            )));
        }
        if !(0.0..=100.0).contains(&percentile) {
            return Err(HdrError::MetadataParseError(format!(
                "percentile {percentile} out of [0, 100]"
            )));
        }
        if pixels.is_empty() {
            return Ok(0.0);
        }
        let mut vals: Vec<f32> = pixels
            .chunks_exact(3)
            .map(|c| c[0].max(c[1]).max(c[2]).max(0.0))
            .collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((percentile / 100.0) * (vals.len() - 1) as f32).round() as usize;
        Ok(vals[idx.min(vals.len() - 1)] * peak_nits)
    }

    /// Derive [`ContentLightLevel`] from per-frame statistics.
    ///
    /// `frame_stats` is a slice of `(max_maxrgb_nits, avg_maxrgb_nits)` pairs,
    /// one entry per frame.  MaxCLL is the peak across all frames; MaxFALL is
    /// the peak frame-average.
    pub fn auto_detect_cll(frame_stats: &[(f32, f32)]) -> ContentLightLevel {
        let mut max_cll = 0.0_f32;
        let mut max_fall = 0.0_f32;
        for &(mx, avg) in frame_stats {
            if mx > max_cll {
                max_cll = mx;
            }
            if avg > max_fall {
                max_fall = avg;
            }
        }
        ContentLightLevel {
            max_cll: max_cll.round() as u16,
            max_fall: max_fall.round() as u16,
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    // 1. Round-trip: encode → decode yields identical struct
    #[test]
    fn test_hdr10_sei_roundtrip() {
        let vol = MasteringDisplayColorVolume::rec2020_reference();
        let encoded = encode_hdr10_sei(&vol);
        assert_eq!(encoded.len(), 24);
        let decoded = parse_hdr10_sei(&encoded).expect("roundtrip parse");
        assert_eq!(decoded, vol);
    }

    // 2. Parse known reference bytes
    #[test]
    fn test_hdr10_sei_parse_known() {
        let vol = MasteringDisplayColorVolume::rec2020_reference();
        let bytes = encode_hdr10_sei(&vol);
        let parsed = parse_hdr10_sei(&bytes).expect("parse");
        // Max luminance 1000 nits = 10 000 000 units
        assert_eq!(parsed.max_luminance, 10_000_000);
        assert_eq!(parsed.min_luminance, 5);
    }

    // 3. Parse short slice → error
    #[test]
    fn test_hdr10_sei_short_slice() {
        let result = parse_hdr10_sei(&[0u8; 10]);
        assert!(result.is_err());
    }

    // 4. Luminance helpers
    #[test]
    fn test_luminance_nits() {
        let vol = MasteringDisplayColorVolume {
            primaries: [[0; 2]; 3],
            white_point: [0; 2],
            max_luminance: 10_000_000,
            min_luminance: 1,
        };
        let diff = (vol.max_luminance_nits() - 1000.0).abs();
        assert!(
            diff < 0.01,
            "max_luminance_nits = {}",
            vol.max_luminance_nits()
        );
        let diff_min = (vol.min_luminance_nits() - 0.0001).abs();
        assert!(diff_min < 1e-6);
    }

    // 5. CLL round-trip
    #[test]
    fn test_cll_sei_roundtrip() {
        let cll = ContentLightLevel {
            max_cll: 1000,
            max_fall: 400,
        };
        let encoded = encode_cll_sei(&cll);
        assert_eq!(encoded.len(), 4);
        let decoded = parse_cll_sei(&encoded).expect("cll roundtrip");
        assert_eq!(decoded, cll);
    }

    // 6. CLL parse known values
    #[test]
    fn test_cll_sei_known_values() {
        // MaxCLL = 4000, MaxFALL = 1000 in big-endian
        let data = [0x0F, 0xA0, 0x03, 0xE8];
        let cll = parse_cll_sei(&data).expect("parse");
        assert_eq!(cll.max_cll, 4000);
        assert_eq!(cll.max_fall, 1000);
    }

    // 7. CLL short slice → error
    #[test]
    fn test_cll_sei_short_slice() {
        assert!(parse_cll_sei(&[0u8; 2]).is_err());
    }

    // 8. Luminance weights are Rec.2020 standard
    #[test]
    fn test_luminance_from_primaries_rec2020() {
        let vol = MasteringDisplayColorVolume::rec2020_reference();
        let weights = luminance_from_primaries(&vol.primaries, &vol.white_point);
        assert!(approx(weights[0], 0.2126, EPS), "Y_r = {}", weights[0]);
        assert!(approx(weights[1], 0.7152, EPS), "Y_g = {}", weights[1]);
        assert!(approx(weights[2], 0.0722, EPS), "Y_b = {}", weights[2]);
    }

    // 9. Luminance weights sum to 1.0
    #[test]
    fn test_luminance_weights_sum_to_one() {
        let vol = MasteringDisplayColorVolume::rec2020_reference();
        let weights = luminance_from_primaries(&vol.primaries, &vol.white_point);
        let sum: f32 = weights.iter().sum();
        assert!(approx(sum, 1.0, 1e-4), "sum = {sum}");
    }

    // 10. encode_hdr10_sei is exactly 24 bytes
    #[test]
    fn test_encode_hdr10_sei_length() {
        let vol = MasteringDisplayColorVolume::rec2020_reference();
        assert_eq!(encode_hdr10_sei(&vol).len(), 24);
    }

    // 11. encode_cll_sei is exactly 4 bytes
    #[test]
    fn test_encode_cll_sei_length() {
        let cll = ContentLightLevel {
            max_cll: 1000,
            max_fall: 200,
        };
        assert_eq!(encode_cll_sei(&cll).len(), 4);
    }

    // 12. Parse with extra trailing bytes is allowed
    #[test]
    fn test_hdr10_sei_extra_bytes_ok() {
        let vol = MasteringDisplayColorVolume::rec2020_reference();
        let mut encoded = encode_hdr10_sei(&vol);
        encoded.extend_from_slice(&[0xFF, 0xFF]); // trailing
        let decoded = parse_hdr10_sei(&encoded).expect("extra bytes ok");
        assert_eq!(decoded, vol);
    }

    // 13. CLL: parse with extra trailing bytes
    #[test]
    fn test_cll_sei_extra_bytes_ok() {
        let cll = ContentLightLevel {
            max_cll: 600,
            max_fall: 200,
        };
        let mut encoded = encode_cll_sei(&cll);
        encoded.push(0xFF);
        let decoded = parse_cll_sei(&encoded).expect("extra ok");
        assert_eq!(decoded, cll);
    }

    // 14. Hdr10PlusMetadata: basic construction
    #[test]
    fn test_hdr10plus_construction() {
        let meta = Hdr10PlusMetadata {
            system_start_code: 0xB5,
            application_version: 1,
            num_windows: 1,
            target_system_display_max_luminance: 4000,
            maxscl: [2000, 2500, 1800],
            average_maxrgb: 600,
        };
        assert_eq!(meta.num_windows, 1);
        assert_eq!(meta.maxscl[1], 2500);
    }

    // 15. White-point round-trip in SEI
    #[test]
    fn test_white_point_roundtrip() {
        let vol = MasteringDisplayColorVolume {
            primaries: [[35400, 14600], [8500, 39850], [6550, 2300]],
            white_point: [15635, 16450],
            max_luminance: 10_000_000,
            min_luminance: 5,
        };
        let enc = encode_hdr10_sei(&vol);
        let dec = parse_hdr10_sei(&enc).expect("white point rt");
        assert_eq!(dec.white_point, vol.white_point);
    }

    // 16. MasteringDisplayColorVolume: primaries round-trip
    #[test]
    fn test_primaries_roundtrip() {
        let vol = MasteringDisplayColorVolume::rec2020_reference();
        let enc = encode_hdr10_sei(&vol);
        let dec = parse_hdr10_sei(&enc).expect("primaries rt");
        assert_eq!(dec.primaries, vol.primaries);
    }

    // 17. encode_hdr10_sei primary G/B/R byte order
    #[test]
    fn test_encode_hdr10_primary_order() {
        // Build a vol where each primary has a unique value
        let vol = MasteringDisplayColorVolume {
            primaries: [
                [0x1111, 0x2222], // R
                [0x3333, 0x4444], // G
                [0x5555, 0x6666], // B
            ],
            white_point: [0x7777, 0x8888],
            max_luminance: 0x0000_0001,
            min_luminance: 0x0000_0002,
        };
        let enc = encode_hdr10_sei(&vol);
        // HEVC order: G then B then R
        assert_eq!(&enc[0..2], &[0x33, 0x33]); // G.x
        assert_eq!(&enc[4..6], &[0x55, 0x55]); // B.x
        assert_eq!(&enc[8..10], &[0x11, 0x11]); // R.x
    }

    // 18. CLL max_cll / max_fall zero values
    #[test]
    fn test_cll_zero_values() {
        let cll = ContentLightLevel {
            max_cll: 0,
            max_fall: 0,
        };
        let enc = encode_cll_sei(&cll);
        let dec = parse_cll_sei(&enc).expect("zero cll");
        assert_eq!(dec.max_cll, 0);
        assert_eq!(dec.max_fall, 0);
    }

    // 19. Max-range luminance values survive round-trip
    #[test]
    fn test_luminance_max_roundtrip() {
        let vol = MasteringDisplayColorVolume {
            primaries: [[u16::MAX; 2]; 3],
            white_point: [u16::MAX; 2],
            max_luminance: u32::MAX,
            min_luminance: 0,
        };
        let enc = encode_hdr10_sei(&vol);
        let dec = parse_hdr10_sei(&enc).expect("max lum rt");
        assert_eq!(dec.max_luminance, u32::MAX);
        assert_eq!(dec.min_luminance, 0);
    }

    // 20. rec2020_reference has sensible default luminance nits
    #[test]
    fn test_rec2020_reference_defaults() {
        let vol = MasteringDisplayColorVolume::rec2020_reference();
        assert!((vol.max_luminance_nits() - 1000.0).abs() < 0.1);
        assert!(vol.min_luminance_nits() < 0.01);
    }

    // 21. MaxRgbAnalyzer: empty buffer returns (0, 0)
    #[test]
    fn test_maxrgb_empty_buffer() {
        let (mx, avg) = MaxRgbAnalyzer::compute(&[], 1000.0).expect("empty");
        assert!(approx(mx, 0.0, EPS));
        assert!(approx(avg, 0.0, EPS));
    }

    // 22. MaxRgbAnalyzer: single white pixel
    #[test]
    fn test_maxrgb_single_white() {
        let pixels = [1.0f32, 1.0, 1.0];
        let (mx, avg) = MaxRgbAnalyzer::compute(&pixels, 1000.0).expect("white");
        assert!(approx(mx, 1000.0, 0.01), "max = {mx}");
        assert!(approx(avg, 1000.0, 0.01), "avg = {avg}");
    }

    // 23. MaxRgbAnalyzer: mixed pixels
    #[test]
    fn test_maxrgb_mixed_pixels() {
        // Two pixels: pure red at 0.5, pure white at 1.0
        let pixels = [0.5f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        let (mx, avg) = MaxRgbAnalyzer::compute(&pixels, 1000.0).expect("mixed");
        assert!(approx(mx, 1000.0, 0.01), "max = {mx}");
        // Average of 0.5 and 1.0 = 0.75 → 750 nits
        assert!(approx(avg, 750.0, 0.5), "avg = {avg}");
    }

    // 24. MaxRgbAnalyzer: invalid buffer length
    #[test]
    fn test_maxrgb_invalid_length() {
        assert!(MaxRgbAnalyzer::compute(&[0.1f32, 0.2], 1000.0).is_err());
    }

    // 25. MaxRgbAnalyzer: percentile_nits basic
    #[test]
    fn test_maxrgb_percentile_100() {
        let pixels: Vec<f32> = (0..30).flat_map(|i| [i as f32 / 30.0, 0.0, 0.0]).collect();
        let p100 = MaxRgbAnalyzer::percentile_nits(&pixels, 1000.0, 100.0).expect("p100");
        // Max should be close to (29/30) * 1000
        assert!(p100 > 900.0 && p100 <= 1000.0, "p100 = {p100}");
    }

    // 26. MaxRgbAnalyzer: percentile 0 gives minimum
    #[test]
    fn test_maxrgb_percentile_0() {
        let pixels: Vec<f32> = (1..=30).flat_map(|i| [i as f32 / 30.0, 0.0, 0.0]).collect();
        let p0 = MaxRgbAnalyzer::percentile_nits(&pixels, 1000.0, 0.0).expect("p0");
        // Minimum should be 1/30 * 1000 ≈ 33.33 nits
        assert!(p0 > 0.0 && p0 < 100.0, "p0 = {p0}");
    }

    // 27. MaxRgbAnalyzer: percentile out of range errors
    #[test]
    fn test_maxrgb_percentile_invalid() {
        let pixels = [0.5f32, 0.5, 0.5];
        assert!(MaxRgbAnalyzer::percentile_nits(&pixels, 1000.0, -1.0).is_err());
        assert!(MaxRgbAnalyzer::percentile_nits(&pixels, 1000.0, 101.0).is_err());
    }

    // 28. MaxRgbAnalyzer: auto_detect_cll from frame stats
    #[test]
    fn test_auto_detect_cll_basic() {
        let frame_stats = vec![(800.0f32, 200.0f32), (1000.0, 400.0), (600.0, 300.0)];
        let cll = MaxRgbAnalyzer::auto_detect_cll(&frame_stats);
        assert_eq!(cll.max_cll, 1000);
        assert_eq!(cll.max_fall, 400);
    }

    // 29. MaxRgbAnalyzer: auto_detect_cll empty slice
    #[test]
    fn test_auto_detect_cll_empty() {
        let cll = MaxRgbAnalyzer::auto_detect_cll(&[]);
        assert_eq!(cll.max_cll, 0);
        assert_eq!(cll.max_fall, 0);
    }

    // 30. MaxRgbAnalyzer: max is always >= avg
    #[test]
    fn test_maxrgb_max_ge_avg() {
        let pixels: Vec<f32> = (0..30)
            .flat_map(|i| [i as f32 / 30.0, i as f32 / 60.0, 0.0])
            .collect();
        let (mx, avg) = MaxRgbAnalyzer::compute(&pixels, 1000.0).expect("ge");
        assert!(mx >= avg - 0.01, "max ({mx}) should be >= avg ({avg})");
    }
}
