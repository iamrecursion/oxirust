//! JPEG 2000 codestream marker *writers* (ISO/IEC 15444-1 §A) — the exact
//! forward counterpart of [`super::markers`].
//!
//! Emits the main-header markers SOC, SIZ, COD, QCD and the tile markers SOT,
//! SOD, EOC, producing a raw J2K codestream (`.j2k`). Each writer mirrors the
//! corresponding parser in [`super::markers`] field-for-field so that the
//! produced codestream parses back identically. All multi-byte integers are
//! big-endian.

use super::markers::{COD, EOC, QCD, SIZ, SOC, SOD, SOT};
use super::{Jp2Error, Jp2Result};

/// Per-component SIZ parameters for the encoder.
#[derive(Debug, Clone, Copy)]
pub struct ComponentSpec {
    /// Precision minus one, with bit 7 the sign flag (the `Ssiz` field).
    pub ssiz: u8,
    /// Horizontal subsampling factor.
    pub xr_siz: u8,
    /// Vertical subsampling factor.
    pub yr_siz: u8,
}

impl ComponentSpec {
    /// Build an unsigned component spec of the given bit depth (1..=38) with no
    /// subsampling.
    #[must_use]
    pub fn unsigned(bit_depth: u8) -> Self {
        Self {
            ssiz: bit_depth.saturating_sub(1) & 0x7F,
            xr_siz: 1,
            yr_siz: 1,
        }
    }
}

/// Append the SOC (start of codestream) marker.
pub fn write_soc(out: &mut Vec<u8>) {
    out.extend_from_slice(&SOC.to_be_bytes());
}

/// Append a SIZ (image and tile size) marker.
///
/// Mirrors [`super::markers`] `parse_siz`: 36 fixed bytes plus 3 bytes per
/// component, preceded by the marker code and the `Lsiz` length field.
pub fn write_siz(
    out: &mut Vec<u8>,
    width: u32,
    height: u32,
    tile_width: u32,
    tile_height: u32,
    components: &[ComponentSpec],
) -> Jp2Result<()> {
    if components.is_empty() {
        return Err(Jp2Error::InternalError(
            "SIZ requires at least one component".to_string(),
        ));
    }
    let csiz = u16::try_from(components.len())
        .map_err(|_| Jp2Error::InternalError("too many components for SIZ".to_string()))?;
    let lsiz_usize = 2 + 36 + 3 * components.len();
    let lsiz = u16::try_from(lsiz_usize)
        .map_err(|_| Jp2Error::InternalError("SIZ marker too long".to_string()))?;

    out.extend_from_slice(&SIZ.to_be_bytes());
    out.extend_from_slice(&lsiz.to_be_bytes());
    out.extend_from_slice(&0u16.to_be_bytes()); // Rsiz
    out.extend_from_slice(&width.to_be_bytes()); // Xsiz
    out.extend_from_slice(&height.to_be_bytes()); // Ysiz
    out.extend_from_slice(&0u32.to_be_bytes()); // XOsiz
    out.extend_from_slice(&0u32.to_be_bytes()); // YOsiz
    out.extend_from_slice(&tile_width.to_be_bytes()); // XTsiz
    out.extend_from_slice(&tile_height.to_be_bytes()); // YTsiz
    out.extend_from_slice(&0u32.to_be_bytes()); // XTOsiz
    out.extend_from_slice(&0u32.to_be_bytes()); // YTOsiz
    out.extend_from_slice(&csiz.to_be_bytes()); // Csiz
    for c in components {
        out.push(c.ssiz);
        out.push(c.xr_siz);
        out.push(c.yr_siz);
    }
    Ok(())
}

/// Append a COD (coding style default) marker for the lossless 5-3 path.
///
/// `num_decomp_levels` decomposition levels, code-block exponents `xcb`/`ycb`
/// (block size = `2^(exp+2)`), LRCP progression, one quality layer, no
/// multi-component transform, 5-3 reversible wavelet (filter = 1).
pub fn write_cod(out: &mut Vec<u8>, num_decomp_levels: u8, xcb: u8, ycb: u8) {
    write_cod_with_filter(out, num_decomp_levels, xcb, ycb, 1);
}

/// Append a COD (coding style default) marker for the lossy 9-7 path.
///
/// Identical to [`write_cod`] except for the wavelet filter byte: the CDF 9/7
/// irreversible filter is selected (transformation byte = 0, per
/// [`super::markers::CodMarker::is_irreversible_97`]).
pub fn write_cod_lossy(out: &mut Vec<u8>, num_decomp_levels: u8, xcb: u8, ycb: u8) {
    write_cod_with_filter(out, num_decomp_levels, xcb, ycb, 0);
}

fn write_cod_with_filter(
    out: &mut Vec<u8>,
    num_decomp_levels: u8,
    xcb: u8,
    ycb: u8,
    wavelet_filter: u8,
) {
    let lcod: u16 = 12; // 2 + 10
    out.extend_from_slice(&COD.to_be_bytes());
    out.extend_from_slice(&lcod.to_be_bytes());
    out.push(0); // Scod: no precincts / SOP / EPH
    out.push(0); // SGcod: progression order = 0 (LRCP)
    out.extend_from_slice(&1u16.to_be_bytes()); // SGcod: num_layers = 1
    out.push(0); // SGcod: MCT = 0
    out.push(num_decomp_levels); // SPcod: decomposition levels
    out.push(xcb); // SPcod: code-block width exponent
    out.push(ycb); // SPcod: code-block height exponent
    out.push(0); // SPcod: code-block style
    out.push(wavelet_filter); // SPcod: wavelet filter (1 = 5-3 / 0 = 9-7)
}

/// Append a QCD (quantization default) marker for the lossless path.
///
/// Quantization style 0 (no quantization). One `SPqcd` byte per subband
/// (`1 + 3 * num_decomp_levels`), each zero — matching `parse_qcd` for style 0.
pub fn write_qcd(out: &mut Vec<u8>, num_decomp_levels: u8) -> Jp2Result<()> {
    let num_subbands = 1usize + 3 * usize::from(num_decomp_levels);
    let lqcd_usize = 2 + 1 + num_subbands;
    let lqcd = u16::try_from(lqcd_usize)
        .map_err(|_| Jp2Error::InternalError("QCD marker too long".to_string()))?;
    out.extend_from_slice(&QCD.to_be_bytes());
    out.extend_from_slice(&lqcd.to_be_bytes());
    out.push(0); // Sqcd: guard bits = 0, style = 0 (no quantization)
    for _ in 0..num_subbands {
        out.push(0x00); // SPqcd: 8-bit exponent 0 per subband (lossless)
    }
    Ok(())
}

/// Append a QCD (quantization default) marker for the lossy 9-7 path
/// (Sqcd style 2, scalar expounded — one 16-bit `(ε, μ)` pair per subband).
///
/// `epsilon_mu` must contain exactly `1 + 3 * num_decomp_levels` entries in the
/// QCD subband order: `[LL_N, HL_N, LH_N, HH_N, HL_{N-1}, LH_{N-1}, HH_{N-1},
/// …]` (coarsest to finest), where `N = num_decomp_levels`. Each entry is
/// packed as `(ε << 11) | μ`; the decoder reads back `ε = (raw >> 11) & 0x1F`
/// and `μ = raw & 0x7FF` (see
/// [`super::markers::QcdMarker::step_size_for_subband`]).
///
/// `guard_bits` ∈ 0..=7 is stored in the top three bits of `Sqcd` (style 2 in
/// the low five bits). Callers in this crate pass `0`.
pub fn write_qcd_lossy(
    out: &mut Vec<u8>,
    num_decomp_levels: u8,
    epsilon_mu: &[(u8, u16)],
) -> Jp2Result<()> {
    let num_subbands = 1usize + 3 * usize::from(num_decomp_levels);
    if epsilon_mu.len() != num_subbands {
        return Err(Jp2Error::InternalError(format!(
            "write_qcd_lossy expects {num_subbands} (ε, μ) pairs, got {}",
            epsilon_mu.len()
        )));
    }
    let lqcd_usize = 2 + 1 + 2 * num_subbands;
    let lqcd = u16::try_from(lqcd_usize)
        .map_err(|_| Jp2Error::InternalError("QCD marker too long".to_string()))?;
    out.extend_from_slice(&QCD.to_be_bytes());
    out.extend_from_slice(&lqcd.to_be_bytes());
    // Sqcd: guard bits = 0 (top 3 bits), style = 2 (low 5 bits).
    out.push(0x02);
    for &(eps, mu) in epsilon_mu {
        if eps > 0x1F {
            return Err(Jp2Error::InternalError(format!(
                "QCD ε value {eps} out of range (0..=31)"
            )));
        }
        if mu > 0x07FF {
            return Err(Jp2Error::InternalError(format!(
                "QCD μ value {mu} out of range (0..=2047)"
            )));
        }
        let raw: u16 = (u16::from(eps) << 11) | (mu & 0x07FF);
        out.extend_from_slice(&raw.to_be_bytes());
    }
    Ok(())
}

/// Append a SOT (start of tile-part) marker.
///
/// `psot` is the tile-part length in bytes including the SOT segment, or 0 for
/// "unknown" (the decoder delimits the tile by the next SOT/EOC).
pub fn write_sot(out: &mut Vec<u8>, isot: u16, psot: u32, tpsot: u8, tnsot: u8) {
    let lsot: u16 = 10; // 2 + 8
    out.extend_from_slice(&SOT.to_be_bytes());
    out.extend_from_slice(&lsot.to_be_bytes());
    out.extend_from_slice(&isot.to_be_bytes());
    out.extend_from_slice(&psot.to_be_bytes());
    out.push(tpsot);
    out.push(tnsot);
}

/// Append the SOD (start of data) marker. Tile data follows immediately.
pub fn write_sod(out: &mut Vec<u8>) {
    out.extend_from_slice(&SOD.to_be_bytes());
}

/// Append the EOC (end of codestream) marker.
pub fn write_eoc(out: &mut Vec<u8>) {
    out.extend_from_slice(&EOC.to_be_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpeg2000::markers::{parse_codestream, MarkerSegment};

    #[test]
    fn write_and_parse_roundtrip() {
        let mut v = Vec::new();
        write_soc(&mut v);
        write_siz(&mut v, 32, 24, 32, 24, &[ComponentSpec::unsigned(8)]).expect("siz");
        write_cod(&mut v, 2, 2, 2);
        write_qcd(&mut v, 2).expect("qcd");
        write_sot(&mut v, 0, 0, 0, 1);
        write_sod(&mut v);
        v.push(0x00); // empty packet body
        write_eoc(&mut v);

        let segments = parse_codestream(&v).expect("parse");
        let siz = segments
            .iter()
            .find_map(|s| match s {
                MarkerSegment::Siz(z) => Some(z),
                _ => None,
            })
            .expect("SIZ");
        assert_eq!(siz.image_width(), 32);
        assert_eq!(siz.image_height(), 24);
        assert_eq!(siz.csiz, 1);
        assert_eq!(siz.components[0].bit_depth(), 8);
        assert!(!siz.components[0].is_signed());

        let cod = segments
            .iter()
            .find_map(|s| match s {
                MarkerSegment::Cod(c) => Some(c),
                _ => None,
            })
            .expect("COD");
        assert_eq!(cod.num_decomp_levels, 2);
        assert!(cod.is_lossless_wavelet());
        assert_eq!(cod.num_layers, 1);
        assert_eq!(cod.xcb, 2);
        assert_eq!(cod.ycb, 2);

        let qcd = segments
            .iter()
            .find_map(|s| match s {
                MarkerSegment::Qcd(q) => Some(q),
                _ => None,
            })
            .expect("QCD");
        assert_eq!(qcd.quant_style(), 0);

        assert!(segments.iter().any(|s| matches!(s, MarkerSegment::Eoc)));
    }

    #[test]
    fn lossy_cod_and_qcd_roundtrip() {
        let mut v = Vec::new();
        write_soc(&mut v);
        write_siz(&mut v, 32, 32, 32, 32, &[ComponentSpec::unsigned(8)]).expect("siz");
        write_cod_lossy(&mut v, 2, 3, 3);
        // ε=8, μ=0 for all (1 + 3*2) = 7 subbands.
        let pairs: Vec<(u8, u16)> = (0..7).map(|_| (8u8, 0u16)).collect();
        write_qcd_lossy(&mut v, 2, &pairs).expect("qcd lossy");
        write_sot(&mut v, 0, 0, 0, 1);
        write_sod(&mut v);
        v.push(0x00);
        write_eoc(&mut v);

        let segments = parse_codestream(&v).expect("parse");
        let cod = segments
            .iter()
            .find_map(|s| match s {
                MarkerSegment::Cod(c) => Some(c),
                _ => None,
            })
            .expect("COD");
        assert!(cod.is_irreversible_97(), "wavelet filter must be 9-7");
        assert!(!cod.is_lossless_wavelet());
        assert_eq!(cod.num_decomp_levels, 2);

        let qcd = segments
            .iter()
            .find_map(|s| match s {
                MarkerSegment::Qcd(q) => Some(q),
                _ => None,
            })
            .expect("QCD");
        assert_eq!(qcd.quant_style(), 2, "Sqcd style must be expounded");
        assert_eq!(qcd.guard_bits(), 0);
        assert_eq!(qcd.step_sizes.len(), 7);
        for &raw in &qcd.step_sizes {
            let eps = (raw >> 11) & 0x1F;
            let mu = raw & 0x07FF;
            assert_eq!(eps, 8);
            assert_eq!(mu, 0);
        }
    }

    #[test]
    fn write_qcd_lossy_rejects_wrong_length() {
        let mut v = Vec::new();
        let err = write_qcd_lossy(&mut v, 2, &[(8u8, 0u16); 6]);
        assert!(err.is_err());
    }

    #[test]
    fn write_qcd_lossy_rejects_out_of_range() {
        let mut v = Vec::new();
        // ε > 31 is invalid
        let err = write_qcd_lossy(
            &mut v,
            1,
            &[(32u8, 0u16), (8u8, 0u16), (8u8, 0u16), (8u8, 0u16)],
        );
        assert!(err.is_err());
        // μ > 2047 is invalid
        let err = write_qcd_lossy(
            &mut v,
            1,
            &[(8u8, 2048u16), (8u8, 0u16), (8u8, 0u16), (8u8, 0u16)],
        );
        assert!(err.is_err());
    }

    #[test]
    fn siz_multi_component() {
        let mut v = Vec::new();
        write_soc(&mut v);
        let comps = [ComponentSpec::unsigned(8); 3];
        write_siz(&mut v, 16, 16, 16, 16, &comps).expect("siz");
        write_cod(&mut v, 1, 2, 2);
        write_qcd(&mut v, 1).expect("qcd");
        write_sot(&mut v, 0, 0, 0, 1);
        write_sod(&mut v);
        v.push(0x00);
        write_eoc(&mut v);

        let segments = parse_codestream(&v).expect("parse");
        let siz = segments
            .iter()
            .find_map(|s| match s {
                MarkerSegment::Siz(z) => Some(z),
                _ => None,
            })
            .expect("SIZ");
        assert_eq!(siz.csiz, 3);
        assert_eq!(siz.components.len(), 3);
    }
}
