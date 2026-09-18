//! JPEG-LS marker writers (ISO 14495-1 Annex C).
//!
//! These functions emit the framing markers consumed by
//! [`super::markers::parse_headers`]. The byte layout mirrors the parsers in
//! [`super::markers`] exactly, so a header written here parses back to an
//! equal [`JlsHeaders`](super::markers::JlsHeaders).

use super::markers::{ComponentSpec, JlsPresetParams, EOI, LSE, SOF55, SOI, SOS};

/// One component descriptor as written into the SOF55 frame header.
#[derive(Debug, Clone, Copy)]
pub struct FrameComponent {
    /// Component identifier byte (Ci).
    pub id: u8,
    /// Horizontal sampling factor (Hi); JPEG-LS uses 1.
    pub h_factor: u8,
    /// Vertical sampling factor (Vi); JPEG-LS uses 1.
    pub v_factor: u8,
    /// Quantisation-table selector (Tqi); always 0 for JPEG-LS.
    pub quant_table_idx: u8,
}

impl FrameComponent {
    /// Construct a standard JPEG-LS component descriptor (`Hi = Vi = 1`, `Tqi = 0`).
    #[must_use]
    pub fn standard(id: u8) -> Self {
        Self {
            id,
            h_factor: 1,
            v_factor: 1,
            quant_table_idx: 0,
        }
    }
}

impl From<&ComponentSpec> for FrameComponent {
    fn from(spec: &ComponentSpec) -> Self {
        Self {
            id: spec.id,
            h_factor: spec.h_factor,
            v_factor: spec.v_factor,
            quant_table_idx: spec.quant_table_idx,
        }
    }
}

/// Append the SOI (Start Of Image) marker.
pub fn write_soi(out: &mut Vec<u8>) {
    out.extend_from_slice(&SOI.to_be_bytes());
}

/// Append the EOI (End Of Image) marker.
pub fn write_eoi(out: &mut Vec<u8>) {
    out.extend_from_slice(&EOI.to_be_bytes());
}

/// Append a SOF55 (JPEG-LS Start Of Frame) marker.
///
/// Layout (after the 2-byte marker): `Lf` (u16), `P` (u8), `Y` (u16),
/// `X` (u16), `Nf` (u8), then for each component `Ci` (u8),
/// `Hi<<4 | Vi` (u8), `Tqi` (u8). `Lf` counts itself, so
/// `Lf = 8 + 3 * Nf`.
pub fn write_sof55(
    out: &mut Vec<u8>,
    precision: u8,
    height: u16,
    width: u16,
    components: &[FrameComponent],
) {
    let nf = components.len() as u16;
    let seg_len: u16 = 8 + 3 * nf;

    out.extend_from_slice(&SOF55.to_be_bytes());
    out.extend_from_slice(&seg_len.to_be_bytes());
    out.push(precision);
    out.extend_from_slice(&height.to_be_bytes());
    out.extend_from_slice(&width.to_be_bytes());
    out.push(components.len() as u8);
    for comp in components {
        out.push(comp.id);
        out.push((comp.h_factor << 4) | (comp.v_factor & 0x0F));
        out.push(comp.quant_table_idx);
    }
}

/// Append an LSE (JPEG-LS preset parameters) marker with `Id = 1`.
///
/// Layout (after the 2-byte marker): `Ll` (u16), `Id = 1` (u8),
/// `MaxVal` (u16), `T1` (u16), `T2` (u16), `T3` (u16), `Reset` (u16).
/// `Ll` counts itself, so `Ll = 13`.
///
/// The decoder only honours `Id == 1`; emit this when non-default preset
/// thresholds are required (e.g. custom T1/T2/T3 or MaxVal). When the presets
/// equal [`JlsPresetParams::default_for_precision`] this marker may be omitted.
pub fn write_lse_preset(out: &mut Vec<u8>, presets: &JlsPresetParams) {
    // Ll = 2 (length) + 1 (Id) + 2*5 (MaxVal,T1,T2,T3,Reset) = 13.
    let seg_len: u16 = 13;
    out.extend_from_slice(&LSE.to_be_bytes());
    out.extend_from_slice(&seg_len.to_be_bytes());
    out.push(1u8); // Id = 1 (preset coding parameters)
    out.extend_from_slice(&presets.max_val.to_be_bytes());
    out.extend_from_slice(&(presets.t1 as u16).to_be_bytes());
    out.extend_from_slice(&(presets.t2 as u16).to_be_bytes());
    out.extend_from_slice(&(presets.t3 as u16).to_be_bytes());
    out.extend_from_slice(&presets.reset.to_be_bytes());
}

/// Append a SOS (Start Of Scan) marker.
///
/// Layout (after the 2-byte marker): `Ls` (u16), `Ns` (u8), then for each
/// scan component `Csj` (u8) and a `Tdj<<4 | Taj` byte (0 for JPEG-LS),
/// followed by `NEAR` (u8), `ILV` (u8), `Pt` (u8). `Ls` counts itself, so
/// `Ls = 6 + 2 * Ns`.
pub fn write_sos(out: &mut Vec<u8>, component_ids: &[u8], near: u8, ilv: u8, point_transform: u8) {
    let ns = component_ids.len() as u16;
    let seg_len: u16 = 6 + 2 * ns;

    out.extend_from_slice(&SOS.to_be_bytes());
    out.extend_from_slice(&seg_len.to_be_bytes());
    out.push(component_ids.len() as u8);
    for &cs in component_ids {
        out.push(cs);
        out.push(0u8); // Td = 0, Ta = 0 (unused by JPEG-LS)
    }
    out.push(near);
    out.push(ilv);
    out.push(point_transform);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpegls::markers::{parse_headers, JlsPresetParams};

    #[test]
    fn sof55_sos_roundtrip_greyscale() {
        let mut out = Vec::new();
        write_soi(&mut out);
        write_sof55(&mut out, 8, 24, 32, &[FrameComponent::standard(1)]);
        write_sos(&mut out, &[1], 0, 0, 0);
        // A trailing byte so the marker walk does not hit EOI/truncation early.
        out.push(0x00);

        let headers = parse_headers(&out).expect("headers parse");
        assert_eq!(headers.frame.precision, 8);
        assert_eq!(headers.frame.height, 24);
        assert_eq!(headers.frame.width, 32);
        assert_eq!(headers.frame.num_components, 1);
        assert_eq!(headers.frame.components.len(), 1);
        assert_eq!(headers.frame.components[0].id, 1);
        assert_eq!(headers.frame.components[0].h_factor, 1);
        assert_eq!(headers.frame.components[0].v_factor, 1);
        assert_eq!(headers.scan.near, 0);
        assert_eq!(headers.scan.ilv, 0);
        assert_eq!(headers.scan.pt, 0);
        assert_eq!(headers.scan.component_ids, vec![1]);
    }

    #[test]
    fn sof55_sos_roundtrip_rgb_ilv2_near() {
        let comps = [
            FrameComponent::standard(1),
            FrameComponent::standard(2),
            FrameComponent::standard(3),
        ];
        let mut out = Vec::new();
        write_soi(&mut out);
        write_sof55(&mut out, 8, 4, 4, &comps);
        write_sos(&mut out, &[1, 2, 3], 2, 2, 0);
        out.push(0x00);

        let headers = parse_headers(&out).expect("headers parse");
        assert_eq!(headers.frame.num_components, 3);
        assert_eq!(headers.frame.components.len(), 3);
        assert_eq!(headers.scan.near, 2);
        assert_eq!(headers.scan.ilv, 2);
        assert_eq!(headers.scan.component_ids, vec![1, 2, 3]);
    }

    #[test]
    fn lse_preset_roundtrip() {
        // Use non-default presets to verify the LSE marker is honoured.
        let presets = JlsPresetParams {
            max_val: 200,
            t1: 5,
            t2: 17,
            t3: 60,
            reset: 48,
        };
        let mut out = Vec::new();
        write_soi(&mut out);
        write_sof55(&mut out, 8, 4, 4, &[FrameComponent::standard(1)]);
        write_lse_preset(&mut out, &presets);
        write_sos(&mut out, &[1], 0, 0, 0);
        out.push(0x00);

        let headers = parse_headers(&out).expect("headers parse");
        assert_eq!(headers.presets.max_val, 200);
        assert_eq!(headers.presets.t1, 5);
        assert_eq!(headers.presets.t2, 17);
        assert_eq!(headers.presets.t3, 60);
        assert_eq!(headers.presets.reset, 48);
    }
}
