// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! JPEG XS marker writers (encoder side).
//!
//! Each writer emits a marker exactly mirroring the field layout parsed in
//! [`super::markers`], so that a written marker re-parses to identical field
//! values. The byte layouts follow ISO/IEC 21122-1:2019 Annex A as interpreted
//! by the project's decoder ([`super::markers::parse_headers`]).
//!
//! All marker payloads carry a 2-byte big-endian length field `L` immediately
//! after the 2-byte marker code; `L` counts the length field itself plus the
//! payload bytes (i.e. `L = 2 + payload_len`).

use super::markers::{CDT, CWD, EOC, NLT, PIH, SLH, SOC, WGT};
use super::{JxsError, JxsResult};

/// Append a big-endian `u16` to `buf`.
fn push_u16(buf: &mut Vec<u8>, v: u16) {
    buf.extend_from_slice(&v.to_be_bytes());
}

/// Append a 3-byte big-endian unsigned integer (`u24`) to `buf`.
///
/// # Errors
/// Returns `JxsError::InvalidHeader` if `v` does not fit in 24 bits.
fn push_u24(buf: &mut Vec<u8>, v: u32) -> JxsResult<()> {
    if v > 0x00FF_FFFF {
        return Err(JxsError::InvalidHeader(format!(
            "u24 field overflow: {v} > 0xFFFFFF"
        )));
    }
    buf.push(((v >> 16) & 0xFF) as u8);
    buf.push(((v >> 8) & 0xFF) as u8);
    buf.push((v & 0xFF) as u8);
    Ok(())
}

/// Write the Start Of Codestream (SOC, `0xFF10`) marker — no payload.
pub fn write_soc(buf: &mut Vec<u8>) {
    push_u16(buf, SOC);
}

/// Write the End Of Codestream (EOC, `0xFF11`) marker — no payload.
pub fn write_eoc(buf: &mut Vec<u8>) {
    push_u16(buf, EOC);
}

/// Parameters for the Picture Header (PIH, `0xFF12`) marker.
#[derive(Debug, Clone, Copy)]
pub struct PihFields {
    /// Total codestream length in bytes (`Lcod`); `0` = undefined / VBR.
    pub codestream_len: u32,
    /// Profile identifier (`Ppih`).
    pub profile: u16,
    /// Level identifier (`Plev`).
    pub level: u16,
    /// Frame width in pixels (`Wf` / `Lc`).
    pub width: u16,
    /// Frame height in lines (`Hf` / `Lr`).
    pub height: u16,
    /// Codegroup width (`Cw`); the decoder ignores this — set to `width`.
    pub codegroup_width: u16,
    /// Slice height in lines (`Hsl` / `Sl`).
    pub slice_height: u16,
    /// Number of image components (`Nc`).
    pub num_components: u8,
    /// Ganging factor (`Ng`); `0` = none.
    pub ganging: u8,
    /// Sample bit depth (`Ss` / `Bw`-base), 1..=16.
    pub bit_depth: u8,
    /// Extended bit-width field (`Bw`); `0` for standard depths.
    pub bw_ext: u8,
    /// Fractional QP bits (`Fq`).
    pub fq: u8,
    /// Target bitrate (`Br`); `0` = VBR.
    pub bitrate: u32,
    /// Slice fraction (`Fsl`).
    pub fsl: u8,
    /// Progressive order (`Ppoc`).
    pub ppoc: u8,
    /// Coding-mode bitmask (`Cpih`).
    pub cpih: u8,
}

/// Write the Picture Header (PIH, `0xFF12`) marker.
///
/// The 26-byte payload layout exactly matches [`super::markers`] parsing:
/// `Lcod`(3) `Ppih`(2) `Plev`(2) `Wf`(2) `Hf`(2) `Cw`(2) `Hsl`(2) `Nc`(1)
/// `Ng`(1) `Ss`(1) `Bw`(1) `Fq`(1) `Br`(3) `Fsl`(1) `Ppoc`(1) `Cpih`(1).
///
/// # Errors
/// Returns `JxsError::InvalidHeader` on invalid component count / bit depth, or
/// if a `u24` field overflows.
pub fn write_pih(buf: &mut Vec<u8>, f: &PihFields) -> JxsResult<()> {
    if f.num_components == 0 {
        return Err(JxsError::InvalidHeader(
            "PIH Nc=0: at least one component required".to_string(),
        ));
    }
    if f.bit_depth == 0 || f.bit_depth > 16 {
        return Err(JxsError::InvalidHeader(format!(
            "PIH Ss={}: bit depth must be 1-16",
            f.bit_depth
        )));
    }
    if f.width == 0 || f.height == 0 {
        return Err(JxsError::InvalidHeader(format!(
            "PIH frame size {}x{}: dimensions must be non-zero",
            f.width, f.height
        )));
    }

    push_u16(buf, PIH);
    // Payload is exactly 26 bytes → Lp = 28.
    const PIH_PAYLOAD_LEN: u16 = 26;
    push_u16(buf, PIH_PAYLOAD_LEN + 2);

    push_u24(buf, f.codestream_len)?; // Lcod
    push_u16(buf, f.profile); // Ppih
    push_u16(buf, f.level); // Plev
    push_u16(buf, f.width); // Wf
    push_u16(buf, f.height); // Hf
    push_u16(buf, f.codegroup_width); // Cw
    push_u16(buf, f.slice_height); // Hsl
    buf.push(f.num_components); // Nc
    buf.push(f.ganging); // Ng
    buf.push(f.bit_depth); // Ss
    buf.push(f.bw_ext); // Bw
    buf.push(f.fq); // Fq
    push_u24(buf, f.bitrate)?; // Br
    buf.push(f.fsl); // Fsl
    buf.push(f.ppoc); // Ppoc
    buf.push(f.cpih); // Cpih
    Ok(())
}

/// One component descriptor as written into a CDT marker.
#[derive(Debug, Clone, Copy)]
pub struct CdtComponent {
    /// Component bit depth (`Bdc`).
    pub bit_depth: u8,
    /// Horizontal subsampling factor (`Sx`); `1` = no subsampling.
    pub sx: u8,
    /// Vertical subsampling factor (`Sy`); `1` = no subsampling.
    pub sy: u8,
}

/// Write the Component Table (CDT, `0xFF13`) marker.
///
/// Layout: 3 bytes per component — `Bdc`(1) `Sx`(1) `Sy`(1).
///
/// # Errors
/// Returns `JxsError::InvalidHeader` if `components` is empty.
pub fn write_cdt(buf: &mut Vec<u8>, components: &[CdtComponent]) -> JxsResult<()> {
    if components.is_empty() {
        return Err(JxsError::InvalidHeader(
            "CDT requires at least one component".to_string(),
        ));
    }
    push_u16(buf, CDT);
    let payload_len = components.len() * 3;
    push_u16(buf, (payload_len + 2) as u16);
    for c in components {
        buf.push(c.bit_depth);
        buf.push(c.sx);
        buf.push(c.sy);
    }
    Ok(())
}

/// Write the Weights Table (WGT, `0xFF14`) marker.
///
/// Each weight is a big-endian `u16`. The decoder reads exactly
/// `payload_len / 2` weights, matching this writer.
///
/// # Errors
/// Returns `JxsError::InvalidHeader` if `weights` is empty.
pub fn write_wgt(buf: &mut Vec<u8>, weights: &[u16]) -> JxsResult<()> {
    if weights.is_empty() {
        return Err(JxsError::InvalidHeader(
            "WGT requires at least one weight".to_string(),
        ));
    }
    push_u16(buf, WGT);
    let payload_len = weights.len() * 2;
    push_u16(buf, (payload_len + 2) as u16);
    for &w in weights {
        push_u16(buf, w);
    }
    Ok(())
}

/// Write the Nonlinear Transform (NLT, `0xFF15`) marker (quadratic variant).
///
/// Payload: `Tnlt`(1) `T1`(2) `T2`(2) = 5 bytes, matching
/// [`super::nlt::parse_nlt_payload`].
///
/// `tnlt` is `0` for quadratic and `1` for the (deferred) extended variant.
pub fn write_nlt(buf: &mut Vec<u8>, tnlt: u8, t1: u16, t2: u16) {
    push_u16(buf, NLT);
    push_u16(buf, 7); // L = 2 + 5
    buf.push(tnlt);
    push_u16(buf, t1);
    push_u16(buf, t2);
}

/// Write the Codeword Mapping (CWD, `0xFF17`) marker.
///
/// The project's decoder uses its fixed default VLC tables and skips the CWD
/// payload via its generic marker-skip path, so this writer simply emits a
/// length-prefixed payload that the decoder will step over.
///
/// `payload` may be empty (the marker still carries its 2-byte length).
pub fn write_cwd(buf: &mut Vec<u8>, payload: &[u8]) {
    push_u16(buf, CWD);
    push_u16(buf, (payload.len() + 2) as u16);
    buf.extend_from_slice(payload);
}

/// Write the Slice Header (SLH, `0xFF19`) marker.
///
/// Payload: `Qpih`(2) — the slice quantisation step. The decoder reads this
/// `u16` at payload offset 0. Compressed slice data must follow immediately
/// after this marker.
pub fn write_slh(buf: &mut Vec<u8>, qp: u16) {
    push_u16(buf, SLH);
    push_u16(buf, 4); // L = 2 + 2 (Qpih)
    push_u16(buf, qp);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpegxs::markers::{parse_headers, PROFILE_MAIN};

    fn default_pih(width: u16, height: u16, nc: u8, bit_depth: u8) -> PihFields {
        PihFields {
            codestream_len: 0,
            profile: PROFILE_MAIN,
            level: 0,
            width,
            height,
            codegroup_width: width,
            slice_height: height,
            num_components: nc,
            ganging: 0,
            bit_depth,
            bw_ext: 0,
            fq: 0,
            bitrate: 0,
            fsl: 0,
            ppoc: 0,
            cpih: 0,
        }
    }

    #[test]
    fn pih_write_then_parse_roundtrip() {
        let mut buf = Vec::new();
        write_soc(&mut buf);
        write_pih(&mut buf, &default_pih(640, 480, 3, 8)).unwrap();
        write_cdt(
            &mut buf,
            &[
                CdtComponent {
                    bit_depth: 8,
                    sx: 1,
                    sy: 1,
                },
                CdtComponent {
                    bit_depth: 8,
                    sx: 1,
                    sy: 1,
                },
                CdtComponent {
                    bit_depth: 8,
                    sx: 1,
                    sy: 1,
                },
            ],
        )
        .unwrap();
        write_eoc(&mut buf);

        let (headers, _) = parse_headers(&buf).expect("parse");
        assert_eq!(headers.pih.width, 640);
        assert_eq!(headers.pih.height, 480);
        assert_eq!(headers.pih.slice_height, 480);
        assert_eq!(headers.pih.num_components, 3);
        assert_eq!(headers.pih.bit_depth, 8);
        assert_eq!(headers.pih.profile, PROFILE_MAIN);
        assert_eq!(headers.components.len(), 3);
    }

    #[test]
    fn pih_16x16_yuv_fields_match() {
        let mut buf = Vec::new();
        write_soc(&mut buf);
        write_pih(&mut buf, &default_pih(16, 16, 3, 8)).unwrap();
        write_cdt(
            &mut buf,
            &[CdtComponent {
                bit_depth: 8,
                sx: 1,
                sy: 1,
            }; 3],
        )
        .unwrap();
        write_eoc(&mut buf);
        let (h, _) = parse_headers(&buf).unwrap();
        assert_eq!(h.pih.width, 16);
        assert_eq!(h.pih.height, 16);
        assert_eq!(h.pih.num_components, 3);
        assert_eq!(h.pih.bit_depth, 8);
    }

    #[test]
    fn wgt_write_then_parse() {
        let mut buf = Vec::new();
        write_soc(&mut buf);
        write_pih(&mut buf, &default_pih(8, 8, 1, 8)).unwrap();
        write_cdt(
            &mut buf,
            &[CdtComponent {
                bit_depth: 8,
                sx: 1,
                sy: 1,
            }],
        )
        .unwrap();
        write_wgt(&mut buf, &[256, 256, 256, 256]).unwrap();
        write_eoc(&mut buf);
        let (h, _) = parse_headers(&buf).unwrap();
        assert_eq!(h.weights, vec![256, 256, 256, 256]);
    }

    #[test]
    fn nlt_write_then_parse_payload() {
        use crate::jpegxs::nlt::{parse_nlt_payload, NltType};
        let mut buf = Vec::new();
        write_soc(&mut buf);
        write_pih(&mut buf, &default_pih(8, 8, 1, 8)).unwrap();
        write_cdt(
            &mut buf,
            &[CdtComponent {
                bit_depth: 8,
                sx: 1,
                sy: 1,
            }],
        )
        .unwrap();
        write_nlt(&mut buf, 0, 64, 192);
        write_eoc(&mut buf);
        let (h, _) = parse_headers(&buf).unwrap();
        assert!(h.has_nlt);
        let payload = h.nlt_payload.expect("nlt payload");
        let params = parse_nlt_payload(&payload).unwrap();
        assert_eq!(params.nlt_type, NltType::Quadratic);
        assert_eq!(params.t1, 64);
        assert_eq!(params.t2, 192);
    }

    #[test]
    fn pih_rejects_zero_components() {
        let mut buf = Vec::new();
        let r = write_pih(&mut buf, &default_pih(8, 8, 0, 8));
        assert!(r.is_err());
    }

    #[test]
    fn pih_rejects_bad_bit_depth() {
        let mut buf = Vec::new();
        let r = write_pih(&mut buf, &default_pih(8, 8, 1, 17));
        assert!(r.is_err());
    }

    #[test]
    fn cdt_rejects_empty() {
        let mut buf = Vec::new();
        assert!(write_cdt(&mut buf, &[]).is_err());
    }

    #[test]
    fn wgt_rejects_empty() {
        let mut buf = Vec::new();
        assert!(write_wgt(&mut buf, &[]).is_err());
    }

    #[test]
    fn slh_writes_qp_field() {
        let mut buf = Vec::new();
        write_slh(&mut buf, 0x1234);
        // FF 19 | 00 04 | 12 34
        assert_eq!(buf, vec![0xFF, 0x19, 0x00, 0x04, 0x12, 0x34]);
    }

    #[test]
    fn cwd_marker_is_skipped_by_parser() {
        let mut buf = Vec::new();
        write_soc(&mut buf);
        write_pih(&mut buf, &default_pih(8, 8, 1, 8)).unwrap();
        write_cdt(
            &mut buf,
            &[CdtComponent {
                bit_depth: 8,
                sx: 1,
                sy: 1,
            }],
        )
        .unwrap();
        write_cwd(&mut buf, &[0xDE, 0xAD, 0xBE, 0xEF]);
        write_eoc(&mut buf);
        // Parser should skip CWD via its generic marker path without error.
        let (h, _) = parse_headers(&buf).unwrap();
        assert_eq!(h.pih.width, 8);
    }

    #[test]
    fn u24_overflow_rejected() {
        let mut buf = Vec::new();
        assert!(push_u24(&mut buf, 0x0100_0000).is_err());
    }
}
