//! VR180 V3D metadata box parsing and serialisation.
//!
//! The `v3d ` (Video 3D / VR180) box is an ISOBMFF extension used by Google's
//! VR180 format to signal:
//!
//! * **Stereo mode** — left-right or top-bottom packed halves.
//! * **Projection type** — equirectangular or fisheye.
//! * **Field-of-view** — the horizontal and vertical FOV of the fisheye lens.
//! * **Principal point offset** — optical centre relative to the image centre.
//!
//! The specification is described in the *VR180 Video Metadata Specification*
//! published by Google, 2018.
//!
//! ## Box layout
//!
//! ```text
//! v3d  — VR180 3D box  (fourcc = "v3d ")
//!   4 bytes: total box size (big-endian u32)
//!   4 bytes: fourcc "v3d "  (note trailing space)
//!   1 byte:  version (0)
//!   3 bytes: flags  (0,0,0)
//!   1 byte:  stereo_mode (0=LR, 1=TB)
//!   1 byte:  projection   (0=equirect, 1=fisheye)
//!   2 bytes: reserved (0,0)
//!   4 bytes: hfov_deg as fixed-point 16.16 big-endian
//!   4 bytes: vfov_deg as fixed-point 16.16 big-endian
//!   4 bytes: ppx_offset as fixed-point 16.16 big-endian (signed)
//!   4 bytes: ppy_offset as fixed-point 16.16 big-endian (signed)
//! ```
//!
//! Total: **28 bytes**.
//!
//! ## Usage
//!
//! ```rust
//! use oximedia_360::v3d::{V3dBox, Vr180StereoMode, Vr180Projection};
//!
//! let b = V3dBox::fisheye_lr(180.0, 180.0);
//! let bytes = b.to_bytes();
//! let parsed = V3dBox::parse(&bytes).expect("ok");
//! assert_eq!(parsed.stereo_mode, Vr180StereoMode::LeftRight);
//! ```

use crate::VrError;

// ─── Stereo mode ──────────────────────────────────────────────────────────────

/// Stereo packing mode used by a VR180 video.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vr180StereoMode {
    /// Left eye on the left half, right eye on the right half.
    LeftRight,
    /// Left eye on the top half, right eye on the bottom half.
    TopBottom,
}

impl Vr180StereoMode {
    fn as_u8(self) -> u8 {
        match self {
            Vr180StereoMode::LeftRight => 0,
            Vr180StereoMode::TopBottom => 1,
        }
    }

    fn from_u8(v: u8) -> Self {
        match v {
            1 => Vr180StereoMode::TopBottom,
            _ => Vr180StereoMode::LeftRight,
        }
    }
}

// ─── Projection type ──────────────────────────────────────────────────────────

/// Projection type signalled in the VR180 `v3d ` box.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vr180Projection {
    /// Equirectangular (full sphere or half-sphere).
    Equirectangular,
    /// Fisheye (typically equidistant, up to 180° FOV per eye).
    Fisheye,
}

impl Vr180Projection {
    fn as_u8(self) -> u8 {
        match self {
            Vr180Projection::Equirectangular => 0,
            Vr180Projection::Fisheye => 1,
        }
    }

    fn from_u8(v: u8) -> Self {
        match v {
            1 => Vr180Projection::Fisheye,
            _ => Vr180Projection::Equirectangular,
        }
    }
}

// ─── V3dBox ───────────────────────────────────────────────────────────────────

/// VR180 3D metadata box (`v3d `).
///
/// Signals stereo mode, projection type, and lens FOV/calibration data for
/// VR180 video content.
#[derive(Debug, Clone, PartialEq)]
pub struct V3dBox {
    /// How the two eye views are packed in the frame.
    pub stereo_mode: Vr180StereoMode,
    /// Projection type (equirectangular or fisheye).
    pub projection: Vr180Projection,
    /// Horizontal field of view in degrees (typically 180.0 for VR180).
    pub hfov_deg: f32,
    /// Vertical field of view in degrees (typically 180.0 for VR180).
    pub vfov_deg: f32,
    /// Principal point X offset from image centre, as a fraction of image width.
    /// 0.0 = perfectly centred.
    pub ppx_offset: f32,
    /// Principal point Y offset from image centre, as a fraction of image height.
    /// 0.0 = perfectly centred.
    pub ppy_offset: f32,
}

impl V3dBox {
    /// Create a VR180 fisheye left-right stereo box with the given FOV.
    ///
    /// The principal point is assumed to be perfectly centred.
    pub fn fisheye_lr(hfov_deg: f32, vfov_deg: f32) -> Self {
        Self {
            stereo_mode: Vr180StereoMode::LeftRight,
            projection: Vr180Projection::Fisheye,
            hfov_deg,
            vfov_deg,
            ppx_offset: 0.0,
            ppy_offset: 0.0,
        }
    }

    /// Create a VR180 equirectangular left-right stereo box.
    pub fn equirectangular_lr() -> Self {
        Self {
            stereo_mode: Vr180StereoMode::LeftRight,
            projection: Vr180Projection::Equirectangular,
            hfov_deg: 180.0,
            vfov_deg: 180.0,
            ppx_offset: 0.0,
            ppy_offset: 0.0,
        }
    }

    /// Serialize to ISOBMFF bytes (28 bytes total).
    ///
    /// All multi-byte integers are big-endian.  Floating-point values are
    /// encoded as 16.16 fixed-point.
    pub fn to_bytes(&self) -> Vec<u8> {
        // 4 (size) + 4 (fourcc) + 1 (version) + 3 (flags)
        // + 1 (stereo_mode) + 1 (projection) + 2 (reserved)
        // + 4 (hfov) + 4 (vfov) + 4 (ppx) + 4 (ppy) = 32 bytes
        let total_size: u32 = 32;
        let mut out = Vec::with_capacity(total_size as usize);

        out.extend_from_slice(&total_size.to_be_bytes());
        out.extend_from_slice(b"v3d "); // fourcc with trailing space
        out.push(0u8); // version
        out.push(0u8); // flags[0]
        out.push(0u8); // flags[1]
        out.push(0u8); // flags[2]
        out.push(self.stereo_mode.as_u8());
        out.push(self.projection.as_u8());
        out.push(0u8); // reserved
        out.push(0u8); // reserved
        out.extend_from_slice(&f32_to_fixed_16_16(self.hfov_deg));
        out.extend_from_slice(&f32_to_fixed_16_16(self.vfov_deg));
        out.extend_from_slice(&f32_to_signed_fixed_16_16(self.ppx_offset));
        out.extend_from_slice(&f32_to_signed_fixed_16_16(self.ppy_offset));

        out
    }

    /// Parse a `V3dBox` from its serialised ISOBMFF bytes.
    ///
    /// # Errors
    /// Returns [`VrError::ParseError`] if:
    /// * The data is too short (< 32 bytes).
    /// * The fourcc is not `"v3d "`.
    /// * The declared size does not match the expected layout.
    pub fn parse(data: &[u8]) -> Result<Self, VrError> {
        if data.len() < 32 {
            return Err(VrError::ParseError(
                "v3d box must be at least 32 bytes".into(),
            ));
        }

        let total_size = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
        if &data[4..8] != b"v3d " {
            return Err(VrError::ParseError(
                "expected fourcc 'v3d ' (with trailing space)".into(),
            ));
        }
        if total_size < 32 {
            return Err(VrError::ParseError(
                "v3d box size field is too small".into(),
            ));
        }
        if data.len() < total_size as usize {
            return Err(VrError::ParseError("v3d box truncated".into()));
        }

        // version at [8], flags at [9..12]
        let stereo_mode = Vr180StereoMode::from_u8(data[12]);
        let projection = Vr180Projection::from_u8(data[13]);
        // reserved: data[14], data[15]

        let hfov_deg =
            fixed_16_16_to_f32(u32::from_be_bytes([data[16], data[17], data[18], data[19]]));
        let vfov_deg =
            fixed_16_16_to_f32(u32::from_be_bytes([data[20], data[21], data[22], data[23]]));
        let ppx_offset =
            signed_fixed_16_16_to_f32(i32::from_be_bytes([data[24], data[25], data[26], data[27]]));
        let ppy_offset =
            signed_fixed_16_16_to_f32(i32::from_be_bytes([data[28], data[29], data[30], data[31]]));

        Ok(Self {
            stereo_mode,
            projection,
            hfov_deg,
            vfov_deg,
            ppx_offset,
            ppy_offset,
        })
    }
}

// ─── Fixed-point helpers ──────────────────────────────────────────────────────

/// Encode an `f32` as unsigned fixed-point 16.16 (big-endian bytes).
fn f32_to_fixed_16_16(v: f32) -> [u8; 4] {
    let fixed = (v.max(0.0) * 65536.0).round() as u32;
    fixed.to_be_bytes()
}

/// Decode an unsigned fixed-point 16.16 `u32` to `f32`.
fn fixed_16_16_to_f32(raw: u32) -> f32 {
    raw as f32 / 65536.0
}

/// Encode an `f32` as signed fixed-point 16.16 (big-endian bytes).
///
/// The integer part occupies the upper 16 bits (two's-complement),
/// the fractional part the lower 16 bits.
fn f32_to_signed_fixed_16_16(v: f32) -> [u8; 4] {
    let fixed = (v * 65536.0).round() as i32;
    fixed.to_be_bytes()
}

/// Decode a signed fixed-point 16.16 `i32` to `f32`.
fn signed_fixed_16_16_to_f32(raw: i32) -> f32 {
    raw as f32 / 65536.0
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Fixed-point helpers ──────────────────────────────────────────────────

    #[test]
    fn fixed_16_16_roundtrip_positive() {
        let v = 180.0_f32;
        let raw = u32::from_be_bytes(f32_to_fixed_16_16(v));
        let back = fixed_16_16_to_f32(raw);
        assert!((back - v).abs() < 0.01, "back={back}");
    }

    #[test]
    fn signed_fixed_16_16_roundtrip_negative() {
        let v = -0.05_f32;
        let raw = i32::from_be_bytes(f32_to_signed_fixed_16_16(v));
        let back = signed_fixed_16_16_to_f32(raw);
        assert!((back - v).abs() < 0.0002, "back={back}");
    }

    // ── Vr180StereoMode ──────────────────────────────────────────────────────

    #[test]
    fn stereo_mode_lr_is_zero() {
        assert_eq!(Vr180StereoMode::LeftRight.as_u8(), 0);
    }

    #[test]
    fn stereo_mode_tb_is_one() {
        assert_eq!(Vr180StereoMode::TopBottom.as_u8(), 1);
    }

    #[test]
    fn stereo_mode_roundtrip() {
        for mode in [Vr180StereoMode::LeftRight, Vr180StereoMode::TopBottom] {
            assert_eq!(Vr180StereoMode::from_u8(mode.as_u8()), mode);
        }
    }

    // ── Vr180Projection ──────────────────────────────────────────────────────

    #[test]
    fn projection_fisheye_is_one() {
        assert_eq!(Vr180Projection::Fisheye.as_u8(), 1);
    }

    #[test]
    fn projection_equirect_is_zero() {
        assert_eq!(Vr180Projection::Equirectangular.as_u8(), 0);
    }

    // ── V3dBox serialisation ─────────────────────────────────────────────────

    #[test]
    fn v3d_box_size_is_32() {
        let b = V3dBox::fisheye_lr(180.0, 180.0);
        let bytes = b.to_bytes();
        assert_eq!(bytes.len(), 32);
    }

    #[test]
    fn v3d_box_size_field_matches_length() {
        let b = V3dBox::fisheye_lr(180.0, 180.0);
        let bytes = b.to_bytes();
        let size = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(size as usize, bytes.len());
    }

    #[test]
    fn v3d_box_fourcc_is_correct() {
        let b = V3dBox::fisheye_lr(180.0, 180.0);
        let bytes = b.to_bytes();
        assert_eq!(&bytes[4..8], b"v3d ");
    }

    #[test]
    fn v3d_box_version_is_zero() {
        let b = V3dBox::fisheye_lr(180.0, 180.0);
        let bytes = b.to_bytes();
        assert_eq!(bytes[8], 0); // version
    }

    // ── V3dBox round-trip ────────────────────────────────────────────────────

    #[test]
    fn v3d_box_fisheye_lr_roundtrip() {
        let original = V3dBox::fisheye_lr(180.0, 180.0);
        let bytes = original.to_bytes();
        let parsed = V3dBox::parse(&bytes).expect("parse ok");
        assert_eq!(parsed.stereo_mode, Vr180StereoMode::LeftRight);
        assert_eq!(parsed.projection, Vr180Projection::Fisheye);
        assert!(
            (parsed.hfov_deg - 180.0).abs() < 0.01,
            "hfov={}",
            parsed.hfov_deg
        );
        assert!(
            (parsed.vfov_deg - 180.0).abs() < 0.01,
            "vfov={}",
            parsed.vfov_deg
        );
    }

    #[test]
    fn v3d_box_equirectangular_lr_roundtrip() {
        let original = V3dBox::equirectangular_lr();
        let bytes = original.to_bytes();
        let parsed = V3dBox::parse(&bytes).expect("parse ok");
        assert_eq!(parsed.projection, Vr180Projection::Equirectangular);
        assert_eq!(parsed.stereo_mode, Vr180StereoMode::LeftRight);
    }

    #[test]
    fn v3d_box_top_bottom_mode_roundtrip() {
        let original = V3dBox {
            stereo_mode: Vr180StereoMode::TopBottom,
            projection: Vr180Projection::Fisheye,
            hfov_deg: 170.0,
            vfov_deg: 150.0,
            ppx_offset: 0.02,
            ppy_offset: -0.01,
        };
        let bytes = original.to_bytes();
        let parsed = V3dBox::parse(&bytes).expect("parse ok");
        assert_eq!(parsed.stereo_mode, Vr180StereoMode::TopBottom);
        assert!(
            (parsed.hfov_deg - 170.0).abs() < 0.05,
            "hfov={}",
            parsed.hfov_deg
        );
        assert!(
            (parsed.vfov_deg - 150.0).abs() < 0.05,
            "vfov={}",
            parsed.vfov_deg
        );
        assert!(
            (parsed.ppx_offset - 0.02).abs() < 0.0002,
            "ppx={}",
            parsed.ppx_offset
        );
        assert!(
            (parsed.ppy_offset - (-0.01)).abs() < 0.0002,
            "ppy={}",
            parsed.ppy_offset
        );
    }

    // ── Parse error handling ─────────────────────────────────────────────────

    #[test]
    fn v3d_parse_error_too_short() {
        assert!(V3dBox::parse(&[0u8; 8]).is_err());
    }

    #[test]
    fn v3d_parse_error_wrong_fourcc() {
        let mut bytes = V3dBox::fisheye_lr(180.0, 180.0).to_bytes();
        bytes[4] = b'X'; // corrupt fourcc
        assert!(V3dBox::parse(&bytes).is_err());
    }

    #[test]
    fn v3d_parse_error_size_too_small() {
        let mut bytes = V3dBox::fisheye_lr(180.0, 180.0).to_bytes();
        // Set declared size to 10 (too small)
        bytes[0] = 0;
        bytes[1] = 0;
        bytes[2] = 0;
        bytes[3] = 10;
        assert!(V3dBox::parse(&bytes).is_err());
    }
}
