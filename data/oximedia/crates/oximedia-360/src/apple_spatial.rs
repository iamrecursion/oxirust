//! Apple Spatial Video metadata support.
//!
//! Apple Spatial Video uses MV-HEVC (Multi-View HEVC) for encoding stereo 3D
//! video and embeds rich metadata in the ISOBMFF container to describe:
//!
//! * Stereo eye layout (`hero` / `paired`)
//! * Horizontal disparity baseline and limits
//! * `ProjectionBox` (`prhd`) with FOV + projection type
//! * `StereoPairBox` (`eyes`) for left/right eye track association
//! * `SpatialVideoMetadataBox` (`vexu`) wrapping all spatial metadata
//! * Apple `cmfy` (comfort / advisory) box for divergence limits
//!
//! The full specification is given in **Apple HEVC Stereo Video — Interoperability
//! Profile** (2023) and **ISO/IEC 14496-12**.  This module implements the
//! structures needed to write and parse compliant boxes.
//!
//! ## Box hierarchy
//!
//! ```text
//! vexu  — Spatial Video Metadata Box
//! ├── eyes  — Stereo Pair Box
//! │   ├── hero  — Hero Eye Box (identifies left/right hero eye)
//! │   └── trak_ref  — Track references (hero + pear track IDs)
//! └── must  — Must-Understand Box (opaque extension flags)
//!
//! prhd  — Projection Header Box (inside each video sample entry)
//!   └── prji  — Projection Info Box
//!       └── hfov  — Horizontal FOV (optional sub-box)
//!
//! cmfy  — Comfort Box (divergence advisory)
//! ```
//!
//! ## Usage
//!
//! ```rust
//! use oximedia_360::apple_spatial::{
//!     AppleSpatialVideoMeta, EyeConfig, ProjectionBox, ComfortBox,
//! };
//!
//! let meta = AppleSpatialVideoMeta::default();
//! let bytes = meta.to_bytes();
//! ```

use crate::VrError;

// ─── Hero eye / stereo config ─────────────────────────────────────────────────

/// Which eye is the "hero" (primary) eye used for monoscopic fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeroEye {
    /// Left eye is the hero eye.
    Left,
    /// Right eye is the hero eye.
    Right,
}

impl HeroEye {
    /// Numeric eye ID as defined by Apple's spec: Left = 1, Right = 2.
    pub fn eye_id(self) -> u8 {
        match self {
            HeroEye::Left => 1,
            HeroEye::Right => 2,
        }
    }
}

/// Stereo pair configuration for a spatial video track pair.
#[derive(Debug, Clone, PartialEq)]
pub struct EyeConfig {
    /// Track ID of the hero (primary) eye.
    pub hero_track_id: u32,
    /// Track ID of the paired (secondary) eye.
    pub pair_track_id: u32,
    /// Which eye is the hero eye.
    pub hero_eye: HeroEye,
}

impl EyeConfig {
    /// Construct a standard left-hero stereo pair.
    pub fn left_hero(left_track: u32, right_track: u32) -> Self {
        Self {
            hero_track_id: left_track,
            pair_track_id: right_track,
            hero_eye: HeroEye::Left,
        }
    }
}

// ─── Projection type ──────────────────────────────────────────────────────────

/// Projection type used by the spatial video (for `prhd` box).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialProjection {
    /// Standard rectilinear (perspective) projection — used for most
    /// Apple Spatial Video and visionOS content.
    Rectilinear,
    /// Equirectangular (360°) projection — used for Apple Immersive Video.
    Equirectangular,
    /// Half-equirectangular (180°) — Apple Spatial Video at wide FOV.
    HalfEquirectangular,
}

impl SpatialProjection {
    /// 4-byte FourCC identifier used in `prji` sub-box.
    pub fn fourcc(self) -> [u8; 4] {
        match self {
            SpatialProjection::Rectilinear => *b"rect",
            SpatialProjection::Equirectangular => *b"equi",
            SpatialProjection::HalfEquirectangular => *b"hequ",
        }
    }
}

// ─── Projection Header Box (prhd) ────────────────────────────────────────────

/// Projection Header Box (`prhd`) containing projection type and optional FOV.
///
/// Placed inside each MV-HEVC video sample entry.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectionBox {
    /// Projection type.
    pub projection: SpatialProjection,
    /// Horizontal FOV in degrees, if explicitly signalled (optional).
    /// For `Rectilinear`, this is the full horizontal FOV of the camera.
    /// For 360/180 projections this field is normally omitted (`None`).
    pub hfov_deg: Option<f32>,
}

impl ProjectionBox {
    /// Standard rectilinear projection with 90° HFOV (typical iPhone default).
    pub fn rectilinear_90() -> Self {
        Self {
            projection: SpatialProjection::Rectilinear,
            hfov_deg: Some(90.0),
        }
    }

    /// Equirectangular projection (no explicit FOV required).
    pub fn equirectangular() -> Self {
        Self {
            projection: SpatialProjection::Equirectangular,
            hfov_deg: None,
        }
    }

    /// Serialize to ISOBMFF bytes.
    ///
    /// Layout:
    /// ```text
    /// prhd box:
    ///   4 bytes: total size
    ///   4 bytes: fourcc "prhd"
    ///   1 byte:  version (0)
    ///   3 bytes: flags (0,0,0)
    ///   prji sub-box:
    ///     4 bytes: size
    ///     4 bytes: "prji"
    ///     4 bytes: projection fourcc
    ///   [optional hfov sub-box]:
    ///     4 bytes: size
    ///     4 bytes: "hfov"
    ///     4 bytes: FOV as fixed-point 16.16 degrees
    /// ```
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut inner: Vec<u8> = Vec::new();

        // prji sub-box (12 bytes)
        let prji_size: u32 = 12;
        inner.extend_from_slice(&prji_size.to_be_bytes());
        inner.extend_from_slice(b"prji");
        inner.extend_from_slice(&self.projection.fourcc());

        // Optional hfov sub-box (12 bytes if present)
        if let Some(fov) = self.hfov_deg {
            let hfov_size: u32 = 12;
            inner.extend_from_slice(&hfov_size.to_be_bytes());
            inner.extend_from_slice(b"hfov");
            // Fixed-point 16.16: integer part + fractional part
            let fixed = (fov * 65536.0) as u32;
            inner.extend_from_slice(&fixed.to_be_bytes());
        }

        // prhd box header: 4 (size) + 4 (fourcc) + 1 (version) + 3 (flags)
        let prhd_header_size: u32 = 12;
        let total_size = prhd_header_size + inner.len() as u32;

        let mut out = Vec::with_capacity(total_size as usize);
        out.extend_from_slice(&total_size.to_be_bytes());
        out.extend_from_slice(b"prhd");
        out.push(0u8); // version
        out.push(0u8); // flags[0]
        out.push(0u8); // flags[1]
        out.push(0u8); // flags[2]
        out.extend_from_slice(&inner);
        out
    }

    /// Parse a `ProjectionBox` from its serialised bytes.
    ///
    /// # Errors
    /// Returns [`VrError::ParseError`] if the bytes are malformed.
    pub fn parse(data: &[u8]) -> Result<Self, VrError> {
        if data.len() < 12 {
            return Err(VrError::ParseError("prhd box too short".into()));
        }
        let total_size = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
        if &data[4..8] != b"prhd" {
            return Err(VrError::ParseError("expected fourcc 'prhd'".into()));
        }
        if data.len() < total_size as usize {
            return Err(VrError::ParseError("prhd box truncated".into()));
        }
        // Skip version + flags (4 bytes at offset 8)
        let mut pos = 12usize;
        let mut projection = SpatialProjection::Rectilinear;
        let mut hfov_deg = None;

        while pos + 8 <= total_size as usize {
            let sub_size =
                u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                    as usize;
            if sub_size < 8 || pos + sub_size > total_size as usize {
                break;
            }
            let fourcc = &data[pos + 4..pos + 8];
            if fourcc == b"prji" && pos + 12 <= total_size as usize {
                let fc = &data[pos + 8..pos + 12];
                projection = match fc {
                    b"rect" => SpatialProjection::Rectilinear,
                    b"equi" => SpatialProjection::Equirectangular,
                    b"hequ" => SpatialProjection::HalfEquirectangular,
                    _ => SpatialProjection::Rectilinear,
                };
            } else if fourcc == b"hfov" && pos + 12 <= total_size as usize {
                let fixed = u32::from_be_bytes([
                    data[pos + 8],
                    data[pos + 9],
                    data[pos + 10],
                    data[pos + 11],
                ]);
                hfov_deg = Some(fixed as f32 / 65536.0);
            }
            pos += sub_size;
        }

        Ok(Self {
            projection,
            hfov_deg,
        })
    }
}

// ─── Comfort Box (cmfy) ───────────────────────────────────────────────────────

/// Apple Comfort Box (`cmfy`) — signals horizontal disparity limits.
///
/// The comfort advisory informs renderers about the maximum positive and
/// negative disparity values so they can apply comfort-zone enforcement.
#[derive(Debug, Clone, PartialEq)]
pub struct ComfortBox {
    /// Maximum positive disparity (screen-plane depth, normalised 0..1).
    pub max_positive_disparity: f32,
    /// Maximum negative disparity (behind-screen depth, normalised 0..1).
    pub max_negative_disparity: f32,
    /// Recommended divergence threshold (in degrees of arc) for content.
    pub divergence_deg: f32,
}

impl ComfortBox {
    /// Sensible default advisory values for typical 3D video.
    pub fn default_advisory() -> Self {
        Self {
            max_positive_disparity: 0.02,
            max_negative_disparity: 0.03,
            divergence_deg: 1.0,
        }
    }

    /// Serialize to ISOBMFF bytes.
    ///
    /// Layout:
    /// ```text
    /// cmfy box:
    ///   4 bytes: total size
    ///   4 bytes: fourcc "cmfy"
    ///   1 byte:  version (0)
    ///   3 bytes: flags (0,0,0)
    ///   4 bytes: max_positive_disparity (IEEE 754 float, big-endian)
    ///   4 bytes: max_negative_disparity (IEEE 754 float, big-endian)
    ///   4 bytes: divergence_deg (IEEE 754 float, big-endian)
    /// ```
    pub fn to_bytes(&self) -> Vec<u8> {
        // 4 (size) + 4 (fourcc) + 4 (version+flags) + 4+4+4 (payload) = 24 bytes
        let total_size: u32 = 24;
        let mut out = Vec::with_capacity(total_size as usize);
        out.extend_from_slice(&total_size.to_be_bytes());
        out.extend_from_slice(b"cmfy");
        out.push(0u8); // version
        out.push(0u8); // flags
        out.push(0u8);
        out.push(0u8);
        out.extend_from_slice(&self.max_positive_disparity.to_be_bytes());
        out.extend_from_slice(&self.max_negative_disparity.to_be_bytes());
        out.extend_from_slice(&self.divergence_deg.to_be_bytes());
        out
    }

    /// Parse a `ComfortBox` from its serialised bytes.
    ///
    /// # Errors
    /// Returns [`VrError::ParseError`] if the bytes are malformed.
    pub fn parse(data: &[u8]) -> Result<Self, VrError> {
        if data.len() < 24 {
            return Err(VrError::ParseError("cmfy box too short".into()));
        }
        if &data[4..8] != b"cmfy" {
            return Err(VrError::ParseError("expected fourcc 'cmfy'".into()));
        }
        // version+flags at [8..12], payload at [12..24]
        let pos_disp = f32::from_be_bytes([data[12], data[13], data[14], data[15]]);
        let neg_disp = f32::from_be_bytes([data[16], data[17], data[18], data[19]]);
        let divg = f32::from_be_bytes([data[20], data[21], data[22], data[23]]);
        Ok(Self {
            max_positive_disparity: pos_disp,
            max_negative_disparity: neg_disp,
            divergence_deg: divg,
        })
    }
}

// ─── Stereo Pair Box (eyes) ───────────────────────────────────────────────────

/// Stereo Pair Box (`eyes`) — identifies the hero and paired eye tracks.
///
/// Per Apple spec, this box resides inside `vexu` and links the two MV-HEVC
/// tracks that make up the stereo pair.
#[derive(Debug, Clone, PartialEq)]
pub struct StereoPairBox {
    pub eye_config: EyeConfig,
}

impl StereoPairBox {
    /// Create a new stereo pair box.
    pub fn new(eye_config: EyeConfig) -> Self {
        Self { eye_config }
    }

    /// Serialize to ISOBMFF bytes.
    ///
    /// Layout:
    /// ```text
    /// eyes box:
    ///   4 bytes: total size
    ///   4 bytes: fourcc "eyes"
    ///   hero sub-box:
    ///     4 bytes: size (= 9)
    ///     4 bytes: "hero"
    ///     1 byte:  eye_id (1=left, 2=right)
    ///   trak sub-box (track IDs):
    ///     4 bytes: size (= 16)
    ///     4 bytes: "trak"
    ///     4 bytes: hero_track_id
    ///     4 bytes: pair_track_id
    /// ```
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut inner: Vec<u8> = Vec::new();

        // hero sub-box: 4 (size) + 4 (fourcc) + 1 (eye_id) = 9 bytes
        let hero_size: u32 = 9;
        inner.extend_from_slice(&hero_size.to_be_bytes());
        inner.extend_from_slice(b"hero");
        inner.push(self.eye_config.hero_eye.eye_id());

        // trak sub-box: 4 + 4 + 4 + 4 = 16 bytes
        let trak_size: u32 = 16;
        inner.extend_from_slice(&trak_size.to_be_bytes());
        inner.extend_from_slice(b"trak");
        inner.extend_from_slice(&self.eye_config.hero_track_id.to_be_bytes());
        inner.extend_from_slice(&self.eye_config.pair_track_id.to_be_bytes());

        let total_size = 8u32 + inner.len() as u32;
        let mut out = Vec::with_capacity(total_size as usize);
        out.extend_from_slice(&total_size.to_be_bytes());
        out.extend_from_slice(b"eyes");
        out.extend_from_slice(&inner);
        out
    }

    /// Parse a `StereoPairBox` from its serialised bytes.
    ///
    /// # Errors
    /// Returns [`VrError::ParseError`] if the bytes are malformed.
    pub fn parse(data: &[u8]) -> Result<Self, VrError> {
        if data.len() < 8 {
            return Err(VrError::ParseError("eyes box too short".into()));
        }
        let total_size = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize;
        if &data[4..8] != b"eyes" {
            return Err(VrError::ParseError("expected fourcc 'eyes'".into()));
        }
        if data.len() < total_size {
            return Err(VrError::ParseError("eyes box truncated".into()));
        }

        let mut pos = 8usize;
        let mut hero_eye = HeroEye::Left;
        let mut hero_track_id = 1u32;
        let mut pair_track_id = 2u32;

        while pos + 8 <= total_size {
            let sub_size =
                u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                    as usize;
            if sub_size < 8 || pos + sub_size > total_size {
                break;
            }
            let fourcc = &data[pos + 4..pos + 8];
            if fourcc == b"hero" && pos + 9 <= total_size {
                hero_eye = match data[pos + 8] {
                    2 => HeroEye::Right,
                    _ => HeroEye::Left,
                };
            } else if fourcc == b"trak" && pos + 16 <= total_size {
                hero_track_id = u32::from_be_bytes([
                    data[pos + 8],
                    data[pos + 9],
                    data[pos + 10],
                    data[pos + 11],
                ]);
                pair_track_id = u32::from_be_bytes([
                    data[pos + 12],
                    data[pos + 13],
                    data[pos + 14],
                    data[pos + 15],
                ]);
            }
            pos += sub_size;
        }

        Ok(Self {
            eye_config: EyeConfig {
                hero_track_id,
                pair_track_id,
                hero_eye,
            },
        })
    }
}

// ─── Spatial Video Metadata Box (vexu) ───────────────────────────────────────

/// Top-level Apple Spatial Video Metadata Box (`vexu`).
///
/// This box wraps the stereo pair and optional comfort metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct AppleSpatialVideoMeta {
    /// Stereo pair track association.
    pub stereo_pair: StereoPairBox,
    /// Optional comfort/divergence advisory.
    pub comfort: Option<ComfortBox>,
    /// Projection and FOV metadata for the primary track.
    pub projection: ProjectionBox,
}

impl AppleSpatialVideoMeta {
    /// Create standard Apple Spatial Video metadata for a left-hero stereo pair
    /// with rectilinear 90° FOV and default comfort advisory.
    pub fn new_standard(hero_track: u32, pair_track: u32) -> Self {
        Self {
            stereo_pair: StereoPairBox::new(EyeConfig::left_hero(hero_track, pair_track)),
            comfort: Some(ComfortBox::default_advisory()),
            projection: ProjectionBox::rectilinear_90(),
        }
    }

    /// Serialize to ISOBMFF bytes.
    ///
    /// Layout:
    /// ```text
    /// vexu box:
    ///   4 bytes: total size
    ///   4 bytes: fourcc "vexu"
    ///   <eyes box>
    ///   [<cmfy box>]   — optional
    ///   <prhd box>
    /// ```
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut inner: Vec<u8> = Vec::new();
        inner.extend_from_slice(&self.stereo_pair.to_bytes());
        if let Some(ref cmfy) = self.comfort {
            inner.extend_from_slice(&cmfy.to_bytes());
        }
        inner.extend_from_slice(&self.projection.to_bytes());

        let total_size = 8u32 + inner.len() as u32;
        let mut out = Vec::with_capacity(total_size as usize);
        out.extend_from_slice(&total_size.to_be_bytes());
        out.extend_from_slice(b"vexu");
        out.extend_from_slice(&inner);
        out
    }

    /// Parse an `AppleSpatialVideoMeta` from its serialised `vexu` bytes.
    ///
    /// # Errors
    /// Returns [`VrError::ParseError`] if the bytes are malformed or required
    /// sub-boxes are missing.
    pub fn parse(data: &[u8]) -> Result<Self, VrError> {
        if data.len() < 8 {
            return Err(VrError::ParseError("vexu box too short".into()));
        }
        let total_size = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize;
        if &data[4..8] != b"vexu" {
            return Err(VrError::ParseError("expected fourcc 'vexu'".into()));
        }
        if data.len() < total_size {
            return Err(VrError::ParseError("vexu box truncated".into()));
        }

        let mut pos = 8usize;
        let mut stereo_pair: Option<StereoPairBox> = None;
        let mut comfort: Option<ComfortBox> = None;
        let mut projection: Option<ProjectionBox> = None;

        while pos + 8 <= total_size {
            let sub_size =
                u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                    as usize;
            if sub_size < 8 || pos + sub_size > total_size {
                break;
            }
            let fourcc = &data[pos..pos + 4 + 4]; // 8 bytes for size+fourcc check
            let fc = &data[pos + 4..pos + 8];

            if fc == b"eyes" {
                stereo_pair = Some(StereoPairBox::parse(&data[pos..pos + sub_size])?);
            } else if fc == b"cmfy" {
                comfort = Some(ComfortBox::parse(&data[pos..pos + sub_size])?);
            } else if fc == b"prhd" {
                projection = Some(ProjectionBox::parse(&data[pos..pos + sub_size])?);
            }

            let _ = fourcc; // used for FC detection above
            pos += sub_size;
        }

        let stereo_pair =
            stereo_pair.ok_or_else(|| VrError::ParseError("missing 'eyes' box".into()))?;
        let projection =
            projection.ok_or_else(|| VrError::ParseError("missing 'prhd' box".into()))?;

        Ok(Self {
            stereo_pair,
            comfort,
            projection,
        })
    }
}

impl Default for AppleSpatialVideoMeta {
    fn default() -> Self {
        Self::new_standard(1, 2)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── HeroEye ──────────────────────────────────────────────────────────────

    #[test]
    fn hero_eye_ids() {
        assert_eq!(HeroEye::Left.eye_id(), 1);
        assert_eq!(HeroEye::Right.eye_id(), 2);
    }

    // ── SpatialProjection ─────────────────────────────────────────────────────

    #[test]
    fn spatial_projection_fourccs() {
        assert_eq!(SpatialProjection::Rectilinear.fourcc(), *b"rect");
        assert_eq!(SpatialProjection::Equirectangular.fourcc(), *b"equi");
        assert_eq!(SpatialProjection::HalfEquirectangular.fourcc(), *b"hequ");
    }

    // ── ProjectionBox ─────────────────────────────────────────────────────────

    #[test]
    fn projection_box_rectilinear_has_correct_fourcc() {
        let pb = ProjectionBox::rectilinear_90();
        let bytes = pb.to_bytes();
        assert_eq!(&bytes[4..8], b"prhd");
    }

    #[test]
    fn projection_box_size_field_matches_length() {
        let pb = ProjectionBox::rectilinear_90();
        let bytes = pb.to_bytes();
        let size = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(size as usize, bytes.len());
    }

    #[test]
    fn projection_box_contains_prji_fourcc() {
        let pb = ProjectionBox::rectilinear_90();
        let bytes = pb.to_bytes();
        // prji sub-box starts at offset 12 (after prhd header)
        assert_eq!(&bytes[16..20], b"prji");
    }

    #[test]
    fn projection_box_contains_rect_fourcc() {
        let pb = ProjectionBox::rectilinear_90();
        let bytes = pb.to_bytes();
        // rect fourcc at offset 20 (prji payload)
        assert_eq!(&bytes[20..24], b"rect");
    }

    #[test]
    fn projection_box_with_fov_contains_hfov_box() {
        let pb = ProjectionBox::rectilinear_90();
        let bytes = pb.to_bytes();
        // hfov sub-box should be present after prji
        let hfov_pos = bytes.windows(4).position(|w| w == b"hfov");
        assert!(hfov_pos.is_some(), "hfov box not found");
    }

    #[test]
    fn projection_box_without_fov_has_no_hfov() {
        let pb = ProjectionBox {
            projection: SpatialProjection::Equirectangular,
            hfov_deg: None,
        };
        let bytes = pb.to_bytes();
        let hfov_pos = bytes.windows(4).position(|w| w == b"hfov");
        assert!(hfov_pos.is_none());
    }

    #[test]
    fn projection_box_roundtrip_rectilinear() {
        let original = ProjectionBox::rectilinear_90();
        let bytes = original.to_bytes();
        let parsed = ProjectionBox::parse(&bytes).expect("parse ok");
        assert_eq!(parsed.projection, original.projection);
        // FOV should round-trip within floating-point precision
        if let (Some(a), Some(b)) = (original.hfov_deg, parsed.hfov_deg) {
            assert!((a - b).abs() < 0.01, "fov mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn projection_box_roundtrip_equirectangular() {
        let original = ProjectionBox::equirectangular();
        let bytes = original.to_bytes();
        let parsed = ProjectionBox::parse(&bytes).expect("parse ok");
        assert_eq!(parsed.projection, SpatialProjection::Equirectangular);
        assert!(parsed.hfov_deg.is_none());
    }

    #[test]
    fn projection_box_parse_error_wrong_fourcc() {
        let mut bytes = ProjectionBox::rectilinear_90().to_bytes();
        bytes[4] = b'X'; // corrupt fourcc
        assert!(ProjectionBox::parse(&bytes).is_err());
    }

    #[test]
    fn projection_box_parse_error_too_short() {
        assert!(ProjectionBox::parse(&[0u8; 4]).is_err());
    }

    // ── ComfortBox ───────────────────────────────────────────────────────────

    #[test]
    fn comfort_box_size_field_matches_length() {
        let cb = ComfortBox::default_advisory();
        let bytes = cb.to_bytes();
        let size = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(size as usize, bytes.len());
        assert_eq!(bytes.len(), 24);
    }

    #[test]
    fn comfort_box_fourcc_is_cmfy() {
        let cb = ComfortBox::default_advisory();
        let bytes = cb.to_bytes();
        assert_eq!(&bytes[4..8], b"cmfy");
    }

    #[test]
    fn comfort_box_roundtrip() {
        let original = ComfortBox {
            max_positive_disparity: 0.025,
            max_negative_disparity: 0.035,
            divergence_deg: 1.5,
        };
        let bytes = original.to_bytes();
        let parsed = ComfortBox::parse(&bytes).expect("parse ok");
        assert!((parsed.max_positive_disparity - original.max_positive_disparity).abs() < 1e-5);
        assert!((parsed.max_negative_disparity - original.max_negative_disparity).abs() < 1e-5);
        assert!((parsed.divergence_deg - original.divergence_deg).abs() < 1e-5);
    }

    #[test]
    fn comfort_box_parse_error_wrong_fourcc() {
        let mut bytes = ComfortBox::default_advisory().to_bytes();
        bytes[7] = b'X'; // corrupt last byte of fourcc
        assert!(ComfortBox::parse(&bytes).is_err());
    }

    #[test]
    fn comfort_box_parse_error_too_short() {
        assert!(ComfortBox::parse(&[0u8; 8]).is_err());
    }

    // ── StereoPairBox (eyes) ──────────────────────────────────────────────────

    #[test]
    fn stereo_pair_box_fourcc() {
        let spb = StereoPairBox::new(EyeConfig::left_hero(1, 2));
        let bytes = spb.to_bytes();
        assert_eq!(&bytes[4..8], b"eyes");
    }

    #[test]
    fn stereo_pair_box_size_field_matches_length() {
        let spb = StereoPairBox::new(EyeConfig::left_hero(1, 2));
        let bytes = spb.to_bytes();
        let size = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(size as usize, bytes.len());
    }

    #[test]
    fn stereo_pair_box_contains_hero_box() {
        let spb = StereoPairBox::new(EyeConfig::left_hero(1, 2));
        let bytes = spb.to_bytes();
        let hero_pos = bytes.windows(4).position(|w| w == b"hero");
        assert!(hero_pos.is_some());
    }

    #[test]
    fn stereo_pair_box_roundtrip_left_hero() {
        let original = StereoPairBox::new(EyeConfig::left_hero(3, 4));
        let bytes = original.to_bytes();
        let parsed = StereoPairBox::parse(&bytes).expect("parse ok");
        assert_eq!(parsed.eye_config.hero_eye, HeroEye::Left);
        assert_eq!(parsed.eye_config.hero_track_id, 3);
        assert_eq!(parsed.eye_config.pair_track_id, 4);
    }

    #[test]
    fn stereo_pair_box_roundtrip_right_hero() {
        let cfg = EyeConfig {
            hero_track_id: 5,
            pair_track_id: 6,
            hero_eye: HeroEye::Right,
        };
        let original = StereoPairBox::new(cfg);
        let bytes = original.to_bytes();
        let parsed = StereoPairBox::parse(&bytes).expect("parse ok");
        assert_eq!(parsed.eye_config.hero_eye, HeroEye::Right);
        assert_eq!(parsed.eye_config.hero_track_id, 5);
        assert_eq!(parsed.eye_config.pair_track_id, 6);
    }

    #[test]
    fn stereo_pair_box_parse_error_wrong_fourcc() {
        let mut bytes = StereoPairBox::new(EyeConfig::left_hero(1, 2)).to_bytes();
        bytes[4] = b'X';
        assert!(StereoPairBox::parse(&bytes).is_err());
    }

    // ── AppleSpatialVideoMeta (vexu) ──────────────────────────────────────────

    #[test]
    fn vexu_fourcc() {
        let meta = AppleSpatialVideoMeta::default();
        let bytes = meta.to_bytes();
        assert_eq!(&bytes[4..8], b"vexu");
    }

    #[test]
    fn vexu_size_field_matches_length() {
        let meta = AppleSpatialVideoMeta::default();
        let bytes = meta.to_bytes();
        let size = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(size as usize, bytes.len());
    }

    #[test]
    fn vexu_contains_eyes_box() {
        let meta = AppleSpatialVideoMeta::default();
        let bytes = meta.to_bytes();
        let has_eyes = bytes.windows(4).any(|w| w == b"eyes");
        assert!(has_eyes);
    }

    #[test]
    fn vexu_contains_prhd_box() {
        let meta = AppleSpatialVideoMeta::default();
        let bytes = meta.to_bytes();
        let has_prhd = bytes.windows(4).any(|w| w == b"prhd");
        assert!(has_prhd);
    }

    #[test]
    fn vexu_contains_cmfy_when_set() {
        let meta = AppleSpatialVideoMeta::default();
        let bytes = meta.to_bytes();
        let has_cmfy = bytes.windows(4).any(|w| w == b"cmfy");
        assert!(has_cmfy, "comfort box should be present in default meta");
    }

    #[test]
    fn vexu_without_comfort() {
        let meta = AppleSpatialVideoMeta {
            stereo_pair: StereoPairBox::new(EyeConfig::left_hero(1, 2)),
            comfort: None,
            projection: ProjectionBox::equirectangular(),
        };
        let bytes = meta.to_bytes();
        let has_cmfy = bytes.windows(4).any(|w| w == b"cmfy");
        assert!(!has_cmfy);
    }

    #[test]
    fn vexu_roundtrip_standard() {
        let original = AppleSpatialVideoMeta::new_standard(1, 2);
        let bytes = original.to_bytes();
        let parsed = AppleSpatialVideoMeta::parse(&bytes).expect("parse ok");
        assert_eq!(parsed.stereo_pair.eye_config.hero_eye, HeroEye::Left);
        assert_eq!(parsed.stereo_pair.eye_config.hero_track_id, 1);
        assert_eq!(parsed.stereo_pair.eye_config.pair_track_id, 2);
        assert_eq!(parsed.projection.projection, SpatialProjection::Rectilinear);
        assert!(parsed.comfort.is_some());
    }

    #[test]
    fn vexu_roundtrip_equirectangular_no_comfort() {
        let original = AppleSpatialVideoMeta {
            stereo_pair: StereoPairBox::new(EyeConfig::left_hero(3, 4)),
            comfort: None,
            projection: ProjectionBox::equirectangular(),
        };
        let bytes = original.to_bytes();
        let parsed = AppleSpatialVideoMeta::parse(&bytes).expect("parse ok");
        assert_eq!(
            parsed.projection.projection,
            SpatialProjection::Equirectangular
        );
        assert!(parsed.comfort.is_none());
        assert_eq!(parsed.stereo_pair.eye_config.hero_track_id, 3);
    }

    #[test]
    fn vexu_parse_error_wrong_fourcc() {
        let mut bytes = AppleSpatialVideoMeta::default().to_bytes();
        bytes[4] = b'X';
        assert!(AppleSpatialVideoMeta::parse(&bytes).is_err());
    }

    #[test]
    fn vexu_parse_error_too_short() {
        assert!(AppleSpatialVideoMeta::parse(&[0u8; 4]).is_err());
    }
}
