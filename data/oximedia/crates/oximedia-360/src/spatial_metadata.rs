//! Google Spatial Media v2 / XMP metadata and ISOBMFF box helpers.
//!
//! Implements:
//! * [`SpatialMediaV2`] — projection / stereo metadata struct with XMP serialisation
//! * [`Sv3dBox`]        — ISOBMFF `sv3d` box for 360° metadata in MP4
//! * [`StereoVideoBox`] — ISOBMFF stereo video box

use crate::stereo::StereoLayout;

// ─── Projection type ─────────────────────────────────────────────────────────

/// The projection type used by a 360° video.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProjectionType {
    Equirectangular,
    CubemapEquiangular,
    Fisheye,
    Mesh,
    Unknown,
}

impl ProjectionType {
    fn as_gpano_str(&self) -> &'static str {
        match self {
            ProjectionType::Equirectangular => "equirectangular",
            ProjectionType::CubemapEquiangular => "cubemap",
            ProjectionType::Fisheye => "fisheye",
            ProjectionType::Mesh => "mesh",
            ProjectionType::Unknown => "unknown",
        }
    }

    fn from_gpano_str(s: &str) -> ProjectionType {
        match s.to_lowercase().as_str() {
            "equirectangular" => ProjectionType::Equirectangular,
            "cubemap" => ProjectionType::CubemapEquiangular,
            "fisheye" => ProjectionType::Fisheye,
            "mesh" => ProjectionType::Mesh,
            _ => ProjectionType::Unknown,
        }
    }
}

// ─── SpatialMediaV2 ──────────────────────────────────────────────────────────

/// Google Spatial Media v2 metadata describing a 360° / VR video.
///
/// This metadata is typically embedded in the file's XMP sidecar or in a
/// dedicated `uuid` box in the MP4 container.
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialMediaV2 {
    /// Spherical projection type.
    pub projection: ProjectionType,
    /// Stereo packing layout.
    pub stereo: StereoLayout,
    /// Initial horizontal viewing direction in degrees.
    pub initial_view_yaw_deg: f32,
    /// Initial vertical viewing direction in degrees.
    pub initial_view_pitch_deg: f32,
}

impl SpatialMediaV2 {
    /// Create an equirectangular mono metadata descriptor.
    pub fn equirectangular_mono() -> Self {
        Self {
            projection: ProjectionType::Equirectangular,
            stereo: StereoLayout::Mono,
            initial_view_yaw_deg: 0.0,
            initial_view_pitch_deg: 0.0,
        }
    }

    /// Serialize to an XMP XML document string using the Google Panorama
    /// namespace (`http://ns.google.com/photos/1.0/panorama/`).
    pub fn to_xmp(&self) -> String {
        let stereo_str = match self.stereo {
            StereoLayout::TopBottom => "top-bottom",
            StereoLayout::LeftRight => "left-right",
            StereoLayout::Alternating => "stereo-custom",
            StereoLayout::Mono => "mono",
        };

        format!(
            r#"<?xpacket begin='' id='W5M0MpCehiHzreSzNTczkc9d'?>
<x:xmpmeta xmlns:x='adobe:ns:meta/' x:xmptk='OxiMedia 360 XMP Toolkit'>
<rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>
<rdf:Description rdf:about=''
  xmlns:GPano='http://ns.google.com/photos/1.0/panorama/'>
  <GPano:UsePanoramaViewer>True</GPano:UsePanoramaViewer>
  <GPano:ProjectionType>{projection}</GPano:ProjectionType>
  <GPano:StereoMode>{stereo}</GPano:StereoMode>
  <GPano:InitialViewHeadingDegrees>{yaw}</GPano:InitialViewHeadingDegrees>
  <GPano:InitialViewPitchDegrees>{pitch}</GPano:InitialViewPitchDegrees>
</rdf:Description>
</rdf:RDF>
</x:xmpmeta>
<?xpacket end='w'?>"#,
            projection = self.projection.as_gpano_str(),
            stereo = stereo_str,
            yaw = self.initial_view_yaw_deg,
            pitch = self.initial_view_pitch_deg,
        )
    }

    /// Parse a `SpatialMediaV2` from an XMP string.
    ///
    /// Uses a minimal string scanner rather than a full XML parser.
    /// Returns `None` if no recognisable GPano metadata is found.
    pub fn parse_xmp(xmp: &str) -> Option<SpatialMediaV2> {
        let projection_str = extract_gpano_value(xmp, "GPano:ProjectionType")?;
        let projection = ProjectionType::from_gpano_str(&projection_str);

        let stereo_str =
            extract_gpano_value(xmp, "GPano:StereoMode").unwrap_or_else(|| "mono".to_string());
        let stereo = match stereo_str.to_lowercase().as_str() {
            "top-bottom" => StereoLayout::TopBottom,
            "left-right" => StereoLayout::LeftRight,
            "stereo-custom" => StereoLayout::Alternating,
            _ => StereoLayout::Mono,
        };

        let yaw = extract_gpano_value(xmp, "GPano:InitialViewHeadingDegrees")
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.0);

        let pitch = extract_gpano_value(xmp, "GPano:InitialViewPitchDegrees")
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.0);

        Some(SpatialMediaV2 {
            projection,
            stereo,
            initial_view_yaw_deg: yaw,
            initial_view_pitch_deg: pitch,
        })
    }
}

/// Extract the text content of `<tag>content</tag>` from `xml`.
fn extract_gpano_value(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = xml.find(&open)?;
    let content_start = start + open.len();
    let end = xml[content_start..].find(&close)?;
    Some(xml[content_start..content_start + end].trim().to_string())
}

// ─── ISOBMFF boxes ───────────────────────────────────────────────────────────

/// ISOBMFF `sv3d` (Spherical Video v2) box.
///
/// Written as a minimal implementation of the Google Spatial Media specification.
/// The box contains a nested `proj` box with a `prji` (projection info) sub-box.
#[derive(Debug, Clone)]
pub struct Sv3dBox {
    /// Projection type indicator string (4 bytes: "equi", "cbmp", etc.).
    pub projection_type_fourcc: [u8; 4],
}

impl Sv3dBox {
    /// Create an equirectangular `sv3d` box.
    pub fn equirectangular() -> Self {
        Self {
            projection_type_fourcc: *b"equi",
        }
    }

    /// Create a cubemap `sv3d` box.
    pub fn cubemap() -> Self {
        Self {
            projection_type_fourcc: *b"cbmp",
        }
    }

    /// Serialize to bytes.
    ///
    /// Layout (all integers big-endian):
    /// ```text
    /// sv3d box:
    ///   4 bytes: total box size (uint32)
    ///   4 bytes: fourcc "sv3d"
    ///   proj sub-box:
    ///     4 bytes: size
    ///     4 bytes: "proj"
    ///     prji sub-box:
    ///       4 bytes: size
    ///       4 bytes: "prji"
    ///       4 bytes: projection fourcc
    /// ```
    pub fn to_bytes(&self) -> Vec<u8> {
        // prji box: 4 (size) + 4 (fourcc) + 4 (projection fourcc) = 12 bytes
        let prji_size: u32 = 12;
        // proj box: 4 (size) + 4 (fourcc) + prji_size = 20 bytes
        let proj_size: u32 = 8 + prji_size;
        // sv3d box: 4 (size) + 4 (fourcc) + proj_size = 28 bytes
        let sv3d_size: u32 = 8 + proj_size;

        let mut out = Vec::with_capacity(sv3d_size as usize);

        // sv3d box header
        out.extend_from_slice(&sv3d_size.to_be_bytes());
        out.extend_from_slice(b"sv3d");

        // proj sub-box header
        out.extend_from_slice(&proj_size.to_be_bytes());
        out.extend_from_slice(b"proj");

        // prji sub-box
        out.extend_from_slice(&prji_size.to_be_bytes());
        out.extend_from_slice(b"prji");
        out.extend_from_slice(&self.projection_type_fourcc);

        out
    }
}

/// ISOBMFF Stereo Video Box (`st3d`).
///
/// Signals the stereo mode of a 360° video track.
#[derive(Debug, Clone)]
pub struct StereoVideoBox {
    /// Stereo mode: 0 = mono, 1 = top-bottom, 2 = left-right, 3 = stereo-custom.
    pub stereo_mode: u8,
}

impl StereoVideoBox {
    /// Create from a [`StereoLayout`].
    pub fn from_layout(layout: StereoLayout) -> Self {
        let stereo_mode = match layout {
            StereoLayout::Mono => 0,
            StereoLayout::TopBottom => 1,
            StereoLayout::LeftRight => 2,
            StereoLayout::Alternating => 3,
        };
        Self { stereo_mode }
    }

    /// Serialize to ISOBMFF bytes (big-endian).
    ///
    /// Layout:
    /// ```text
    /// 4 bytes: box size (uint32, big-endian)
    /// 4 bytes: fourcc "st3d"
    /// 1 byte:  version (0)
    /// 3 bytes: flags (0,0,0)
    /// 1 byte:  stereo_mode
    /// ```
    pub fn to_bytes(&self) -> Vec<u8> {
        // 4 (size) + 4 (fourcc) + 1 (version) + 3 (flags) + 1 (stereo_mode) = 13 bytes
        let size: u32 = 13;
        let mut out = Vec::with_capacity(size as usize);
        out.extend_from_slice(&size.to_be_bytes());
        out.extend_from_slice(b"st3d");
        out.push(0u8); // version
        out.push(0u8); // flags[0]
        out.push(0u8); // flags[1]
        out.push(0u8); // flags[2]
        out.push(self.stereo_mode);
        out
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ProjectionType ───────────────────────────────────────────────────────

    #[test]
    fn projection_type_to_gpano_str() {
        assert_eq!(
            ProjectionType::Equirectangular.as_gpano_str(),
            "equirectangular"
        );
        assert_eq!(ProjectionType::CubemapEquiangular.as_gpano_str(), "cubemap");
        assert_eq!(ProjectionType::Fisheye.as_gpano_str(), "fisheye");
        assert_eq!(ProjectionType::Mesh.as_gpano_str(), "mesh");
        assert_eq!(ProjectionType::Unknown.as_gpano_str(), "unknown");
    }

    #[test]
    fn projection_type_from_gpano_str_roundtrip() {
        for proj in [
            ProjectionType::Equirectangular,
            ProjectionType::CubemapEquiangular,
            ProjectionType::Fisheye,
            ProjectionType::Mesh,
        ] {
            let s = proj.as_gpano_str();
            let back = ProjectionType::from_gpano_str(s);
            assert_eq!(back, proj);
        }
    }

    #[test]
    fn projection_type_unknown_fallback() {
        let p = ProjectionType::from_gpano_str("totally_unknown_projection_xyz");
        assert_eq!(p, ProjectionType::Unknown);
    }

    // ── SpatialMediaV2: to_xmp ───────────────────────────────────────────────

    #[test]
    fn to_xmp_contains_projection_type() {
        let sm = SpatialMediaV2::equirectangular_mono();
        let xmp = sm.to_xmp();
        assert!(xmp.contains("equirectangular"), "XMP: {xmp}");
    }

    #[test]
    fn to_xmp_contains_stereo_mode() {
        let sm = SpatialMediaV2 {
            projection: ProjectionType::Equirectangular,
            stereo: StereoLayout::TopBottom,
            initial_view_yaw_deg: 0.0,
            initial_view_pitch_deg: 0.0,
        };
        let xmp = sm.to_xmp();
        assert!(xmp.contains("top-bottom"), "XMP: {xmp}");
    }

    #[test]
    fn to_xmp_contains_xpacket_header() {
        let sm = SpatialMediaV2::equirectangular_mono();
        let xmp = sm.to_xmp();
        assert!(xmp.contains("<?xpacket"));
        assert!(xmp.contains("W5M0MpCehiHzreSzNTczkc9d"));
    }

    #[test]
    fn to_xmp_contains_gpano_namespace() {
        let sm = SpatialMediaV2::equirectangular_mono();
        let xmp = sm.to_xmp();
        assert!(xmp.contains("ns.google.com/photos/1.0/panorama/"));
    }

    #[test]
    fn to_xmp_contains_initial_view_angles() {
        let sm = SpatialMediaV2 {
            projection: ProjectionType::Equirectangular,
            stereo: StereoLayout::Mono,
            initial_view_yaw_deg: 45.0,
            initial_view_pitch_deg: -15.0,
        };
        let xmp = sm.to_xmp();
        assert!(xmp.contains("45"), "should contain yaw 45: {xmp}");
        assert!(xmp.contains("-15"), "should contain pitch -15: {xmp}");
    }

    // ── SpatialMediaV2: parse_xmp ────────────────────────────────────────────

    #[test]
    fn parse_xmp_roundtrip_equirect_mono() {
        let original = SpatialMediaV2 {
            projection: ProjectionType::Equirectangular,
            stereo: StereoLayout::Mono,
            initial_view_yaw_deg: 0.0,
            initial_view_pitch_deg: 0.0,
        };
        let xmp = original.to_xmp();
        let parsed = SpatialMediaV2::parse_xmp(&xmp).expect("should parse");
        assert_eq!(parsed.projection, original.projection);
        assert_eq!(parsed.stereo, original.stereo);
    }

    #[test]
    fn parse_xmp_roundtrip_topbottom() {
        let original = SpatialMediaV2 {
            projection: ProjectionType::Equirectangular,
            stereo: StereoLayout::TopBottom,
            initial_view_yaw_deg: 90.0,
            initial_view_pitch_deg: 10.0,
        };
        let xmp = original.to_xmp();
        let parsed = SpatialMediaV2::parse_xmp(&xmp).expect("should parse");
        assert_eq!(parsed.stereo, StereoLayout::TopBottom);
        assert!((parsed.initial_view_yaw_deg - 90.0).abs() < 0.5);
        assert!((parsed.initial_view_pitch_deg - 10.0).abs() < 0.5);
    }

    #[test]
    fn parse_xmp_returns_none_for_empty_string() {
        let result = SpatialMediaV2::parse_xmp("");
        assert!(result.is_none());
    }

    #[test]
    fn parse_xmp_returns_none_for_non_gpano_xml() {
        let xml = "<rdf:RDF><rdf:Description></rdf:Description></rdf:RDF>";
        let result = SpatialMediaV2::parse_xmp(xml);
        assert!(result.is_none());
    }

    #[test]
    fn parse_xmp_fisheye_projection() {
        let original = SpatialMediaV2 {
            projection: ProjectionType::Fisheye,
            stereo: StereoLayout::LeftRight,
            initial_view_yaw_deg: 0.0,
            initial_view_pitch_deg: 0.0,
        };
        let xmp = original.to_xmp();
        let parsed = SpatialMediaV2::parse_xmp(&xmp).expect("parse");
        assert_eq!(parsed.projection, ProjectionType::Fisheye);
        assert_eq!(parsed.stereo, StereoLayout::LeftRight);
    }

    // ── Sv3dBox ──────────────────────────────────────────────────────────────

    #[test]
    fn sv3d_box_equirect_length() {
        let b = Sv3dBox::equirectangular();
        let bytes = b.to_bytes();
        // sv3d_size = 28
        assert_eq!(bytes.len(), 28);
    }

    #[test]
    fn sv3d_box_starts_with_sv3d_fourcc() {
        let b = Sv3dBox::equirectangular();
        let bytes = b.to_bytes();
        assert_eq!(&bytes[4..8], b"sv3d");
    }

    #[test]
    fn sv3d_box_contains_equi_fourcc() {
        let b = Sv3dBox::equirectangular();
        let bytes = b.to_bytes();
        // equi fourcc is at position 24..28 (prji payload)
        assert_eq!(&bytes[24..28], b"equi");
    }

    #[test]
    fn sv3d_box_cubemap_fourcc() {
        let b = Sv3dBox::cubemap();
        let bytes = b.to_bytes();
        assert_eq!(&bytes[24..28], b"cbmp");
    }

    #[test]
    fn sv3d_box_size_field_matches_length() {
        let b = Sv3dBox::equirectangular();
        let bytes = b.to_bytes();
        let size = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(size as usize, bytes.len());
    }

    // ── StereoVideoBox ───────────────────────────────────────────────────────

    #[test]
    fn st3d_box_mono_stereo_mode() {
        let b = StereoVideoBox::from_layout(StereoLayout::Mono);
        let bytes = b.to_bytes();
        assert_eq!(bytes[12], 0); // stereo_mode = 0
    }

    #[test]
    fn st3d_box_top_bottom_stereo_mode() {
        let b = StereoVideoBox::from_layout(StereoLayout::TopBottom);
        let bytes = b.to_bytes();
        assert_eq!(bytes[12], 1);
    }

    #[test]
    fn st3d_box_left_right_stereo_mode() {
        let b = StereoVideoBox::from_layout(StereoLayout::LeftRight);
        let bytes = b.to_bytes();
        assert_eq!(bytes[12], 2);
    }

    #[test]
    fn st3d_box_fourcc() {
        let b = StereoVideoBox::from_layout(StereoLayout::Mono);
        let bytes = b.to_bytes();
        assert_eq!(&bytes[4..8], b"st3d");
    }

    #[test]
    fn st3d_box_size_field() {
        let b = StereoVideoBox::from_layout(StereoLayout::Mono);
        let bytes = b.to_bytes();
        let size = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(size as usize, bytes.len());
    }

    #[test]
    fn st3d_box_version_is_zero() {
        let b = StereoVideoBox::from_layout(StereoLayout::Mono);
        let bytes = b.to_bytes();
        assert_eq!(bytes[8], 0); // version
    }
}
