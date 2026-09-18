//! HDR10+ dynamic metadata (SMPTE ST 2094-40).
//!
//! Provides serialisable structs for per-frame dynamic tone-mapping metadata
//! together with simplified SEI payload encoding and decoding.

use crate::{HdrError, Result};

// ── Structs ───────────────────────────────────────────────────────────────────

/// A single luminance-analysis window for HDR10+ dynamic metadata.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Hdr10PlusWindow {
    /// Upper-left corner of the rectangular window (x, y) in pixels.
    pub window_upper_left: (u16, u16),
    /// Lower-right corner of the rectangular window (x, y) in pixels.
    pub window_lower_right: (u16, u16),
    /// Centre of the analysis ellipse (x, y).
    pub center_of_ellipse: (u16, u16),
    /// Rotation angle of the ellipse in degrees (0–360 stored as u8).
    pub rotation_angle: u8,
    /// Semi-major axis of the bounding ellipse (external).
    pub semimajor_axis_external: u16,
    /// Semi-minor axis of the bounding ellipse (external).
    pub semiminor_axis_external: u16,
    /// Semi-major axis of the inner ellipse.
    pub semimajor_axis_internal: u16,
    /// Semi-minor axis of the inner ellipse.
    pub semiminor_axis_internal: u16,
    /// Overlap process option.
    pub overlap_process_option: u8,
    /// Maximum scene luminance per R/G/B channel (linear, ×1000).
    pub maxscl: [u32; 3],
    /// Average MaxRGB value for this window.
    pub average_maxrgb: u16,
}

/// HDR10+ dynamic metadata for one video frame (SMPTE ST 2094-40).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Hdr10PlusDynamicMetadata {
    /// ITU-T T.35 country code (0xB5 = USA).
    pub country_code: u8,
    /// ITU-T T.35 terminal provider code.
    pub terminal_provider_code: u16,
    /// Application identifier (1 for HDR10+).
    pub application_identifier: u8,
    /// Application version.
    pub application_version: u8,
    /// Number of analysis windows.
    pub num_windows: u8,
    /// Per-window metadata.
    pub windows: Vec<Hdr10PlusWindow>,
    /// Targeted system display maximum luminance, expressed as nits × 10.
    pub targeted_system_display_max_luminance: u32,
    /// Average MaxRGB across the frame.
    pub average_maxrgb: u16,
    /// Distribution values (9 percentile points).
    pub distribution_values: [u16; 9],
    /// Fraction of pixels above the knee point (0–255).
    pub fraction_bright_pixels: u8,
}

/// A dynamic metadata packet associated with a specific video frame.
pub struct DynamicMetadataFrame {
    /// Zero-based frame index in the stream.
    pub frame_index: u64,
    /// Presentation timestamp in milliseconds.
    pub pts_ms: u64,
    /// HDR10+ metadata for this frame.
    pub metadata: Hdr10PlusDynamicMetadata,
}

// ── Implementations ───────────────────────────────────────────────────────────

impl Hdr10PlusWindow {
    fn default_window() -> Self {
        Self {
            window_upper_left: (0, 0),
            window_lower_right: (3840, 2160),
            center_of_ellipse: (1920, 1080),
            rotation_angle: 0,
            semimajor_axis_external: 1920,
            semiminor_axis_external: 1080,
            semimajor_axis_internal: 1920,
            semiminor_axis_internal: 1080,
            overlap_process_option: 0,
            maxscl: [0u32; 3],
            average_maxrgb: 0,
        }
    }

    /// Serialised byte size of a single window record in the simplified payload.
    ///
    /// Layout (little-endian):
    /// - 4 × u16 : upper_left x,y  lower_right x,y
    /// - 2 × u16 : center x,y
    /// - 1 × u8  : rotation_angle
    /// - 4 × u16 : semimajor_ext, semiminor_ext, semimajor_int, semiminor_int
    /// - 1 × u8  : overlap_process_option
    /// - 3 × u32 : maxscl[0..3]
    /// - 1 × u16 : average_maxrgb
    ///
    /// Total = 8 + 4 + 1 + 8 + 1 + 12 + 2 = 36 bytes.
    const ENCODED_SIZE: usize = 36;

    #[allow(unused_assignments)]
    fn encode(&self) -> [u8; Self::ENCODED_SIZE] {
        let mut b = [0u8; Self::ENCODED_SIZE];
        let mut off = 0usize;

        macro_rules! write_u16 {
            ($v:expr) => {{
                b[off..off + 2].copy_from_slice(&($v as u16).to_le_bytes());
                off += 2;
            }};
        }
        macro_rules! write_u32 {
            ($v:expr) => {{
                b[off..off + 4].copy_from_slice(&($v as u32).to_le_bytes());
                off += 4;
            }};
        }
        macro_rules! write_u8 {
            ($v:expr) => {{
                b[off] = $v as u8;
                off += 1;
            }};
        }

        write_u16!(self.window_upper_left.0);
        write_u16!(self.window_upper_left.1);
        write_u16!(self.window_lower_right.0);
        write_u16!(self.window_lower_right.1);
        write_u16!(self.center_of_ellipse.0);
        write_u16!(self.center_of_ellipse.1);
        write_u8!(self.rotation_angle);
        write_u16!(self.semimajor_axis_external);
        write_u16!(self.semiminor_axis_external);
        write_u16!(self.semimajor_axis_internal);
        write_u16!(self.semiminor_axis_internal);
        write_u8!(self.overlap_process_option);
        write_u32!(self.maxscl[0]);
        write_u32!(self.maxscl[1]);
        write_u32!(self.maxscl[2]);
        write_u16!(self.average_maxrgb);

        b
    }

    #[allow(unused_assignments)]
    fn decode(data: &[u8]) -> Result<Self> {
        if data.len() < Self::ENCODED_SIZE {
            return Err(HdrError::MetadataParseError(format!(
                "window record too short: {} bytes (need {})",
                data.len(),
                Self::ENCODED_SIZE
            )));
        }

        let mut off = 0usize;
        macro_rules! read_u16 {
            () => {{
                let v = u16::from_le_bytes([data[off], data[off + 1]]);
                off += 2;
                v
            }};
        }
        macro_rules! read_u32 {
            () => {{
                let v =
                    u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
                off += 4;
                v
            }};
        }
        macro_rules! read_u8 {
            () => {{
                let v = data[off];
                off += 1;
                v
            }};
        }

        let window_upper_left = (read_u16!(), read_u16!());
        let window_lower_right = (read_u16!(), read_u16!());
        let center_of_ellipse = (read_u16!(), read_u16!());
        let rotation_angle = read_u8!();
        let semimajor_axis_external = read_u16!();
        let semiminor_axis_external = read_u16!();
        let semimajor_axis_internal = read_u16!();
        let semiminor_axis_internal = read_u16!();
        let overlap_process_option = read_u8!();
        let maxscl = [read_u32!(), read_u32!(), read_u32!()];
        let average_maxrgb = read_u16!();

        Ok(Self {
            window_upper_left,
            window_lower_right,
            center_of_ellipse,
            rotation_angle,
            semimajor_axis_external,
            semiminor_axis_external,
            semimajor_axis_internal,
            semiminor_axis_internal,
            overlap_process_option,
            maxscl,
            average_maxrgb,
        })
    }
}

impl Hdr10PlusDynamicMetadata {
    /// Create a simple single-window metadata block for the given peak display luminance.
    pub fn new_simple(targeted_nits: u32) -> Self {
        let mut window = Hdr10PlusWindow::default_window();
        window.maxscl = [
            targeted_nits * 1000,
            targeted_nits * 1000,
            targeted_nits * 1000,
        ];
        window.average_maxrgb = (targeted_nits / 2).min(u32::from(u16::MAX)) as u16;

        Self {
            country_code: 0xB5, // USA
            terminal_provider_code: 0x003C,
            application_identifier: 4,
            application_version: 0,
            num_windows: 1,
            windows: vec![window],
            targeted_system_display_max_luminance: targeted_nits * 10,
            average_maxrgb: (targeted_nits / 2).min(u32::from(u16::MAX)) as u16,
            distribution_values: [0u16; 9],
            fraction_bright_pixels: 0,
        }
    }

    /// Encode this metadata to a simplified MPEG-4 ITU-T T.35 SEI payload.
    ///
    /// Layout (little-endian):
    /// - 1 × u8   : country_code
    /// - 1 × u16  : terminal_provider_code
    /// - 1 × u8   : application_identifier
    /// - 1 × u8   : application_version
    /// - 1 × u8   : num_windows
    /// - n × 36 B : window records (n = num_windows)
    /// - 1 × u32  : targeted_system_display_max_luminance
    /// - 1 × u16  : average_maxrgb
    /// - 9 × u16  : distribution_values
    /// - 1 × u8   : fraction_bright_pixels
    pub fn encode(&self) -> Vec<u8> {
        let n = self.windows.len();
        let header_size = 1 + 2 + 1 + 1 + 1; // 6 bytes
        let window_size = n * Hdr10PlusWindow::ENCODED_SIZE;
        let tail_size = 4 + 2 + 9 * 2 + 1; // 25 bytes
        let total = header_size + window_size + tail_size;

        let mut buf = Vec::with_capacity(total);

        buf.push(self.country_code);
        buf.extend_from_slice(&self.terminal_provider_code.to_le_bytes());
        buf.push(self.application_identifier);
        buf.push(self.application_version);
        buf.push(self.num_windows);

        for window in &self.windows {
            buf.extend_from_slice(&window.encode());
        }

        buf.extend_from_slice(&self.targeted_system_display_max_luminance.to_le_bytes());
        buf.extend_from_slice(&self.average_maxrgb.to_le_bytes());
        for dv in &self.distribution_values {
            buf.extend_from_slice(&dv.to_le_bytes());
        }
        buf.push(self.fraction_bright_pixels);

        buf
    }

    /// Decode a payload previously produced by `encode`.
    ///
    /// # Errors
    /// Returns `HdrError::MetadataParseError` if the payload is too short or inconsistent.
    pub fn decode(data: &[u8]) -> Result<Self> {
        const HEADER_MIN: usize = 6; // country_code + prov_code + app_id + app_ver + num_windows
        if data.len() < HEADER_MIN {
            return Err(HdrError::MetadataParseError(format!(
                "HDR10+ payload too short: {} bytes",
                data.len()
            )));
        }

        let country_code = data[0];
        let terminal_provider_code = u16::from_le_bytes([data[1], data[2]]);
        let application_identifier = data[3];
        let application_version = data[4];
        let num_windows = data[5];

        let window_bytes = usize::from(num_windows) * Hdr10PlusWindow::ENCODED_SIZE;
        const TAIL_SIZE: usize = 4 + 2 + 9 * 2 + 1;
        let required = HEADER_MIN + window_bytes + TAIL_SIZE;

        if data.len() < required {
            return Err(HdrError::MetadataParseError(format!(
                "HDR10+ payload too short for {num_windows} windows: need {required}, got {}",
                data.len()
            )));
        }

        let mut windows = Vec::with_capacity(usize::from(num_windows));
        let mut off = HEADER_MIN;
        for _ in 0..num_windows {
            let window = Hdr10PlusWindow::decode(&data[off..])?;
            off += Hdr10PlusWindow::ENCODED_SIZE;
            windows.push(window);
        }

        let targeted_system_display_max_luminance =
            u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
        off += 4;

        let average_maxrgb = u16::from_le_bytes([data[off], data[off + 1]]);
        off += 2;

        let mut distribution_values = [0u16; 9];
        for dv in &mut distribution_values {
            *dv = u16::from_le_bytes([data[off], data[off + 1]]);
            off += 2;
        }

        let fraction_bright_pixels = data[off];

        Ok(Self {
            country_code,
            terminal_provider_code,
            application_identifier,
            application_version,
            num_windows,
            windows,
            targeted_system_display_max_luminance,
            average_maxrgb,
            distribution_values,
            fraction_bright_pixels,
        })
    }
}

impl DynamicMetadataFrame {
    /// Create a `DynamicMetadataFrame` for the given frame index and peak luminance.
    pub fn for_frame(index: u64, pts_ms: u64, peak_nits: u32) -> Self {
        Self {
            frame_index: index,
            pts_ms,
            metadata: Hdr10PlusDynamicMetadata::new_simple(peak_nits),
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_simple_country_code() {
        let m = Hdr10PlusDynamicMetadata::new_simple(1000);
        assert_eq!(m.country_code, 0xB5);
    }

    #[test]
    fn test_new_simple_targeted_luminance() {
        let m = Hdr10PlusDynamicMetadata::new_simple(4000);
        assert_eq!(m.targeted_system_display_max_luminance, 40_000);
    }

    #[test]
    fn test_new_simple_num_windows() {
        let m = Hdr10PlusDynamicMetadata::new_simple(1000);
        assert_eq!(m.num_windows, 1);
        assert_eq!(m.windows.len(), 1);
    }

    #[test]
    fn test_encode_decode_round_trip() {
        let orig = Hdr10PlusDynamicMetadata::new_simple(2000);
        let payload = orig.encode();
        let decoded = Hdr10PlusDynamicMetadata::decode(&payload).expect("decode HDR10+");
        assert_eq!(decoded.country_code, orig.country_code);
        assert_eq!(
            decoded.targeted_system_display_max_luminance,
            orig.targeted_system_display_max_luminance
        );
        assert_eq!(decoded.num_windows, orig.num_windows);
        assert_eq!(decoded.average_maxrgb, orig.average_maxrgb);
    }

    #[test]
    fn test_encode_decode_window_fields() {
        let orig = Hdr10PlusDynamicMetadata::new_simple(1000);
        let payload = orig.encode();
        let decoded = Hdr10PlusDynamicMetadata::decode(&payload).expect("decode window");
        let ow = &orig.windows[0];
        let dw = &decoded.windows[0];
        assert_eq!(dw.window_lower_right, ow.window_lower_right);
        assert_eq!(dw.maxscl, ow.maxscl);
        assert_eq!(dw.average_maxrgb, ow.average_maxrgb);
    }

    #[test]
    fn test_decode_too_short_error() {
        let result = Hdr10PlusDynamicMetadata::decode(&[0u8; 4]);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_incomplete_windows_error() {
        // Exactly header but not enough bytes for 1 window
        let data = [0u8; 6]; // 1 window promised, no window bytes
        let result = Hdr10PlusDynamicMetadata::decode(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_dynamic_metadata_frame_for_frame() {
        let frame = DynamicMetadataFrame::for_frame(42, 1400, 600);
        assert_eq!(frame.frame_index, 42);
        assert_eq!(frame.pts_ms, 1400);
        assert_eq!(frame.metadata.targeted_system_display_max_luminance, 6000);
    }

    #[test]
    fn test_encode_payload_length_consistent() {
        let m = Hdr10PlusDynamicMetadata::new_simple(800);
        let payload = m.encode();
        // Verify we can round-trip and that the length matches expectation
        let expected_len = 6 + Hdr10PlusWindow::ENCODED_SIZE + 4 + 2 + 18 + 1;
        assert_eq!(payload.len(), expected_len);
    }
}
