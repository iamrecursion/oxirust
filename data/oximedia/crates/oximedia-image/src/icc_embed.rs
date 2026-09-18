//! ICC profile embedding: attach, strip, and convert colour profiles.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_truncation)]

use std::fmt;

// ── Known colour spaces ───────────────────────────────────────────────────────

/// Well-known ICC colour-space identifiers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IccColorSpace {
    /// sRGB IEC 61966-2-1.
    Srgb,
    /// Linear sRGB (no gamma).
    LinearSrgb,
    /// Adobe RGB (1998).
    AdobeRgb,
    /// Display P3 (D65).
    DisplayP3,
    /// DCI P3 (D60).
    DciP3,
    /// Rec. 2020.
    Rec2020,
    /// ProPhoto RGB (ROMM RGB).
    ProPhotoRgb,
    /// Generic grayscale.
    Grayscale,
    /// CMYK.
    Cmyk,
    /// Unknown / custom profile.
    Unknown,
}

impl fmt::Display for IccColorSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Srgb => "sRGB",
            Self::LinearSrgb => "Linear sRGB",
            Self::AdobeRgb => "Adobe RGB (1998)",
            Self::DisplayP3 => "Display P3",
            Self::DciP3 => "DCI P3",
            Self::Rec2020 => "Rec. 2020",
            Self::ProPhotoRgb => "ProPhoto RGB",
            Self::Grayscale => "Grayscale",
            Self::Cmyk => "CMYK",
            Self::Unknown => "Unknown",
        };
        f.write_str(name)
    }
}

// ── ICC profile blob ──────────────────────────────────────────────────────────

/// An ICC profile blob with optional colour-space hint.
#[derive(Clone, Debug, PartialEq)]
pub struct IccProfile {
    /// Raw ICC profile bytes.
    pub data: Vec<u8>,
    /// Colour space inferred from the profile header, if known.
    pub color_space: IccColorSpace,
    /// Human-readable description extracted from the profile.
    pub description: Option<String>,
}

impl IccProfile {
    /// Construct a profile from raw bytes.
    ///
    /// This does a best-effort parse of the 4-byte colour-space field in the
    /// ICC v2/v4 header at offset 16.
    #[must_use]
    pub fn from_bytes(data: Vec<u8>) -> Self {
        let color_space = if data.len() >= 20 {
            let tag = &data[16..20];
            match tag {
                b"RGB " => IccColorSpace::Srgb,
                b"GRAY" => IccColorSpace::Grayscale,
                b"CMYK" => IccColorSpace::Cmyk,
                _ => IccColorSpace::Unknown,
            }
        } else {
            IccColorSpace::Unknown
        };

        Self {
            data,
            color_space,
            description: None,
        }
    }

    /// Returns the profile size in bytes.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the profile blob is non-empty.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.data.len() >= 128
    }

    /// Minimal synthetic sRGB stub for testing (not a real ICC profile).
    #[cfg(test)]
    fn stub_srgb() -> Self {
        let mut data = vec![0u8; 128];
        // Set colour-space field at offset 16.
        data[16..20].copy_from_slice(b"RGB ");
        Self {
            data,
            color_space: IccColorSpace::Srgb,
            description: Some("sRGB IEC61966-2.1".to_string()),
        }
    }
}

// ── Embed / strip operations ──────────────────────────────────────────────────

/// Errors that can occur during profile operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IccEmbedError {
    /// The image container already carries a profile.
    ProfileAlreadyPresent,
    /// No profile is present; cannot strip.
    NoProfilePresent,
    /// Profile data is malformed.
    InvalidProfile(String),
    /// Source and destination colour spaces are incompatible.
    IncompatibleSpaces(IccColorSpace, IccColorSpace),
}

impl fmt::Display for IccEmbedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProfileAlreadyPresent => write!(f, "a profile is already embedded"),
            Self::NoProfilePresent => write!(f, "no embedded profile found"),
            Self::InvalidProfile(msg) => write!(f, "invalid ICC profile: {msg}"),
            Self::IncompatibleSpaces(s, d) => {
                write!(f, "incompatible colour spaces: {s} -> {d}")
            }
        }
    }
}

/// A lightweight image container used for profile attachment tests.
#[derive(Clone, Debug, Default)]
pub struct ImageWithProfile {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Pixel data (format-agnostic byte buffer).
    pub pixels: Vec<u8>,
    /// Embedded ICC profile, if any.
    pub profile: Option<IccProfile>,
}

impl ImageWithProfile {
    /// Create a new image without a profile.
    #[must_use]
    pub fn new(width: u32, height: u32, pixels: Vec<u8>) -> Self {
        Self {
            width,
            height,
            pixels,
            profile: None,
        }
    }

    /// Attach an ICC profile, returning an error if one is already present.
    ///
    /// # Errors
    ///
    /// Returns [`IccEmbedError::ProfileAlreadyPresent`] if a profile is already embedded.
    /// Returns [`IccEmbedError::InvalidProfile`] if the profile blob is too small.
    pub fn embed_profile(&mut self, profile: IccProfile) -> Result<(), IccEmbedError> {
        if self.profile.is_some() {
            return Err(IccEmbedError::ProfileAlreadyPresent);
        }
        if !profile.is_valid() {
            return Err(IccEmbedError::InvalidProfile(
                "profile blob smaller than 128 bytes".to_string(),
            ));
        }
        self.profile = Some(profile);
        Ok(())
    }

    /// Remove the embedded profile.
    ///
    /// # Errors
    ///
    /// Returns [`IccEmbedError::NoProfilePresent`] if there is no profile to remove.
    pub fn strip_profile(&mut self) -> Result<IccProfile, IccEmbedError> {
        self.profile.take().ok_or(IccEmbedError::NoProfilePresent)
    }

    /// Replace the current profile with `new_profile`, performing a
    /// colour-space compatibility check and transforming pixel data as needed.
    ///
    /// CMYK↔RGB conversion is performed in-place using component-based transform:
    /// - CMYK→RGB: R = 255*(1-C)*(1-K), G = 255*(1-M)*(1-K), B = 255*(1-Y)*(1-K)
    /// - RGB→CMYK: K = 1 - max(R,G,B); C,M,Y derived from GCR
    ///
    /// Grayscale↔CMYK conversion is not supported and returns [`IccEmbedError::IncompatibleSpaces`].
    ///
    /// # Errors
    ///
    /// Returns [`IccEmbedError::NoProfilePresent`] if no source profile exists.
    /// Returns [`IccEmbedError::IncompatibleSpaces`] for unsupported space pairs.
    pub fn convert_on_embed(
        &mut self,
        new_profile: IccProfile,
    ) -> Result<IccColorSpace, IccEmbedError> {
        let old_cs = self
            .profile
            .as_ref()
            .map(|p| p.color_space)
            .ok_or(IccEmbedError::NoProfilePresent)?;

        let new_cs = new_profile.color_space;

        // Determine whether conversion involves CMYK
        let old_is_cmyk = old_cs == IccColorSpace::Cmyk;
        let new_is_cmyk = new_cs == IccColorSpace::Cmyk;
        let old_is_rgb_family = is_rgb_family(old_cs);
        let new_is_rgb_family = is_rgb_family(new_cs);

        match (old_is_cmyk, new_is_cmyk) {
            (true, false) => {
                // CMYK → RGB family
                if !new_is_rgb_family {
                    return Err(IccEmbedError::IncompatibleSpaces(old_cs, new_cs));
                }
                cmyk_to_rgb(&mut self.pixels);
            }
            (false, true) => {
                // RGB family → CMYK
                if !old_is_rgb_family {
                    return Err(IccEmbedError::IncompatibleSpaces(old_cs, new_cs));
                }
                rgb_to_cmyk(&mut self.pixels);
            }
            _ => {
                // Same family (CMYK→CMYK or RGB→RGB): no pixel transform needed
            }
        }

        self.profile = Some(new_profile);
        Ok(new_cs)
    }
}

// ── Colour-space helpers ──────────────────────────────────────────────────────

/// Returns `true` for any RGB-family colour space.
fn is_rgb_family(cs: IccColorSpace) -> bool {
    matches!(
        cs,
        IccColorSpace::Srgb
            | IccColorSpace::LinearSrgb
            | IccColorSpace::AdobeRgb
            | IccColorSpace::DisplayP3
            | IccColorSpace::DciP3
            | IccColorSpace::Rec2020
            | IccColorSpace::ProPhotoRgb
    )
}

/// Convert CMYK pixel buffer (4 bytes/pixel) to RGB (3 bytes/pixel) in-place.
///
/// Formula: R = 255*(1-C/255)*(1-K/255), G = 255*(1-M/255)*(1-K/255),
///          B = 255*(1-Y/255)*(1-K/255)
fn cmyk_to_rgb(pixels: &mut Vec<u8>) {
    let n = pixels.len() / 4;
    let mut out = Vec::with_capacity(n * 3);
    for chunk in pixels.chunks_exact(4) {
        let c = chunk[0] as f32 / 255.0;
        let m = chunk[1] as f32 / 255.0;
        let y = chunk[2] as f32 / 255.0;
        let k = chunk[3] as f32 / 255.0;
        out.push(((1.0 - c) * (1.0 - k) * 255.0).round().clamp(0.0, 255.0) as u8);
        out.push(((1.0 - m) * (1.0 - k) * 255.0).round().clamp(0.0, 255.0) as u8);
        out.push(((1.0 - y) * (1.0 - k) * 255.0).round().clamp(0.0, 255.0) as u8);
    }
    *pixels = out;
}

/// Convert RGB pixel buffer (3 bytes/pixel) to CMYK (4 bytes/pixel) in-place.
///
/// GCR (Grey Component Replacement): K = 1 - max(R/255, G/255, B/255).
fn rgb_to_cmyk(pixels: &mut Vec<u8>) {
    let n = pixels.len() / 3;
    let mut out = Vec::with_capacity(n * 4);
    for chunk in pixels.chunks_exact(3) {
        let r = chunk[0] as f32 / 255.0;
        let g = chunk[1] as f32 / 255.0;
        let b = chunk[2] as f32 / 255.0;
        let k = 1.0_f32 - r.max(g).max(b);
        let denom = 1.0 - k;
        let (c, m, y) = if denom < 1e-6 {
            (0.0_f32, 0.0_f32, 0.0_f32)
        } else {
            (
                (1.0 - r - k) / denom,
                (1.0 - g - k) / denom,
                (1.0 - b - k) / denom,
            )
        };
        out.push((c * 255.0).round().clamp(0.0, 255.0) as u8);
        out.push((m * 255.0).round().clamp(0.0, 255.0) as u8);
        out.push((y * 255.0).round().clamp(0.0, 255.0) as u8);
        out.push((k * 255.0).round().clamp(0.0, 255.0) as u8);
    }
    *pixels = out;
}

// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn make_image() -> ImageWithProfile {
        ImageWithProfile::new(4, 4, vec![128u8; 4 * 4 * 3])
    }

    #[test]
    fn test_icc_color_space_display_srgb() {
        assert_eq!(IccColorSpace::Srgb.to_string(), "sRGB");
    }

    #[test]
    fn test_icc_color_space_display_adobe() {
        assert_eq!(IccColorSpace::AdobeRgb.to_string(), "Adobe RGB (1998)");
    }

    #[test]
    fn test_icc_profile_from_bytes_rgb() {
        let mut data = vec![0u8; 128];
        data[16..20].copy_from_slice(b"RGB ");
        let p = IccProfile::from_bytes(data);
        assert_eq!(p.color_space, IccColorSpace::Srgb);
    }

    #[test]
    fn test_icc_profile_from_bytes_gray() {
        let mut data = vec![0u8; 128];
        data[16..20].copy_from_slice(b"GRAY");
        let p = IccProfile::from_bytes(data);
        assert_eq!(p.color_space, IccColorSpace::Grayscale);
    }

    #[test]
    fn test_icc_profile_from_bytes_short() {
        let data = vec![0u8; 10];
        let p = IccProfile::from_bytes(data);
        assert_eq!(p.color_space, IccColorSpace::Unknown);
    }

    #[test]
    fn test_icc_profile_is_valid() {
        let p = IccProfile::stub_srgb();
        assert!(p.is_valid());
    }

    #[test]
    fn test_icc_profile_is_valid_small() {
        let p = IccProfile::from_bytes(vec![0u8; 64]);
        assert!(!p.is_valid());
    }

    #[test]
    fn test_embed_profile_success() {
        let mut img = make_image();
        let p = IccProfile::stub_srgb();
        assert!(img.embed_profile(p).is_ok());
        assert!(img.profile.is_some());
    }

    #[test]
    fn test_embed_profile_already_present() {
        let mut img = make_image();
        img.embed_profile(IccProfile::stub_srgb())
            .expect("should succeed in test");
        let result = img.embed_profile(IccProfile::stub_srgb());
        assert_eq!(result, Err(IccEmbedError::ProfileAlreadyPresent));
    }

    #[test]
    fn test_embed_invalid_profile() {
        let mut img = make_image();
        let small = IccProfile::from_bytes(vec![0u8; 64]);
        assert!(matches!(
            img.embed_profile(small),
            Err(IccEmbedError::InvalidProfile(_))
        ));
    }

    #[test]
    fn test_strip_profile_success() {
        let mut img = make_image();
        img.embed_profile(IccProfile::stub_srgb())
            .expect("should succeed in test");
        let stripped = img.strip_profile().expect("should succeed in test");
        assert_eq!(stripped.color_space, IccColorSpace::Srgb);
        assert!(img.profile.is_none());
    }

    #[test]
    fn test_strip_profile_none() {
        let mut img = make_image();
        let result = img.strip_profile();
        assert_eq!(result, Err(IccEmbedError::NoProfilePresent));
    }

    #[test]
    fn test_convert_on_embed_no_source() {
        let mut img = make_image();
        let result = img.convert_on_embed(IccProfile::stub_srgb());
        assert_eq!(result, Err(IccEmbedError::NoProfilePresent));
    }

    #[test]
    fn test_convert_on_embed_success() {
        let mut img = make_image();
        img.embed_profile(IccProfile::stub_srgb())
            .expect("should succeed in test");

        let mut p3_data = vec![0u8; 128];
        p3_data[16..20].copy_from_slice(b"RGB ");
        let p3 = IccProfile {
            data: p3_data,
            color_space: IccColorSpace::DisplayP3,
            description: Some("Display P3".into()),
        };

        let new_cs = img.convert_on_embed(p3).expect("should succeed in test");
        assert_eq!(new_cs, IccColorSpace::DisplayP3);
    }

    #[test]
    fn test_convert_on_embed_incompatible_grayscale_to_cmyk() {
        // Grayscale↔CMYK is explicitly unsupported.
        let mut img = ImageWithProfile::new(2, 2, vec![128u8; 4]);

        let mut gray_data = vec![0u8; 128];
        gray_data[16..20].copy_from_slice(b"GRAY");
        let gray_profile = IccProfile {
            data: gray_data,
            color_space: IccColorSpace::Grayscale,
            description: None,
        };
        img.embed_profile(gray_profile)
            .expect("should succeed in test");

        let mut cmyk_data = vec![0u8; 128];
        cmyk_data[16..20].copy_from_slice(b"CMYK");
        let cmyk = IccProfile {
            data: cmyk_data,
            color_space: IccColorSpace::Cmyk,
            description: None,
        };

        let result = img.convert_on_embed(cmyk);
        assert!(matches!(
            result,
            Err(IccEmbedError::IncompatibleSpaces(_, _))
        ));
    }

    fn stub_cmyk_profile() -> IccProfile {
        let mut data = vec![0u8; 128];
        data[16..20].copy_from_slice(b"CMYK");
        IccProfile {
            data,
            color_space: IccColorSpace::Cmyk,
            description: Some("Test CMYK".into()),
        }
    }

    #[test]
    fn test_convert_cmyk_to_rgb_black_pixel() {
        // CMYK (0,0,0,255) = pure black → RGB (0,0,0)
        let mut img = ImageWithProfile::new(1, 1, vec![0u8, 0u8, 0u8, 255u8]);
        img.embed_profile(stub_cmyk_profile())
            .expect("should succeed in test");
        let new_cs = img
            .convert_on_embed(IccProfile::stub_srgb())
            .expect("conversion should succeed");
        assert_eq!(new_cs, IccColorSpace::Srgb);
        assert_eq!(img.pixels.len(), 3);
        assert_eq!(img.pixels, vec![0u8, 0u8, 0u8]);
    }

    #[test]
    fn test_convert_cmyk_to_rgb_white_pixel() {
        // CMYK (0,0,0,0) = pure white → RGB (255,255,255)
        let mut img = ImageWithProfile::new(1, 1, vec![0u8, 0u8, 0u8, 0u8]);
        img.embed_profile(stub_cmyk_profile())
            .expect("should succeed in test");
        img.convert_on_embed(IccProfile::stub_srgb())
            .expect("conversion should succeed");
        assert_eq!(img.pixels, vec![255u8, 255u8, 255u8]);
    }

    #[test]
    fn test_convert_rgb_to_cmyk_white_pixel() {
        // RGB (255,255,255) → CMYK (0,0,0,0)
        let mut img = make_image();
        img.pixels = vec![255u8, 255u8, 255u8];
        img.embed_profile(IccProfile::stub_srgb())
            .expect("should succeed in test");
        img.convert_on_embed(stub_cmyk_profile())
            .expect("conversion should succeed");
        assert_eq!(img.pixels.len(), 4);
        assert_eq!(img.pixels, vec![0u8, 0u8, 0u8, 0u8]);
    }

    #[test]
    fn test_convert_rgb_to_cmyk_black_pixel() {
        // RGB (0,0,0) → CMYK (0,0,0,255)
        let mut img = make_image();
        img.pixels = vec![0u8, 0u8, 0u8];
        img.embed_profile(IccProfile::stub_srgb())
            .expect("should succeed in test");
        img.convert_on_embed(stub_cmyk_profile())
            .expect("conversion should succeed");
        assert_eq!(img.pixels.len(), 4);
        assert_eq!(img.pixels, vec![0u8, 0u8, 0u8, 255u8]);
    }

    #[test]
    fn test_convert_cmyk_to_rgb_multi_pixel() {
        // 2 pixels: (0,0,0,0) white and (0,0,0,255) black
        let pixels = vec![0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 255u8];
        let mut img = ImageWithProfile::new(2, 1, pixels);
        img.embed_profile(stub_cmyk_profile())
            .expect("should succeed in test");
        img.convert_on_embed(IccProfile::stub_srgb())
            .expect("conversion should succeed");
        assert_eq!(img.pixels.len(), 6);
        // First pixel: white
        assert_eq!(&img.pixels[0..3], &[255u8, 255u8, 255u8]);
        // Second pixel: black
        assert_eq!(&img.pixels[3..6], &[0u8, 0u8, 0u8]);
    }

    #[test]
    fn test_cmyk_round_trip_approximate() {
        // Round-trip RGB→CMYK→RGB should be close (within ±2 due to float rounding)
        let original = vec![128u8, 64u8, 200u8];
        let mut img = ImageWithProfile::new(1, 1, original.clone());
        img.embed_profile(IccProfile::stub_srgb())
            .expect("should succeed in test");
        img.convert_on_embed(stub_cmyk_profile())
            .expect("rgb->cmyk should succeed");
        assert_eq!(img.pixels.len(), 4);
        img.convert_on_embed(IccProfile::stub_srgb())
            .expect("cmyk->rgb should succeed");
        assert_eq!(img.pixels.len(), 3);
        for (orig, result) in original.iter().zip(img.pixels.iter()) {
            let diff = (*orig as i32 - *result as i32).unsigned_abs();
            assert!(
                diff <= 2,
                "Channel differs by {diff} (orig={orig}, result={result})"
            );
        }
    }
}
