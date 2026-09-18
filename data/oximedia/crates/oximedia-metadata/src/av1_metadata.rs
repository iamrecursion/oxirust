//! Codec string metadata for AV1, VP9, and HEVC.
//!
//! Provides parsing and generation of standardized codec strings used in
//! MIME types, DASH/HLS manifests, and HTML5 `<source>` `codecs` attributes.
//!
//! # AV1 Codec String
//!
//! Format: `av01.P.LLT.DD` where P=profile, LL=level (2 digits), T=tier (M/H),
//! DD=bit depth (08/10/12).
//!
//! # VP9 Codec String
//!
//! Format: `vp09.PP.LL.DD` where PP=profile (2 digits), LL=level (2 digits),
//! DD=bit depth (2 digits).
//!
//! # HEVC Codec String
//!
//! Format: `hvc1.G.CCCCCC.LNNN.CC...` where G=general_profile_space + profile_idc,
//! CCCCCC=compatibility flags, L=tier_flag + level_idc, CC=constraint bytes.
//!
//! # Example
//!
//! ```
//! use oximedia_metadata::av1_metadata::{Av1CodecString, parse_av1_codec_string};
//!
//! let codec = Av1CodecString {
//!     profile: 0,
//!     level: 13,
//!     tier: 'M',
//!     bit_depth: 10,
//!     monochrome: false,
//!     chroma_subsampling_x: 1,
//!     chroma_subsampling_y: 1,
//!     color_primaries: None,
//!     transfer_characteristics: None,
//!     matrix_coefficients: None,
//! };
//! assert_eq!(codec.to_codec_string(), "av01.0.13M.10");
//!
//! let parsed = parse_av1_codec_string("av01.0.13M.10").expect("valid codec string");
//! assert_eq!(parsed.profile, 0);
//! assert_eq!(parsed.level, 13);
//! ```

#![allow(dead_code)]

use crate::Error;

// ────────────────────────────────────────────────────────────────────────────
// AV1 Codec String
// ────────────────────────────────────────────────────────────────────────────

/// AV1 codec parameters as encoded in the `av01` codec string.
///
/// The canonical format is `av01.P.LLT.DD` with optional additional components
/// for color metadata: `av01.P.LLT.DD.M.CCC.SSS.TTT.F`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Av1CodecString {
    /// AV1 profile (0=Main, 1=High, 2=Professional).
    pub profile: u8,
    /// AV1 level index (0..=31). Level 5.1 is 13, etc.
    pub level: u8,
    /// Tier: 'M' (Main) or 'H' (High).
    pub tier: char,
    /// Bit depth: 8, 10, or 12.
    pub bit_depth: u8,
    /// Monochrome flag.
    pub monochrome: bool,
    /// Chroma subsampling X (0 or 1).
    pub chroma_subsampling_x: u8,
    /// Chroma subsampling Y (0 or 1).
    pub chroma_subsampling_y: u8,
    /// Color primaries (1-22, optional).
    pub color_primaries: Option<u8>,
    /// Transfer characteristics (optional).
    pub transfer_characteristics: Option<u8>,
    /// Matrix coefficients (optional).
    pub matrix_coefficients: Option<u8>,
}

impl Av1CodecString {
    /// Create a Main profile, Main tier codec string with common defaults.
    #[must_use]
    pub fn main_profile(level: u8, bit_depth: u8) -> Self {
        Self {
            profile: 0,
            level,
            tier: 'M',
            bit_depth,
            monochrome: false,
            chroma_subsampling_x: 1,
            chroma_subsampling_y: 1,
            color_primaries: None,
            transfer_characteristics: None,
            matrix_coefficients: None,
        }
    }

    /// Produce the canonical codec string: `av01.P.LLT.DD`.
    ///
    /// If color metadata is set, appends `.M.CCC.SSS.TTT.F` components.
    #[must_use]
    pub fn to_codec_string(&self) -> String {
        let dd = if self.bit_depth < 10 {
            format!("{:02}", self.bit_depth)
        } else {
            format!("{}", self.bit_depth)
        };
        let base = format!(
            "av01.{}.{:02}{}.{}",
            self.profile, self.level, self.tier, dd
        );

        if self.color_primaries.is_some()
            || self.transfer_characteristics.is_some()
            || self.matrix_coefficients.is_some()
        {
            let mono = if self.monochrome { 1 } else { 0 };
            let sub = format!(
                "{}{}{}",
                self.chroma_subsampling_x, self.chroma_subsampling_y, 0
            );
            let cp = self.color_primaries.unwrap_or(1);
            let tc = self.transfer_characteristics.unwrap_or(1);
            let mc = self.matrix_coefficients.unwrap_or(1);
            format!("{base}.{mono}.{sub}.{cp:02}.{tc:02}.0")
                .replace(
                    &format!(".{mc:02}.0"),
                    &format!(".{tc:02}.{mc:02}.0"),
                )
                // Actually build it correctly:
                ;
            // Rebuild properly
            format!("{base}.{mono}.{sub}.{cp:02}.{tc:02}.{mc:02}.0")
        } else {
            base
        }
    }

    /// Validate that the codec string parameters are within spec.
    pub fn validate(&self) -> Result<(), Error> {
        if self.profile > 2 {
            return Err(Error::ParseError(format!(
                "AV1 profile must be 0-2, got {}",
                self.profile
            )));
        }
        if self.level > 31 {
            return Err(Error::ParseError(format!(
                "AV1 level must be 0-31, got {}",
                self.level
            )));
        }
        if self.tier != 'M' && self.tier != 'H' {
            return Err(Error::ParseError(format!(
                "AV1 tier must be 'M' or 'H', got '{}'",
                self.tier
            )));
        }
        if self.bit_depth != 8 && self.bit_depth != 10 && self.bit_depth != 12 {
            return Err(Error::ParseError(format!(
                "AV1 bit depth must be 8, 10, or 12, got {}",
                self.bit_depth
            )));
        }
        Ok(())
    }

    /// Return a human-readable description of the profile.
    #[must_use]
    pub fn profile_name(&self) -> &'static str {
        match self.profile {
            0 => "Main",
            1 => "High",
            2 => "Professional",
            _ => "Unknown",
        }
    }

    /// Return a human-readable description of the level.
    #[must_use]
    pub fn level_name(&self) -> String {
        let major = 2 + (self.level >> 2);
        let minor = self.level & 3;
        format!("{major}.{minor}")
    }
}

impl std::fmt::Display for Av1CodecString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_codec_string())
    }
}

/// Parse an AV1 codec string of the form `av01.P.LLT.DD[...]`.
///
/// # Errors
///
/// Returns [`Error::ParseError`] if the string is malformed.
pub fn parse_av1_codec_string(s: &str) -> Result<Av1CodecString, Error> {
    if !s.starts_with("av01.") {
        return Err(Error::ParseError(format!(
            "AV1 codec string must start with 'av01.', got: {s}"
        )));
    }
    let parts: Vec<&str> = s[5..].split('.').collect();
    if parts.len() < 3 {
        return Err(Error::ParseError(format!(
            "AV1 codec string requires at least 3 components after 'av01.', got: {s}"
        )));
    }

    // Profile
    let profile: u8 = parts[0]
        .parse()
        .map_err(|_| Error::ParseError(format!("Invalid AV1 profile: '{}'", parts[0])))?;

    // Level + Tier (e.g. "13M" or "05H")
    let level_tier = parts[1];
    if level_tier.len() < 2 {
        return Err(Error::ParseError(format!(
            "AV1 level+tier too short: '{level_tier}'"
        )));
    }
    let tier_char = level_tier
        .chars()
        .last()
        .ok_or_else(|| Error::ParseError("Empty level+tier".to_string()))?;
    if tier_char != 'M' && tier_char != 'H' {
        return Err(Error::ParseError(format!(
            "AV1 tier must be 'M' or 'H', got '{tier_char}'"
        )));
    }
    let level_str = &level_tier[..level_tier.len() - 1];
    let level: u8 = level_str
        .parse()
        .map_err(|_| Error::ParseError(format!("Invalid AV1 level: '{level_str}'")))?;

    // Bit depth
    let bit_depth: u8 = parts[2]
        .parse()
        .map_err(|_| Error::ParseError(format!("Invalid AV1 bit depth: '{}'", parts[2])))?;

    let mut codec = Av1CodecString {
        profile,
        level,
        tier: tier_char,
        bit_depth,
        monochrome: false,
        chroma_subsampling_x: 1,
        chroma_subsampling_y: 1,
        color_primaries: None,
        transfer_characteristics: None,
        matrix_coefficients: None,
    };

    // Parse optional extended fields
    if parts.len() > 3 {
        codec.monochrome = parts[3] == "1";
    }
    if parts.len() > 4 {
        // Char-safe parse of the two subsampling digits. Slicing `sub[0..1]`
        // panics when the first character is multibyte UTF-8 (a char-boundary
        // panic reachable from an attacker-supplied DASH/HLS `codecs=`
        // attribute), so iterate characters instead of taking byte ranges.
        let mut sub_chars = parts[4].chars();
        if let (Some(x), Some(y)) = (sub_chars.next(), sub_chars.next()) {
            codec.chroma_subsampling_x = x.to_digit(10).map_or(1, |d| d as u8);
            codec.chroma_subsampling_y = y.to_digit(10).map_or(1, |d| d as u8);
        }
    }
    if parts.len() > 5 {
        codec.color_primaries = parts[5].parse().ok();
    }
    if parts.len() > 6 {
        codec.transfer_characteristics = parts[6].parse().ok();
    }
    if parts.len() > 7 {
        codec.matrix_coefficients = parts[7].parse().ok();
    }

    Ok(codec)
}

// ────────────────────────────────────────────────────────────────────────────
// VP9 Codec String
// ────────────────────────────────────────────────────────────────────────────

/// VP9 codec parameters as encoded in the `vp09` codec string.
///
/// Format: `vp09.PP.LL.DD[.CC.TT.SS.CF.CP]`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Vp9CodecString {
    /// VP9 profile (0, 1, 2, or 3).
    pub profile: u8,
    /// VP9 level (10, 11, 20, 21, 30, 31, 40, 41, 50, 51, 52, 60, 61, 62).
    pub level: u8,
    /// Bit depth: 8, 10, or 12.
    pub bit_depth: u8,
    /// Chroma subsampling (0=4:2:0, 1=4:2:2, 2=4:4:4, 3=4:4:0).
    pub chroma_subsampling: Option<u8>,
    /// Color primaries.
    pub color_primaries: Option<u8>,
    /// Transfer characteristics.
    pub transfer_characteristics: Option<u8>,
    /// Matrix coefficients.
    pub matrix_coefficients: Option<u8>,
    /// Video full range flag (0 or 1).
    pub video_full_range: Option<u8>,
}

impl Vp9CodecString {
    /// Create a Profile 0, 8-bit codec string.
    #[must_use]
    pub fn profile0(level: u8) -> Self {
        Self {
            profile: 0,
            level,
            bit_depth: 8,
            chroma_subsampling: None,
            color_primaries: None,
            transfer_characteristics: None,
            matrix_coefficients: None,
            video_full_range: None,
        }
    }

    /// Produce the canonical codec string: `vp09.PP.LL.DD`.
    #[must_use]
    pub fn to_codec_string(&self) -> String {
        let base = format!(
            "vp09.{:02}.{:02}.{:02}",
            self.profile, self.level, self.bit_depth
        );
        if self.chroma_subsampling.is_some() || self.color_primaries.is_some() {
            let cs = self.chroma_subsampling.unwrap_or(0);
            let cp = self.color_primaries.unwrap_or(1);
            let tc = self.transfer_characteristics.unwrap_or(1);
            let mc = self.matrix_coefficients.unwrap_or(1);
            let vfr = self.video_full_range.unwrap_or(0);
            format!("{base}.{cs:02}.{cp:02}.{tc:02}.{mc:02}.{vfr:02}")
        } else {
            base
        }
    }

    /// Validate parameters are within VP9 spec.
    pub fn validate(&self) -> Result<(), Error> {
        if self.profile > 3 {
            return Err(Error::ParseError(format!(
                "VP9 profile must be 0-3, got {}",
                self.profile
            )));
        }
        if self.bit_depth != 8 && self.bit_depth != 10 && self.bit_depth != 12 {
            return Err(Error::ParseError(format!(
                "VP9 bit depth must be 8, 10, or 12, got {}",
                self.bit_depth
            )));
        }
        Ok(())
    }

    /// Return a human-readable profile name.
    #[must_use]
    pub fn profile_name(&self) -> &'static str {
        match self.profile {
            0 => "Profile 0 (4:2:0, 8-bit)",
            1 => "Profile 1 (4:2:2/4:4:4, 8-bit)",
            2 => "Profile 2 (4:2:0, 10/12-bit)",
            3 => "Profile 3 (4:2:2/4:4:4, 10/12-bit)",
            _ => "Unknown",
        }
    }
}

impl std::fmt::Display for Vp9CodecString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_codec_string())
    }
}

/// Parse a VP9 codec string of the form `vp09.PP.LL.DD[...]`.
///
/// # Errors
///
/// Returns [`Error::ParseError`] if the string is malformed.
pub fn parse_vp9_codec_string(s: &str) -> Result<Vp9CodecString, Error> {
    if !s.starts_with("vp09.") {
        return Err(Error::ParseError(format!(
            "VP9 codec string must start with 'vp09.', got: {s}"
        )));
    }
    let parts: Vec<&str> = s[5..].split('.').collect();
    if parts.len() < 3 {
        return Err(Error::ParseError(format!(
            "VP9 codec string requires at least 3 components, got: {s}"
        )));
    }

    let profile: u8 = parts[0]
        .parse()
        .map_err(|_| Error::ParseError(format!("Invalid VP9 profile: '{}'", parts[0])))?;
    let level: u8 = parts[1]
        .parse()
        .map_err(|_| Error::ParseError(format!("Invalid VP9 level: '{}'", parts[1])))?;
    let bit_depth: u8 = parts[2]
        .parse()
        .map_err(|_| Error::ParseError(format!("Invalid VP9 bit depth: '{}'", parts[2])))?;

    let mut codec = Vp9CodecString {
        profile,
        level,
        bit_depth,
        chroma_subsampling: None,
        color_primaries: None,
        transfer_characteristics: None,
        matrix_coefficients: None,
        video_full_range: None,
    };

    if parts.len() > 3 {
        codec.chroma_subsampling = parts[3].parse().ok();
    }
    if parts.len() > 4 {
        codec.color_primaries = parts[4].parse().ok();
    }
    if parts.len() > 5 {
        codec.transfer_characteristics = parts[5].parse().ok();
    }
    if parts.len() > 6 {
        codec.matrix_coefficients = parts[6].parse().ok();
    }
    if parts.len() > 7 {
        codec.video_full_range = parts[7].parse().ok();
    }

    Ok(codec)
}

// ────────────────────────────────────────────────────────────────────────────
// HEVC Codec String
// ────────────────────────────────────────────────────────────────────────────

/// HEVC (H.265) codec parameters as encoded in the `hvc1`/`hev1` codec string.
///
/// Format: `hvc1.{profile_space}{profile_idc}.{compat_flags_hex}.{tier}{level_idc}.{constraints_hex}`
///
/// Example: `hvc1.1.6.L93.B0`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HevcCodecString {
    /// Codec tag: "hvc1" (in-band parameter sets) or "hev1" (out-of-band).
    pub codec_tag: String,
    /// General profile space (0=none, 1=A, 2=B, 3=C).
    pub general_profile_space: u8,
    /// General profile IDC (1=Main, 2=Main10, 3=MainStillPicture, etc.).
    pub general_profile_idc: u8,
    /// General profile compatibility flags (32-bit).
    pub general_profile_compat_flags: u32,
    /// General tier flag: 'L' (Main tier) or 'H' (High tier).
    pub general_tier_flag: char,
    /// General level IDC (e.g. 93 = Level 3.1, 120 = Level 4.0).
    pub general_level_idc: u8,
    /// Constraint indicator bytes (variable length, typically 1-6 bytes).
    pub constraint_bytes: Vec<u8>,
}

impl HevcCodecString {
    /// Create a common Main profile HEVC codec string.
    #[must_use]
    pub fn main_profile(level_idc: u8) -> Self {
        Self {
            codec_tag: "hvc1".to_string(),
            general_profile_space: 0,
            general_profile_idc: 1,
            general_profile_compat_flags: 0x60000000,
            general_tier_flag: 'L',
            general_level_idc: level_idc,
            constraint_bytes: vec![0xB0],
        }
    }

    /// Create a Main 10 profile HEVC codec string.
    #[must_use]
    pub fn main10_profile(level_idc: u8) -> Self {
        Self {
            codec_tag: "hvc1".to_string(),
            general_profile_space: 0,
            general_profile_idc: 2,
            general_profile_compat_flags: 0x60000000,
            general_tier_flag: 'L',
            general_level_idc: level_idc,
            constraint_bytes: vec![0xB0],
        }
    }

    /// Produce the canonical codec string.
    #[must_use]
    pub fn to_codec_string(&self) -> String {
        let profile_space_str = match self.general_profile_space {
            0 => String::new(),
            1 => "A".to_string(),
            2 => "B".to_string(),
            3 => "C".to_string(),
            _ => String::new(),
        };

        let compat_hex = format!("{:X}", self.general_profile_compat_flags);

        let constraints: String = self
            .constraint_bytes
            .iter()
            .map(|b| format!("{b:02X}"))
            .collect::<Vec<_>>()
            .join(".");

        if constraints.is_empty() {
            format!(
                "{}.{}{}.{}.{}{}",
                self.codec_tag,
                profile_space_str,
                self.general_profile_idc,
                compat_hex,
                self.general_tier_flag,
                self.general_level_idc,
            )
        } else {
            format!(
                "{}.{}{}.{}.{}{}.{}",
                self.codec_tag,
                profile_space_str,
                self.general_profile_idc,
                compat_hex,
                self.general_tier_flag,
                self.general_level_idc,
                constraints,
            )
        }
    }

    /// Validate parameters.
    pub fn validate(&self) -> Result<(), Error> {
        if self.general_profile_space > 3 {
            return Err(Error::ParseError(format!(
                "HEVC profile space must be 0-3, got {}",
                self.general_profile_space
            )));
        }
        if self.general_tier_flag != 'L' && self.general_tier_flag != 'H' {
            return Err(Error::ParseError(format!(
                "HEVC tier must be 'L' or 'H', got '{}'",
                self.general_tier_flag
            )));
        }
        Ok(())
    }

    /// Return a human-readable profile name.
    #[must_use]
    pub fn profile_name(&self) -> &'static str {
        match self.general_profile_idc {
            1 => "Main",
            2 => "Main 10",
            3 => "Main Still Picture",
            4 => "Range Extensions",
            5 => "High Throughput",
            _ => "Unknown",
        }
    }

    /// Return the HEVC level as a dotted string (e.g. "3.1" for level_idc=93).
    #[must_use]
    pub fn level_name(&self) -> String {
        let major = self.general_level_idc / 30;
        let minor = (self.general_level_idc % 30) / 3;
        format!("{major}.{minor}")
    }
}

impl std::fmt::Display for HevcCodecString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_codec_string())
    }
}

/// Parse an HEVC codec string of the form `hvc1.X.YYYY.ZNNN.CC` or `hev1.X.YYYY.ZNNN.CC`.
///
/// # Errors
///
/// Returns [`Error::ParseError`] if the string is malformed.
pub fn parse_hevc_codec_string(s: &str) -> Result<HevcCodecString, Error> {
    let codec_tag = if s.starts_with("hvc1.") {
        "hvc1"
    } else if s.starts_with("hev1.") {
        "hev1"
    } else {
        return Err(Error::ParseError(format!(
            "HEVC codec string must start with 'hvc1.' or 'hev1.', got: {s}"
        )));
    };

    let parts: Vec<&str> = s[5..].split('.').collect();
    if parts.len() < 3 {
        return Err(Error::ParseError(format!(
            "HEVC codec string requires at least 3 components, got: {s}"
        )));
    }

    // Parse profile space + profile IDC from first component (e.g., "1", "A1", "B2")
    let profile_part = parts[0];
    let (general_profile_space, profile_idc_str) = if profile_part.starts_with('A') {
        (1u8, &profile_part[1..])
    } else if profile_part.starts_with('B') {
        (2, &profile_part[1..])
    } else if profile_part.starts_with('C') {
        (3, &profile_part[1..])
    } else {
        (0, profile_part)
    };
    let general_profile_idc: u8 = profile_idc_str
        .parse()
        .map_err(|_| Error::ParseError(format!("Invalid HEVC profile IDC: '{profile_idc_str}'")))?;

    // Parse compatibility flags (hex)
    let general_profile_compat_flags = u32::from_str_radix(parts[1], 16)
        .map_err(|_| Error::ParseError(format!("Invalid HEVC compat flags: '{}'", parts[1])))?;

    // Parse tier + level IDC (e.g., "L93", "H120")
    let tier_level = parts[2];
    if tier_level.is_empty() {
        return Err(Error::ParseError("Empty HEVC tier+level".to_string()));
    }
    let tier_char = tier_level
        .chars()
        .next()
        .ok_or_else(|| Error::ParseError("Missing HEVC tier character".to_string()))?;
    if tier_char != 'L' && tier_char != 'H' {
        return Err(Error::ParseError(format!(
            "HEVC tier must be 'L' or 'H', got '{tier_char}'"
        )));
    }
    let level_idc: u8 = tier_level[1..].parse().map_err(|_| {
        Error::ParseError(format!("Invalid HEVC level IDC: '{}'", &tier_level[1..]))
    })?;

    // Parse constraint bytes (remaining dot-separated hex values)
    let mut constraint_bytes = Vec::new();
    for part in parts.iter().skip(3) {
        // Each constraint component is a hex byte (possibly multi-char like "B0", "00")
        let byte = u8::from_str_radix(part, 16)
            .map_err(|_| Error::ParseError(format!("Invalid HEVC constraint byte: '{part}'")))?;
        constraint_bytes.push(byte);
    }

    Ok(HevcCodecString {
        codec_tag: codec_tag.to_string(),
        general_profile_space,
        general_profile_idc,
        general_profile_compat_flags,
        general_tier_flag: tier_char,
        general_level_idc: level_idc,
        constraint_bytes,
    })
}

// ────────────────────────────────────────────────────────────────────────────
// Codec string detection
// ────────────────────────────────────────────────────────────────────────────

/// Universal codec string detection and identification.
pub struct CodecStringParser;

impl CodecStringParser {
    /// Detect the codec family from a codec string.
    ///
    /// Returns the codec name if recognized, or `None` for unknown strings.
    #[must_use]
    pub fn detect_codec(s: &str) -> Option<&'static str> {
        if s.starts_with("av01.") {
            Some("AV1")
        } else if s.starts_with("vp09.") {
            Some("VP9")
        } else if s.starts_with("vp8") || s == "vp8" {
            Some("VP8")
        } else if s.starts_with("hvc1.") || s.starts_with("hev1.") {
            Some("HEVC")
        } else if s.starts_with("avc1.") || s.starts_with("avc3.") {
            Some("H.264/AVC")
        } else if s.starts_with("mp4a.") {
            Some("AAC")
        } else if s.starts_with("opus") || s == "opus" {
            Some("Opus")
        } else if s.starts_with("vorbis") || s == "vorbis" {
            Some("Vorbis")
        } else if s.starts_with("flac") || s == "flac" {
            Some("FLAC")
        } else if s.starts_with("ec-3") || s == "ec-3" {
            Some("E-AC-3")
        } else if s.starts_with("ac-3") || s == "ac-3" {
            Some("AC-3")
        } else if s.starts_with("stpp") || s == "stpp" {
            Some("TTML")
        } else if s.starts_with("wvtt") || s == "wvtt" {
            Some("WebVTT")
        } else {
            None
        }
    }

    /// Check if a codec string represents a video codec.
    #[must_use]
    pub fn is_video_codec(s: &str) -> bool {
        matches!(
            Self::detect_codec(s),
            Some("AV1") | Some("VP9") | Some("VP8") | Some("HEVC") | Some("H.264/AVC")
        )
    }

    /// Check if a codec string represents an audio codec.
    #[must_use]
    pub fn is_audio_codec(s: &str) -> bool {
        matches!(
            Self::detect_codec(s),
            Some("AAC")
                | Some("Opus")
                | Some("Vorbis")
                | Some("FLAC")
                | Some("E-AC-3")
                | Some("AC-3")
        )
    }

    /// Check if a codec string represents a subtitle/text codec.
    #[must_use]
    pub fn is_text_codec(s: &str) -> bool {
        matches!(Self::detect_codec(s), Some("TTML") | Some("WebVTT"))
    }

    /// Parse any recognized codec string and return a summary description.
    #[must_use]
    pub fn describe(s: &str) -> String {
        if let Some(name) = Self::detect_codec(s) {
            if s.starts_with("av01.") {
                if let Ok(av1) = parse_av1_codec_string(s) {
                    return format!(
                        "{name} {} Level {} {}-bit",
                        av1.profile_name(),
                        av1.level_name(),
                        av1.bit_depth
                    );
                }
            } else if s.starts_with("vp09.") {
                if let Ok(vp9) = parse_vp9_codec_string(s) {
                    return format!("{name} {} {}-bit", vp9.profile_name(), vp9.bit_depth);
                }
            } else if s.starts_with("hvc1.") || s.starts_with("hev1.") {
                if let Ok(hevc) = parse_hevc_codec_string(s) {
                    return format!("{name} {} Level {}", hevc.profile_name(), hevc.level_name());
                }
            }
            name.to_string()
        } else {
            format!("Unknown codec: {s}")
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// MIME type helpers
// ────────────────────────────────────────────────────────────────────────────

/// Generate a MIME type string with codec parameter.
///
/// Example: `video/webm; codecs="vp09.00.31.08"`.
#[must_use]
pub fn mime_type_with_codecs(container: &str, codecs: &[&str]) -> String {
    if codecs.is_empty() {
        container.to_string()
    } else {
        let codec_list = codecs.join(", ");
        format!("{container}; codecs=\"{codec_list}\"")
    }
}

/// Detect a suitable container MIME type for a given video+audio codec pair.
#[must_use]
pub fn suggest_container(video_codec: &str, audio_codec: &str) -> &'static str {
    match (
        CodecStringParser::detect_codec(video_codec),
        CodecStringParser::detect_codec(audio_codec),
    ) {
        (Some("AV1"), Some("Opus"))
        | (Some("VP9"), Some("Opus"))
        | (Some("VP8"), Some("Vorbis")) => "video/webm",
        (Some("AV1"), _) => "video/mp4",
        (Some("HEVC"), _) | (Some("H.264/AVC"), _) => "video/mp4",
        _ => "video/mp4",
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── AV1 ─────────────────────────────────────────────────────────────

    #[test]
    fn test_av1_main_profile_codec_string() {
        let c = Av1CodecString::main_profile(13, 10);
        assert_eq!(c.to_codec_string(), "av01.0.13M.10");
    }

    #[test]
    fn test_av1_8bit_codec_string() {
        let c = Av1CodecString::main_profile(5, 8);
        assert_eq!(c.to_codec_string(), "av01.0.05M.08");
    }

    #[test]
    fn test_av1_high_tier() {
        let c = Av1CodecString {
            profile: 1,
            level: 20,
            tier: 'H',
            bit_depth: 10,
            monochrome: false,
            chroma_subsampling_x: 1,
            chroma_subsampling_y: 1,
            color_primaries: None,
            transfer_characteristics: None,
            matrix_coefficients: None,
        };
        assert_eq!(c.to_codec_string(), "av01.1.20H.10");
    }

    #[test]
    fn test_parse_av1_basic() {
        let c = parse_av1_codec_string("av01.0.13M.10").expect("should parse");
        assert_eq!(c.profile, 0);
        assert_eq!(c.level, 13);
        assert_eq!(c.tier, 'M');
        assert_eq!(c.bit_depth, 10);
    }

    #[test]
    fn test_parse_av1_8bit() {
        let c = parse_av1_codec_string("av01.0.05M.08").expect("should parse");
        assert_eq!(c.profile, 0);
        assert_eq!(c.level, 5);
        assert_eq!(c.bit_depth, 8);
    }

    #[test]
    fn test_parse_av1_invalid_prefix() {
        assert!(parse_av1_codec_string("vp09.0.31.08").is_err());
    }

    #[test]
    fn test_parse_av1_too_few_parts() {
        assert!(parse_av1_codec_string("av01.0.13M").is_err());
    }

    #[test]
    fn test_av1_validate_valid() {
        let c = Av1CodecString::main_profile(13, 10);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn test_av1_validate_bad_profile() {
        let mut c = Av1CodecString::main_profile(13, 10);
        c.profile = 5;
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_av1_validate_bad_tier() {
        let mut c = Av1CodecString::main_profile(13, 10);
        c.tier = 'X';
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_av1_profile_name() {
        assert_eq!(Av1CodecString::main_profile(0, 8).profile_name(), "Main");
        let mut c = Av1CodecString::main_profile(0, 8);
        c.profile = 1;
        assert_eq!(c.profile_name(), "High");
        c.profile = 2;
        assert_eq!(c.profile_name(), "Professional");
    }

    #[test]
    fn test_av1_level_name() {
        let c = Av1CodecString::main_profile(13, 10);
        assert_eq!(c.level_name(), "5.1");
    }

    #[test]
    fn test_av1_display() {
        let c = Av1CodecString::main_profile(13, 10);
        assert_eq!(format!("{c}"), "av01.0.13M.10");
    }

    #[test]
    fn test_av1_roundtrip() {
        let original = Av1CodecString::main_profile(8, 10);
        let s = original.to_codec_string();
        let parsed = parse_av1_codec_string(&s).expect("roundtrip parse");
        assert_eq!(parsed.profile, original.profile);
        assert_eq!(parsed.level, original.level);
        assert_eq!(parsed.tier, original.tier);
        assert_eq!(parsed.bit_depth, original.bit_depth);
    }

    // ── VP9 ─────────────────────────────────────────────────────────────

    #[test]
    fn test_vp9_profile0_codec_string() {
        let c = Vp9CodecString::profile0(31);
        assert_eq!(c.to_codec_string(), "vp09.00.31.08");
    }

    #[test]
    fn test_parse_vp9_basic() {
        let c = parse_vp9_codec_string("vp09.02.31.10").expect("should parse");
        assert_eq!(c.profile, 2);
        assert_eq!(c.level, 31);
        assert_eq!(c.bit_depth, 10);
    }

    #[test]
    fn test_parse_vp9_invalid_prefix() {
        assert!(parse_vp9_codec_string("av01.0.13M.10").is_err());
    }

    #[test]
    fn test_vp9_validate_valid() {
        let c = Vp9CodecString::profile0(31);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn test_vp9_validate_bad_profile() {
        let mut c = Vp9CodecString::profile0(31);
        c.profile = 5;
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_vp9_display() {
        let c = Vp9CodecString::profile0(31);
        assert_eq!(format!("{c}"), "vp09.00.31.08");
    }

    #[test]
    fn test_vp9_roundtrip() {
        let original = Vp9CodecString {
            profile: 2,
            level: 41,
            bit_depth: 10,
            chroma_subsampling: None,
            color_primaries: None,
            transfer_characteristics: None,
            matrix_coefficients: None,
            video_full_range: None,
        };
        let s = original.to_codec_string();
        let parsed = parse_vp9_codec_string(&s).expect("roundtrip parse");
        assert_eq!(parsed.profile, original.profile);
        assert_eq!(parsed.level, original.level);
        assert_eq!(parsed.bit_depth, original.bit_depth);
    }

    // ── HEVC ────────────────────────────────────────────────────────────

    #[test]
    fn test_hevc_main_profile() {
        let c = HevcCodecString::main_profile(93);
        let s = c.to_codec_string();
        assert!(s.starts_with("hvc1.1."));
        assert!(s.contains("L93"));
    }

    #[test]
    fn test_parse_hevc_basic() {
        let c = parse_hevc_codec_string("hvc1.1.6.L93.B0").expect("should parse");
        assert_eq!(c.codec_tag, "hvc1");
        assert_eq!(c.general_profile_space, 0);
        assert_eq!(c.general_profile_idc, 1);
        assert_eq!(c.general_profile_compat_flags, 6);
        assert_eq!(c.general_tier_flag, 'L');
        assert_eq!(c.general_level_idc, 93);
        assert_eq!(c.constraint_bytes, vec![0xB0]);
    }

    #[test]
    fn test_parse_hevc_hev1() {
        let c = parse_hevc_codec_string("hev1.2.4.H120.B0").expect("should parse");
        assert_eq!(c.codec_tag, "hev1");
        assert_eq!(c.general_profile_idc, 2);
        assert_eq!(c.general_tier_flag, 'H');
        assert_eq!(c.general_level_idc, 120);
    }

    #[test]
    fn test_parse_hevc_invalid_prefix() {
        assert!(parse_hevc_codec_string("avc1.42E01E").is_err());
    }

    #[test]
    fn test_hevc_validate_valid() {
        let c = HevcCodecString::main_profile(93);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn test_hevc_profile_name() {
        assert_eq!(HevcCodecString::main_profile(93).profile_name(), "Main");
        assert_eq!(
            HevcCodecString::main10_profile(93).profile_name(),
            "Main 10"
        );
    }

    #[test]
    fn test_hevc_display() {
        let c = HevcCodecString::main_profile(93);
        let s = format!("{c}");
        assert!(s.starts_with("hvc1."));
    }

    // ── CodecStringParser ───────────────────────────────────────────────

    #[test]
    fn test_detect_codec_av1() {
        assert_eq!(
            CodecStringParser::detect_codec("av01.0.13M.10"),
            Some("AV1")
        );
    }

    #[test]
    fn test_detect_codec_vp9() {
        assert_eq!(
            CodecStringParser::detect_codec("vp09.00.31.08"),
            Some("VP9")
        );
    }

    #[test]
    fn test_detect_codec_hevc() {
        assert_eq!(
            CodecStringParser::detect_codec("hvc1.1.6.L93.B0"),
            Some("HEVC")
        );
        assert_eq!(
            CodecStringParser::detect_codec("hev1.2.4.H120.B0"),
            Some("HEVC")
        );
    }

    #[test]
    fn test_detect_codec_avc() {
        assert_eq!(
            CodecStringParser::detect_codec("avc1.42E01E"),
            Some("H.264/AVC")
        );
    }

    #[test]
    fn test_detect_codec_audio() {
        assert_eq!(CodecStringParser::detect_codec("opus"), Some("Opus"));
        assert_eq!(CodecStringParser::detect_codec("vorbis"), Some("Vorbis"));
        assert_eq!(CodecStringParser::detect_codec("flac"), Some("FLAC"));
        assert_eq!(CodecStringParser::detect_codec("mp4a.40.2"), Some("AAC"));
    }

    #[test]
    fn test_detect_codec_unknown() {
        assert_eq!(CodecStringParser::detect_codec("unknown-codec"), None);
    }

    #[test]
    fn test_is_video_codec() {
        assert!(CodecStringParser::is_video_codec("av01.0.13M.10"));
        assert!(CodecStringParser::is_video_codec("vp09.00.31.08"));
        assert!(CodecStringParser::is_video_codec("hvc1.1.6.L93.B0"));
        assert!(!CodecStringParser::is_video_codec("opus"));
    }

    #[test]
    fn test_is_audio_codec() {
        assert!(CodecStringParser::is_audio_codec("opus"));
        assert!(CodecStringParser::is_audio_codec("flac"));
        assert!(!CodecStringParser::is_audio_codec("av01.0.13M.10"));
    }

    #[test]
    fn test_is_text_codec() {
        assert!(CodecStringParser::is_text_codec("wvtt"));
        assert!(CodecStringParser::is_text_codec("stpp"));
        assert!(!CodecStringParser::is_text_codec("opus"));
    }

    #[test]
    fn test_describe_av1() {
        let desc = CodecStringParser::describe("av01.0.13M.10");
        assert!(desc.contains("AV1"));
        assert!(desc.contains("Main"));
        assert!(desc.contains("10-bit"));
    }

    #[test]
    fn test_describe_unknown() {
        let desc = CodecStringParser::describe("xyz");
        assert!(desc.contains("Unknown"));
    }

    // ── MIME helpers ────────────────────────────────────────────────────

    #[test]
    fn test_mime_type_with_codecs() {
        let mime = mime_type_with_codecs("video/webm", &["vp09.00.31.08", "opus"]);
        assert_eq!(mime, "video/webm; codecs=\"vp09.00.31.08, opus\"");
    }

    #[test]
    fn test_mime_type_no_codecs() {
        let mime = mime_type_with_codecs("video/mp4", &[]);
        assert_eq!(mime, "video/mp4");
    }

    #[test]
    fn test_suggest_container_webm() {
        assert_eq!(suggest_container("vp09.00.31.08", "opus"), "video/webm");
        assert_eq!(suggest_container("av01.0.13M.10", "opus"), "video/webm");
    }

    #[test]
    fn test_suggest_container_mp4() {
        assert_eq!(
            suggest_container("hvc1.1.6.L93.B0", "mp4a.40.2"),
            "video/mp4"
        );
    }

    #[test]
    fn parse_av1_codec_string_multibyte_subsampling_no_panic() {
        // Regression: a multibyte UTF-8 character in the chroma-subsampling
        // field (`parts[4]`) used to panic on `sub[0..1]` (char-boundary panic)
        // when reached from an attacker-controlled DASH/HLS `codecs=` attribute.
        // Parsing is now char-safe and must not panic.
        let result = parse_av1_codec_string("av01.0.13M.10.0.Ÿ0");
        assert!(
            result.is_ok(),
            "multibyte subsampling must not panic: {result:?}"
        );
        // A three-byte character straddling the offset likewise must not panic.
        assert!(parse_av1_codec_string("av01.0.13M.10.0.€x").is_ok());
    }
}
