//! Rights metadata standards: ISRC, ISWC, and ISAN.
//!
//! Provides strongly-typed representations and parsing/formatting for the
//! three principal identifiers used in content-rights management:
//!
//! * **ISRC** – International Standard Recording Code (ISO 3901)
//! * **ISWC** – International Standard Musical Work Code (ISO 15707)
//! * **ISAN** – International Standard Audiovisual Number (ISO 15706)
//!
//! Each type validates its inputs on construction and exposes round-trip
//! `parse` / `to_string` methods.

#![allow(missing_docs)]

// ── Isrc ─────────────────────────────────────────────────────────────────────

/// International Standard Recording Code (ISO 3901).
///
/// Format: `CC-XXX-YY-NNNNN`
/// where
/// - `CC`    = 2-character ISO 3166-1 country code
/// - `XXX`   = 3-character alphanumeric registrant code
/// - `YY`    = 2-digit year of reference
/// - `NNNNN` = 5-digit designation number (00000 – 99999)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Isrc {
    /// ISO 3166-1 alpha-2 country code (uppercase).
    pub country: String,
    /// 3-character alphanumeric registrant code (uppercase).
    pub registrant: String,
    /// 2-digit year of reference.
    pub year: u16,
    /// 5-digit sequential designation number.
    pub designation: u32,
}

impl Isrc {
    /// Construct a new ISRC, validating each component.
    ///
    /// # Errors
    /// Returns `Err` if any component is out of range or uses invalid
    /// characters.
    pub fn new(
        country: &str,
        registrant: &str,
        year: u16,
        designation: u32,
    ) -> Result<Self, String> {
        // Validate country code: exactly 2 ASCII alphabetic characters
        if country.len() != 2 || !country.chars().all(|c| c.is_ascii_alphabetic()) {
            return Err(format!(
                "Invalid ISRC country code '{}': must be 2 ASCII alphabetic characters",
                country
            ));
        }
        // Validate registrant: exactly 3 ASCII alphanumeric characters
        if registrant.len() != 3 || !registrant.chars().all(|c| c.is_ascii_alphanumeric()) {
            return Err(format!(
                "Invalid ISRC registrant '{}': must be 3 ASCII alphanumeric characters",
                registrant
            ));
        }
        // Year of reference: 0–99 (stored as the two-digit year value)
        if year > 99 {
            return Err(format!("Invalid ISRC year {}: must be 0–99", year));
        }
        // Designation: 0–99999
        if designation > 99_999 {
            return Err(format!(
                "Invalid ISRC designation {}: must be 0–99999",
                designation
            ));
        }
        Ok(Self {
            country: country.to_uppercase(),
            registrant: registrant.to_uppercase(),
            year,
            designation,
        })
    }

    /// Parse an ISRC from the canonical `CC-XXX-YY-NNNNN` string.
    ///
    /// # Errors
    /// Returns `Err` with a diagnostic message if the format is invalid.
    pub fn parse(s: &str) -> Result<Self, String> {
        let parts: Vec<&str> = s.split('-').collect();
        if parts.len() != 4 {
            return Err(format!(
                "Invalid ISRC '{}': expected 4 dash-separated components",
                s
            ));
        }
        let country = parts[0];
        let registrant = parts[1];
        let year_str = parts[2];
        let designation_str = parts[3];

        let year: u16 = year_str
            .parse()
            .map_err(|_| format!("Invalid ISRC year segment '{}': not a number", year_str))?;
        let designation: u32 = designation_str.parse().map_err(|_| {
            format!(
                "Invalid ISRC designation segment '{}': not a number",
                designation_str
            )
        })?;

        Self::new(country, registrant, year, designation)
    }

    /// Format this ISRC as `CC-XXX-YY-NNNNN`.
    pub fn to_isrc_string(&self) -> String {
        format!(
            "{}-{}-{:02}-{:05}",
            self.country, self.registrant, self.year, self.designation
        )
    }
}

impl std::fmt::Display for Isrc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_isrc_string())
    }
}

// ── Iswc ─────────────────────────────────────────────────────────────────────

/// International Standard Musical Work Code (ISO 15707).
///
/// Format: `T-NNNNNNNNN-C`
/// where
/// - `T`         = literal prefix character
/// - `NNNNNNNNN` = 9-digit work identifier
/// - `C`         = single check digit computed via a Luhn-like algorithm
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Iswc {
    /// 9-digit work identifier (0 – 999_999_999).
    pub id: u32,
    /// Computed single-digit check value (0–9).
    pub check_digit: u8,
}

impl Iswc {
    /// Construct a new ISWC from a 9-digit work identifier.
    ///
    /// The check digit is computed automatically using the ISO 15707
    /// weighted-sum algorithm.
    pub fn new(id: u32) -> Self {
        let check_digit = Self::compute_check(id);
        Self { id, check_digit }
    }

    /// Compute the ISWC check digit for the given 9-digit work identifier.
    ///
    /// Algorithm (ISO 15707 Annex A):
    /// 1. Expand the 9-digit identifier into individual decimal digits d₁…d₉
    ///    (d₁ = most significant).
    /// 2. Multiply each digit by its 1-based position weight (1×d₁ + 2×d₂ + …).
    /// 3. Sum all products.
    /// 4. Check digit = (10 - (sum mod 10)) mod 10.
    fn compute_check(id: u32) -> u8 {
        let digits = Self::expand_digits(id);
        let weighted_sum: u32 = digits
            .iter()
            .enumerate()
            .map(|(i, &d)| (i as u32 + 1) * d as u32)
            .sum();
        ((10 - (weighted_sum % 10)) % 10) as u8
    }

    /// Expand a 9-digit integer into an array of individual digits (most
    /// significant first).
    fn expand_digits(id: u32) -> [u8; 9] {
        let mut digits = [0u8; 9];
        let mut n = id;
        for i in (0..9).rev() {
            digits[i] = (n % 10) as u8;
            n /= 10;
        }
        digits
    }

    /// Verify that the stored check digit matches a fresh computation.
    pub fn verify(&self) -> bool {
        Self::compute_check(self.id) == self.check_digit
    }

    /// Parse an ISWC from the canonical `T-NNNNNNNNN-C` string.
    ///
    /// # Errors
    /// Returns `Err` if the format is invalid or the check digit fails.
    pub fn parse(s: &str) -> Result<Self, String> {
        // Accept both `T-NNNNNNNNN-C` and `T-NNN-NNN-NNN-C` (spaces normalised)
        let normalised = s.trim();
        // Must start with T-
        let body = normalised
            .strip_prefix("T-")
            .ok_or_else(|| format!("Invalid ISWC '{}': must start with 'T-'", s))?;

        // Body is `NNNNNNNNN-C`
        let dash_pos = body
            .rfind('-')
            .ok_or_else(|| format!("Invalid ISWC '{}': missing check-digit separator", s))?;
        let id_str = &body[..dash_pos];
        let check_str = &body[dash_pos + 1..];

        // Remove any embedded dashes from the work-identifier segment.
        let id_clean: String = id_str.chars().filter(|c| c.is_ascii_digit()).collect();
        if id_clean.len() != 9 {
            return Err(format!(
                "Invalid ISWC '{}': work identifier must have exactly 9 digits, got {}",
                s,
                id_clean.len()
            ));
        }
        let id: u32 = id_clean
            .parse()
            .map_err(|_| format!("Invalid ISWC '{}': non-numeric work identifier", s))?;

        if check_str.len() != 1 {
            return Err(format!(
                "Invalid ISWC '{}': check digit must be a single character",
                s
            ));
        }
        let check_digit: u8 = check_str
            .chars()
            .next()
            .ok_or_else(|| format!("Invalid ISWC '{}': empty check digit", s))?
            .to_digit(10)
            .ok_or_else(|| format!("Invalid ISWC '{}': check digit is not a decimal digit", s))?
            as u8;

        let iswc = Self { id, check_digit };
        if !iswc.verify() {
            return Err(format!(
                "Invalid ISWC '{}': check digit {} does not match computed {}",
                s,
                check_digit,
                Self::compute_check(id)
            ));
        }
        Ok(iswc)
    }

    /// Format this ISWC as `T-NNNNNNNNN-C`.
    pub fn to_iswc_string(&self) -> String {
        format!("T-{:09}-{}", self.id, self.check_digit)
    }
}

impl std::fmt::Display for Iswc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_iswc_string())
    }
}

// ── Isan ─────────────────────────────────────────────────────────────────────

/// International Standard Audiovisual Number (ISO 15706).
///
/// An ISAN consists of:
/// - an 8-byte (16 hex char) root identifying the work,
/// - a 4-byte (8 hex char) episode segment (all zeros for non-episodic works),
/// - a single check byte.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Isan {
    /// 64-bit root identifier.
    pub root: [u8; 8],
    /// 32-bit episode identifier (zeros for non-episodic).
    pub episode: [u8; 4],
    /// Check byte.
    pub check_digit: u8,
}

impl Isan {
    /// Parse an ISAN from a hex string.
    ///
    /// Accepted forms (case-insensitive, optional dashes):
    /// - `AAAAAAAABBBBBBBBCCCCCCCCDD` (25 hex chars + 1 check)
    /// - `AAAAAAAAAAAAAAAA-BBBBBBBB-CC` (dashes optional)
    ///
    /// The string must provide exactly 16 root hex chars + 8 episode hex chars
    /// + 2 check hex chars = 26 hex characters total (ignoring dashes).
    ///
    /// # Errors
    /// Returns `Err` if the string cannot be parsed as a valid ISAN.
    pub fn parse(s: &str) -> Result<Self, String> {
        // Strip dashes and whitespace, normalise to uppercase.
        let hex: String = s
            .chars()
            .filter(|c| !c.is_whitespace() && *c != '-')
            .map(|c| c.to_ascii_uppercase())
            .collect();

        if hex.len() != 26 {
            return Err(format!(
                "Invalid ISAN '{}': expected 26 hex characters (ignoring dashes), got {}",
                s,
                hex.len()
            ));
        }

        if !hex.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(format!("Invalid ISAN '{}': non-hex character(s)", s));
        }

        let mut root = [0u8; 8];
        for i in 0..8 {
            root[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16)
                .map_err(|_| format!("Invalid ISAN '{}': root parse error at byte {}", s, i))?;
        }

        let mut episode = [0u8; 4];
        for i in 0..4 {
            episode[i] = u8::from_str_radix(&hex[16 + i * 2..16 + i * 2 + 2], 16)
                .map_err(|_| format!("Invalid ISAN '{}': episode parse error at byte {}", s, i))?;
        }

        let check_digit = u8::from_str_radix(&hex[24..26], 16)
            .map_err(|_| format!("Invalid ISAN '{}': check byte parse error", s))?;

        Ok(Self {
            root,
            episode,
            check_digit,
        })
    }

    /// Format this ISAN as uppercase hex with group dashes:
    /// `RRRRRRRRRRRRRRRR-EEEEEEEE-CC`
    pub fn to_isan_string(&self) -> String {
        let root_hex: String = self.root.iter().map(|b| format!("{:02X}", b)).collect();
        let ep_hex: String = self.episode.iter().map(|b| format!("{:02X}", b)).collect();
        format!("{}-{}-{:02X}", root_hex, ep_hex, self.check_digit)
    }
}

impl std::fmt::Display for Isan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_isan_string())
    }
}

// ── RightsMetadata ────────────────────────────────────────────────────────────

/// Aggregated rights metadata for a single media asset.
#[derive(Debug, Clone, Default)]
pub struct RightsMetadata {
    /// Recording identifier.
    pub isrc: Option<Isrc>,
    /// Musical work identifier.
    pub iswc: Option<Iswc>,
    /// Audiovisual work identifier.
    pub isan: Option<Isan>,
    /// Entities holding rights to this asset.
    pub rights_holders: Vec<String>,
    /// Music publisher name.
    pub publisher: Option<String>,
    /// Record label.
    pub label: Option<String>,
    /// Year of original release.
    pub release_year: Option<u16>,
    /// Genre tags.
    pub genres: Vec<String>,
}

impl RightsMetadata {
    /// Construct a minimal `RightsMetadata` from an ISRC only.
    pub fn from_isrc(isrc: Isrc) -> Self {
        Self {
            isrc: Some(isrc),
            ..Default::default()
        }
    }

    /// Return `true` if this asset has at least one recognised identifier.
    pub fn has_identifier(&self) -> bool {
        self.isrc.is_some() || self.iswc.is_some() || self.isan.is_some()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Isrc ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_isrc_new_valid() {
        let isrc = Isrc::new("US", "ABC", 24, 12345).expect("valid ISRC");
        assert_eq!(isrc.country, "US");
        assert_eq!(isrc.registrant, "ABC");
        assert_eq!(isrc.year, 24);
        assert_eq!(isrc.designation, 12345);
    }

    #[test]
    fn test_isrc_new_invalid_country() {
        assert!(Isrc::new("USA", "ABC", 24, 0).is_err());
    }

    #[test]
    fn test_isrc_new_invalid_registrant_length() {
        assert!(Isrc::new("US", "AB", 24, 0).is_err());
    }

    #[test]
    fn test_isrc_new_invalid_designation_overflow() {
        assert!(Isrc::new("US", "ABC", 24, 100_000).is_err());
    }

    #[test]
    fn test_isrc_roundtrip_parse() {
        let original = Isrc::new("GB", "XYZ", 5, 99999).expect("valid");
        let s = original.to_isrc_string();
        let parsed = Isrc::parse(&s).expect("parse should succeed");
        assert_eq!(original, parsed);
    }

    #[test]
    fn test_isrc_to_string_format() {
        let isrc = Isrc::new("US", "ABC", 3, 1).expect("valid");
        assert_eq!(isrc.to_isrc_string(), "US-ABC-03-00001");
    }

    #[test]
    fn test_isrc_parse_invalid_format() {
        assert!(Isrc::parse("US-ABC-24").is_err()); // too few segments
    }

    // ── Iswc ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_iswc_new_check_digit_computed() {
        // Known example: T-000000001-? — verify check is deterministic.
        let iswc = Iswc::new(1);
        assert!(iswc.verify(), "check digit should verify");
    }

    #[test]
    fn test_iswc_verify_valid() {
        let iswc = Iswc::new(123_456_789);
        assert!(iswc.verify());
    }

    #[test]
    fn test_iswc_verify_tampered_fails() {
        let mut iswc = Iswc::new(123_456_789);
        iswc.check_digit = (iswc.check_digit + 1) % 10;
        assert!(!iswc.verify());
    }

    #[test]
    fn test_iswc_roundtrip_parse() {
        let original = Iswc::new(987_654_321);
        let s = original.to_iswc_string();
        let parsed = Iswc::parse(&s).expect("parse should succeed");
        assert_eq!(original.id, parsed.id);
        assert_eq!(original.check_digit, parsed.check_digit);
    }

    // ── Isan ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_isan_parse_roundtrip() {
        // Construct a known 26-hex-char ISAN string and round-trip it.
        let s = "00000000AB12CD340000000056";
        let isan = Isan::parse(s).expect("parse should succeed");
        let back = isan.to_isan_string().replace('-', "");
        // Compare hex ignoring dashes and case
        assert_eq!(back.to_uppercase(), s.to_uppercase());
    }

    #[test]
    fn test_isan_parse_too_short() {
        assert!(Isan::parse("0000AB").is_err());
    }

    // ── RightsMetadata ────────────────────────────────────────────────────────

    #[test]
    fn test_rights_metadata_from_isrc() {
        let isrc = Isrc::new("JP", "KAS", 22, 500).expect("valid");
        let meta = RightsMetadata::from_isrc(isrc);
        assert!(meta.isrc.is_some());
        assert!(meta.iswc.is_none());
        assert!(meta.has_identifier());
    }

    #[test]
    fn test_rights_metadata_empty_has_no_identifier() {
        let meta = RightsMetadata::default();
        assert!(!meta.has_identifier());
    }
}
