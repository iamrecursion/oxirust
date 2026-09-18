//! Real-time DJ features: beat-matching and compatible key detection.
//!
//! # Beat matching
//!
//! [`BeatMatcher`](crate::dj_features::BeatMatcher) computes the pitch-shift / time-stretch factor needed to
//! align two tracks at the same BPM, supporting integer and half/double
//! tempo corrections.
//!
//! # Camelot Wheel
//!
//! The [Camelot Wheel](https://www.harmonic-mixing.com/howto.aspx) is a
//! disc-jockey tool for harmonic mixing: it assigns each musical key a
//! position code (1A–12A for minor keys, 1B–12B for major keys) such that
//! adjacent positions are harmonically compatible.
//!
//! [`CamelotWheel`](crate::dj_features::CamelotWheel) converts between standard key names and Camelot codes,
//! and exposes a compatibility matrix for harmonic mixing decisions.

use std::fmt;

// ---------------------------------------------------------------------------
// Camelot Wheel
// ---------------------------------------------------------------------------

/// Camelot code: a number 1–12 plus a letter A (minor) or B (major).
///
/// Adjacent numbers on the wheel are a perfect fifth apart; A/B pairs share
/// the same root (relative major/minor).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CamelotCode {
    /// Position on the wheel (1–12).
    pub number: u8,
    /// `true` = major (B side), `false` = minor (A side).
    pub is_major: bool,
}

impl CamelotCode {
    /// Create a Camelot code from a number and mode flag.
    ///
    /// # Panics
    ///
    /// Does NOT panic — invalid numbers are clamped to [1, 12].
    #[must_use]
    pub fn new(number: u8, is_major: bool) -> Self {
        Self {
            number: number.clamp(1, 12),
            is_major,
        }
    }

    /// Letter component: 'A' for minor, 'B' for major.
    #[must_use]
    pub fn letter(&self) -> char {
        if self.is_major {
            'B'
        } else {
            'A'
        }
    }

    /// Check whether `other` is harmonically compatible with `self`.
    ///
    /// Compatible combinations:
    /// - Same code (identical keys).
    /// - Same number, opposite letter (relative major/minor).
    /// - Adjacent number (±1, wrapping), same letter (perfect fifth relationship).
    #[must_use]
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        if self == other {
            return true;
        }
        // Relative major/minor: same number, different letter
        if self.number == other.number && self.is_major != other.is_major {
            return true;
        }
        // Adjacent ±1 on the wheel (same mode): perfect fifth
        let diff = wheel_distance(self.number, other.number);
        diff == 1 && self.is_major == other.is_major
    }

    /// "Energy boost" mixing: one semitone up from `self` (adjacent energy key).
    ///
    /// Returns the code that is 1 step clockwise (number + 1, wrapping).
    #[must_use]
    pub fn energy_boost(&self) -> Self {
        Self::new(wheel_inc(self.number), self.is_major)
    }

    /// Dominant key: one step counter-clockwise (number - 1, wrapping).
    #[must_use]
    pub fn dominant(&self) -> Self {
        Self::new(wheel_dec(self.number), self.is_major)
    }
}

impl fmt::Display for CamelotCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.number, self.letter())
    }
}

/// Compute the circular distance between two wheel positions [1..12].
fn wheel_distance(a: u8, b: u8) -> u8 {
    let diff = (a as i8 - b as i8).unsigned_abs();
    diff.min(12 - diff)
}

fn wheel_inc(n: u8) -> u8 {
    if n >= 12 {
        1
    } else {
        n + 1
    }
}

fn wheel_dec(n: u8) -> u8 {
    if n <= 1 {
        12
    } else {
        n - 1
    }
}

// ---------------------------------------------------------------------------
// Camelot Wheel mapping table
// ---------------------------------------------------------------------------

/// Full mapping between (root_note 0–11, is_major) and Camelot code.
///
/// Root notes follow standard chroma ordering: 0 = C, 1 = C#, … 11 = B.
///
/// The mapping is the standard Camelot Wheel used by Mixed In Key and similar
/// software.
static CAMELOT_TABLE: &[(u8, bool, u8, bool)] = &[
    // (root, is_major, camelot_number, camelot_is_major)
    (0, false, 5, false),   // C minor  → 5A
    (0, true, 8, true),     // C major  → 8B
    (1, false, 12, false),  // C# minor → 12A
    (1, true, 3, true),     // C# major → 3B
    (2, false, 7, false),   // D minor  → 7A
    (2, true, 10, true),    // D major  → 10B
    (3, false, 2, false),   // Eb minor → 2A
    (3, true, 5, true),     // Eb major → 5B
    (4, false, 9, false),   // E minor  → 9A
    (4, true, 12, true),    // E major  → 12B
    (5, false, 4, false),   // F minor  → 4A
    (5, true, 7, true),     // F major  → 7B
    (6, false, 11, false),  // F# minor → 11A
    (6, true, 2, true),     // F# major → 2B
    (7, false, 6, false),   // G minor  → 6A
    (7, true, 9, true),     // G major  → 9B
    (8, false, 1, false),   // Ab minor → 1A
    (8, true, 4, true),     // Ab major → 4B
    (9, false, 8, false),   // A minor  → 8A
    (9, true, 11, true),    // A major  → 11B
    (10, false, 3, false),  // Bb minor → 3A
    (10, true, 6, true),    // Bb major → 6B
    (11, false, 10, false), // B minor  → 10A
    (11, true, 1, true),    // B major  → 1B
];

/// Helper for Camelot Wheel lookups and compatibility queries.
pub struct CamelotWheel;

impl CamelotWheel {
    /// Convert a `(root_note, is_major)` pair to a Camelot code.
    ///
    /// `root_note`: 0 = C, 1 = C#, … 11 = B.
    /// Returns `None` if the root note is out of range.
    #[must_use]
    pub fn from_key(root_note: u8, is_major: bool) -> Option<CamelotCode> {
        CAMELOT_TABLE
            .iter()
            .find(|(r, m, _, _)| *r == root_note && *m == is_major)
            .map(|(_, _, cn, cm)| CamelotCode::new(*cn, *cm))
    }

    /// Convert a Camelot code back to a `(root_note, is_major)` pair.
    ///
    /// Returns `None` if the code is not in the table.
    #[must_use]
    pub fn to_key(code: &CamelotCode) -> Option<(u8, bool)> {
        CAMELOT_TABLE
            .iter()
            .find(|(_, _, cn, cm)| *cn == code.number && *cm == code.is_major)
            .map(|(r, m, _, _)| (*r, *m))
    }

    /// Return all Camelot codes that are harmonically compatible with `code`.
    ///
    /// Includes `code` itself and up to 3 neighbours:
    /// - Relative major/minor (same number, other letter).
    /// - Adjacent clockwise (number+1, same letter).
    /// - Adjacent counter-clockwise (number-1, same letter).
    #[must_use]
    pub fn compatible_keys(code: &CamelotCode) -> Vec<CamelotCode> {
        let mut compatible = vec![
            *code,
            CamelotCode::new(code.number, !code.is_major),
            CamelotCode::new(wheel_inc(code.number), code.is_major),
            CamelotCode::new(wheel_dec(code.number), code.is_major),
        ];
        compatible.dedup();
        compatible
    }

    /// Determine whether two keys are harmonically compatible for DJ mixing.
    #[must_use]
    pub fn are_compatible(a_root: u8, a_major: bool, b_root: u8, b_major: bool) -> bool {
        match (
            Self::from_key(a_root, a_major),
            Self::from_key(b_root, b_major),
        ) {
            (Some(ca), Some(cb)) => ca.is_compatible_with(&cb),
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------------
// BeatMatcher
// ---------------------------------------------------------------------------

/// Result of a beat-matching analysis.
#[derive(Debug, Clone)]
pub struct BeatMatchResult {
    /// BPM of track A.
    pub bpm_a: f32,
    /// BPM of track B.
    pub bpm_b: f32,
    /// Recommended target BPM for mixing (average of the two after correction).
    pub target_bpm: f32,
    /// Stretch/pitch factor to apply to track B to match track A.
    ///
    /// Values > 1.0 mean speed up track B; < 1.0 mean slow it down.
    pub stretch_factor: f32,
    /// Whether a half-tempo correction was applied to track B.
    pub half_tempo_correction: bool,
    /// Whether a double-tempo correction was applied to track B.
    pub double_tempo_correction: bool,
    /// Harmonic compatibility between the two keys.
    pub keys_compatible: bool,
    /// Camelot code for track A (if key info provided).
    pub camelot_a: Option<CamelotCode>,
    /// Camelot code for track B (if key info provided).
    pub camelot_b: Option<CamelotCode>,
}

/// Real-time beat-matcher for DJ applications.
///
/// Given BPMs (and optionally key information) for two tracks, computes the
/// stretch factor needed to align them at a common tempo.
pub struct BeatMatcher {
    /// Tolerance for half/double tempo detection (fraction, e.g. 0.05 = 5%).
    pub tempo_tolerance: f32,
}

impl Default for BeatMatcher {
    fn default() -> Self {
        Self {
            tempo_tolerance: 0.05,
        }
    }
}

impl BeatMatcher {
    /// Create a new beat matcher with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute beat-match parameters to mix track B into track A.
    ///
    /// # Arguments
    ///
    /// * `bpm_a`       — BPM of the playing deck (track A, "anchor").
    /// * `bpm_b`       — BPM of the incoming track (track B, to be adjusted).
    /// * `key_a`       — Optional `(root_note, is_major)` for track A.
    /// * `key_b`       — Optional `(root_note, is_major)` for track B.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn match_tracks(
        &self,
        bpm_a: f32,
        bpm_b: f32,
        key_a: Option<(u8, bool)>,
        key_b: Option<(u8, bool)>,
    ) -> BeatMatchResult {
        let mut effective_bpm_b = bpm_b;
        let mut half_correction = false;
        let mut double_correction = false;

        // Detect half-tempo: if track B is roughly half of A, double it
        let ratio = bpm_a / effective_bpm_b;
        if (ratio - 2.0).abs() < self.tempo_tolerance * 2.0 {
            effective_bpm_b *= 2.0;
            double_correction = true;
        } else if (ratio - 0.5).abs() < self.tempo_tolerance * 0.5 {
            // A is half of B: halve B
            effective_bpm_b *= 0.5;
            half_correction = true;
        }

        // Stretch factor: how much to speed up / slow down track B
        let stretch_factor = bpm_a / effective_bpm_b;

        // Target BPM: average (DJ convention: meet in the middle)
        let target_bpm = (bpm_a + effective_bpm_b) / 2.0;

        // Key compatibility
        let camelot_a = key_a.and_then(|(r, m)| CamelotWheel::from_key(r, m));
        let camelot_b = key_b.and_then(|(r, m)| CamelotWheel::from_key(r, m));

        let keys_compatible = match (&camelot_a, &camelot_b) {
            (Some(ca), Some(cb)) => ca.is_compatible_with(cb),
            _ => false,
        };

        BeatMatchResult {
            bpm_a,
            bpm_b,
            target_bpm,
            stretch_factor,
            half_tempo_correction: half_correction,
            double_tempo_correction: double_correction,
            keys_compatible,
            camelot_a,
            camelot_b,
        }
    }

    /// Suggest the minimum pitch/tempo shift (in semitones) to make two keys
    /// harmonically compatible.
    ///
    /// Returns `None` if the keys are already compatible or if key info is missing.
    /// Returns `Some(semitones)` where `semitones ∈ [-6, 6]` is the smallest
    /// shift that moves `key_b` onto a wheel position compatible with `key_a`.
    #[must_use]
    pub fn suggest_key_shift(root_a: u8, major_a: bool, root_b: u8, major_b: bool) -> Option<i8> {
        let ca = CamelotWheel::from_key(root_a, major_a)?;
        let cb = CamelotWheel::from_key(root_b, major_b)?;

        if ca.is_compatible_with(&cb) {
            return None;
        }

        // Try all 12 semitone shifts on track B and find the smallest one that
        // yields a compatible key.
        let mut best_shift: Option<i8> = None;
        for shift in -6_i8..=6 {
            let new_root = ((root_b as i8 + shift).rem_euclid(12)) as u8;
            if let Some(shifted_code) = CamelotWheel::from_key(new_root, major_b) {
                if ca.is_compatible_with(&shifted_code) {
                    match best_shift {
                        None => best_shift = Some(shift),
                        Some(prev) if shift.unsigned_abs() < prev.unsigned_abs() => {
                            best_shift = Some(shift);
                        }
                        _ => {}
                    }
                }
            }
        }

        best_shift
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_camelot_code_display() {
        let code = CamelotCode::new(8, true);
        assert_eq!(code.to_string(), "8B");
        let code2 = CamelotCode::new(5, false);
        assert_eq!(code2.to_string(), "5A");
    }

    #[test]
    fn test_camelot_from_key_c_major() {
        // C major → 8B
        let code = CamelotWheel::from_key(0, true).expect("C major must map");
        assert_eq!(code.number, 8);
        assert!(code.is_major);
    }

    #[test]
    fn test_camelot_from_key_a_minor() {
        // A minor (root=9, minor) → 8A
        let code = CamelotWheel::from_key(9, false).expect("A minor must map");
        assert_eq!(code.number, 8);
        assert!(!code.is_major);
    }

    #[test]
    fn test_camelot_all_24_keys_map() {
        for root in 0_u8..12 {
            let major = CamelotWheel::from_key(root, true);
            let minor = CamelotWheel::from_key(root, false);
            assert!(major.is_some(), "Root {root} major missing");
            assert!(minor.is_some(), "Root {root} minor missing");
        }
    }

    #[test]
    fn test_camelot_to_key_roundtrip() {
        for (root, is_major, cn, cm) in CAMELOT_TABLE {
            let code = CamelotCode::new(*cn, *cm);
            let (r, m) = CamelotWheel::to_key(&code).expect("roundtrip must work");
            assert_eq!(r, *root, "root mismatch for {root}/{is_major}");
            assert_eq!(m, *is_major, "mode mismatch for {root}/{is_major}");
        }
    }

    #[test]
    fn test_compatible_same_key() {
        let c_major = CamelotWheel::from_key(0, true).expect("C major must map");
        assert!(
            c_major.is_compatible_with(&c_major),
            "Key compatible with itself"
        );
    }

    #[test]
    fn test_compatible_relative_minor() {
        // C major (8B) and A minor (8A) share number 8 → compatible
        let c_major = CamelotWheel::from_key(0, true).expect("C major must map");
        let a_minor = CamelotWheel::from_key(9, false).expect("A minor must map");
        assert!(c_major.is_compatible_with(&a_minor));
    }

    #[test]
    fn test_compatible_adjacent_fifth() {
        // G major (9B) is adjacent to C major (8B) → compatible
        let c_major = CamelotWheel::from_key(0, true).expect("C major must map");
        let g_major = CamelotWheel::from_key(7, true).expect("G major must map");
        assert!(c_major.is_compatible_with(&g_major));
    }

    #[test]
    fn test_incompatible_distant_keys() {
        // C major and F# major are 6 semitones apart — incompatible
        let c_major = CamelotWheel::from_key(0, true).expect("C major must map");
        let fs_major = CamelotWheel::from_key(6, true).expect("F# major must map");
        assert!(!c_major.is_compatible_with(&fs_major));
    }

    #[test]
    fn test_compatible_keys_returns_four() {
        let code = CamelotWheel::from_key(0, true).expect("C major must map"); // C major = 8B
        let compat = CamelotWheel::compatible_keys(&code);
        assert!(compat.len() >= 3 && compat.len() <= 4);
        // Must include self
        assert!(compat.contains(&code));
    }

    #[test]
    fn test_beat_matcher_same_bpm() {
        let matcher = BeatMatcher::new();
        let result = matcher.match_tracks(120.0, 120.0, None, None);
        assert!((result.stretch_factor - 1.0).abs() < 1e-4);
        assert!(!result.half_tempo_correction);
        assert!(!result.double_tempo_correction);
    }

    #[test]
    fn test_beat_matcher_double_tempo() {
        // Track B at 60 BPM played against 120 BPM → double correction
        let matcher = BeatMatcher::new();
        let result = matcher.match_tracks(120.0, 60.0, None, None);
        assert!(result.double_tempo_correction);
        assert!((result.stretch_factor - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_beat_matcher_half_tempo() {
        // Track B at 240 BPM against 120 BPM → halved
        let matcher = BeatMatcher::new();
        let result = matcher.match_tracks(120.0, 240.0, None, None);
        assert!(result.half_tempo_correction);
    }

    #[test]
    fn test_beat_matcher_key_compatibility() {
        let matcher = BeatMatcher::new();
        // C major (root=0, major) vs A minor (root=9, minor) → compatible
        let result = matcher.match_tracks(120.0, 120.0, Some((0, true)), Some((9, false)));
        assert!(result.keys_compatible);
        assert!(result.camelot_a.is_some());
        assert!(result.camelot_b.is_some());
    }

    #[test]
    fn test_beat_matcher_key_incompatible() {
        let matcher = BeatMatcher::new();
        // C major vs F# major → incompatible
        let result = matcher.match_tracks(120.0, 120.0, Some((0, true)), Some((6, true)));
        assert!(!result.keys_compatible);
    }

    #[test]
    fn test_suggest_key_shift_already_compatible() {
        // Same key — no shift needed
        let shift = BeatMatcher::suggest_key_shift(0, true, 0, true);
        assert!(shift.is_none());
    }

    #[test]
    fn test_suggest_key_shift_finds_small_shift() {
        // C major vs F# major (tritone) — should find a small shift
        let shift = BeatMatcher::suggest_key_shift(0, true, 6, true);
        assert!(shift.is_some(), "Should find a valid shift");
        let s = shift.expect("shift must be Some");
        assert!(
            s.unsigned_abs() <= 6,
            "Shift should be at most 6 semitones, got {s}"
        );
    }

    #[test]
    fn test_wheel_distance_wrap() {
        // Distance between 1 and 12 on a 12-point wheel = 1
        assert_eq!(wheel_distance(1, 12), 1);
        assert_eq!(wheel_distance(12, 1), 1);
    }

    #[test]
    fn test_camelot_energy_boost_and_dominant() {
        let c_major = CamelotWheel::from_key(0, true).expect("C major must map"); // 8B
        let energy = c_major.energy_boost(); // 9B
        let dominant = c_major.dominant(); // 7B
        assert_eq!(energy.number, 9);
        assert_eq!(dominant.number, 7);
        assert!(energy.is_major);
        assert!(dominant.is_major);
    }
}
