//! Configuration types for line-breaking: `LineBreakConfig`, `AudienceProfile`,
//! `CpsCache`, and reading-speed helpers.

use std::collections::HashMap;

use super::balance::compute_cps;

/// Configuration for line-breaking behaviour.
#[derive(Debug, Clone, PartialEq)]
pub struct LineBreakConfig {
    /// Maximum characters per line.
    pub max_chars_per_line: u8,
    /// Maximum reading speed in characters per second.
    pub max_cps: f32,
    /// Maximum number of lines in a caption block.
    pub max_lines: u8,
    /// Minimum gap between successive caption blocks in milliseconds.
    pub min_gap_ms: u32,
    /// Hard maximum characters per line (enforced even if `max_chars_per_line`
    /// would allow more).  `None` means no additional constraint.
    pub hard_max_chars: Option<u8>,
}

impl LineBreakConfig {
    /// Sensible broadcast defaults: 42 chars/line, 17 CPS, 2 lines, 80ms gap.
    pub fn default_broadcast() -> Self {
        Self {
            max_chars_per_line: 42,
            max_cps: 17.0,
            max_lines: 2,
            min_gap_ms: 80,
            hard_max_chars: None,
        }
    }

    /// Effective maximum characters per line considering the hard cap.
    pub fn effective_max_chars(&self) -> u8 {
        match self.hard_max_chars {
            Some(hard) => self.max_chars_per_line.min(hard),
            None => self.max_chars_per_line,
        }
    }
}

// ─── Target audience reading speed ────────────────────────────────────────────

/// The intended viewing audience, used to select appropriate CPS limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudienceProfile {
    /// Young children (ages 4–7): very slow readers.
    YoungChildren,
    /// Older children (ages 8–12): moderate readers.
    OlderChildren,
    /// General adult audience: standard broadcast speed.
    Adults,
    /// Specialised/technical audience: faster reading expected.
    TechnicalAdults,
}

impl AudienceProfile {
    /// Maximum recommended reading speed (CPS) for this audience.
    pub fn max_cps(self) -> f32 {
        match self {
            AudienceProfile::YoungChildren => 5.0,
            AudienceProfile::OlderChildren => 10.0,
            AudienceProfile::Adults => 17.0,
            AudienceProfile::TechnicalAdults => 22.0,
        }
    }

    /// Minimum recommended display duration (ms) for this audience.
    pub fn min_display_ms(self) -> u32 {
        match self {
            AudienceProfile::YoungChildren => 3000,
            AudienceProfile::OlderChildren => 1500,
            AudienceProfile::Adults => 1000,
            AudienceProfile::TechnicalAdults => 700,
        }
    }
}

/// Validate reading speed for a specific audience profile.
///
/// Returns `true` if the CPS is within acceptable range for the audience.
pub fn reading_speed_ok_for_audience(
    text: &str,
    duration_ms: u64,
    audience: AudienceProfile,
) -> bool {
    super::balance::reading_speed_ok(text, duration_ms, audience.max_cps())
}

// ─── CPS cache ────────────────────────────────────────────────────────────────

/// A cache for CPS (characters-per-second) computations.
///
/// This avoids recomputing CPS for the same `(text, duration_ms)` pairs when
/// captions are re-broken multiple times (e.g., during layout refinement).
#[derive(Debug, Default)]
pub struct CpsCache {
    cache: HashMap<(u64, u64), f32>, // key: (text_hash, duration_ms)
}

impl CpsCache {
    /// Create a new empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute or retrieve cached CPS for `(text, duration_ms)`.
    pub fn compute_cps(&mut self, text: &str, duration_ms: u64) -> f32 {
        let key = (hash_str(text), duration_ms);
        *self
            .cache
            .entry(key)
            .or_insert_with(|| compute_cps(text, duration_ms))
    }

    /// Return the number of entries currently in the cache.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Return `true` if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Clear all cached entries.
    pub fn clear(&mut self) {
        self.cache.clear();
    }
}

/// Simple FNV-1a 64-bit hash for a string.
fn hash_str(s: &str) -> u64 {
    const FNV_OFFSET: u64 = 14695981039346656037;
    const FNV_PRIME: u64 = 1099511628211;
    s.bytes().fold(FNV_OFFSET, |acc, b| {
        (acc ^ b as u64).wrapping_mul(FNV_PRIME)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_break_config_default_broadcast_values() {
        let cfg = LineBreakConfig::default_broadcast();
        assert_eq!(cfg.max_chars_per_line, 42);
        assert_eq!(cfg.max_lines, 2);
        assert_eq!(cfg.min_gap_ms, 80);
        assert_eq!(cfg.hard_max_chars, None);
    }

    #[test]
    fn line_break_config_hard_max_chars_constrains_effective() {
        let mut cfg = LineBreakConfig::default_broadcast();
        cfg.hard_max_chars = Some(30);
        assert_eq!(cfg.effective_max_chars(), 30); // hard cap wins
        cfg.hard_max_chars = Some(50);
        assert_eq!(cfg.effective_max_chars(), 42); // max_chars_per_line wins
    }

    #[test]
    fn audience_profile_children_have_lower_cps() {
        assert!(AudienceProfile::YoungChildren.max_cps() < AudienceProfile::Adults.max_cps());
        assert!(AudienceProfile::OlderChildren.max_cps() < AudienceProfile::Adults.max_cps());
    }

    #[test]
    fn audience_profile_children_have_longer_min_display() {
        assert!(
            AudienceProfile::YoungChildren.min_display_ms()
                > AudienceProfile::Adults.min_display_ms()
        );
    }

    #[test]
    fn reading_speed_ok_for_audience_children() {
        // 10 chars at 3 seconds = 3.3 cps < 5 cps (YoungChildren threshold)
        assert!(reading_speed_ok_for_audience(
            "Hello world",
            3000,
            AudienceProfile::YoungChildren
        ));
    }

    #[test]
    fn reading_speed_too_fast_for_children() {
        // 100 chars at 2 seconds = 50 cps > 5 cps
        let text = "A".repeat(100);
        assert!(!reading_speed_ok_for_audience(
            &text,
            2000,
            AudienceProfile::YoungChildren
        ));
    }

    #[test]
    fn cps_cache_returns_same_value_twice() {
        let mut cache = CpsCache::new();
        let v1 = cache.compute_cps("Hello world", 2000);
        let v2 = cache.compute_cps("Hello world", 2000);
        assert!((v1 - v2).abs() < 1e-6);
    }

    #[test]
    fn cps_cache_stores_entry() {
        let mut cache = CpsCache::new();
        assert_eq!(cache.len(), 0);
        cache.compute_cps("Hello", 1000);
        assert_eq!(cache.len(), 1);
        // Same key → no new entry.
        cache.compute_cps("Hello", 1000);
        assert_eq!(cache.len(), 1);
        // Different text → new entry.
        cache.compute_cps("World", 1000);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn cps_cache_clear_removes_all_entries() {
        let mut cache = CpsCache::new();
        cache.compute_cps("Hello", 1000);
        cache.clear();
        assert!(cache.is_empty());
    }
}
