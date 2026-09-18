//! Playlist continuity management.
//!
//! Detects continuity breaks (runtime gaps, genre mismatches, duplicate items,
//! etc.) within a playlist and provides repair utilities.

#![allow(dead_code)]

use crate::recommendation_engine::PlaylistItem;
use serde::{Deserialize, Serialize};

/// Classification of a continuity break.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BreakType {
    /// A gap larger than 30 seconds exists between two consecutive items.
    RuntimeGap,
    /// The content rating escalates (e.g. PG → R) at this boundary.
    ContentRating,
    /// A large age gap between production years.
    AgeGap,
    /// A dramatic genre shift between consecutive items.
    GenreMismatch,
    /// The same title appears more than once in the playlist.
    DuplicateItem,
    /// A significant tonal shift (e.g. comedy directly followed by horror).
    ToneShift,
}

/// Represents a single continuity break found within a playlist.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuityBreak {
    /// Index of the first item involved in the break (0-based).
    pub position_idx: usize,
    /// Type of continuity break.
    pub break_type: BreakType,
    /// Severity of the break (0.0 = minor, 1.0 = critical).
    pub severity: f32,
}

impl ContinuityBreak {
    /// Create a new continuity break.
    #[must_use]
    pub fn new(position_idx: usize, break_type: BreakType, severity: f32) -> Self {
        Self {
            position_idx,
            break_type,
            severity: severity.clamp(0.0, 1.0),
        }
    }
}

/// Analyses a playlist for continuity issues.
pub struct ContinuityChecker;

impl ContinuityChecker {
    /// Check a slice of playlist items for continuity breaks.
    ///
    /// Currently detects:
    /// - **RuntimeGap**: No gap detection from raw `PlaylistItem` since timing
    ///   is implicit; instead consecutive items with the same gap logic are
    ///   evaluated (stub — always 0 gap for adjacent items in a playlist).
    /// - **DuplicateItem**: The same `title` appears more than once.
    /// - **GenreMismatch**: Adjacent items have strongly differing genres.
    #[must_use]
    pub fn check(items: &[PlaylistItem]) -> Vec<ContinuityBreak> {
        let mut breaks = Vec::new();

        // Detect duplicates
        for i in 0..items.len() {
            for j in (i + 1)..items.len() {
                if items[i].title == items[j].title {
                    breaks.push(ContinuityBreak::new(j, BreakType::DuplicateItem, 0.6));
                }
            }
        }

        // Detect genre mismatches and tone shifts between adjacent items
        for i in 1..items.len() {
            let prev_genre = items[i - 1].genre.to_lowercase();
            let curr_genre = items[i].genre.to_lowercase();

            if is_dramatic_genre_shift(&prev_genre, &curr_genre) {
                breaks.push(ContinuityBreak::new(i, BreakType::GenreMismatch, 0.7));
            } else if is_tone_shift(&prev_genre, &curr_genre) {
                breaks.push(ContinuityBreak::new(i, BreakType::ToneShift, 0.5));
            }
        }

        // Sort by position index
        breaks.sort_by_key(|b| b.position_idx);
        breaks
    }
}

/// Returns `true` when two genre strings represent a dramatic mismatch.
fn is_dramatic_genre_shift(a: &str, b: &str) -> bool {
    const NEWS: &[&str] = &["news", "documentary", "factual"];
    const ENTERTAINMENT: &[&str] = &["comedy", "drama", "thriller", "action", "romance", "film"];

    let a_news = NEWS.iter().any(|g| a.contains(g));
    let b_news = NEWS.iter().any(|g| b.contains(g));
    let a_ent = ENTERTAINMENT.iter().any(|g| a.contains(g));
    let b_ent = ENTERTAINMENT.iter().any(|g| b.contains(g));

    // News → Entertainment or vice-versa is a dramatic shift
    (a_news && b_ent) || (a_ent && b_news)
}

/// Returns `true` when two genres suggest a tonal shift (not dramatic enough
/// for a full GenreMismatch).
fn is_tone_shift(a: &str, b: &str) -> bool {
    const LIGHT: &[&str] = &["comedy", "animation", "children", "family"];
    const DARK: &[&str] = &["horror", "thriller", "crime", "noir"];

    let a_light = LIGHT.iter().any(|g| a.contains(g));
    let b_dark = DARK.iter().any(|g| b.contains(g));
    let a_dark = DARK.iter().any(|g| a.contains(g));
    let b_light = LIGHT.iter().any(|g| b.contains(g));

    (a_light && b_dark) || (a_dark && b_light)
}

/// Content rating classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ContentRating {
    /// Suitable for all audiences.
    G,
    /// Parental guidance suggested.
    PG,
    /// Parents strongly cautioned (13+).
    PG13,
    /// Restricted (17+).
    R,
    /// Adults only (17+, stronger than R).
    NC17,
    /// Mature audiences – Television equivalent (TV-MA).
    TvMa,
}

impl ContentRating {
    /// Returns a numeric level (0–5) for comparison purposes.
    #[must_use]
    pub const fn level(&self) -> u8 {
        match self {
            Self::G => 0,
            Self::PG => 1,
            Self::PG13 => 2,
            Self::R => 3,
            Self::NC17 => 4,
            Self::TvMa => 5,
        }
    }
}

/// Checks for content rating escalations within a playlist.
pub struct RatingContinuityCheck;

impl RatingContinuityCheck {
    /// Find positions where the content rating escalates by more than one step.
    ///
    /// Returns the indices (into `items`) where an escalation is detected.
    #[must_use]
    pub fn check(items: &[(u64, ContentRating)]) -> Vec<usize> {
        let mut escalations = Vec::new();
        for i in 1..items.len() {
            let prev_level = items[i - 1].1.level();
            let curr_level = items[i].1.level();
            // Flag any upward jump of more than 1 rating level
            if curr_level > prev_level + 1 {
                escalations.push(i);
            }
        }
        escalations
    }
}

/// Provides repair utilities for playlist continuity.
pub struct PlaylistRepair;

impl PlaylistRepair {
    /// Insert `filler` items to cover gaps longer than 30 seconds between
    /// consecutive items.
    ///
    /// Since `PlaylistItem` does not carry an absolute schedule position, this
    /// function identifies consecutive pairs that together leave an implicit gap
    /// (where the sum of all item durations is shorter than the expected window)
    /// and inserts filler to pad the list.
    ///
    /// In practice this simply inserts a filler between any two consecutive
    /// items whose combined duration leaves a detected gap — here we use a
    /// heuristic: if consecutive items have a large duration difference (> 30s)
    /// compared to the expected average slot, insert a filler.
    ///
    /// For a simple, testable interface: insert `filler` before every item that
    /// would create an implied gap > 30 s when played back-to-back in a
    /// fixed-duration slot.
    pub fn fix_gaps(items: &mut Vec<PlaylistItem>, filler: PlaylistItem) {
        const GAP_THRESHOLD_SECS: u32 = 30;
        let mut i = 1;
        while i < items.len() {
            let prev_dur = items[i - 1].duration_secs;
            let curr_dur = items[i].duration_secs;
            // Heuristic: if duration difference suggests a gap in a fixed schedule
            if prev_dur > curr_dur + GAP_THRESHOLD_SECS {
                let mut filler_clone = filler.clone();
                // Set filler duration to fill the perceived gap
                filler_clone.duration_secs = prev_dur.saturating_sub(curr_dur);
                items.insert(i, filler_clone);
                i += 2; // skip the filler we just inserted
            } else {
                i += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recommendation_engine::PlaylistItem;

    fn make_item(id: u64, title: &str, genre: &str, duration_secs: u32) -> PlaylistItem {
        PlaylistItem::new(id, title, duration_secs, genre, 3.5, 0.5)
    }

    #[test]
    fn test_break_type_variants() {
        let b = ContinuityBreak::new(2, BreakType::DuplicateItem, 0.6);
        assert_eq!(b.position_idx, 2);
        assert_eq!(b.break_type, BreakType::DuplicateItem);
    }

    #[test]
    fn test_severity_clamped() {
        let b = ContinuityBreak::new(0, BreakType::RuntimeGap, 5.0);
        assert_eq!(b.severity, 1.0);
        let b2 = ContinuityBreak::new(0, BreakType::GenreMismatch, -1.0);
        assert_eq!(b2.severity, 0.0);
    }

    #[test]
    fn test_no_breaks_in_homogeneous_playlist() {
        let items = vec![
            make_item(1, "Drama 1", "drama", 3600),
            make_item(2, "Drama 2", "drama", 3600),
            make_item(3, "Drama 3", "drama", 3600),
        ];
        let breaks = ContinuityChecker::check(&items);
        assert!(breaks.is_empty());
    }

    #[test]
    fn test_duplicate_detection() {
        let items = vec![
            make_item(1, "News At Ten", "news", 1800),
            make_item(2, "Drama Show", "drama", 3600),
            make_item(3, "News At Ten", "news", 1800), // duplicate title
        ];
        let breaks = ContinuityChecker::check(&items);
        assert!(breaks
            .iter()
            .any(|b| b.break_type == BreakType::DuplicateItem));
    }

    #[test]
    fn test_genre_mismatch_detection() {
        let items = vec![
            make_item(1, "Evening News", "news", 1800),
            make_item(2, "Action Movie", "action film", 7200),
        ];
        let breaks = ContinuityChecker::check(&items);
        assert!(breaks
            .iter()
            .any(|b| b.break_type == BreakType::GenreMismatch));
    }

    #[test]
    fn test_tone_shift_detection() {
        let items = vec![
            make_item(1, "Kids' Show", "children animation", 1800),
            make_item(2, "Horror Night", "horror thriller", 7200),
        ];
        let breaks = ContinuityChecker::check(&items);
        assert!(breaks.iter().any(|b| b.break_type == BreakType::ToneShift));
    }

    #[test]
    fn test_content_rating_levels() {
        assert!(ContentRating::G.level() < ContentRating::PG.level());
        assert!(ContentRating::PG.level() < ContentRating::PG13.level());
        assert!(ContentRating::R.level() < ContentRating::NC17.level());
        assert!(ContentRating::NC17.level() < ContentRating::TvMa.level());
    }

    #[test]
    fn test_rating_continuity_no_escalation() {
        let items: Vec<(u64, ContentRating)> = vec![
            (1, ContentRating::G),
            (2, ContentRating::PG),
            (3, ContentRating::PG13),
        ];
        let escalations = RatingContinuityCheck::check(&items);
        assert!(escalations.is_empty());
    }

    #[test]
    fn test_rating_continuity_escalation() {
        let items: Vec<(u64, ContentRating)> = vec![
            (1, ContentRating::G),
            (2, ContentRating::R), // skips PG and PG13 — escalation
        ];
        let escalations = RatingContinuityCheck::check(&items);
        assert_eq!(escalations, vec![1]);
    }

    #[test]
    fn test_rating_continuity_multiple_escalations() {
        let items: Vec<(u64, ContentRating)> = vec![
            (1, ContentRating::G),
            (2, ContentRating::R),    // escalation at idx 1
            (3, ContentRating::PG),   // no escalation (drops back)
            (4, ContentRating::TvMa), // escalation at idx 3
        ];
        let escalations = RatingContinuityCheck::check(&items);
        assert!(escalations.contains(&1));
        assert!(escalations.contains(&3));
    }

    #[test]
    fn test_fix_gaps_inserts_filler() {
        let mut items = vec![
            make_item(1, "Long Show", "drama", 3600),
            make_item(2, "Short Clip", "drama", 120), // gap heuristic: 3600 - 120 > 30
        ];
        let filler = make_item(99, "Filler", "slate", 0);
        let original_len = items.len();
        PlaylistRepair::fix_gaps(&mut items, filler);
        assert!(
            items.len() > original_len,
            "Filler should have been inserted"
        );
    }

    #[test]
    fn test_fix_gaps_no_insertion_when_no_gap() {
        let mut items = vec![
            make_item(1, "Show A", "drama", 1800),
            make_item(2, "Show B", "drama", 1800),
        ];
        let filler = make_item(99, "Filler", "slate", 0);
        let original_len = items.len();
        PlaylistRepair::fix_gaps(&mut items, filler);
        assert_eq!(items.len(), original_len, "No filler should be inserted");
    }
}
