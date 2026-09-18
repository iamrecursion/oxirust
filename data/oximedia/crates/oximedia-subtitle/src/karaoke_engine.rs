//! Full ASS karaoke timing engine with syllable-level highlighting.
//!
//! Provides a `KaraokeTrack` that holds `KaraokeSyllable` entries with precise
//! timing, and a progress-based renderer that computes per-syllable highlight state.

use crate::style::Color;

/// A single syllable in a karaoke track.
#[derive(Clone, Debug, PartialEq)]
pub struct KaraokeSyllable {
    /// The syllable text.
    pub text: String,
    /// Start time in milliseconds (absolute).
    pub start_ms: i64,
    /// Duration of this syllable in milliseconds.
    pub duration_ms: i64,
}

impl KaraokeSyllable {
    /// Create a new karaoke syllable.
    #[must_use]
    pub fn new(text: impl Into<String>, start_ms: i64, duration_ms: i64) -> Self {
        Self {
            text: text.into(),
            start_ms,
            duration_ms,
        }
    }

    /// End time in milliseconds.
    #[must_use]
    pub fn end_ms(&self) -> i64 {
        self.start_ms + self.duration_ms
    }

    /// Whether this syllable is active at the given timestamp.
    #[must_use]
    pub fn is_active(&self, timestamp_ms: i64) -> bool {
        timestamp_ms >= self.start_ms && timestamp_ms < self.end_ms()
    }

    /// Progress within this syllable (0.0 to 1.0).
    /// Returns 0.0 before start, 1.0 after end.
    #[must_use]
    pub fn progress(&self, timestamp_ms: i64) -> f32 {
        if self.duration_ms <= 0 {
            return if timestamp_ms >= self.start_ms {
                1.0
            } else {
                0.0
            };
        }
        let elapsed = timestamp_ms - self.start_ms;
        if elapsed <= 0 {
            0.0
        } else if elapsed >= self.duration_ms {
            1.0
        } else {
            elapsed as f32 / self.duration_ms as f32
        }
    }
}

/// Highlight state for a syllable at a given point in time.
#[derive(Clone, Debug, PartialEq)]
pub enum SyllableState {
    /// Syllable has not started yet.
    Pending,
    /// Syllable is currently being highlighted (progress 0.0..1.0).
    Active {
        /// How far through the syllable we are (0.0 to 1.0).
        progress: f32,
    },
    /// Syllable has already been sung.
    Completed,
}

/// A complete karaoke track with syllable-level timing.
#[derive(Clone, Debug)]
pub struct KaraokeTrack {
    /// The syllables in this track.
    pub syllables: Vec<KaraokeSyllable>,
    /// Color for un-highlighted (pending) text.
    pub pending_color: Color,
    /// Color for highlighted (active/completed) text.
    pub highlight_color: Color,
    /// Color for the active syllable wipe fill.
    pub wipe_color: Color,
}

impl KaraokeTrack {
    /// Create a new empty karaoke track with default colors.
    #[must_use]
    pub fn new() -> Self {
        Self {
            syllables: Vec::new(),
            pending_color: Color::white(),
            highlight_color: Color::rgb(255, 255, 0),
            wipe_color: Color::rgb(255, 200, 0),
        }
    }

    /// Create a track from a list of syllables.
    #[must_use]
    pub fn from_syllables(syllables: Vec<KaraokeSyllable>) -> Self {
        Self {
            syllables,
            ..Self::new()
        }
    }

    /// Set the pending (un-sung) color.
    #[must_use]
    pub fn with_pending_color(mut self, color: Color) -> Self {
        self.pending_color = color;
        self
    }

    /// Set the highlight (sung) color.
    #[must_use]
    pub fn with_highlight_color(mut self, color: Color) -> Self {
        self.highlight_color = color;
        self
    }

    /// Set the wipe (active fill) color.
    #[must_use]
    pub fn with_wipe_color(mut self, color: Color) -> Self {
        self.wipe_color = color;
        self
    }

    /// Add a syllable to the track.
    pub fn add_syllable(&mut self, syllable: KaraokeSyllable) {
        self.syllables.push(syllable);
    }

    /// Total duration of the karaoke track.
    #[must_use]
    pub fn total_duration_ms(&self) -> i64 {
        self.syllables
            .iter()
            .map(|s| s.end_ms())
            .max()
            .unwrap_or(0)
            .saturating_sub(self.start_ms())
    }

    /// Start time of the first syllable.
    #[must_use]
    pub fn start_ms(&self) -> i64 {
        self.syllables.iter().map(|s| s.start_ms).min().unwrap_or(0)
    }

    /// End time of the last syllable.
    #[must_use]
    pub fn end_ms(&self) -> i64 {
        self.syllables.iter().map(|s| s.end_ms()).max().unwrap_or(0)
    }

    /// Get the full text of the track.
    #[must_use]
    pub fn full_text(&self) -> String {
        self.syllables.iter().map(|s| s.text.as_str()).collect()
    }

    /// Get the state of each syllable at a given timestamp.
    #[must_use]
    pub fn syllable_states(&self, timestamp_ms: i64) -> Vec<SyllableState> {
        self.syllables
            .iter()
            .map(|s| {
                if timestamp_ms < s.start_ms {
                    SyllableState::Pending
                } else if timestamp_ms >= s.end_ms() {
                    SyllableState::Completed
                } else {
                    SyllableState::Active {
                        progress: s.progress(timestamp_ms),
                    }
                }
            })
            .collect()
    }

    /// Compute the color for each syllable at a given timestamp.
    ///
    /// Returns a vector of `(text, color)` tuples for rendering.
    #[must_use]
    pub fn render_colors(&self, timestamp_ms: i64) -> Vec<(&str, Color)> {
        let states = self.syllable_states(timestamp_ms);
        self.syllables
            .iter()
            .zip(states.iter())
            .map(|(syl, state)| {
                let color = match state {
                    SyllableState::Pending => self.pending_color,
                    SyllableState::Completed => self.highlight_color,
                    SyllableState::Active { progress } => {
                        interpolate_color(self.pending_color, self.wipe_color, *progress)
                    }
                };
                (syl.text.as_str(), color)
            })
            .collect()
    }

    /// Get the overall progress of the track (0.0 to 1.0).
    #[must_use]
    pub fn overall_progress(&self, timestamp_ms: i64) -> f32 {
        if self.syllables.is_empty() {
            return 0.0;
        }
        let start = self.start_ms();
        let end = self.end_ms();
        if end <= start {
            return if timestamp_ms >= start { 1.0 } else { 0.0 };
        }
        let elapsed = timestamp_ms - start;
        if elapsed <= 0 {
            0.0
        } else if elapsed >= end - start {
            1.0
        } else {
            elapsed as f32 / (end - start) as f32
        }
    }
}

impl Default for KaraokeTrack {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse ASS karaoke tags from text and produce syllables.
///
/// Supports `\k`, `\K`, `\kf`, `\ko` duration tags (centiseconds).
/// The `base_start_ms` is the absolute start time of the dialogue line.
///
/// # Example
///
/// ```
/// use oximedia_subtitle::karaoke_engine::parse_ass_karaoke;
/// let syllables = parse_ass_karaoke(r"{\k50}Hel{\k30}lo {\k80}World", 0);
/// assert_eq!(syllables.len(), 3);
/// assert_eq!(syllables[0].text, "Hel");
/// assert_eq!(syllables[0].duration_ms, 500);
/// ```
#[must_use]
pub fn parse_ass_karaoke(text: &str, base_start_ms: i64) -> Vec<KaraokeSyllable> {
    let mut syllables = Vec::new();
    let mut cursor_ms = base_start_ms;
    let mut remaining = text;

    while !remaining.is_empty() {
        // Look for next override block
        if let Some(brace_start) = remaining.find('{') {
            // Text before the brace belongs to the previous syllable if any
            let before = &remaining[..brace_start];
            if !before.is_empty() {
                // Append to last syllable or create a zero-duration one
                if let Some(last) = syllables.last_mut() {
                    let last: &mut KaraokeSyllable = last;
                    last.text.push_str(before);
                } else {
                    syllables.push(KaraokeSyllable::new(before, cursor_ms, 0));
                }
            }

            // Find the closing brace
            let after_brace = &remaining[brace_start + 1..];
            if let Some(brace_end) = after_brace.find('}') {
                let tag_content = &after_brace[..brace_end];
                remaining = &after_brace[brace_end + 1..];

                // Parse karaoke tags within the block
                if let Some(duration_cs) = parse_karaoke_tag(tag_content) {
                    let duration_ms = i64::from(duration_cs) * 10;
                    syllables.push(KaraokeSyllable::new("", cursor_ms, duration_ms));
                    cursor_ms += duration_ms;
                }
            } else {
                // Unclosed brace, treat rest as text
                if let Some(last) = syllables.last_mut() {
                    let last: &mut KaraokeSyllable = last;
                    last.text.push_str(remaining);
                } else {
                    syllables.push(KaraokeSyllable::new(remaining, cursor_ms, 0));
                }
                break;
            }
        } else {
            // No more braces, rest is text for last syllable
            if let Some(last) = syllables.last_mut() {
                let last: &mut KaraokeSyllable = last;
                last.text.push_str(remaining);
            } else if !remaining.is_empty() {
                syllables.push(KaraokeSyllable::new(remaining, cursor_ms, 0));
            }
            break;
        }
    }

    // Remove empty-text syllables (from consecutive tags)
    syllables.retain(|s| !s.text.is_empty());

    syllables
}

/// Parse a karaoke duration tag from override block content.
///
/// Supports `\k<N>`, `\K<N>`, `\kf<N>`, `\ko<N>` where N is centiseconds.
fn parse_karaoke_tag(content: &str) -> Option<u32> {
    // Tags can have multiple overrides; find the karaoke one
    for part in content.split('\\') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        // Match k, K, kf, ko followed by digits
        let numeric_start = if part.starts_with("kf") || part.starts_with("ko") {
            2
        } else if part.starts_with('k') || part.starts_with('K') {
            1
        } else {
            continue;
        };

        let digits = &part[numeric_start..];
        if let Ok(val) = digits.trim().parse::<u32>() {
            return Some(val);
        }
    }
    None
}

/// Linearly interpolate between two colors.
#[must_use]
fn interpolate_color(a: Color, b: Color, t: f32) -> Color {
    let t = t.clamp(0.0, 1.0);
    let inv = 1.0 - t;
    Color::new(
        (f32::from(a.r) * inv + f32::from(b.r) * t) as u8,
        (f32::from(a.g) * inv + f32::from(b.g) * t) as u8,
        (f32::from(a.b) * inv + f32::from(b.b) * t) as u8,
        (f32::from(a.a) * inv + f32::from(b.a) * t) as u8,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syllable_basic() {
        let s = KaraokeSyllable::new("Hello", 1000, 500);
        assert_eq!(s.end_ms(), 1500);
        assert!(s.is_active(1200));
        assert!(!s.is_active(900));
        assert!(!s.is_active(1500));
    }

    #[test]
    fn test_syllable_progress() {
        let s = KaraokeSyllable::new("X", 0, 1000);
        assert!((s.progress(0) - 0.0).abs() < f32::EPSILON);
        assert!((s.progress(500) - 0.5).abs() < f32::EPSILON);
        assert!((s.progress(1000) - 1.0).abs() < f32::EPSILON);
        assert!((s.progress(-100) - 0.0).abs() < f32::EPSILON);
        assert!((s.progress(2000) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_syllable_zero_duration() {
        let s = KaraokeSyllable::new("X", 100, 0);
        assert!((s.progress(50) - 0.0).abs() < f32::EPSILON);
        assert!((s.progress(100) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_parse_ass_karaoke_basic() {
        let syllables = parse_ass_karaoke(r"{\k50}Hel{\k30}lo {\k80}World", 0);
        assert_eq!(syllables.len(), 3);
        assert_eq!(syllables[0].text, "Hel");
        assert_eq!(syllables[0].start_ms, 0);
        assert_eq!(syllables[0].duration_ms, 500);
        assert_eq!(syllables[1].text, "lo ");
        assert_eq!(syllables[1].start_ms, 500);
        assert_eq!(syllables[1].duration_ms, 300);
        assert_eq!(syllables[2].text, "World");
        assert_eq!(syllables[2].start_ms, 800);
        assert_eq!(syllables[2].duration_ms, 800);
    }

    #[test]
    fn test_parse_ass_karaoke_kf_ko() {
        let syllables = parse_ass_karaoke(r"{\kf100}A{\ko50}B", 1000);
        assert_eq!(syllables.len(), 2);
        assert_eq!(syllables[0].start_ms, 1000);
        assert_eq!(syllables[0].duration_ms, 1000);
        assert_eq!(syllables[1].start_ms, 2000);
        assert_eq!(syllables[1].duration_ms, 500);
    }

    #[test]
    fn test_parse_ass_karaoke_no_tags() {
        let syllables = parse_ass_karaoke("Hello World", 0);
        assert_eq!(syllables.len(), 1);
        assert_eq!(syllables[0].text, "Hello World");
    }

    #[test]
    fn test_karaoke_track_states() {
        let track = KaraokeTrack::from_syllables(vec![
            KaraokeSyllable::new("A", 0, 500),
            KaraokeSyllable::new("B", 500, 500),
            KaraokeSyllable::new("C", 1000, 500),
        ]);

        let states = track.syllable_states(600);
        assert_eq!(states[0], SyllableState::Completed);
        assert!(matches!(states[1], SyllableState::Active { .. }));
        assert_eq!(states[2], SyllableState::Pending);
    }

    #[test]
    fn test_karaoke_track_full_text() {
        let track = KaraokeTrack::from_syllables(vec![
            KaraokeSyllable::new("Hel", 0, 500),
            KaraokeSyllable::new("lo", 500, 300),
        ]);
        assert_eq!(track.full_text(), "Hello");
    }

    #[test]
    fn test_karaoke_track_timing() {
        let track = KaraokeTrack::from_syllables(vec![
            KaraokeSyllable::new("A", 100, 200),
            KaraokeSyllable::new("B", 300, 400),
        ]);
        assert_eq!(track.start_ms(), 100);
        assert_eq!(track.end_ms(), 700);
        assert_eq!(track.total_duration_ms(), 600);
    }

    #[test]
    fn test_karaoke_track_overall_progress() {
        let track = KaraokeTrack::from_syllables(vec![
            KaraokeSyllable::new("A", 0, 1000),
            KaraokeSyllable::new("B", 1000, 1000),
        ]);
        assert!((track.overall_progress(0) - 0.0).abs() < f32::EPSILON);
        assert!((track.overall_progress(1000) - 0.5).abs() < f32::EPSILON);
        assert!((track.overall_progress(2000) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_karaoke_render_colors() {
        let track = KaraokeTrack::from_syllables(vec![
            KaraokeSyllable::new("A", 0, 500),
            KaraokeSyllable::new("B", 500, 500),
        ])
        .with_pending_color(Color::white())
        .with_highlight_color(Color::rgb(255, 255, 0));

        let colors = track.render_colors(600);
        assert_eq!(colors.len(), 2);
        assert_eq!(colors[0].0, "A");
        assert_eq!(colors[0].1, Color::rgb(255, 255, 0)); // completed
        assert_eq!(colors[1].0, "B");
        // Active: should be interpolated between pending and wipe
    }

    #[test]
    fn test_karaoke_track_empty() {
        let track = KaraokeTrack::new();
        assert_eq!(track.total_duration_ms(), 0);
        assert_eq!(track.full_text(), "");
        assert!((track.overall_progress(100) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_interpolate_color_extremes() {
        let a = Color::rgb(0, 0, 0);
        let b = Color::rgb(255, 255, 255);
        let mid = interpolate_color(a, b, 0.5);
        assert_eq!(mid.r, 127);
        assert_eq!(mid.g, 127);
        assert_eq!(mid.b, 127);
    }

    #[test]
    fn test_karaoke_add_syllable() {
        let mut track = KaraokeTrack::new();
        track.add_syllable(KaraokeSyllable::new("X", 0, 100));
        assert_eq!(track.syllables.len(), 1);
    }

    #[test]
    fn test_parse_ass_karaoke_with_base_offset() {
        let syllables = parse_ass_karaoke(r"{\k100}Test", 5000);
        assert_eq!(syllables[0].start_ms, 5000);
        assert_eq!(syllables[0].duration_ms, 1000);
    }

    #[test]
    fn test_karaoke_default_colors() {
        let track = KaraokeTrack::new();
        assert_eq!(track.pending_color, Color::white());
        assert_eq!(track.highlight_color, Color::rgb(255, 255, 0));
    }
}
