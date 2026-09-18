#![allow(dead_code)]
//! Screen reader accessibility support for media player interfaces.
//!
//! This module provides ARIA-like role definitions, live region announcements,
//! semantic descriptions for media UI elements, and Braille display output so
//! that screen reader software and refreshable Braille displays can present
//! the interface effectively to blind users.

use std::collections::{HashMap, VecDeque};
use std::fmt;

/// ARIA-like role for a UI element.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AriaRole {
    /// A clickable button.
    Button,
    /// A range slider (volume, seek).
    Slider,
    /// A progress bar.
    ProgressBar,
    /// A timer/clock display.
    Timer,
    /// A status indicator.
    Status,
    /// A region that receives live updates.
    LiveRegion,
    /// A group container.
    Group,
    /// A toolbar container.
    Toolbar,
    /// A dialog overlay.
    Dialog,
    /// A menu.
    Menu,
    /// A menu item.
    MenuItem,
    /// A checkbox toggle.
    Checkbox,
    /// A generic region.
    Region,
}

impl fmt::Display for AriaRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Button => write!(f, "button"),
            Self::Slider => write!(f, "slider"),
            Self::ProgressBar => write!(f, "progressbar"),
            Self::Timer => write!(f, "timer"),
            Self::Status => write!(f, "status"),
            Self::LiveRegion => write!(f, "log"),
            Self::Group => write!(f, "group"),
            Self::Toolbar => write!(f, "toolbar"),
            Self::Dialog => write!(f, "dialog"),
            Self::Menu => write!(f, "menu"),
            Self::MenuItem => write!(f, "menuitem"),
            Self::Checkbox => write!(f, "checkbox"),
            Self::Region => write!(f, "region"),
        }
    }
}

/// Politeness level for live region announcements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LivePoliteness {
    /// The update is not announced unless the user navigates to it.
    Off,
    /// The update is announced at the next graceful opportunity.
    Polite,
    /// The update is announced immediately, interrupting current speech.
    Assertive,
}

impl fmt::Display for LivePoliteness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Off => write!(f, "off"),
            Self::Polite => write!(f, "polite"),
            Self::Assertive => write!(f, "assertive"),
        }
    }
}

/// Describes a single UI element for screen reader consumption.
#[derive(Debug, Clone)]
pub struct AccessibleElement {
    /// Unique identifier for the element.
    pub id: String,
    /// ARIA-like role.
    pub role: AriaRole,
    /// Accessible label (what the screen reader reads).
    pub label: String,
    /// Optional description for additional context.
    pub description: Option<String>,
    /// Whether the element is currently disabled.
    pub disabled: bool,
    /// Whether the element is currently hidden from the accessibility tree.
    pub hidden: bool,
    /// Optional current value (for sliders, progress bars).
    pub value: Option<String>,
    /// Optional minimum value.
    pub value_min: Option<String>,
    /// Optional maximum value.
    pub value_max: Option<String>,
    /// Whether the element is in a pressed/checked state.
    pub pressed: Option<bool>,
}

impl AccessibleElement {
    /// Create a new accessible element.
    #[must_use]
    pub fn new(id: impl Into<String>, role: AriaRole, label: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            role,
            label: label.into(),
            description: None,
            disabled: false,
            hidden: false,
            value: None,
            value_min: None,
            value_max: None,
            pressed: None,
        }
    }

    /// Set the description.
    #[must_use]
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set disabled state.
    #[must_use]
    pub fn with_disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }

    /// Set hidden state.
    #[must_use]
    pub fn with_hidden(mut self, hidden: bool) -> Self {
        self.hidden = hidden;
        self
    }

    /// Set a current value (for sliders/progress).
    #[must_use]
    pub fn with_value(mut self, value: impl Into<String>) -> Self {
        self.value = Some(value.into());
        self
    }

    /// Set a value range (min/max).
    #[must_use]
    pub fn with_range(mut self, min: impl Into<String>, max: impl Into<String>) -> Self {
        self.value_min = Some(min.into());
        self.value_max = Some(max.into());
        self
    }

    /// Set pressed/checked state.
    #[must_use]
    pub fn with_pressed(mut self, pressed: bool) -> Self {
        self.pressed = Some(pressed);
        self
    }

    /// Generate the full screen reader announcement for this element.
    #[must_use]
    pub fn announce(&self) -> String {
        let mut parts = Vec::new();
        parts.push(self.label.clone());
        parts.push(self.role.to_string());

        if let Some(v) = &self.value {
            parts.push(format!("value {v}"));
        }
        if let Some(pressed) = self.pressed {
            parts.push(if pressed {
                "pressed".to_string()
            } else {
                "not pressed".to_string()
            });
        }
        if self.disabled {
            parts.push("disabled".to_string());
        }

        parts.join(", ")
    }
}

/// A live region announcement to be spoken by the screen reader.
#[derive(Debug, Clone)]
pub struct Announcement {
    /// The text to be spoken.
    pub text: String,
    /// The politeness level.
    pub politeness: LivePoliteness,
    /// Timestamp in milliseconds when the announcement was created.
    pub timestamp_ms: u64,
}

impl Announcement {
    /// Create a new polite announcement.
    #[must_use]
    pub fn polite(text: impl Into<String>, timestamp_ms: u64) -> Self {
        Self {
            text: text.into(),
            politeness: LivePoliteness::Polite,
            timestamp_ms,
        }
    }

    /// Create a new assertive announcement.
    #[must_use]
    pub fn assertive(text: impl Into<String>, timestamp_ms: u64) -> Self {
        Self {
            text: text.into(),
            politeness: LivePoliteness::Assertive,
            timestamp_ms,
        }
    }
}

/// Manages a queue of screen reader announcements.
#[derive(Debug, Clone)]
pub struct AnnouncementQueue {
    /// Queue of pending announcements.
    queue: VecDeque<Announcement>,
    /// Maximum queue size before old entries are dropped.
    max_size: usize,
}

impl AnnouncementQueue {
    /// Create a new announcement queue.
    #[must_use]
    pub fn new(max_size: usize) -> Self {
        Self {
            queue: VecDeque::new(),
            max_size: max_size.max(1),
        }
    }

    /// Push a new announcement. Drops oldest if over capacity.
    pub fn push(&mut self, announcement: Announcement) {
        if self.queue.len() >= self.max_size {
            self.queue.pop_front();
        }
        self.queue.push_back(announcement);
    }

    /// Pop the next announcement to be spoken.
    pub fn pop(&mut self) -> Option<Announcement> {
        // Assertive announcements have priority
        if let Some(idx) = self
            .queue
            .iter()
            .position(|a| a.politeness == LivePoliteness::Assertive)
        {
            return self.queue.remove(idx);
        }
        self.queue.pop_front()
    }

    /// Get the number of pending announcements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Check whether the queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Clear all pending announcements.
    pub fn clear(&mut self) {
        self.queue.clear();
    }
}

impl Default for AnnouncementQueue {
    fn default() -> Self {
        Self::new(50)
    }
}

/// Generates standard media player screen reader announcements.
#[derive(Debug)]
pub struct MediaAnnouncer;

impl MediaAnnouncer {
    /// Announce playback state change.
    #[must_use]
    pub fn playback_state(playing: bool, timestamp_ms: u64) -> Announcement {
        let text = if playing { "Playing" } else { "Paused" };
        Announcement::assertive(text, timestamp_ms)
    }

    /// Announce volume change.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn volume_change(level: u32, max_level: u32, timestamp_ms: u64) -> Announcement {
        let pct = if max_level == 0 {
            0
        } else {
            #[allow(clippy::cast_possible_truncation)]
            #[allow(clippy::cast_sign_loss)]
            {
                ((f64::from(level) / f64::from(max_level)) * 100.0).round() as u32
            }
        };
        Announcement::polite(format!("Volume {pct} percent"), timestamp_ms)
    }

    /// Announce current time position.
    #[must_use]
    pub fn time_position(
        current_seconds: u64,
        total_seconds: u64,
        timestamp_ms: u64,
    ) -> Announcement {
        let cur_min = current_seconds / 60;
        let cur_sec = current_seconds % 60;
        let tot_min = total_seconds / 60;
        let tot_sec = total_seconds % 60;
        Announcement::polite(
            format!("{cur_min}:{cur_sec:02} of {tot_min}:{tot_sec:02}"),
            timestamp_ms,
        )
    }

    /// Announce mute state.
    #[must_use]
    pub fn mute_state(muted: bool, timestamp_ms: u64) -> Announcement {
        let text = if muted { "Muted" } else { "Unmuted" };
        Announcement::assertive(text, timestamp_ms)
    }

    /// Announce caption toggle.
    #[must_use]
    pub fn caption_state(enabled: bool, timestamp_ms: u64) -> Announcement {
        let text = if enabled {
            "Captions on"
        } else {
            "Captions off"
        };
        Announcement::polite(text, timestamp_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aria_role_display() {
        assert_eq!(AriaRole::Button.to_string(), "button");
        assert_eq!(AriaRole::Slider.to_string(), "slider");
        assert_eq!(AriaRole::ProgressBar.to_string(), "progressbar");
        assert_eq!(AriaRole::Timer.to_string(), "timer");
    }

    #[test]
    fn test_live_politeness_display() {
        assert_eq!(LivePoliteness::Off.to_string(), "off");
        assert_eq!(LivePoliteness::Polite.to_string(), "polite");
        assert_eq!(LivePoliteness::Assertive.to_string(), "assertive");
    }

    #[test]
    fn test_accessible_element_basic() {
        let elem = AccessibleElement::new("play_btn", AriaRole::Button, "Play");
        assert_eq!(elem.id, "play_btn");
        assert_eq!(elem.role, AriaRole::Button);
        assert_eq!(elem.label, "Play");
        assert!(!elem.disabled);
        assert!(!elem.hidden);
    }

    #[test]
    fn test_accessible_element_announce() {
        let elem = AccessibleElement::new("play_btn", AriaRole::Button, "Play");
        let text = elem.announce();
        assert!(text.contains("Play"));
        assert!(text.contains("button"));
    }

    #[test]
    fn test_accessible_element_slider() {
        let elem = AccessibleElement::new("vol", AriaRole::Slider, "Volume")
            .with_value("75")
            .with_range("0", "100");
        let text = elem.announce();
        assert!(text.contains("Volume"));
        assert!(text.contains("slider"));
        assert!(text.contains("value 75"));
    }

    #[test]
    fn test_accessible_element_disabled() {
        let elem = AccessibleElement::new("btn", AriaRole::Button, "Save").with_disabled(true);
        let text = elem.announce();
        assert!(text.contains("disabled"));
    }

    #[test]
    fn test_accessible_element_pressed() {
        let elem = AccessibleElement::new("mute", AriaRole::Button, "Mute").with_pressed(true);
        let text = elem.announce();
        assert!(text.contains("pressed"));
    }

    #[test]
    fn test_announcement_polite() {
        let a = Announcement::polite("Now playing", 1000);
        assert_eq!(a.politeness, LivePoliteness::Polite);
        assert_eq!(a.text, "Now playing");
    }

    #[test]
    fn test_announcement_assertive() {
        let a = Announcement::assertive("Error occurred", 2000);
        assert_eq!(a.politeness, LivePoliteness::Assertive);
    }

    #[test]
    fn test_queue_push_pop() {
        let mut q = AnnouncementQueue::new(10);
        q.push(Announcement::polite("Hello", 100));
        q.push(Announcement::polite("World", 200));
        assert_eq!(q.len(), 2);
        let a = q.pop().expect("a should be valid");
        assert_eq!(a.text, "Hello");
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn test_queue_assertive_priority() {
        let mut q = AnnouncementQueue::new(10);
        q.push(Announcement::polite("Low priority", 100));
        q.push(Announcement::assertive("High priority", 200));
        q.push(Announcement::polite("Also low", 300));
        let a = q.pop().expect("a should be valid");
        assert_eq!(a.text, "High priority");
    }

    #[test]
    fn test_queue_max_size() {
        let mut q = AnnouncementQueue::new(2);
        q.push(Announcement::polite("A", 100));
        q.push(Announcement::polite("B", 200));
        q.push(Announcement::polite("C", 300));
        // "A" should have been dropped
        assert_eq!(q.len(), 2);
        let first = q.pop().expect("first should be valid");
        assert_eq!(first.text, "B");
    }

    #[test]
    fn test_queue_clear() {
        let mut q = AnnouncementQueue::new(10);
        q.push(Announcement::polite("A", 100));
        q.clear();
        assert!(q.is_empty());
    }

    #[test]
    fn test_media_announcer_playback() {
        let a = MediaAnnouncer::playback_state(true, 0);
        assert_eq!(a.text, "Playing");
        assert_eq!(a.politeness, LivePoliteness::Assertive);

        let a = MediaAnnouncer::playback_state(false, 0);
        assert_eq!(a.text, "Paused");
    }

    #[test]
    fn test_media_announcer_volume() {
        let a = MediaAnnouncer::volume_change(50, 100, 0);
        assert_eq!(a.text, "Volume 50 percent");
        assert_eq!(a.politeness, LivePoliteness::Polite);
    }

    #[test]
    fn test_media_announcer_time() {
        let a = MediaAnnouncer::time_position(65, 120, 0);
        assert_eq!(a.text, "1:05 of 2:00");
    }

    #[test]
    fn test_media_announcer_mute() {
        let a = MediaAnnouncer::mute_state(true, 0);
        assert_eq!(a.text, "Muted");
        let a = MediaAnnouncer::mute_state(false, 0);
        assert_eq!(a.text, "Unmuted");
    }

    #[test]
    fn test_media_announcer_captions() {
        let a = MediaAnnouncer::caption_state(true, 0);
        assert_eq!(a.text, "Captions on");
        let a = MediaAnnouncer::caption_state(false, 0);
        assert_eq!(a.text, "Captions off");
    }
}

// ─── Braille Display Output ─────────────────────────────────────────────────

/// A single Braille cell encoded as a bitmask of raised dots.
///
/// Dots are numbered 1–8 in the standard Braille arrangement:
/// ```text
/// 1 4
/// 2 5
/// 3 6
/// 7 8   (dots 7 and 8 are used in 8-dot Braille only)
/// ```
/// The bitmask stores dot N at bit (N-1): bit 0 = dot 1, bit 1 = dot 2, …
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BrailleCell(pub u8);

impl BrailleCell {
    /// The blank (space) cell — no dots raised.
    pub const BLANK: Self = Self(0b0000_0000);

    /// Create a cell from individual dot flags (dots 1–6 only; 6-dot Braille).
    #[must_use]
    pub fn from_dots(d1: bool, d2: bool, d3: bool, d4: bool, d5: bool, d6: bool) -> Self {
        let mut mask = 0u8;
        if d1 {
            mask |= 1 << 0;
        }
        if d2 {
            mask |= 1 << 1;
        }
        if d3 {
            mask |= 1 << 2;
        }
        if d4 {
            mask |= 1 << 3;
        }
        if d5 {
            mask |= 1 << 4;
        }
        if d6 {
            mask |= 1 << 5;
        }
        Self(mask)
    }

    /// Whether dot N (1-based) is raised.
    #[must_use]
    pub fn dot(&self, n: u8) -> bool {
        if n == 0 || n > 8 {
            return false;
        }
        (self.0 >> (n - 1)) & 1 == 1
    }

    /// Convert to the Unicode Braille Pattern character (U+2800..U+28FF).
    ///
    /// The Unicode Braille block uses the same bitmask layout as this struct,
    /// so the mapping is a direct offset from U+2800.
    #[must_use]
    pub fn to_unicode(&self) -> char {
        // SAFETY: 0x2800..=0x28FF are valid Unicode scalar values.
        char::from_u32(0x2800 | u32::from(self.0)).unwrap_or('⠀')
    }

    /// Raised-dot count (number of dots raised in this cell).
    #[must_use]
    pub fn dot_count(&self) -> u32 {
        u32::from(self.0.count_ones())
    }
}

impl std::fmt::Display for BrailleCell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_unicode())
    }
}

/// Braille grade (translation fidelity level).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrailleGrade {
    /// Grade 1: character-by-character transliteration (uncontracted).
    Grade1,
    /// Grade 2: contracted Braille with common word abbreviations.
    Grade2,
}

/// Configuration for a refreshable Braille display.
#[derive(Debug, Clone)]
pub struct BrailleDisplayConfig {
    /// Number of cells visible in the display viewport.
    pub cell_width: usize,
    /// Braille translation grade.
    pub grade: BrailleGrade,
    /// Whether to wrap text at cell_width boundaries.
    pub wrap: bool,
}

impl BrailleDisplayConfig {
    /// Standard 40-cell Grade 2 display.
    #[must_use]
    pub fn standard_40() -> Self {
        Self {
            cell_width: 40,
            grade: BrailleGrade::Grade2,
            wrap: true,
        }
    }

    /// Compact 20-cell Grade 1 display.
    #[must_use]
    pub fn compact_20() -> Self {
        Self {
            cell_width: 20,
            grade: BrailleGrade::Grade1,
            wrap: false,
        }
    }
}

impl Default for BrailleDisplayConfig {
    fn default() -> Self {
        Self::standard_40()
    }
}

/// Translates text into sequences of [`BrailleCell`]s.
///
/// Supports Grade 1 (uncontracted) and Grade 2 (contracted) English Braille
/// using the English Braille American Edition (EBAE) standard.
pub struct BrailleEncoder {
    grade: BrailleGrade,
    /// Grade-2 contraction table: text pattern -> replacement cells.
    contractions: Vec<(String, Vec<BrailleCell>)>,
    /// Grade-1 character table: ASCII char -> cell.
    char_table: HashMap<char, BrailleCell>,
}

impl BrailleEncoder {
    /// Create a new encoder for the given grade.
    #[must_use]
    pub fn new(grade: BrailleGrade) -> Self {
        let char_table = Self::build_char_table();
        let contractions = if grade == BrailleGrade::Grade2 {
            Self::build_contractions(&char_table)
        } else {
            Vec::new()
        };
        Self {
            grade,
            contractions,
            char_table,
        }
    }

    /// Encode a text string into a vector of Braille cells.
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<BrailleCell> {
        if self.grade == BrailleGrade::Grade2 {
            self.encode_grade2(text)
        } else {
            self.encode_grade1(text)
        }
    }

    /// Grade-1 encoding: map each character individually.
    fn encode_grade1(&self, text: &str) -> Vec<BrailleCell> {
        text.chars()
            .map(|c| {
                self.char_table
                    .get(&c.to_ascii_lowercase())
                    .copied()
                    .unwrap_or(BrailleCell::BLANK)
            })
            .collect()
    }

    /// Grade-2 encoding: apply contractions where possible, then fall back to Grade 1.
    fn encode_grade2(&self, text: &str) -> Vec<BrailleCell> {
        let lower = text.to_lowercase();
        let mut result = Vec::new();
        let mut pos = 0usize;
        let bytes = lower.as_bytes();

        while pos < bytes.len() {
            // Try to match the longest contraction first
            let mut matched = false;
            for (pattern, cells) in &self.contractions {
                let pb = pattern.as_bytes();
                if bytes.len() >= pos + pb.len() && &bytes[pos..pos + pb.len()] == pb {
                    // Ensure we're at a word boundary for multi-char contractions
                    let at_start = pos == 0 || bytes[pos - 1] == b' ';
                    let at_end = pos + pb.len() == bytes.len()
                        || bytes[pos + pb.len()] == b' '
                        || pb.len() == 1;
                    if pb.len() == 1 || (at_start && at_end) {
                        result.extend_from_slice(cells);
                        pos += pb.len();
                        matched = true;
                        break;
                    }
                }
            }
            if !matched {
                let ch = lower[pos..].chars().next().unwrap_or(' ');
                result.push(
                    self.char_table
                        .get(&ch)
                        .copied()
                        .unwrap_or(BrailleCell::BLANK),
                );
                pos += ch.len_utf8();
            }
        }
        result
    }

    /// Convert a cell slice back to a Unicode Braille string.
    #[must_use]
    pub fn cells_to_unicode(cells: &[BrailleCell]) -> String {
        cells.iter().map(BrailleCell::to_unicode).collect()
    }

    // ─── Tables ────────────────────────────────────────────────────────────

    /// Build the EBAE Grade-1 character table (a–z, digits, space, punctuation).
    fn build_char_table() -> HashMap<char, BrailleCell> {
        // Dot masks for letters a-z follow the standard EBAE pattern.
        // Reference: https://www.brailleauthority.org/ueb/
        #[rustfmt::skip]
        let letter_masks: &[(char, u8)] = &[
            ('a', 0b000001), ('b', 0b000011), ('c', 0b001001), ('d', 0b011001),
            ('e', 0b010001), ('f', 0b001011), ('g', 0b011011), ('h', 0b010011),
            ('i', 0b001010), ('j', 0b011010), ('k', 0b000101), ('l', 0b000111),
            ('m', 0b001101), ('n', 0b011101), ('o', 0b010101), ('p', 0b001111),
            ('q', 0b011111), ('r', 0b010111), ('s', 0b001110), ('t', 0b011110),
            ('u', 0b100101), ('v', 0b100111), ('w', 0b111010), ('x', 0b101101),
            ('y', 0b111101), ('z', 0b110101),
        ];
        // Digits use the number indicator (dots 3456 = 0b111100) prefix; here
        // we store the digit cells directly without the indicator for simplicity.
        #[rustfmt::skip]
        let digit_masks: &[(char, u8)] = &[
            ('1', 0b000001), ('2', 0b000011), ('3', 0b001001), ('4', 0b011001),
            ('5', 0b010001), ('6', 0b001011), ('7', 0b011011), ('8', 0b010011),
            ('9', 0b001010), ('0', 0b011010),
        ];
        #[rustfmt::skip]
        let punct_masks: &[(char, u8)] = &[
            (' ', 0b000000),
            (',', 0b000010), (';', 0b000110), (':', 0b010010), ('.', 0b110010),
            ('!', 0b010110), ('?', 0b100110), ('\'', 0b000100), ('-', 0b100100),
            ('"', 0b100010),
        ];

        let mut table = HashMap::new();
        for (ch, mask) in letter_masks.iter().chain(digit_masks).chain(punct_masks) {
            table.insert(*ch, BrailleCell(*mask));
        }
        table
    }

    /// Build the Grade-2 contraction table ordered longest-first.
    fn build_contractions(
        char_table: &HashMap<char, BrailleCell>,
    ) -> Vec<(String, Vec<BrailleCell>)> {
        // Common EBAE Grade 2 whole-word contractions
        // Sorted longest-first so greedy matching picks the right one.
        let patterns: &[(&str, &[u8])] = &[
            // Whole-word contractions (dot masks)
            ("the", &[0b011110, 0b010011, 0b010001]), // ⠞⠓⠑  (t-h-e)
            ("and", &[0b000001, 0b011101, 0b011001]), // ⠁⠝⠙  (a-n-d)
            ("for", &[0b001011, 0b010101, 0b010111]), // ⠋⠕⠗  (f-o-r)
            ("of", &[0b010101, 0b001011]),            // ⠕⠋  (o-f)
            ("with", &[0b110111, 0b001010, 0b011110, 0b010011]), // ⠺⠊⠞⠓ (w-i-t-h)
            // Single-letter contractions for very common words
            ("a", &[0b000001]), // letter a
            ("i", &[0b001010]), // letter i
        ];

        let mut result: Vec<(String, Vec<BrailleCell>)> = patterns
            .iter()
            .map(|(pat, masks)| {
                let cells: Vec<BrailleCell> = masks.iter().map(|&m| BrailleCell(m)).collect();
                (pat.to_string(), cells)
            })
            .collect();

        // Add all single-character mappings from the char_table as fallback
        // (already covered in the Grade-1 path, but including ensures Grade-2
        // can handle any character).
        for (ch, cell) in char_table {
            if !result.iter().any(|(p, _)| p == &ch.to_string()) {
                result.push((ch.to_string(), vec![*cell]));
            }
        }

        // Sort: longest patterns first for greedy matching
        result.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        result
    }
}

impl Default for BrailleEncoder {
    fn default() -> Self {
        Self::new(BrailleGrade::Grade2)
    }
}

/// A refreshable Braille display with a panning viewport.
///
/// The full encoded output is stored internally; the viewport shows a window
/// of `config.cell_width` cells.  Call `pan_right` / `pan_left` to
/// advance or retreat the viewport.
pub struct BrailleDisplay {
    encoder: BrailleEncoder,
    config: BrailleDisplayConfig,
    /// Full encoded cell buffer for the current text.
    buffer: Vec<BrailleCell>,
    /// Offset (in cells) of the left edge of the viewport into `buffer`.
    pan_offset: usize,
}

impl BrailleDisplay {
    /// Create a new display with the given configuration.
    #[must_use]
    pub fn new(config: BrailleDisplayConfig) -> Self {
        let encoder = BrailleEncoder::new(config.grade);
        Self {
            encoder,
            config,
            buffer: Vec::new(),
            pan_offset: 0,
        }
    }

    /// Load text into the display, resetting the viewport to the beginning.
    pub fn load(&mut self, text: &str) {
        self.buffer = self.encoder.encode(text);
        self.pan_offset = 0;
    }

    /// Current viewport as a slice of [`BrailleCell`]s.
    ///
    /// Returns up to `config.cell_width` cells starting at `pan_offset`.
    #[must_use]
    pub fn viewport(&self) -> &[BrailleCell] {
        let start = self.pan_offset.min(self.buffer.len());
        let end = (start + self.config.cell_width).min(self.buffer.len());
        &self.buffer[start..end]
    }

    /// Render the current viewport as a Unicode Braille string.
    #[must_use]
    pub fn render_unicode(&self) -> String {
        BrailleEncoder::cells_to_unicode(self.viewport())
    }

    /// Render the entire cell buffer as a single UTF-8 Braille pattern string.
    ///
    /// Unlike [`render_unicode`] which only shows the current viewport, this
    /// method serialises every cell in the buffer from start to finish.
    /// Each `BrailleCell` value maps directly to the Unicode Braille Patterns
    /// block (U+2800–U+28FF): `U+2800 + dot_bitmask`.
    ///
    /// [`render_unicode`]: BrailleDisplay::render_unicode
    #[must_use]
    pub fn render(&self) -> String {
        self.buffer.iter().map(|cell| cell.to_unicode()).collect()
    }

    /// Render the entire cell buffer as a newline-separated Braille grid.
    ///
    /// The buffer is split into rows of exactly `width` cells (the last row
    /// may be shorter).  Each row is converted to a Unicode Braille string and
    /// the rows are joined with `'\n'`.  A `width` of 0 is treated as 1 to
    /// avoid a panic on the chunk split.
    #[must_use]
    pub fn render_grid(&self, width: usize) -> String {
        let w = width.max(1);
        self.buffer
            .chunks(w)
            .map(|row| row.iter().map(|cell| cell.to_unicode()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Pan the viewport right by one full page (cell_width cells).
    ///
    /// Returns `true` if there is more content after panning; `false` if we
    /// were already at or near the end.
    pub fn pan_right(&mut self) -> bool {
        let new_offset = self.pan_offset + self.config.cell_width;
        if new_offset < self.buffer.len() {
            self.pan_offset = new_offset;
            true
        } else {
            false
        }
    }

    /// Pan the viewport left by one full page (cell_width cells).
    ///
    /// Returns `true` if the viewport moved (was not already at the start).
    pub fn pan_left(&mut self) -> bool {
        if self.pan_offset == 0 {
            return false;
        }
        self.pan_offset = self.pan_offset.saturating_sub(self.config.cell_width);
        true
    }

    /// Pan to the very beginning of the buffer.
    pub fn pan_home(&mut self) {
        self.pan_offset = 0;
    }

    /// Pan to the end of the buffer.
    pub fn pan_end(&mut self) {
        self.pan_offset = self.buffer.len().saturating_sub(self.config.cell_width);
    }

    /// Total number of encoded cells in the current text.
    #[must_use]
    pub fn total_cells(&self) -> usize {
        self.buffer.len()
    }

    /// Current pan offset (left edge of viewport).
    #[must_use]
    pub fn pan_offset(&self) -> usize {
        self.pan_offset
    }

    /// Whether the viewport is at the very end of the buffer.
    #[must_use]
    pub fn at_end(&self) -> bool {
        self.pan_offset + self.config.cell_width >= self.buffer.len()
    }

    /// Number of pages required to display the full text.
    #[must_use]
    pub fn page_count(&self) -> usize {
        if self.buffer.is_empty() {
            return 0;
        }
        (self.buffer.len() + self.config.cell_width - 1) / self.config.cell_width
    }
}

impl Default for BrailleDisplay {
    fn default() -> Self {
        Self::new(BrailleDisplayConfig::default())
    }
}

#[cfg(test)]
mod braille_tests {
    use super::*;

    // ─── BrailleCell tests ─────────────────────────────────────────────────

    #[test]
    fn test_blank_cell() {
        let cell = BrailleCell::BLANK;
        assert_eq!(cell.0, 0);
        assert_eq!(cell.to_unicode(), '⠀');
    }

    #[test]
    fn test_cell_from_dots() {
        let cell = BrailleCell::from_dots(true, false, false, false, false, false);
        assert!(cell.dot(1));
        assert!(!cell.dot(2));
        assert_eq!(cell.0, 0b000001);
    }

    #[test]
    fn test_cell_dot_out_of_range() {
        let cell = BrailleCell(0xFF);
        assert!(!cell.dot(0));
        assert!(!cell.dot(9));
    }

    #[test]
    fn test_cell_to_unicode_a() {
        // Letter 'a' in Braille is dots 1 only = mask 0b000001 = U+2801
        let cell = BrailleCell(0b000001);
        assert_eq!(cell.to_unicode(), '\u{2801}');
    }

    #[test]
    fn test_cell_dot_count() {
        let cell = BrailleCell(0b000111); // dots 1, 2, 3
        assert_eq!(cell.dot_count(), 3);
    }

    #[test]
    fn test_cell_display() {
        let cell = BrailleCell(0b000001);
        let s = format!("{cell}");
        assert_eq!(s.chars().count(), 1);
    }

    // ─── BrailleEncoder Grade-1 tests ──────────────────────────────────────

    #[test]
    fn test_encoder_grade1_single_char() {
        let enc = BrailleEncoder::new(BrailleGrade::Grade1);
        let cells = enc.encode("a");
        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0], BrailleCell(0b000001));
    }

    #[test]
    fn test_encoder_grade1_space() {
        let enc = BrailleEncoder::new(BrailleGrade::Grade1);
        let cells = enc.encode(" ");
        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0], BrailleCell::BLANK);
    }

    #[test]
    fn test_encoder_grade1_hello() {
        let enc = BrailleEncoder::new(BrailleGrade::Grade1);
        let cells = enc.encode("hello");
        assert_eq!(cells.len(), 5);
        // h=0b010011, e=0b010001, l=0b000111, l, o=0b010101
        assert_eq!(cells[0], BrailleCell(0b010011));
        assert_eq!(cells[1], BrailleCell(0b010001));
    }

    #[test]
    fn test_encoder_grade1_unknown_char() {
        let enc = BrailleEncoder::new(BrailleGrade::Grade1);
        let cells = enc.encode("@"); // not in table
        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0], BrailleCell::BLANK);
    }

    #[test]
    fn test_encoder_grade1_case_insensitive() {
        let enc = BrailleEncoder::new(BrailleGrade::Grade1);
        let lower = enc.encode("abc");
        let upper = enc.encode("ABC");
        assert_eq!(lower, upper);
    }

    // ─── BrailleEncoder Grade-2 tests ──────────────────────────────────────

    #[test]
    fn test_encoder_grade2_the_contraction() {
        let enc = BrailleEncoder::new(BrailleGrade::Grade2);
        let cells = enc.encode("the");
        // Grade 2 contraction for "the" is 3 cells
        assert_eq!(cells.len(), 3);
    }

    #[test]
    fn test_encoder_grade2_and_contraction() {
        let enc = BrailleEncoder::new(BrailleGrade::Grade2);
        let cells = enc.encode("and");
        assert_eq!(cells.len(), 3);
    }

    #[test]
    fn test_encoder_grade2_cells_to_unicode() {
        let enc = BrailleEncoder::new(BrailleGrade::Grade2);
        let cells = enc.encode("hello");
        let unicode = BrailleEncoder::cells_to_unicode(&cells);
        assert!(!unicode.is_empty());
        // Every character should be in the Braille Unicode block
        for ch in unicode.chars() {
            let code = ch as u32;
            assert!(
                code >= 0x2800 && code <= 0x28FF,
                "char {ch:?} (U+{code:04X}) not in Braille block"
            );
        }
    }

    #[test]
    fn test_encoder_grade2_falls_back_for_non_contraction() {
        let enc = BrailleEncoder::new(BrailleGrade::Grade2);
        let cells = enc.encode("xyz");
        // No contractions for xyz — should still produce 3 cells
        assert_eq!(cells.len(), 3);
    }

    // ─── BrailleDisplay tests ──────────────────────────────────────────────

    #[test]
    fn test_display_default_empty() {
        let display = BrailleDisplay::default();
        assert_eq!(display.total_cells(), 0);
        assert_eq!(display.viewport().len(), 0);
    }

    #[test]
    fn test_display_load_text() {
        let mut display = BrailleDisplay::default();
        display.load("hello world");
        assert!(display.total_cells() > 0);
        assert_eq!(display.pan_offset(), 0);
    }

    #[test]
    fn test_display_viewport_width() {
        let cfg = BrailleDisplayConfig {
            cell_width: 10,
            grade: BrailleGrade::Grade1,
            wrap: true,
        };
        let mut display = BrailleDisplay::new(cfg);
        display.load("abcdefghijklmnopqrstuvwxyz");
        let vp = display.viewport();
        assert_eq!(vp.len(), 10);
    }

    #[test]
    fn test_display_pan_right() {
        let cfg = BrailleDisplayConfig {
            cell_width: 5,
            grade: BrailleGrade::Grade1,
            wrap: false,
        };
        let mut display = BrailleDisplay::new(cfg);
        display.load("abcdefghijklmnopqrst"); // 20 chars
        assert_eq!(display.pan_offset(), 0);
        let moved = display.pan_right();
        assert!(moved);
        assert_eq!(display.pan_offset(), 5);
    }

    #[test]
    fn test_display_pan_left() {
        let cfg = BrailleDisplayConfig {
            cell_width: 5,
            grade: BrailleGrade::Grade1,
            wrap: false,
        };
        let mut display = BrailleDisplay::new(cfg);
        display.load("abcdefghijklmnopqrst");
        display.pan_right();
        let moved = display.pan_left();
        assert!(moved);
        assert_eq!(display.pan_offset(), 0);
    }

    #[test]
    fn test_display_pan_left_at_start() {
        let mut display = BrailleDisplay::default();
        display.load("hello");
        let moved = display.pan_left();
        assert!(!moved);
    }

    #[test]
    fn test_display_pan_right_at_end() {
        let cfg = BrailleDisplayConfig {
            cell_width: 40,
            grade: BrailleGrade::Grade1,
            wrap: false,
        };
        let mut display = BrailleDisplay::new(cfg);
        display.load("hi"); // only 2 cells, far less than 40
        let moved = display.pan_right();
        assert!(!moved);
    }

    #[test]
    fn test_display_pan_home() {
        let cfg = BrailleDisplayConfig {
            cell_width: 5,
            grade: BrailleGrade::Grade1,
            wrap: false,
        };
        let mut display = BrailleDisplay::new(cfg);
        display.load("abcdefghijklmnopqrst");
        display.pan_right();
        display.pan_right();
        display.pan_home();
        assert_eq!(display.pan_offset(), 0);
    }

    #[test]
    fn test_display_pan_end() {
        let cfg = BrailleDisplayConfig {
            cell_width: 5,
            grade: BrailleGrade::Grade1,
            wrap: false,
        };
        let mut display = BrailleDisplay::new(cfg);
        display.load("abcdefghijklmnopqrst");
        display.pan_end();
        assert!(display.at_end());
    }

    #[test]
    fn test_display_render_unicode() {
        let mut display = BrailleDisplay::default();
        display.load("hello");
        let rendered = display.render_unicode();
        assert!(!rendered.is_empty());
        for ch in rendered.chars() {
            let code = ch as u32;
            assert!(code >= 0x2800 && code <= 0x28FF);
        }
    }

    #[test]
    fn test_display_page_count_empty() {
        let display = BrailleDisplay::default();
        assert_eq!(display.page_count(), 0);
    }

    #[test]
    fn test_display_page_count() {
        let cfg = BrailleDisplayConfig {
            cell_width: 10,
            grade: BrailleGrade::Grade1,
            wrap: true,
        };
        let mut display = BrailleDisplay::new(cfg);
        display.load("abcdefghijklmnopqrst"); // 20 chars -> 2 pages of 10
        assert_eq!(display.page_count(), 2);
    }

    #[test]
    fn test_display_load_resets_offset() {
        let cfg = BrailleDisplayConfig {
            cell_width: 5,
            grade: BrailleGrade::Grade1,
            wrap: false,
        };
        let mut display = BrailleDisplay::new(cfg);
        display.load("abcdefghij");
        display.pan_right();
        assert_eq!(display.pan_offset(), 5);
        display.load("new text");
        assert_eq!(display.pan_offset(), 0);
    }

    #[test]
    fn test_braille_display_config_standard_40() {
        let cfg = BrailleDisplayConfig::standard_40();
        assert_eq!(cfg.cell_width, 40);
        assert_eq!(cfg.grade, BrailleGrade::Grade2);
    }

    #[test]
    fn test_braille_display_config_compact_20() {
        let cfg = BrailleDisplayConfig::compact_20();
        assert_eq!(cfg.cell_width, 20);
        assert_eq!(cfg.grade, BrailleGrade::Grade1);
    }

    // ─── render() / render_grid() tests ───────────────────────────────────

    #[test]
    fn test_braille_render_basic() {
        // Cell value 0x01 (dot 1 only) must map to U+2801 ('⠁').
        let cfg = BrailleDisplayConfig {
            cell_width: 40,
            grade: BrailleGrade::Grade1,
            wrap: false,
        };
        let mut display = BrailleDisplay::new(cfg);
        // 'a' in Grade-1 Braille is mask 0b000001 = 0x01 → U+2801
        display.load("a");
        let rendered = display.render();
        assert_eq!(rendered.chars().count(), 1, "one cell → one char");
        let ch = rendered.chars().next().expect("render must be non-empty");
        assert_eq!(ch, '\u{2801}', "dot-1-only cell must render as U+2801 (⠁)");
    }

    #[test]
    fn test_braille_render_grid() {
        // Load 4 cells (letters a, b, c, d) and render at width=2
        // → should produce exactly 2 lines separated by '\n'.
        let cfg = BrailleDisplayConfig {
            cell_width: 40,
            grade: BrailleGrade::Grade1,
            wrap: false,
        };
        let mut display = BrailleDisplay::new(cfg);
        display.load("abcd"); // 4 cells in Grade-1
        let grid = display.render_grid(2);
        let lines: Vec<&str> = grid.split('\n').collect();
        assert_eq!(lines.len(), 2, "4 cells at width=2 must produce 2 lines");
        assert_eq!(
            lines[0].chars().count(),
            2,
            "first line must have exactly 2 Braille chars"
        );
        assert_eq!(
            lines[1].chars().count(),
            2,
            "second line must have exactly 2 Braille chars"
        );
        // Every character must be in the Braille Unicode block.
        for ch in grid.chars().filter(|&c| c != '\n') {
            let code = ch as u32;
            assert!(
                code >= 0x2800 && code <= 0x28FF,
                "char U+{code:04X} outside Braille block"
            );
        }
    }
}
