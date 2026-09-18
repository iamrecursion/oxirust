//! Real-time caption display with roll-up and paint-on modes.
//!
//! Supports two display modes:
//! - **Roll-up**: Captions scroll from bottom to top with configurable visible lines.
//! - **Paint-on**: Characters appear one at a time (typewriter effect).

/// Caption display mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptionMode {
    /// Roll-up mode: lines scroll upward.
    RollUp,
    /// Paint-on mode: characters revealed one by one.
    PaintOn,
}

/// A single caption event (text with timing).
#[derive(Clone, Debug)]
pub struct CaptionEvent {
    /// The text content.
    pub text: String,
    /// Timestamp when this event starts (milliseconds).
    pub start_ms: i64,
}

impl CaptionEvent {
    /// Create a new caption event.
    #[must_use]
    pub fn new(text: impl Into<String>, start_ms: i64) -> Self {
        Self {
            text: text.into(),
            start_ms,
        }
    }
}

/// Configuration for the live caption display.
#[derive(Clone, Debug)]
pub struct LiveCaptionConfig {
    /// Display mode.
    pub mode: CaptionMode,
    /// Number of visible lines for roll-up mode (typically 2-4).
    pub visible_lines: usize,
    /// Duration each line remains visible in roll-up mode (milliseconds).
    pub line_duration_ms: i64,
    /// Characters per second for paint-on mode.
    pub chars_per_second: f32,
    /// Maximum characters per line before wrapping.
    pub max_chars_per_line: usize,
}

impl Default for LiveCaptionConfig {
    fn default() -> Self {
        Self {
            mode: CaptionMode::RollUp,
            visible_lines: 3,
            line_duration_ms: 5000,
            chars_per_second: 30.0,
            max_chars_per_line: 32,
        }
    }
}

impl LiveCaptionConfig {
    /// Create a roll-up configuration.
    #[must_use]
    pub fn roll_up(visible_lines: usize) -> Self {
        Self {
            mode: CaptionMode::RollUp,
            visible_lines,
            ..Self::default()
        }
    }

    /// Create a paint-on configuration.
    #[must_use]
    pub fn paint_on(chars_per_second: f32) -> Self {
        Self {
            mode: CaptionMode::PaintOn,
            chars_per_second,
            ..Self::default()
        }
    }
}

/// A line in the caption display with its own timing.
#[derive(Clone, Debug)]
struct DisplayLine {
    /// The text content.
    text: String,
    /// When this line was created / scrolled in (milliseconds).
    created_ms: i64,
}

/// Real-time caption display engine.
///
/// Manages a scrolling/painting display of caption text that can be
/// updated incrementally as new text arrives.
#[derive(Clone, Debug)]
pub struct LiveCaptionDisplay {
    /// Configuration.
    config: LiveCaptionConfig,
    /// All lines accumulated so far (for roll-up mode).
    lines: Vec<DisplayLine>,
    /// Current paint-on buffer (text being revealed).
    paint_buffer: String,
    /// When the current paint-on text started.
    paint_start_ms: i64,
    /// Full text pending for paint-on reveal.
    paint_full_text: String,
}

impl LiveCaptionDisplay {
    /// Create a new display with the given configuration.
    #[must_use]
    pub fn new(config: LiveCaptionConfig) -> Self {
        Self {
            config,
            lines: Vec::new(),
            paint_buffer: String::new(),
            paint_start_ms: 0,
            paint_full_text: String::new(),
        }
    }

    /// Create a display with default roll-up configuration.
    #[must_use]
    pub fn default_roll_up() -> Self {
        Self::new(LiveCaptionConfig::roll_up(3))
    }

    /// Create a display with default paint-on configuration.
    #[must_use]
    pub fn default_paint_on() -> Self {
        Self::new(LiveCaptionConfig::paint_on(30.0))
    }

    /// Get the current display mode.
    #[must_use]
    pub fn mode(&self) -> CaptionMode {
        self.config.mode
    }

    /// Get the number of visible lines configured.
    #[must_use]
    pub fn visible_lines(&self) -> usize {
        self.config.visible_lines
    }

    /// Feed a new caption event into the display.
    pub fn feed(&mut self, event: &CaptionEvent) {
        match self.config.mode {
            CaptionMode::RollUp => {
                self.feed_roll_up(event);
            }
            CaptionMode::PaintOn => {
                self.feed_paint_on(event);
            }
        }
    }

    /// Feed text in roll-up mode.
    fn feed_roll_up(&mut self, event: &CaptionEvent) {
        // Word-wrap the text into lines
        let wrapped = word_wrap(&event.text, self.config.max_chars_per_line);
        for line_text in wrapped {
            self.lines.push(DisplayLine {
                text: line_text,
                created_ms: event.start_ms,
            });
        }
    }

    /// Feed text in paint-on mode.
    fn feed_paint_on(&mut self, event: &CaptionEvent) {
        self.paint_full_text = event.text.clone();
        self.paint_start_ms = event.start_ms;
        self.paint_buffer.clear();
    }

    /// Render the current display at the given timestamp.
    ///
    /// Returns the visible lines of text.
    #[must_use]
    pub fn render(&self, timestamp_ms: i64) -> Vec<String> {
        match self.config.mode {
            CaptionMode::RollUp => self.render_roll_up(timestamp_ms),
            CaptionMode::PaintOn => self.render_paint_on(timestamp_ms),
        }
    }

    /// Render roll-up display.
    fn render_roll_up(&self, timestamp_ms: i64) -> Vec<String> {
        // Filter to lines that are still visible (not expired)
        let visible: Vec<&DisplayLine> = self
            .lines
            .iter()
            .filter(|line| {
                let age = timestamp_ms - line.created_ms;
                age >= 0 && age < self.config.line_duration_ms
            })
            .collect();

        // Take only the last N visible lines
        let start = if visible.len() > self.config.visible_lines {
            visible.len() - self.config.visible_lines
        } else {
            0
        };

        visible[start..]
            .iter()
            .map(|line| line.text.clone())
            .collect()
    }

    /// Render paint-on display.
    fn render_paint_on(&self, timestamp_ms: i64) -> Vec<String> {
        if self.paint_full_text.is_empty() {
            return Vec::new();
        }

        let elapsed_ms = timestamp_ms - self.paint_start_ms;
        if elapsed_ms < 0 {
            return Vec::new();
        }

        // Calculate how many characters should be visible
        let elapsed_secs = elapsed_ms as f32 / 1000.0;
        let chars_to_show = (elapsed_secs * self.config.chars_per_second) as usize;
        let chars_to_show = chars_to_show.min(self.paint_full_text.len());

        if chars_to_show == 0 {
            return Vec::new();
        }

        // Get the visible portion
        let visible_text: String = self.paint_full_text.chars().take(chars_to_show).collect();

        // Word-wrap the visible portion
        word_wrap(&visible_text, self.config.max_chars_per_line)
    }

    /// Clear all displayed captions.
    pub fn clear(&mut self) {
        self.lines.clear();
        self.paint_buffer.clear();
        self.paint_full_text.clear();
    }

    /// Get the total number of lines accumulated (for roll-up mode).
    #[must_use]
    pub fn total_lines(&self) -> usize {
        self.lines.len()
    }

    /// Check if the display is currently empty.
    #[must_use]
    pub fn is_empty(&self, timestamp_ms: i64) -> bool {
        self.render(timestamp_ms).is_empty()
    }

    /// Get the scroll progress for roll-up mode.
    ///
    /// Returns a value between 0.0 (no scroll needed) and 1.0 (fully scrolled).
    /// This can be used to animate smooth scrolling between lines.
    #[must_use]
    pub fn scroll_progress(&self, timestamp_ms: i64) -> f32 {
        if self.config.mode != CaptionMode::RollUp || self.lines.is_empty() {
            return 0.0;
        }

        // Find the most recently added line
        if let Some(last) = self.lines.last() {
            let age = timestamp_ms - last.created_ms;
            if age < 0 {
                return 0.0;
            }
            // Smooth scroll over 300ms when a new line arrives
            let scroll_time = 300.0_f32;
            (age as f32 / scroll_time).min(1.0)
        } else {
            0.0
        }
    }

    /// Get paint-on reveal progress (0.0 to 1.0).
    #[must_use]
    pub fn paint_progress(&self, timestamp_ms: i64) -> f32 {
        if self.config.mode != CaptionMode::PaintOn || self.paint_full_text.is_empty() {
            return 0.0;
        }

        let elapsed_ms = timestamp_ms - self.paint_start_ms;
        if elapsed_ms < 0 {
            return 0.0;
        }

        let elapsed_secs = elapsed_ms as f32 / 1000.0;
        let chars_to_show = elapsed_secs * self.config.chars_per_second;
        let total_chars = self.paint_full_text.len() as f32;

        if total_chars <= 0.0 {
            return 0.0;
        }

        (chars_to_show / total_chars).min(1.0)
    }
}

/// Simple word-wrap function that breaks text into lines.
fn word_wrap(text: &str, max_chars: usize) -> Vec<String> {
    if max_chars == 0 {
        return vec![text.to_string()];
    }

    let mut lines = Vec::new();
    let mut current_line = String::new();

    for word in text.split_whitespace() {
        if current_line.is_empty() {
            if word.len() > max_chars {
                // Break long words
                let mut remaining = word;
                while remaining.len() > max_chars {
                    let (chunk, rest) = remaining.split_at(max_chars);
                    lines.push(chunk.to_string());
                    remaining = rest;
                }
                current_line = remaining.to_string();
            } else {
                current_line = word.to_string();
            }
        } else if current_line.len() + 1 + word.len() > max_chars {
            lines.push(current_line);
            current_line = word.to_string();
        } else {
            current_line.push(' ');
            current_line.push_str(word);
        }
    }

    if !current_line.is_empty() {
        lines.push(current_line);
    }

    if lines.is_empty() && !text.is_empty() {
        lines.push(String::new());
    }

    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_caption_event_basic() {
        let event = CaptionEvent::new("Hello", 1000);
        assert_eq!(event.text, "Hello");
        assert_eq!(event.start_ms, 1000);
    }

    #[test]
    fn test_config_default() {
        let config = LiveCaptionConfig::default();
        assert_eq!(config.mode, CaptionMode::RollUp);
        assert_eq!(config.visible_lines, 3);
    }

    #[test]
    fn test_config_roll_up() {
        let config = LiveCaptionConfig::roll_up(4);
        assert_eq!(config.mode, CaptionMode::RollUp);
        assert_eq!(config.visible_lines, 4);
    }

    #[test]
    fn test_config_paint_on() {
        let config = LiveCaptionConfig::paint_on(20.0);
        assert_eq!(config.mode, CaptionMode::PaintOn);
        assert!((config.chars_per_second - 20.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_roll_up_basic() {
        let mut display = LiveCaptionDisplay::new(LiveCaptionConfig::roll_up(2));
        display.feed(&CaptionEvent::new("Line 1", 0));
        display.feed(&CaptionEvent::new("Line 2", 100));
        display.feed(&CaptionEvent::new("Line 3", 200));

        let rendered = display.render(300);
        // Should show only the last 2 lines
        assert_eq!(rendered.len(), 2);
        assert_eq!(rendered[0], "Line 2");
        assert_eq!(rendered[1], "Line 3");
    }

    #[test]
    fn test_roll_up_expiry() {
        let mut display = LiveCaptionDisplay::new(LiveCaptionConfig {
            mode: CaptionMode::RollUp,
            visible_lines: 3,
            line_duration_ms: 1000,
            chars_per_second: 30.0,
            max_chars_per_line: 32,
        });
        display.feed(&CaptionEvent::new("Old line", 0));
        display.feed(&CaptionEvent::new("New line", 200));

        // At t=500, both should be visible (old age=500 < 1000, new age=300 < 1000)
        let rendered = display.render(500);
        assert_eq!(rendered.len(), 2);

        // At t=1100, old line should be expired (age=1100 >= 1000), new still visible (age=900 < 1000)
        let rendered = display.render(1100);
        assert_eq!(rendered.len(), 1);
        assert_eq!(rendered[0], "New line");
    }

    #[test]
    fn test_paint_on_basic() {
        let mut display = LiveCaptionDisplay::new(LiveCaptionConfig {
            mode: CaptionMode::PaintOn,
            visible_lines: 3,
            line_duration_ms: 5000,
            chars_per_second: 10.0, // 10 chars per second
            max_chars_per_line: 32,
        });
        display.feed(&CaptionEvent::new("Hello World", 0));

        // At t=0, no characters yet (need at least 100ms for 1 char)
        let rendered = display.render(0);
        assert!(rendered.is_empty());

        // At t=500, should show 5 characters
        let rendered = display.render(500);
        assert_eq!(rendered.len(), 1);
        assert_eq!(rendered[0], "Hello");

        // At t=2000, should show full text
        let rendered = display.render(2000);
        assert_eq!(rendered.len(), 1);
        assert_eq!(rendered[0], "Hello World");
    }

    #[test]
    fn test_paint_on_progress() {
        let mut display = LiveCaptionDisplay::new(LiveCaptionConfig::paint_on(10.0));
        display.feed(&CaptionEvent::new("ABCDEFGHIJ", 0)); // 10 chars

        assert!((display.paint_progress(0) - 0.0).abs() < f32::EPSILON);
        assert!((display.paint_progress(500) - 0.5).abs() < 0.01);
        assert!((display.paint_progress(1000) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_display_clear() {
        let mut display = LiveCaptionDisplay::default_roll_up();
        display.feed(&CaptionEvent::new("Test", 0));
        assert!(!display.is_empty(0));
        display.clear();
        assert!(display.is_empty(0));
    }

    #[test]
    fn test_display_mode() {
        let display = LiveCaptionDisplay::default_roll_up();
        assert_eq!(display.mode(), CaptionMode::RollUp);
        let display = LiveCaptionDisplay::default_paint_on();
        assert_eq!(display.mode(), CaptionMode::PaintOn);
    }

    #[test]
    fn test_word_wrap() {
        let lines = word_wrap("Hello World", 5);
        assert_eq!(lines, vec!["Hello", "World"]);
    }

    #[test]
    fn test_word_wrap_long_word() {
        let lines = word_wrap("Supercalifragilistic", 10);
        assert_eq!(lines, vec!["Supercalif", "ragilistic"]);
    }

    #[test]
    fn test_word_wrap_fits() {
        let lines = word_wrap("Hi", 10);
        assert_eq!(lines, vec!["Hi"]);
    }

    #[test]
    fn test_scroll_progress() {
        let mut display = LiveCaptionDisplay::default_roll_up();
        display.feed(&CaptionEvent::new("Test", 0));
        assert!((display.scroll_progress(0) - 0.0).abs() < f32::EPSILON);
        assert!((display.scroll_progress(300) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_visible_lines_config() {
        let display = LiveCaptionDisplay::new(LiveCaptionConfig::roll_up(4));
        assert_eq!(display.visible_lines(), 4);
    }

    #[test]
    fn test_total_lines() {
        let mut display = LiveCaptionDisplay::default_roll_up();
        assert_eq!(display.total_lines(), 0);
        display.feed(&CaptionEvent::new("A", 0));
        display.feed(&CaptionEvent::new("B", 100));
        assert_eq!(display.total_lines(), 2);
    }

    #[test]
    fn test_roll_up_before_start() {
        let mut display = LiveCaptionDisplay::default_roll_up();
        display.feed(&CaptionEvent::new("Future", 1000));
        let rendered = display.render(500);
        assert!(rendered.is_empty());
    }

    #[test]
    fn test_paint_on_before_start() {
        let mut display = LiveCaptionDisplay::default_paint_on();
        display.feed(&CaptionEvent::new("Future", 1000));
        let rendered = display.render(500);
        assert!(rendered.is_empty());
    }

    #[test]
    fn test_paint_on_empty() {
        let display = LiveCaptionDisplay::default_paint_on();
        let rendered = display.render(1000);
        assert!(rendered.is_empty());
    }

    #[test]
    fn test_scroll_progress_paint_on_mode() {
        let display = LiveCaptionDisplay::default_paint_on();
        assert!((display.scroll_progress(100) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_paint_progress_roll_up_mode() {
        let display = LiveCaptionDisplay::default_roll_up();
        assert!((display.paint_progress(100) - 0.0).abs() < f32::EPSILON);
    }
}
