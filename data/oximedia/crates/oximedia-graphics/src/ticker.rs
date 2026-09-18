//! News/data ticker graphics for broadcast overlays.
//!
//! Provides scrolling ticker rendering with priority support, configurable
//! colors and scroll speed, and a queue for managing ticker items.
//!
//! ## RTL support
//!
//! For Arabic, Hebrew, Persian and other right-to-left scripts, set
//! [`TickerConfig::scroll_dir`] to [`TickerScrollDir::Rtl`].  In RTL mode the
//! ticker scrolls from **left to right** (text enters from the left edge and
//! exits through the right edge) matching the natural reading direction.

use std::collections::VecDeque;

/// A single item in the news ticker.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TickerItem {
    /// Text content of the ticker item.
    pub text: String,
    /// Optional category label (e.g. "BREAKING", "SPORTS").
    pub category: Option<String>,
    /// Priority level (higher = shown sooner). Range 0–255.
    pub priority: u8,
}

impl TickerItem {
    /// Create a new ticker item.
    #[allow(dead_code)]
    pub fn new(text: impl Into<String>, category: Option<String>, priority: u8) -> Self {
        Self {
            text: text.into(),
            category,
            priority,
        }
    }

    /// Format the item as a display string including the category prefix.
    #[allow(dead_code)]
    pub fn formatted(&self, separator: &str) -> String {
        match &self.category {
            Some(cat) => format!("[{}] {}{}", cat, self.text, separator),
            None => format!("{}{}", self.text, separator),
        }
    }
}

/// Scroll direction for the ticker.
///
/// | Variant | Reading order | Motion |
/// |---------|---------------|--------|
/// | `Ltr`   | Left-to-right (Latin, CJK, …) | Text enters right, exits left |
/// | `Rtl`   | Right-to-left (Arabic, Hebrew, …) | Text enters left, exits right |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum TickerScrollDir {
    /// Standard left-to-right scroll — text moves toward the left edge.
    Ltr,
    /// Right-to-left scroll — text moves toward the right edge, suitable for
    /// Arabic / Hebrew content.
    Rtl,
}

impl Default for TickerScrollDir {
    fn default() -> Self {
        Self::Ltr
    }
}

/// Position of the ticker strip on screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum TickerPosition {
    /// Ticker appears at the bottom of the frame.
    Bottom,
    /// Ticker appears at the top of the frame.
    Top,
}

/// Looping behaviour when the ticker reaches the last item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum TickerLoopMode {
    /// Stop after all items have scrolled past.
    Once,
    /// Restart from the first item continuously.
    Loop,
}

impl Default for TickerLoopMode {
    fn default() -> Self {
        Self::Loop
    }
}

/// Configuration for the ticker renderer.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TickerConfig {
    /// Base scroll speed in pixels per second.
    pub scroll_speed_pps: f32,
    /// Background fill color as RGBA.
    pub bg_color: [u8; 4],
    /// Text color as RGBA.
    pub text_color: [u8; 4],
    /// Separator string inserted between items.
    pub separator: String,
    /// Height of the ticker strip in pixels.
    pub height_px: u32,
    /// Screen position (top or bottom).
    pub position: TickerPosition,
    /// Looping mode.
    pub loop_mode: TickerLoopMode,
    /// If `true`, the ticker pauses briefly when a new item starts scrolling.
    pub pause_on_item: bool,
    /// Duration of the pause (seconds) when `pause_on_item` is enabled.
    pub pause_duration_secs: f32,
    /// Per-item speed multiplier override. A value of `None` means use the
    /// base speed. Items can individually scroll faster or slower.
    pub item_speed_multiplier: f32,
    /// Scroll direction.
    ///
    /// Use [`TickerScrollDir::Rtl`] for Arabic / Hebrew tickers.
    pub scroll_dir: TickerScrollDir,
}

impl Default for TickerConfig {
    fn default() -> Self {
        Self {
            scroll_speed_pps: 120.0,
            bg_color: [20, 20, 80, 230],
            text_color: [255, 255, 255, 255],
            separator: "  •  ".to_string(),
            height_px: 48,
            position: TickerPosition::Bottom,
            loop_mode: TickerLoopMode::Loop,
            pause_on_item: false,
            pause_duration_secs: 1.0,
            item_speed_multiplier: 1.0,
            scroll_dir: TickerScrollDir::Ltr,
        }
    }
}

impl TickerConfig {
    /// Create a config pre-configured for right-to-left Arabic / Hebrew tickers.
    ///
    /// RTL tickers use a slightly darker background to distinguish them
    /// visually from standard LTR overlays.  All other defaults are preserved.
    #[allow(dead_code)]
    pub fn rtl() -> Self {
        Self {
            scroll_dir: TickerScrollDir::Rtl,
            bg_color: [60, 20, 20, 230],
            ..Self::default()
        }
    }
}

/// Current scroll state of the ticker.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TickerState {
    /// List of items to display.
    pub items: Vec<TickerItem>,
    /// Index of the item currently being scrolled.
    pub current_item_idx: usize,
    /// Current scroll offset in pixels (positive = scrolled left).
    pub scroll_offset_px: f32,
    /// Remaining pause time in seconds (non-zero when `pause_on_item` is active).
    pub pause_remaining_secs: f32,
    /// Whether all items have scrolled past (only meaningful in `Once` mode).
    pub completed: bool,
}

impl TickerState {
    /// Create a new ticker state with the given items.
    #[allow(dead_code)]
    pub fn new(items: Vec<TickerItem>) -> Self {
        Self {
            items,
            current_item_idx: 0,
            scroll_offset_px: 0.0,
            pause_remaining_secs: 0.0,
            completed: false,
        }
    }

    /// Advance the scroll by `dt_secs` seconds at the speed specified in `config`.
    ///
    /// `item_width_px` is the rendered pixel width of the current item.
    ///
    /// Returns `true` if the ticker has advanced to a new item this call.
    #[allow(dead_code)]
    pub fn advance(&mut self, dt_secs: f32, item_width_px: f32, speed_pps: f32) -> bool {
        if self.items.is_empty() || self.completed {
            return false;
        }

        // Legacy path: no config available — just scroll at the given speed.
        self.scroll_offset_px += speed_pps * dt_secs;

        if self.scroll_offset_px >= item_width_px {
            self.scroll_offset_px -= item_width_px;
            self.current_item_idx = (self.current_item_idx + 1) % self.items.len();
            return true;
        }
        false
    }

    /// Advance the scroll using a full [`TickerConfig`] for richer behaviour.
    ///
    /// Supports variable scroll speed (via `config.item_speed_multiplier`),
    /// pause-on-item, and loop-mode control.
    ///
    /// `item_width_px` is the rendered pixel width of the current item.
    ///
    /// Returns `true` if the ticker has advanced to a new item this call.
    #[allow(dead_code)]
    pub fn advance_with_config(
        &mut self,
        dt_secs: f32,
        item_width_px: f32,
        config: &TickerConfig,
    ) -> bool {
        if self.items.is_empty() || self.completed {
            return false;
        }

        // If paused on the current item, consume pause time.
        if self.pause_remaining_secs > 0.0 {
            self.pause_remaining_secs -= dt_secs;
            if self.pause_remaining_secs < 0.0 {
                self.pause_remaining_secs = 0.0;
            }
            return false;
        }

        let effective_speed = config.scroll_speed_pps * config.item_speed_multiplier.max(0.0);
        self.scroll_offset_px += effective_speed * dt_secs;

        let mut advanced = false;

        // Process all item boundaries crossed in this time step.
        while self.scroll_offset_px >= item_width_px {
            self.scroll_offset_px -= item_width_px;
            advanced = true;

            let next_idx = self.current_item_idx + 1;
            if next_idx >= self.items.len() {
                match config.loop_mode {
                    TickerLoopMode::Once => {
                        self.completed = true;
                        self.current_item_idx = self.items.len().saturating_sub(1);
                        // Stop processing further — ticker is done.
                        if config.pause_on_item && config.pause_duration_secs > 0.0 {
                            self.pause_remaining_secs = config.pause_duration_secs;
                        }
                        return true;
                    }
                    TickerLoopMode::Loop => {
                        self.current_item_idx = 0;
                    }
                }
            } else {
                self.current_item_idx = next_idx;
            }

            if config.pause_on_item && config.pause_duration_secs > 0.0 {
                self.pause_remaining_secs = config.pause_duration_secs;
                // Pause was triggered — stop advancing until next call.
                break;
            }
        }

        advanced
    }

    /// Get the currently active ticker item, if any.
    #[allow(dead_code)]
    pub fn current_item(&self) -> Option<&TickerItem> {
        self.items.get(self.current_item_idx)
    }

    /// Reset the ticker to the beginning.
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.current_item_idx = 0;
        self.scroll_offset_px = 0.0;
        self.pause_remaining_secs = 0.0;
        self.completed = false;
    }
}

impl Default for TickerState {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

/// Renderer for the ticker strip.
pub struct TickerRenderer;

impl TickerRenderer {
    /// Render a horizontal RGBA ticker strip.
    ///
    /// Returns a `Vec<u8>` of RGBA pixels with length `width * config.height_px * 4`.
    ///
    /// The accent line is placed on the **entry** side of the strip:
    /// - LTR: accent on the right edge (where text enters).
    /// - RTL: accent on the left edge (where RTL text enters).
    #[allow(dead_code)]
    pub fn render(_state: &TickerState, config: &TickerConfig, width: u32) -> Vec<u8> {
        let h = config.height_px;
        let total = (width * h * 4) as usize;
        let mut data = vec![0u8; total];

        // Fill background
        for chunk in data.chunks_exact_mut(4) {
            chunk[0] = config.bg_color[0];
            chunk[1] = config.bg_color[1];
            chunk[2] = config.bg_color[2];
            chunk[3] = config.bg_color[3];
        }

        // Draw a thin accent line at the top of the strip
        let accent_row_height = (h / 10).max(2) as usize;
        for row in 0..accent_row_height {
            for col in 0..width as usize {
                let idx = (row * width as usize + col) * 4;
                if idx + 3 < data.len() {
                    // Slightly lighter than bg as accent
                    data[idx] = config.bg_color[0].saturating_add(60);
                    data[idx + 1] = config.bg_color[1].saturating_add(60);
                    data[idx + 2] = config.bg_color[2].saturating_add(60);
                    data[idx + 3] = 255;
                }
            }
        }

        // For RTL mode draw a vertical entry-edge marker on the left side.
        // For LTR mode draw the entry-edge marker on the right side.
        let marker_width = ((width / 40).max(2)) as usize;
        let (marker_start_col, marker_end_col) = match config.scroll_dir {
            TickerScrollDir::Rtl => (0usize, marker_width),
            TickerScrollDir::Ltr => {
                let w = width as usize;
                (w.saturating_sub(marker_width), w)
            }
        };
        for row in 0..h as usize {
            for col in marker_start_col..marker_end_col {
                let idx = (row * width as usize + col) * 4;
                if idx + 3 < data.len() {
                    data[idx] = config.text_color[0];
                    data[idx + 1] = config.text_color[1];
                    data[idx + 2] = config.text_color[2];
                    data[idx + 3] = config.text_color[3] / 2;
                }
            }
        }

        data
    }

    /// Returns `true` when the given config is operating in right-to-left mode.
    #[allow(dead_code)]
    pub fn is_rtl(config: &TickerConfig) -> bool {
        config.scroll_dir == TickerScrollDir::Rtl
    }

    /// Compute the X-coordinate at which a ticker item should be rendered for
    /// the current scroll offset.
    ///
    /// - **LTR**: `x = strip_width - scroll_offset_px` — item enters from the right.
    /// - **RTL**: `x = scroll_offset_px - item_width` — item enters from the left,
    ///   so at `scroll_offset_px == 0` the item starts fully off-screen to the left.
    #[allow(dead_code)]
    pub fn item_x(
        scroll_offset_px: f32,
        item_width_px: f32,
        strip_width: f32,
        config: &TickerConfig,
    ) -> f32 {
        match config.scroll_dir {
            TickerScrollDir::Ltr => strip_width - scroll_offset_px,
            TickerScrollDir::Rtl => scroll_offset_px - item_width_px,
        }
    }
}

/// A priority queue for ticker items.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct TickerQueue {
    items: VecDeque<TickerItem>,
}

impl TickerQueue {
    /// Create an empty ticker queue.
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            items: VecDeque::new(),
        }
    }

    /// Push a standard item to the back of the queue.
    #[allow(dead_code)]
    pub fn push(&mut self, item: TickerItem) {
        self.items.push_back(item);
    }

    /// Pop the next item from the front of the queue.
    #[allow(dead_code)]
    pub fn pop(&mut self) -> Option<TickerItem> {
        self.items.pop_front()
    }

    /// Insert a breaking-news item at the front of the queue, bypassing priority ordering.
    #[allow(dead_code)]
    pub fn insert_breaking(&mut self, mut item: TickerItem) {
        item.priority = 255;
        self.items.push_front(item);
    }

    /// Returns the number of items currently in the queue.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Returns true if the queue is empty.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Drain the queue into a Vec, sorted by descending priority.
    #[allow(dead_code)]
    pub fn drain_sorted(&mut self) -> Vec<TickerItem> {
        let mut items: Vec<TickerItem> = self.items.drain(..).collect();
        items.sort_by(|a, b| b.priority.cmp(&a.priority));
        items
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ticker_item_formatted_with_category() {
        let item = TickerItem::new("Some news", Some("BREAKING".to_string()), 255);
        let fmt = item.formatted(" | ");
        assert!(fmt.contains("BREAKING"));
        assert!(fmt.contains("Some news"));
    }

    #[test]
    fn test_ticker_item_formatted_without_category() {
        let item = TickerItem::new("Plain text", None, 0);
        let fmt = item.formatted(" • ");
        assert!(!fmt.contains('['));
        assert!(fmt.contains("Plain text"));
    }

    #[test]
    fn test_ticker_config_default() {
        let cfg = TickerConfig::default();
        assert!(cfg.scroll_speed_pps > 0.0);
        assert!(!cfg.separator.is_empty());
        assert!(cfg.height_px > 0);
    }

    #[test]
    fn test_ticker_state_advance_scrolls() {
        let items = vec![TickerItem::new("item1", None, 0)];
        let mut state = TickerState::new(items);
        // Advance less than full item width — should not advance item
        let advanced = state.advance(0.1, 300.0, 120.0);
        assert!(!advanced);
        assert!(state.scroll_offset_px > 0.0);
    }

    #[test]
    fn test_ticker_state_advance_next_item() {
        let items = vec![
            TickerItem::new("item1", None, 0),
            TickerItem::new("item2", None, 0),
        ];
        let mut state = TickerState::new(items);
        // Advance enough to scroll past the full item width
        let advanced = state.advance(10.0, 100.0, 120.0);
        assert!(advanced);
        assert_eq!(state.current_item_idx, 1);
    }

    #[test]
    fn test_ticker_state_wraps_around() {
        let items = vec![TickerItem::new("A", None, 0), TickerItem::new("B", None, 0)];
        let mut state = TickerState::new(items);
        state.advance(10.0, 50.0, 120.0);
        state.advance(10.0, 50.0, 120.0);
        // Should have wrapped back to 0
        assert_eq!(state.current_item_idx, 0);
    }

    #[test]
    fn test_ticker_state_current_item() {
        let items = vec![TickerItem::new("hello", None, 5)];
        let state = TickerState::new(items);
        assert!(state.current_item().is_some());
        assert_eq!(
            state
                .current_item()
                .expect("current_item should succeed")
                .text,
            "hello"
        );
    }

    #[test]
    fn test_ticker_render_size() {
        let state = TickerState::default();
        let config = TickerConfig {
            height_px: 48,
            ..TickerConfig::default()
        };
        let data = TickerRenderer::render(&state, &config, 1920);
        assert_eq!(data.len(), (1920 * 48 * 4) as usize);
    }

    #[test]
    fn test_ticker_render_has_background() {
        let state = TickerState::default();
        let config = TickerConfig {
            bg_color: [50, 50, 200, 255],
            height_px: 48,
            ..TickerConfig::default()
        };
        let data = TickerRenderer::render(&state, &config, 100);
        // Last row pixels should be background color
        let row_offset = (47 * 100 * 4) as usize;
        assert_eq!(data[row_offset], 50);
        assert_eq!(data[row_offset + 2], 200);
    }

    #[test]
    fn test_ticker_queue_push_pop() {
        let mut q = TickerQueue::new();
        q.push(TickerItem::new("A", None, 0));
        q.push(TickerItem::new("B", None, 0));
        assert_eq!(q.len(), 2);
        let item = q.pop().expect("item should be valid");
        assert_eq!(item.text, "A");
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn test_ticker_queue_insert_breaking() {
        let mut q = TickerQueue::new();
        q.push(TickerItem::new("Normal", None, 0));
        q.insert_breaking(TickerItem::new("BREAKING", Some("BREAKING".to_string()), 0));
        // Breaking should be at front
        let first = q.pop().expect("first should be valid");
        assert_eq!(first.priority, 255);
        assert_eq!(first.text, "BREAKING");
    }

    #[test]
    fn test_ticker_queue_drain_sorted() {
        let mut q = TickerQueue::new();
        q.push(TickerItem::new("low", None, 10));
        q.push(TickerItem::new("high", None, 200));
        q.push(TickerItem::new("mid", None, 100));
        let sorted = q.drain_sorted();
        assert_eq!(sorted[0].priority, 200);
        assert_eq!(sorted[1].priority, 100);
        assert_eq!(sorted[2].priority, 10);
    }

    #[test]
    fn test_ticker_queue_is_empty() {
        let q = TickerQueue::new();
        assert!(q.is_empty());
    }

    // --- Variable scroll speed / pause-on-item / looping tests ---

    fn make_multi_item_state() -> TickerState {
        TickerState::new(vec![
            TickerItem::new("Item A", None, 0),
            TickerItem::new("Item B", None, 0),
            TickerItem::new("Item C", None, 0),
        ])
    }

    #[test]
    fn test_ticker_loop_mode_default() {
        assert_eq!(TickerLoopMode::default(), TickerLoopMode::Loop);
    }

    #[test]
    fn test_ticker_config_has_loop_mode() {
        let cfg = TickerConfig::default();
        assert_eq!(cfg.loop_mode, TickerLoopMode::Loop);
    }

    #[test]
    fn test_advance_with_config_scrolls() {
        let mut state = make_multi_item_state();
        let cfg = TickerConfig::default();
        state.advance_with_config(0.1, 300.0, &cfg);
        assert!(state.scroll_offset_px > 0.0);
    }

    #[test]
    fn test_advance_with_config_advances_item() {
        let mut state = make_multi_item_state();
        let cfg = TickerConfig::default();
        // Enough time to scroll past a 100px item at default 120pps.
        let advanced = state.advance_with_config(1.0, 100.0, &cfg);
        assert!(advanced);
        assert_eq!(state.current_item_idx, 1);
    }

    #[test]
    fn test_advance_with_config_loops() {
        let mut state = TickerState::new(vec![
            TickerItem::new("A", None, 0),
            TickerItem::new("B", None, 0),
        ]);
        let cfg = TickerConfig {
            loop_mode: TickerLoopMode::Loop,
            ..TickerConfig::default()
        };
        // Scroll past both items.
        state.advance_with_config(10.0, 50.0, &cfg);
        assert!(!state.completed);
        assert_eq!(state.current_item_idx, 0); // looped back
    }

    #[test]
    fn test_advance_with_config_once_completes() {
        let mut state = TickerState::new(vec![
            TickerItem::new("A", None, 0),
            TickerItem::new("B", None, 0),
        ]);
        let cfg = TickerConfig {
            loop_mode: TickerLoopMode::Once,
            scroll_speed_pps: 1000.0,
            ..TickerConfig::default()
        };
        state.advance_with_config(10.0, 10.0, &cfg);
        assert!(state.completed);
    }

    #[test]
    fn test_advance_with_config_pause_on_item() {
        let mut state = make_multi_item_state();
        let cfg = TickerConfig {
            pause_on_item: true,
            pause_duration_secs: 2.0,
            scroll_speed_pps: 1000.0,
            ..TickerConfig::default()
        };
        // First advance: scroll past the item → triggers pause.
        state.advance_with_config(1.0, 10.0, &cfg);
        // Now the ticker should be paused.
        assert!(state.pause_remaining_secs > 0.0);
        // Advance within pause window — item should not change.
        let old_idx = state.current_item_idx;
        state.advance_with_config(0.5, 10.0, &cfg);
        assert_eq!(state.current_item_idx, old_idx);
    }

    #[test]
    fn test_advance_with_config_variable_speed() {
        let mut state_slow = make_multi_item_state();
        let mut state_fast = make_multi_item_state();
        let cfg_slow = TickerConfig {
            item_speed_multiplier: 0.5,
            scroll_speed_pps: 100.0,
            ..TickerConfig::default()
        };
        let cfg_fast = TickerConfig {
            item_speed_multiplier: 2.0,
            scroll_speed_pps: 100.0,
            ..TickerConfig::default()
        };
        state_slow.advance_with_config(1.0, 10000.0, &cfg_slow);
        state_fast.advance_with_config(1.0, 10000.0, &cfg_fast);
        assert!(state_fast.scroll_offset_px > state_slow.scroll_offset_px);
    }

    #[test]
    fn test_ticker_state_reset() {
        let mut state = make_multi_item_state();
        let cfg = TickerConfig::default();
        state.advance_with_config(5.0, 100.0, &cfg);
        state.reset();
        assert_eq!(state.current_item_idx, 0);
        assert!((state.scroll_offset_px).abs() < f32::EPSILON);
        assert!(!state.completed);
    }

    // --- RTL (Right-to-Left) scroll direction tests ---

    #[test]
    fn test_ticker_scroll_dir_default_is_ltr() {
        assert_eq!(TickerScrollDir::default(), TickerScrollDir::Ltr);
    }

    #[test]
    fn test_ticker_config_default_is_ltr() {
        let cfg = TickerConfig::default();
        assert_eq!(cfg.scroll_dir, TickerScrollDir::Ltr);
    }

    #[test]
    fn test_ticker_config_rtl_preset() {
        let cfg = TickerConfig::rtl();
        assert_eq!(cfg.scroll_dir, TickerScrollDir::Rtl);
        // Should still have sensible defaults.
        assert!(cfg.scroll_speed_pps > 0.0);
        assert!(cfg.height_px > 0);
    }

    #[test]
    fn test_ticker_renderer_is_rtl_false_for_ltr() {
        let cfg = TickerConfig::default();
        assert!(!TickerRenderer::is_rtl(&cfg));
    }

    #[test]
    fn test_ticker_renderer_is_rtl_true_for_rtl() {
        let cfg = TickerConfig::rtl();
        assert!(TickerRenderer::is_rtl(&cfg));
    }

    #[test]
    fn test_ticker_renderer_item_x_ltr_at_zero_offset() {
        let cfg = TickerConfig::default(); // LTR
                                           // At scroll_offset 0, item starts just off the right edge.
        let x = TickerRenderer::item_x(0.0, 200.0, 1920.0, &cfg);
        assert!((x - 1920.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_ticker_renderer_item_x_ltr_partial_scroll() {
        let cfg = TickerConfig::default(); // LTR
                                           // After scrolling 300px, item is at 1920 - 300 = 1620.
        let x = TickerRenderer::item_x(300.0, 200.0, 1920.0, &cfg);
        assert!((x - 1620.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_ticker_renderer_item_x_rtl_at_zero_offset() {
        let cfg = TickerConfig::rtl(); // RTL
                                       // At scroll_offset 0, item starts fully off the left edge: 0 - 200 = -200.
        let x = TickerRenderer::item_x(0.0, 200.0, 1920.0, &cfg);
        assert!((x - (-200.0_f32)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_ticker_renderer_item_x_rtl_partial_scroll() {
        let cfg = TickerConfig::rtl(); // RTL
                                       // After scrolling 300px, item's left edge is at 300 - 200 = 100.
        let x = TickerRenderer::item_x(300.0, 200.0, 1920.0, &cfg);
        assert!((x - 100.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_ticker_rtl_render_size() {
        let state = TickerState::default();
        let config = TickerConfig {
            height_px: 48,
            scroll_dir: TickerScrollDir::Rtl,
            ..TickerConfig::default()
        };
        let data = TickerRenderer::render(&state, &config, 1920);
        assert_eq!(data.len(), (1920 * 48 * 4) as usize);
    }

    #[test]
    fn test_ticker_rtl_render_left_marker_differs_from_background() {
        // In RTL mode the left-edge marker pixels differ from the background.
        let state = TickerState::default();
        let config = TickerConfig {
            height_px: 48,
            scroll_dir: TickerScrollDir::Rtl,
            bg_color: [20, 20, 80, 230],
            text_color: [255, 255, 255, 255],
            ..TickerConfig::default()
        };
        let data = TickerRenderer::render(&state, &config, 1920);
        // The first pixel (col 0, row 0) should be tinted toward text_color.
        // It's the alpha-blended marker, so alpha > bg_alpha / 2.
        let alpha_0 = data[3];
        let alpha_mid = data[(960 * 4) + 3]; // col ~960
                                             // Left marker pixel should have a different alpha from mid-strip bg.
                                             // marker alpha = text_color[3] / 2 = 127, bg alpha = 230 → different.
        assert_ne!(alpha_0, alpha_mid);
    }

    #[test]
    fn test_advance_with_config_rtl_scrolls_same_as_ltr() {
        // Scroll mechanics are direction-agnostic; RTL and LTR advance identically.
        let mut state_ltr = make_multi_item_state();
        let mut state_rtl = make_multi_item_state();
        let cfg_ltr = TickerConfig::default();
        let cfg_rtl = TickerConfig::rtl();
        state_ltr.advance_with_config(1.0, 300.0, &cfg_ltr);
        state_rtl.advance_with_config(1.0, 300.0, &cfg_rtl);
        assert!((state_ltr.scroll_offset_px - state_rtl.scroll_offset_px).abs() < f32::EPSILON);
    }
}
