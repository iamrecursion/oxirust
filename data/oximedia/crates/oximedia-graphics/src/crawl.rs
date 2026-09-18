//! Vertical text crawl for broadcast end-credits and programme listings.
//!
//! A crawl scrolls lines of text vertically (upward, typical for end credits)
//! or downward. Each line may have independent styling metadata. The renderer
//! produces a full-frame RGBA pixel buffer.

/// Direction of the crawl scroll.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrawlDirection {
    /// Text scrolls upward (classic end-credits direction).
    Up,
    /// Text scrolls downward.
    Down,
}

impl Default for CrawlDirection {
    fn default() -> Self {
        Self::Up
    }
}

/// A single line entry in the crawl.
#[derive(Debug, Clone)]
pub struct CrawlLine {
    /// Text content.
    pub text: String,
    /// Optional role / category annotation (e.g. "DIRECTOR", "CAST").
    pub role: Option<String>,
    /// RGBA foreground color.
    pub color: [u8; 4],
    /// Relative font-size scale (1.0 = normal, 1.5 = 50% larger).
    pub font_scale: f32,
    /// Extra blank-line padding above this entry (pixels).
    pub padding_top_px: f32,
}

impl CrawlLine {
    /// Create a plain text crawl line.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            role: None,
            color: [255, 255, 255, 255],
            font_scale: 1.0,
            padding_top_px: 0.0,
        }
    }

    /// Create a role+name pair (e.g. "DIRECTOR" / "Jane Doe").
    pub fn with_role(text: impl Into<String>, role: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            role: Some(role.into()),
            color: [220, 220, 220, 255],
            font_scale: 1.0,
            padding_top_px: 8.0,
        }
    }

    /// Set font scale and return `self`.
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.font_scale = scale.max(0.1);
        self
    }

    /// Set the top padding and return `self`.
    pub fn with_padding_top(mut self, px: f32) -> Self {
        self.padding_top_px = px.max(0.0);
        self
    }

    /// Effective pixel height for this line given a base line height.
    pub fn effective_height(&self, base_line_height_px: f32) -> f32 {
        self.padding_top_px + base_line_height_px * self.font_scale
    }
}

/// Configuration for the vertical crawl renderer.
#[derive(Debug, Clone)]
pub struct CrawlConfig {
    /// Width of the output frame in pixels.
    pub frame_width: u32,
    /// Height of the output frame in pixels.
    pub frame_height: u32,
    /// Scroll speed in pixels per second.
    pub scroll_speed_pps: f32,
    /// Scroll direction.
    pub direction: CrawlDirection,
    /// Base line height in pixels (before font_scale).
    pub base_line_height_px: f32,
    /// Background color (RGBA). Use `[0, 0, 0, 0]` for transparent.
    pub bg_color: [u8; 4],
    /// Horizontal alignment fraction: 0.0 = left, 0.5 = center, 1.0 = right.
    pub h_align: f32,
    /// Whether to loop the crawl after all lines have scrolled past.
    pub looping: bool,
}

impl Default for CrawlConfig {
    fn default() -> Self {
        Self {
            frame_width: 1920,
            frame_height: 1080,
            scroll_speed_pps: 60.0,
            direction: CrawlDirection::Up,
            base_line_height_px: 40.0,
            bg_color: [0, 0, 0, 192],
            h_align: 0.5,
            looping: false,
        }
    }
}

/// State for the vertical crawl.
#[derive(Debug, Clone)]
pub struct CrawlState {
    /// Crawl lines in display order.
    pub lines: Vec<CrawlLine>,
    /// Current vertical scroll offset in pixels.
    pub scroll_offset_px: f32,
    /// Whether the crawl has completed (all lines scrolled off screen).
    pub completed: bool,
}

impl CrawlState {
    /// Create a new crawl state.
    pub fn new(lines: Vec<CrawlLine>) -> Self {
        Self {
            lines,
            scroll_offset_px: 0.0,
            completed: false,
        }
    }

    /// Total content height for all lines with a given config.
    pub fn total_content_height(&self, config: &CrawlConfig) -> f32 {
        self.lines
            .iter()
            .map(|l| l.effective_height(config.base_line_height_px))
            .sum()
    }

    /// Advance the crawl by `dt_secs` seconds.
    ///
    /// Returns `true` if all content has scrolled off screen.
    pub fn advance(&mut self, dt_secs: f32, config: &CrawlConfig) -> bool {
        if self.completed {
            return true;
        }

        let delta = config.scroll_speed_pps * dt_secs;
        self.scroll_offset_px += delta;

        let total_h = self.total_content_height(config);
        let end_offset = total_h + config.frame_height as f32;

        if self.scroll_offset_px >= end_offset {
            if config.looping {
                self.scroll_offset_px -= end_offset;
            } else {
                self.completed = true;
            }
        }

        self.completed
    }

    /// Reset the crawl to the beginning.
    pub fn reset(&mut self) {
        self.scroll_offset_px = 0.0;
        self.completed = false;
    }
}

/// Renderer for the vertical text crawl.
pub struct CrawlRenderer;

impl CrawlRenderer {
    /// Render the crawl as an RGBA pixel buffer.
    ///
    /// Returns `Vec<u8>` of length `frame_width * frame_height * 4`.
    pub fn render(state: &CrawlState, config: &CrawlConfig) -> Vec<u8> {
        let w = config.frame_width as usize;
        let h = config.frame_height as usize;
        let mut data = vec![0u8; w * h * 4];

        // Fill background.
        for chunk in data.chunks_exact_mut(4) {
            chunk[0] = config.bg_color[0];
            chunk[1] = config.bg_color[1];
            chunk[2] = config.bg_color[2];
            chunk[3] = config.bg_color[3];
        }

        if state.lines.is_empty() {
            return data;
        }

        // Draw line indicators (simplified — text rendering needs a font system).
        // Represent each visible line as a colored horizontal strip.
        let mut y_pos = match config.direction {
            CrawlDirection::Up => config.frame_height as f32 - state.scroll_offset_px,
            CrawlDirection::Down => state.scroll_offset_px - config.base_line_height_px,
        };

        for line in &state.lines {
            let line_h = line.effective_height(config.base_line_height_px);

            // Advance past padding.
            if config.direction == CrawlDirection::Up {
                y_pos += line.padding_top_px;
            }

            let top_y = y_pos as i32;
            let bot_y = (y_pos + config.base_line_height_px * line.font_scale) as i32;

            for row in top_y.max(0)..bot_y.min(h as i32) {
                // Render a thin representative strip in the line color.
                // Stripe width proportional to text estimate.
                let text_len = line.text.chars().count();
                let approx_w =
                    (text_len as f32 * config.base_line_height_px * 0.5 * line.font_scale)
                        .min(config.frame_width as f32 * 0.9) as usize;
                let x_start = ((config.frame_width as f32 * config.h_align - approx_w as f32 / 2.0)
                    .max(0.0)) as usize;
                let x_end = (x_start + approx_w).min(w);

                for col in x_start..x_end {
                    let idx = (row as usize * w + col) * 4;
                    if idx + 3 < data.len() {
                        let alpha_f = line.color[3] as f32 / 255.0;
                        let bg_a = config.bg_color[3] as f32 / 255.0;
                        let inv_a = 1.0 - alpha_f;
                        data[idx] = (line.color[0] as f32 * alpha_f
                            + config.bg_color[0] as f32 * bg_a * inv_a)
                            as u8;
                        data[idx + 1] = (line.color[1] as f32 * alpha_f
                            + config.bg_color[1] as f32 * bg_a * inv_a)
                            as u8;
                        data[idx + 2] = (line.color[2] as f32 * alpha_f
                            + config.bg_color[2] as f32 * bg_a * inv_a)
                            as u8;
                        data[idx + 3] = 255;
                    }
                }
            }

            // Advance y for next line.
            match config.direction {
                CrawlDirection::Up => {
                    y_pos += config.base_line_height_px * line.font_scale;
                }
                CrawlDirection::Down => {
                    y_pos += line_h;
                }
            }
        }

        data
    }
}

// ============================================================================
// Column layout for credits
// ============================================================================

/// A role:name pair for column-based credit layout.
#[derive(Debug, Clone)]
pub struct CreditEntry {
    /// Role/title (e.g. "Director", "Producer"). Displayed in the left column.
    pub role: String,
    /// Name(s) associated with this role. Displayed in the right column.
    pub name: String,
    /// RGBA color for the role text.
    pub role_color: [u8; 4],
    /// RGBA color for the name text.
    pub name_color: [u8; 4],
    /// Font scale relative to the base line height.
    pub font_scale: f32,
}

impl CreditEntry {
    /// Create a simple credit entry.
    pub fn new(role: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            name: name.into(),
            role_color: [180, 180, 180, 255],
            name_color: [255, 255, 255, 255],
            font_scale: 1.0,
        }
    }

    /// Builder: set a uniform color for both role and name.
    pub fn with_color(mut self, color: [u8; 4]) -> Self {
        self.role_color = color;
        self.name_color = color;
        self
    }

    /// Builder: set separate role/name colors.
    pub fn with_colors(mut self, role_color: [u8; 4], name_color: [u8; 4]) -> Self {
        self.role_color = role_color;
        self.name_color = name_color;
        self
    }

    /// Builder: set font scale.
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.font_scale = scale.max(0.1);
        self
    }
}

/// Configuration for a column-based credit crawl.
#[derive(Debug, Clone)]
pub struct ColumnCreditConfig {
    /// Frame width in pixels.
    pub frame_width: u32,
    /// Frame height in pixels.
    pub frame_height: u32,
    /// Scroll speed in pixels per second.
    pub scroll_speed_pps: f32,
    /// Base line height in pixels.
    pub base_line_height_px: f32,
    /// Fraction of frame width used for the role column [0.0, 1.0].
    pub role_column_fraction: f32,
    /// Gap between role and name columns in pixels.
    pub column_gap_px: f32,
    /// Extra vertical spacing between entries in pixels.
    pub entry_spacing_px: f32,
    /// Background color (RGBA).
    pub bg_color: [u8; 4],
    /// Whether to loop.
    pub looping: bool,
}

impl Default for ColumnCreditConfig {
    fn default() -> Self {
        Self {
            frame_width: 1920,
            frame_height: 1080,
            scroll_speed_pps: 50.0,
            base_line_height_px: 36.0,
            role_column_fraction: 0.4,
            column_gap_px: 20.0,
            entry_spacing_px: 4.0,
            bg_color: [0, 0, 0, 200],
            looping: false,
        }
    }
}

/// State for a column-based credit crawl.
#[derive(Debug, Clone)]
pub struct ColumnCreditState {
    /// Credit entries.
    pub entries: Vec<CreditEntry>,
    /// Current scroll offset in pixels.
    pub scroll_offset_px: f32,
    /// Whether the crawl has completed.
    pub completed: bool,
}

impl ColumnCreditState {
    /// Create a new column credit state.
    pub fn new(entries: Vec<CreditEntry>) -> Self {
        Self {
            entries,
            scroll_offset_px: 0.0,
            completed: false,
        }
    }

    /// Total content height.
    pub fn total_content_height(&self, config: &ColumnCreditConfig) -> f32 {
        self.entries
            .iter()
            .map(|e| config.base_line_height_px * e.font_scale + config.entry_spacing_px)
            .sum()
    }

    /// Advance the crawl.
    pub fn advance(&mut self, dt_secs: f32, config: &ColumnCreditConfig) -> bool {
        if self.completed {
            return true;
        }
        self.scroll_offset_px += config.scroll_speed_pps * dt_secs;
        let end = self.total_content_height(config) + config.frame_height as f32;
        if self.scroll_offset_px >= end {
            if config.looping {
                self.scroll_offset_px -= end;
            } else {
                self.completed = true;
            }
        }
        self.completed
    }

    /// Reset the crawl.
    pub fn reset(&mut self) {
        self.scroll_offset_px = 0.0;
        self.completed = false;
    }
}

// ============================================================================
// Pause-on-highlight
// ============================================================================

/// Configuration for pause-on-highlight behavior.
///
/// When certain entries are "highlighted", the crawl speed is reduced or the
/// crawl pauses entirely for a configured duration.
#[derive(Debug, Clone)]
pub struct HighlightPauseConfig {
    /// Indices of highlighted entries (0-based into the crawl's entry list).
    pub highlighted_indices: Vec<usize>,
    /// Speed multiplier when a highlighted entry is visible (0.0 = full pause,
    /// 0.5 = half speed, 1.0 = normal speed).
    pub speed_multiplier: f32,
    /// Duration in seconds to hold the pause/slow-down once triggered.
    pub hold_duration_secs: f32,
    /// Vertical region (fraction of frame height from center) within which a
    /// highlighted entry triggers the slow-down. E.g. 0.3 means the middle 30%.
    pub trigger_zone_fraction: f32,
}

impl Default for HighlightPauseConfig {
    fn default() -> Self {
        Self {
            highlighted_indices: Vec::new(),
            speed_multiplier: 0.0,
            hold_duration_secs: 2.0,
            trigger_zone_fraction: 0.3,
        }
    }
}

impl HighlightPauseConfig {
    /// Create a pause config that fully pauses on the given indices.
    pub fn pause_on(indices: Vec<usize>) -> Self {
        Self {
            highlighted_indices: indices,
            speed_multiplier: 0.0,
            hold_duration_secs: 2.0,
            trigger_zone_fraction: 0.3,
        }
    }

    /// Builder: set speed multiplier.
    pub fn with_speed(mut self, multiplier: f32) -> Self {
        self.speed_multiplier = multiplier.clamp(0.0, 1.0);
        self
    }

    /// Builder: set hold duration.
    pub fn with_hold_duration(mut self, secs: f32) -> Self {
        self.hold_duration_secs = secs.max(0.0);
        self
    }

    /// Returns `true` if the given entry index is highlighted.
    pub fn is_highlighted(&self, index: usize) -> bool {
        self.highlighted_indices.contains(&index)
    }
}

/// State tracker for the pause-on-highlight feature.
#[derive(Debug, Clone)]
pub struct HighlightPauseState {
    /// Remaining hold time in seconds (> 0 means we are paused/slowed).
    pub remaining_hold_secs: f32,
    /// The index of the entry currently triggering the pause, if any.
    pub active_highlight_index: Option<usize>,
}

impl Default for HighlightPauseState {
    fn default() -> Self {
        Self {
            remaining_hold_secs: 0.0,
            active_highlight_index: None,
        }
    }
}

impl HighlightPauseState {
    /// Compute the effective speed multiplier for this frame.
    ///
    /// `visible_entry_indices` lists all entry indices currently visible
    /// on screen.
    pub fn update(
        &mut self,
        dt_secs: f32,
        visible_entry_indices: &[usize],
        config: &HighlightPauseConfig,
    ) -> f32 {
        // Decrement hold timer
        if self.remaining_hold_secs > 0.0 {
            self.remaining_hold_secs = (self.remaining_hold_secs - dt_secs).max(0.0);
            if self.remaining_hold_secs > 0.0 {
                return config.speed_multiplier;
            }
            self.active_highlight_index = None;
        }

        // Check if any highlighted entry is in the visible set
        for &idx in visible_entry_indices {
            if config.is_highlighted(idx) {
                // Only trigger if it's a new highlight (not the one we just finished)
                if self.active_highlight_index != Some(idx) {
                    self.active_highlight_index = Some(idx);
                    self.remaining_hold_secs = config.hold_duration_secs;
                    return config.speed_multiplier;
                }
            }
        }

        1.0 // normal speed
    }

    /// Returns `true` if currently paused or slowed.
    pub fn is_active(&self) -> bool {
        self.remaining_hold_secs > 0.0
    }
}

// ============================================================================
// Logo insertion in credit rolls
// ============================================================================

/// A logo to be inserted into the credit roll at a specific position.
#[derive(Debug, Clone)]
pub struct CrawlLogo {
    /// RGBA pixel data of the logo.
    pub pixels: Vec<u8>,
    /// Width of the logo in pixels.
    pub width: u32,
    /// Height of the logo in pixels.
    pub height: u32,
    /// Horizontal alignment fraction (0.0 = left, 0.5 = center, 1.0 = right).
    pub h_align: f32,
    /// Extra padding above the logo in pixels.
    pub padding_top_px: f32,
    /// Extra padding below the logo in pixels.
    pub padding_bottom_px: f32,
    /// Opacity in [0.0, 1.0].
    pub opacity: f32,
}

impl CrawlLogo {
    /// Create a logo entry.
    ///
    /// `pixels` must have length `width * height * 4` (RGBA).
    pub fn new(pixels: Vec<u8>, width: u32, height: u32) -> Self {
        Self {
            pixels,
            width,
            height,
            h_align: 0.5,
            padding_top_px: 20.0,
            padding_bottom_px: 20.0,
            opacity: 1.0,
        }
    }

    /// Builder: set horizontal alignment.
    pub fn with_align(mut self, align: f32) -> Self {
        self.h_align = align.clamp(0.0, 1.0);
        self
    }

    /// Builder: set padding.
    pub fn with_padding(mut self, top: f32, bottom: f32) -> Self {
        self.padding_top_px = top.max(0.0);
        self.padding_bottom_px = bottom.max(0.0);
        self
    }

    /// Builder: set opacity.
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity.clamp(0.0, 1.0);
        self
    }

    /// Total vertical space consumed by this logo.
    pub fn total_height(&self) -> f32 {
        self.padding_top_px + self.height as f32 + self.padding_bottom_px
    }

    /// Returns `true` if the pixel data is the correct size.
    pub fn is_valid(&self) -> bool {
        self.pixels.len() == (self.width as usize * self.height as usize * 4)
    }

    /// Composite the logo onto an RGBA buffer at the given position.
    ///
    /// `dest` is the destination buffer, `dest_width` is the buffer's row stride
    /// in pixels. `x_offset` and `y_offset` are the top-left corner in dest.
    pub fn composite(&self, dest: &mut [u8], dest_width: usize, x_offset: usize, y_offset: usize) {
        if !self.is_valid() {
            return;
        }
        let lw = self.width as usize;
        let lh = self.height as usize;
        for row in 0..lh {
            let dy = y_offset + row;
            for col in 0..lw {
                let dx = x_offset + col;
                let src_idx = (row * lw + col) * 4;
                let dst_idx = (dy * dest_width + dx) * 4;
                if src_idx + 3 >= self.pixels.len() || dst_idx + 3 >= dest.len() {
                    continue;
                }
                let src_a = self.pixels[src_idx + 3] as f32 / 255.0 * self.opacity;
                let inv_a = 1.0 - src_a;
                dest[dst_idx] =
                    (self.pixels[src_idx] as f32 * src_a + dest[dst_idx] as f32 * inv_a) as u8;
                dest[dst_idx + 1] = (self.pixels[src_idx + 1] as f32 * src_a
                    + dest[dst_idx + 1] as f32 * inv_a) as u8;
                dest[dst_idx + 2] = (self.pixels[src_idx + 2] as f32 * src_a
                    + dest[dst_idx + 2] as f32 * inv_a) as u8;
                dest[dst_idx + 3] =
                    ((src_a + dest[dst_idx + 3] as f32 / 255.0 * inv_a) * 255.0).min(255.0) as u8;
            }
        }
    }
}

/// An item in an enhanced credit roll: either a text line, a credit entry pair,
/// or an inline logo.
#[derive(Debug, Clone)]
pub enum CrawlItem {
    /// A simple text line.
    Text(CrawlLine),
    /// A role:name column entry.
    Credit(CreditEntry),
    /// An inline logo image.
    Logo(CrawlLogo),
}

impl CrawlItem {
    /// Effective height of this item given a base line height.
    pub fn effective_height(&self, base_line_height_px: f32) -> f32 {
        match self {
            Self::Text(line) => line.effective_height(base_line_height_px),
            Self::Credit(entry) => base_line_height_px * entry.font_scale,
            Self::Logo(logo) => logo.total_height(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state() -> CrawlState {
        CrawlState::new(vec![
            CrawlLine::new("Directed by"),
            CrawlLine::with_role("Jane Doe", "DIRECTOR"),
            CrawlLine::new("Music by"),
            CrawlLine::with_role("John Smith", "COMPOSER"),
        ])
    }

    #[test]
    fn test_crawl_direction_default() {
        assert_eq!(CrawlDirection::default(), CrawlDirection::Up);
    }

    #[test]
    fn test_crawl_line_new() {
        let l = CrawlLine::new("Test line");
        assert_eq!(l.text, "Test line");
        assert!(l.role.is_none());
        assert!((l.font_scale - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_crawl_line_with_role() {
        let l = CrawlLine::with_role("Jane Doe", "DIRECTOR");
        assert_eq!(l.text, "Jane Doe");
        assert_eq!(l.role.as_deref(), Some("DIRECTOR"));
        assert!(l.padding_top_px > 0.0);
    }

    #[test]
    fn test_crawl_line_with_scale() {
        let l = CrawlLine::new("Title").with_scale(2.0);
        assert!((l.font_scale - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_crawl_line_with_padding_top() {
        let l = CrawlLine::new("Line").with_padding_top(20.0);
        assert!((l.padding_top_px - 20.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_crawl_line_effective_height_no_padding() {
        let l = CrawlLine::new("Line");
        assert!((l.effective_height(40.0) - 40.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_crawl_line_effective_height_with_scale() {
        let l = CrawlLine::new("Big").with_scale(2.0);
        assert!((l.effective_height(40.0) - 80.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_crawl_config_default() {
        let cfg = CrawlConfig::default();
        assert!(cfg.scroll_speed_pps > 0.0);
        assert_eq!(cfg.direction, CrawlDirection::Up);
        assert!(!cfg.looping);
    }

    #[test]
    fn test_crawl_state_new() {
        let state = make_state();
        assert_eq!(state.lines.len(), 4);
        assert!(!state.completed);
        assert!((state.scroll_offset_px).abs() < f32::EPSILON);
    }

    #[test]
    fn test_crawl_state_total_content_height() {
        let state = CrawlState::new(vec![CrawlLine::new("A"), CrawlLine::new("B")]);
        let cfg = CrawlConfig::default();
        let h = state.total_content_height(&cfg);
        assert!((h - 80.0).abs() < f32::EPSILON); // 2 × 40px
    }

    #[test]
    fn test_crawl_state_advance_moves_offset() {
        let mut state = make_state();
        let cfg = CrawlConfig::default();
        state.advance(1.0, &cfg);
        assert!(state.scroll_offset_px > 0.0);
    }

    #[test]
    fn test_crawl_state_advance_completes() {
        let mut state = CrawlState::new(vec![CrawlLine::new("Short")]);
        let cfg = CrawlConfig::default();
        // Advance well past the total height.
        state.advance(10000.0, &cfg);
        assert!(state.completed);
    }

    #[test]
    fn test_crawl_state_looping_does_not_complete() {
        let mut state = CrawlState::new(vec![CrawlLine::new("Line")]);
        let cfg = CrawlConfig {
            looping: true,
            ..CrawlConfig::default()
        };
        state.advance(10000.0, &cfg);
        assert!(!state.completed);
    }

    #[test]
    fn test_crawl_state_reset() {
        let mut state = make_state();
        let cfg = CrawlConfig::default();
        state.advance(500.0, &cfg);
        state.reset();
        assert!(!state.completed);
        assert!((state.scroll_offset_px).abs() < f32::EPSILON);
    }

    #[test]
    fn test_crawl_renderer_output_size() {
        let state = make_state();
        let cfg = CrawlConfig {
            frame_width: 320,
            frame_height: 240,
            ..CrawlConfig::default()
        };
        let data = CrawlRenderer::render(&state, &cfg);
        assert_eq!(data.len(), 320 * 240 * 4);
    }

    #[test]
    fn test_crawl_renderer_non_empty_output() {
        let state = make_state();
        let cfg = CrawlConfig {
            frame_width: 320,
            frame_height: 240,
            ..CrawlConfig::default()
        };
        let data = CrawlRenderer::render(&state, &cfg);
        assert!(data.iter().any(|&b| b > 0));
    }

    #[test]
    fn test_crawl_renderer_empty_lines_fills_bg() {
        let state = CrawlState::new(vec![]);
        let cfg = CrawlConfig {
            frame_width: 64,
            frame_height: 64,
            bg_color: [100, 0, 0, 255],
            ..CrawlConfig::default()
        };
        let data = CrawlRenderer::render(&state, &cfg);
        assert_eq!(data[0], 100); // bg red channel
    }

    // --- CreditEntry tests ---

    #[test]
    fn test_credit_entry_new() {
        let e = CreditEntry::new("Director", "Jane Doe");
        assert_eq!(e.role, "Director");
        assert_eq!(e.name, "Jane Doe");
        assert!((e.font_scale - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_credit_entry_with_color() {
        let e = CreditEntry::new("Role", "Name").with_color([255, 0, 0, 255]);
        assert_eq!(e.role_color, [255, 0, 0, 255]);
        assert_eq!(e.name_color, [255, 0, 0, 255]);
    }

    #[test]
    fn test_credit_entry_with_colors() {
        let e = CreditEntry::new("Role", "Name")
            .with_colors([100, 100, 100, 255], [200, 200, 200, 255]);
        assert_eq!(e.role_color, [100, 100, 100, 255]);
        assert_eq!(e.name_color, [200, 200, 200, 255]);
    }

    #[test]
    fn test_credit_entry_with_scale() {
        let e = CreditEntry::new("Role", "Name").with_scale(1.5);
        assert!((e.font_scale - 1.5).abs() < f32::EPSILON);
    }

    // --- ColumnCreditConfig tests ---

    #[test]
    fn test_column_credit_config_default() {
        let cfg = ColumnCreditConfig::default();
        assert!(cfg.scroll_speed_pps > 0.0);
        assert!(cfg.role_column_fraction > 0.0 && cfg.role_column_fraction < 1.0);
        assert!(!cfg.looping);
    }

    // --- ColumnCreditState tests ---

    #[test]
    fn test_column_credit_state_new() {
        let entries = vec![
            CreditEntry::new("Director", "Jane Doe"),
            CreditEntry::new("Producer", "John Smith"),
        ];
        let state = ColumnCreditState::new(entries);
        assert_eq!(state.entries.len(), 2);
        assert!(!state.completed);
    }

    #[test]
    fn test_column_credit_state_total_height() {
        let entries = vec![CreditEntry::new("A", "B"), CreditEntry::new("C", "D")];
        let state = ColumnCreditState::new(entries);
        let cfg = ColumnCreditConfig::default();
        let h = state.total_content_height(&cfg);
        assert!(h > 0.0);
    }

    #[test]
    fn test_column_credit_state_advance_completes() {
        let entries = vec![CreditEntry::new("Role", "Name")];
        let mut state = ColumnCreditState::new(entries);
        let cfg = ColumnCreditConfig::default();
        state.advance(10000.0, &cfg);
        assert!(state.completed);
    }

    #[test]
    fn test_column_credit_state_reset() {
        let entries = vec![CreditEntry::new("Role", "Name")];
        let mut state = ColumnCreditState::new(entries);
        let cfg = ColumnCreditConfig::default();
        state.advance(10000.0, &cfg);
        state.reset();
        assert!(!state.completed);
        assert!((state.scroll_offset_px).abs() < f32::EPSILON);
    }

    // --- HighlightPauseConfig tests ---

    #[test]
    fn test_highlight_pause_config_default() {
        let cfg = HighlightPauseConfig::default();
        assert!(cfg.highlighted_indices.is_empty());
        assert!((cfg.speed_multiplier).abs() < f32::EPSILON);
    }

    #[test]
    fn test_highlight_pause_config_pause_on() {
        let cfg = HighlightPauseConfig::pause_on(vec![2, 5]);
        assert!(cfg.is_highlighted(2));
        assert!(cfg.is_highlighted(5));
        assert!(!cfg.is_highlighted(0));
    }

    #[test]
    fn test_highlight_pause_config_with_speed() {
        let cfg = HighlightPauseConfig::pause_on(vec![0]).with_speed(0.5);
        assert!((cfg.speed_multiplier - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_highlight_pause_config_with_hold_duration() {
        let cfg = HighlightPauseConfig::pause_on(vec![0]).with_hold_duration(5.0);
        assert!((cfg.hold_duration_secs - 5.0).abs() < f32::EPSILON);
    }

    // --- HighlightPauseState tests ---

    #[test]
    fn test_highlight_pause_state_default_not_active() {
        let state = HighlightPauseState::default();
        assert!(!state.is_active());
    }

    #[test]
    fn test_highlight_pause_state_triggers_on_highlighted() {
        let mut state = HighlightPauseState::default();
        let cfg = HighlightPauseConfig::pause_on(vec![3]);
        let mult = state.update(0.016, &[1, 3, 5], &cfg);
        assert!(
            mult < 1.0,
            "Should slow down when highlighted entry is visible"
        );
        assert!(state.is_active());
    }

    #[test]
    fn test_highlight_pause_state_normal_speed_without_highlight() {
        let mut state = HighlightPauseState::default();
        let cfg = HighlightPauseConfig::pause_on(vec![10]);
        let mult = state.update(0.016, &[0, 1, 2], &cfg);
        assert!((mult - 1.0).abs() < f32::EPSILON);
        assert!(!state.is_active());
    }

    // --- CrawlLogo tests ---

    #[test]
    fn test_crawl_logo_valid() {
        let w = 4;
        let h = 2;
        let pixels = vec![128u8; (w * h * 4) as usize];
        let logo = CrawlLogo::new(pixels, w, h);
        assert!(logo.is_valid());
        assert!(logo.total_height() > 0.0);
    }

    #[test]
    fn test_crawl_logo_invalid_size() {
        let logo = CrawlLogo::new(vec![0u8; 10], 4, 4);
        assert!(!logo.is_valid());
    }

    #[test]
    fn test_crawl_logo_with_align() {
        let logo = CrawlLogo::new(vec![], 0, 0).with_align(0.0);
        assert!((logo.h_align).abs() < f32::EPSILON);
    }

    #[test]
    fn test_crawl_logo_with_padding() {
        let logo = CrawlLogo::new(vec![], 0, 0).with_padding(10.0, 15.0);
        assert!((logo.padding_top_px - 10.0).abs() < f32::EPSILON);
        assert!((logo.padding_bottom_px - 15.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_crawl_logo_composite() {
        let w = 2u32;
        let h = 2u32;
        let logo_pixels = vec![255u8; (w * h * 4) as usize];
        let logo = CrawlLogo::new(logo_pixels, w, h);

        let dest_w = 4usize;
        let dest_h = 4usize;
        let mut dest = vec![0u8; dest_w * dest_h * 4];
        logo.composite(&mut dest, dest_w, 1, 1);

        // Pixel at (1,1) should have been blended
        let idx = (1 * dest_w + 1) * 4;
        assert!(dest[idx] > 0, "Logo should have composited onto dest");
    }

    // --- CrawlItem tests ---

    #[test]
    fn test_crawl_item_text_height() {
        let item = CrawlItem::Text(CrawlLine::new("Hello"));
        assert!((item.effective_height(40.0) - 40.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_crawl_item_credit_height() {
        let entry = CreditEntry::new("Role", "Name").with_scale(2.0);
        let item = CrawlItem::Credit(entry);
        assert!((item.effective_height(40.0) - 80.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_crawl_item_logo_height() {
        let logo = CrawlLogo::new(vec![0u8; 16], 2, 2).with_padding(10.0, 10.0);
        let item = CrawlItem::Logo(logo);
        // 10 + 2 + 10 = 22
        assert!((item.effective_height(40.0) - 22.0).abs() < f32::EPSILON);
    }
}
