//! Lower-third graphics template system for broadcast graphics.
//!
//! Provides a complete lower-third rendering pipeline with configurable styles,
//! animation phases, and timeline management.

/// Style variants for lower-third graphics.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub enum LowerThirdStyle {
    /// Classic broadcast style with solid background bar.
    Classic,
    /// Modern style with gradient and accent line.
    Modern,
    /// Minimal style with transparent background.
    Minimal,
    /// News ticker style with bold text.
    News,
    /// Sports style with high-contrast colors.
    Sports,
    /// Corporate style with subdued colors.
    Corporate,
}

impl LowerThirdStyle {
    /// Returns a human-readable description of the style.
    #[allow(dead_code)]
    pub fn description(&self) -> &str {
        match self {
            Self::Classic => "Classic broadcast lower-third",
            Self::Modern => "Modern gradient lower-third",
            Self::Minimal => "Minimal transparent lower-third",
            Self::News => "News ticker style",
            Self::Sports => "Sports high-contrast style",
            Self::Corporate => "Corporate subdued style",
        }
    }
}

/// Configuration for a lower-third graphic element.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct LowerThirdConfig {
    /// Primary name displayed prominently.
    pub name: String,
    /// Title or role beneath the name.
    pub title: String,
    /// Optional subtitle for additional context.
    pub subtitle: Option<String>,
    /// Visual style variant.
    pub style: LowerThirdStyle,
    /// Accent color as RGBA.
    pub accent_color: [u8; 4],
    /// Text color as RGBA.
    pub text_color: [u8; 4],
    /// Background color as RGBA.
    pub background_color: [u8; 4],
    /// Vertical position as fraction of frame height (0.0 = top, 1.0 = bottom).
    pub position_y_pct: f32,
}

impl Default for LowerThirdConfig {
    fn default() -> Self {
        Self {
            name: "John Smith".to_string(),
            title: "Reporter".to_string(),
            subtitle: None,
            style: LowerThirdStyle::Classic,
            accent_color: [255, 165, 0, 255],
            text_color: [255, 255, 255, 255],
            background_color: [0, 0, 0, 200],
            position_y_pct: 0.8,
        }
    }
}

/// Phase of the lower-third animation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum AnimationPhase {
    /// Slide-in animation at the start.
    In,
    /// Static hold during main display.
    Hold,
    /// Slide-out animation at the end.
    Out,
}

impl AnimationPhase {
    /// Returns the duration in frames for this phase at the given FPS.
    ///
    /// - In: 15 frames
    /// - Hold: 150 frames
    /// - Out: 15 frames
    #[allow(dead_code)]
    pub fn duration_frames(&self, _fps: f32) -> u32 {
        match self {
            Self::In => 15,
            Self::Hold => 150,
            Self::Out => 15,
        }
    }

    /// Determine the animation phase for a given frame within total frames.
    #[allow(dead_code)]
    pub fn for_frame(frame: u32, total_frames: u32, fps: f32) -> Self {
        let in_frames = AnimationPhase::In.duration_frames(fps);
        let out_frames = AnimationPhase::Out.duration_frames(fps);

        if frame < in_frames {
            Self::In
        } else if frame >= total_frames.saturating_sub(out_frames) {
            Self::Out
        } else {
            Self::Hold
        }
    }
}

/// Renderer for lower-third graphics.
pub struct LowerThirdRenderer;

impl LowerThirdRenderer {
    /// Render a lower-third frame as RGBA pixel data.
    ///
    /// # Arguments
    ///
    /// * `config` - Lower-third configuration.
    /// * `frame` - Current frame number.
    /// * `total_frames` - Total frames for the lower-third display.
    /// * `width` - Output image width in pixels.
    /// * `height` - Output image height in pixels.
    ///
    /// Returns a `Vec<u8>` of RGBA pixels with length `width * height * 4`.
    #[allow(dead_code)]
    pub fn render(
        config: &LowerThirdConfig,
        frame: u32,
        total_frames: u32,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let mut data = vec![0u8; (width * height * 4) as usize];

        let fps = 30.0_f32;
        let phase = AnimationPhase::for_frame(frame, total_frames, fps);

        // Calculate slide-in offset (slide from left)
        let slide_offset = match phase {
            AnimationPhase::In => {
                let in_dur = AnimationPhase::In.duration_frames(fps);
                let progress = if in_dur > 0 {
                    frame as f32 / in_dur as f32
                } else {
                    1.0
                };
                let eased = progress * progress; // quad ease-in
                ((1.0 - eased) * -(width as f32)) as i32
            }
            AnimationPhase::Hold => 0,
            AnimationPhase::Out => {
                let out_dur = AnimationPhase::Out.duration_frames(fps);
                let out_start = total_frames.saturating_sub(out_dur);
                let progress = if out_dur > 0 {
                    (frame - out_start) as f32 / out_dur as f32
                } else {
                    1.0
                };
                let eased = progress * progress;
                (eased * -(width as f32)) as i32
            }
        };

        // Bar dimensions
        let bar_y = (config.position_y_pct * height as f32) as i32;
        let bar_height = (height as f32 * 0.12) as u32;
        let bar_height = bar_height.max(40);

        let accent_bar_height = (bar_height as f32 * 0.1) as u32;
        let accent_bar_height = accent_bar_height.max(4);

        // Draw background bar
        for row in 0..bar_height {
            let y = bar_y + row as i32;
            if y < 0 || y >= height as i32 {
                continue;
            }
            let y = y as u32;

            for x in 0..width {
                let sx = x as i32 - slide_offset;
                if sx < 0 {
                    continue;
                }

                // Determine if this pixel is in the accent stripe
                let is_accent = row < accent_bar_height;
                let color = if is_accent {
                    config.accent_color
                } else {
                    config.background_color
                };

                let idx = ((y * width + x) * 4) as usize;
                if idx + 3 < data.len() {
                    data[idx] = color[0];
                    data[idx + 1] = color[1];
                    data[idx + 2] = color[2];
                    data[idx + 3] = color[3];
                }
            }
        }

        data
    }
}

/// A single lower-third segment in a timeline.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct LowerThirdSegment {
    /// Start frame of this segment.
    pub start_frame: u64,
    /// End frame of this segment.
    pub end_frame: u64,
    /// Lower-third configuration for this segment.
    pub config: LowerThirdConfig,
}

impl LowerThirdSegment {
    /// Create a new lower-third segment.
    #[allow(dead_code)]
    pub fn new(start_frame: u64, end_frame: u64, config: LowerThirdConfig) -> Self {
        Self {
            start_frame,
            end_frame,
            config,
        }
    }

    /// Duration in frames.
    #[allow(dead_code)]
    pub fn duration_frames(&self) -> u64 {
        self.end_frame.saturating_sub(self.start_frame)
    }
}

/// Timeline of lower-third segments.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct LowerThirdTimeline {
    /// Ordered list of lower-third segments.
    pub segments: Vec<LowerThirdSegment>,
}

impl LowerThirdTimeline {
    /// Create an empty timeline.
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
        }
    }

    /// Add a segment to the timeline.
    #[allow(dead_code)]
    pub fn add(&mut self, segment: LowerThirdSegment) {
        self.segments.push(segment);
        self.segments.sort_by_key(|s| s.start_frame);
    }

    /// Find the active segment at the given frame, if any.
    #[allow(dead_code)]
    pub fn at_frame(&self, frame: u64) -> Option<&LowerThirdSegment> {
        self.segments
            .iter()
            .find(|s| frame >= s.start_frame && frame < s.end_frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lower_third_style_description() {
        assert!(!LowerThirdStyle::Classic.description().is_empty());
        assert!(!LowerThirdStyle::Modern.description().is_empty());
        assert!(!LowerThirdStyle::Minimal.description().is_empty());
        assert!(!LowerThirdStyle::News.description().is_empty());
        assert!(!LowerThirdStyle::Sports.description().is_empty());
        assert!(!LowerThirdStyle::Corporate.description().is_empty());
    }

    #[test]
    fn test_animation_phase_duration() {
        assert_eq!(AnimationPhase::In.duration_frames(30.0), 15);
        assert_eq!(AnimationPhase::Hold.duration_frames(30.0), 150);
        assert_eq!(AnimationPhase::Out.duration_frames(30.0), 15);
    }

    #[test]
    fn test_animation_phase_for_frame_in() {
        let phase = AnimationPhase::for_frame(0, 180, 30.0);
        assert_eq!(phase, AnimationPhase::In);
    }

    #[test]
    fn test_animation_phase_for_frame_hold() {
        let phase = AnimationPhase::for_frame(80, 180, 30.0);
        assert_eq!(phase, AnimationPhase::Hold);
    }

    #[test]
    fn test_animation_phase_for_frame_out() {
        let phase = AnimationPhase::for_frame(170, 180, 30.0);
        assert_eq!(phase, AnimationPhase::Out);
    }

    #[test]
    fn test_lower_third_config_default() {
        let cfg = LowerThirdConfig::default();
        assert!(!cfg.name.is_empty());
        assert!(!cfg.title.is_empty());
        assert!(cfg.subtitle.is_none());
        assert!(cfg.position_y_pct >= 0.0 && cfg.position_y_pct <= 1.0);
    }

    #[test]
    fn test_render_returns_correct_size() {
        let cfg = LowerThirdConfig::default();
        let data = LowerThirdRenderer::render(&cfg, 0, 180, 320, 240);
        assert_eq!(data.len(), 320 * 240 * 4);
    }

    #[test]
    fn test_render_hold_frame_draws_background() {
        let cfg = LowerThirdConfig {
            background_color: [255, 0, 0, 255],
            position_y_pct: 0.5,
            ..LowerThirdConfig::default()
        };
        // Hold phase frame (after In frames)
        let data = LowerThirdRenderer::render(&cfg, 50, 180, 320, 240);
        // Check that some pixels have been drawn (non-zero)
        let has_color = data.chunks(4).any(|p| p[3] > 0);
        assert!(has_color, "Render should produce visible pixels");
    }

    #[test]
    fn test_lower_third_segment_duration() {
        let cfg = LowerThirdConfig::default();
        let seg = LowerThirdSegment::new(0, 180, cfg);
        assert_eq!(seg.duration_frames(), 180);
    }

    #[test]
    fn test_timeline_at_frame_hit() {
        let mut tl = LowerThirdTimeline::new();
        let cfg = LowerThirdConfig::default();
        tl.add(LowerThirdSegment::new(100, 200, cfg));

        assert!(tl.at_frame(100).is_some());
        assert!(tl.at_frame(150).is_some());
        assert!(tl.at_frame(199).is_some());
    }

    #[test]
    fn test_timeline_at_frame_miss() {
        let mut tl = LowerThirdTimeline::new();
        let cfg = LowerThirdConfig::default();
        tl.add(LowerThirdSegment::new(100, 200, cfg));

        assert!(tl.at_frame(99).is_none());
        assert!(tl.at_frame(200).is_none());
    }

    #[test]
    fn test_timeline_multiple_segments() {
        let mut tl = LowerThirdTimeline::new();
        tl.add(LowerThirdSegment::new(0, 100, LowerThirdConfig::default()));
        tl.add(LowerThirdSegment::new(
            200,
            300,
            LowerThirdConfig::default(),
        ));

        assert!(tl.at_frame(50).is_some());
        assert!(tl.at_frame(150).is_none());
        assert!(tl.at_frame(250).is_some());
    }

    #[test]
    fn test_timeline_sorted_on_add() {
        let mut tl = LowerThirdTimeline::new();
        // Add out of order
        tl.add(LowerThirdSegment::new(
            200,
            300,
            LowerThirdConfig::default(),
        ));
        tl.add(LowerThirdSegment::new(0, 100, LowerThirdConfig::default()));

        assert_eq!(tl.segments[0].start_frame, 0);
        assert_eq!(tl.segments[1].start_frame, 200);
    }
}
