//! Logo bug — persistent corner logo placement with fade in/out animation.
//!
//! A "bug" (station identifier or watermark) is a small, semi-transparent logo
//! rendered in a corner of the broadcast frame. This module handles:
//! - Corner positioning with configurable padding.
//! - Fade-in and fade-out animations.
//! - Steady-state opacity (typically 50–75% for broadcast compliance).
//! - Optional pulse animation that briefly increases opacity to attract attention.

/// Corner placement for the logo bug.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BugCorner {
    /// Top-left corner.
    TopLeft,
    /// Top-right corner.
    TopRight,
    /// Bottom-left corner.
    BottomLeft,
    /// Bottom-right corner (most common for network branding).
    BottomRight,
}

impl Default for BugCorner {
    fn default() -> Self {
        Self::BottomRight
    }
}

/// Animation state for the logo bug.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BugAnimState {
    /// Bug is fading in.
    FadeIn,
    /// Bug is at steady-state opacity.
    Steady,
    /// Bug is fading out.
    FadeOut,
    /// Bug is fully hidden (opacity = 0).
    Hidden,
    /// Bug is pulsing (brief opacity increase).
    Pulse,
}

impl Default for BugAnimState {
    fn default() -> Self {
        Self::FadeIn
    }
}

/// Configuration for a logo bug.
#[derive(Debug, Clone)]
pub struct LogoBugConfig {
    /// Frame width in pixels.
    pub frame_width: u32,
    /// Frame height in pixels.
    pub frame_height: u32,
    /// Logo width in pixels.
    pub logo_width: u32,
    /// Logo height in pixels.
    pub logo_height: u32,
    /// Corner to place the bug.
    pub corner: BugCorner,
    /// Padding from the frame edge in pixels.
    pub padding_px: u32,
    /// Steady-state opacity in [0.0, 1.0].
    pub steady_opacity: f32,
    /// Duration of fade-in in seconds.
    pub fade_in_secs: f32,
    /// Duration of fade-out in seconds.
    pub fade_out_secs: f32,
    /// Duration of the pulse hold in seconds.
    pub pulse_hold_secs: f32,
    /// Peak opacity during pulse animation.
    pub pulse_peak_opacity: f32,
    /// RGBA fill color used as a placeholder when no logo image is supplied.
    pub placeholder_color: [u8; 4],
}

impl Default for LogoBugConfig {
    fn default() -> Self {
        Self {
            frame_width: 1920,
            frame_height: 1080,
            logo_width: 120,
            logo_height: 60,
            corner: BugCorner::BottomRight,
            padding_px: 20,
            steady_opacity: 0.6,
            fade_in_secs: 0.5,
            fade_out_secs: 0.5,
            pulse_hold_secs: 0.25,
            pulse_peak_opacity: 1.0,
            placeholder_color: [255, 255, 255, 200],
        }
    }
}

impl LogoBugConfig {
    /// Compute the pixel position of the bug's top-left corner.
    pub fn bug_origin(&self) -> (u32, u32) {
        let pad = self.padding_px;
        match self.corner {
            BugCorner::TopLeft => (pad, pad),
            BugCorner::TopRight => (self.frame_width.saturating_sub(self.logo_width + pad), pad),
            BugCorner::BottomLeft => (
                pad,
                self.frame_height.saturating_sub(self.logo_height + pad),
            ),
            BugCorner::BottomRight => (
                self.frame_width.saturating_sub(self.logo_width + pad),
                self.frame_height.saturating_sub(self.logo_height + pad),
            ),
        }
    }
}

/// Runtime state for a logo bug.
#[derive(Debug, Clone)]
pub struct LogoBugState {
    /// Current animation state.
    pub anim_state: BugAnimState,
    /// Elapsed time within the current animation phase (seconds).
    pub phase_elapsed_secs: f32,
    /// Current computed opacity in [0.0, 1.0].
    pub current_opacity: f32,
}

impl Default for LogoBugState {
    fn default() -> Self {
        Self {
            anim_state: BugAnimState::FadeIn,
            phase_elapsed_secs: 0.0,
            current_opacity: 0.0,
        }
    }
}

impl LogoBugState {
    /// Create a new bug state starting in fade-in.
    pub fn new() -> Self {
        Self::default()
    }

    /// Trigger the fade-out animation.
    pub fn fade_out(&mut self) {
        self.anim_state = BugAnimState::FadeOut;
        self.phase_elapsed_secs = 0.0;
    }

    /// Trigger a pulse animation.
    pub fn pulse(&mut self) {
        if self.anim_state == BugAnimState::Steady {
            self.anim_state = BugAnimState::Pulse;
            self.phase_elapsed_secs = 0.0;
        }
    }

    /// Advance the bug animation by `dt_secs`.
    ///
    /// Returns `true` when the bug has completed its current action
    /// (e.g. fully faded in, pulse complete, fully hidden).
    pub fn advance(&mut self, dt_secs: f32, config: &LogoBugConfig) -> bool {
        self.phase_elapsed_secs += dt_secs;
        match self.anim_state {
            BugAnimState::FadeIn => {
                let t = if config.fade_in_secs > 0.0 {
                    (self.phase_elapsed_secs / config.fade_in_secs).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                self.current_opacity = ease_in_out(t) * config.steady_opacity;
                if t >= 1.0 {
                    self.anim_state = BugAnimState::Steady;
                    self.phase_elapsed_secs = 0.0;
                    return true;
                }
                false
            }
            BugAnimState::Steady => {
                self.current_opacity = config.steady_opacity;
                false
            }
            BugAnimState::FadeOut => {
                let t = if config.fade_out_secs > 0.0 {
                    (self.phase_elapsed_secs / config.fade_out_secs).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                self.current_opacity = (1.0 - ease_in_out(t)) * config.steady_opacity;
                if t >= 1.0 {
                    self.anim_state = BugAnimState::Hidden;
                    self.current_opacity = 0.0;
                    return true;
                }
                false
            }
            BugAnimState::Hidden => {
                self.current_opacity = 0.0;
                false
            }
            BugAnimState::Pulse => {
                // Pulse up then back down over `pulse_hold_secs`.
                let pulse_dur = config.pulse_hold_secs;
                let t = if pulse_dur > 0.0 {
                    (self.phase_elapsed_secs / pulse_dur).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                // Triangle: opacity rises to peak at t=0.5, back to steady at t=1.0.
                let pulse_factor = if t < 0.5 { t * 2.0 } else { (1.0 - t) * 2.0 };
                self.current_opacity = config.steady_opacity
                    + (config.pulse_peak_opacity - config.steady_opacity) * pulse_factor;
                if t >= 1.0 {
                    self.anim_state = BugAnimState::Steady;
                    self.current_opacity = config.steady_opacity;
                    self.phase_elapsed_secs = 0.0;
                    return true;
                }
                false
            }
        }
    }

    /// Returns `true` when the bug is fully hidden.
    pub fn is_hidden(&self) -> bool {
        self.anim_state == BugAnimState::Hidden
    }

    /// Returns `true` when the bug is visible to any degree.
    pub fn is_visible(&self) -> bool {
        self.current_opacity > 0.0
    }
}

/// Ease-in-out (cubic) easing function.
fn ease_in_out(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Renderer for the logo bug overlay.
pub struct LogoBugRenderer;

impl LogoBugRenderer {
    /// Render the logo bug as a full-frame RGBA pixel buffer.
    ///
    /// The logo region is filled with the placeholder color (or a provided
    /// logo slice) at the computed opacity. All other pixels are transparent.
    ///
    /// Returns `Vec<u8>` of length `frame_width * frame_height * 4`.
    pub fn render(
        state: &LogoBugState,
        config: &LogoBugConfig,
        logo_rgba: Option<&[u8]>,
    ) -> Vec<u8> {
        let w = config.frame_width as usize;
        let h = config.frame_height as usize;
        let mut data = vec![0u8; w * h * 4];

        if !state.is_visible() {
            return data;
        }

        let (ox, oy) = config.bug_origin();
        let lw = config.logo_width as usize;
        let lh = config.logo_height as usize;
        let opacity = state.current_opacity.clamp(0.0, 1.0);

        for row in 0..lh {
            let fy = oy as usize + row;
            if fy >= h {
                break;
            }
            for col in 0..lw {
                let fx = ox as usize + col;
                if fx >= w {
                    break;
                }
                let dst_idx = (fy * w + fx) * 4;

                let (r, g, b, base_a) = if let Some(logo) = logo_rgba {
                    let src_idx = (row * lw + col) * 4;
                    if src_idx + 3 < logo.len() {
                        (
                            logo[src_idx],
                            logo[src_idx + 1],
                            logo[src_idx + 2],
                            logo[src_idx + 3],
                        )
                    } else {
                        let c = config.placeholder_color;
                        (c[0], c[1], c[2], c[3])
                    }
                } else {
                    let c = config.placeholder_color;
                    (c[0], c[1], c[2], c[3])
                };

                let effective_a = (base_a as f32 / 255.0 * opacity * 255.0) as u8;
                if dst_idx + 3 < data.len() {
                    data[dst_idx] = r;
                    data[dst_idx + 1] = g;
                    data[dst_idx + 2] = b;
                    data[dst_idx + 3] = effective_a;
                }
            }
        }

        data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> LogoBugConfig {
        LogoBugConfig {
            frame_width: 320,
            frame_height: 240,
            logo_width: 60,
            logo_height: 30,
            padding_px: 10,
            ..LogoBugConfig::default()
        }
    }

    #[test]
    fn test_bug_corner_default() {
        assert_eq!(BugCorner::default(), BugCorner::BottomRight);
    }

    #[test]
    fn test_bug_anim_state_default() {
        assert_eq!(BugAnimState::default(), BugAnimState::FadeIn);
    }

    #[test]
    fn test_logo_bug_config_origin_bottom_right() {
        let cfg = small_config();
        let (x, y) = cfg.bug_origin();
        assert_eq!(x, 320 - 60 - 10);
        assert_eq!(y, 240 - 30 - 10);
    }

    #[test]
    fn test_logo_bug_config_origin_top_left() {
        let cfg = LogoBugConfig {
            corner: BugCorner::TopLeft,
            ..small_config()
        };
        let (x, y) = cfg.bug_origin();
        assert_eq!(x, 10);
        assert_eq!(y, 10);
    }

    #[test]
    fn test_logo_bug_config_origin_top_right() {
        let cfg = LogoBugConfig {
            corner: BugCorner::TopRight,
            ..small_config()
        };
        let (x, y) = cfg.bug_origin();
        assert_eq!(x, 320 - 60 - 10);
        assert_eq!(y, 10);
    }

    #[test]
    fn test_logo_bug_config_origin_bottom_left() {
        let cfg = LogoBugConfig {
            corner: BugCorner::BottomLeft,
            ..small_config()
        };
        let (x, y) = cfg.bug_origin();
        assert_eq!(x, 10);
        assert_eq!(y, 240 - 30 - 10);
    }

    #[test]
    fn test_bug_state_initial() {
        let state = LogoBugState::new();
        assert_eq!(state.anim_state, BugAnimState::FadeIn);
        assert!((state.current_opacity).abs() < f32::EPSILON);
    }

    #[test]
    fn test_bug_state_advance_fade_in() {
        let cfg = small_config();
        let mut state = LogoBugState::new();
        state.advance(cfg.fade_in_secs * 0.5, &cfg);
        assert!(state.current_opacity > 0.0);
        assert_eq!(state.anim_state, BugAnimState::FadeIn);
    }

    #[test]
    fn test_bug_state_advance_to_steady() {
        let cfg = small_config();
        let mut state = LogoBugState::new();
        let done = state.advance(cfg.fade_in_secs + 0.01, &cfg);
        assert!(done);
        assert_eq!(state.anim_state, BugAnimState::Steady);
        assert!((state.current_opacity - cfg.steady_opacity).abs() < 0.01);
    }

    #[test]
    fn test_bug_state_fade_out() {
        let cfg = small_config();
        let mut state = LogoBugState {
            anim_state: BugAnimState::Steady,
            current_opacity: cfg.steady_opacity,
            phase_elapsed_secs: 0.0,
        };
        state.fade_out();
        assert_eq!(state.anim_state, BugAnimState::FadeOut);
        let done = state.advance(cfg.fade_out_secs + 0.01, &cfg);
        assert!(done);
        assert_eq!(state.anim_state, BugAnimState::Hidden);
        assert!(state.is_hidden());
    }

    #[test]
    fn test_bug_state_pulse() {
        let cfg = small_config();
        let mut state = LogoBugState {
            anim_state: BugAnimState::Steady,
            current_opacity: cfg.steady_opacity,
            phase_elapsed_secs: 0.0,
        };
        state.pulse();
        assert_eq!(state.anim_state, BugAnimState::Pulse);
        // Advance to pulse peak (halfway through pulse hold).
        state.advance(cfg.pulse_hold_secs * 0.5, &cfg);
        assert!(state.current_opacity > cfg.steady_opacity);
    }

    #[test]
    fn test_bug_state_pulse_only_from_steady() {
        let mut state = LogoBugState::new(); // FadeIn state
        state.pulse();
        // Should not transition to Pulse from FadeIn.
        assert_eq!(state.anim_state, BugAnimState::FadeIn);
    }

    #[test]
    fn test_bug_state_is_visible() {
        let state = LogoBugState {
            anim_state: BugAnimState::Steady,
            current_opacity: 0.5,
            phase_elapsed_secs: 0.0,
        };
        assert!(state.is_visible());
        let hidden = LogoBugState {
            anim_state: BugAnimState::Hidden,
            current_opacity: 0.0,
            phase_elapsed_secs: 0.0,
        };
        assert!(!hidden.is_visible());
    }

    #[test]
    fn test_logo_bug_renderer_output_size() {
        let state = LogoBugState {
            anim_state: BugAnimState::Steady,
            current_opacity: 0.6,
            phase_elapsed_secs: 0.0,
        };
        let cfg = small_config();
        let data = LogoBugRenderer::render(&state, &cfg, None);
        assert_eq!(data.len(), 320 * 240 * 4);
    }

    #[test]
    fn test_logo_bug_renderer_hidden_is_transparent() {
        let state = LogoBugState {
            anim_state: BugAnimState::Hidden,
            current_opacity: 0.0,
            phase_elapsed_secs: 0.0,
        };
        let cfg = small_config();
        let data = LogoBugRenderer::render(&state, &cfg, None);
        let all_zero = data.iter().all(|&b| b == 0);
        assert!(all_zero);
    }

    #[test]
    fn test_logo_bug_renderer_steady_writes_pixels() {
        let cfg = small_config();
        let state = LogoBugState {
            anim_state: BugAnimState::Steady,
            current_opacity: 1.0,
            phase_elapsed_secs: 0.0,
        };
        let data = LogoBugRenderer::render(&state, &cfg, None);
        let has_nonzero = data.iter().any(|&b| b > 0);
        assert!(has_nonzero);
    }

    #[test]
    fn test_ease_in_out_bounds() {
        assert!((ease_in_out(0.0)).abs() < f32::EPSILON);
        assert!((ease_in_out(1.0) - 1.0).abs() < f32::EPSILON);
        assert!(ease_in_out(0.5) > 0.0 && ease_in_out(0.5) < 1.0);
    }
}
