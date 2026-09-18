//! Stinger transition module for animated full-screen transitions with alpha channel.
//!
//! A stinger is a short animated graphic that sweeps across the full frame,
//! hiding the cut between two video sources. It typically consists of three
//! phases: a "reveal" wipe that moves an animated shape across the frame,
//! a brief "hold" at full coverage, and a "wipe off" phase that uncovers
//! the incoming source.

/// Phase of a stinger transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StingerPhase {
    /// The stinger is animating onto the screen (covering the outgoing source).
    WipeIn,
    /// The stinger is fully covering the frame (cut point occurs here).
    Hold,
    /// The stinger is animating off the screen (revealing the incoming source).
    WipeOut,
    /// The transition is complete.
    Complete,
}

impl Default for StingerPhase {
    fn default() -> Self {
        Self::WipeIn
    }
}

/// Shape of the wipe edge used for the stinger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WipeShape {
    /// Straight diagonal wipe.
    Diagonal,
    /// Horizontal band wipe.
    HorizontalBand,
    /// Radial/iris wipe from center.
    IrisCircle,
    /// Vertical band wipe.
    VerticalBand,
}

impl Default for WipeShape {
    fn default() -> Self {
        Self::Diagonal
    }
}

/// Configuration for a stinger transition.
#[derive(Debug, Clone)]
pub struct StingerConfig {
    /// Frame width.
    pub frame_width: u32,
    /// Frame height.
    pub frame_height: u32,
    /// Duration of the wipe-in phase in seconds.
    pub wipe_in_duration_secs: f32,
    /// Duration of the hold phase in seconds.
    pub hold_duration_secs: f32,
    /// Duration of the wipe-out phase in seconds.
    pub wipe_out_duration_secs: f32,
    /// RGBA color of the stinger graphic.
    pub stinger_color: [u8; 4],
    /// Optional accent color for a secondary element (e.g. logo, edge highlight).
    pub accent_color: [u8; 4],
    /// Shape of the wipe edge.
    pub wipe_shape: WipeShape,
    /// Softness of the wipe edge in pixels.
    pub edge_softness_px: f32,
}

impl Default for StingerConfig {
    fn default() -> Self {
        Self {
            frame_width: 1920,
            frame_height: 1080,
            wipe_in_duration_secs: 0.3,
            hold_duration_secs: 0.1,
            wipe_out_duration_secs: 0.3,
            stinger_color: [10, 40, 120, 255],
            accent_color: [255, 180, 0, 255],
            wipe_shape: WipeShape::Diagonal,
            edge_softness_px: 8.0,
        }
    }
}

impl StingerConfig {
    /// Total duration of the full transition in seconds.
    pub fn total_duration(&self) -> f32 {
        self.wipe_in_duration_secs + self.hold_duration_secs + self.wipe_out_duration_secs
    }
}

/// Runtime state for a stinger transition.
#[derive(Debug, Clone)]
pub struct StingerState {
    /// Elapsed time since the transition started (seconds).
    pub elapsed_secs: f32,
    /// Current phase.
    pub phase: StingerPhase,
    /// Whether the video cut has been performed (occurs during Hold).
    pub cut_performed: bool,
}

impl Default for StingerState {
    fn default() -> Self {
        Self {
            elapsed_secs: 0.0,
            phase: StingerPhase::WipeIn,
            cut_performed: false,
        }
    }
}

impl StingerState {
    /// Create a fresh stinger state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Advance the stinger by `dt_secs` seconds.
    ///
    /// Returns `true` when the transition has completed.
    pub fn advance(&mut self, dt_secs: f32, config: &StingerConfig) -> bool {
        if self.phase == StingerPhase::Complete {
            return true;
        }
        self.elapsed_secs += dt_secs;
        self.update_phase(config);
        self.phase == StingerPhase::Complete
    }

    fn update_phase(&mut self, config: &StingerConfig) {
        let t = self.elapsed_secs;
        let wipe_in_end = config.wipe_in_duration_secs;
        let hold_end = wipe_in_end + config.hold_duration_secs;
        let wipe_out_end = hold_end + config.wipe_out_duration_secs;

        if t < wipe_in_end {
            self.phase = StingerPhase::WipeIn;
        } else if t < hold_end {
            self.phase = StingerPhase::Hold;
            self.cut_performed = true;
        } else if t < wipe_out_end {
            self.phase = StingerPhase::WipeOut;
        } else {
            self.phase = StingerPhase::Complete;
        }
    }

    /// Progress within the current phase, in [0.0, 1.0].
    pub fn phase_progress(&self, config: &StingerConfig) -> f32 {
        let t = self.elapsed_secs;
        match self.phase {
            StingerPhase::WipeIn => {
                if config.wipe_in_duration_secs > 0.0 {
                    (t / config.wipe_in_duration_secs).clamp(0.0, 1.0)
                } else {
                    1.0
                }
            }
            StingerPhase::Hold => {
                let hold_start = config.wipe_in_duration_secs;
                let dur = config.hold_duration_secs;
                if dur > 0.0 {
                    ((t - hold_start) / dur).clamp(0.0, 1.0)
                } else {
                    1.0
                }
            }
            StingerPhase::WipeOut => {
                let out_start = config.wipe_in_duration_secs + config.hold_duration_secs;
                let dur = config.wipe_out_duration_secs;
                if dur > 0.0 {
                    ((t - out_start) / dur).clamp(0.0, 1.0)
                } else {
                    1.0
                }
            }
            StingerPhase::Complete => 1.0,
        }
    }

    /// Reset the stinger to the beginning.
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

/// Renderer for stinger transitions producing an RGBA alpha-channel frame.
pub struct StingerRenderer;

impl StingerRenderer {
    /// Render one frame of the stinger transition as RGBA pixels.
    ///
    /// Returns a `Vec<u8>` of length `frame_width * frame_height * 4`.
    /// The alpha channel encodes the stinger's coverage: alpha=255 means
    /// the stinger fully covers that pixel; alpha=0 means transparent.
    pub fn render(state: &StingerState, config: &StingerConfig) -> Vec<u8> {
        let w = config.frame_width as usize;
        let h = config.frame_height as usize;
        let mut data = vec![0u8; w * h * 4];

        if state.phase == StingerPhase::Complete {
            return data;
        }

        let progress = state.phase_progress(config);
        let cover_progress = match state.phase {
            StingerPhase::WipeIn => progress,
            StingerPhase::Hold => 1.0,
            StingerPhase::WipeOut => 1.0 - progress,
            StingerPhase::Complete => 0.0,
        };

        let softness = config.edge_softness_px.max(1.0);

        for row in 0..h {
            for col in 0..w {
                let norm_x = col as f32 / w as f32;
                let norm_y = row as f32 / h as f32;

                let edge_t = compute_wipe_edge(
                    norm_x,
                    norm_y,
                    cover_progress,
                    config.wipe_shape,
                    softness,
                    w as f32,
                    h as f32,
                );

                if edge_t <= 0.0 {
                    continue;
                }

                let idx = (row * w + col) * 4;
                let alpha = (edge_t * config.stinger_color[3] as f32 / 255.0 * 255.0) as u8;
                let blend_fg = edge_t;
                let blend_bg = 1.0 - blend_fg;

                data[idx] = (config.stinger_color[0] as f32 * blend_fg) as u8;
                data[idx + 1] = (config.stinger_color[1] as f32 * blend_fg
                    + data[idx + 1] as f32 * blend_bg) as u8;
                data[idx + 2] = (config.stinger_color[2] as f32 * blend_fg
                    + data[idx + 2] as f32 * blend_bg) as u8;
                data[idx + 3] = alpha;
            }
        }

        data
    }
}

/// Compute wipe edge alpha for a pixel.
///
/// Returns a value in [0.0, 1.0] representing the stinger coverage at this pixel.
fn compute_wipe_edge(
    nx: f32,
    ny: f32,
    cover_progress: f32,
    shape: WipeShape,
    softness: f32,
    frame_w: f32,
    frame_h: f32,
) -> f32 {
    let softness_norm_x = softness / frame_w;
    let softness_norm_y = softness / frame_h;
    let softness_norm = (softness_norm_x + softness_norm_y) * 0.5;

    let edge_distance = match shape {
        WipeShape::Diagonal => {
            // Diagonal from top-left to bottom-right.
            let diagonal_t = (nx + ny) * 0.5;
            cover_progress - diagonal_t
        }
        WipeShape::HorizontalBand => {
            // Wipe from top to bottom.
            cover_progress - ny
        }
        WipeShape::VerticalBand => {
            // Wipe from left to right.
            cover_progress - nx
        }
        WipeShape::IrisCircle => {
            // Iris wipe: expands from center.
            let dx = nx - 0.5;
            let dy = ny - 0.5;
            let dist = (dx * dx + dy * dy).sqrt() * std::f32::consts::SQRT_2;
            cover_progress - dist
        }
    };

    if softness_norm < f32::EPSILON {
        if edge_distance >= 0.0 {
            1.0
        } else {
            0.0
        }
    } else {
        (edge_distance / softness_norm + 0.5).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> StingerConfig {
        StingerConfig {
            frame_width: 320,
            frame_height: 240,
            ..StingerConfig::default()
        }
    }

    #[test]
    fn test_stinger_phase_default() {
        assert_eq!(StingerPhase::default(), StingerPhase::WipeIn);
    }

    #[test]
    fn test_wipe_shape_default() {
        assert_eq!(WipeShape::default(), WipeShape::Diagonal);
    }

    #[test]
    fn test_stinger_config_total_duration() {
        let cfg = StingerConfig::default();
        let expected = 0.3 + 0.1 + 0.3;
        assert!((cfg.total_duration() - expected).abs() < 0.001);
    }

    #[test]
    fn test_stinger_state_initial() {
        let state = StingerState::new();
        assert_eq!(state.phase, StingerPhase::WipeIn);
        assert!(!state.cut_performed);
    }

    #[test]
    fn test_stinger_advance_wipe_in_to_hold() {
        let cfg = default_config();
        let mut state = StingerState::new();
        state.advance(0.35, &cfg); // past wipe_in (0.3s)
        assert_eq!(state.phase, StingerPhase::Hold);
        assert!(state.cut_performed);
    }

    #[test]
    fn test_stinger_advance_to_wipe_out() {
        let cfg = default_config();
        let mut state = StingerState::new();
        state.advance(0.45, &cfg); // past hold (0.3+0.1=0.4s)
        assert_eq!(state.phase, StingerPhase::WipeOut);
    }

    #[test]
    fn test_stinger_advance_to_complete() {
        let cfg = default_config();
        let mut state = StingerState::new();
        let done = state.advance(1.0, &cfg); // past total duration
        assert!(done);
        assert_eq!(state.phase, StingerPhase::Complete);
    }

    #[test]
    fn test_stinger_phase_progress_wipe_in() {
        let cfg = default_config();
        let mut state = StingerState::new();
        state.elapsed_secs = 0.15; // half way through wipe_in
        state.phase = StingerPhase::WipeIn;
        let p = state.phase_progress(&cfg);
        assert!((p - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_stinger_phase_progress_complete() {
        let cfg = default_config();
        let state = StingerState {
            phase: StingerPhase::Complete,
            elapsed_secs: 1.0,
            cut_performed: true,
        };
        assert!((state.phase_progress(&cfg) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_stinger_reset() {
        let cfg = default_config();
        let mut state = StingerState::new();
        state.advance(1.0, &cfg);
        state.reset();
        assert_eq!(state.phase, StingerPhase::WipeIn);
        assert!(!state.cut_performed);
    }

    #[test]
    fn test_stinger_renderer_output_size() {
        let state = StingerState::new();
        let cfg = default_config();
        let data = StingerRenderer::render(&state, &cfg);
        assert_eq!(data.len(), 320 * 240 * 4);
    }

    #[test]
    fn test_stinger_renderer_complete_is_transparent() {
        let state = StingerState {
            phase: StingerPhase::Complete,
            elapsed_secs: 1.0,
            cut_performed: true,
        };
        let cfg = default_config();
        let data = StingerRenderer::render(&state, &cfg);
        // All alpha should be 0 for complete state.
        let all_transparent = data.chunks_exact(4).all(|p| p[3] == 0);
        assert!(all_transparent);
    }

    #[test]
    fn test_stinger_renderer_hold_has_pixels() {
        let state = StingerState {
            phase: StingerPhase::Hold,
            elapsed_secs: 0.35,
            cut_performed: true,
        };
        let cfg = default_config();
        let data = StingerRenderer::render(&state, &cfg);
        let has_opaque = data.chunks_exact(4).any(|p| p[3] > 0);
        assert!(has_opaque);
    }

    #[test]
    fn test_stinger_renderer_horizontal_band_wipe() {
        let state = StingerState {
            phase: StingerPhase::Hold,
            elapsed_secs: 0.35,
            cut_performed: true,
        };
        let cfg = StingerConfig {
            frame_width: 64,
            frame_height: 64,
            wipe_shape: WipeShape::HorizontalBand,
            ..StingerConfig::default()
        };
        let data = StingerRenderer::render(&state, &cfg);
        assert_eq!(data.len(), 64 * 64 * 4);
    }

    #[test]
    fn test_stinger_renderer_iris_wipe() {
        let state = StingerState {
            phase: StingerPhase::Hold,
            elapsed_secs: 0.35,
            cut_performed: true,
        };
        let cfg = StingerConfig {
            frame_width: 64,
            frame_height: 64,
            wipe_shape: WipeShape::IrisCircle,
            ..StingerConfig::default()
        };
        let data = StingerRenderer::render(&state, &cfg);
        assert_eq!(data.len(), 64 * 64 * 4);
    }
}
