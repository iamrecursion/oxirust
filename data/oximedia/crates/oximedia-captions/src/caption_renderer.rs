//! Caption rendering configuration and target selection.
//!
//! Provides abstractions for rendering captions to different output surfaces,
//! configuring font metrics, safe-area margins, and background styling.
//!
//! # Batch Parallel Rendering
//!
//! For burn-in export of large caption tracks, use [`render_captions_batch_parallel`]
//! which processes each [`CaptionFrame`] independently via Rayon's work-stealing
//! thread pool.  The per-frame work is pure layout arithmetic (no shared mutable
//! state), so parallelism is always safe.

use rayon::prelude::*;

/// Output surface to which captions are rendered.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RenderTarget {
    /// Software RGBA frame buffer at the given dimensions.
    FrameBuffer {
        /// Frame width in pixels.
        width: u32,
        /// Frame height in pixels.
        height: u32,
    },
    /// Burn-in directly onto an encoded video stream.
    VideoBurnIn,
    /// HTML/CSS overlay for web players.
    WebOverlay,
    /// Native platform accessibility layer (e.g. macOS `VoiceOver`).
    AccessibilityLayer,
    /// Dedicated sidecar subtitle stream (not burned in).
    SidecarStream {
        /// MIME type of the sidecar format (e.g. `"text/vtt"`).
        mime_type: String,
    },
}

impl RenderTarget {
    /// Returns `true` when captions are composited onto the image.
    #[must_use]
    pub fn is_burned_in(&self) -> bool {
        matches!(self, Self::FrameBuffer { .. } | Self::VideoBurnIn)
    }

    /// Returns `true` when the target produces a separate stream.
    #[must_use]
    pub fn is_sidecar(&self) -> bool {
        matches!(self, Self::SidecarStream { .. } | Self::WebOverlay)
    }
}

/// RGBA colour with components in `[0, 255]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RgbaColor {
    /// Red component.
    pub r: u8,
    /// Green component.
    pub g: u8,
    /// Blue component.
    pub b: u8,
    /// Alpha component (`0` = transparent, `255` = opaque).
    pub a: u8,
}

impl RgbaColor {
    /// Opaque white.
    pub const WHITE: Self = Self {
        r: 255,
        g: 255,
        b: 255,
        a: 255,
    };
    /// Fully transparent black.
    pub const TRANSPARENT: Self = Self {
        r: 0,
        g: 0,
        b: 0,
        a: 0,
    };
    /// Semi-transparent black (typical caption background).
    pub const CAPTION_BG: Self = Self {
        r: 0,
        g: 0,
        b: 0,
        a: 192,
    };

    /// Create a new colour.
    #[must_use]
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Returns `true` when the colour is fully transparent.
    #[must_use]
    pub fn is_transparent(self) -> bool {
        self.a == 0
    }
}

/// Safe-area inset expressed as a fraction of the frame dimension (`0.0`–`1.0`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SafeAreaInsets {
    /// Left inset as a fraction.
    pub left: f32,
    /// Right inset as a fraction.
    pub right: f32,
    /// Top inset as a fraction.
    pub top: f32,
    /// Bottom inset as a fraction.
    pub bottom: f32,
}

impl SafeAreaInsets {
    /// Standard 10% EBU/SMPTE safe area.
    #[must_use]
    pub fn standard() -> Self {
        Self {
            left: 0.1,
            right: 0.1,
            top: 0.1,
            bottom: 0.1,
        }
    }

    /// No insets (fill the full frame).
    #[must_use]
    pub fn none() -> Self {
        Self {
            left: 0.0,
            right: 0.0,
            top: 0.0,
            bottom: 0.0,
        }
    }
}

impl Default for SafeAreaInsets {
    fn default() -> Self {
        Self::standard()
    }
}

/// Configuration for the caption renderer.
#[derive(Debug, Clone)]
pub struct CaptionRenderConfig {
    /// Target output surface.
    pub target: RenderTarget,
    /// Font size in points.
    pub font_size_pt: f32,
    /// Font family name.
    pub font_family: String,
    /// Default text colour.
    pub text_color: RgbaColor,
    /// Background box colour (`TRANSPARENT` to disable).
    pub background_color: RgbaColor,
    /// Safe-area insets applied to caption positioning.
    pub safe_area: SafeAreaInsets,
    /// Whether to enable drop-shadow for readability.
    pub drop_shadow: bool,
    /// Maximum number of caption rows displayed simultaneously.
    pub max_rows: u8,
}

impl Default for CaptionRenderConfig {
    fn default() -> Self {
        Self {
            target: RenderTarget::FrameBuffer {
                width: 1920,
                height: 1080,
            },
            font_size_pt: 36.0,
            font_family: "Arial".to_string(),
            text_color: RgbaColor::WHITE,
            background_color: RgbaColor::CAPTION_BG,
            safe_area: SafeAreaInsets::standard(),
            drop_shadow: true,
            max_rows: 3,
        }
    }
}

/// A rendered caption item ready for compositing.
#[derive(Debug, Clone)]
pub struct RenderedCaption {
    /// The caption text after layout.
    pub text: String,
    /// Normalised x position within the safe area (`0.0`–`1.0`).
    pub x: f32,
    /// Normalised y position within the safe area (`0.0`–`1.0`).
    pub y: f32,
    /// Text colour used during rendering.
    pub color: RgbaColor,
}

/// Prepares caption text for compositing given a render configuration.
///
/// In a production system this would invoke a font rasteriser; here it
/// performs the layout calculations needed to position captions.
#[derive(Debug)]
pub struct CaptionRenderer {
    config: CaptionRenderConfig,
}

impl CaptionRenderer {
    /// Create a new renderer with the given configuration.
    #[must_use]
    pub fn new(config: CaptionRenderConfig) -> Self {
        Self { config }
    }

    /// Access the current render configuration.
    #[must_use]
    pub fn config(&self) -> &CaptionRenderConfig {
        &self.config
    }

    /// Update the render configuration.
    pub fn set_config(&mut self, config: CaptionRenderConfig) {
        self.config = config;
    }

    /// Lay out a caption text string for rendering.
    ///
    /// Returns a `RenderedCaption` positioned at the bottom-centre of the
    /// safe area (the standard broadcast position).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn render(&self, text: &str, row: u8) -> RenderedCaption {
        // Centre horizontally; position from bottom of safe area.
        let row_offset = f32::from(row) * 0.08;
        let y = (1.0_f32 - self.config.safe_area.bottom - row_offset).clamp(0.0, 1.0);

        RenderedCaption {
            text: text.to_string(),
            x: 0.5,
            y,
            color: self.config.text_color,
        }
    }

    /// Render multiple lines, clamped to `max_rows`.
    #[must_use]
    pub fn render_lines(&self, lines: &[&str]) -> Vec<RenderedCaption> {
        lines
            .iter()
            .take(self.config.max_rows as usize)
            .enumerate()
            .map(|(i, line)| {
                #[allow(clippy::cast_possible_truncation)]
                self.render(line, i as u8)
            })
            .collect()
    }

    /// Returns `true` when the current target burns captions into the image.
    #[must_use]
    pub fn is_burned_in(&self) -> bool {
        self.config.target.is_burned_in()
    }

    /// Renders a single caption frame (all its lines) in a stateless, allocation-minimal way.
    ///
    /// This is the per-frame work unit used by [`render_captions_batch_parallel`].
    /// It is deliberately free of any interior mutability so that it may be called
    /// concurrently from multiple Rayon threads without synchronisation overhead.
    #[must_use]
    pub fn render_single_caption(&self, frame: &CaptionFrame) -> RenderedCaptionBatch {
        let lines: Vec<&str> = frame.lines.iter().map(String::as_str).collect();
        let captions = self.render_lines(&lines);
        RenderedCaptionBatch {
            frame_index: frame.frame_index,
            timestamp_ms: frame.timestamp_ms,
            captions,
        }
    }
}

// ============================================================================
// Batch parallel rendering
// ============================================================================

/// A captioned video frame descriptor used as input to batch rendering.
///
/// Each `CaptionFrame` carries the frame's index in the sequence, its
/// presentation timestamp, and the lines of text to be composited.
#[derive(Debug, Clone)]
pub struct CaptionFrame {
    /// Zero-based index of the frame in the export sequence.
    pub frame_index: usize,
    /// Presentation timestamp in milliseconds.
    pub timestamp_ms: u64,
    /// Caption lines for this frame (at most `CaptionRenderConfig::max_rows` are used).
    pub lines: Vec<String>,
}

impl CaptionFrame {
    /// Creates a new `CaptionFrame`.
    #[must_use]
    pub fn new(frame_index: usize, timestamp_ms: u64, lines: Vec<String>) -> Self {
        Self {
            frame_index,
            timestamp_ms,
            lines,
        }
    }
}

/// The rendering result for a single [`CaptionFrame`].
#[derive(Debug, Clone)]
pub struct RenderedCaptionBatch {
    /// Frame index matching the input [`CaptionFrame::frame_index`].
    pub frame_index: usize,
    /// Presentation timestamp matching [`CaptionFrame::timestamp_ms`].
    pub timestamp_ms: u64,
    /// Per-row rendering output, in row order.
    pub captions: Vec<RenderedCaption>,
}

/// Renders a batch of caption frames in parallel using Rayon's global thread pool.
///
/// The output `Vec` preserves the same order as the input `captions` slice.
/// Each frame is processed by [`CaptionRenderer::render_single_caption`]; no
/// shared mutable state is accessed.
///
/// # Example
///
/// ```rust
/// use oximedia_captions::caption_renderer::{
///     CaptionFrame, CaptionRenderConfig, CaptionRenderer, render_captions_batch_parallel,
/// };
///
/// let config = CaptionRenderConfig::default();
/// let renderer = CaptionRenderer::new(config);
/// let frames = vec![
///     CaptionFrame::new(0, 0,    vec!["Hello".to_string()]),
///     CaptionFrame::new(1, 1000, vec!["World".to_string()]),
/// ];
/// let results = render_captions_batch_parallel(&frames, &renderer);
/// assert_eq!(results.len(), 2);
/// assert_eq!(results[0].frame_index, 0);
/// ```
#[must_use]
pub fn render_captions_batch_parallel(
    captions: &[CaptionFrame],
    renderer: &CaptionRenderer,
) -> Vec<RenderedCaptionBatch> {
    captions
        .par_iter()
        .map(|frame| renderer.render_single_caption(frame))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_target_is_burned_in() {
        assert!(RenderTarget::FrameBuffer {
            width: 1920,
            height: 1080
        }
        .is_burned_in());
        assert!(RenderTarget::VideoBurnIn.is_burned_in());
        assert!(!RenderTarget::WebOverlay.is_burned_in());
    }

    #[test]
    fn test_render_target_is_sidecar() {
        assert!(RenderTarget::WebOverlay.is_sidecar());
        assert!(RenderTarget::SidecarStream {
            mime_type: "text/vtt".to_string()
        }
        .is_sidecar());
        assert!(!RenderTarget::VideoBurnIn.is_sidecar());
    }

    #[test]
    fn test_rgba_constants() {
        assert_eq!(RgbaColor::WHITE.a, 255);
        assert_eq!(RgbaColor::TRANSPARENT.a, 0);
        assert!(!RgbaColor::CAPTION_BG.is_transparent());
    }

    #[test]
    fn test_rgba_is_transparent() {
        assert!(RgbaColor::TRANSPARENT.is_transparent());
        assert!(!RgbaColor::WHITE.is_transparent());
    }

    #[test]
    fn test_safe_area_standard() {
        let sa = SafeAreaInsets::standard();
        assert!((sa.left - 0.1).abs() < 1e-6);
        assert!((sa.bottom - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_safe_area_none() {
        let sa = SafeAreaInsets::none();
        assert_eq!(sa.left, 0.0);
        assert_eq!(sa.top, 0.0);
    }

    #[test]
    fn test_default_config() {
        let cfg = CaptionRenderConfig::default();
        assert_eq!(cfg.max_rows, 3);
        assert!(cfg.drop_shadow);
    }

    #[test]
    fn test_renderer_render_row_0() {
        let renderer = CaptionRenderer::new(CaptionRenderConfig::default());
        let cap = renderer.render("Hello world", 0);
        assert_eq!(cap.text, "Hello world");
        // row 0: y = 1 - 0.1 - 0 = 0.9
        assert!((cap.y - 0.9).abs() < 1e-5, "y={}", cap.y);
        assert!((cap.x - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_renderer_render_row_1_lower() {
        let renderer = CaptionRenderer::new(CaptionRenderConfig::default());
        let row0 = renderer.render("line0", 0);
        let row1 = renderer.render("line1", 1);
        // row 1 should be higher on screen (smaller y)
        assert!(row1.y < row0.y);
    }

    #[test]
    fn test_render_lines_respects_max_rows() {
        let renderer = CaptionRenderer::new(CaptionRenderConfig::default());
        let lines = vec!["a", "b", "c", "d", "e"];
        let rendered = renderer.render_lines(&lines);
        assert_eq!(rendered.len(), 3); // max_rows = 3
    }

    #[test]
    fn test_render_lines_fewer_than_max() {
        let renderer = CaptionRenderer::new(CaptionRenderConfig::default());
        let rendered = renderer.render_lines(&["only"]);
        assert_eq!(rendered.len(), 1);
    }

    #[test]
    fn test_is_burned_in_true() {
        let renderer = CaptionRenderer::new(CaptionRenderConfig::default());
        assert!(renderer.is_burned_in());
    }

    #[test]
    fn test_is_burned_in_false_for_web() {
        let mut cfg = CaptionRenderConfig::default();
        cfg.target = RenderTarget::WebOverlay;
        let renderer = CaptionRenderer::new(cfg);
        assert!(!renderer.is_burned_in());
    }

    #[test]
    fn test_set_config() {
        let mut renderer = CaptionRenderer::new(CaptionRenderConfig::default());
        let mut new_cfg = CaptionRenderConfig::default();
        new_cfg.font_size_pt = 48.0;
        renderer.set_config(new_cfg);
        assert!((renderer.config().font_size_pt - 48.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Batch parallel rendering tests
    // -----------------------------------------------------------------------

    fn make_frames(count: usize) -> Vec<CaptionFrame> {
        (0..count)
            .map(|i| {
                CaptionFrame::new(
                    i,
                    (i as u64) * 1000,
                    vec![format!("Caption line {i}"), format!("Row 2 for {i}")],
                )
            })
            .collect()
    }

    /// Parallel and sequential rendering must produce identical output.
    #[test]
    fn test_parallel_render_matches_sequential() {
        let renderer = CaptionRenderer::new(CaptionRenderConfig::default());
        let frames = make_frames(50);

        // Sequential reference.
        let sequential: Vec<RenderedCaptionBatch> = frames
            .iter()
            .map(|f| renderer.render_single_caption(f))
            .collect();

        // Parallel under test.
        let parallel = render_captions_batch_parallel(&frames, &renderer);

        assert_eq!(sequential.len(), parallel.len());
        for (seq, par) in sequential.iter().zip(parallel.iter()) {
            assert_eq!(seq.frame_index, par.frame_index, "frame_index mismatch");
            assert_eq!(seq.timestamp_ms, par.timestamp_ms, "timestamp_ms mismatch");
            assert_eq!(
                seq.captions.len(),
                par.captions.len(),
                "captions count mismatch for frame {}",
                seq.frame_index
            );
            for (sc, pc) in seq.captions.iter().zip(par.captions.iter()) {
                assert_eq!(sc.text, pc.text, "text mismatch");
                assert!((sc.x - pc.x).abs() < 1e-6, "x mismatch");
                assert!((sc.y - pc.y).abs() < 1e-6, "y mismatch");
                assert_eq!(sc.color, pc.color, "color mismatch");
            }
        }
    }

    /// Empty frame list returns an empty result without panicking.
    #[test]
    fn test_parallel_render_empty_input() {
        let renderer = CaptionRenderer::new(CaptionRenderConfig::default());
        let result = render_captions_batch_parallel(&[], &renderer);
        assert!(result.is_empty());
    }

    /// Output is ordered by frame_index (matches input order even if rayon reorders).
    #[test]
    fn test_parallel_render_output_order() {
        let renderer = CaptionRenderer::new(CaptionRenderConfig::default());
        let frames = make_frames(20);
        let result = render_captions_batch_parallel(&frames, &renderer);
        for (i, batch) in result.iter().enumerate() {
            assert_eq!(batch.frame_index, i, "output order must match input order");
        }
    }

    /// `render_single_caption` respects `max_rows` limit.
    #[test]
    fn test_render_single_caption_max_rows() {
        let mut cfg = CaptionRenderConfig::default();
        cfg.max_rows = 2;
        let renderer = CaptionRenderer::new(cfg);
        let frame = CaptionFrame::new(
            0,
            0,
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
        );
        let batch = renderer.render_single_caption(&frame);
        assert_eq!(
            batch.captions.len(),
            2,
            "max_rows=2 must truncate to 2 captions"
        );
    }
}
