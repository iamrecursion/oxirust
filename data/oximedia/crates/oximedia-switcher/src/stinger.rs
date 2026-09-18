//! Stinger transition support for video switchers.
//!
//! A stinger transition uses an animated overlay clip (e.g. a station ID logo
//! animation) to mask the switch between program and preview sources.  The
//! overlay is composited over the program/preview mix.  At the `cut_point`
//! frame index the underlying source switches from *program* to *preview*;
//! before that frame the program is shown beneath the overlay, and from that
//! frame onwards the preview is shown beneath the overlay.
//!
//! # Frame layout
//!
//! Each `FrameData` in `StingerTransition::overlay_frames` is an RGBA8 pixel
//! buffer with dimensions `(width, height)`.  The alpha channel is used for
//! compositing: alpha=255 means the overlay is fully opaque, alpha=0 means
//! the underlying source shows through.

use thiserror::Error;

/// Errors that can occur with stinger transitions.
#[derive(Error, Debug, Clone)]
pub enum StingerError {
    /// The cut-point frame index is out of bounds.
    #[error("Cut point {0} is out of bounds for overlay with {1} frames")]
    CutPointOutOfBounds(usize, usize),

    /// The overlay has no frames.
    #[error("Stinger overlay contains no frames")]
    EmptyOverlay,

    /// Frame dimension mismatch.
    #[error("Frame dimension mismatch: expected {0}x{1}, got {2}x{3}")]
    DimensionMismatch(u32, u32, u32, u32),

    /// Invalid frame data length.
    #[error("Invalid frame data: expected {0} bytes, got {1}")]
    InvalidFrameData(usize, usize),
}

/// Raw RGBA8 frame data.
///
/// Each pixel occupies 4 bytes: R, G, B, A.
#[derive(Debug, Clone)]
pub struct FrameData {
    /// Pixel data in RGBA8 format (row-major).
    pub data: Vec<u8>,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
}

impl FrameData {
    /// Create a new `FrameData`, validating that `data.len() == width * height * 4`.
    pub fn new(data: Vec<u8>, width: u32, height: u32) -> Result<Self, StingerError> {
        let expected = (width as usize) * (height as usize) * 4;
        if data.len() != expected {
            return Err(StingerError::InvalidFrameData(expected, data.len()));
        }
        Ok(Self {
            data,
            width,
            height,
        })
    }

    /// Create an all-transparent (black) frame of the given dimensions.
    pub fn transparent(width: u32, height: u32) -> Self {
        let len = (width as usize) * (height as usize) * 4;
        Self {
            data: vec![0u8; len],
            width,
            height,
        }
    }

    /// Create a fully-opaque black frame.
    pub fn black(width: u32, height: u32) -> Self {
        let len = (width as usize) * (height as usize) * 4;
        let mut data = vec![0u8; len];
        // Set alpha to 255 for all pixels.
        for i in (3..len).step_by(4) {
            data[i] = 255;
        }
        Self {
            data,
            width,
            height,
        }
    }

    /// Composite this RGBA overlay onto an RGB8 background frame in-place.
    ///
    /// The background slice must have length `width * height * 3`.
    /// Alpha blending: `out = overlay_rgb * alpha + bg_rgb * (1 - alpha)`.
    pub fn composite_onto_rgb(&self, background: &mut [u8]) -> Result<(), StingerError> {
        let expected_bg = (self.width as usize) * (self.height as usize) * 3;
        if background.len() != expected_bg {
            return Err(StingerError::InvalidFrameData(
                expected_bg,
                background.len(),
            ));
        }

        let pixel_count = (self.width as usize) * (self.height as usize);
        for i in 0..pixel_count {
            let ov_idx = i * 4;
            let bg_idx = i * 3;

            if ov_idx + 3 >= self.data.len() {
                break;
            }

            let ov_r = self.data[ov_idx] as f32;
            let ov_g = self.data[ov_idx + 1] as f32;
            let ov_b = self.data[ov_idx + 2] as f32;
            let alpha = self.data[ov_idx + 3] as f32 / 255.0;

            let bg_r = background[bg_idx] as f32;
            let bg_g = background[bg_idx + 1] as f32;
            let bg_b = background[bg_idx + 2] as f32;

            background[bg_idx] = (ov_r * alpha + bg_r * (1.0 - alpha)) as u8;
            background[bg_idx + 1] = (ov_g * alpha + bg_g * (1.0 - alpha)) as u8;
            background[bg_idx + 2] = (ov_b * alpha + bg_b * (1.0 - alpha)) as u8;
        }

        Ok(())
    }
}

/// A stinger transition definition.
///
/// The stinger animation is stored as a sequence of `FrameData` (RGBA8).
/// At frame index `cut_point` the underlying source switches from *program*
/// to *preview*.  The overlay continues to play until all frames are exhausted.
///
/// # Example
///
/// ```rust
/// use oximedia_switcher::stinger::{FrameData, StingerTransition, StingerTransitionEngine};
///
/// let overlay = vec![
///     FrameData::transparent(4, 4),
///     FrameData::black(4, 4),
///     FrameData::transparent(4, 4),
/// ];
/// let stinger = StingerTransition::new(overlay, 1).expect("valid");
/// let mut engine = StingerTransitionEngine::new(stinger);
///
/// let program = vec![255u8; 4 * 4 * 3];
/// let preview = vec![0u8; 4 * 4 * 3];
///
/// let result = engine.process_frame(0, &program, &preview).expect("ok");
/// assert_eq!(result.len(), 4 * 4 * 3);
/// ```
#[derive(Debug, Clone)]
pub struct StingerTransition {
    /// Overlay frames (RGBA8).
    pub overlay_frames: Vec<FrameData>,
    /// Frame index at which the source cut occurs (0-based).
    pub cut_point: usize,
}

impl StingerTransition {
    /// Create a new stinger transition.
    ///
    /// Returns an error if the overlay is empty or if `cut_point` is out of
    /// bounds for the provided overlay.
    pub fn new(overlay_frames: Vec<FrameData>, cut_point: usize) -> Result<Self, StingerError> {
        if overlay_frames.is_empty() {
            return Err(StingerError::EmptyOverlay);
        }
        if cut_point >= overlay_frames.len() {
            return Err(StingerError::CutPointOutOfBounds(
                cut_point,
                overlay_frames.len(),
            ));
        }
        Ok(Self {
            overlay_frames,
            cut_point,
        })
    }

    /// Total number of overlay frames.
    pub fn frame_count(&self) -> usize {
        self.overlay_frames.len()
    }

    /// Width of the overlay (taken from the first frame).
    pub fn width(&self) -> u32 {
        self.overlay_frames[0].width
    }

    /// Height of the overlay (taken from the first frame).
    pub fn height(&self) -> u32 {
        self.overlay_frames[0].height
    }
}

/// Stateful engine that drives a `StingerTransition` frame by frame.
///
/// Call `process_frame` for each frame index to obtain the composited output.
/// The engine does not track time itself; the caller is responsible for
/// advancing the frame index in sync with the video clock.
pub struct StingerTransitionEngine {
    transition: StingerTransition,
    /// Whether the transition is finished (past all overlay frames).
    finished: bool,
}

impl StingerTransitionEngine {
    /// Create a new engine for the given stinger transition.
    pub fn new(transition: StingerTransition) -> Self {
        Self {
            transition,
            finished: false,
        }
    }

    /// Process a single frame of the stinger transition.
    ///
    /// # Parameters
    /// - `frame_idx`: zero-based index of the current overlay frame.
    /// - `program`: RGB8 pixel data of the current program source.
    /// - `preview`: RGB8 pixel data of the preview (incoming) source.
    ///
    /// # Returns
    /// The composited RGB8 output frame.  The underlying source (program or
    /// preview) is determined by whether `frame_idx` has reached `cut_point`.
    ///
    /// Once `frame_idx` >= `overlay_frames.len()` the engine is finished and
    /// the raw preview buffer is returned unchanged.
    pub fn process_frame(
        &mut self,
        frame_idx: usize,
        program: &[u8],
        preview: &[u8],
    ) -> Result<Vec<u8>, StingerError> {
        if program.len() != preview.len() {
            // Derive expected dimensions from overlay.
            let ow = self.transition.width();
            let oh = self.transition.height();
            return Err(StingerError::DimensionMismatch(ow, oh, 0, 0));
        }

        // Past all overlay frames — return preview passthrough.
        if frame_idx >= self.transition.frame_count() {
            self.finished = true;
            return Ok(preview.to_vec());
        }

        // Select underlying source based on cut point.
        let underlying: Vec<u8> = if frame_idx < self.transition.cut_point {
            program.to_vec()
        } else {
            preview.to_vec()
        };

        // Composite overlay over the chosen underlying source.
        let mut output = underlying;
        let overlay = &self.transition.overlay_frames[frame_idx];
        overlay.composite_onto_rgb(&mut output)?;

        Ok(output)
    }

    /// Returns `true` once the transition has played past all overlay frames.
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Reset the engine so it can be reused from frame 0.
    pub fn reset(&mut self) {
        self.finished = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rgb_frame(width: u32, height: u32, value: u8) -> Vec<u8> {
        vec![value; (width as usize) * (height as usize) * 3]
    }

    fn make_rgba_frame(width: u32, height: u32, r: u8, g: u8, b: u8, a: u8) -> FrameData {
        let len = (width as usize) * (height as usize) * 4;
        let mut data = vec![0u8; len];
        for i in 0..(len / 4) {
            data[i * 4] = r;
            data[i * 4 + 1] = g;
            data[i * 4 + 2] = b;
            data[i * 4 + 3] = a;
        }
        FrameData {
            data,
            width,
            height,
        }
    }

    #[test]
    fn test_frame_data_new_valid() {
        let data = vec![0u8; 4 * 4 * 4];
        let fd = FrameData::new(data, 4, 4);
        assert!(fd.is_ok());
    }

    #[test]
    fn test_frame_data_new_invalid_length() {
        let data = vec![0u8; 10];
        let fd = FrameData::new(data, 4, 4);
        assert!(fd.is_err());
    }

    #[test]
    fn test_stinger_transition_empty_overlay_errors() {
        let result = StingerTransition::new(vec![], 0);
        assert!(result.is_err());
        assert!(matches!(result, Err(StingerError::EmptyOverlay)));
    }

    #[test]
    fn test_stinger_transition_cut_point_out_of_bounds() {
        let overlay = vec![FrameData::transparent(4, 4)];
        let result = StingerTransition::new(overlay, 5);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(StingerError::CutPointOutOfBounds(5, 1))
        ));
    }

    #[test]
    fn test_stinger_before_cut_point_shows_program() {
        // Fully transparent overlay so underlying source is fully visible.
        let overlay = vec![
            FrameData::transparent(2, 2), // frame 0 — before cut
            FrameData::transparent(2, 2), // frame 1 — at cut
            FrameData::transparent(2, 2), // frame 2 — after cut
        ];
        let stinger = StingerTransition::new(overlay, 1).expect("valid");
        let mut engine = StingerTransitionEngine::new(stinger);

        let program = make_rgb_frame(2, 2, 200);
        let preview = make_rgb_frame(2, 2, 50);

        // Frame 0: before cut_point → program source
        let out = engine.process_frame(0, &program, &preview).expect("ok");
        assert_eq!(out, program, "frame 0 should show program");
    }

    #[test]
    fn test_stinger_at_cut_point_shows_preview() {
        let overlay = vec![
            FrameData::transparent(2, 2),
            FrameData::transparent(2, 2),
            FrameData::transparent(2, 2),
        ];
        let stinger = StingerTransition::new(overlay, 1).expect("valid");
        let mut engine = StingerTransitionEngine::new(stinger);

        let program = make_rgb_frame(2, 2, 200);
        let preview = make_rgb_frame(2, 2, 50);

        // Frame 1: at cut_point → preview source
        let out = engine.process_frame(1, &program, &preview).expect("ok");
        assert_eq!(out, preview, "frame at cut_point should show preview");
    }

    #[test]
    fn test_stinger_opaque_overlay_covers_source() {
        // Fully opaque red overlay.
        let overlay = vec![make_rgba_frame(2, 2, 255, 0, 0, 255)];
        let stinger = StingerTransition::new(overlay, 0).expect("valid");
        let mut engine = StingerTransitionEngine::new(stinger);

        let program = make_rgb_frame(2, 2, 128);
        let preview = make_rgb_frame(2, 2, 64);

        let out = engine.process_frame(0, &program, &preview).expect("ok");

        // Every pixel should be red (255, 0, 0).
        for i in 0..(2 * 2) {
            assert_eq!(out[i * 3], 255, "R should be 255 at pixel {i}");
            assert_eq!(out[i * 3 + 1], 0, "G should be 0 at pixel {i}");
            assert_eq!(out[i * 3 + 2], 0, "B should be 0 at pixel {i}");
        }
    }

    #[test]
    fn test_stinger_past_all_frames_returns_preview() {
        let overlay = vec![FrameData::transparent(2, 2)];
        let stinger = StingerTransition::new(overlay, 0).expect("valid");
        let mut engine = StingerTransitionEngine::new(stinger);

        let program = make_rgb_frame(2, 2, 200);
        let preview = make_rgb_frame(2, 2, 50);

        let out = engine.process_frame(99, &program, &preview).expect("ok");
        assert_eq!(out, preview);
        assert!(engine.is_finished());
    }

    #[test]
    fn test_stinger_reset() {
        let overlay = vec![FrameData::transparent(2, 2)];
        let stinger = StingerTransition::new(overlay, 0).expect("valid");
        let mut engine = StingerTransitionEngine::new(stinger);

        let program = make_rgb_frame(2, 2, 200);
        let preview = make_rgb_frame(2, 2, 50);

        // Drive past all frames.
        engine.process_frame(99, &program, &preview).expect("ok");
        assert!(engine.is_finished());

        engine.reset();
        assert!(!engine.is_finished());
    }

    #[test]
    fn test_stinger_frame_count_and_dimensions() {
        let overlay = vec![
            FrameData::transparent(1920, 1080),
            FrameData::transparent(1920, 1080),
        ];
        let stinger = StingerTransition::new(overlay, 1).expect("valid");
        assert_eq!(stinger.frame_count(), 2);
        assert_eq!(stinger.width(), 1920);
        assert_eq!(stinger.height(), 1080);
    }
}
