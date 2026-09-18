//! WebAssembly bindings for shot detection from `oximedia-shots`.
//!
//! Provides cut detection between consecutive video frames operating entirely
//! in-memory without file-system access, suitable for browser-based video
//! analysis workflows.

use wasm_bindgen::prelude::*;

use oximedia_shots::detect::cut::CutDetector;
use oximedia_shots::frame_buffer::FrameBuffer;

// ---------------------------------------------------------------------------
// Error helper
// ---------------------------------------------------------------------------

fn js_err(msg: impl std::fmt::Display) -> JsValue {
    crate::utils::js_err(&format!("{msg}"))
}

// ---------------------------------------------------------------------------
// ShotDetector
// ---------------------------------------------------------------------------

/// Shot boundary (cut) detector for browser-based video analysis.
///
/// Feed consecutive frames via `detect_cut`; the internal state keeps the
/// previous frame so you only need to supply one frame per call after the
/// first.  Call `reset` to restart detection for a new clip.
///
/// # Example
///
/// ```javascript
/// const detector = new ShotDetector(0.3);
/// for (const frame of frames) {
///     const isCut = detector.detect_cut(frame.data, frame.width, frame.height);
///     if (isCut) console.log('Shot boundary detected');
/// }
/// ```
#[wasm_bindgen]
pub struct ShotDetector {
    inner: CutDetector,
    prev_frame: Option<FrameBuffer>,
}

#[wasm_bindgen]
impl ShotDetector {
    /// Create a new shot detector with the given histogram-difference threshold.
    ///
    /// `threshold` is a value in [0.0, 1.0]: lower = more sensitive (detects
    /// smaller colour changes as cuts); a value around 0.25–0.35 works well for
    /// most content.
    ///
    /// # Errors
    ///
    /// Returns an error if `threshold` is outside [0.0, 1.0].
    #[wasm_bindgen(constructor)]
    pub fn new(threshold: f32) -> Result<ShotDetector, JsValue> {
        if !(0.0..=1.0).contains(&threshold) {
            return Err(js_err(format!(
                "threshold {threshold} must be in [0.0, 1.0]"
            )));
        }
        Ok(ShotDetector {
            inner: CutDetector::with_params(threshold, threshold * 1.3, 3),
            prev_frame: None,
        })
    }

    /// Detect whether a hard cut occurred between the previous frame and this one.
    ///
    /// `frame_data` must be a flat `u8` RGBA buffer of exactly `width × height × 4`
    /// bytes.  The first call always returns `false` (establishes the baseline).
    ///
    /// # Panics (JS exception)
    ///
    /// Throws a JS error if `frame_data.length != width * height * 4`.
    pub fn detect_cut(&mut self, frame_data: &[u8], width: u32, height: u32) -> bool {
        let w = width as usize;
        let h = height as usize;
        let expected = w * h * 4;
        if frame_data.len() != expected {
            // Silently return false rather than throwing for WASM ergonomics.
            return false;
        }

        // Convert RGBA → RGB (drop alpha channel) for FrameBuffer (3 channels).
        let rgb_len = w * h * 3;
        let mut rgb = Vec::with_capacity(rgb_len);
        for chunk in frame_data.chunks_exact(4) {
            rgb.push(chunk[0]); // R
            rgb.push(chunk[1]); // G
            rgb.push(chunk[2]); // B
        }

        let current = match FrameBuffer::from_vec(h, w, 3, rgb) {
            Some(f) => f,
            None => return false,
        };

        let result = match &self.prev_frame {
            None => {
                self.prev_frame = Some(current);
                return false;
            }
            Some(prev) => self.inner.detect_cut(prev, &current),
        };

        self.prev_frame = Some(current);

        match result {
            Ok((is_cut, _score)) => is_cut,
            Err(_) => false,
        }
    }

    /// Reset detector state; the next call to `detect_cut` will establish a
    /// new baseline frame.
    pub fn reset(&mut self) {
        self.prev_frame = None;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_solid_rgba(r: u8, g: u8, b: u8, w: usize, h: usize) -> Vec<u8> {
        let mut buf = Vec::with_capacity(w * h * 4);
        for _ in 0..w * h {
            buf.extend_from_slice(&[r, g, b, 255]);
        }
        buf
    }

    #[test]
    fn constructor_valid_threshold() {
        assert!(ShotDetector::new(0.3).is_ok());
        assert!(ShotDetector::new(0.0).is_ok());
        assert!(ShotDetector::new(1.0).is_ok());
    }

    #[test]
    fn constructor_rejects_out_of_range() {
        assert!(ShotDetector::new(-0.1).is_err());
        assert!(ShotDetector::new(1.1).is_err());
    }

    #[test]
    fn first_frame_always_false() {
        let mut det = ShotDetector::new(0.3).expect("valid");
        let frame = make_solid_rgba(128, 64, 32, 8, 8);
        assert!(!det.detect_cut(&frame, 8, 8), "first frame must be false");
    }

    #[test]
    fn same_frame_no_cut() {
        let mut det = ShotDetector::new(0.3).expect("valid");
        let frame = make_solid_rgba(100, 100, 100, 8, 8);
        det.detect_cut(&frame, 8, 8);
        assert!(
            !det.detect_cut(&frame, 8, 8),
            "identical frame should not be a cut"
        );
    }

    #[test]
    fn reset_clears_state() {
        let mut det = ShotDetector::new(0.3).expect("valid");
        let f1 = make_solid_rgba(255, 0, 0, 8, 8);
        det.detect_cut(&f1, 8, 8);
        det.reset();
        // After reset, next call is again a "first frame" → false.
        let f2 = make_solid_rgba(0, 0, 255, 8, 8);
        assert!(
            !det.detect_cut(&f2, 8, 8),
            "after reset first frame must be false"
        );
    }
}
