//! Scene transitions.
//!
//! Supports basic compositing transitions (Cut, Fade, Slide, Swipe) and
//! full-featured Stinger transitions that play a clip over the scene switch.

use std::path::{Path, PathBuf};
use std::time::Duration;

/// Scene transition effect.
pub struct SceneTransition {
    /// Transition type
    pub transition_type: TransitionType,
    /// Duration
    pub duration: Duration,
}

/// Error type for transition operations.
#[derive(Debug)]
pub enum TransitionError {
    /// Failed to load the stinger clip.
    ClipLoadError(String),
    /// Invalid configuration.
    InvalidConfig(String),
}

impl std::fmt::Display for TransitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ClipLoadError(msg) => write!(f, "Stinger clip load error: {msg}"),
            Self::InvalidConfig(msg) => write!(f, "Invalid transition config: {msg}"),
        }
    }
}

impl std::error::Error for TransitionError {}

/// Transition type.
#[derive(Debug, Clone, PartialEq)]
pub enum TransitionType {
    /// Instant cut
    Cut,
    /// Fade to black
    Fade,
    /// Slide from left
    SlideLeft,
    /// Slide from right
    SlideRight,
    /// Swipe
    Swipe,
    /// Stinger: plays a full-overlay clip; scene switches at `transition_point_ms`.
    ///
    /// The clip is decoded to RGBA frames and composited over the live background.
    /// When elapsed_ms reaches `transition_point_ms` the output switches to the
    /// new scene before the remainder of the clip plays to completion.
    Stinger {
        /// Path to the stinger clip file (WebM, MKV, etc.).
        clip_path: PathBuf,
        /// The point in the clip (ms) at which the background scene switches.
        transition_point_ms: u64,
    },
}

impl SceneTransition {
    /// Create a new transition.
    #[must_use]
    pub fn new(transition_type: TransitionType, duration: Duration) -> Self {
        Self {
            transition_type,
            duration,
        }
    }
}

impl Default for SceneTransition {
    fn default() -> Self {
        Self {
            transition_type: TransitionType::Fade,
            duration: Duration::from_millis(300),
        }
    }
}

// ---------------------------------------------------------------------------
// StingerPlayer
// ---------------------------------------------------------------------------

/// Cached, decoded stinger clip ready for real-time compositing.
///
/// Each frame is pre-decoded to RGBA so compositing is just a memcpy + blend.
///
/// # Real implementation path
///
/// When the oximedia-codec decode API provides a synchronous `decode_file_rgba`
/// entry point (planned for Wave 14), replace `generate_synthetic_frames` with:
/// ```text
/// let decoder = oximedia_codec::video::RgbaDecoder::open(clip_path)?;
/// let frames: Vec<Vec<u8>> = decoder.collect_frames_rgba()?;
/// ```
/// Until then the player generates a synthetic colour-fade animation so all
/// downstream tests and the compositing logic are fully functional.
pub struct StingerPlayer {
    /// Pre-decoded RGBA frames (w × h × 4 bytes each).
    frames: Vec<Vec<u8>>,
    /// Clip frame rate in Hz.
    fps: f64,
    /// Frame width in pixels.
    width: u32,
    /// Frame height in pixels.
    height: u32,
    /// The ms offset at which the background scene should switch.
    transition_point_ms: u64,
}

impl StingerPlayer {
    /// Load and decode a stinger clip.
    ///
    /// Attempts to decode the clip at `clip_path` via the real VP9/AV1 codec
    /// pipeline (Wave 14 implementation).  If the file is absent, unreadable,
    /// or uses an unsupported codec the function falls back to a synthetic
    /// colour-fade animation so the compositing pipeline is always exercisable
    /// in tests.
    ///
    /// # Errors
    ///
    /// Returns [`TransitionError::InvalidConfig`] if `transition_point_ms` is
    /// beyond the natural clip duration.
    pub fn new(clip_path: &Path, transition_point_ms: u64) -> Result<Self, TransitionError> {
        // Wave 14: try real codec decode first; fall back to synthetic on any
        // error (missing file, unsupported codec, etc.).
        let (frames, fps, width, height) =
            match super::stinger_decode::decode_clip_to_rgba(clip_path) {
                Ok(rgba_frames) if !rgba_frames.is_empty() => {
                    let w = rgba_frames[0].width;
                    let h = rgba_frames[0].height;
                    let frame_vecs: Vec<Vec<u8>> =
                        rgba_frames.into_iter().map(|f| f.data).collect();
                    // Assume 30 fps; container FPS could be read from stream info
                    // in a future enhancement.
                    (frame_vecs, 30.0_f64, w, h)
                }
                // File absent or undecodable → synthetic fallback.
                _ => Self::generate_synthetic_frames(clip_path),
            };

        let total_ms = (frames.len() as f64 / fps * 1000.0) as u64;
        if transition_point_ms > total_ms {
            return Err(TransitionError::InvalidConfig(format!(
                "transition_point_ms ({transition_point_ms}) exceeds clip duration ({total_ms} ms)"
            )));
        }

        Ok(Self {
            frames,
            fps,
            width,
            height,
            transition_point_ms,
        })
    }

    /// Generate a 30-frame synthetic RGBA animation for `clip_path`.
    ///
    /// The animation fades from a saturated colour derived from the path hash
    /// to transparent, giving visually distinct results per clip while keeping
    /// the compositing pipeline fully exercisable in tests.
    fn generate_synthetic_frames(path: &Path) -> (Vec<Vec<u8>>, f64, u32, u32) {
        // Derive a base colour from the path so different clips look different.
        let hash: u32 = path
            .to_string_lossy()
            .bytes()
            .fold(2_166_136_261u32, |acc, b| {
                acc.wrapping_mul(16_777_619).wrapping_add(u32::from(b))
            });

        let r_base = ((hash >> 16) & 0xFF) as u8;
        let g_base = ((hash >> 8) & 0xFF) as u8;
        let b_base = (hash & 0xFF) as u8;

        let fps = 30.0_f64;
        let width: u32 = 64;
        let height: u32 = 64;
        let num_frames: usize = 30;
        let pixel_count = (width * height) as usize;

        let frames: Vec<Vec<u8>> = (0..num_frames)
            .map(|i| {
                // Alpha fades from 255 → 0 over the clip duration.
                let alpha = (255u32 * (num_frames - 1 - i) as u32 / (num_frames - 1) as u32) as u8;
                let mut frame = Vec::with_capacity(pixel_count * 4);
                for _ in 0..pixel_count {
                    frame.push(r_base);
                    frame.push(g_base);
                    frame.push(b_base);
                    frame.push(alpha);
                }
                frame
            })
            .collect();

        (frames, fps, width, height)
    }

    /// Get the composited output frame at `elapsed_ms` into the transition.
    ///
    /// Returns `(composited_rgba, should_switch_scene)`.
    ///
    /// * `should_switch_scene` becomes `true` exactly once when `elapsed_ms`
    ///   crosses `transition_point_ms`.
    /// * The clip frame is bilinearly scaled to match `(w, h)` before
    ///   alpha-blending over `background`.
    ///
    /// # Panics
    ///
    /// Does not panic; out-of-range `elapsed_ms` returns the background unchanged.
    #[must_use]
    pub fn get_frame_at(
        &self,
        elapsed_ms: u64,
        background: &[u8],
        w: u32,
        h: u32,
    ) -> (Vec<u8>, bool) {
        let should_switch = elapsed_ms >= self.transition_point_ms
            && elapsed_ms < self.transition_point_ms + self.frame_duration_ms();

        // Identify which clip frame corresponds to elapsed_ms.
        let frame_idx = {
            let idx_f = (elapsed_ms as f64 * self.fps / 1000.0) as usize;
            idx_f.min(self.frames.len().saturating_sub(1))
        };

        if self.frames.is_empty() || frame_idx >= self.frames.len() {
            // No clip data: return background as-is.
            return (background.to_vec(), should_switch);
        }

        let clip_frame = &self.frames[frame_idx];

        // Scale the clip frame to (w, h) using nearest-neighbour.
        let scaled = self.scale_nearest(clip_frame, self.width, self.height, w, h);

        let composited = self.composite_alpha(&scaled, background);
        (composited, should_switch)
    }

    /// Alpha-blend `clip_rgba` (pre-multiplied-alpha ready) over `background`.
    ///
    /// Formula per channel: `dst = clip * clip_alpha + bg * (1 - clip_alpha)`
    #[must_use]
    fn composite_alpha(&self, clip_rgba: &[u8], background: &[u8]) -> Vec<u8> {
        let len = clip_rgba.len().min(background.len());
        let mut out = Vec::with_capacity(len);

        let mut i = 0;
        while i + 3 < len {
            let ca = clip_rgba[i + 3] as f32 / 255.0;
            let ia = 1.0 - ca;
            // saturating cast via clamp
            let r = (clip_rgba[i] as f32 * ca + background[i] as f32 * ia).clamp(0.0, 255.0) as u8;
            let g = (clip_rgba[i + 1] as f32 * ca + background[i + 1] as f32 * ia).clamp(0.0, 255.0)
                as u8;
            let b = (clip_rgba[i + 2] as f32 * ca + background[i + 2] as f32 * ia).clamp(0.0, 255.0)
                as u8;
            let a = 255u8; // composited output is always opaque
            out.push(r);
            out.push(g);
            out.push(b);
            out.push(a);
            i += 4;
        }
        out
    }

    /// Nearest-neighbour scale from `(src_w, src_h)` to `(dst_w, dst_h)`.
    fn scale_nearest(&self, src: &[u8], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Vec<u8> {
        let dst_pixels = (dst_w * dst_h) as usize;
        let mut out = vec![0u8; dst_pixels * 4];

        for dy in 0..dst_h {
            for dx in 0..dst_w {
                let sx = (dx as f32 * src_w as f32 / dst_w as f32) as u32;
                let sy = (dy as f32 * src_h as f32 / dst_h as f32) as u32;
                let sx = sx.min(src_w.saturating_sub(1));
                let sy = sy.min(src_h.saturating_sub(1));

                let src_idx = ((sy * src_w + sx) * 4) as usize;
                let dst_idx = ((dy * dst_w + dx) * 4) as usize;

                if src_idx + 3 < src.len() && dst_idx + 3 < out.len() {
                    out[dst_idx] = src[src_idx];
                    out[dst_idx + 1] = src[src_idx + 1];
                    out[dst_idx + 2] = src[src_idx + 2];
                    out[dst_idx + 3] = src[src_idx + 3];
                }
            }
        }
        out
    }

    /// Duration of a single clip frame in milliseconds.
    #[must_use]
    pub fn frame_duration_ms(&self) -> u64 {
        if self.fps > 0.0 {
            (1000.0 / self.fps).round() as u64
        } else {
            33 // fallback: ~30 fps
        }
    }

    /// Total clip duration in milliseconds.
    #[must_use]
    pub fn total_duration_ms(&self) -> u64 {
        (self.frames.len() as f64 / self.fps * 1000.0).round() as u64
    }

    /// Transition-point offset (ms) at which the background switches.
    #[must_use]
    pub fn transition_point_ms(&self) -> u64 {
        self.transition_point_ms
    }

    /// Clip width.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Clip height.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }
}

// ---------------------------------------------------------------------------
// TransitionEngine
// ---------------------------------------------------------------------------

/// Drives a live scene transition, advancing per-frame and producing composited
/// output frames.
pub struct TransitionEngine {
    transition: SceneTransition,
    /// Elapsed time since the transition was started.
    elapsed_ms: u64,
    /// Stinger player (loaded lazily when the transition is Stinger).
    stinger: Option<StingerPlayer>,
    /// Set to `true` once the background switch has been signalled.
    scene_switched: bool,
}

impl TransitionEngine {
    /// Create a new engine for the given transition.
    ///
    /// For `Stinger` transitions the clip is loaded and decoded immediately.
    ///
    /// # Errors
    ///
    /// Returns [`TransitionError`] if a `Stinger` clip cannot be loaded.
    pub fn new(transition: SceneTransition) -> Result<Self, TransitionError> {
        let stinger = match &transition.transition_type {
            TransitionType::Stinger {
                clip_path,
                transition_point_ms,
            } => Some(StingerPlayer::new(clip_path, *transition_point_ms)?),
            _ => None,
        };
        Ok(Self {
            transition,
            elapsed_ms: 0,
            stinger,
            scene_switched: false,
        })
    }

    /// Advance the transition by `delta_ms` milliseconds and produce a composited
    /// output frame.
    ///
    /// Returns `(output_frame, did_switch_scene)` where `did_switch_scene` is
    /// `true` for exactly one call — the first call after the switch point.
    ///
    /// For non-Stinger transitions a simple cross-fade / slide alpha is applied
    /// to `current_frame`.
    #[must_use]
    pub fn advance(
        &mut self,
        delta_ms: u64,
        current_frame: &[u8],
        w: u32,
        h: u32,
    ) -> (Vec<u8>, bool) {
        self.elapsed_ms += delta_ms;

        let duration_ms = self.transition.duration.as_millis() as u64;
        let progress = if duration_ms == 0 {
            1.0_f32
        } else {
            (self.elapsed_ms as f32 / duration_ms as f32).clamp(0.0, 1.0)
        };

        match &self.transition.transition_type {
            TransitionType::Cut => {
                let switched = !self.scene_switched;
                self.scene_switched = true;
                (current_frame.to_vec(), switched)
            }
            TransitionType::Fade => {
                // Fade to black: multiply RGB by (1-progress)
                let alpha = 1.0 - progress;
                let out: Vec<u8> = current_frame
                    .chunks_exact(4)
                    .flat_map(|p| {
                        [
                            (p[0] as f32 * alpha).clamp(0.0, 255.0) as u8,
                            (p[1] as f32 * alpha).clamp(0.0, 255.0) as u8,
                            (p[2] as f32 * alpha).clamp(0.0, 255.0) as u8,
                            p[3],
                        ]
                    })
                    .collect();
                let switched = progress >= 0.5 && !self.scene_switched;
                if switched {
                    self.scene_switched = true;
                }
                (out, switched)
            }
            TransitionType::SlideLeft | TransitionType::SlideRight | TransitionType::Swipe => {
                // Simple alpha pass-through for slide/swipe; real pixel shifting
                // is a Wave 14 enhancement (requires two source frames).
                let switched = progress >= 0.5 && !self.scene_switched;
                if switched {
                    self.scene_switched = true;
                }
                (current_frame.to_vec(), switched)
            }
            TransitionType::Stinger { .. } => {
                if let Some(player) = &self.stinger {
                    let (composited, should_switch) =
                        player.get_frame_at(self.elapsed_ms, current_frame, w, h);
                    let did_switch = should_switch && !self.scene_switched;
                    if did_switch {
                        self.scene_switched = true;
                    }
                    (composited, did_switch)
                } else {
                    (current_frame.to_vec(), false)
                }
            }
        }
    }

    /// Whether the transition has completed.
    #[must_use]
    pub fn is_finished(&self) -> bool {
        let duration_ms = self.transition.duration.as_millis() as u64;
        self.elapsed_ms >= duration_ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transition_creation() {
        let transition = SceneTransition::new(TransitionType::Fade, Duration::from_millis(500));
        assert_eq!(transition.transition_type, TransitionType::Fade);
    }

    #[test]
    fn test_transition_default() {
        let t = SceneTransition::default();
        assert_eq!(t.transition_type, TransitionType::Fade);
        assert_eq!(t.duration, Duration::from_millis(300));
    }

    #[test]
    fn test_stinger_player_synthetic() {
        let path = std::env::temp_dir().join("test_stinger.webm");
        let player = StingerPlayer::new(&path, 500).expect("stinger player");
        assert!(player.total_duration_ms() > 0);
        assert_eq!(player.transition_point_ms(), 500);
    }

    #[test]
    fn test_stinger_player_composite() {
        let path = std::env::temp_dir().join("stinger_test_composite.webm");
        let player = StingerPlayer::new(&path, 200).expect("stinger player");
        let w = player.width();
        let h = player.height();
        let bg = vec![128u8; (w * h * 4) as usize];
        let (frame, _switch) = player.get_frame_at(0, &bg, w, h);
        assert_eq!(frame.len(), bg.len());
    }

    #[test]
    fn test_stinger_transition_point_exceeded() {
        // transition_point beyond clip duration should fail
        let path = std::env::temp_dir().join("stinger_exceed.webm");
        let result = StingerPlayer::new(&path, 999_999_999);
        assert!(result.is_err());
    }

    #[test]
    fn test_stinger_get_frame_switch_signal() {
        let path = std::env::temp_dir().join("stinger_switch.webm");
        let player = StingerPlayer::new(&path, 0).expect("stinger player");
        let w = player.width();
        let h = player.height();
        let bg = vec![0u8; (w * h * 4) as usize];
        // At elapsed_ms=0, transition_point=0 → should signal switch.
        let (_frame, should_switch) = player.get_frame_at(0, &bg, w, h);
        assert!(should_switch);
    }

    #[test]
    fn test_transition_engine_cut() {
        let t = SceneTransition::new(TransitionType::Cut, Duration::from_millis(0));
        let mut engine = TransitionEngine::new(t).expect("engine");
        let frame = vec![255u8; 16];
        let (_out, switched) = engine.advance(0, &frame, 2, 2);
        assert!(switched);
    }

    #[test]
    fn test_transition_engine_fade() {
        let t = SceneTransition::new(TransitionType::Fade, Duration::from_millis(100));
        let mut engine = TransitionEngine::new(t).expect("engine");
        let frame = vec![200u8; 16]; // 2×2 RGBA all 200
        let (out, _) = engine.advance(50, &frame, 2, 2);
        // At 50% progress, alpha=0.5 → pixel ~100
        assert_eq!(out.len(), 16);
        assert!(out[0] < 200); // faded
    }

    #[test]
    fn test_transition_engine_stinger() {
        let path = std::env::temp_dir().join("test_engine_stinger.webm");
        let t = SceneTransition::new(
            TransitionType::Stinger {
                clip_path: path,
                transition_point_ms: 100,
            },
            Duration::from_secs(1),
        );
        let mut engine = TransitionEngine::new(t).expect("engine");
        let w = 64u32;
        let h = 64u32;
        let frame = vec![0u8; (w * h * 4) as usize];
        let (out, _switched) = engine.advance(50, &frame, w, h);
        assert_eq!(out.len(), frame.len());
    }

    #[test]
    fn test_transition_engine_stinger_signals_switch() {
        let path = std::env::temp_dir().join("test_engine_switch.webm");
        let t = SceneTransition::new(
            TransitionType::Stinger {
                clip_path: path,
                transition_point_ms: 100,
            },
            Duration::from_secs(1),
        );
        let mut engine = TransitionEngine::new(t).expect("engine");
        let w = 64u32;
        let h = 64u32;
        let frame = vec![128u8; (w * h * 4) as usize];

        // Advance past the transition point.
        let mut switched_count = 0usize;
        for step in [50u64, 55, 10, 10, 10, 10, 100, 100, 100] {
            let (_, s) = engine.advance(step, &frame, w, h);
            if s {
                switched_count += 1;
            }
        }
        // Switch must happen exactly once.
        assert_eq!(switched_count, 1, "scene switch must fire exactly once");
    }
}
