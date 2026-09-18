//! Frame rate cadence conversion.
//!
//! Converts video between common frame rates (e.g. 24→30, 25→30, 50→60)
//! using several strategies: frame duplication, frame dropping, frame
//! blending, and 2:3 / 3:2 pulldown insertion.
//!
//! # Example
//!
//! ```rust
//! use oximedia_video::cadence_convert::{CadenceConverter, CadenceStrategy, Rational};
//!
//! let conv = CadenceConverter::new(
//!     Rational::new(24, 1),
//!     Rational::new(30, 1),
//!     CadenceStrategy::Pulldown23,
//! );
//! let out_count = conv.convert_frame_count(24);
//! assert_eq!(out_count, 30);
//! ```

// -----------------------------------------------------------------------
// Public types
// -----------------------------------------------------------------------

/// A rational number `numerator / denominator` used for frame rates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rational {
    /// Numerator.
    pub num: u64,
    /// Denominator (must not be zero).
    pub den: u64,
}

impl Rational {
    /// Create a new `Rational`.  Panics if `den` is zero in debug builds.
    pub fn new(num: u64, den: u64) -> Self {
        debug_assert!(den != 0, "Rational denominator must not be zero");
        Self { num, den }
    }

    /// Convert to `f64`.
    pub fn as_f64(self) -> f64 {
        if self.den == 0 {
            return 0.0;
        }
        self.num as f64 / self.den as f64
    }
}

impl std::fmt::Display for Rational {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.den == 1 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}

/// How the cadence converter maps source frames to output frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CadenceStrategy {
    /// Blend adjacent source frames in proportion to the fractional timestamp offset.
    Blend,
    /// Duplicate the nearest source frame (nearest-neighbour assignment).
    Duplicate,
    /// Drop frames when the target rate is lower than the source rate.
    Drop,
    /// Insert 2:3 pulldown cadence (film 24 fps → video 30 fps).
    Pulldown23,
    /// Insert 3:2 pulldown cadence (alternate pattern: 3 interlaced + 2 progressive per cycle).
    Pulldown32,
}

/// A reference to an output frame — either a direct copy of a single source
/// frame or a weighted blend of two adjacent source frames.
#[derive(Debug, Clone, PartialEq)]
pub enum FrameRef {
    /// Use source frame at index `n` directly.
    Single(u64),
    /// Blend source frame `a` and source frame `b` with weight `w` (0 = all `a`, 1 = all `b`).
    Blend(u64, u64, f32),
}

/// Frame rate converter that maps source frame indices to output `FrameRef`s.
pub struct CadenceConverter {
    /// Source frame rate.
    pub source_fps: Rational,
    /// Target frame rate.
    pub target_fps: Rational,
    /// Strategy to use.
    pub strategy: CadenceStrategy,
}

impl CadenceConverter {
    /// Create a new `CadenceConverter`.
    pub fn new(source_fps: Rational, target_fps: Rational, strategy: CadenceStrategy) -> Self {
        Self {
            source_fps,
            target_fps,
            strategy,
        }
    }

    /// Return the number of output frames produced from `source_frames` source frames.
    pub fn convert_frame_count(&self, source_frames: u64) -> u64 {
        match self.strategy {
            CadenceStrategy::Pulldown23 => {
                // 24 source frames → 30 output frames per second.
                // Scale: output = source * (target / source)
                scale_frames(source_frames, self.source_fps, self.target_fps)
            }
            CadenceStrategy::Pulldown32 => {
                scale_frames(source_frames, self.source_fps, self.target_fps)
            }
            CadenceStrategy::Duplicate | CadenceStrategy::Blend => {
                scale_frames(source_frames, self.source_fps, self.target_fps)
            }
            CadenceStrategy::Drop => scale_frames(source_frames, self.source_fps, self.target_fps),
        }
    }

    /// Return the `FrameRef` list for all output frames derived from `source_frames`
    /// source frames.
    pub fn get_output_frame_indices(&self, source_frames: u64) -> Vec<FrameRef> {
        let out_count = self.convert_frame_count(source_frames);
        if out_count == 0 || source_frames == 0 {
            return Vec::new();
        }

        match self.strategy {
            CadenceStrategy::Duplicate => duplicate_indices(out_count, source_frames),
            CadenceStrategy::Drop => drop_indices(out_count, source_frames),
            CadenceStrategy::Blend => blend_indices(out_count, source_frames),
            CadenceStrategy::Pulldown23 => pulldown23_indices(out_count, source_frames),
            CadenceStrategy::Pulldown32 => pulldown32_indices(out_count, source_frames),
        }
    }
}

// -----------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------

/// Scale `source_frames` by `(target / source)` with proper rounding.
fn scale_frames(source_frames: u64, source_fps: Rational, target_fps: Rational) -> u64 {
    if source_fps.den == 0 || source_fps.num == 0 {
        return 0;
    }
    // out = source_frames * target_fps / source_fps
    // = source_frames * (target.num * source.den) / (target.den * source.num)
    let numer = source_frames
        .saturating_mul(target_fps.num)
        .saturating_mul(source_fps.den);
    let denom = target_fps.den.saturating_mul(source_fps.num);
    if denom == 0 {
        return 0;
    }
    (numer + denom / 2) / denom
}

/// Map `out_count` output frames to nearest source frames (nearest-neighbour / duplicate).
fn duplicate_indices(out_count: u64, source_frames: u64) -> Vec<FrameRef> {
    (0..out_count)
        .map(|out_idx| {
            // src_idx = round(out_idx * (source_frames - 1) / (out_count - 1))
            let src_idx = if out_count <= 1 {
                0
            } else {
                let numerator = out_idx * (source_frames - 1);
                let denominator = out_count - 1;
                (numerator + denominator / 2) / denominator
            };
            FrameRef::Single(src_idx.min(source_frames - 1))
        })
        .collect()
}

/// Map `out_count` output frames by dropping frames (for rate decreases).
fn drop_indices(out_count: u64, source_frames: u64) -> Vec<FrameRef> {
    (0..out_count)
        .map(|out_idx| {
            let src_idx = if out_count <= 1 {
                0
            } else {
                let numerator = out_idx * (source_frames - 1);
                let denominator = out_count - 1;
                (numerator + denominator / 2) / denominator
            };
            FrameRef::Single(src_idx.min(source_frames - 1))
        })
        .collect()
}

/// Map `out_count` output frames by blending pairs of source frames.
fn blend_indices(out_count: u64, source_frames: u64) -> Vec<FrameRef> {
    (0..out_count)
        .map(|out_idx| {
            // Exact fractional position in the source timeline.
            let src_pos = if out_count <= 1 {
                0.0f64
            } else {
                out_idx as f64 * (source_frames - 1) as f64 / (out_count - 1) as f64
            };
            let src_floor = src_pos.floor() as u64;
            let frac = (src_pos - src_floor as f64) as f32;

            if src_floor + 1 >= source_frames || frac < 1e-6 {
                FrameRef::Single(src_floor.min(source_frames - 1))
            } else {
                FrameRef::Blend(src_floor, src_floor + 1, frac)
            }
        })
        .collect()
}

/// 2:3 pulldown: each group of 5 source frames (at 24 fps) produces 5 output
/// slots (at 30 fps) using the cadence A A B B C → A AB B BC C.
///
/// More precisely in the 2:3 pattern, source frame N is assigned to output
/// frames as follows (0-indexed within each 5-output cycle):
///   output 0 → src 0       (Single)
///   output 1 → src 0       (Single, duplicate)
///   output 2 → blend(0,1)  (Blend 50/50)
///   output 3 → src 1       (Single)
///   output 4 → src 1       (Single, duplicate)
///
/// This repeats every 4 source frames / 5 output frames.
fn pulldown23_indices(out_count: u64, source_frames: u64) -> Vec<FrameRef> {
    let mut refs = Vec::with_capacity(out_count as usize);
    // Pattern over 5 output frames uses 4 source frames.
    // out_phase 0 → src+0 (Single)
    // out_phase 1 → src+0 (Single dup)
    // out_phase 2 → Blend(src+0, src+1, 0.5)
    // out_phase 3 → src+1 (Single)
    // out_phase 4 → src+1 (Single dup) --- then next cycle starts with src+2
    // Wait — standard 2:3 pulldown maps 4 film frames to 5 video fields (10 fields).
    // We model it as: 4 src → 5 output:
    //   cycle = out_idx / 5, phase = out_idx % 5
    //   src_base = cycle * 4
    //   phase 0 → src_base + 0
    //   phase 1 → src_base + 1
    //   phase 2 → Blend(src_base+1, src_base+2, 0.5)
    //   phase 3 → src_base + 2
    //   phase 4 → src_base + 3

    for out_idx in 0..out_count {
        let cycle = out_idx / 5;
        let phase = out_idx % 5;
        let src_base = cycle * 4;

        let frame_ref = match phase {
            0 => single_clamped(src_base, source_frames),
            1 => single_clamped(src_base + 1, source_frames),
            2 => blend_clamped(src_base + 1, src_base + 2, 0.5, source_frames),
            3 => single_clamped(src_base + 2, source_frames),
            4 => single_clamped(src_base + 3, source_frames),
            _ => unreachable!(),
        };
        refs.push(frame_ref);
    }
    refs
}

/// 3:2 pulldown: alternate pattern — 3 interlaced + 2 progressive per cycle.
/// Modelled here as 5 output frames per 4 source frames with a different phase
/// assignment:
///   phase 0 → src_base + 0
///   phase 1 → Blend(src_base+0, src_base+1, 0.5)
///   phase 2 → src_base + 1
///   phase 3 → src_base + 2
///   phase 4 → Blend(src_base+2, src_base+3, 0.5)
fn pulldown32_indices(out_count: u64, source_frames: u64) -> Vec<FrameRef> {
    let mut refs = Vec::with_capacity(out_count as usize);

    for out_idx in 0..out_count {
        let cycle = out_idx / 5;
        let phase = out_idx % 5;
        let src_base = cycle * 4;

        let frame_ref = match phase {
            0 => single_clamped(src_base, source_frames),
            1 => blend_clamped(src_base, src_base + 1, 0.5, source_frames),
            2 => single_clamped(src_base + 1, source_frames),
            3 => single_clamped(src_base + 2, source_frames),
            4 => blend_clamped(src_base + 2, src_base + 3, 0.5, source_frames),
            _ => unreachable!(),
        };
        refs.push(frame_ref);
    }
    refs
}

#[inline]
fn single_clamped(idx: u64, source_frames: u64) -> FrameRef {
    FrameRef::Single(idx.min(source_frames.saturating_sub(1)))
}

#[inline]
fn blend_clamped(a: u64, b: u64, w: f32, source_frames: u64) -> FrameRef {
    let max_idx = source_frames.saturating_sub(1);
    let ca = a.min(max_idx);
    let cb = b.min(max_idx);
    if ca == cb {
        FrameRef::Single(ca)
    } else {
        FrameRef::Blend(ca, cb, w)
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn fps(n: u64) -> Rational {
        Rational::new(n, 1)
    }

    // 1. convert_frame_count: 24→30, 24 source frames → 30 output frames
    #[test]
    fn test_24_to_30_frame_count() {
        let conv = CadenceConverter::new(fps(24), fps(30), CadenceStrategy::Duplicate);
        assert_eq!(conv.convert_frame_count(24), 30);
    }

    // 2. convert_frame_count: 25→30, 25 source frames → 30 output frames
    #[test]
    fn test_25_to_30_frame_count() {
        let conv = CadenceConverter::new(fps(25), fps(30), CadenceStrategy::Duplicate);
        assert_eq!(conv.convert_frame_count(25), 30);
    }

    // 3. convert_frame_count: 50→60, 50 source → 60 output
    #[test]
    fn test_50_to_60_frame_count() {
        let conv = CadenceConverter::new(fps(50), fps(60), CadenceStrategy::Blend);
        assert_eq!(conv.convert_frame_count(50), 60);
    }

    // 4. convert_frame_count: Pulldown23 24→30
    #[test]
    fn test_pulldown23_frame_count() {
        let conv = CadenceConverter::new(fps(24), fps(30), CadenceStrategy::Pulldown23);
        assert_eq!(conv.convert_frame_count(24), 30);
    }

    // 5. convert_frame_count: Drop 60→24
    #[test]
    fn test_drop_60_to_24() {
        let conv = CadenceConverter::new(fps(60), fps(24), CadenceStrategy::Drop);
        assert_eq!(conv.convert_frame_count(60), 24);
    }

    // 6. Duplicate: indices length matches convert_frame_count
    #[test]
    fn test_duplicate_index_count_matches() {
        let conv = CadenceConverter::new(fps(24), fps(30), CadenceStrategy::Duplicate);
        let indices = conv.get_output_frame_indices(24);
        assert_eq!(indices.len(), conv.convert_frame_count(24) as usize);
    }

    // 7. Blend: all FrameRef indices within valid range
    #[test]
    fn test_blend_indices_in_range() {
        let conv = CadenceConverter::new(fps(25), fps(30), CadenceStrategy::Blend);
        let source = 25u64;
        let indices = conv.get_output_frame_indices(source);
        for r in &indices {
            match r {
                FrameRef::Single(i) => assert!(*i < source, "index {i} out of range"),
                FrameRef::Blend(a, b, w) => {
                    assert!(*a < source);
                    assert!(*b < source);
                    assert!(*w >= 0.0 && *w <= 1.0);
                }
            }
        }
    }

    // 8. Pulldown23: every 4 source → 5 output in 24-frame sequence
    #[test]
    fn test_pulldown23_ratio() {
        let conv = CadenceConverter::new(fps(24), fps(30), CadenceStrategy::Pulldown23);
        let indices = conv.get_output_frame_indices(24);
        assert_eq!(indices.len(), 30);
    }

    // 9. Pulldown32: 24 source → 30 output
    #[test]
    fn test_pulldown32_ratio() {
        let conv = CadenceConverter::new(fps(24), fps(30), CadenceStrategy::Pulldown32);
        let indices = conv.get_output_frame_indices(24);
        assert_eq!(indices.len(), 30);
    }

    // 10. Drop: first and last output frame are within source range
    #[test]
    fn test_drop_first_last_in_range() {
        let conv = CadenceConverter::new(fps(60), fps(24), CadenceStrategy::Drop);
        let source = 60u64;
        let indices = conv.get_output_frame_indices(source);
        assert!(!indices.is_empty());
        if let FrameRef::Single(first) = &indices[0] {
            assert!(*first < source);
        }
        if let FrameRef::Single(last) = indices.last().expect("at least one") {
            assert!(*last < source);
        }
    }

    // 11. Zero source frames → empty output
    #[test]
    fn test_zero_source_frames_empty() {
        let conv = CadenceConverter::new(fps(24), fps(30), CadenceStrategy::Blend);
        let indices = conv.get_output_frame_indices(0);
        assert!(indices.is_empty());
    }

    // 12. Rational::as_f64 correct
    #[test]
    fn test_rational_as_f64() {
        let r = Rational::new(30, 1);
        assert!((r.as_f64() - 30.0).abs() < 1e-9);
        let r2 = Rational::new(2997, 100);
        assert!((r2.as_f64() - 29.97).abs() < 0.001);
    }

    // 13. Blend weight for 50→60: middle output frame has fractional weight
    #[test]
    fn test_blend_50_to_60_has_blend_frames() {
        let conv = CadenceConverter::new(fps(50), fps(60), CadenceStrategy::Blend);
        let indices = conv.get_output_frame_indices(50);
        let has_blend = indices
            .iter()
            .any(|r| matches!(r, FrameRef::Blend(_, _, _)));
        assert!(has_blend, "50→60 blend should produce some blended frames");
    }

    // 14. Duplicate: first output is always Single(0)
    #[test]
    fn test_duplicate_first_is_single_zero() {
        let conv = CadenceConverter::new(fps(24), fps(30), CadenceStrategy::Duplicate);
        let indices = conv.get_output_frame_indices(24);
        assert_eq!(indices[0], FrameRef::Single(0));
    }

    // 15. convert_frame_count: identity (same rate, same count)
    #[test]
    fn test_same_rate_identity() {
        let conv = CadenceConverter::new(fps(30), fps(30), CadenceStrategy::Blend);
        assert_eq!(conv.convert_frame_count(30), 30);
    }

    // 16. Rational Display
    #[test]
    fn test_rational_display() {
        assert_eq!(Rational::new(30, 1).to_string(), "30");
        assert_eq!(Rational::new(24000, 1001).to_string(), "24000/1001");
    }
}
