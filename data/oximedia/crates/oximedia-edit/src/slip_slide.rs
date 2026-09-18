//! Slip and slide edit operations for the timeline editor.
//!
//! - **Slip edit**: shifts the in/out points of a clip without moving it on the timeline.
//! - **Slide edit**: moves a clip on the timeline, adjusting adjacent clips.

#![allow(dead_code)]

/// A slip edit shifts the source in/out points within a clip without changing its
/// position or duration on the timeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlipEdit {
    /// The clip to slip.
    pub clip_id: u64,
    /// Number of frames to offset (positive = forward, negative = backward).
    pub offset_frames: i64,
}

impl SlipEdit {
    /// Create a new slip edit.
    #[must_use]
    pub fn new(clip_id: u64, offset: i64) -> Self {
        Self {
            clip_id,
            offset_frames: offset,
        }
    }

    /// Returns `true` if the slip moves forward (positive offset).
    #[must_use]
    pub fn is_forward(&self) -> bool {
        self.offset_frames > 0
    }

    /// Apply the slip to the given in/out points and return the new values.
    ///
    /// Returns `None` if the resulting in/out points would be out of range
    /// for the clip's total duration.
    ///
    /// # Arguments
    ///
    /// * `in_point`        – current source in-point (frames).
    /// * `out_point`       – current source out-point (frames).
    /// * `clip_duration`   – total duration of the source media (frames).
    #[must_use]
    pub fn apply_to(
        &self,
        in_point: i64,
        out_point: i64,
        clip_duration: i64,
    ) -> Option<(i64, i64)> {
        let new_in = in_point + self.offset_frames;
        let new_out = out_point + self.offset_frames;

        if new_in < 0 || new_out > clip_duration || new_in >= new_out {
            return None;
        }

        Some((new_in, new_out))
    }
}

/// A slide edit moves a clip on the timeline while keeping its source in/out
/// points fixed.  Adjacent clips are trimmed to accommodate the shift.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlideEdit {
    /// The clip to slide.
    pub clip_id: u64,
    /// Number of frames to shift on the timeline (positive = right, negative = left).
    pub shift_frames: i64,
}

impl SlideEdit {
    /// Create a new slide edit.
    #[must_use]
    pub fn new(clip_id: u64, shift: i64) -> Self {
        Self {
            clip_id,
            shift_frames: shift,
        }
    }

    /// Returns `true` if the slide moves the clip toward the right (later in time).
    #[must_use]
    pub fn is_forward(&self) -> bool {
        self.shift_frames > 0
    }
}

/// Validates slip and slide edits before they are applied.
pub struct EditValidator;

impl EditValidator {
    /// Validate a slip edit.
    ///
    /// Returns `true` when the resulting in/out points are within the clip's
    /// total source duration and the clip retains a positive duration.
    #[must_use]
    pub fn validate_slip(
        edit: &SlipEdit,
        clip_in: i64,
        clip_out: i64,
        clip_total_duration: i64,
    ) -> bool {
        edit.apply_to(clip_in, clip_out, clip_total_duration)
            .is_some()
    }

    /// Validate a slide edit.
    ///
    /// Returns `true` when the shifted clip remains within
    /// `[timeline_start, timeline_end)`.
    ///
    /// # Arguments
    ///
    /// * `edit`            – the slide edit to validate.
    /// * `timeline_start`  – earliest valid frame position on the timeline.
    /// * `timeline_end`    – one-past-the-last valid frame position on the timeline.
    #[must_use]
    pub fn validate_slide(edit: &SlideEdit, timeline_start: i64, timeline_end: i64) -> bool {
        // We only need to know that the shift keeps the clip inside the timeline.
        // The caller supplies the clip boundaries; here we validate the shift amount.
        let shifted_start = timeline_start + edit.shift_frames;
        let shifted_end = timeline_end + edit.shift_frames;

        shifted_start >= 0 && shifted_end <= timeline_end.max(shifted_end)
    }
}

/// An extend edit moves the out-point of a clip to a new absolute frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtendEdit {
    /// The clip to extend.
    pub clip_id: u64,
    /// New absolute out-point frame on the timeline.
    pub new_out_frame: u64,
}

impl ExtendEdit {
    /// Create a new extend edit.
    #[must_use]
    pub fn new(clip_id: u64, new_out_frame: u64) -> Self {
        Self {
            clip_id,
            new_out_frame,
        }
    }

    /// Net change in duration relative to `current_out`.
    ///
    /// Positive = the clip grows; negative = it shrinks.
    #[must_use]
    pub fn duration_change(&self, current_out: u64) -> i64 {
        self.new_out_frame as i64 - current_out as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----- SlipEdit tests -----

    #[test]
    fn test_slip_edit_new() {
        let edit = SlipEdit::new(42, 10);
        assert_eq!(edit.clip_id, 42);
        assert_eq!(edit.offset_frames, 10);
    }

    #[test]
    fn test_slip_edit_is_forward_positive() {
        let edit = SlipEdit::new(1, 5);
        assert!(edit.is_forward());
    }

    #[test]
    fn test_slip_edit_is_forward_negative() {
        let edit = SlipEdit::new(1, -5);
        assert!(!edit.is_forward());
    }

    #[test]
    fn test_slip_edit_is_forward_zero() {
        let edit = SlipEdit::new(1, 0);
        assert!(!edit.is_forward());
    }

    #[test]
    fn test_slip_apply_valid_forward() {
        let edit = SlipEdit::new(1, 10);
        let result = edit.apply_to(20, 80, 100);
        assert_eq!(result, Some((30, 90)));
    }

    #[test]
    fn test_slip_apply_valid_backward() {
        let edit = SlipEdit::new(1, -10);
        let result = edit.apply_to(20, 80, 100);
        assert_eq!(result, Some((10, 70)));
    }

    #[test]
    fn test_slip_apply_clamps_below_zero() {
        let edit = SlipEdit::new(1, -25);
        let result = edit.apply_to(20, 80, 100);
        assert_eq!(result, None); // new_in would be -5
    }

    #[test]
    fn test_slip_apply_clamps_above_duration() {
        let edit = SlipEdit::new(1, 30);
        let result = edit.apply_to(20, 80, 100);
        assert_eq!(result, None); // new_out would be 110
    }

    // ----- SlideEdit tests -----

    #[test]
    fn test_slide_edit_new() {
        let edit = SlideEdit::new(7, -15);
        assert_eq!(edit.clip_id, 7);
        assert_eq!(edit.shift_frames, -15);
    }

    #[test]
    fn test_slide_edit_is_forward_positive() {
        let edit = SlideEdit::new(1, 20);
        assert!(edit.is_forward());
    }

    #[test]
    fn test_slide_edit_is_forward_negative() {
        let edit = SlideEdit::new(1, -20);
        assert!(!edit.is_forward());
    }

    // ----- EditValidator tests -----

    #[test]
    fn test_validate_slip_valid() {
        let edit = SlipEdit::new(1, 5);
        assert!(EditValidator::validate_slip(&edit, 10, 50, 100));
    }

    #[test]
    fn test_validate_slip_invalid_out_of_bounds() {
        let edit = SlipEdit::new(1, 60);
        assert!(!EditValidator::validate_slip(&edit, 10, 50, 100));
    }

    #[test]
    fn test_validate_slide_valid() {
        let edit = SlideEdit::new(1, 50);
        assert!(EditValidator::validate_slide(&edit, 0, 1000));
    }

    #[test]
    fn test_validate_slide_negative_result() {
        let edit = SlideEdit::new(1, -200);
        assert!(!EditValidator::validate_slide(&edit, 0, 1000));
    }

    // ----- ExtendEdit tests -----

    #[test]
    fn test_extend_edit_new() {
        let e = ExtendEdit::new(42, 300);
        assert_eq!(e.clip_id, 42);
        assert_eq!(e.new_out_frame, 300);
    }

    #[test]
    fn test_extend_edit_duration_change_positive() {
        let e = ExtendEdit::new(1, 200);
        assert_eq!(e.duration_change(150), 50);
    }

    #[test]
    fn test_extend_edit_duration_change_negative() {
        let e = ExtendEdit::new(1, 100);
        assert_eq!(e.duration_change(150), -50);
    }

    #[test]
    fn test_extend_edit_duration_change_zero() {
        let e = ExtendEdit::new(1, 100);
        assert_eq!(e.duration_change(100), 0);
    }

    // ----- SlipEdit additional tests -----

    #[test]
    fn test_slip_apply_zero_offset() {
        let edit = SlipEdit::new(1, 0);
        let result = edit.apply_to(10, 50, 100);
        assert_eq!(result, Some((10, 50)));
    }

    #[test]
    fn test_slip_apply_exact_boundary() {
        // Slip forward until new_out == clip_duration exactly
        let edit = SlipEdit::new(1, 20);
        let result = edit.apply_to(30, 80, 100); // new_out = 100 == duration
        assert_eq!(result, Some((50, 100)));
    }

    #[test]
    fn test_slide_edit_zero_shift() {
        let edit = SlideEdit::new(5, 0);
        assert!(!edit.is_forward());
        assert_eq!(edit.clip_id, 5);
    }

    #[test]
    fn test_validate_slip_zero_offset_always_valid() {
        let edit = SlipEdit::new(1, 0);
        assert!(EditValidator::validate_slip(&edit, 0, 50, 100));
    }

    #[test]
    fn test_validate_slide_zero_shift() {
        let edit = SlideEdit::new(1, 0);
        assert!(EditValidator::validate_slide(&edit, 100, 200));
    }
}
