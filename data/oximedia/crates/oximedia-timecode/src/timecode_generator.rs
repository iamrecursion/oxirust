#![allow(dead_code)]
//! Free-running timecode generator with configurable start time and frame rate.
//!
//! [`TimecodeGenerator`] provides an incrementing source of SMPTE timecodes
//! suitable for free-running playout, offline conforming, or real-time capture
//! scenarios.  It correctly handles drop-frame minute boundaries and midnight
//! roll-over.

use crate::{frame_rate_from_info, FrameRate, Timecode, TimecodeError};

/// A free-running timecode generator.
///
/// The generator owns a current position expressed as a [`Timecode`] and
/// advances it one frame at a time each time [`next`](TimecodeGenerator::next)
/// is called.  The generator can be paused (`running = false`), reset to an
/// arbitrary position, seeked, and fast-forwarded/rewound by an arbitrary
/// number of frames.
#[derive(Debug, Clone)]
pub struct TimecodeGenerator {
    /// Current timecode position (the value that will be returned by the next
    /// call to [`next`](TimecodeGenerator::next)).
    current: Timecode,
    /// Whether the generator is advancing on each call to `next`.
    pub running: bool,
}

impl TimecodeGenerator {
    /// Create a new generator starting at `start` with the given `frame_rate`.
    ///
    /// The generator starts in the **running** state.
    ///
    /// # Errors
    ///
    /// Forwards any error from [`Timecode::new`] if `start` describes an
    /// invalid timecode.
    pub fn new(start: Timecode) -> Self {
        Self {
            current: start,
            running: true,
        }
    }

    /// Create a generator starting at midnight (`00:00:00:00`) for the given
    /// frame rate.
    ///
    /// # Errors
    ///
    /// Returns an error if `frame_rate` cannot produce a valid midnight
    /// timecode (should never occur for well-defined frame rates).
    pub fn at_midnight(frame_rate: FrameRate) -> Result<Self, TimecodeError> {
        let tc = Timecode::new(0, 0, 0, 0, frame_rate)?;
        Ok(Self::new(tc))
    }

    /// Advance the generator by one frame (if running) and return the
    /// timecode **before** the increment.
    ///
    /// If `running` is `false` the current position is returned without
    /// advancing.
    ///
    /// Midnight roll-over is handled transparently; the generator continues
    /// from `00:00:00:00` after `23:59:59:FF`.
    pub fn next(&mut self) -> Timecode {
        let out = self.current;
        if self.running {
            // Silently ignore increment errors (they should not occur for valid TC)
            let _ = self.current.increment();
        }
        out
    }

    /// Return the current timecode position without advancing.
    pub fn peek(&self) -> Timecode {
        self.current
    }

    /// Reset the generator to midnight (`00:00:00:00`) for its current frame
    /// rate.
    ///
    /// # Errors
    ///
    /// Returns an error if building the midnight timecode fails.
    pub fn reset(&mut self) -> Result<(), TimecodeError> {
        let rate = frame_rate_from_info(&self.current.frame_rate);
        self.current = Timecode::new(0, 0, 0, 0, rate)?;
        Ok(())
    }

    /// Seek to (reset to) an arbitrary timecode.
    ///
    /// The generator adopts the frame rate embedded in `tc`.
    pub fn reset_to(&mut self, tc: Timecode) {
        self.current = tc;
    }

    /// Seek to a specific timecode (alias of [`reset_to`](Self::reset_to)).
    pub fn seek(&mut self, tc: Timecode) {
        self.current = tc;
    }

    /// Skip forward (`n > 0`) or backward (`n < 0`) by `n` frames.
    ///
    /// The operation wraps around midnight boundaries correctly using the
    /// modular arithmetic built into [`Timecode::from_frames`].
    ///
    /// # Errors
    ///
    /// Returns an error if the resulting frame count cannot be converted back
    /// to a valid timecode.
    pub fn skip_frames(&mut self, n: i64) -> Result<(), TimecodeError> {
        let rate = frame_rate_from_info(&self.current.frame_rate);
        let fps = self.current.frame_rate.fps as i64;
        let frames_per_day = fps * 86_400;

        let current_frames = self.current.to_frames() as i64;
        // Modular arithmetic to handle both forward and backward wrapping
        let new_frames = if frames_per_day > 0 {
            ((current_frames + n).rem_euclid(frames_per_day)) as u64
        } else {
            (current_frames + n).max(0) as u64
        };

        self.current = Timecode::from_frames(new_frames, rate)?;
        Ok(())
    }

    /// Start the generator (set `running = true`).
    pub fn start(&mut self) {
        self.running = true;
    }

    /// Stop the generator (set `running = false`).
    pub fn stop(&mut self) {
        self.running = false;
    }

    /// Return the frame rate of the current timecode.
    pub fn frame_rate(&self) -> FrameRate {
        frame_rate_from_info(&self.current.frame_rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gen_25() -> TimecodeGenerator {
        TimecodeGenerator::at_midnight(FrameRate::Fps25).expect("midnight ok")
    }

    #[test]
    fn test_generator_starts_at_midnight() {
        let gen = make_gen_25();
        let tc = gen.peek();
        assert_eq!(tc.hours, 0);
        assert_eq!(tc.minutes, 0);
        assert_eq!(tc.seconds, 0);
        assert_eq!(tc.frames, 0);
    }

    #[test]
    fn test_next_increments() {
        let mut gen = make_gen_25();
        let tc0 = gen.next();
        let tc1 = gen.next();
        assert_eq!(tc0.to_frames() + 1, tc1.to_frames());
    }

    #[test]
    fn test_next_returns_current_before_increment() {
        let mut gen = make_gen_25();
        let peek = gen.peek();
        let got = gen.next();
        assert_eq!(peek, got);
    }

    #[test]
    fn test_stop_freezes_position() {
        let mut gen = make_gen_25();
        gen.stop();
        let a = gen.next();
        let b = gen.next();
        assert_eq!(a, b);
    }

    #[test]
    fn test_start_resumes_after_stop() {
        let mut gen = make_gen_25();
        gen.stop();
        let _ = gen.next();
        gen.start();
        let before = gen.peek().to_frames();
        let _ = gen.next();
        let after = gen.peek().to_frames();
        assert_eq!(after, before + 1);
    }

    #[test]
    fn test_reset_to_midnight() {
        let mut gen = make_gen_25();
        // Advance a few frames
        for _ in 0..100 {
            let _ = gen.next();
        }
        gen.reset().expect("reset ok");
        assert_eq!(gen.peek().to_frames(), 0);
    }

    #[test]
    fn test_reset_to_arbitrary_tc() {
        let mut gen = make_gen_25();
        let target = Timecode::new(12, 34, 56, 10, FrameRate::Fps25).expect("valid");
        gen.reset_to(target);
        assert_eq!(gen.peek(), target);
    }

    #[test]
    fn test_seek_alias() {
        let mut gen = make_gen_25();
        let target = Timecode::new(1, 0, 0, 0, FrameRate::Fps25).expect("valid");
        gen.seek(target);
        assert_eq!(gen.peek(), target);
    }

    #[test]
    fn test_skip_forward() {
        let mut gen = make_gen_25();
        gen.skip_frames(100).expect("skip ok");
        assert_eq!(gen.peek().to_frames(), 100);
    }

    #[test]
    fn test_skip_backward_wraps() {
        let mut gen = make_gen_25();
        // Skipping backward from midnight wraps to end of day
        gen.skip_frames(-1).expect("skip ok");
        let frames_per_day = 25u64 * 86_400;
        assert_eq!(gen.peek().to_frames(), frames_per_day - 1);
    }

    #[test]
    fn test_skip_forward_wraps_midnight() {
        let mut gen = make_gen_25();
        let frames_per_day = 25i64 * 86_400;
        // skip exactly one full day forward — should land back at midnight
        gen.skip_frames(frames_per_day).expect("skip ok");
        assert_eq!(gen.peek().to_frames(), 0);
    }

    #[test]
    fn test_drop_frame_generator_next_at_minute_boundary() {
        // Seek to just before 1-minute mark in 29.97 DF and verify next() skips frames 0+1
        let start = Timecode::new(0, 0, 59, 29, FrameRate::Fps2997DF).expect("valid");
        let mut gen = TimecodeGenerator::new(start);
        let tc = gen.next(); // returns 00:00:59:29
        assert_eq!(tc.frames, 29);
        let next = gen.next(); // should advance to 00:01:00:02
        assert_eq!(next.minutes, 1);
        assert_eq!(next.seconds, 0);
        assert_eq!(next.frames, 2);
    }
}
