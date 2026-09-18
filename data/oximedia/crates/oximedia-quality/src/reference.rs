//! Reference frame management for quality assessment.
//!
//! Manages reference frames for full-reference quality metrics,
//! including frame alignment and temporal synchronization.

use crate::Frame;
use std::collections::VecDeque;

/// Reference frame manager for temporal quality assessment.
pub struct ReferenceManager {
    /// Buffer of reference frames
    buffer: VecDeque<Frame>,
    /// Maximum buffer size
    max_buffer_size: usize,
    /// Current frame index
    current_index: usize,
}

impl ReferenceManager {
    /// Creates a new reference manager with default buffer size.
    #[must_use]
    pub fn new() -> Self {
        Self::with_buffer_size(30) // Default: 1 second at 30fps
    }

    /// Creates a reference manager with custom buffer size.
    #[must_use]
    pub fn with_buffer_size(size: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(size),
            max_buffer_size: size,
            current_index: 0,
        }
    }

    /// Adds a reference frame to the buffer.
    pub fn add_frame(&mut self, frame: Frame) {
        if self.buffer.len() >= self.max_buffer_size {
            self.buffer.pop_front();
        }
        self.buffer.push_back(frame);
        self.current_index += 1;
    }

    /// Gets the most recent reference frame.
    #[must_use]
    pub fn current_frame(&self) -> Option<&Frame> {
        self.buffer.back()
    }

    /// Gets a reference frame at a specific offset from current.
    ///
    /// Offset 0 = current frame, -1 = previous frame, etc.
    #[must_use]
    pub fn frame_at_offset(&self, offset: i32) -> Option<&Frame> {
        if offset == 0 {
            return self.buffer.back();
        }

        if offset < 0 {
            let abs_offset = offset.unsigned_abs() as usize;
            if abs_offset < self.buffer.len() {
                self.buffer.get(self.buffer.len() - 1 - abs_offset)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Gets the number of buffered frames.
    #[must_use]
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    /// Gets the current frame index.
    #[must_use]
    pub fn current_index(&self) -> usize {
        self.current_index
    }

    /// Clears the buffer.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.current_index = 0;
    }

    /// Finds the best matching frame for the given distorted frame.
    ///
    /// This can be used for temporal alignment when reference and distorted
    /// videos are not perfectly synchronized.
    #[must_use]
    pub fn find_best_match(&self, distorted: &Frame) -> Option<&Frame> {
        if self.buffer.is_empty() {
            return None;
        }

        // Simple matching based on mean absolute difference
        let mut best_frame = None;
        let mut best_score = f64::INFINITY;

        for frame in &self.buffer {
            if frame.width != distorted.width || frame.height != distorted.height {
                continue;
            }

            let score = self.compute_mad(&frame.planes[0], &distorted.planes[0]);
            if score < best_score {
                best_score = score;
                best_frame = Some(frame);
            }
        }

        best_frame
    }

    /// Computes mean absolute difference between two planes.
    fn compute_mad(&self, plane1: &[u8], plane2: &[u8]) -> f64 {
        if plane1.len() != plane2.len() {
            return f64::INFINITY;
        }

        let sum: u64 = plane1
            .iter()
            .zip(plane2.iter())
            .map(|(a, b)| u64::from((i32::from(*a) - i32::from(*b)).unsigned_abs()))
            .sum();

        sum as f64 / plane1.len() as f64
    }
}

impl Default for ReferenceManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::PixelFormat;

    fn create_test_frame(width: usize, height: usize, value: u8) -> Frame {
        let mut frame =
            Frame::new(width, height, PixelFormat::Yuv420p).expect("should succeed in test");
        frame.planes[0].fill(value);
        frame
    }

    #[test]
    fn test_add_and_get_frame() {
        let mut manager = ReferenceManager::new();
        let frame = create_test_frame(64, 64, 128);

        manager.add_frame(frame.clone());
        assert_eq!(manager.buffer_len(), 1);

        let current = manager.current_frame().expect("should succeed in test");
        assert_eq!(current.width, 64);
        assert_eq!(current.height, 64);
    }

    #[test]
    fn test_buffer_overflow() {
        let mut manager = ReferenceManager::with_buffer_size(3);

        for i in 0..5 {
            let frame = create_test_frame(64, 64, i as u8);
            manager.add_frame(frame);
        }

        // Should only keep last 3 frames
        assert_eq!(manager.buffer_len(), 3);
        assert_eq!(manager.current_index(), 5);
    }

    #[test]
    fn test_frame_at_offset() {
        let mut manager = ReferenceManager::new();

        for i in 0..5 {
            let frame = create_test_frame(64, 64, i as u8);
            manager.add_frame(frame);
        }

        // Current frame (offset 0)
        let current = manager.frame_at_offset(0).expect("should succeed in test");
        assert_eq!(current.planes[0][0], 4);

        // Previous frame (offset -1)
        let prev = manager.frame_at_offset(-1).expect("should succeed in test");
        assert_eq!(prev.planes[0][0], 3);

        // Two frames back (offset -2)
        let prev2 = manager.frame_at_offset(-2).expect("should succeed in test");
        assert_eq!(prev2.planes[0][0], 2);
    }

    #[test]
    fn test_clear() {
        let mut manager = ReferenceManager::new();

        for i in 0..5 {
            let frame = create_test_frame(64, 64, i as u8);
            manager.add_frame(frame);
        }

        assert_eq!(manager.buffer_len(), 5);

        manager.clear();
        assert_eq!(manager.buffer_len(), 0);
        assert_eq!(manager.current_index(), 0);
    }

    #[test]
    fn test_find_best_match() {
        let mut manager = ReferenceManager::new();

        let frame1 = create_test_frame(64, 64, 100);
        let frame2 = create_test_frame(64, 64, 120);
        let frame3 = create_test_frame(64, 64, 140);

        manager.add_frame(frame1);
        manager.add_frame(frame2);
        manager.add_frame(frame3);

        let distorted = create_test_frame(64, 64, 122);

        let best_match = manager
            .find_best_match(&distorted)
            .expect("should succeed in test");
        // Should match frame2 (120) as closest
        assert_eq!(best_match.planes[0][0], 120);
    }

    #[test]
    fn test_compute_mad() {
        let manager = ReferenceManager::new();

        let plane1 = vec![100u8; 100];
        let plane2 = vec![110u8; 100];

        let mad = manager.compute_mad(&plane1, &plane2);
        assert!((mad - 10.0).abs() < 0.01);
    }
}
