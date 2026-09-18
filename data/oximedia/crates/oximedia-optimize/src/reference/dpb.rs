//! Decoded Picture Buffer optimization.

/// DPB statistics.
#[derive(Debug, Clone, Copy)]
pub struct DpbStats {
    /// Number of frames in DPB.
    pub frame_count: usize,
    /// Maximum DPB size.
    pub max_size: usize,
    /// Average utilization (0-1).
    pub utilization: f64,
}

impl Default for DpbStats {
    fn default() -> Self {
        Self {
            frame_count: 0,
            max_size: 8,
            utilization: 0.0,
        }
    }
}

/// DPB optimizer.
pub struct DpbOptimizer {
    max_dpb_size: usize,
    reorder_depth: usize,
}

impl Default for DpbOptimizer {
    fn default() -> Self {
        Self::new(8, 4)
    }
}

impl DpbOptimizer {
    /// Creates a new DPB optimizer.
    #[must_use]
    pub const fn new(max_dpb_size: usize, reorder_depth: usize) -> Self {
        Self {
            max_dpb_size,
            reorder_depth,
        }
    }

    /// Calculates optimal DPB size for encoding parameters.
    #[must_use]
    pub fn calculate_dpb_size(&self, gop_size: usize, num_ref_frames: usize) -> usize {
        let min_size = num_ref_frames + self.reorder_depth + 1;
        let suggested_size = (gop_size / 2).max(min_size);
        suggested_size.min(self.max_dpb_size)
    }

    /// Manages DPB eviction policy.
    #[allow(dead_code)]
    #[must_use]
    pub fn select_eviction_candidate(&self, frames: &[DpbFrame]) -> Option<usize> {
        if frames.is_empty() {
            return None;
        }

        // Find frame with lowest reference count that's not currently in use
        let mut best_idx = 0;
        let mut lowest_score = f64::MAX;

        for (idx, frame) in frames.iter().enumerate() {
            if !frame.in_use {
                let score = self.calculate_eviction_score(frame);
                if score < lowest_score {
                    lowest_score = score;
                    best_idx = idx;
                }
            }
        }

        if lowest_score < f64::MAX {
            Some(best_idx)
        } else {
            None
        }
    }

    fn calculate_eviction_score(&self, frame: &DpbFrame) -> f64 {
        // Lower score = more likely to evict
        let ref_score = frame.reference_count as f64 * 100.0;
        let age_score = frame.age as f64;
        ref_score - age_score // Prefer older frames with fewer references
    }

    /// Gets DPB statistics.
    #[must_use]
    pub fn get_stats(&self, current_frame_count: usize) -> DpbStats {
        DpbStats {
            frame_count: current_frame_count,
            max_size: self.max_dpb_size,
            utilization: current_frame_count as f64 / self.max_dpb_size as f64,
        }
    }

    /// Checks if DPB is full.
    #[must_use]
    pub fn is_full(&self, current_count: usize) -> bool {
        current_count >= self.max_dpb_size
    }
}

/// Frame in DPB.
#[derive(Debug, Clone)]
pub struct DpbFrame {
    /// Frame index.
    pub frame_idx: usize,
    /// Number of times referenced.
    pub reference_count: usize,
    /// Age in frames.
    pub age: usize,
    /// Whether frame is currently in use.
    pub in_use: bool,
}

impl DpbFrame {
    /// Creates a new DPB frame.
    #[must_use]
    pub const fn new(frame_idx: usize) -> Self {
        Self {
            frame_idx,
            reference_count: 0,
            age: 0,
            in_use: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dpb_optimizer_creation() {
        let optimizer = DpbOptimizer::default();
        assert_eq!(optimizer.max_dpb_size, 8);
        assert_eq!(optimizer.reorder_depth, 4);
    }

    #[test]
    fn test_calculate_dpb_size() {
        let optimizer = DpbOptimizer::default();
        let size = optimizer.calculate_dpb_size(16, 3);
        assert!(size >= 3); // At least num_ref_frames
        assert!(size <= 8); // At most max_dpb_size
    }

    #[test]
    fn test_is_full() {
        let optimizer = DpbOptimizer::default();
        assert!(!optimizer.is_full(5));
        assert!(optimizer.is_full(8));
        assert!(optimizer.is_full(10));
    }

    #[test]
    fn test_dpb_stats() {
        let optimizer = DpbOptimizer::default();
        let stats = optimizer.get_stats(4);
        assert_eq!(stats.frame_count, 4);
        assert_eq!(stats.max_size, 8);
        assert_eq!(stats.utilization, 0.5);
    }

    #[test]
    fn test_eviction_candidate_selection() {
        let optimizer = DpbOptimizer::default();
        let frames = vec![
            DpbFrame {
                frame_idx: 0,
                reference_count: 5,
                age: 10,
                in_use: false,
            },
            DpbFrame {
                frame_idx: 1,
                reference_count: 1,
                age: 5,
                in_use: false,
            },
        ];

        let candidate = optimizer.select_eviction_candidate(&frames);
        assert!(candidate.is_some());
    }

    #[test]
    fn test_dpb_frame_creation() {
        let frame = DpbFrame::new(42);
        assert_eq!(frame.frame_idx, 42);
        assert_eq!(frame.reference_count, 0);
        assert_eq!(frame.age, 0);
        assert!(!frame.in_use);
    }
}
