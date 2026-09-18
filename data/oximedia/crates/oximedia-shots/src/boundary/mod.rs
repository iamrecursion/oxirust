//! Shot boundary detection algorithms.

/// Method used for boundary detection.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BoundaryMethod {
    /// Mean absolute pixel difference between frames.
    PixelDifference,
    /// Chi-squared histogram distance.
    Histogram,
    /// Edge change ratio (union vs intersection of edges).
    EdgeChange,
    /// Motion vector based detection.
    MotionVector,
}

/// A frame difference measurement.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub struct FrameDiff {
    /// Frame index (of the second frame in the pair).
    pub frame_idx: u64,
    /// Computed difference score.
    pub score: f32,
    /// Whether this is classified as a boundary.
    pub is_boundary: bool,
}

impl FrameDiff {
    /// Create a new `FrameDiff`.
    #[allow(dead_code)]
    #[must_use]
    pub fn new(frame_idx: u64, score: f32, is_boundary: bool) -> Self {
        Self {
            frame_idx,
            score,
            is_boundary,
        }
    }
}

/// Detects shot boundaries via mean absolute pixel difference.
#[allow(dead_code)]
pub struct PixelDiffDetector;

impl PixelDiffDetector {
    /// Compute the mean absolute difference between two frames.
    ///
    /// Frames must have the same length. Returns 0.0 if empty.
    #[allow(dead_code)]
    #[must_use]
    pub fn compute_diff(frame_a: &[f32], frame_b: &[f32]) -> f32 {
        let len = frame_a.len().min(frame_b.len());
        if len == 0 {
            return 0.0;
        }
        let sum: f32 = frame_a
            .iter()
            .zip(frame_b.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        sum / len as f32
    }
}

/// Detects shot boundaries via chi-squared histogram distance.
#[allow(dead_code)]
pub struct HistogramDiff;

impl HistogramDiff {
    /// Compute chi-squared distance between two 256-bin histograms.
    ///
    /// `chi^2 = sum((A_i - B_i)^2 / (A_i + B_i))` (skips bins where both are 0).
    #[allow(dead_code)]
    #[must_use]
    pub fn compute(hist_a: &[u32; 256], hist_b: &[u32; 256]) -> f32 {
        let mut dist = 0.0_f32;
        for (a, b) in hist_a.iter().zip(hist_b.iter()) {
            let sum = a + b;
            if sum > 0 {
                let diff = (*a as f32) - (*b as f32);
                dist += diff * diff / sum as f32;
            }
        }
        dist
    }
}

/// Detects shot boundaries via edge change ratio.
#[allow(dead_code)]
pub struct EdgeChangeDiff;

impl EdgeChangeDiff {
    /// Compute edge change ratio: `|A∪B - A∩B| / |A∪B|`.
    ///
    /// Values of edges are treated as binary (> 0.5 = edge).
    /// Returns 0.0 if both frames have no edges.
    #[allow(dead_code)]
    #[must_use]
    pub fn compute(edges_a: &[f32], edges_b: &[f32]) -> f32 {
        let len = edges_a.len().min(edges_b.len());
        if len == 0 {
            return 0.0;
        }

        let mut union_count = 0u32;
        let mut intersection_count = 0u32;

        for (a, b) in edges_a.iter().zip(edges_b.iter()) {
            let ea = *a > 0.5;
            let eb = *b > 0.5;
            if ea || eb {
                union_count += 1;
            }
            if ea && eb {
                intersection_count += 1;
            }
        }

        if union_count == 0 {
            return 0.0;
        }

        let symmetric_diff = union_count - intersection_count;
        symmetric_diff as f32 / union_count as f32
    }
}

/// Adaptive threshold using Welford's online algorithm.
#[allow(dead_code)]
pub struct AdaptiveThreshold {
    /// Running count of samples.
    count: u64,
    /// Running mean.
    mean: f64,
    /// Running M2 (variance accumulator).
    m2: f64,
    /// Number of standard deviations above mean to set threshold.
    sigma_factor: f64,
    /// Minimum threshold to avoid false triggers on very stable content.
    min_threshold: f32,
}

impl AdaptiveThreshold {
    /// Create a new adaptive threshold estimator.
    #[allow(dead_code)]
    #[must_use]
    pub fn new(sigma_factor: f64, min_threshold: f32) -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            sigma_factor,
            min_threshold,
        }
    }

    /// Update the running statistics with a new score.
    #[allow(dead_code)]
    pub fn update(&mut self, score: f32) {
        self.count += 1;
        let x = f64::from(score);
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    /// Current running mean.
    #[allow(dead_code)]
    #[must_use]
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Current running variance (population).
    #[allow(dead_code)]
    #[must_use]
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        self.m2 / self.count as f64
    }

    /// Current standard deviation.
    #[allow(dead_code)]
    #[must_use]
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Determine whether `score` exceeds the adaptive threshold.
    #[allow(dead_code)]
    #[must_use]
    pub fn threshold(&self, score: f32) -> bool {
        let computed = (self.mean + self.sigma_factor * self.std_dev()) as f32;
        let t = computed.max(self.min_threshold);
        score > t
    }
}

impl Default for AdaptiveThreshold {
    fn default() -> Self {
        Self::new(2.5, 0.1)
    }
}

/// A weighted method configuration for the combined boundary detector.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct MethodWeight {
    /// Which method.
    pub method: BoundaryMethod,
    /// Weight in the voting combination (0.0..=1.0).
    pub weight: f32,
}

/// Combines multiple boundary detection methods with weighted voting.
#[allow(dead_code)]
pub struct BoundaryDetector {
    /// Weights for each method.
    pub methods: Vec<MethodWeight>,
    /// Threshold for the combined score (0.0..=1.0).
    pub threshold: f32,
}

impl BoundaryDetector {
    /// Create a new boundary detector with specified method weights.
    #[allow(dead_code)]
    #[must_use]
    pub fn new(methods: Vec<MethodWeight>, threshold: f32) -> Self {
        Self { methods, threshold }
    }

    /// Combine individual method scores into a boundary decision.
    ///
    /// `scores` must correspond 1:1 to `self.methods`.
    #[allow(dead_code)]
    #[must_use]
    pub fn decide(&self, scores: &[f32]) -> bool {
        if self.methods.is_empty() || scores.is_empty() {
            return false;
        }

        let total_weight: f32 = self.methods.iter().map(|m| m.weight).sum();
        if total_weight <= 0.0 {
            return false;
        }

        let weighted_score: f32 = self
            .methods
            .iter()
            .zip(scores.iter())
            .map(|(m, &s)| m.weight * s)
            .sum::<f32>()
            / total_weight;

        weighted_score > self.threshold
    }

    /// Process a sequence of per-method per-frame scores.
    ///
    /// `frame_scores[i]` is a slice of scores for frame `i`, one per method.
    #[allow(dead_code)]
    #[must_use]
    pub fn detect_boundaries(&self, frame_scores: &[Vec<f32>]) -> Vec<FrameDiff> {
        frame_scores
            .iter()
            .enumerate()
            .map(|(i, scores)| {
                let is_boundary = self.decide(scores);
                let avg_score: f32 = if scores.is_empty() {
                    0.0
                } else {
                    scores.iter().sum::<f32>() / scores.len() as f32
                };
                FrameDiff::new(i as u64, avg_score, is_boundary)
            })
            .collect()
    }
}

impl Default for BoundaryDetector {
    fn default() -> Self {
        Self::new(
            vec![
                MethodWeight {
                    method: BoundaryMethod::PixelDifference,
                    weight: 0.4,
                },
                MethodWeight {
                    method: BoundaryMethod::Histogram,
                    weight: 0.35,
                },
                MethodWeight {
                    method: BoundaryMethod::EdgeChange,
                    weight: 0.25,
                },
            ],
            0.5,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pixel_diff_identical_frames() {
        let frame = vec![0.5_f32; 100];
        let diff = PixelDiffDetector::compute_diff(&frame, &frame);
        assert!(diff.abs() < 1e-6);
    }

    #[test]
    fn test_pixel_diff_opposite_frames() {
        let frame_a = vec![0.0_f32; 100];
        let frame_b = vec![1.0_f32; 100];
        let diff = PixelDiffDetector::compute_diff(&frame_a, &frame_b);
        assert!((diff - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_pixel_diff_empty() {
        let diff = PixelDiffDetector::compute_diff(&[], &[]);
        assert_eq!(diff, 0.0);
    }

    #[test]
    fn test_histogram_diff_identical() {
        let hist = [10_u32; 256];
        let dist = HistogramDiff::compute(&hist, &hist);
        assert!(dist.abs() < 1e-6);
    }

    #[test]
    fn test_histogram_diff_different() {
        let mut hist_a = [0_u32; 256];
        let mut hist_b = [0_u32; 256];
        hist_a[0] = 100;
        hist_b[255] = 100;
        let dist = HistogramDiff::compute(&hist_a, &hist_b);
        // Both bins have non-overlapping content, should be > 0
        assert!(dist > 0.0);
    }

    #[test]
    fn test_histogram_diff_all_zeros() {
        let hist = [0_u32; 256];
        let dist = HistogramDiff::compute(&hist, &hist);
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_edge_change_identical() {
        let edges = vec![1.0_f32; 50];
        let ratio = EdgeChangeDiff::compute(&edges, &edges);
        // Union == Intersection => ratio = 0
        assert!(ratio.abs() < 1e-6);
    }

    #[test]
    fn test_edge_change_no_overlap() {
        let mut edges_a = vec![0.0_f32; 10];
        let mut edges_b = vec![0.0_f32; 10];
        edges_a[0] = 1.0;
        edges_b[1] = 1.0;
        let ratio = EdgeChangeDiff::compute(&edges_a, &edges_b);
        // Union=2, Intersection=0, ratio=1.0
        assert!((ratio - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_edge_change_no_edges() {
        let edges = vec![0.0_f32; 20];
        let ratio = EdgeChangeDiff::compute(&edges, &edges);
        assert_eq!(ratio, 0.0);
    }

    #[test]
    fn test_adaptive_threshold_welford() {
        let mut at = AdaptiveThreshold::new(2.0, 0.0);
        for v in [1.0, 2.0, 3.0, 4.0, 5.0_f32] {
            at.update(v);
        }
        // Mean should be 3.0
        assert!((at.mean() - 3.0).abs() < 1e-6);
        // Variance = ((1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2) / 5 = 10/5 = 2.0
        assert!((at.variance() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_adaptive_threshold_below() {
        let mut at = AdaptiveThreshold::new(2.0, 0.0);
        for v in [1.0_f32; 100] {
            at.update(v);
        }
        // Mean = 1.0, std=0.0; threshold = 1.0 + 2*0 = 1.0; score=1.0 is NOT > 1.0
        assert!(!at.threshold(1.0));
    }

    #[test]
    fn test_adaptive_threshold_above() {
        let mut at = AdaptiveThreshold::new(2.0, 0.1);
        for v in [0.0_f32; 100] {
            at.update(v);
        }
        // Mean=0, std=0, threshold = max(0.0, 0.1) = 0.1; score=0.5 > 0.1
        assert!(at.threshold(0.5));
    }

    #[test]
    fn test_boundary_detector_default() {
        let det = BoundaryDetector::default();
        // Score of 0.0 across all methods -> below threshold
        assert!(!det.decide(&[0.0, 0.0, 0.0]));
        // Score of 1.0 across all methods -> above threshold
        assert!(det.decide(&[1.0, 1.0, 1.0]));
    }

    #[test]
    fn test_boundary_detector_empty() {
        let det = BoundaryDetector::new(vec![], 0.5);
        assert!(!det.decide(&[1.0]));
    }

    #[test]
    fn test_boundary_detector_detect_boundaries() {
        let det = BoundaryDetector::default();
        let frame_scores = vec![
            vec![0.0, 0.0, 0.0], // not boundary
            vec![1.0, 1.0, 1.0], // boundary
        ];
        let diffs = det.detect_boundaries(&frame_scores);
        assert_eq!(diffs.len(), 2);
        assert!(!diffs[0].is_boundary);
        assert!(diffs[1].is_boundary);
    }
}
