//! Video tracking and motion estimation.
//!
//! This module provides comprehensive tracking algorithms including:
//!
//! # Single Object Tracking
//!
//! - [`mosse`]: MOSSE (Minimum Output Sum of Squared Error) - Fast correlation filter
//! - [`kcf`]: KCF (Kernelized Correlation Filter) - High-speed tracking with kernels
//! - [`csrt`][]: CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability)
//! - [`medianflow`]: MedianFlow - Robust tracking with forward-backward error checking
//! - [`tld`]: TLD (Tracking-Learning-Detection) - Long-term tracking with learning
//!
//! # Multi-Object Tracking
//!
//! - [`sort`]: SORT (Simple Online and Realtime Tracking) - Kalman filter + Hungarian algorithm
//! - [`centroid`]: Centroid tracker - Simple distance-based tracking
//! - [`iou_tracker`]: IoU tracker - Fast overlap-based tracking
//!
//! # Utilities
//!
//! - [`kalman`]: Kalman filter for motion prediction and state estimation
//! - [`assignment`]: Hungarian algorithm for optimal assignment
//! - [`optical_flow`]: Optical flow estimation (Lucas-Kanade, Farneback, Dense RLOF)
//! - [`feature_tracker`]: Feature tracking with KLT algorithm
//! - [`object_tracker`]: Legacy object tracking interface
//!
//! # Example
//!
//! ```
//! use oximedia_cv::tracking::mosse::MosseTracker;
//! use oximedia_cv::tracking::sort::SortTracker;
//! use oximedia_cv::detect::BoundingBox;
//!
//! // Single object tracking
//! let bbox = BoundingBox::new(50.0, 50.0, 100.0, 100.0);
//! let tracker = MosseTracker::new(bbox);
//!
//! // Multi-object tracking
//! let mut sort = SortTracker::new();
//! let detections = vec![BoundingBox::new(100.0, 100.0, 50.0, 50.0)];
//! let tracks = sort.update(&detections);
//! ```

// Legacy modules (maintained for backwards compatibility)
pub mod feature_tracker;
pub mod kalman;
pub mod object_tracker;
pub mod optical_flow;

// Single object trackers
pub mod csrt;
pub mod kcf;
pub mod medianflow;
pub mod mosse;
pub mod tld;

// Multi-object trackers
pub mod centroid;
pub mod iou_tracker;
pub mod optical_flow_tracker;
pub mod sort;
pub mod sort_enhanced;

// Utilities
pub mod assignment;

// Re-export commonly used items from legacy modules
pub use feature_tracker::{FeatureTracker, TrackedFeature};
pub use kalman::KalmanFilter;
pub use object_tracker::{ObjectTracker, TrackerType};
pub use optical_flow::{
    compute_lk_bouguet_sparse, FlowField, FlowMethod, LkConfig, LkFlowPoint, OpticalFlow,
};

// Re-export single object trackers
pub use csrt::CsrtTracker;
pub use kcf::KcfTracker;
pub use medianflow::MedianFlowTracker;
pub use mosse::MosseTracker;
pub use tld::TldTracker;

// Re-export multi-object trackers
pub use centroid::CentroidTracker;
pub use iou_tracker::{IouTracker, IouTrackerAdvanced};
pub use optical_flow_tracker::{FlowPoint, GrayFrame, LkTracker};
pub use sort::{DeepSortTracker, SortTracker};
pub use sort_enhanced::{BBox as SortBBox, KalmanTrack, SortTrackerV2, TrackedObject};

// Re-export utilities
pub use assignment::{compute_iou, create_iou_cost_matrix, greedy_assignment, hungarian_algorithm};

/// 2D point representation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2D {
    /// X coordinate.
    pub x: f32,
    /// Y coordinate.
    pub y: f32,
}

impl Point2D {
    /// Create a new 2D point.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::tracking::Point2D;
    ///
    /// let point = Point2D::new(10.0, 20.0);
    /// assert_eq!(point.x, 10.0);
    /// ```
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Calculate distance to another point.
    #[must_use]
    pub fn distance(&self, other: &Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Calculate squared distance to another point (faster).
    #[must_use]
    pub fn distance_squared(&self, other: &Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }

    /// Calculate dot product with another point.
    #[must_use]
    pub fn dot(&self, other: &Self) -> f32 {
        self.x * other.x + self.y * other.y
    }
}

impl Default for Point2D {
    fn default() -> Self {
        Self::new(0.0, 0.0)
    }
}

impl From<(f32, f32)> for Point2D {
    fn from((x, y): (f32, f32)) -> Self {
        Self::new(x, y)
    }
}

impl From<Point2D> for (f32, f32) {
    fn from(p: Point2D) -> Self {
        (p.x, p.y)
    }
}

/// Tracker quality metrics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackQuality {
    /// Tracking confidence (0.0 to 1.0).
    pub confidence: f64,
    /// Peak-to-sidelobe ratio (for correlation filters).
    pub psr: f64,
    /// Forward-backward error (for optical flow trackers).
    pub fb_error: f64,
    /// Number of tracked features.
    pub num_features: usize,
}

impl TrackQuality {
    /// Create new track quality metrics.
    #[must_use]
    pub const fn new(confidence: f64) -> Self {
        Self {
            confidence,
            psr: 0.0,
            fb_error: 0.0,
            num_features: 0,
        }
    }

    /// Check if tracking is considered good.
    #[must_use]
    pub const fn is_good(&self) -> bool {
        self.confidence > 0.7
    }

    /// Check if tracking has failed.
    #[must_use]
    pub const fn has_failed(&self) -> bool {
        self.confidence < 0.3
    }
}

impl Default for TrackQuality {
    fn default() -> Self {
        Self::new(1.0)
    }
}
