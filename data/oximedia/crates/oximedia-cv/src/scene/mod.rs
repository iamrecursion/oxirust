//! Video scene detection and shot boundary detection.
//!
//! This module provides algorithms for detecting scene changes and shot boundaries
//! in video sequences. It supports various detection methods including:
//!
//! - **Histogram-based**: Compares color/intensity histograms between frames
//! - **Edge-based**: Detects changes in edge patterns
//! - **Motion-based**: Analyzes motion vectors and flow
//! - **Adaptive**: Uses adaptive thresholding for varying content
//!
//! # Scene Changes
//!
//! Scene changes can be:
//! - **Hard cuts**: Abrupt transitions between scenes
//! - **Gradual transitions**: Fades, dissolves, wipes
//!
//! # Example
//!
//! ```
//! use oximedia_cv::scene::{SceneDetector, SceneConfig, DetectionMethod};
//! use oximedia_codec::VideoFrame;
//!
//! let config = SceneConfig::default()
//!     .with_threshold(0.3)
//!     .with_method(DetectionMethod::Histogram);
//!
//! let detector = SceneDetector::new(config);
//! // let changes = detector.detect_scenes(&frames);
//! ```

pub mod adaptive;
pub mod classification;
pub mod edge;
pub mod histogram;
pub mod motion;
pub mod transition_detect;

use crate::error::{CvError, CvResult};
use oximedia_codec::VideoFrame;
use oximedia_core::Timestamp;

/// Type of scene change detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeType {
    /// Hard cut - abrupt transition.
    Cut,
    /// Fade in or out.
    Fade,
    /// Dissolve transition.
    Dissolve,
    /// Unknown gradual transition.
    GradualUnknown,
}

impl ChangeType {
    /// Check if this is a gradual transition.
    #[must_use]
    pub const fn is_gradual(&self) -> bool {
        matches!(self, Self::Fade | Self::Dissolve | Self::GradualUnknown)
    }

    /// Check if this is a hard cut.
    #[must_use]
    pub const fn is_cut(&self) -> bool {
        matches!(self, Self::Cut)
    }
}

/// A detected scene change.
#[derive(Debug, Clone)]
pub struct SceneChange {
    /// Frame number where the change occurs.
    pub frame_number: usize,
    /// Timestamp of the scene change.
    pub timestamp: Timestamp,
    /// Confidence score (0.0 to 1.0).
    pub confidence: f64,
    /// Type of change detected.
    pub change_type: ChangeType,
    /// Additional metadata about the detection.
    pub metadata: SceneMetadata,
}

impl SceneChange {
    /// Create a new scene change.
    #[must_use]
    pub fn new(
        frame_number: usize,
        timestamp: Timestamp,
        confidence: f64,
        change_type: ChangeType,
    ) -> Self {
        Self {
            frame_number,
            timestamp,
            confidence: confidence.clamp(0.0, 1.0),
            change_type,
            metadata: SceneMetadata::default(),
        }
    }

    /// Check if this change meets a minimum confidence threshold.
    #[must_use]
    pub fn meets_threshold(&self, threshold: f64) -> bool {
        self.confidence >= threshold
    }
}

/// Metadata about a scene detection.
#[derive(Debug, Clone, Default)]
pub struct SceneMetadata {
    /// Histogram difference score.
    pub histogram_diff: Option<f64>,
    /// Edge change ratio.
    pub edge_change_ratio: Option<f64>,
    /// Motion score.
    pub motion_score: Option<f64>,
    /// Average color difference.
    pub color_diff: Option<f64>,
    /// Duration of gradual transition (in frames).
    pub transition_duration: Option<usize>,
}

/// Detection method to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectionMethod {
    /// Histogram-based detection (RGB).
    Histogram,
    /// Histogram-based detection (HSV color space).
    HistogramHsv,
    /// Edge-based detection.
    Edge,
    /// Motion-based detection.
    Motion,
    /// Adaptive threshold (combines multiple methods).
    Adaptive,
    /// Hybrid approach using multiple methods.
    Hybrid,
}

/// Configuration for scene detection.
#[derive(Debug, Clone)]
pub struct SceneConfig {
    /// Detection method to use.
    pub method: DetectionMethod,
    /// Threshold for scene change detection (0.0 to 1.0).
    pub threshold: f64,
    /// Minimum frames between scene changes (to avoid false positives).
    pub min_scene_length: usize,
    /// Enable gradual transition detection.
    pub detect_gradual: bool,
    /// Window size for gradual transition detection.
    pub gradual_window: usize,
    /// Threshold for gradual transitions (lower than hard cuts).
    pub gradual_threshold: f64,
    /// Enable temporal coherence analysis.
    pub use_temporal_coherence: bool,
    /// Adaptive threshold settings.
    pub adaptive_config: adaptive::AdaptiveConfig,
    /// Histogram settings.
    pub histogram_config: histogram::HistogramConfig,
    /// Edge detection settings.
    pub edge_config: edge::EdgeConfig,
    /// Motion detection settings.
    pub motion_config: motion::MotionConfig,
}

impl Default for SceneConfig {
    fn default() -> Self {
        Self {
            method: DetectionMethod::Histogram,
            threshold: 0.3,
            min_scene_length: 15, // ~0.5 seconds at 30fps
            detect_gradual: true,
            gradual_window: 10,
            gradual_threshold: 0.15,
            use_temporal_coherence: true,
            adaptive_config: adaptive::AdaptiveConfig::default(),
            histogram_config: histogram::HistogramConfig::default(),
            edge_config: edge::EdgeConfig::default(),
            motion_config: motion::MotionConfig::default(),
        }
    }
}

impl SceneConfig {
    /// Create a new scene configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the detection method.
    #[must_use]
    pub const fn with_method(mut self, method: DetectionMethod) -> Self {
        self.method = method;
        self
    }

    /// Set the threshold.
    #[must_use]
    pub const fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set the minimum scene length.
    #[must_use]
    pub const fn with_min_scene_length(mut self, min_scene_length: usize) -> Self {
        self.min_scene_length = min_scene_length;
        self
    }

    /// Enable or disable gradual transition detection.
    #[must_use]
    pub const fn with_detect_gradual(mut self, detect_gradual: bool) -> Self {
        self.detect_gradual = detect_gradual;
        self
    }

    /// Set the gradual transition window size.
    #[must_use]
    pub const fn with_gradual_window(mut self, window: usize) -> Self {
        self.gradual_window = window;
        self
    }

    /// Set temporal coherence analysis.
    #[must_use]
    pub const fn with_temporal_coherence(mut self, enabled: bool) -> Self {
        self.use_temporal_coherence = enabled;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> CvResult<()> {
        if self.threshold < 0.0 || self.threshold > 1.0 {
            return Err(CvError::invalid_parameter(
                "threshold",
                format!("{} (must be 0.0-1.0)", self.threshold),
            ));
        }

        if self.gradual_threshold < 0.0 || self.gradual_threshold > 1.0 {
            return Err(CvError::invalid_parameter(
                "gradual_threshold",
                format!("{} (must be 0.0-1.0)", self.gradual_threshold),
            ));
        }

        if self.min_scene_length == 0 {
            return Err(CvError::invalid_parameter(
                "min_scene_length",
                "must be greater than 0",
            ));
        }

        if self.gradual_window == 0 {
            return Err(CvError::invalid_parameter(
                "gradual_window",
                "must be greater than 0",
            ));
        }

        Ok(())
    }
}

/// Scene detector for video analysis.
pub struct SceneDetector {
    /// Configuration.
    config: SceneConfig,
}

impl SceneDetector {
    /// Create a new scene detector with the given configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::scene::{SceneDetector, SceneConfig};
    ///
    /// let config = SceneConfig::default();
    /// let detector = SceneDetector::new(config);
    /// ```
    #[must_use]
    pub fn new(config: SceneConfig) -> Self {
        Self { config }
    }

    /// Create a scene detector with default configuration.
    #[must_use]
    pub fn default_detector() -> Self {
        Self::new(SceneConfig::default())
    }

    /// Get the current configuration.
    #[must_use]
    pub const fn config(&self) -> &SceneConfig {
        &self.config
    }

    /// Detect scene changes in a sequence of video frames.
    ///
    /// # Arguments
    ///
    /// * `frames` - Slice of video frames to analyze
    ///
    /// # Returns
    ///
    /// Vector of detected scene changes.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid or frame processing fails.
    pub fn detect_scenes(&self, frames: &[VideoFrame]) -> CvResult<Vec<SceneChange>> {
        self.config.validate()?;

        if frames.len() < 2 {
            return Ok(Vec::new());
        }

        let mut changes = match self.config.method {
            DetectionMethod::Histogram => self.detect_histogram(frames)?,
            DetectionMethod::HistogramHsv => self.detect_histogram_hsv(frames)?,
            DetectionMethod::Edge => self.detect_edge(frames)?,
            DetectionMethod::Motion => self.detect_motion(frames)?,
            DetectionMethod::Adaptive => self.detect_adaptive(frames)?,
            DetectionMethod::Hybrid => self.detect_hybrid(frames)?,
        };

        // Post-process: enforce minimum scene length
        changes = self.filter_min_scene_length(changes);

        // Post-process: temporal coherence
        if self.config.use_temporal_coherence {
            changes = self.apply_temporal_coherence(changes);
        }

        // Detect gradual transitions if enabled
        if self.config.detect_gradual {
            let gradual = self.detect_gradual_transitions(frames)?;
            changes.extend(gradual);
            changes.sort_by_key(|c| c.frame_number);
        }

        Ok(changes)
    }

    /// Detect scenes using histogram-based method (RGB).
    fn detect_histogram(&self, frames: &[VideoFrame]) -> CvResult<Vec<SceneChange>> {
        histogram::detect_histogram_changes(frames, &self.config)
    }

    /// Detect scenes using histogram-based method (HSV).
    fn detect_histogram_hsv(&self, frames: &[VideoFrame]) -> CvResult<Vec<SceneChange>> {
        histogram::detect_histogram_hsv_changes(frames, &self.config)
    }

    /// Detect scenes using edge-based method.
    fn detect_edge(&self, frames: &[VideoFrame]) -> CvResult<Vec<SceneChange>> {
        edge::detect_edge_changes(frames, &self.config)
    }

    /// Detect scenes using motion-based method.
    fn detect_motion(&self, frames: &[VideoFrame]) -> CvResult<Vec<SceneChange>> {
        motion::detect_motion_changes(frames, &self.config)
    }

    /// Detect scenes using adaptive method.
    fn detect_adaptive(&self, frames: &[VideoFrame]) -> CvResult<Vec<SceneChange>> {
        adaptive::detect_adaptive_changes(frames, &self.config)
    }

    /// Detect scenes using hybrid method.
    fn detect_hybrid(&self, frames: &[VideoFrame]) -> CvResult<Vec<SceneChange>> {
        let hist_changes = self.detect_histogram(frames)?;
        let edge_changes = self.detect_edge(frames)?;
        let motion_changes = self.detect_motion(frames)?;

        // Combine changes: a frame is a scene change if detected by multiple methods
        let mut combined = Vec::new();
        let all_frames: std::collections::HashSet<usize> = hist_changes
            .iter()
            .chain(edge_changes.iter())
            .chain(motion_changes.iter())
            .map(|c| c.frame_number)
            .collect();

        for frame_num in all_frames {
            let hist_score = hist_changes
                .iter()
                .find(|c| c.frame_number == frame_num)
                .map_or(0.0, |c| c.confidence);

            let edge_score = edge_changes
                .iter()
                .find(|c| c.frame_number == frame_num)
                .map_or(0.0, |c| c.confidence);

            let motion_score = motion_changes
                .iter()
                .find(|c| c.frame_number == frame_num)
                .map_or(0.0, |c| c.confidence);

            // Vote count: how many methods detected this change
            let vote_count =
                (hist_score > 0.0) as u32 + (edge_score > 0.0) as u32 + (motion_score > 0.0) as u32;

            // Require at least 2 out of 3 methods to agree
            if vote_count >= 2 {
                let avg_confidence = (hist_score + edge_score + motion_score) / 3.0;

                combined.push(SceneChange {
                    frame_number: frame_num,
                    timestamp: frames[frame_num].timestamp,
                    confidence: avg_confidence,
                    change_type: ChangeType::Cut,
                    metadata: SceneMetadata {
                        histogram_diff: Some(hist_score),
                        edge_change_ratio: Some(edge_score),
                        motion_score: Some(motion_score),
                        ..Default::default()
                    },
                });
            }
        }

        combined.sort_by_key(|c| c.frame_number);
        Ok(combined)
    }

    /// Detect gradual transitions (fades, dissolves).
    fn detect_gradual_transitions(&self, frames: &[VideoFrame]) -> CvResult<Vec<SceneChange>> {
        if frames.len() < self.config.gradual_window {
            return Ok(Vec::new());
        }

        let mut gradual_changes = Vec::new();
        let window = self.config.gradual_window;

        for i in 0..frames.len() - window {
            let start_frame = &frames[i];
            let end_frame = &frames[i + window];

            // Compute similarity between start and end of window
            let similarity = histogram::compute_frame_similarity(start_frame, end_frame)?;
            let diff = 1.0 - similarity;

            // Check if there's a gradual change
            if diff > self.config.gradual_threshold && diff < self.config.threshold {
                // Determine transition type
                let change_type = self.classify_gradual_transition(frames, i, i + window)?;

                gradual_changes.push(SceneChange {
                    frame_number: i + window / 2, // Middle of transition
                    timestamp: frames[i + window / 2].timestamp,
                    confidence: diff,
                    change_type,
                    metadata: SceneMetadata {
                        transition_duration: Some(window),
                        histogram_diff: Some(diff),
                        ..Default::default()
                    },
                });
            }
        }

        Ok(gradual_changes)
    }

    /// Classify the type of gradual transition.
    fn classify_gradual_transition(
        &self,
        frames: &[VideoFrame],
        start: usize,
        end: usize,
    ) -> CvResult<ChangeType> {
        // Compute brightness trend
        let start_brightness = histogram::compute_average_brightness(&frames[start])?;
        let end_brightness = histogram::compute_average_brightness(&frames[end])?;

        let brightness_change = (end_brightness - start_brightness).abs();
        let brightness_ratio = if start_brightness > 0.0 {
            brightness_change / start_brightness
        } else {
            0.0
        };

        // If brightness changes significantly, it's likely a fade
        if brightness_ratio > 0.5 {
            return Ok(ChangeType::Fade);
        }

        // Otherwise, it's likely a dissolve
        Ok(ChangeType::Dissolve)
    }

    /// Filter scene changes by minimum scene length.
    fn filter_min_scene_length(&self, changes: Vec<SceneChange>) -> Vec<SceneChange> {
        if changes.is_empty() {
            return changes;
        }

        let mut filtered = Vec::new();
        let mut last_frame = 0;

        for change in changes {
            if change.frame_number - last_frame >= self.config.min_scene_length {
                filtered.push(change.clone());
                last_frame = change.frame_number;
            }
        }

        filtered
    }

    /// Apply temporal coherence to reduce false positives.
    fn apply_temporal_coherence(&self, changes: Vec<SceneChange>) -> Vec<SceneChange> {
        // Simple temporal coherence: remove isolated detections
        if changes.len() < 2 {
            return changes;
        }

        let mut coherent = Vec::new();
        let window = 3; // Look at neighbors within 3 frames

        for (i, change) in changes.iter().enumerate() {
            // Count nearby detections
            let nearby_count = changes
                .iter()
                .enumerate()
                .filter(|(j, c)| {
                    *j != i && (c.frame_number as i64 - change.frame_number as i64).abs() <= window
                })
                .count();

            // Keep if there are nearby detections or if confidence is very high
            if nearby_count > 0 || change.confidence > self.config.threshold * 1.5 {
                coherent.push(change.clone());
            }
        }

        coherent
    }

    /// Analyze a pair of consecutive frames.
    pub fn analyze_frame_pair(
        &self,
        frame1: &VideoFrame,
        frame2: &VideoFrame,
    ) -> CvResult<SceneChange> {
        let similarity = match self.config.method {
            DetectionMethod::Histogram => histogram::compute_frame_similarity(frame1, frame2)?,
            DetectionMethod::HistogramHsv => {
                histogram::compute_frame_similarity_hsv(frame1, frame2)?
            }
            DetectionMethod::Edge => {
                edge::compute_edge_similarity(frame1, frame2, &self.config.edge_config)?
            }
            DetectionMethod::Motion => {
                motion::compute_motion_score(frame1, frame2, &self.config.motion_config)?
            }
            _ => {
                // For adaptive/hybrid, use histogram as default
                histogram::compute_frame_similarity(frame1, frame2)?
            }
        };

        let diff = 1.0 - similarity;
        let change_type = if diff > self.config.threshold {
            ChangeType::Cut
        } else {
            ChangeType::GradualUnknown
        };

        Ok(SceneChange::new(
            0, // Frame number unknown in pair analysis
            frame2.timestamp,
            diff,
            change_type,
        ))
    }
}

impl Default for SceneDetector {
    fn default() -> Self {
        Self::default_detector()
    }
}

/// Compute the difference between two frames using the specified method.
///
/// Returns a value between 0.0 (identical) and 1.0 (completely different).
pub fn compute_frame_difference(
    frame1: &VideoFrame,
    frame2: &VideoFrame,
    method: DetectionMethod,
) -> CvResult<f64> {
    let similarity = match method {
        DetectionMethod::Histogram => histogram::compute_frame_similarity(frame1, frame2)?,
        DetectionMethod::HistogramHsv => histogram::compute_frame_similarity_hsv(frame1, frame2)?,
        DetectionMethod::Edge => {
            let config = edge::EdgeConfig::default();
            edge::compute_edge_similarity(frame1, frame2, &config)?
        }
        DetectionMethod::Motion => {
            let config = motion::MotionConfig::default();
            motion::compute_motion_score(frame1, frame2, &config)?
        }
        DetectionMethod::Adaptive | DetectionMethod::Hybrid => {
            // Use histogram as default
            histogram::compute_frame_similarity(frame1, frame2)?
        }
    };

    Ok(1.0 - similarity)
}
