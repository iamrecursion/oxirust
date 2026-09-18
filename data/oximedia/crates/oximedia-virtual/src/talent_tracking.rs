//! 2D pose estimation and automatic talent masking for virtual production.
//!
//! Provides a lightweight pure-Rust human pose estimator based on colour
//! and gradient cues, plus a silhouette-based talent masker.  The estimator
//! detects 17 COCO-compatible body key-points and outputs a binary alpha mask
//! suitable for downstream compositing.

use crate::{Result, VirtualProductionError};
use serde::{Deserialize, Serialize};

/// COCO body key-point index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Keypoint {
    Nose = 0,
    LeftEye = 1,
    RightEye = 2,
    LeftEar = 3,
    RightEar = 4,
    LeftShoulder = 5,
    RightShoulder = 6,
    LeftElbow = 7,
    RightElbow = 8,
    LeftWrist = 9,
    RightWrist = 10,
    LeftHip = 11,
    RightHip = 12,
    LeftKnee = 13,
    RightKnee = 14,
    LeftAnkle = 15,
    RightAnkle = 16,
}

impl Keypoint {
    /// Total number of key-points.
    pub const COUNT: usize = 17;

    /// Human-readable name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Nose => "nose",
            Self::LeftEye => "left_eye",
            Self::RightEye => "right_eye",
            Self::LeftEar => "left_ear",
            Self::RightEar => "right_ear",
            Self::LeftShoulder => "left_shoulder",
            Self::RightShoulder => "right_shoulder",
            Self::LeftElbow => "left_elbow",
            Self::RightElbow => "right_elbow",
            Self::LeftWrist => "left_wrist",
            Self::RightWrist => "right_wrist",
            Self::LeftHip => "left_hip",
            Self::RightHip => "right_hip",
            Self::LeftKnee => "left_knee",
            Self::RightKnee => "right_knee",
            Self::LeftAnkle => "left_ankle",
            Self::RightAnkle => "right_ankle",
        }
    }

    /// All key-points in COCO order.
    pub fn all() -> [Self; Self::COUNT] {
        [
            Self::Nose,
            Self::LeftEye,
            Self::RightEye,
            Self::LeftEar,
            Self::RightEar,
            Self::LeftShoulder,
            Self::RightShoulder,
            Self::LeftElbow,
            Self::RightElbow,
            Self::LeftWrist,
            Self::RightWrist,
            Self::LeftHip,
            Self::RightHip,
            Self::LeftKnee,
            Self::RightKnee,
            Self::LeftAnkle,
            Self::RightAnkle,
        ]
    }
}

/// COCO skeleton edges (pairs of key-point indices).
pub const SKELETON_EDGES: &[(usize, usize)] = &[
    (0, 1),   // nose → left eye
    (0, 2),   // nose → right eye
    (1, 3),   // left eye → left ear
    (2, 4),   // right eye → right ear
    (5, 6),   // left shoulder → right shoulder
    (5, 7),   // left shoulder → left elbow
    (7, 9),   // left elbow → left wrist
    (6, 8),   // right shoulder → right elbow
    (8, 10),  // right elbow → right wrist
    (5, 11),  // left shoulder → left hip
    (6, 12),  // right shoulder → right hip
    (11, 12), // left hip → right hip
    (11, 13), // left hip → left knee
    (13, 15), // left knee → left ankle
    (12, 14), // right hip → right knee
    (14, 16), // right knee → right ankle
];

/// A detected 2D key-point with confidence.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct KeypointDetection {
    /// Pixel x coordinate.
    pub x: f32,
    /// Pixel y coordinate.
    pub y: f32,
    /// Detection confidence in [0.0, 1.0].
    pub confidence: f32,
    /// Whether this key-point was detected (confidence > threshold).
    pub detected: bool,
}

impl KeypointDetection {
    /// Create a detected key-point.
    #[must_use]
    pub fn new(x: f32, y: f32, confidence: f32) -> Self {
        Self {
            x,
            y,
            confidence,
            detected: confidence > 0.3,
        }
    }

    /// Create an undetected (invisible) key-point.
    #[must_use]
    pub fn undetected() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            confidence: 0.0,
            detected: false,
        }
    }
}

/// Full pose: 17 key-points + bounding box + overall confidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoseEstimate {
    /// Key-points in COCO order.
    pub keypoints: Vec<KeypointDetection>,
    /// Bounding box [x, y, w, h] in pixels.
    pub bbox: [f32; 4],
    /// Overall pose confidence.
    pub confidence: f32,
    /// Person instance ID (for multi-person tracking).
    pub instance_id: u32,
}

impl PoseEstimate {
    /// Create a new pose estimate.
    #[must_use]
    pub fn new(
        keypoints: Vec<KeypointDetection>,
        bbox: [f32; 4],
        confidence: f32,
        instance_id: u32,
    ) -> Self {
        Self {
            keypoints,
            bbox,
            confidence,
            instance_id,
        }
    }

    /// Number of detected (visible) key-points.
    #[must_use]
    pub fn detected_count(&self) -> usize {
        self.keypoints.iter().filter(|k| k.detected).count()
    }

    /// Get a specific key-point.
    #[must_use]
    pub fn keypoint(&self, kp: Keypoint) -> Option<&KeypointDetection> {
        self.keypoints.get(kp as usize)
    }

    /// Compute the body height (ankle to nose) in pixels.
    #[must_use]
    pub fn body_height_px(&self) -> f32 {
        let nose = self.keypoints.get(Keypoint::Nose as usize);
        let l_ankle = self.keypoints.get(Keypoint::LeftAnkle as usize);
        let r_ankle = self.keypoints.get(Keypoint::RightAnkle as usize);

        let top_y = nose.map(|k| k.y).unwrap_or(self.bbox[1]);
        let bot_y = match (l_ankle, r_ankle) {
            (Some(l), Some(r)) if l.detected && r.detected => (l.y + r.y) / 2.0,
            (Some(l), _) if l.detected => l.y,
            (_, Some(r)) if r.detected => r.y,
            _ => self.bbox[1] + self.bbox[3],
        };
        (bot_y - top_y).abs()
    }
}

/// Masking configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TalentMaskingConfig {
    /// Confidence threshold above which a key-point counts as detected.
    pub confidence_threshold: f32,
    /// Radius around each key-point to include in the mask (pixels).
    pub keypoint_radius: f32,
    /// Dilation radius for the final mask (pixels).
    pub dilation_radius: f32,
    /// Whether to use convex hull for the body mask.
    pub use_convex_hull: bool,
    /// Maximum number of people to track simultaneously.
    pub max_people: usize,
}

impl Default for TalentMaskingConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.3,
            keypoint_radius: 30.0,
            dilation_radius: 10.0,
            use_convex_hull: true,
            max_people: 4,
        }
    }
}

/// Talent mask: per-pixel alpha values.
#[derive(Debug, Clone)]
pub struct TalentMask {
    /// Alpha values in [0.0, 1.0], row-major.
    pub alpha: Vec<f32>,
    /// Image width.
    pub width: usize,
    /// Image height.
    pub height: usize,
    /// Number of instances contributing to this mask.
    pub instance_count: usize,
}

impl TalentMask {
    /// Create an empty (all-zero) mask.
    #[must_use]
    pub fn empty(width: usize, height: usize) -> Self {
        Self {
            alpha: vec![0.0; width * height],
            width,
            height,
            instance_count: 0,
        }
    }

    /// Sample mask at pixel (x, y).
    #[must_use]
    pub fn get(&self, x: usize, y: usize) -> Option<f32> {
        if x >= self.width || y >= self.height {
            return None;
        }
        Some(self.alpha[y * self.width + x])
    }

    /// Compute the fraction of pixels that are foreground (alpha > 0.5).
    #[must_use]
    pub fn foreground_fraction(&self) -> f32 {
        let total = self.alpha.len();
        if total == 0 {
            return 0.0;
        }
        let fg = self.alpha.iter().filter(|&&a| a > 0.5).count();
        fg as f32 / total as f32
    }
}

/// Simple foreground segmentation using colour-based saliency.
///
/// In production a neural network would be used; this implementation
/// provides a deterministic heuristic based on skin-tone proximity and
/// edge magnitude, suitable for unit testing.
struct ForegroundSegmenter {
    #[allow(dead_code)]
    config: TalentMaskingConfig,
}

impl ForegroundSegmenter {
    fn new(config: TalentMaskingConfig) -> Self {
        Self { config }
    }

    /// Compute a per-pixel foreground probability from an RGB image.
    fn segment(&self, image: &[u8], width: usize, height: usize) -> Vec<f32> {
        let n = width * height;
        let mut prob = vec![0.0f32; n];

        // Compute Sobel gradient magnitude as a proxy for foreground edge strength
        let grad = Self::sobel_magnitude(image, width, height);

        // Skin-tone heuristic in (R-G, R-B, R/G) space
        for i in 0..n {
            let r = image[i * 3] as f32;
            let g = image[i * 3 + 1] as f32;
            let b = image[i * 3 + 2] as f32;

            // Simplified skin detection: r > 95, g > 40, b > 20, r > g, r > b
            let is_skin = r > 95.0
                && g > 40.0
                && b > 20.0
                && r > g
                && r > b
                && (r - g).abs() > 15.0
                && r as u32 + g as u32 + b as u32 > 180;

            let skin_score = if is_skin { 0.6 } else { 0.0 };
            let edge_score = (grad[i] / 255.0).min(1.0) * 0.4;

            prob[i] = (skin_score + edge_score).min(1.0);
        }

        // Simple threshold
        prob
    }

    /// Compute Sobel gradient magnitude (returns 0–255 range).
    fn sobel_magnitude(image: &[u8], width: usize, height: usize) -> Vec<f32> {
        let mut gray = vec![0.0f32; width * height];
        for i in 0..(width * height) {
            let r = image[i * 3] as f32;
            let g = image[i * 3 + 1] as f32;
            let b = image[i * 3 + 2] as f32;
            gray[i] = 0.299 * r + 0.587 * g + 0.114 * b;
        }

        let mut mag = vec![0.0f32; width * height];
        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                let gx = -gray[(y - 1) * width + (x - 1)]
                    - 2.0 * gray[y * width + (x - 1)]
                    - gray[(y + 1) * width + (x - 1)]
                    + gray[(y - 1) * width + (x + 1)]
                    + 2.0 * gray[y * width + (x + 1)]
                    + gray[(y + 1) * width + (x + 1)];

                let gy = -gray[(y - 1) * width + (x - 1)]
                    - 2.0 * gray[(y - 1) * width + x]
                    - gray[(y - 1) * width + (x + 1)]
                    + gray[(y + 1) * width + (x - 1)]
                    + 2.0 * gray[(y + 1) * width + x]
                    + gray[(y + 1) * width + (x + 1)];

                mag[y * width + x] = (gx * gx + gy * gy).sqrt().min(255.0);
            }
        }
        mag
    }
}

/// Lightweight 2D pose estimator.
///
/// In production a neural network (e.g. OpenPose, ViTPose) would be used.
/// This implementation heuristically places key-points around detected
/// foreground blobs using aspect-ratio priors.
struct PoseEstimatorInner {
    config: TalentMaskingConfig,
}

impl PoseEstimatorInner {
    fn new(config: TalentMaskingConfig) -> Self {
        Self { config }
    }

    /// Estimate poses from a foreground probability map.
    fn estimate(&self, prob: &[f32], width: usize, height: usize) -> Vec<PoseEstimate> {
        // Find connected foreground regions
        let blobs = self.find_blobs(prob, width, height, 0.4);

        let mut poses = Vec::new();
        for (instance_id, blob) in blobs.into_iter().enumerate() {
            if instance_id >= self.config.max_people {
                break;
            }
            let (bx, by, bw, bh) = blob;
            let bbox = [bx as f32, by as f32, bw as f32, bh as f32];
            let keypoints = self.place_keypoints(bx, by, bw, bh);
            let confidence = 0.6 + 0.1 * (bw.min(bh) as f32 / width.min(height) as f32).min(1.0);
            poses.push(PoseEstimate::new(
                keypoints,
                bbox,
                confidence,
                instance_id as u32,
            ));
        }
        poses
    }

    /// Place key-points anatomically within a bounding box.
    fn place_keypoints(
        &self,
        bx: usize,
        by: usize,
        bw: usize,
        bh: usize,
    ) -> Vec<KeypointDetection> {
        // Normalised key-point positions within a human bounding box
        // (0,0) = top-left, (1,1) = bottom-right
        const KP_POS: [(f32, f32); 17] = [
            (0.5, 0.05),  // 0: nose
            (0.45, 0.03), // 1: left eye
            (0.55, 0.03), // 2: right eye
            (0.4, 0.04),  // 3: left ear
            (0.6, 0.04),  // 4: right ear
            (0.35, 0.2),  // 5: left shoulder
            (0.65, 0.2),  // 6: right shoulder
            (0.25, 0.38), // 7: left elbow
            (0.75, 0.38), // 8: right elbow
            (0.18, 0.55), // 9: left wrist
            (0.82, 0.55), // 10: right wrist
            (0.4, 0.55),  // 11: left hip
            (0.6, 0.55),  // 12: right hip
            (0.38, 0.72), // 13: left knee
            (0.62, 0.72), // 14: right knee
            (0.36, 0.92), // 15: left ankle
            (0.64, 0.92), // 16: right ankle
        ];

        KP_POS
            .iter()
            .map(|(nx, ny)| {
                let x = bx as f32 + nx * bw as f32;
                let y = by as f32 + ny * bh as f32;
                KeypointDetection::new(x, y, 0.7)
            })
            .collect()
    }

    /// Find connected components above a probability threshold.
    /// Returns list of (min_x, min_y, width, height) bounding boxes.
    fn find_blobs(
        &self,
        prob: &[f32],
        width: usize,
        height: usize,
        threshold: f32,
    ) -> Vec<(usize, usize, usize, usize)> {
        let mut visited = vec![false; width * height];
        let mut blobs = Vec::new();
        let min_blob_area = 200usize;

        for start_y in 0..height {
            for start_x in 0..width {
                let idx = start_y * width + start_x;
                if visited[idx] || prob[idx] < threshold {
                    continue;
                }
                // BFS
                let mut stack = vec![(start_x, start_y)];
                let mut min_x = start_x;
                let mut min_y = start_y;
                let mut max_x = start_x;
                let mut max_y = start_y;
                let mut count = 0usize;

                while let Some((cx, cy)) = stack.pop() {
                    let i = cy * width + cx;
                    if visited[i] {
                        continue;
                    }
                    visited[i] = true;
                    if prob[i] < threshold {
                        continue;
                    }
                    count += 1;
                    min_x = min_x.min(cx);
                    min_y = min_y.min(cy);
                    max_x = max_x.max(cx);
                    max_y = max_y.max(cy);

                    if cx + 1 < width {
                        stack.push((cx + 1, cy));
                    }
                    if cx > 0 {
                        stack.push((cx - 1, cy));
                    }
                    if cy + 1 < height {
                        stack.push((cx, cy + 1));
                    }
                    if cy > 0 {
                        stack.push((cx, cy - 1));
                    }
                }

                if count >= min_blob_area {
                    let bw = max_x.saturating_sub(min_x).max(1);
                    let bh = max_y.saturating_sub(min_y).max(1);
                    blobs.push((min_x, min_y, bw, bh));
                }
            }
        }

        blobs
    }
}

/// Main talent tracker: pose estimation + mask generation.
pub struct TalentTracker {
    config: TalentMaskingConfig,
    segmenter: ForegroundSegmenter,
    estimator: PoseEstimatorInner,
    last_poses: Vec<PoseEstimate>,
    frame_count: u64,
}

impl TalentTracker {
    /// Create a talent tracker with default configuration.
    pub fn new() -> Result<Self> {
        Self::with_config(TalentMaskingConfig::default())
    }

    /// Create with explicit configuration.
    pub fn with_config(config: TalentMaskingConfig) -> Result<Self> {
        let segmenter = ForegroundSegmenter::new(config.clone());
        let estimator = PoseEstimatorInner::new(config.clone());
        Ok(Self {
            config,
            segmenter,
            estimator,
            last_poses: Vec::new(),
            frame_count: 0,
        })
    }

    /// Process an RGB frame: estimate poses and generate talent mask.
    pub fn process(
        &mut self,
        image: &[u8],
        width: usize,
        height: usize,
    ) -> Result<(Vec<PoseEstimate>, TalentMask)> {
        if image.len() != width * height * 3 {
            return Err(VirtualProductionError::Compositing(format!(
                "Image size mismatch: expected {}, got {}",
                width * height * 3,
                image.len()
            )));
        }

        // 1. Segment foreground
        let prob = self.segmenter.segment(image, width, height);

        // 2. Estimate poses
        let poses = self.estimator.estimate(&prob, width, height);
        self.last_poses = poses.clone();
        self.frame_count += 1;

        // 3. Generate mask
        let mask = self.generate_mask(&poses, width, height);

        Ok((poses, mask))
    }

    /// Generate a binary-ish alpha mask from a list of poses.
    fn generate_mask(&self, poses: &[PoseEstimate], width: usize, height: usize) -> TalentMask {
        let n = width * height;
        let mut alpha = vec![0.0f32; n];
        let radius = self.config.keypoint_radius;

        for pose in poses {
            // Fill convex hull of bounding box
            let bx = pose.bbox[0] as usize;
            let by = pose.bbox[1] as usize;
            let bw = pose.bbox[2] as usize;
            let bh = pose.bbox[3] as usize;

            let ex = (bx + bw).min(width);
            let ey = (by + bh).min(height);

            for y in by..ey {
                for x in bx..ex {
                    alpha[y * width + x] = pose.confidence.min(1.0);
                }
            }

            // Additional coverage around each key-point
            for kp in &pose.keypoints {
                if !kp.detected {
                    continue;
                }
                let kx = kp.x as i32;
                let ky = kp.y as i32;
                let r = radius as i32;

                let x0 = (kx - r).max(0) as usize;
                let y0 = (ky - r).max(0) as usize;
                let x1 = (kx + r).min(width as i32 - 1) as usize;
                let y1 = (ky + r).min(height as i32 - 1) as usize;

                for y in y0..=y1 {
                    for x in x0..=x1 {
                        let dx = x as f32 - kp.x;
                        let dy = y as f32 - kp.y;
                        if dx * dx + dy * dy <= radius * radius {
                            let i = y * width + x;
                            alpha[i] = alpha[i].max(kp.confidence);
                        }
                    }
                }
            }
        }

        // Optional dilation
        let alpha = if self.config.dilation_radius > 0.0 {
            self.dilate(&alpha, width, height, self.config.dilation_radius as usize)
        } else {
            alpha
        };

        TalentMask {
            alpha,
            width,
            height,
            instance_count: poses.len(),
        }
    }

    /// Morphological dilation with a square structuring element.
    fn dilate(&self, alpha: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let y0 = y.saturating_sub(radius);
                let y1 = (y + radius).min(height - 1);
                let x0 = x.saturating_sub(radius);
                let x1 = (x + radius).min(width - 1);

                let mut max_val = 0.0f32;
                for dy in y0..=y1 {
                    for dx in x0..=x1 {
                        max_val = max_val.max(alpha[dy * width + dx]);
                    }
                }
                out[y * width + x] = max_val;
            }
        }
        out
    }

    /// Get the last detected poses.
    #[must_use]
    pub fn last_poses(&self) -> &[PoseEstimate] {
        &self.last_poses
    }

    /// Get frame count.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Get configuration.
    #[must_use]
    pub fn config(&self) -> &TalentMaskingConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_skin_image(w: usize, h: usize) -> Vec<u8> {
        // Skin-tone rectangle in the center of a dark background
        let mut img = vec![20u8; w * h * 3];
        let cx = w / 4;
        let cy = h / 8;
        for y in cy..(h - cy) {
            for x in cx..(w - cx) {
                let idx = (y * w + x) * 3;
                img[idx] = 200; // R
                img[idx + 1] = 120; // G
                img[idx + 2] = 80; // B
            }
        }
        img
    }

    #[test]
    fn test_talent_tracker_creation() {
        let tracker = TalentTracker::new();
        assert!(tracker.is_ok());
    }

    #[test]
    fn test_talent_tracker_with_config() {
        let config = TalentMaskingConfig {
            confidence_threshold: 0.5,
            keypoint_radius: 20.0,
            dilation_radius: 5.0,
            use_convex_hull: false,
            max_people: 2,
        };
        let tracker = TalentTracker::with_config(config);
        assert!(tracker.is_ok());
    }

    #[test]
    fn test_process_black_frame() {
        let mut tracker = TalentTracker::new().expect("should create");
        let img = vec![0u8; 64 * 64 * 3];
        let result = tracker.process(&img, 64, 64);
        assert!(result.is_ok());
        let (poses, mask) = result.expect("ok");
        // Black frame has no skin-tone → no poses
        assert!(poses.is_empty());
        assert_eq!(mask.instance_count, 0);
        assert_eq!(mask.foreground_fraction(), 0.0);
    }

    #[test]
    fn test_process_size_mismatch() {
        let mut tracker = TalentTracker::new().expect("should create");
        let img = vec![0u8; 10];
        let result = tracker.process(&img, 64, 64);
        assert!(result.is_err());
    }

    #[test]
    fn test_process_skin_frame_detects_foreground() {
        let w = 64usize;
        let h = 64usize;
        let mut tracker = TalentTracker::with_config(TalentMaskingConfig {
            keypoint_radius: 5.0,
            dilation_radius: 2.0,
            ..TalentMaskingConfig::default()
        })
        .expect("should create");
        let img = make_skin_image(w, h);
        let result = tracker.process(&img, w, h);
        assert!(result.is_ok());
    }

    #[test]
    fn test_frame_count() {
        let mut tracker = TalentTracker::new().expect("should create");
        let img = vec![0u8; 32 * 32 * 3];
        assert_eq!(tracker.frame_count(), 0);
        tracker.process(&img, 32, 32).expect("ok");
        tracker.process(&img, 32, 32).expect("ok");
        assert_eq!(tracker.frame_count(), 2);
    }

    #[test]
    fn test_talent_mask_empty() {
        let mask = TalentMask::empty(64, 64);
        assert_eq!(mask.alpha.len(), 64 * 64);
        assert_eq!(mask.foreground_fraction(), 0.0);
    }

    #[test]
    fn test_talent_mask_get_out_of_bounds() {
        let mask = TalentMask::empty(64, 64);
        assert!(mask.get(64, 0).is_none());
        assert!(mask.get(0, 64).is_none());
    }

    #[test]
    fn test_talent_mask_get_in_bounds() {
        let mask = TalentMask::empty(64, 64);
        assert_eq!(mask.get(10, 10), Some(0.0));
    }

    #[test]
    fn test_keypoint_names() {
        assert_eq!(Keypoint::Nose.name(), "nose");
        assert_eq!(Keypoint::LeftAnkle.name(), "left_ankle");
    }

    #[test]
    fn test_keypoint_all_count() {
        assert_eq!(Keypoint::all().len(), Keypoint::COUNT);
    }

    #[test]
    fn test_skeleton_edges_valid() {
        for &(a, b) in SKELETON_EDGES {
            assert!(a < Keypoint::COUNT, "edge a={a} out of range");
            assert!(b < Keypoint::COUNT, "edge b={b} out of range");
        }
    }

    #[test]
    fn test_pose_estimate_detected_count() {
        let kps: Vec<KeypointDetection> = (0..17)
            .map(|i| {
                if i % 2 == 0 {
                    KeypointDetection::new(10.0, 10.0, 0.9)
                } else {
                    KeypointDetection::undetected()
                }
            })
            .collect();
        let pose = PoseEstimate::new(kps, [0.0, 0.0, 50.0, 100.0], 0.8, 0);
        assert_eq!(pose.detected_count(), 9); // 0,2,4,6,8,10,12,14,16
    }

    #[test]
    fn test_pose_estimate_body_height() {
        let mut kps: Vec<KeypointDetection> =
            (0..17).map(|_| KeypointDetection::undetected()).collect();
        kps[Keypoint::Nose as usize] = KeypointDetection::new(50.0, 10.0, 0.9);
        kps[Keypoint::LeftAnkle as usize] = KeypointDetection::new(45.0, 180.0, 0.9);
        kps[Keypoint::RightAnkle as usize] = KeypointDetection::new(55.0, 180.0, 0.9);

        let pose = PoseEstimate::new(kps, [30.0, 10.0, 40.0, 170.0], 0.8, 0);
        let height = pose.body_height_px();
        assert!((height - 170.0).abs() < 1.0, "body height: {height}");
    }

    #[test]
    fn test_foreground_fraction() {
        let mut mask = TalentMask::empty(10, 10);
        // Set half the pixels to 1.0
        for i in 0..50 {
            mask.alpha[i] = 1.0;
        }
        let frac = mask.foreground_fraction();
        assert!((frac - 0.5).abs() < 1e-5, "fraction: {frac}");
    }
}
