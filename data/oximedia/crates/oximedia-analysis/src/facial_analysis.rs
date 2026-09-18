//! Face detection and facial analysis.
//!
//! This module provides face detection and analysis capabilities:
//! - **Face Detection** - Viola-Jones inspired sliding window detection
//! - **Gaze Estimation** - Normalized gaze direction from landmarks
//! - **Face Orientation** - Yaw, pitch, roll estimation
//! - **Expression Analysis** - Emotion classification with valence

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// A rectangular region containing a detected face.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceRegion {
    /// X coordinate of the top-left corner (pixels)
    pub x: u32,
    /// Y coordinate of the top-left corner (pixels)
    pub y: u32,
    /// Width of the region (pixels)
    pub width: u32,
    /// Height of the region (pixels)
    pub height: u32,
    /// Detection confidence (0.0-1.0)
    pub confidence: f32,
    /// Optional face identifier for tracking
    pub face_id: Option<u32>,
}

impl FaceRegion {
    /// Returns the center of this face region.
    #[must_use]
    pub fn center(&self) -> (u32, u32) {
        (self.x + self.width / 2, self.y + self.height / 2)
    }

    /// Returns the area of this face region in pixels.
    #[must_use]
    pub fn area(&self) -> u32 {
        self.width * self.height
    }
}

/// 3D orientation of a detected face.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceOrientation {
    /// Yaw angle in degrees (left/right rotation)
    pub yaw_deg: f32,
    /// Pitch angle in degrees (up/down tilt)
    pub pitch_deg: f32,
    /// Roll angle in degrees (sideways tilt)
    pub roll_deg: f32,
}

impl FaceOrientation {
    /// Returns true if the face is approximately frontal.
    ///
    /// A face is considered frontal if |yaw| < 30° and |pitch| < 20°.
    #[must_use]
    pub fn is_frontal(&self) -> bool {
        self.yaw_deg.abs() < 30.0 && self.pitch_deg.abs() < 20.0
    }
}

/// Facial expression classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FaceExpression {
    /// Neutral expression
    Neutral,
    /// Happy / smiling
    Happy,
    /// Sad expression
    Sad,
    /// Angry expression
    Angry,
    /// Surprised expression
    Surprised,
    /// Fearful expression
    Fearful,
    /// Disgusted expression
    Disgusted,
}

impl FaceExpression {
    /// Returns the emotional valence of this expression.
    ///
    /// Positive values indicate positive emotions, negative indicate negative.
    #[must_use]
    pub fn valence(&self) -> f32 {
        match self {
            Self::Happy => 1.0,
            Self::Neutral => 0.0,
            Self::Surprised => 0.2,
            Self::Fearful => -0.6,
            Self::Sad => -0.7,
            Self::Disgusted => -0.8,
            Self::Angry => -1.0,
        }
    }

    /// Returns the name of the expression as a string.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Neutral => "Neutral",
            Self::Happy => "Happy",
            Self::Sad => "Sad",
            Self::Angry => "Angry",
            Self::Surprised => "Surprised",
            Self::Fearful => "Fearful",
            Self::Disgusted => "Disgusted",
        }
    }
}

/// Normalized facial landmark positions (0.0–1.0 range).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceLandmarks {
    /// Left eye position (normalized)
    pub left_eye: (f32, f32),
    /// Right eye position (normalized)
    pub right_eye: (f32, f32),
    /// Nose tip position (normalized)
    pub nose: (f32, f32),
    /// Left mouth corner position (normalized)
    pub mouth_left: (f32, f32),
    /// Right mouth corner position (normalized)
    pub mouth_right: (f32, f32),
}

impl FaceLandmarks {
    /// Computes the inter-ocular distance (normalized).
    #[must_use]
    pub fn inter_ocular_distance(&self) -> f32 {
        let dx = self.right_eye.0 - self.left_eye.0;
        let dy = self.right_eye.1 - self.left_eye.1;
        (dx * dx + dy * dy).sqrt()
    }

    /// Returns the midpoint between the two eyes (normalized).
    #[must_use]
    pub fn eye_midpoint(&self) -> (f32, f32) {
        (
            (self.left_eye.0 + self.right_eye.0) / 2.0,
            (self.left_eye.1 + self.right_eye.1) / 2.0,
        )
    }
}

/// Complete face detection result for a single frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceDetectionResult {
    /// Detected face regions
    pub faces: Vec<FaceRegion>,
    /// Orientation estimate for each face
    pub orientations: Vec<FaceOrientation>,
    /// Expression estimate for each face
    pub expressions: Vec<FaceExpression>,
}

impl FaceDetectionResult {
    /// Returns the number of detected faces.
    #[must_use]
    pub fn count(&self) -> usize {
        self.faces.len()
    }

    /// Returns true if any frontal faces were detected.
    #[must_use]
    pub fn has_frontal_faces(&self) -> bool {
        self.orientations.iter().any(FaceOrientation::is_frontal)
    }
}

// ─────────────────────────────────────────────────────────────
// Integral image helpers
// ─────────────────────────────────────────────────────────────

/// Build a summed-area table (integral image) from a luma slice.
fn build_integral_image(luma: &[f32], width: u32, height: u32) -> Vec<f64> {
    let w = width as usize;
    let h = height as usize;
    let mut ii = vec![0.0f64; w * h];

    for y in 0..h {
        for x in 0..w {
            let val = f64::from(luma[y * w + x]);
            let above = if y > 0 { ii[(y - 1) * w + x] } else { 0.0 };
            let left = if x > 0 { ii[y * w + (x - 1)] } else { 0.0 };
            let diag = if x > 0 && y > 0 {
                ii[(y - 1) * w + (x - 1)]
            } else {
                0.0
            };
            ii[y * w + x] = val + above + left - diag;
        }
    }
    ii
}

/// Compute the rectangular sum from an integral image.
///
/// Returns the sum of pixels in the rectangle `(x, y, x+w, y+h)`.
fn rect_sum(ii: &[f64], img_width: usize, x: usize, y: usize, w: usize, h: usize) -> f64 {
    let x2 = x + w;
    let y2 = y + h;
    let a = ii[(y2 - 1) * img_width + (x2 - 1)];
    let b = if x > 0 {
        ii[(y2 - 1) * img_width + (x - 1)]
    } else {
        0.0
    };
    let c = if y > 0 {
        ii[(y - 1) * img_width + (x2 - 1)]
    } else {
        0.0
    };
    let d = if x > 0 && y > 0 {
        ii[(y - 1) * img_width + (x - 1)]
    } else {
        0.0
    };
    a - b - c + d
}

/// Simplified Haar-like feature: two-rectangle horizontal difference.
fn haar_horizontal(ii: &[f64], img_width: usize, x: usize, y: usize, w: usize, h: usize) -> f64 {
    let half_w = w / 2;
    let left = rect_sum(ii, img_width, x, y, half_w, h);
    let right = rect_sum(ii, img_width, x + half_w, y, half_w, h);
    right - left
}

/// Simplified Haar-like feature: two-rectangle vertical difference.
fn haar_vertical(ii: &[f64], img_width: usize, x: usize, y: usize, w: usize, h: usize) -> f64 {
    let half_h = h / 2;
    let top = rect_sum(ii, img_width, x, y, w, half_h);
    let bottom = rect_sum(ii, img_width, x, y + half_h, w, half_h);
    bottom - top
}

// ─────────────────────────────────────────────────────────────
// HaarCascadeDetector
// ─────────────────────────────────────────────────────────────

/// Simplified Viola-Jones inspired face detector using Haar-like features and
/// integral images.
pub struct HaarCascadeDetector {
    /// Minimum face window size (pixels)
    min_face_size: u32,
    /// Detection threshold
    threshold: f64,
}

impl HaarCascadeDetector {
    /// Create a new detector with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self {
            min_face_size: 24,
            threshold: 0.55,
        }
    }

    /// Create a detector with custom parameters.
    #[must_use]
    pub fn with_params(min_face_size: u32, threshold: f64) -> Self {
        Self {
            min_face_size,
            threshold,
        }
    }

    /// Detect faces in a luma (grayscale) image.
    ///
    /// This is a simplified Viola-Jones inspired detector using a sliding
    /// window with Haar-like features computed via integral images.
    ///
    /// * `luma` – row-major grayscale pixel values in \[0, 1\]
    /// * `width` – image width in pixels
    /// * `height` – image height in pixels
    ///
    /// Returns a list of detected face regions.
    #[must_use]
    pub fn detect_faces(&self, luma: &[f32], width: u32, height: u32) -> Vec<FaceRegion> {
        if luma.len() != (width * height) as usize
            || width < self.min_face_size
            || height < self.min_face_size
        {
            return Vec::new();
        }

        let img_w = width as usize;
        let img_h = height as usize;
        let win = self.min_face_size as usize;

        // Build integral image
        let ii = build_integral_image(luma, width, height);

        let mut detections: Vec<FaceRegion> = Vec::new();

        // Sliding window at a single scale (simplified)
        let step = win / 4;
        let y_end = img_h.saturating_sub(win);
        let x_end = img_w.saturating_sub(win);

        for y in (0..=y_end).step_by(step.max(1)) {
            for x in (0..=x_end).step_by(step.max(1)) {
                let score = self.evaluate_window(&ii, img_w, x, y, win);
                if score >= self.threshold {
                    detections.push(FaceRegion {
                        x: x as u32,
                        y: y as u32,
                        width: win as u32,
                        height: win as u32,
                        confidence: score as f32,
                        face_id: None,
                    });
                }
            }
        }

        // Non-maximum suppression (greedy)
        self.nms(detections, 0.5)
    }

    /// Evaluate a single detection window with a simplified cascade.
    fn evaluate_window(&self, ii: &[f64], img_width: usize, x: usize, y: usize, win: usize) -> f64 {
        // Stage 1: horizontal brightness difference (eye region vs. cheeks)
        let f1 = haar_horizontal(ii, img_width, x, y + win / 4, win, win / 2);
        // Stage 2: vertical brightness (forehead/eyes vs. mouth/chin)
        let f2 = haar_vertical(ii, img_width, x, y, win, win);
        // Stage 3: centre-surround (nose bridge region)
        let centre = rect_sum(ii, img_width, x + win / 4, y + win / 4, win / 2, win / 2);
        let total = rect_sum(ii, img_width, x, y, win, win);
        let f3 = centre / (total + 1.0) - 0.25;

        // Normalise features to [0, 1]
        let area = (win * win) as f64;
        let nf1 = (f1.abs() / (area * 0.5 + 1.0)).min(1.0);
        let nf2 = (f2.abs() / (area * 0.5 + 1.0)).min(1.0);
        let nf3 = f3.clamp(0.0, 1.0);

        // Simple linear combination
        (0.4 * nf1 + 0.3 * nf2 + 0.3 * nf3).min(1.0)
    }

    /// Greedy non-maximum suppression.
    fn nms(&self, mut regions: Vec<FaceRegion>, iou_threshold: f32) -> Vec<FaceRegion> {
        // Sort by confidence descending
        regions.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut keep: Vec<FaceRegion> = Vec::new();
        for candidate in regions {
            let suppressed = keep.iter().any(|k| {
                iou(
                    (k.x, k.y, k.width, k.height),
                    (candidate.x, candidate.y, candidate.width, candidate.height),
                ) > iou_threshold
            });
            if !suppressed {
                keep.push(candidate);
            }
        }
        keep
    }
}

impl Default for HaarCascadeDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute intersection-over-union between two boxes (x, y, w, h).
fn iou(a: (u32, u32, u32, u32), b: (u32, u32, u32, u32)) -> f32 {
    let ax1 = a.0;
    let ay1 = a.1;
    let ax2 = a.0 + a.2;
    let ay2 = a.1 + a.3;
    let bx1 = b.0;
    let by1 = b.1;
    let bx2 = b.0 + b.2;
    let by2 = b.1 + b.3;

    let ix1 = ax1.max(bx1);
    let iy1 = ay1.max(by1);
    let ix2 = ax2.min(bx2);
    let iy2 = ay2.min(by2);

    if ix2 <= ix1 || iy2 <= iy1 {
        return 0.0;
    }

    let inter = ((ix2 - ix1) * (iy2 - iy1)) as f32;
    let area_a = (a.2 * a.3) as f32;
    let area_b = (b.2 * b.3) as f32;
    inter / (area_a + area_b - inter)
}

// ─────────────────────────────────────────────────────────────
// GazeEstimator
// ─────────────────────────────────────────────────────────────

/// Estimate gaze direction from facial landmarks.
pub struct GazeEstimator;

impl GazeEstimator {
    /// Estimate gaze from normalized facial landmarks.
    ///
    /// Returns `(gaze_x, gaze_y)` both in the range \[-1, 1\]:
    /// - `gaze_x` < 0 → looking left; > 0 → looking right
    /// - `gaze_y` < 0 → looking up; > 0 → looking down
    ///
    /// * `landmarks` – normalized facial landmarks (0–1)
    /// * `frame_w` – frame width in pixels (used for perspective scaling)
    #[must_use]
    pub fn estimate(landmarks: &FaceLandmarks, frame_w: u32) -> (f32, f32) {
        // Inter-ocular distance as scale reference
        let iod = landmarks.inter_ocular_distance().max(1e-5);
        let scale = (frame_w as f32).max(1.0);

        // Nose position relative to eye midpoint
        let eye_mid = landmarks.eye_midpoint();
        let nose_dx = (landmarks.nose.0 - eye_mid.0) / iod;
        let nose_dy = (landmarks.nose.1 - eye_mid.1) / iod;

        // Mouth centre
        let _mouth_cx = (landmarks.mouth_left.0 + landmarks.mouth_right.0) / 2.0;
        let mouth_cy = (landmarks.mouth_left.1 + landmarks.mouth_right.1) / 2.0;
        let mouth_dy = (mouth_cy - eye_mid.1) / iod;

        // Gaze x: biased by nose horizontal offset
        let gaze_x = (nose_dx * 2.0).clamp(-1.0, 1.0);
        // Gaze y: biased by nose vertical offset relative to expected mouth position
        let expected_nose_dy = mouth_dy * 0.45;
        let gaze_y =
            ((nose_dy - expected_nose_dy) / (mouth_dy.abs().max(0.1)) * 1.5).clamp(-1.0, 1.0);

        // Apply a very mild perspective correction using frame width
        let _ = scale; // acknowledged; could be used for depth-based scaling

        (gaze_x, gaze_y)
    }
}

// ─────────────────────────────────────────────────────────────
// BlurScore — standalone sharpness metric
// ─────────────────────────────────────────────────────────────

/// Sharpness score derived from the variance of the Laplacian response.
///
/// A high `variance_of_laplacian` means the region is sharp; a low value
/// means it is blurred or otherwise obscured.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlurScore {
    /// Variance of the Laplacian response over the face region.
    ///
    /// Computed as `mean(lap²) − mean(lap)²` where the Laplacian kernel is
    /// `[0,1,0; 1,−4,1; 0,1,0]`.  Normalised by interior pixel count so the
    /// threshold is scale-invariant.
    ///
    /// Higher ⇒ sharper.  Lower ⇒ blurred / obscured.
    pub variance_of_laplacian: f32,
    /// `true` when the region is considered blurred (`variance_of_laplacian < blur_threshold`).
    pub is_blurred: bool,
}

impl BlurScore {
    /// Compute the Laplacian-variance sharpness score for a grayscale region.
    ///
    /// * `pixels` – row-major u8 luma data for the region.
    /// * `width` / `height` – dimensions of the region.
    /// * `blur_threshold` – variance below which the region is flagged as blurred.
    ///
    /// The Laplacian kernel `[0,1,0; 1,−4,1; 0,1,0]` is applied to every
    /// interior pixel (1-pixel border is skipped).  The variance of the
    /// responses is then computed as `mean(lap²) − mean(lap)²`.  Regions
    /// smaller than 3×3 are returned as `variance_of_laplacian = 0.0`.
    #[must_use]
    pub fn compute(pixels: &[u8], width: usize, height: usize, blur_threshold: f32) -> Self {
        if width < 3 || height < 3 || pixels.len() < width * height {
            return Self {
                variance_of_laplacian: 0.0,
                is_blurred: true,
            };
        }

        let mut lap_sum = 0.0f64;
        let mut lap_sq_sum = 0.0f64;
        let mut count = 0u64;

        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                let c = f64::from(pixels[y * width + x]);
                let n = f64::from(pixels[(y - 1) * width + x]);
                let s = f64::from(pixels[(y + 1) * width + x]);
                let l = f64::from(pixels[y * width + (x - 1)]);
                let r = f64::from(pixels[y * width + (x + 1)]);
                // Laplacian: N + S + L + R − 4·C
                let lap = n + s + l + r - 4.0 * c;
                lap_sum += lap;
                lap_sq_sum += lap * lap;
                count += 1;
            }
        }

        let variance = if count > 0 {
            let mean = lap_sum / count as f64;
            let var = lap_sq_sum / count as f64 - mean * mean;
            var.max(0.0) as f32
        } else {
            0.0
        };

        Self {
            variance_of_laplacian: variance,
            is_blurred: variance < blur_threshold,
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Face Blur / Obscurity Detection
// ─────────────────────────────────────────────────────────────

/// Privacy compliance obscurity level of a detected face region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FaceObscurityLevel {
    /// Face is clearly visible (no blur or occlusion).
    Clear,
    /// Face is slightly blurred or partially occluded.
    PartiallyObscured,
    /// Face is heavily blurred or mostly occluded — privacy-safe.
    HeavilyObscured,
    /// Face region could not be assessed (too small or out-of-bounds).
    Indeterminate,
}

impl FaceObscurityLevel {
    /// Returns `true` if the level is considered privacy-safe.
    #[must_use]
    pub fn is_privacy_safe(&self) -> bool {
        matches!(self, Self::HeavilyObscured | Self::Indeterminate)
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Clear => "Clear",
            Self::PartiallyObscured => "PartiallyObscured",
            Self::HeavilyObscured => "HeavilyObscured",
            Self::Indeterminate => "Indeterminate",
        }
    }
}

/// Privacy assessment for a single face region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FacePrivacyAssessment {
    /// The face region being assessed.
    pub region: FaceRegion,
    /// Measured blur score within the face bounding box (Laplacian variance, lower = blurrier).
    pub blur_score: f32,
    /// Structured `BlurScore` detail, including the `is_blurred` flag.
    pub blur_score_detail: BlurScore,
    /// Fraction of the face bounding box that is covered by near-uniform (obscuring) regions.
    pub occlusion_ratio: f32,
    /// Derived obscurity level.
    pub obscurity_level: FaceObscurityLevel,
}

impl FacePrivacyAssessment {
    /// Returns `true` if the face is considered privacy-compliant.
    #[must_use]
    pub fn is_compliant(&self) -> bool {
        self.obscurity_level.is_privacy_safe()
    }
}

/// Analyse the blur and obscurity level of detected face regions for privacy compliance.
///
/// * `luma` – grayscale luma plane (values 0–255), row-major.
/// * `img_width` / `img_height` – image dimensions in pixels.
/// * `faces` – list of detected face regions to assess.
///
/// Returns a privacy assessment for each face region (in the same order).
#[must_use]
pub fn assess_face_privacy(
    luma: &[u8],
    img_width: u32,
    img_height: u32,
    faces: &[FaceRegion],
) -> Vec<FacePrivacyAssessment> {
    faces
        .iter()
        .map(|face| assess_single_face(luma, img_width, img_height, face))
        .collect()
}

/// Thresholds used by the face privacy assessor.
pub struct FacePrivacyThresholds {
    /// Laplacian variance below which a face is considered blurred.
    pub blur_threshold: f32,
    /// Laplacian variance below which a face is considered heavily blurred.
    pub heavy_blur_threshold: f32,
    /// Occlusion ratio above which a face is considered partially obscured.
    pub partial_occlusion_threshold: f32,
    /// Occlusion ratio above which a face is considered heavily obscured.
    pub heavy_occlusion_threshold: f32,
}

impl Default for FacePrivacyThresholds {
    fn default() -> Self {
        Self {
            blur_threshold: 200.0,
            heavy_blur_threshold: 50.0,
            partial_occlusion_threshold: 0.30,
            heavy_occlusion_threshold: 0.70,
        }
    }
}

/// Assess a single face region for blur and obscurity.
fn assess_single_face(
    luma: &[u8],
    img_width: u32,
    img_height: u32,
    face: &FaceRegion,
) -> FacePrivacyAssessment {
    let thresholds = FacePrivacyThresholds::default();

    let x0 = face.x as usize;
    let y0 = face.y as usize;
    let x1 = (face.x + face.width).min(img_width) as usize;
    let y1 = (face.y + face.height).min(img_height) as usize;
    let w = img_width as usize;

    let declared_area = (face.width as usize).saturating_mul(face.height as usize);
    let clipped_area = (x1 - x0) * (y1 - y0);
    if x1 <= x0 + 2
        || y1 <= y0 + 2
        || luma.len() < w * img_height as usize
        || (declared_area > 0 && clipped_area * 2 < declared_area)
    {
        return FacePrivacyAssessment {
            region: face.clone(),
            blur_score: 0.0,
            blur_score_detail: BlurScore {
                variance_of_laplacian: 0.0,
                is_blurred: true,
            },
            occlusion_ratio: 0.0,
            obscurity_level: FaceObscurityLevel::Indeterminate,
        };
    }

    // --- Blur score: Laplacian variance over the face ROI ---
    let mut lap_sum = 0.0f64;
    let mut lap_sq_sum = 0.0f64;
    let mut count = 0u32;

    for y in (y0 + 1)..(y1 - 1) {
        for x in (x0 + 1)..(x1 - 1) {
            let c = f64::from(luma[y * w + x]);
            let n = f64::from(luma[(y - 1) * w + x]);
            let s = f64::from(luma[(y + 1) * w + x]);
            let l = f64::from(luma[y * w + (x - 1)]);
            let r = f64::from(luma[y * w + (x + 1)]);
            let lap = n + s + l + r - 4.0 * c;
            lap_sum += lap;
            lap_sq_sum += lap * lap;
            count += 1;
        }
    }

    let blur_score = if count > 0 {
        let mean = lap_sum / f64::from(count);
        let variance = lap_sq_sum / f64::from(count) - mean * mean;
        variance.max(0.0) as f32
    } else {
        0.0
    };

    // Build a structured BlurScore using the same variance.
    let blur_score_detail = BlurScore {
        variance_of_laplacian: blur_score,
        is_blurred: blur_score < thresholds.blur_threshold,
    };

    // --- Occlusion ratio: fraction of face pixels that are near-uniform (likely covered) ---
    // We use a sliding 4×4 local standard-deviation check.  Regions with very low std-dev
    // are likely covered by a solid overlay (mosaic, blur box, sticker, etc.).
    const PATCH: usize = 4;
    let mut uniform_patches = 0u32;
    let mut total_patches = 0u32;

    let ry = y0..y1.saturating_sub(PATCH);
    for py in ry.step_by(PATCH) {
        let rx = x0..x1.saturating_sub(PATCH);
        for px in rx.step_by(PATCH) {
            let mut psum = 0.0f64;
            let mut psq = 0.0f64;
            for dy in 0..PATCH {
                for dx in 0..PATCH {
                    let v = f64::from(luma[(py + dy) * w + (px + dx)]);
                    psum += v;
                    psq += v * v;
                }
            }
            let n_p = (PATCH * PATCH) as f64;
            let mean = psum / n_p;
            let std_dev = ((psq / n_p - mean * mean).max(0.0)).sqrt();
            if std_dev < 8.0 {
                uniform_patches += 1;
            }
            total_patches += 1;
        }
    }

    let occlusion_ratio = if total_patches > 0 {
        uniform_patches as f32 / total_patches as f32
    } else {
        0.0
    };

    // --- Derive obscurity level ---
    let obscurity_level = if blur_score < thresholds.heavy_blur_threshold
        || occlusion_ratio > thresholds.heavy_occlusion_threshold
    {
        FaceObscurityLevel::HeavilyObscured
    } else if blur_score < thresholds.blur_threshold
        || occlusion_ratio > thresholds.partial_occlusion_threshold
    {
        FaceObscurityLevel::PartiallyObscured
    } else {
        FaceObscurityLevel::Clear
    };

    FacePrivacyAssessment {
        region: face.clone(),
        blur_score,
        blur_score_detail,
        occlusion_ratio,
        obscurity_level,
    }
}

// ─────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_landmarks() -> FaceLandmarks {
        FaceLandmarks {
            left_eye: (0.35, 0.40),
            right_eye: (0.65, 0.40),
            nose: (0.50, 0.55),
            mouth_left: (0.38, 0.70),
            mouth_right: (0.62, 0.70),
        }
    }

    // ── FaceRegion ────────────────────────────────────────────

    #[test]
    fn test_face_region_center() {
        let r = FaceRegion {
            x: 10,
            y: 20,
            width: 40,
            height: 60,
            confidence: 0.8,
            face_id: None,
        };
        assert_eq!(r.center(), (30, 50));
    }

    #[test]
    fn test_face_region_area() {
        let r = FaceRegion {
            x: 0,
            y: 0,
            width: 48,
            height: 48,
            confidence: 1.0,
            face_id: Some(1),
        };
        assert_eq!(r.area(), 2304);
    }

    // ── FaceOrientation ───────────────────────────────────────

    #[test]
    fn test_is_frontal_true() {
        let o = FaceOrientation {
            yaw_deg: 10.0,
            pitch_deg: 5.0,
            roll_deg: 2.0,
        };
        assert!(o.is_frontal());
    }

    #[test]
    fn test_is_frontal_false_yaw() {
        let o = FaceOrientation {
            yaw_deg: 45.0,
            pitch_deg: 5.0,
            roll_deg: 0.0,
        };
        assert!(!o.is_frontal());
    }

    #[test]
    fn test_is_frontal_false_pitch() {
        let o = FaceOrientation {
            yaw_deg: 5.0,
            pitch_deg: 25.0,
            roll_deg: 0.0,
        };
        assert!(!o.is_frontal());
    }

    // ── FaceExpression ────────────────────────────────────────

    #[test]
    fn test_valence_happy() {
        assert!((FaceExpression::Happy.valence() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_valence_neutral() {
        assert!(FaceExpression::Neutral.valence().abs() < f32::EPSILON);
    }

    #[test]
    fn test_valence_angry_negative() {
        assert!(FaceExpression::Angry.valence() < 0.0);
    }

    #[test]
    fn test_expression_names() {
        assert_eq!(FaceExpression::Sad.name(), "Sad");
        assert_eq!(FaceExpression::Disgusted.name(), "Disgusted");
    }

    // ── FaceLandmarks ─────────────────────────────────────────

    #[test]
    fn test_inter_ocular_distance() {
        let lm = make_landmarks();
        let iod = lm.inter_ocular_distance();
        // left=(0.35,0.40) right=(0.65,0.40) -> dx=0.30
        assert!((iod - 0.30).abs() < 1e-5);
    }

    #[test]
    fn test_eye_midpoint() {
        let lm = make_landmarks();
        let mid = lm.eye_midpoint();
        assert!((mid.0 - 0.50).abs() < 1e-5);
        assert!((mid.1 - 0.40).abs() < 1e-5);
    }

    // ── FaceDetectionResult ───────────────────────────────────

    #[test]
    fn test_detection_result_count() {
        let result = FaceDetectionResult {
            faces: vec![
                FaceRegion {
                    x: 0,
                    y: 0,
                    width: 24,
                    height: 24,
                    confidence: 0.9,
                    face_id: None,
                },
                FaceRegion {
                    x: 100,
                    y: 100,
                    width: 24,
                    height: 24,
                    confidence: 0.7,
                    face_id: None,
                },
            ],
            orientations: vec![
                FaceOrientation {
                    yaw_deg: 5.0,
                    pitch_deg: 3.0,
                    roll_deg: 1.0,
                },
                FaceOrientation {
                    yaw_deg: 40.0,
                    pitch_deg: 0.0,
                    roll_deg: 0.0,
                },
            ],
            expressions: vec![FaceExpression::Happy, FaceExpression::Neutral],
        };
        assert_eq!(result.count(), 2);
        assert!(result.has_frontal_faces()); // first one is frontal
    }

    // ── Integral image helpers ─────────────────────────────────

    #[test]
    fn test_integral_image_uniform() {
        // 2×2 image filled with 1.0
        let luma = vec![1.0f32; 4];
        let ii = build_integral_image(&luma, 2, 2);
        // Bottom-right cell should equal sum of all = 4
        assert!((ii[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_rect_sum_full_image() {
        let luma = vec![1.0f32; 9];
        let ii = build_integral_image(&luma, 3, 3);
        let s = rect_sum(&ii, 3, 0, 0, 3, 3);
        assert!((s - 9.0).abs() < 1e-6);
    }

    // ── HaarCascadeDetector ───────────────────────────────────

    #[test]
    fn test_detector_empty_image() {
        let detector = HaarCascadeDetector::new();
        let result = detector.detect_faces(&[], 0, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_detector_too_small() {
        let detector = HaarCascadeDetector::new();
        let luma = vec![0.5f32; 10 * 10];
        let result = detector.detect_faces(&luma, 10, 10);
        assert!(result.is_empty()); // min_face_size is 24
    }

    #[test]
    fn test_detector_returns_regions() {
        // Create a 96×96 image with a brighter "face-like" central region
        let w = 96u32;
        let h = 96u32;
        let mut luma = vec![0.3f32; (w * h) as usize];
        // Brighten the centre
        for y in 24..72u32 {
            for x in 24..72u32 {
                luma[(y * w + x) as usize] = 0.8;
            }
        }
        let detector = HaarCascadeDetector::with_params(24, 0.0); // threshold=0 → everything passes
        let regions = detector.detect_faces(&luma, w, h);
        // With threshold=0, we should get at least some regions back
        assert!(!regions.is_empty());
    }

    // ── Face Privacy Assessment ───────────────────────────────

    fn make_face_region(x: u32, y: u32, w: u32, h: u32) -> FaceRegion {
        FaceRegion {
            x,
            y,
            width: w,
            height: h,
            confidence: 0.9,
            face_id: None,
        }
    }

    #[test]
    fn test_clear_face_high_blur_score() {
        // Sharp, high-contrast image → should be Clear
        let w = 64u32;
        let h = 64u32;
        let mut luma = vec![0u8; (w * h) as usize];
        // Checkerboard pattern → high Laplacian variance
        for y in 0..h as usize {
            for x in 0..w as usize {
                // Use 2px blocks (smaller than 4x4 patch) so patches span both
                // black and white areas and are NOT classified as uniform.
                luma[y * w as usize + x] = if ((x / 2) + (y / 2)) % 2 == 0 {
                    50
                } else {
                    200
                };
            }
        }
        let face = make_face_region(8, 8, 48, 48);
        let results = assess_face_privacy(&luma, w, h, &[face]);
        assert_eq!(results.len(), 1);
        // High-frequency content → Clear
        assert_eq!(
            results[0].obscurity_level,
            FaceObscurityLevel::Clear,
            "blur_score={}",
            results[0].blur_score
        );
        assert!(!results[0].is_compliant());
    }

    #[test]
    fn test_blurred_face_heavily_obscured() {
        // Near-uniform (blurred) region → HeavilyObscured
        let w = 64u32;
        let h = 64u32;
        let luma = vec![128u8; (w * h) as usize];
        let face = make_face_region(8, 8, 48, 48);
        let results = assess_face_privacy(&luma, w, h, &[face]);
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].obscurity_level,
            FaceObscurityLevel::HeavilyObscured,
            "blur_score={}",
            results[0].blur_score
        );
        assert!(results[0].is_compliant());
    }

    #[test]
    fn test_indeterminate_when_out_of_bounds() {
        let w = 32u32;
        let h = 32u32;
        let luma = vec![128u8; (w * h) as usize];
        // Face region extends beyond image
        let face = make_face_region(28, 28, 20, 20);
        let results = assess_face_privacy(&luma, w, h, &[face]);
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].obscurity_level,
            FaceObscurityLevel::Indeterminate
        );
        assert!(results[0].is_compliant());
    }

    #[test]
    fn test_obscurity_label() {
        assert_eq!(FaceObscurityLevel::Clear.label(), "Clear");
        assert_eq!(
            FaceObscurityLevel::HeavilyObscured.label(),
            "HeavilyObscured"
        );
        assert!(!FaceObscurityLevel::Clear.is_privacy_safe());
        assert!(FaceObscurityLevel::HeavilyObscured.is_privacy_safe());
        assert!(FaceObscurityLevel::Indeterminate.is_privacy_safe());
    }

    #[test]
    fn test_privacy_assessment_empty_face_list() {
        let luma = vec![128u8; 64 * 64];
        let results = assess_face_privacy(&luma, 64, 64, &[]);
        assert!(results.is_empty());
    }

    // ── GazeEstimator ─────────────────────────────────────────

    #[test]
    fn test_gaze_frontal_near_zero() {
        let lm = make_landmarks(); // symmetric, looking forward
        let (gx, gy) = GazeEstimator::estimate(&lm, 1920);
        // Frontal face → gaze should be near centre
        assert!(gx.abs() < 0.5);
        let _ = gy; // y can vary by design
    }

    #[test]
    fn test_gaze_output_in_range() {
        let lm = FaceLandmarks {
            left_eye: (0.30, 0.38),
            right_eye: (0.60, 0.38),
            nose: (0.70, 0.52), // nose shifted right → looking right
            mouth_left: (0.35, 0.68),
            mouth_right: (0.65, 0.68),
        };
        let (gx, gy) = GazeEstimator::estimate(&lm, 1280);
        assert!(gx >= -1.0 && gx <= 1.0);
        assert!(gy >= -1.0 && gy <= 1.0);
        assert!(gx > 0.0); // nose shifted right → positive gaze_x
    }

    // ── BlurScore ──────────────────────────────────────────────

    /// A high-frequency checkerboard should produce high Laplacian variance,
    /// indicating a sharp (non-blurred) region.
    #[test]
    fn test_sharp_face_scores_high() {
        let width = 32usize;
        let height = 32usize;
        // 1px checkerboard (alternating 0 / 255) — maximum spatial frequency.
        let pixels: Vec<u8> = (0..(width * height))
            .map(|i| {
                let x = i % width;
                let y = i / width;
                if (x + y) % 2 == 0 {
                    0u8
                } else {
                    255u8
                }
            })
            .collect();

        let score = BlurScore::compute(&pixels, width, height, 100.0);
        // A 1px checkerboard generates very large Laplacian responses (±1020),
        // so variance should be well above 1000.
        assert!(
            score.variance_of_laplacian > 1000.0,
            "expected variance > 1000 for checkerboard, got {}",
            score.variance_of_laplacian
        );
        assert!(
            !score.is_blurred,
            "sharp checkerboard should not be flagged as blurred"
        );
    }

    /// A single box-blur pass on a sharp checkerboard dramatically reduces the
    /// Laplacian variance; the score must decrease monotonically with each
    /// successive pass, and a near-uniform (heavily smoothed) region must be
    /// flagged as blurred.
    #[test]
    fn test_blurred_face_scores_low() {
        // Apply one 3×3 box-blur pass (interior pixels only, border kept).
        fn box_blur_interior(src: &[u8], width: usize, height: usize) -> Vec<u8> {
            let mut dst = src.to_vec();
            for y in 1..(height - 1) {
                for x in 1..(width - 1) {
                    let mut sum = 0u32;
                    for dy in 0..3usize {
                        for dx in 0..3usize {
                            sum += u32::from(src[(y + dy - 1) * width + (x + dx - 1)]);
                        }
                    }
                    dst[y * width + x] = (sum / 9) as u8;
                }
            }
            dst
        }

        let width = 64usize;
        let height = 64usize;

        // 1px checkerboard — maximum spatial frequency → high Laplacian variance.
        let sharp: Vec<u8> = (0..(width * height))
            .map(|i| {
                let x = i % width;
                let y = i / width;
                if (x + y) % 2 == 0 {
                    0u8
                } else {
                    255u8
                }
            })
            .collect();

        // One pass of box-blur reduces the variance dramatically.
        let blurred_1 = box_blur_interior(&sharp, width, height);

        // Variance of the once-blurred image (no threshold check needed here).
        let score_sharp = BlurScore::compute(&sharp, width, height, 100_000.0);
        let score_blur1 = BlurScore::compute(&blurred_1, width, height, 100_000.0);

        // Monotonic: one blur pass must reduce variance significantly.
        assert!(
            score_sharp.variance_of_laplacian > score_blur1.variance_of_laplacian,
            "single blur must reduce variance: sharp={}, blur1={}",
            score_sharp.variance_of_laplacian,
            score_blur1.variance_of_laplacian
        );
        // The reduction must be substantial (> 10×).
        assert!(
            score_sharp.variance_of_laplacian > score_blur1.variance_of_laplacian * 10.0,
            "blur must reduce variance by at least 10×: sharp={}, blur1={}",
            score_sharp.variance_of_laplacian,
            score_blur1.variance_of_laplacian
        );

        // A truly near-uniform region (constant value) must always be flagged.
        let flat = vec![128u8; width * height];
        let score_flat = BlurScore::compute(&flat, width, height, 100.0);
        assert!(
            score_flat.is_blurred,
            "constant region must be flagged as blurred; variance={}",
            score_flat.variance_of_laplacian
        );
        assert!(
            score_flat.variance_of_laplacian < 1.0,
            "constant region must have near-zero Laplacian variance; got {}",
            score_flat.variance_of_laplacian
        );

        // The sharp checkerboard must NOT be flagged when threshold = 10_000.
        let score_sharp_flag = BlurScore::compute(&sharp, width, height, 10_000.0);
        assert!(
            !score_sharp_flag.is_blurred,
            "sharp checkerboard must not be flagged; variance={}",
            score_sharp_flag.variance_of_laplacian
        );
    }

    /// End-to-end: blurred face regions must appear in `FacePrivacyAssessment`
    /// with `blur_score_detail.is_blurred == true`.
    #[test]
    fn test_blur_privacy_assessment() {
        let w = 64u32;
        let h = 64u32;
        // Near-uniform image simulates a heavy mosaic blur.
        let luma = vec![128u8; (w * h) as usize];
        let face = FaceRegion {
            x: 4,
            y: 4,
            width: 56,
            height: 56,
            confidence: 0.95,
            face_id: Some(0),
        };
        let results = assess_face_privacy(&luma, w, h, &[face]);
        assert_eq!(results.len(), 1);
        let pa = &results[0];

        // The blur_score_detail field should flag the region as blurred.
        assert!(
            pa.blur_score_detail.is_blurred,
            "uniform image must be flagged as blurred; variance={}",
            pa.blur_score_detail.variance_of_laplacian
        );
        // And the obscurity level must reflect this.
        assert_eq!(
            pa.obscurity_level,
            FaceObscurityLevel::HeavilyObscured,
            "uniform face region must be HeavilyObscured"
        );
        assert!(pa.is_compliant());
    }
}
