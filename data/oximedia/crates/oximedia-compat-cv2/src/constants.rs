//! OpenCV-compatible constants for `oximedia-compat-cv2`.
//!
//! Constants are grouped into sub-modules by category. All are re-exported at
//! the crate root so `use oximedia_compat_cv2::*;` brings all constants into
//! scope (matching the `cv2.IMREAD_COLOR` style from Python).

// Mixed-case constant names like COLOR_BGR2Lab mirror the OpenCV API exactly.
#![allow(non_upper_case_globals)]

// ── imread flags ──────────────────────────────────────────────────────────────

/// imread / imdecode flags.
pub mod imread {
    /// Load as 3-channel BGR colour image.
    pub const IMREAD_COLOR: i32 = 1;
    /// Load as grayscale image.
    pub const IMREAD_GRAYSCALE: i32 = 0;
    /// Load including alpha channel (BGRA).
    pub const IMREAD_UNCHANGED: i32 = -1;
    /// Any bit depth, single-channel.
    pub const IMREAD_ANYDEPTH: i32 = 2;
    /// Any channel count.
    pub const IMREAD_ANYCOLOR: i32 = 4;
}

// ── Color conversion codes ────────────────────────────────────────────────────

/// Color space conversion codes (`cvtColor`).
pub mod color {
    /// BGR → BGRA.
    pub const COLOR_BGR2BGRA: i32 = 0;
    /// BGRA → BGR.
    pub const COLOR_BGRA2BGR: i32 = 1;
    /// BGR → RGBA.
    pub const COLOR_BGR2RGBA: i32 = 2;
    /// RGBA → BGR.
    pub const COLOR_RGBA2BGR: i32 = 3;
    /// BGR → RGB (and RGB → BGR — same operation).
    pub const COLOR_BGR2RGB: i32 = 4;
    /// RGB → BGR (same value as `COLOR_BGR2RGB`).
    pub const COLOR_RGB2BGR: i32 = 4;
    /// BGRA → RGBA.
    pub const COLOR_BGRA2RGBA: i32 = 5;
    /// BGR → grayscale.
    pub const COLOR_BGR2GRAY: i32 = 6;
    /// RGB → grayscale.
    pub const COLOR_RGB2GRAY: i32 = 7;
    /// Grayscale → BGR.
    pub const COLOR_GRAY2BGR: i32 = 8;
    /// Grayscale → RGB (same value as `COLOR_GRAY2BGR`).
    pub const COLOR_GRAY2RGB: i32 = 8;
    /// Grayscale → BGRA.
    pub const COLOR_GRAY2BGRA: i32 = 9;
    /// BGRA → grayscale.
    pub const COLOR_BGRA2GRAY: i32 = 10;
    /// BGR → HSV.
    pub const COLOR_BGR2HSV: i32 = 40;
    /// RGB → HSV.
    pub const COLOR_RGB2HSV: i32 = 41;
    /// BGR → CIE Lab.
    #[allow(non_upper_case_globals)]
    pub const COLOR_BGR2Lab: i32 = 44;
    /// RGB → CIE Lab.
    #[allow(non_upper_case_globals)]
    pub const COLOR_RGB2Lab: i32 = 45;
    /// BGR → HLS.
    pub const COLOR_BGR2HLS: i32 = 52;
    /// CIE Lab → BGR.
    #[allow(non_upper_case_globals)]
    pub const COLOR_Lab2BGR: i32 = 56;
    /// CIE Lab → RGB.
    #[allow(non_upper_case_globals)]
    pub const COLOR_Lab2RGB: i32 = 57;
    /// HSV → BGR.
    pub const COLOR_HSV2BGR: i32 = 54;
    /// HSV → RGB.
    pub const COLOR_HSV2RGB: i32 = 55;
    /// HLS → BGR.
    pub const COLOR_HLS2BGR: i32 = 60;
    /// HLS → RGB.
    pub const COLOR_HLS2RGB: i32 = 61;
    /// BGR → YUV.
    pub const COLOR_BGR2YUV: i32 = 82;
    /// RGB → YUV.
    pub const COLOR_RGB2YUV: i32 = 83;
    /// YUV → BGR.
    pub const COLOR_YUV2BGR: i32 = 84;
    /// YUV → RGB.
    pub const COLOR_YUV2RGB: i32 = 85;
    /// YUV 4:2:0 (NV12) → BGR.
    pub const COLOR_YUV2BGR_NV12: i32 = 90;
    /// YUV 4:2:0 → grayscale.
    pub const COLOR_YUV2GRAY_420: i32 = 106;
}

// ── Interpolation flags ───────────────────────────────────────────────────────

/// Interpolation method flags.
pub mod interpolation {
    /// Nearest-neighbour interpolation.
    pub const INTER_NEAREST: i32 = 0;
    /// Bilinear interpolation.
    pub const INTER_LINEAR: i32 = 1;
    /// Bicubic interpolation.
    pub const INTER_CUBIC: i32 = 2;
    /// Resampling using pixel area relation.
    pub const INTER_AREA: i32 = 3;
    /// Lanczos interpolation over 8×8 neighbourhood.
    pub const INTER_LANCZOS4: i32 = 4;
    /// Bit-exact bilinear interpolation.
    pub const INTER_LINEAR_EXACT: i32 = 5;
}

// ── Border types ──────────────────────────────────────────────────────────────

/// Border / padding modes.
pub mod border {
    /// `iiiiii|abcdefgh|iiiiiii` where `i` is a constant value.
    pub const BORDER_CONSTANT: i32 = 0;
    /// `aaaaaa|abcdefgh|hhhhhhh`.
    pub const BORDER_REPLICATE: i32 = 1;
    /// `fedcba|abcdefgh|hgfedcb`.
    pub const BORDER_WRAP: i32 = 3;
    /// `cdefgh|abcdefgh|abcdefg` (same as `BORDER_REFLECT_101`).
    pub const BORDER_REFLECT: i32 = 4;
    /// `cdefgh|abcdefgh|abcdefg` (same as `BORDER_REFLECT`).
    pub const BORDER_REFLECT_101: i32 = 4;
    /// Transparent border (no border).
    pub const BORDER_TRANSPARENT: i32 = 5;
    /// Alias for `BORDER_REFLECT_101` (OpenCV default).
    pub const BORDER_DEFAULT: i32 = 4;
}

// ── Threshold types ───────────────────────────────────────────────────────────

/// Thresholding operation types.
pub mod threshold {
    /// `dst(x,y) = maxval if src(x,y) > thresh else 0`.
    pub const THRESH_BINARY: i32 = 0;
    /// `dst(x,y) = 0 if src(x,y) > thresh else maxval`.
    pub const THRESH_BINARY_INV: i32 = 1;
    /// `dst(x,y) = thresh if src(x,y) > thresh else src(x,y)`.
    pub const THRESH_TRUNC: i32 = 2;
    /// `dst(x,y) = src(x,y) if src(x,y) > thresh else 0`.
    pub const THRESH_TOZERO: i32 = 3;
    /// `dst(x,y) = 0 if src(x,y) > thresh else src(x,y)`.
    pub const THRESH_TOZERO_INV: i32 = 4;
    /// Threshold-type mask.
    pub const THRESH_MASK: i32 = 7;
    /// Flag: use Otsu's method to determine optimal threshold.
    pub const THRESH_OTSU: i32 = 8;
    /// Flag: use triangle algorithm to determine optimal threshold.
    pub const THRESH_TRIANGLE: i32 = 16;
}

// ── Adaptive threshold ────────────────────────────────────────────────────────

/// Adaptive thresholding method selection.
pub mod adaptive_thresh {
    /// Threshold value is the mean of the neighbourhood area.
    pub const ADAPTIVE_THRESH_MEAN_C: i32 = 0;
    /// Threshold value is the weighted sum (Gaussian window) of the neighbourhood.
    pub const ADAPTIVE_THRESH_GAUSSIAN_C: i32 = 1;
}

// ── Morphology shapes ─────────────────────────────────────────────────────────

/// Structuring-element shape for morphological operations.
pub mod morph_shape {
    /// Rectangular structuring element.
    pub const MORPH_RECT: i32 = 0;
    /// Cross-shaped structuring element.
    pub const MORPH_CROSS: i32 = 1;
    /// Elliptical structuring element.
    pub const MORPH_ELLIPSE: i32 = 2;
}

// ── Morphology operations ─────────────────────────────────────────────────────

/// Morphological operation selectors.
pub mod morph_op {
    /// Erosion.
    pub const MORPH_ERODE: i32 = 0;
    /// Dilation.
    pub const MORPH_DILATE: i32 = 1;
    /// Opening (erosion then dilation).
    pub const MORPH_OPEN: i32 = 2;
    /// Closing (dilation then erosion).
    pub const MORPH_CLOSE: i32 = 3;
    /// Morphological gradient (dilation minus erosion).
    pub const MORPH_GRADIENT: i32 = 4;
    /// Top hat (source minus opening).
    pub const MORPH_TOPHAT: i32 = 5;
    /// Black hat (closing minus source).
    pub const MORPH_BLACKHAT: i32 = 6;
}

// ── Contour retrieval modes ───────────────────────────────────────────────────

/// Contour retrieval mode for `findContours`.
pub mod contour_retr {
    /// Retrieve only the extreme outer contours.
    pub const RETR_EXTERNAL: i32 = 0;
    /// Retrieve all contours without establishing a hierarchy.
    pub const RETR_LIST: i32 = 1;
    /// Retrieve all contours and organise them into a two-level hierarchy.
    pub const RETR_CCOMP: i32 = 2;
    /// Retrieve all contours and reconstruct a full hierarchy.
    pub const RETR_TREE: i32 = 3;
}

// ── Contour approximation ─────────────────────────────────────────────────────

/// Contour approximation method for `findContours`.
pub mod chain_approx {
    /// Store all the contour points.
    pub const CHAIN_APPROX_NONE: i32 = 1;
    /// Compress horizontal, vertical, and diagonal segments.
    pub const CHAIN_APPROX_SIMPLE: i32 = 2;
    /// Apply Teh-Chin chain approximation algorithm (L1).
    pub const CHAIN_APPROX_TC89_L1: i32 = 3;
    /// Apply Teh-Chin chain approximation algorithm (k-cosine).
    pub const CHAIN_APPROX_TC89_KCOS: i32 = 4;
}

// ── VideoCapture properties ───────────────────────────────────────────────────

/// `VideoCapture::get` / `set` property identifiers.
pub mod cap_prop {
    /// Position in milliseconds.
    pub const CAP_PROP_POS_MSEC: i32 = 0;
    /// Zero-based index of the next frame to be decoded.
    pub const CAP_PROP_POS_FRAMES: i32 = 1;
    /// Relative position in the video file (0=start, 1=end).
    pub const CAP_PROP_POS_AVI_RATIO: i32 = 2;
    /// Width of the frames.
    pub const CAP_PROP_FRAME_WIDTH: i32 = 3;
    /// Height of the frames.
    pub const CAP_PROP_FRAME_HEIGHT: i32 = 4;
    /// Frame rate.
    pub const CAP_PROP_FPS: i32 = 5;
    /// 4-character codec code.
    pub const CAP_PROP_FOURCC: i32 = 6;
    /// Number of frames in the video file.
    pub const CAP_PROP_FRAME_COUNT: i32 = 7;
    /// Format of the Mat objects returned.
    pub const CAP_PROP_FORMAT: i32 = 8;
    /// Backend-specific value indicating the current capture mode.
    pub const CAP_PROP_MODE: i32 = 9;
}

// ── Rotation codes ────────────────────────────────────────────────────────────

/// Rotation codes for `rotate`.
pub mod rotate {
    /// Rotate 90° clockwise.
    pub const ROTATE_90_CLOCKWISE: i32 = 0;
    /// Rotate 180°.
    pub const ROTATE_180: i32 = 1;
    /// Rotate 90° counter-clockwise (270° clockwise).
    pub const ROTATE_90_COUNTERCLOCKWISE: i32 = 2;
}

// ── Font types ────────────────────────────────────────────────────────────────

/// Font identifiers for `putText`.
pub mod font {
    /// Normal-size sans-serif font.
    pub const FONT_HERSHEY_SIMPLEX: i32 = 0;
    /// Small-size sans-serif font.
    pub const FONT_HERSHEY_PLAIN: i32 = 1;
    /// Normal-size sans-serif font (more complex than simplex).
    pub const FONT_HERSHEY_DUPLEX: i32 = 2;
    /// Normal-size serif font.
    pub const FONT_HERSHEY_COMPLEX: i32 = 3;
    /// Normal-size serif font (more complex than complex).
    pub const FONT_HERSHEY_TRIPLEX: i32 = 4;
    /// Smaller version of `FONT_HERSHEY_COMPLEX`.
    pub const FONT_HERSHEY_COMPLEX_SMALL: i32 = 5;
    /// Hand-writing style font (simplex).
    pub const FONT_HERSHEY_SCRIPT_SIMPLEX: i32 = 6;
    /// Hand-writing style font (complex).
    pub const FONT_HERSHEY_SCRIPT_COMPLEX: i32 = 7;
    /// Flag to make any font italic.
    pub const FONT_ITALIC: i32 = 16;
}

// ── Line types ────────────────────────────────────────────────────────────────

/// Line connectivity / drawing modes.
pub mod line_type {
    /// 4-connected Bresenham line.
    pub const LINE_4: i32 = 4;
    /// 8-connected Bresenham line.
    pub const LINE_8: i32 = 8;
    /// Anti-aliased line.
    pub const LINE_AA: i32 = 16;
    /// Filled shape (used with `thickness` parameter).
    pub const FILLED: i32 = -1;
}

// ── Feature detector flags ────────────────────────────────────────────────────

/// ORB detector score type flags.
pub mod feature_flags {
    /// Use Harris score for keypoint ranking.
    pub const ORB_HARRIS_SCORE: i32 = 0;
    /// Use FAST score for keypoint ranking.
    pub const ORB_FAST_SCORE: i32 = 1;
}

// ── Hough transform ───────────────────────────────────────────────────────────

/// Hough transform variant selector.
pub mod hough {
    /// Classical or standard Hough transform.
    pub const HOUGH_STANDARD: i32 = 0;
    /// Probabilistic Hough transform.
    pub const HOUGH_PROBABILISTIC: i32 = 1;
    /// Multi-scale classical Hough transform.
    pub const HOUGH_MULTI_SCALE: i32 = 2;
    /// Hough gradient (for circles).
    pub const HOUGH_GRADIENT: i32 = 3;
    /// Alternative gradient-based Hough (for circles).
    pub const HOUGH_GRADIENT_ALT: i32 = 4;
}

// ── Norm types ────────────────────────────────────────────────────────────────

/// Norm type selectors for `norm` / `normalize`.
pub mod norm_type {
    /// Infinity norm (max absolute value).
    pub const NORM_INF: i32 = 1;
    /// L1 norm (sum of absolute values).
    pub const NORM_L1: i32 = 2;
    /// L2 norm (Euclidean distance).
    pub const NORM_L2: i32 = 4;
    /// L2 norm squared (sum of squares, not square-rooted).
    pub const NORM_L2SQR: i32 = 5;
    /// Hamming distance (population count of XOR — used by binary descriptors
    /// such as ORB/BRIEF/BRISK).
    pub const NORM_HAMMING: i32 = 6;
    /// Hamming distance computed on 2-bit chunks (used by some MIH variants).
    pub const NORM_HAMMING2: i32 = 7;
    /// Min–max normalization flag.
    pub const NORM_MINMAX: i32 = 32;
}

// ── Template matching methods ─────────────────────────────────────────────────

/// Template matching comparison methods.
pub mod template_match {
    /// Sum of squared differences.
    pub const TM_SQDIFF: i32 = 0;
    /// Normalised sum of squared differences.
    pub const TM_SQDIFF_NORMED: i32 = 1;
    /// Cross-correlation.
    pub const TM_CCORR: i32 = 2;
    /// Normalised cross-correlation.
    pub const TM_CCORR_NORMED: i32 = 3;
    /// Coefficient correlation.
    pub const TM_CCOEFF: i32 = 4;
    /// Normalised coefficient correlation.
    pub const TM_CCOEFF_NORMED: i32 = 5;
}

// ── Comparison operations ─────────────────────────────────────────────────────

/// Per-element comparison operation selectors.
pub mod compare {
    /// `a == b`.
    pub const CMP_EQ: i32 = 0;
    /// `a > b`.
    pub const CMP_GT: i32 = 1;
    /// `a >= b`.
    pub const CMP_GE: i32 = 2;
    /// `a < b`.
    pub const CMP_LT: i32 = 3;
    /// `a <= b`.
    pub const CMP_LE: i32 = 4;
    /// `a != b`.
    pub const CMP_NE: i32 = 5;
}

// ── Data types ────────────────────────────────────────────────────────────────

/// OpenCV data-type depth constants (`CV_8U`, `CV_32F`, …).
pub mod data_type {
    /// 8-bit unsigned integer.
    pub const CV_8U: i32 = 0;
    /// 8-bit signed integer.
    pub const CV_8S: i32 = 1;
    /// 16-bit unsigned integer.
    pub const CV_16U: i32 = 2;
    /// 16-bit signed integer.
    pub const CV_16S: i32 = 3;
    /// 32-bit signed integer.
    pub const CV_32S: i32 = 4;
    /// 32-bit floating point.
    pub const CV_32F: i32 = 5;
    /// 64-bit floating point.
    pub const CV_64F: i32 = 6;
}

// ── Warp flags ────────────────────────────────────────────────────────────────

/// Flags for warp-perspective / warp-affine.
pub mod warp_flags {
    /// Interpret the transformation matrix as its own inverse.
    pub const WARP_INVERSE_MAP: i32 = 16;
}

// ── Optical flow flags ────────────────────────────────────────────────────────

/// Flags for optical-flow algorithms.
pub mod optical_flow_flags {
    /// Use a Gaussian window in Farneback flow (slower but smoother).
    pub const OPTFLOW_FARNEBACK_GAUSSIAN: i32 = 256;
    /// Use the initial flow from the previous call as the starting estimate.
    pub const OPTFLOW_USE_INITIAL_FLOW: i32 = 4;
}

// ── Draw matches flags ────────────────────────────────────────────────────────

/// Flags for `drawMatches` / `drawKeypoints`.
pub mod draw_matches_flags {
    /// Draw only matched keypoints; default for both.
    pub const DRAW_MATCHES_FLAGS_DEFAULT: i32 = 0;
    /// Draw each keypoint as a circle with size proportional to its scale and
    /// a line indicating its orientation.
    pub const DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS: i32 = 4;
    /// Do not draw single (unmatched) keypoints.
    pub const DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS: i32 = 2;
}

// ── Marker types ──────────────────────────────────────────────────────────────

/// Marker shapes for `drawMarker`.
pub mod marker_type {
    /// Crosshair marker.
    pub const MARKER_CROSS: i32 = 0;
    /// Diagonal crosshair (×) marker.
    pub const MARKER_TILTED_CROSS: i32 = 1;
    /// Star marker (*)
    pub const MARKER_STAR: i32 = 2;
    /// Diamond marker.
    pub const MARKER_DIAMOND: i32 = 3;
    /// Square marker.
    pub const MARKER_SQUARE: i32 = 4;
    /// Upward-pointing triangle marker.
    pub const MARKER_TRIANGLE_UP: i32 = 5;
    /// Downward-pointing triangle marker.
    pub const MARKER_TRIANGLE_DOWN: i32 = 6;
}

// ── Distance transform types ──────────────────────────────────────────────────

/// Distance types for `fitLine` and `distanceTransform`.
pub mod dist_type {
    /// L1 norm (city-block distance).
    pub const DIST_L1: i32 = 1;
    /// L2 norm (Euclidean distance).
    pub const DIST_L2: i32 = 2;
    /// L-infinity norm.
    pub const DIST_C: i32 = 3;
    /// L1 approximation (fast).
    pub const DIST_L12: i32 = 4;
    /// Fair M-estimator.
    pub const DIST_FAIR: i32 = 5;
    /// Welsch M-estimator.
    pub const DIST_WELSCH: i32 = 6;
    /// Huber M-estimator.
    pub const DIST_HUBER: i32 = 7;
}

// ── Flat re-exports ───────────────────────────────────────────────────────────

pub use adaptive_thresh::*;
pub use border::*;
pub use cap_prop::*;
pub use chain_approx::*;
pub use color::*;
pub use compare::*;
pub use contour_retr::*;
pub use data_type::*;
pub use dist_type::*;
pub use draw_matches_flags::*;
pub use feature_flags::*;
pub use font::*;
pub use hough::*;
pub use imread::*;
pub use interpolation::*;
pub use line_type::*;
pub use marker_type::*;
pub use morph_op::*;
pub use morph_shape::*;
pub use norm_type::*;
pub use optical_flow_flags::*;
pub use rotate::*;
pub use template_match::*;
pub use threshold::*;
pub use warp_flags::*;
