//! `Mat` — the central image buffer type for the cv2 compatibility layer.
//!
//! BGR channel ordering is the default, matching OpenCV convention.

// OpenCV uses SCREAMING_SNAKE_CASE variant names for type constants (CV_8UC1, etc.).
// This is intentional API mimicry and requires suppressing the lint here.
#![allow(non_camel_case_types)]

use crate::error::{Cv2Error, Cv2Result};

/// Multi-channel image / matrix buffer.
///
/// Data is stored row-major, interleaved.  For `CV_8UC3` the layout is
/// `[B0 G0 R0 | B1 G1 R1 | …]` (OpenCV BGR ordering).
#[derive(Clone, Debug)]
pub struct Mat {
    /// Raw pixel bytes (row-major, interleaved).
    pub data: Vec<u8>,
    /// Number of rows (height in pixels).
    pub rows: usize,
    /// Number of columns (width in pixels).
    pub cols: usize,
    /// Bytes per row (`cols * elem_size`).
    pub step: usize,
    /// Element type and channel count.
    pub mat_type: MatType,
}

/// Element type selector mirroring OpenCV's `CV_` depth+channel constants.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MatType {
    /// 8-bit unsigned, 1 channel (grayscale).
    CV_8UC1,
    /// 8-bit unsigned, 3 channels (e.g. BGR).
    CV_8UC3,
    /// 8-bit unsigned, 4 channels (e.g. BGRA).
    CV_8UC4,
    /// 32-bit float, 1 channel.
    CV_32FC1,
    /// 32-bit float, 2 channels (e.g. optical-flow x/y vectors).
    CV_32FC2,
    /// 32-bit float, 3 channels.
    CV_32FC3,
    /// 64-bit float, 1 channel (used for transformation matrices).
    CV_64FC1,
}

/// A 4-element scalar value (matches `cv::Scalar`).
#[derive(Clone, Copy, Debug, Default)]
pub struct Scalar(pub f64, pub f64, pub f64, pub f64);

/// Integer 2-D point.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Point {
    /// Horizontal coordinate.
    pub x: i32,
    /// Vertical coordinate.
    pub y: i32,
}

/// Floating-point 2-D point.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Point2f {
    /// Horizontal coordinate.
    pub x: f32,
    /// Vertical coordinate.
    pub y: f32,
}

/// Dimensions of a 2-D region.
#[derive(Clone, Copy, Debug, Default)]
pub struct Size {
    /// Width.
    pub width: usize,
    /// Height.
    pub height: usize,
}

/// Axis-aligned 2-D rectangle.
#[derive(Clone, Copy, Debug, Default)]
pub struct Rect {
    /// Left edge (inclusive).
    pub x: i32,
    /// Top edge (inclusive).
    pub y: i32,
    /// Width in pixels.
    pub width: i32,
    /// Height in pixels.
    pub height: i32,
}

// ── MatType helpers ───────────────────────────────────────────────────────────

impl MatType {
    /// Number of channels per pixel.
    #[must_use]
    pub const fn channels(self) -> usize {
        match self {
            Self::CV_8UC1 | Self::CV_32FC1 | Self::CV_64FC1 => 1,
            Self::CV_32FC2 => 2,
            Self::CV_8UC3 | Self::CV_32FC3 => 3,
            Self::CV_8UC4 => 4,
        }
    }

    /// Bytes per individual component (depth).
    #[must_use]
    pub const fn depth_bytes(self) -> usize {
        match self {
            Self::CV_8UC1 | Self::CV_8UC3 | Self::CV_8UC4 => 1,
            Self::CV_32FC1 | Self::CV_32FC2 | Self::CV_32FC3 => 4,
            Self::CV_64FC1 => 8,
        }
    }

    /// Bytes per pixel element (`channels * depth_bytes`).
    #[must_use]
    pub const fn elem_size(self) -> usize {
        self.channels() * self.depth_bytes()
    }
}

// ── Mat implementation ────────────────────────────────────────────────────────

impl Mat {
    /// Allocate a zero-initialized `Mat` with the given shape and type.
    #[must_use]
    pub fn new(rows: usize, cols: usize, mat_type: MatType) -> Self {
        let step = cols * mat_type.elem_size();
        Mat {
            data: vec![0u8; rows * step],
            rows,
            cols,
            step,
            mat_type,
        }
    }

    /// Allocate an 8-bit single-channel (grayscale) `Mat`.
    #[must_use]
    pub fn new_8uc1(rows: usize, cols: usize) -> Self {
        Self::new(rows, cols, MatType::CV_8UC1)
    }

    /// Allocate an 8-bit BGR 3-channel `Mat`.
    #[must_use]
    pub fn new_8uc3(rows: usize, cols: usize) -> Self {
        Self::new(rows, cols, MatType::CV_8UC3)
    }

    /// Allocate an 8-bit BGRA 4-channel `Mat`.
    #[must_use]
    pub fn new_8uc4(rows: usize, cols: usize) -> Self {
        Self::new(rows, cols, MatType::CV_8UC4)
    }

    /// Number of channels determined by `mat_type`.
    #[must_use]
    pub fn channels(&self) -> usize {
        self.mat_type.channels()
    }

    /// Bytes per component determined by `mat_type`.
    #[must_use]
    pub fn depth_bytes(&self) -> usize {
        self.mat_type.depth_bytes()
    }

    /// Total number of pixels (`rows * cols`).
    #[must_use]
    pub fn pixel_count(&self) -> usize {
        self.rows * self.cols
    }

    /// Returns `true` if either dimension is zero.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows == 0 || self.cols == 0
    }

    /// Read the BGR triple at `(row, col)`.
    ///
    /// Panics in debug builds if `mat_type != CV_8UC3`.
    #[must_use]
    pub fn at_8u3(&self, row: usize, col: usize) -> [u8; 3] {
        debug_assert_eq!(self.mat_type, MatType::CV_8UC3);
        let off = row * self.step + col * 3;
        [self.data[off], self.data[off + 1], self.data[off + 2]]
    }

    /// Mutable reference to the BGR triple at `(row, col)`.
    ///
    /// Panics in debug builds if `mat_type != CV_8UC3`.
    pub fn at_8u3_mut(&mut self, row: usize, col: usize) -> &mut [u8] {
        debug_assert_eq!(self.mat_type, MatType::CV_8UC3);
        let off = row * self.step + col * 3;
        &mut self.data[off..off + 3]
    }

    /// Read the grayscale byte at `(row, col)`.
    ///
    /// Panics in debug builds if `mat_type != CV_8UC1`.
    #[must_use]
    pub fn at_8u1(&self, row: usize, col: usize) -> u8 {
        debug_assert_eq!(self.mat_type, MatType::CV_8UC1);
        self.data[row * self.step + col]
    }

    /// Construct a BGR `Mat` from raw byte data.
    ///
    /// `data` must have exactly `rows * cols * 3` bytes.
    #[must_use]
    pub fn from_bgr_bytes(data: Vec<u8>, rows: usize, cols: usize) -> Self {
        let step = cols * 3;
        Mat {
            data,
            rows,
            cols,
            step,
            mat_type: MatType::CV_8UC3,
        }
    }

    /// Construct a grayscale `Mat` from raw byte data.
    ///
    /// `data` must have exactly `rows * cols` bytes.
    #[must_use]
    pub fn from_gray_bytes(data: Vec<u8>, rows: usize, cols: usize) -> Self {
        Mat {
            data,
            rows,
            cols,
            step: cols,
            mat_type: MatType::CV_8UC1,
        }
    }

    /// Convert BGR data to RGB byte order (swaps B and R channels).
    ///
    /// Only valid for `CV_8UC3`; returns `Cv2Error::UnsupportedDtype` otherwise.
    pub fn to_rgb_bytes(&self) -> Cv2Result<Vec<u8>> {
        match self.mat_type {
            MatType::CV_8UC3 => {
                let mut rgb = self.data.clone();
                for chunk in rgb.chunks_exact_mut(3) {
                    chunk.swap(0, 2);
                }
                Ok(rgb)
            }
            _ => Err(Cv2Error::UnsupportedDtype {
                mat_type: self.mat_type,
            }),
        }
    }

    /// Deep-copy this `Mat` into a new `Mat` with independent storage.
    ///
    /// Named `clone_mat` to provide an explicit, unambiguous API alongside the
    /// derived `Clone` impl.
    #[must_use]
    pub fn clone_mat(&self) -> Mat {
        Mat {
            data: self.data.clone(),
            rows: self.rows,
            cols: self.cols,
            step: self.step,
            mat_type: self.mat_type,
        }
    }

    /// Scale-and-shift conversion: `dst[i] = saturate_cast<dst_type>(src[i] * alpha + beta)`.
    ///
    /// Supported conversions:
    /// - `CV_8UC1` → `CV_32FC1`: stores `f32` as native-endian bytes
    /// - `CV_32FC1` → `CV_8UC1`: clamps to `[0, 255]`
    /// - `CV_8UC1` → `CV_8UC1`: saturating scale
    /// - `CV_8UC3` → `CV_8UC3`: per-channel saturating scale
    pub fn convert_to(&self, dst_type: MatType, alpha: f64, beta: f64) -> Cv2Result<Mat> {
        let n = self.rows * self.cols;
        let ch = self.channels();
        match (self.mat_type, dst_type) {
            (MatType::CV_8UC1, MatType::CV_32FC1) => {
                let mut data = vec![0u8; n * 4];
                for i in 0..n {
                    let v = (self.data[i] as f64 * alpha + beta) as f32;
                    data[i * 4..(i + 1) * 4].copy_from_slice(&v.to_ne_bytes());
                }
                Ok(Mat {
                    data,
                    rows: self.rows,
                    cols: self.cols,
                    step: self.cols * 4,
                    mat_type: MatType::CV_32FC1,
                })
            }
            (MatType::CV_32FC1, MatType::CV_8UC1) => {
                let mut data = vec![0u8; n];
                for i in 0..n {
                    let bytes = [
                        self.data[i * 4],
                        self.data[i * 4 + 1],
                        self.data[i * 4 + 2],
                        self.data[i * 4 + 3],
                    ];
                    let v = f32::from_ne_bytes(bytes) as f64;
                    data[i] = (v * alpha + beta).clamp(0.0, 255.0) as u8;
                }
                Ok(Mat {
                    data,
                    rows: self.rows,
                    cols: self.cols,
                    step: self.cols,
                    mat_type: MatType::CV_8UC1,
                })
            }
            (MatType::CV_8UC1, MatType::CV_8UC1) => {
                let data: Vec<u8> = self
                    .data
                    .iter()
                    .map(|&v| (v as f64 * alpha + beta).clamp(0.0, 255.0) as u8)
                    .collect();
                Ok(Mat {
                    data,
                    rows: self.rows,
                    cols: self.cols,
                    step: self.cols,
                    mat_type: MatType::CV_8UC1,
                })
            }
            (MatType::CV_8UC3, MatType::CV_8UC3) => {
                let data: Vec<u8> = self
                    .data
                    .iter()
                    .map(|&v| (v as f64 * alpha + beta).clamp(0.0, 255.0) as u8)
                    .collect();
                Ok(Mat {
                    data,
                    rows: self.rows,
                    cols: self.cols,
                    step: self.cols * ch,
                    mat_type: MatType::CV_8UC3,
                })
            }
            _ => Err(Cv2Error::UnsupportedDtype { mat_type: dst_type }),
        }
    }

    /// Return a deep copy of the sub-image `[x, x+width) × [y, y+height)`.
    ///
    /// Note: unlike `cv::Mat::operator()`, this returns an **owned copy**,
    /// not a view into the parent buffer.
    ///
    /// Returns `Cv2Error::SizeMismatch` if the ROI is empty after clamping to
    /// image bounds.
    pub fn submat(&self, x: i32, y: i32, width: i32, height: i32) -> Cv2Result<Mat> {
        let x0 = x.max(0) as usize;
        let y0 = y.max(0) as usize;
        let x1 = (x + width).min(self.cols as i32).max(0) as usize;
        let y1 = (y + height).min(self.rows as i32).max(0) as usize;

        let w = x1.saturating_sub(x0);
        let h = y1.saturating_sub(y0);

        if w == 0 || h == 0 {
            return Err(Cv2Error::SizeMismatch {
                expected: (1, 1),
                actual: (h, w),
            });
        }

        let elem = self.mat_type.elem_size();
        let mut data = Vec::with_capacity(h * w * elem);
        for row in y0..y1 {
            let row_start = row * self.step + x0 * elem;
            let row_end = row_start + w * elem;
            data.extend_from_slice(&self.data[row_start..row_end]);
        }

        Ok(Mat {
            step: w * elem,
            data,
            rows: h,
            cols: w,
            mat_type: self.mat_type,
        })
    }

    /// Reinterpret buffer dimensions without copying data.
    ///
    /// Valid only when `new_rows * new_cols == self.rows * self.cols`.
    /// Returns `Cv2Error::SizeMismatch` if the total element count would change.
    pub fn reshape(&self, new_rows: i32, new_cols: i32) -> Cv2Result<Mat> {
        let nr = new_rows as usize;
        let nc = new_cols as usize;
        if nr * nc != self.rows * self.cols {
            return Err(Cv2Error::SizeMismatch {
                expected: (self.rows, self.cols),
                actual: (nr, nc),
            });
        }
        Ok(Mat {
            data: self.data.clone(),
            rows: nr,
            cols: nc,
            step: nc * self.mat_type.elem_size(),
            mat_type: self.mat_type,
        })
    }
}
