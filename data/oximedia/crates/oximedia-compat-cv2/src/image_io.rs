//! Image I/O functions: `imread`, `imwrite`, `imdecode`, `imencode`.
//!
//! These functions translate between the OxiMedia image pipeline and the `Mat`
//! type, handling the BGR↔RGB channel-order swap that OpenCV uses.

use crate::constants::imread::{IMREAD_COLOR, IMREAD_GRAYSCALE, IMREAD_UNCHANGED};
use crate::error::{Cv2Error, Cv2Result};
use crate::mat::{Mat, MatType};
use oximedia_image::{
    format_detect::{FormatDetector, ImageFormat},
    jpeg::{write_jpeg, JpegDecoder, JpegEncoder, JpegQuality},
    png::{PngColorType, PngDecoder, PngEncoder, PngImage},
    ColorSpace, ImageData, ImageFrame, PixelType,
};
use std::path::Path;

// ── Rec.601 luma conversion ───────────────────────────────────────────────────

/// Convert interleaved RGB bytes to grayscale using Rec.601 luma coefficients.
fn rgb_to_gray(rgb: &[u8]) -> Vec<u8> {
    rgb.chunks_exact(3)
        .map(|px| {
            let r = f32::from(px[0]);
            let g = f32::from(px[1]);
            let b = f32::from(px[2]);
            (0.299 * r + 0.587 * g + 0.114 * b) as u8
        })
        .collect()
}

// ── PNG helpers ───────────────────────────────────────────────────────────────

/// Decode raw bytes as PNG, returning a `PngImage`.
fn decode_png_bytes(data: &[u8]) -> Cv2Result<PngImage> {
    PngDecoder::new()
        .decode(data)
        .map_err(|e| Cv2Error::Codec(e.to_string()))
}

/// Build a `Mat` from a decoded `PngImage` and the requested `flags`.
fn png_image_to_mat(img: &PngImage, flags: i32) -> Cv2Result<Mat> {
    let w = img.width as usize;
    let h = img.height as usize;

    if flags == IMREAD_UNCHANGED {
        // Return the data as-is (map PNG color type → MatType).
        let mat_type = match img.color_type {
            PngColorType::Grayscale => MatType::CV_8UC1,
            PngColorType::Rgb => MatType::CV_8UC3,
            PngColorType::Rgba => MatType::CV_8UC4,
            PngColorType::GrayscaleAlpha => MatType::CV_8UC4, // promote to 4ch
            PngColorType::Indexed => MatType::CV_8UC1,
        };
        let data = match img.color_type {
            PngColorType::GrayscaleAlpha => {
                // Expand 2-channel to 4-channel (BGRA order, gray duplicated)
                img.pixels
                    .chunks_exact(2)
                    .flat_map(|px| [px[0], px[0], px[0], px[1]])
                    .collect()
            }
            PngColorType::Rgb => {
                // RGB → BGR
                img.pixels
                    .chunks_exact(3)
                    .flat_map(|px| [px[2], px[1], px[0]])
                    .collect()
            }
            PngColorType::Rgba => {
                // RGBA → BGRA
                img.pixels
                    .chunks_exact(4)
                    .flat_map(|px| [px[2], px[1], px[0], px[3]])
                    .collect()
            }
            _ => img.pixels.clone(),
        };
        let step = w * mat_type.elem_size();
        return Ok(Mat {
            data,
            rows: h,
            cols: w,
            step,
            mat_type,
        });
    }

    if flags == IMREAD_GRAYSCALE {
        let gray = match img.color_type {
            PngColorType::Grayscale | PngColorType::Indexed => img.pixels.clone(),
            PngColorType::GrayscaleAlpha => img.pixels.chunks_exact(2).map(|px| px[0]).collect(),
            PngColorType::Rgb => rgb_to_gray(&img.pixels),
            PngColorType::Rgba => {
                // drop alpha then convert
                let rgb: Vec<u8> = img
                    .pixels
                    .chunks_exact(4)
                    .flat_map(|px| [px[0], px[1], px[2]])
                    .collect();
                rgb_to_gray(&rgb)
            }
        };
        return Ok(Mat::from_gray_bytes(gray, h, w));
    }

    // IMREAD_COLOR (default): always produce CV_8UC3 BGR
    let bgr = match img.color_type {
        PngColorType::Grayscale | PngColorType::Indexed => {
            // Expand gray to BGR
            img.pixels.iter().flat_map(|&v| [v, v, v]).collect()
        }
        PngColorType::GrayscaleAlpha => img
            .pixels
            .chunks_exact(2)
            .flat_map(|px| [px[0], px[0], px[0]])
            .collect(),
        PngColorType::Rgb => {
            // RGB → BGR
            img.pixels
                .chunks_exact(3)
                .flat_map(|px| [px[2], px[1], px[0]])
                .collect()
        }
        PngColorType::Rgba => {
            // RGBA → BGR (drop alpha)
            img.pixels
                .chunks_exact(4)
                .flat_map(|px| [px[2], px[1], px[0]])
                .collect()
        }
    };
    Ok(Mat::from_bgr_bytes(bgr, h, w))
}

/// Convert a BGR / BGRA `Mat` to a `PngImage` (RGB / RGBA pixels).
fn mat_to_png_image(mat: &Mat) -> Cv2Result<PngImage> {
    let (pixels, color_type) = match mat.mat_type {
        MatType::CV_8UC3 => {
            let rgb: Vec<u8> = mat
                .data
                .chunks_exact(3)
                .flat_map(|px| [px[2], px[1], px[0]])
                .collect();
            (rgb, PngColorType::Rgb)
        }
        MatType::CV_8UC4 => {
            let rgba: Vec<u8> = mat
                .data
                .chunks_exact(4)
                .flat_map(|px| [px[2], px[1], px[0], px[3]])
                .collect();
            (rgba, PngColorType::Rgba)
        }
        MatType::CV_8UC1 => (mat.data.clone(), PngColorType::Grayscale),
        _ => {
            return Err(Cv2Error::UnsupportedDtype {
                mat_type: mat.mat_type,
            })
        }
    };
    Ok(PngImage {
        width: mat.cols as u32,
        height: mat.rows as u32,
        bit_depth: 8,
        color_type,
        pixels,
        metadata: std::collections::HashMap::new(),
    })
}

// ── JPEG helpers ──────────────────────────────────────────────────────────────

/// Decode raw bytes as JPEG, returning an `ImageFrame`.
fn decode_jpeg_bytes_to_frame(data: &[u8]) -> Cv2Result<ImageFrame> {
    JpegDecoder::new()
        .decode(data)
        .map(|jf| jf.to_image_frame(1))
        .map_err(|e| Cv2Error::Codec(e.to_string()))
}

/// Convert an `ImageFrame` (from JPEG decode) to a `Mat` using `flags`.
///
/// JPEG decode produces interleaved RGB or grayscale pixels.
fn jpeg_frame_to_mat(frame: &ImageFrame, flags: i32) -> Cv2Result<Mat> {
    let w = frame.width as usize;
    let h = frame.height as usize;
    let raw = frame
        .data
        .as_slice()
        .ok_or_else(|| Cv2Error::Codec("JPEG frame data is not interleaved".to_string()))?;

    if flags == IMREAD_UNCHANGED || flags == IMREAD_COLOR {
        if frame.components == 1 {
            if flags == IMREAD_COLOR {
                // Gray → BGR
                let bgr: Vec<u8> = raw.iter().flat_map(|&v| [v, v, v]).collect();
                Ok(Mat::from_bgr_bytes(bgr, h, w))
            } else {
                Ok(Mat::from_gray_bytes(raw.to_vec(), h, w))
            }
        } else {
            // RGB → BGR
            let bgr: Vec<u8> = raw
                .chunks_exact(3)
                .flat_map(|px| [px[2], px[1], px[0]])
                .collect();
            Ok(Mat::from_bgr_bytes(bgr, h, w))
        }
    } else {
        // IMREAD_GRAYSCALE
        let gray = if frame.components == 1 {
            raw.to_vec()
        } else {
            rgb_to_gray(raw)
        };
        Ok(Mat::from_gray_bytes(gray, h, w))
    }
}

/// Convert a `Mat` to an `ImageFrame` suitable for JPEG encoding.
fn mat_to_jpeg_frame(mat: &Mat) -> Cv2Result<ImageFrame> {
    match mat.mat_type {
        MatType::CV_8UC3 => {
            // BGR → RGB for JPEG encoder
            let rgb: Vec<u8> = mat
                .data
                .chunks_exact(3)
                .flat_map(|px| [px[2], px[1], px[0]])
                .collect();
            Ok(ImageFrame::new(
                1,
                mat.cols as u32,
                mat.rows as u32,
                PixelType::U8,
                3,
                ColorSpace::Srgb,
                ImageData::interleaved(rgb),
            ))
        }
        MatType::CV_8UC1 => Ok(ImageFrame::new(
            1,
            mat.cols as u32,
            mat.rows as u32,
            PixelType::U8,
            1,
            ColorSpace::Luma,
            ImageData::interleaved(mat.data.clone()),
        )),
        _ => Err(Cv2Error::UnsupportedDtype {
            mat_type: mat.mat_type,
        }),
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Read an image file into a `Mat`.
///
/// The `flags` parameter controls channel count and bit depth:
/// - [`IMREAD_COLOR`] (1): always return a 3-channel BGR `Mat` (default).
/// - [`IMREAD_GRAYSCALE`] (0): return a 1-channel grayscale `Mat`.
/// - [`IMREAD_UNCHANGED`] (-1): return with original channels (e.g. BGRA for RGBA source).
///
/// Supports PNG and JPEG files. BMP is not yet implemented.
pub fn imread(path: impl AsRef<Path>, flags: i32) -> Cv2Result<Mat> {
    let path = path.as_ref();
    let data = std::fs::read(path).map_err(Cv2Error::Io)?;
    imdecode(&data, flags)
}

/// Decode an image from in-memory bytes into a `Mat`.
///
/// Format is auto-detected by magic bytes. See [`imread`] for `flags` semantics.
pub fn imdecode(data: &[u8], flags: i32) -> Cv2Result<Mat> {
    let fmt = FormatDetector::detect(data);
    match fmt {
        ImageFormat::Png => {
            let img = decode_png_bytes(data)?;
            png_image_to_mat(&img, flags)
        }
        ImageFormat::Jpeg => {
            let frame = decode_jpeg_bytes_to_frame(data)?;
            jpeg_frame_to_mat(&frame, flags)
        }
        ImageFormat::Unknown => {
            // Try PNG first (handles files with bad magic but valid content)
            if let Ok(img) = decode_png_bytes(data) {
                return png_image_to_mat(&img, flags);
            }
            Err(Cv2Error::Codec("Unrecognised image format".to_string()))
        }
        other => Err(Cv2Error::FeatureNotImplemented {
            name: other.name(),
            refinement: "Slice A supports PNG and JPEG only",
        }),
    }
}

/// Write a `Mat` to a file. Format is inferred from the file extension.
///
/// Supports `.png`, `.jpg` / `.jpeg`.
pub fn imwrite(path: impl AsRef<Path>, mat: &Mat) -> Cv2Result<()> {
    let path = path.as_ref();
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();

    match ext.as_str() {
        "png" => {
            let img = mat_to_png_image(mat)?;
            let encoded = PngEncoder::default()
                .encode(&img)
                .map_err(|e| Cv2Error::Codec(e.to_string()))?;
            std::fs::write(path, encoded).map_err(Cv2Error::Io)
        }
        "jpg" | "jpeg" => {
            let frame = mat_to_jpeg_frame(mat)?;
            write_jpeg(path, &frame, 95).map_err(|e| Cv2Error::Codec(e.to_string()))
        }
        other => Err(Cv2Error::UnknownExtension {
            ext: other.to_string(),
        }),
    }
}

/// Encode a `Mat` to bytes in the given format.
///
/// `ext` is the format extension without the leading dot (e.g. `"png"`, `"jpg"`).
pub fn imencode(ext: &str, mat: &Mat) -> Cv2Result<Vec<u8>> {
    match ext.to_ascii_lowercase().as_str() {
        "png" => {
            let img = mat_to_png_image(mat)?;
            PngEncoder::default()
                .encode(&img)
                .map_err(|e| Cv2Error::Codec(e.to_string()))
        }
        "jpg" | "jpeg" => {
            let frame = mat_to_jpeg_frame(mat)?;
            JpegEncoder::new(JpegQuality::new(95))
                .encode(&frame)
                .map_err(|e| Cv2Error::Codec(e.to_string()))
        }
        other => Err(Cv2Error::UnknownExtension {
            ext: other.to_string(),
        }),
    }
}
