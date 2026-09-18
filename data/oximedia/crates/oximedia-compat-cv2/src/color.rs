//! Color space conversion: `cvt_color`.
//!
//! Provides OpenCV-compatible color space conversion operating on [`Mat`] buffers.
//! Channel order follows OpenCV convention: BGR for 3-channel images.

use crate::constants::color::{
    COLOR_BGR2Lab, COLOR_Lab2BGR, COLOR_Lab2RGB, COLOR_RGB2Lab, COLOR_BGR2GRAY, COLOR_BGR2HLS,
    COLOR_BGR2HSV, COLOR_BGR2RGB, COLOR_BGR2YUV, COLOR_BGRA2GRAY, COLOR_GRAY2BGR, COLOR_HLS2BGR,
    COLOR_HLS2RGB, COLOR_HSV2BGR, COLOR_HSV2RGB, COLOR_RGB2GRAY, COLOR_RGB2HSV, COLOR_RGB2YUV,
    COLOR_YUV2BGR, COLOR_YUV2RGB,
};
use crate::error::{Cv2Error, Cv2Result};
use crate::mat::Mat;
#[cfg(test)]
use crate::mat::MatType;

// ── Public API ────────────────────────────────────────────────────────────────

/// Convert a `Mat` from one color space to another.
///
/// Mirrors `cv2.cvtColor(src, code)`.
///
/// # Supported codes
///
/// | code | conversion |
/// |------|-----------|
/// | `COLOR_BGR2RGB` / `COLOR_RGB2BGR` (4) | swap B↔R |
/// | `COLOR_BGR2GRAY` (6) | BT.601 luminance |
/// | `COLOR_RGB2GRAY` (7) | BT.601 luminance (R/G/B order) |
/// | `COLOR_GRAY2BGR` / `COLOR_GRAY2RGB` (8) | replicate gray → 3 ch |
/// | `COLOR_BGRA2GRAY` (10) | BGRA → gray |
/// | `COLOR_BGR2HSV` (40) | H∈[0,180), S/V∈[0,256) |
/// | `COLOR_RGB2HSV` (41) | same but RGB input |
/// | `COLOR_HSV2BGR` (54) | HSV → BGR |
/// | `COLOR_HSV2RGB` (55) | HSV → RGB |
/// | `COLOR_BGR2Lab` (44) | CIE L\*a\*b\* D65 |
/// | `COLOR_RGB2Lab` (45) | same but RGB input |
/// | `COLOR_Lab2BGR` (56) | L\*a\*b\* → BGR |
/// | `COLOR_Lab2RGB` (57) | L\*a\*b\* → RGB |
/// | `COLOR_BGR2HLS` (52) | HLS |
/// | `COLOR_HLS2BGR` (60) | HLS → BGR |
/// | `COLOR_HLS2RGB` (61) | HLS → RGB |
/// | `COLOR_BGR2YUV` (82) | BT.601 YUV |
/// | `COLOR_RGB2YUV` (83) | same but RGB input |
/// | `COLOR_YUV2BGR` (84) | YUV → BGR |
/// | `COLOR_YUV2RGB` (85) | YUV → RGB |
///
/// Unknown codes return `Cv2Error::UnsupportedFlag`.
pub fn cvt_color(src: &Mat, code: i32) -> Cv2Result<Mat> {
    let data = src.data.as_slice();
    let w = src.cols;
    let h = src.rows;
    let ch = src.channels();

    match code {
        c if c == COLOR_BGR2GRAY => {
            let gray = convert_to_gray(data, w, h, ch, false)?;
            Ok(Mat::from_gray_bytes(gray, h, w))
        }
        c if c == COLOR_RGB2GRAY => {
            let gray = convert_to_gray(data, w, h, ch, true)?;
            Ok(Mat::from_gray_bytes(gray, h, w))
        }
        c if c == COLOR_GRAY2BGR => {
            let bgr = gray_to_bgr(data, w, h);
            Ok(Mat::from_bgr_bytes(bgr, h, w))
        }
        c if c == COLOR_BGR2RGB || c == 4 => {
            // COLOR_BGR2RGB and COLOR_RGB2BGR share value 4
            let swapped = swap_rb(data, w, h);
            Ok(Mat::from_bgr_bytes(swapped, h, w))
        }
        c if c == COLOR_BGRA2GRAY => {
            let gray = bgra_to_gray(data, w, h)?;
            Ok(Mat::from_gray_bytes(gray, h, w))
        }
        c if c == COLOR_BGR2HSV => {
            let hsv = bgr_to_hsv(data, w, h, false);
            Ok(Mat::from_bgr_bytes(hsv, h, w))
        }
        c if c == COLOR_RGB2HSV => {
            let hsv = bgr_to_hsv(data, w, h, true);
            Ok(Mat::from_bgr_bytes(hsv, h, w))
        }
        c if c == COLOR_HSV2BGR => {
            let bgr = hsv_to_bgr(data, w, h, false);
            Ok(Mat::from_bgr_bytes(bgr, h, w))
        }
        c if c == COLOR_HSV2RGB => {
            let bgr = hsv_to_bgr(data, w, h, true);
            Ok(Mat::from_bgr_bytes(bgr, h, w))
        }
        c if c == COLOR_BGR2Lab => {
            let lab = bgr_to_lab(data, w, h, false);
            Ok(Mat::from_bgr_bytes(lab, h, w))
        }
        c if c == COLOR_RGB2Lab => {
            let lab = bgr_to_lab(data, w, h, true);
            Ok(Mat::from_bgr_bytes(lab, h, w))
        }
        c if c == COLOR_Lab2BGR => {
            let bgr = lab_to_bgr(data, w, h, false);
            Ok(Mat::from_bgr_bytes(bgr, h, w))
        }
        c if c == COLOR_Lab2RGB => {
            let bgr = lab_to_bgr(data, w, h, true);
            Ok(Mat::from_bgr_bytes(bgr, h, w))
        }
        c if c == COLOR_BGR2HLS => {
            let hls = bgr_to_hls(data, w, h);
            Ok(Mat::from_bgr_bytes(hls, h, w))
        }
        c if c == COLOR_HLS2BGR => {
            let bgr = hls_to_bgr(data, w, h, false);
            Ok(Mat::from_bgr_bytes(bgr, h, w))
        }
        c if c == COLOR_HLS2RGB => {
            let bgr = hls_to_bgr(data, w, h, true);
            Ok(Mat::from_bgr_bytes(bgr, h, w))
        }
        c if c == COLOR_BGR2YUV => {
            let yuv = bgr_to_yuv(data, w, h, false);
            Ok(Mat::from_bgr_bytes(yuv, h, w))
        }
        c if c == COLOR_RGB2YUV => {
            let yuv = bgr_to_yuv(data, w, h, true);
            Ok(Mat::from_bgr_bytes(yuv, h, w))
        }
        c if c == COLOR_YUV2BGR => {
            let bgr = yuv_to_bgr(data, w, h, false);
            Ok(Mat::from_bgr_bytes(bgr, h, w))
        }
        c if c == COLOR_YUV2RGB => {
            let bgr = yuv_to_bgr(data, w, h, true);
            Ok(Mat::from_bgr_bytes(bgr, h, w))
        }
        _ => Err(Cv2Error::UnsupportedFlag {
            name: "cvt_color.code",
            value: code,
        }),
    }
}

// ── Internal conversion helpers ───────────────────────────────────────────────

/// BT.601 luma. `is_rgb = true` means input channel order is R,G,B (index 0=R).
pub(crate) fn convert_to_gray(
    data: &[u8],
    w: usize,
    h: usize,
    ch: usize,
    is_rgb: bool,
) -> Cv2Result<Vec<u8>> {
    let mut out = vec![0u8; h * w];
    if ch == 1 {
        out.copy_from_slice(data);
        return Ok(out);
    }
    if ch < 3 {
        return Err(Cv2Error::Codec(
            "cvtColor: need at least 3 channels for gray conversion".into(),
        ));
    }
    for y in 0..h {
        for x in 0..w {
            let off = (y * w + x) * ch;
            let (r, g, b) = if is_rgb {
                (data[off], data[off + 1], data[off + 2])
            } else {
                (data[off + 2], data[off + 1], data[off])
            };
            let gray = (0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32) as u8;
            out[y * w + x] = gray;
        }
    }
    Ok(out)
}

fn bgra_to_gray(data: &[u8], w: usize, h: usize) -> Cv2Result<Vec<u8>> {
    let ch = if data.len() == h * w * 4 { 4 } else { 3 };
    convert_to_gray(data, w, h, ch, false)
}

pub(crate) fn gray_to_bgr(data: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; h * w * 3];
    for i in 0..h * w {
        let g = data[i];
        out[i * 3] = g;
        out[i * 3 + 1] = g;
        out[i * 3 + 2] = g;
    }
    out
}

pub(crate) fn swap_rb(data: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = data.to_vec();
    for i in 0..h * w {
        out.swap(i * 3, i * 3 + 2);
    }
    out
}

// ── HSV ───────────────────────────────────────────────────────────────────────

pub(crate) fn bgr_to_hsv(data: &[u8], w: usize, h: usize, is_rgb: bool) -> Vec<u8> {
    let mut out = vec![0u8; h * w * 3];
    for i in 0..h * w {
        let off = i * 3;
        let (r, g, b) = if is_rgb {
            (
                data[off] as f32 / 255.0,
                data[off + 1] as f32 / 255.0,
                data[off + 2] as f32 / 255.0,
            )
        } else {
            (
                data[off + 2] as f32 / 255.0,
                data[off + 1] as f32 / 255.0,
                data[off] as f32 / 255.0,
            )
        };
        let (h_val, s, v) = rgb_to_hsv_f(r, g, b);
        out[off] = (h_val * 180.0) as u8; // OpenCV uses 0-180 for H
        out[off + 1] = (s * 255.0) as u8;
        out[off + 2] = (v * 255.0) as u8;
    }
    out
}

pub(crate) fn hsv_to_bgr(data: &[u8], w: usize, h: usize, is_rgb: bool) -> Vec<u8> {
    let mut out = vec![0u8; h * w * 3];
    for i in 0..h * w {
        let off = i * 3;
        let h_val = data[off] as f32 / 180.0; // OpenCV uses 0-180
        let s = data[off + 1] as f32 / 255.0;
        let v = data[off + 2] as f32 / 255.0;
        let (r, g, b) = hsv_to_rgb_f(h_val, s, v);
        if is_rgb {
            out[off] = (r * 255.0) as u8;
            out[off + 1] = (g * 255.0) as u8;
            out[off + 2] = (b * 255.0) as u8;
        } else {
            out[off] = (b * 255.0) as u8;
            out[off + 1] = (g * 255.0) as u8;
            out[off + 2] = (r * 255.0) as u8;
        }
    }
    out
}

fn rgb_to_hsv_f(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let v = max;
    let s = if max > 1e-6 { delta / max } else { 0.0 };
    let h = if delta < 1e-6 {
        0.0
    } else if (max - r).abs() < 1e-6 {
        ((g - b) / delta).rem_euclid(6.0) / 6.0
    } else if (max - g).abs() < 1e-6 {
        ((b - r) / delta + 2.0) / 6.0
    } else {
        ((r - g) / delta + 4.0) / 6.0
    };
    (h, s, v)
}

fn hsv_to_rgb_f(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    if s < 1e-6 {
        return (v, v, v);
    }
    let h6 = h * 6.0;
    let i = h6 as i32;
    let f = h6 - i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    match i % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}

// ── L*a*b* ────────────────────────────────────────────────────────────────────

pub(crate) fn bgr_to_lab(data: &[u8], w: usize, h: usize, is_rgb: bool) -> Vec<u8> {
    let mut out = vec![0u8; h * w * 3];
    for i in 0..h * w {
        let off = i * 3;
        let (r, g, b) = if is_rgb {
            (
                data[off] as f32 / 255.0,
                data[off + 1] as f32 / 255.0,
                data[off + 2] as f32 / 255.0,
            )
        } else {
            (
                data[off + 2] as f32 / 255.0,
                data[off + 1] as f32 / 255.0,
                data[off] as f32 / 255.0,
            )
        };
        let (l, a, b_val) = rgb_to_lab_f(r, g, b);
        // OpenCV scales: L*[0,100]→uint8, a*/b* [-128,127] → +128
        out[off] = (l * 255.0 / 100.0).clamp(0.0, 255.0) as u8;
        out[off + 1] = ((a + 128.0).clamp(0.0, 255.0)) as u8;
        out[off + 2] = ((b_val + 128.0).clamp(0.0, 255.0)) as u8;
    }
    out
}

pub(crate) fn lab_to_bgr(data: &[u8], w: usize, h: usize, is_rgb: bool) -> Vec<u8> {
    let mut out = vec![0u8; h * w * 3];
    for i in 0..h * w {
        let off = i * 3;
        let l = data[off] as f32 * 100.0 / 255.0;
        let a = data[off + 1] as f32 - 128.0;
        let b_val = data[off + 2] as f32 - 128.0;
        let (r, g, b) = lab_to_rgb_f(l, a, b_val);
        if is_rgb {
            out[off] = (r * 255.0).clamp(0.0, 255.0) as u8;
            out[off + 1] = (g * 255.0).clamp(0.0, 255.0) as u8;
            out[off + 2] = (b * 255.0).clamp(0.0, 255.0) as u8;
        } else {
            out[off] = (b * 255.0).clamp(0.0, 255.0) as u8;
            out[off + 1] = (g * 255.0).clamp(0.0, 255.0) as u8;
            out[off + 2] = (r * 255.0).clamp(0.0, 255.0) as u8;
        }
    }
    out
}

fn rgb_to_lab_f(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    // sRGB linearization (gamma removal)
    let lin = |c: f32| -> f32 {
        if c > 0.04045 {
            ((c + 0.055) / 1.055).powf(2.4)
        } else {
            c / 12.92
        }
    };
    let rl = lin(r);
    let gl = lin(g);
    let bl = lin(b);

    // Linear RGB → CIE XYZ (D65)
    let x = rl * 0.4124564 + gl * 0.3575761 + bl * 0.1804375;
    let y = rl * 0.2126729 + gl * 0.7151522 + bl * 0.0721750;
    let z = rl * 0.0193339 + gl * 0.1191920 + bl * 0.9503041;

    // XYZ → L*a*b*
    let f = |t: f32| -> f32 {
        if t > 0.008856 {
            t.powf(1.0 / 3.0)
        } else {
            7.787 * t + 16.0 / 116.0
        }
    };
    let fx = f(x / 0.95047);
    let fy = f(y); // yn = 1.0
    let fz = f(z / 1.08883);
    let l = 116.0 * fy - 16.0;
    let a = 500.0 * (fx - fy);
    let b_val = 200.0 * (fy - fz);
    (l, a, b_val)
}

fn lab_to_rgb_f(l: f32, a: f32, b_val: f32) -> (f32, f32, f32) {
    let fy = (l + 16.0) / 116.0;
    let fx = a / 500.0 + fy;
    let fz = fy - b_val / 200.0;
    let f_inv = |t: f32| -> f32 {
        if t > 0.2069 {
            t.powi(3)
        } else {
            (t - 16.0 / 116.0) / 7.787
        }
    };
    let x = f_inv(fx) * 0.95047;
    let y = f_inv(fy);
    let z = f_inv(fz) * 1.08883;
    // CIE XYZ → linear RGB (D65)
    let r = x * 3.2404542 - y * 1.5371385 - z * 0.4985314;
    let g = -x * 0.9692660 + y * 1.8760108 + z * 0.0415560;
    let b = x * 0.0556434 - y * 0.2040259 + z * 1.0572252;
    // sRGB gamma encoding
    let gamma = |c: f32| -> f32 {
        if c > 0.0031308 {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        } else {
            12.92 * c
        }
    };
    (gamma(r), gamma(g), gamma(b))
}

// ── HLS ───────────────────────────────────────────────────────────────────────

fn bgr_to_hls(data: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; h * w * 3];
    for i in 0..h * w {
        let off = i * 3;
        let b = data[off] as f32 / 255.0;
        let g = data[off + 1] as f32 / 255.0;
        let r = data[off + 2] as f32 / 255.0;
        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let l = (max + min) / 2.0;
        let delta = max - min;
        let s = if delta < 1e-6 {
            0.0
        } else if l < 0.5 {
            delta / (max + min)
        } else {
            delta / (2.0 - max - min)
        };
        let h_val = if delta < 1e-6 {
            0.0
        } else if (max - r).abs() < 1e-6 {
            ((g - b) / delta).rem_euclid(6.0) / 6.0
        } else if (max - g).abs() < 1e-6 {
            ((b - r) / delta + 2.0) / 6.0
        } else {
            ((r - g) / delta + 4.0) / 6.0
        };
        out[off] = (h_val * 180.0) as u8;
        out[off + 1] = (l * 255.0) as u8;
        out[off + 2] = (s * 255.0) as u8;
    }
    out
}

fn hls_to_bgr(data: &[u8], w: usize, h: usize, is_rgb: bool) -> Vec<u8> {
    let mut out = vec![0u8; h * w * 3];
    for i in 0..h * w {
        let off = i * 3;
        let h_val = data[off] as f32 / 180.0;
        let l = data[off + 1] as f32 / 255.0;
        let s = data[off + 2] as f32 / 255.0;
        let (r, g, b) = hls_to_rgb_f(h_val, l, s);
        if is_rgb {
            out[off] = (r * 255.0).clamp(0.0, 255.0) as u8;
            out[off + 1] = (g * 255.0).clamp(0.0, 255.0) as u8;
            out[off + 2] = (b * 255.0).clamp(0.0, 255.0) as u8;
        } else {
            out[off] = (b * 255.0).clamp(0.0, 255.0) as u8;
            out[off + 1] = (g * 255.0).clamp(0.0, 255.0) as u8;
            out[off + 2] = (r * 255.0).clamp(0.0, 255.0) as u8;
        }
    }
    out
}

fn hls_to_rgb_f(h: f32, l: f32, s: f32) -> (f32, f32, f32) {
    if s < 1e-6 {
        return (l, l, l);
    }
    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;
    let hue_to_rgb = |p: f32, q: f32, mut t: f32| -> f32 {
        if t < 0.0 {
            t += 1.0;
        }
        if t > 1.0 {
            t -= 1.0;
        }
        if t < 1.0 / 6.0 {
            return p + (q - p) * 6.0 * t;
        }
        if t < 1.0 / 2.0 {
            return q;
        }
        if t < 2.0 / 3.0 {
            return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
        }
        p
    };
    (
        hue_to_rgb(p, q, h + 1.0 / 3.0),
        hue_to_rgb(p, q, h),
        hue_to_rgb(p, q, h - 1.0 / 3.0),
    )
}

// ── YUV (BT.601) ──────────────────────────────────────────────────────────────

pub(crate) fn bgr_to_yuv(data: &[u8], w: usize, h: usize, is_rgb: bool) -> Vec<u8> {
    let mut out = vec![0u8; h * w * 3];
    for i in 0..h * w {
        let off = i * 3;
        let (r, g, b) = if is_rgb {
            (data[off] as f32, data[off + 1] as f32, data[off + 2] as f32)
        } else {
            (data[off + 2] as f32, data[off + 1] as f32, data[off] as f32)
        };
        let y = 0.299 * r + 0.587 * g + 0.114 * b;
        let u = -0.147 * r - 0.289 * g + 0.436 * b + 128.0;
        let v = 0.615 * r - 0.515 * g - 0.100 * b + 128.0;
        out[off] = y.clamp(0.0, 255.0) as u8;
        out[off + 1] = u.clamp(0.0, 255.0) as u8;
        out[off + 2] = v.clamp(0.0, 255.0) as u8;
    }
    out
}

pub(crate) fn yuv_to_bgr(data: &[u8], w: usize, h: usize, is_rgb: bool) -> Vec<u8> {
    let mut out = vec![0u8; h * w * 3];
    for i in 0..h * w {
        let off = i * 3;
        let y = data[off] as f32;
        let u = data[off + 1] as f32 - 128.0;
        let v = data[off + 2] as f32 - 128.0;
        let r = y + 1.140 * v;
        let g = y - 0.395 * u - 0.581 * v;
        let b = y + 2.032 * u;
        if is_rgb {
            out[off] = r.clamp(0.0, 255.0) as u8;
            out[off + 1] = g.clamp(0.0, 255.0) as u8;
            out[off + 2] = b.clamp(0.0, 255.0) as u8;
        } else {
            out[off] = b.clamp(0.0, 255.0) as u8;
            out[off + 1] = g.clamp(0.0, 255.0) as u8;
            out[off + 2] = r.clamp(0.0, 255.0) as u8;
        }
    }
    out
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bgr_gray_white() {
        let mat = Mat::from_bgr_bytes(vec![255u8, 255, 255], 1, 1);
        let gray = cvt_color(&mat, COLOR_BGR2GRAY).expect("should succeed");
        assert_eq!(gray.mat_type, MatType::CV_8UC1);
        assert_eq!(gray.at_8u1(0, 0), 255);
    }

    #[test]
    fn test_bgr_gray_black() {
        let mat = Mat::from_bgr_bytes(vec![0u8, 0, 0], 1, 1);
        let gray = cvt_color(&mat, COLOR_BGR2GRAY).expect("should succeed");
        assert_eq!(gray.at_8u1(0, 0), 0);
    }

    #[test]
    fn test_bgr_gray_pure_blue() {
        // Pure blue in BGR: [255, 0, 0] → BT.601: 0.114*255 ≈ 29
        let mat = Mat::from_bgr_bytes(vec![255u8, 0, 0], 1, 1);
        let gray = cvt_color(&mat, COLOR_BGR2GRAY).expect("should succeed");
        let expected = (0.114f32 * 255.0) as u8;
        assert!((gray.at_8u1(0, 0) as i32 - expected as i32).abs() <= 2);
    }

    #[test]
    fn test_swap_rb_roundtrip() {
        let data = vec![10u8, 20, 30, 40, 50, 60];
        let mat = Mat::from_bgr_bytes(data, 1, 2);
        let rgb = cvt_color(&mat, COLOR_BGR2RGB).expect("should succeed");
        let back = cvt_color(&rgb, COLOR_BGR2RGB).expect("should succeed");
        assert_eq!(back.data, mat.data);
    }

    #[test]
    fn test_gray_to_bgr() {
        let mat = Mat::from_gray_bytes(vec![128u8], 1, 1);
        let bgr = cvt_color(&mat, COLOR_GRAY2BGR).expect("should succeed");
        assert_eq!(bgr.mat_type, MatType::CV_8UC3);
        let px = bgr.at_8u3(0, 0);
        assert_eq!(px, [128, 128, 128]);
    }

    #[test]
    fn test_hsv_roundtrip_pure_green() {
        let mat = Mat::from_bgr_bytes(vec![0u8, 255, 0], 1, 1);
        let hsv = cvt_color(&mat, COLOR_BGR2HSV).expect("hsv");
        let back = cvt_color(&hsv, COLOR_HSV2BGR).expect("back");
        let px = back.at_8u3(0, 0);
        assert!(px[0] <= 2, "B≈0, got {}", px[0]);
        assert!((px[1] as i32 - 255).abs() <= 2, "G≈255, got {}", px[1]);
        assert!(px[2] <= 2, "R≈0, got {}", px[2]);
    }

    #[test]
    fn test_yuv_roundtrip_midgray() {
        let mat = Mat::from_bgr_bytes(vec![128u8, 128, 128], 1, 1);
        let yuv = cvt_color(&mat, COLOR_BGR2YUV).expect("yuv");
        let back = cvt_color(&yuv, COLOR_YUV2BGR).expect("back");
        let px = back.at_8u3(0, 0);
        for ch in 0..3 {
            assert!((px[ch] as i32 - 128).abs() <= 3, "ch{ch}: {}", px[ch]);
        }
    }

    #[test]
    fn test_unsupported_code_returns_error() {
        let mat = Mat::from_bgr_bytes(vec![0u8, 0, 0], 1, 1);
        assert!(cvt_color(&mat, 999).is_err());
    }
}
