#![allow(dead_code)]
//! Field-based interlaced video scaling
//!
//! Provides scaling operations that respect interlaced field structure.
//! Separates top and bottom fields, scales each independently, and
//! re-interleaves them to prevent combing artifacts.

use std::fmt;

/// Field dominance (which field comes first temporally).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldOrder {
    /// Top field first (upper field first).
    TopFieldFirst,
    /// Bottom field first (lower field first).
    BottomFieldFirst,
    /// Progressive (no fields).
    Progressive,
}

impl fmt::Display for FieldOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::TopFieldFirst => "TFF",
            Self::BottomFieldFirst => "BFF",
            Self::Progressive => "PROG",
        };
        write!(f, "{s}")
    }
}

/// Interpolation method for field scaling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldInterpolation {
    /// Nearest-neighbor (fastest).
    Nearest,
    /// Bilinear interpolation.
    Bilinear,
    /// Bicubic interpolation.
    Bicubic,
}

/// Configuration for field-based scaling.
#[derive(Debug, Clone)]
pub struct FieldScaleConfig {
    /// Source width.
    pub src_width: u32,
    /// Source height (full frame, must be even for interlaced).
    pub src_height: u32,
    /// Destination width.
    pub dst_width: u32,
    /// Destination height (full frame, must be even for interlaced).
    pub dst_height: u32,
    /// Field order of the source.
    pub field_order: FieldOrder,
    /// Interpolation method.
    pub interpolation: FieldInterpolation,
    /// Whether to apply anti-alias filtering between fields.
    pub anti_alias: bool,
}

impl FieldScaleConfig {
    /// Create a new field scale config.
    pub fn new(
        src_width: u32,
        src_height: u32,
        dst_width: u32,
        dst_height: u32,
        field_order: FieldOrder,
    ) -> Self {
        Self {
            src_width,
            src_height,
            dst_width,
            dst_height,
            field_order,
            interpolation: FieldInterpolation::Bilinear,
            anti_alias: true,
        }
    }

    /// Set interpolation method.
    pub fn with_interpolation(mut self, interp: FieldInterpolation) -> Self {
        self.interpolation = interp;
        self
    }

    /// Set anti-alias filtering.
    pub fn with_anti_alias(mut self, enabled: bool) -> Self {
        self.anti_alias = enabled;
        self
    }

    /// Whether the source is interlaced.
    pub fn is_interlaced(&self) -> bool {
        self.field_order != FieldOrder::Progressive
    }

    /// Compute the field height (half the frame height).
    pub fn field_height(&self, frame_height: u32) -> u32 {
        frame_height / 2
    }
}

/// Separate a frame buffer into top and bottom fields (row-based, 1 byte per pixel).
///
/// Assumes row-major layout with `stride` bytes per row.
#[allow(clippy::cast_precision_loss)]
pub fn separate_fields(frame: &[u8], width: u32, height: u32, stride: u32) -> (Vec<u8>, Vec<u8>) {
    let field_h = height / 2;
    let row_bytes = width as usize;
    let mut top = Vec::with_capacity(field_h as usize * row_bytes);
    let mut bottom = Vec::with_capacity(field_h as usize * row_bytes);

    for y in 0..height {
        let start = (y * stride) as usize;
        let end = start + row_bytes;
        if end > frame.len() {
            break;
        }
        let row = &frame[start..end];
        if y % 2 == 0 {
            top.extend_from_slice(row);
        } else {
            bottom.extend_from_slice(row);
        }
    }
    (top, bottom)
}

/// Interleave top and bottom fields back into a full frame.
pub fn interleave_fields(top: &[u8], bottom: &[u8], width: u32, height: u32) -> Vec<u8> {
    let row_bytes = width as usize;
    let field_h = (height / 2) as usize;
    let mut frame = vec![0u8; (width * height) as usize];
    for fy in 0..field_h {
        let src_off = fy * row_bytes;
        // top field → even rows
        let dst_top = fy * 2 * row_bytes;
        if src_off + row_bytes <= top.len() && dst_top + row_bytes <= frame.len() {
            frame[dst_top..dst_top + row_bytes].copy_from_slice(&top[src_off..src_off + row_bytes]);
        }
        // bottom field → odd rows
        let dst_bot = (fy * 2 + 1) * row_bytes;
        if src_off + row_bytes <= bottom.len() && dst_bot + row_bytes <= frame.len() {
            frame[dst_bot..dst_bot + row_bytes]
                .copy_from_slice(&bottom[src_off..src_off + row_bytes]);
        }
    }
    frame
}

/// Scale a single field using bilinear interpolation.
///
/// `src` is the field data (field_height rows, each `src_width` bytes).
#[allow(clippy::cast_precision_loss)]
pub fn scale_field_bilinear(
    src: &[u8],
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
) -> Vec<u8> {
    let mut dst = vec![0u8; (dst_width * dst_height) as usize];
    let x_ratio = if dst_width > 1 {
        (src_width as f64 - 1.0) / (dst_width as f64 - 1.0)
    } else {
        0.0
    };
    let y_ratio = if dst_height > 1 {
        (src_height as f64 - 1.0) / (dst_height as f64 - 1.0)
    } else {
        0.0
    };

    for dy in 0..dst_height {
        for dx in 0..dst_width {
            let sx = x_ratio * dx as f64;
            let sy = y_ratio * dy as f64;
            let x0 = sx.floor() as u32;
            let y0 = sy.floor() as u32;
            let x1 = (x0 + 1).min(src_width - 1);
            let y1 = (y0 + 1).min(src_height - 1);
            let xf = sx - sx.floor();
            let yf = sy - sy.floor();

            let idx = |x: u32, y: u32| -> u8 {
                let i = (y * src_width + x) as usize;
                if i < src.len() {
                    src[i]
                } else {
                    0
                }
            };

            let top_l = idx(x0, y0) as f64;
            let top_r = idx(x1, y0) as f64;
            let bot_l = idx(x0, y1) as f64;
            let bot_r = idx(x1, y1) as f64;

            let top = top_l * (1.0 - xf) + top_r * xf;
            let bot = bot_l * (1.0 - xf) + bot_r * xf;
            let val = top * (1.0 - yf) + bot * yf;

            dst[(dy * dst_width + dx) as usize] = val.round().min(255.0).max(0.0) as u8;
        }
    }
    dst
}

/// Perform a complete field-based scale operation on a full interlaced frame.
#[allow(clippy::cast_precision_loss)]
pub fn field_scale(frame: &[u8], config: &FieldScaleConfig) -> Vec<u8> {
    if !config.is_interlaced() {
        // Progressive: just scale normally
        return scale_field_bilinear(
            frame,
            config.src_width,
            config.src_height,
            config.dst_width,
            config.dst_height,
        );
    }

    let (top, bottom) =
        separate_fields(frame, config.src_width, config.src_height, config.src_width);
    let src_fh = config.field_height(config.src_height);
    let dst_fh = config.field_height(config.dst_height);

    let scaled_top = scale_field_bilinear(&top, config.src_width, src_fh, config.dst_width, dst_fh);
    let scaled_bottom =
        scale_field_bilinear(&bottom, config.src_width, src_fh, config.dst_width, dst_fh);

    interleave_fields(
        &scaled_top,
        &scaled_bottom,
        config.dst_width,
        config.dst_height,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_order_display() {
        assert_eq!(FieldOrder::TopFieldFirst.to_string(), "TFF");
        assert_eq!(FieldOrder::BottomFieldFirst.to_string(), "BFF");
        assert_eq!(FieldOrder::Progressive.to_string(), "PROG");
    }

    #[test]
    fn test_config_is_interlaced() {
        let cfg = FieldScaleConfig::new(720, 480, 1920, 1080, FieldOrder::TopFieldFirst);
        assert!(cfg.is_interlaced());
        let cfg2 = FieldScaleConfig::new(720, 480, 1920, 1080, FieldOrder::Progressive);
        assert!(!cfg2.is_interlaced());
    }

    #[test]
    fn test_field_height() {
        let cfg = FieldScaleConfig::new(720, 480, 1920, 1080, FieldOrder::TopFieldFirst);
        assert_eq!(cfg.field_height(480), 240);
        assert_eq!(cfg.field_height(1080), 540);
    }

    #[test]
    fn test_config_builder() {
        let cfg = FieldScaleConfig::new(720, 480, 1920, 1080, FieldOrder::TopFieldFirst)
            .with_interpolation(FieldInterpolation::Bicubic)
            .with_anti_alias(false);
        assert_eq!(cfg.interpolation, FieldInterpolation::Bicubic);
        assert!(!cfg.anti_alias);
    }

    #[test]
    fn test_separate_fields() {
        // 4x4 frame, values 0..15
        let frame: Vec<u8> = (0..16).collect();
        let (top, bottom) = separate_fields(&frame, 4, 4, 4);
        // Row 0 (even): [0,1,2,3], Row 2 (even): [8,9,10,11]
        assert_eq!(top, vec![0, 1, 2, 3, 8, 9, 10, 11]);
        // Row 1 (odd): [4,5,6,7], Row 3 (odd): [12,13,14,15]
        assert_eq!(bottom, vec![4, 5, 6, 7, 12, 13, 14, 15]);
    }

    #[test]
    fn test_interleave_fields() {
        let top = vec![0, 1, 2, 3, 8, 9, 10, 11];
        let bottom = vec![4, 5, 6, 7, 12, 13, 14, 15];
        let frame = interleave_fields(&top, &bottom, 4, 4);
        let expected: Vec<u8> = (0..16).collect();
        assert_eq!(frame, expected);
    }

    #[test]
    fn test_roundtrip_separate_interleave() {
        let frame: Vec<u8> = (0..64).collect();
        let (top, bottom) = separate_fields(&frame, 8, 8, 8);
        let result = interleave_fields(&top, &bottom, 8, 8);
        assert_eq!(frame, result);
    }

    #[test]
    fn test_scale_field_identity() {
        // 2x2 field scaled to 2x2 should remain the same
        let field = vec![10, 20, 30, 40];
        let scaled = scale_field_bilinear(&field, 2, 2, 2, 2);
        assert_eq!(scaled, field);
    }

    #[test]
    fn test_scale_field_upscale() {
        // 2x2 field scaled to 4x4
        let field = vec![0, 100, 0, 100];
        let scaled = scale_field_bilinear(&field, 2, 2, 4, 4);
        assert_eq!(scaled.len(), 16);
        // Corner values should match source
        assert_eq!(scaled[0], 0);
        assert_eq!(scaled[3], 100);
    }

    #[test]
    fn test_scale_field_downscale() {
        // 4x4 field scaled to 2x2
        let field = vec![
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ];
        let scaled = scale_field_bilinear(&field, 4, 4, 2, 2);
        assert_eq!(scaled.len(), 4);
        // Corners: [0,0] = 10, [1,0] = 40, [0,1]=130, [1,1]=160
        assert_eq!(scaled[0], 10);
        assert_eq!(scaled[1], 40);
    }

    #[test]
    fn test_field_scale_progressive() {
        let frame = vec![10, 20, 30, 40];
        let cfg = FieldScaleConfig::new(2, 2, 2, 2, FieldOrder::Progressive);
        let result = field_scale(&frame, &cfg);
        assert_eq!(result, frame);
    }

    #[test]
    fn test_field_scale_interlaced_roundtrip_size() {
        // 4x4 source → 4x4 dest (identity)
        let frame: Vec<u8> = (0..16).collect();
        let cfg = FieldScaleConfig::new(4, 4, 4, 4, FieldOrder::TopFieldFirst);
        let result = field_scale(&frame, &cfg);
        assert_eq!(result.len(), 16);
    }

    #[test]
    fn test_field_scale_upscale_interlaced() {
        // 4x4 → 8x8 interlaced
        let frame: Vec<u8> = vec![128; 16];
        let cfg = FieldScaleConfig::new(4, 4, 8, 8, FieldOrder::BottomFieldFirst);
        let result = field_scale(&frame, &cfg);
        assert_eq!(result.len(), 64);
        // Uniform input → uniform output
        for &v in &result {
            assert_eq!(v, 128);
        }
    }
}
