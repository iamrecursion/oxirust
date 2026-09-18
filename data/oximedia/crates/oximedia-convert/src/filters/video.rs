// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Video filters.

use serde::{Deserialize, Serialize};

/// Video filter type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VideoFilter {
    /// Deinterlace video
    Deinterlace(DeinterlaceMode),
    /// Denoise video
    Denoise(DenoiseParams),
    /// Sharpen video
    Sharpen(f64),
    /// Adjust brightness
    Brightness(f64),
    /// Adjust contrast
    Contrast(f64),
    /// Adjust saturation
    Saturation(f64),
    /// Rotate video
    Rotate(RotateAngle),
    /// Flip video horizontally
    FlipHorizontal,
    /// Flip video vertically
    FlipVertical,
    /// Crop video
    Crop(CropParams),
    /// Scale video
    Scale(ScaleParams),
    /// Color correction
    ColorCorrect(ColorCorrection),
    /// Overlay watermark
    Watermark(WatermarkParams),
}

/// Deinterlace mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeinterlaceMode {
    /// Discard one field (fastest)
    Discard,
    /// Blend fields
    Blend,
    /// Bob deinterlacing (double framerate)
    Bob,
    /// Yadif (yet another deinterlacing filter)
    Yadif,
}

/// Denoise parameters.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DenoiseParams {
    /// Luma strength (0.0-1.0)
    pub luma: f64,
    /// Chroma strength (0.0-1.0)
    pub chroma: f64,
}

impl Default for DenoiseParams {
    fn default() -> Self {
        Self {
            luma: 0.5,
            chroma: 0.5,
        }
    }
}

/// Rotation angle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RotateAngle {
    /// 90 degrees clockwise
    Rotate90,
    /// 180 degrees
    Rotate180,
    /// 270 degrees clockwise (90 counter-clockwise)
    Rotate270,
}

/// Crop parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CropParams {
    /// X coordinate of top-left corner
    pub x: u32,
    /// Y coordinate of top-left corner
    pub y: u32,
    /// Width of cropped area
    pub width: u32,
    /// Height of cropped area
    pub height: u32,
}

/// Scale parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScaleParams {
    /// Target width
    pub width: u32,
    /// Target height
    pub height: u32,
    /// Scaling algorithm
    pub algorithm: ScaleAlgorithm,
}

/// Scaling algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScaleAlgorithm {
    /// Nearest neighbor (fastest)
    Nearest,
    /// Bilinear interpolation
    Bilinear,
    /// Bicubic interpolation
    Bicubic,
    /// Lanczos resampling (best quality)
    Lanczos,
}

/// Color correction parameters.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ColorCorrection {
    /// Brightness adjustment (-1.0 to 1.0)
    pub brightness: f64,
    /// Contrast adjustment (0.0 to 2.0)
    pub contrast: f64,
    /// Saturation adjustment (0.0 to 2.0)
    pub saturation: f64,
    /// Gamma correction (0.1 to 10.0)
    pub gamma: f64,
}

impl Default for ColorCorrection {
    fn default() -> Self {
        Self {
            brightness: 0.0,
            contrast: 1.0,
            saturation: 1.0,
            gamma: 1.0,
        }
    }
}

/// Watermark parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WatermarkParams {
    /// Path to watermark image
    pub image_path: String,
    /// X position (0.0-1.0, relative to video width)
    pub x: f64,
    /// Y position (0.0-1.0, relative to video height)
    pub y: f64,
    /// Opacity (0.0-1.0)
    pub opacity: f64,
    /// Scale factor
    pub scale: f64,
}

impl Default for WatermarkParams {
    fn default() -> Self {
        Self {
            image_path: String::new(),
            x: 0.0,
            y: 0.0,
            opacity: 1.0,
            scale: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_denoise_params_default() {
        let params = DenoiseParams::default();
        assert_eq!(params.luma, 0.5);
        assert_eq!(params.chroma, 0.5);
    }

    #[test]
    fn test_color_correction_default() {
        let cc = ColorCorrection::default();
        assert_eq!(cc.brightness, 0.0);
        assert_eq!(cc.contrast, 1.0);
        assert_eq!(cc.saturation, 1.0);
        assert_eq!(cc.gamma, 1.0);
    }

    #[test]
    fn test_crop_params() {
        let crop = CropParams {
            x: 100,
            y: 100,
            width: 1280,
            height: 720,
        };
        assert_eq!(crop.x, 100);
        assert_eq!(crop.width, 1280);
    }

    #[test]
    fn test_scale_params() {
        let scale = ScaleParams {
            width: 1920,
            height: 1080,
            algorithm: ScaleAlgorithm::Lanczos,
        };
        assert_eq!(scale.algorithm, ScaleAlgorithm::Lanczos);
    }
}
