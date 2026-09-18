//! Multi-stage scaling pipeline orchestration.
//!
//! Chains several scale operations (pre-crop, horizontal scale, vertical
//! scale, post-sharpen, pad) into a single [`ScalePipeline`] that can be
//! validated and applied to a buffer.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use serde::{Deserialize, Serialize};

// -- PipelineStage -----------------------------------------------------------

/// Individual stages of a scaling pipeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PipelineStage {
    /// Crop the source before scaling.
    Crop {
        /// Left pixel offset.
        left: u32,
        /// Top pixel offset.
        top: u32,
        /// Cropped region width.
        width: u32,
        /// Cropped region height.
        height: u32,
    },
    /// Horizontal scale to `target_width`.
    HorizontalScale {
        /// Target width in pixels.
        target_width: u32,
    },
    /// Vertical scale to `target_height`.
    VerticalScale {
        /// Target height in pixels.
        target_height: u32,
    },
    /// Post-scale sharpening.
    Sharpen {
        /// Sharpening strength (0.0 .. 2.0).
        strength: f32,
    },
    /// Pad the output to a final frame size with a fill value.
    Pad {
        /// Final frame width.
        frame_width: u32,
        /// Final frame height.
        frame_height: u32,
        /// Fill value (e.g. 0 for black, 128 for mid-grey in 8-bit).
        fill_value: u8,
    },
}

impl std::fmt::Display for PipelineStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Crop {
                left,
                top,
                width,
                height,
            } => {
                write!(f, "crop({left},{top},{width}x{height})")
            }
            Self::HorizontalScale { target_width } => write!(f, "h-scale({target_width})"),
            Self::VerticalScale { target_height } => write!(f, "v-scale({target_height})"),
            Self::Sharpen { strength } => write!(f, "sharpen({strength:.2})"),
            Self::Pad {
                frame_width,
                frame_height,
                fill_value,
            } => {
                write!(f, "pad({frame_width}x{frame_height},fill={fill_value})")
            }
        }
    }
}

// -- PipelineError -----------------------------------------------------------

/// Errors that can occur during pipeline validation or execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelineError {
    /// The pipeline has no stages.
    Empty,
    /// A dimension is zero.
    ZeroDimension(String),
    /// Crop exceeds source dimensions.
    CropExceedsSource(String),
    /// Sharpening strength out of range.
    InvalidSharpenStrength,
    /// Stage ordering is invalid.
    InvalidStageOrder(String),
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "pipeline has no stages"),
            Self::ZeroDimension(s) => write!(f, "zero dimension: {s}"),
            Self::CropExceedsSource(s) => write!(f, "crop exceeds source: {s}"),
            Self::InvalidSharpenStrength => write!(f, "sharpen strength out of range [0,2]"),
            Self::InvalidStageOrder(s) => write!(f, "invalid stage order: {s}"),
        }
    }
}

impl std::error::Error for PipelineError {}

// -- ScalePipeline -----------------------------------------------------------

/// A validated, ordered sequence of scaling stages.
///
/// # Example
/// ```
/// use oximedia_scaling::scale_pipeline::{ScalePipeline, PipelineStage};
///
/// let mut pipe = ScalePipeline::new(1920, 1080);
/// pipe.add_stage(PipelineStage::HorizontalScale { target_width: 1280 });
/// pipe.add_stage(PipelineStage::VerticalScale { target_height: 720 });
/// assert_eq!(pipe.stage_count(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct ScalePipeline {
    /// Source width.
    src_width: u32,
    /// Source height.
    src_height: u32,
    /// Ordered stages.
    stages: Vec<PipelineStage>,
}

impl ScalePipeline {
    /// Create a new pipeline for a source of `src_width x src_height`.
    pub fn new(src_width: u32, src_height: u32) -> Self {
        Self {
            src_width,
            src_height,
            stages: Vec::new(),
        }
    }

    /// Append a stage to the pipeline.
    pub fn add_stage(&mut self, stage: PipelineStage) {
        self.stages.push(stage);
    }

    /// Return the number of stages.
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Return a reference to the stages.
    pub fn stages(&self) -> &[PipelineStage] {
        &self.stages
    }

    /// Source dimensions.
    pub fn source_dims(&self) -> (u32, u32) {
        (self.src_width, self.src_height)
    }

    /// Compute the final output dimensions by walking through all stages.
    pub fn output_dims(&self) -> (u32, u32) {
        let mut w = self.src_width;
        let mut h = self.src_height;
        for stage in &self.stages {
            match stage {
                PipelineStage::Crop { width, height, .. } => {
                    w = *width;
                    h = *height;
                }
                PipelineStage::HorizontalScale { target_width } => {
                    w = *target_width;
                }
                PipelineStage::VerticalScale { target_height } => {
                    h = *target_height;
                }
                PipelineStage::Pad {
                    frame_width,
                    frame_height,
                    ..
                } => {
                    w = *frame_width;
                    h = *frame_height;
                }
                PipelineStage::Sharpen { .. } => {} // no dimension change
            }
        }
        (w, h)
    }

    /// Validate the pipeline, returning all errors found.
    pub fn validate(&self) -> Vec<PipelineError> {
        let mut errors = Vec::new();
        if self.stages.is_empty() {
            errors.push(PipelineError::Empty);
            return errors;
        }

        let mut cur_w = self.src_width;
        let mut cur_h = self.src_height;

        for stage in &self.stages {
            match stage {
                PipelineStage::Crop {
                    left,
                    top,
                    width,
                    height,
                } => {
                    if *width == 0 || *height == 0 {
                        errors.push(PipelineError::ZeroDimension(format!(
                            "crop {width}x{height}"
                        )));
                    }
                    if left + width > cur_w || top + height > cur_h {
                        errors.push(PipelineError::CropExceedsSource(format!(
                            "crop ({left},{top},{width}x{height}) exceeds {cur_w}x{cur_h}"
                        )));
                    }
                    cur_w = *width;
                    cur_h = *height;
                }
                PipelineStage::HorizontalScale { target_width } => {
                    if *target_width == 0 {
                        errors.push(PipelineError::ZeroDimension("h-scale width=0".into()));
                    }
                    cur_w = *target_width;
                }
                PipelineStage::VerticalScale { target_height } => {
                    if *target_height == 0 {
                        errors.push(PipelineError::ZeroDimension("v-scale height=0".into()));
                    }
                    cur_h = *target_height;
                }
                PipelineStage::Sharpen { strength } => {
                    if *strength < 0.0 || *strength > 2.0 {
                        errors.push(PipelineError::InvalidSharpenStrength);
                    }
                }
                PipelineStage::Pad {
                    frame_width,
                    frame_height,
                    ..
                } => {
                    if *frame_width == 0 || *frame_height == 0 {
                        errors.push(PipelineError::ZeroDimension(format!(
                            "pad {frame_width}x{frame_height}"
                        )));
                    }
                    cur_w = *frame_width;
                    cur_h = *frame_height;
                }
            }
        }

        errors
    }

    /// Return `true` if the pipeline passes validation.
    pub fn is_valid(&self) -> bool {
        self.validate().is_empty()
    }

    /// Build a simple two-pass (horizontal + vertical) pipeline.
    pub fn simple_scale(src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Self {
        let mut pipe = Self::new(src_w, src_h);
        pipe.add_stage(PipelineStage::HorizontalScale {
            target_width: dst_w,
        });
        pipe.add_stage(PipelineStage::VerticalScale {
            target_height: dst_h,
        });
        pipe
    }

    /// Return a human-readable summary of the pipeline.
    pub fn summary(&self) -> String {
        let stage_strs: Vec<String> = self.stages.iter().map(|s| s.to_string()).collect();
        format!(
            "{}x{} -> [{}] -> {}x{}",
            self.src_width,
            self.src_height,
            stage_strs.join(" | "),
            self.output_dims().0,
            self.output_dims().1,
        )
    }
}

// -- Tests -------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_scale_output_dims() {
        let pipe = ScalePipeline::simple_scale(1920, 1080, 1280, 720);
        assert_eq!(pipe.output_dims(), (1280, 720));
    }

    #[test]
    fn test_simple_scale_is_valid() {
        let pipe = ScalePipeline::simple_scale(1920, 1080, 1280, 720);
        assert!(pipe.is_valid());
    }

    #[test]
    fn test_empty_pipeline_invalid() {
        let pipe = ScalePipeline::new(1920, 1080);
        let errs = pipe.validate();
        assert!(errs.iter().any(|e| *e == PipelineError::Empty));
    }

    #[test]
    fn test_crop_then_scale() {
        let mut pipe = ScalePipeline::new(1920, 1080);
        pipe.add_stage(PipelineStage::Crop {
            left: 100,
            top: 50,
            width: 1720,
            height: 980,
        });
        pipe.add_stage(PipelineStage::HorizontalScale { target_width: 1280 });
        pipe.add_stage(PipelineStage::VerticalScale { target_height: 720 });
        assert!(pipe.is_valid());
        assert_eq!(pipe.output_dims(), (1280, 720));
    }

    #[test]
    fn test_crop_exceeds_source() {
        let mut pipe = ScalePipeline::new(1920, 1080);
        pipe.add_stage(PipelineStage::Crop {
            left: 0,
            top: 0,
            width: 2000,
            height: 1080,
        });
        let errs = pipe.validate();
        assert!(errs
            .iter()
            .any(|e| matches!(e, PipelineError::CropExceedsSource(_))));
    }

    #[test]
    fn test_zero_target_width() {
        let mut pipe = ScalePipeline::new(1920, 1080);
        pipe.add_stage(PipelineStage::HorizontalScale { target_width: 0 });
        let errs = pipe.validate();
        assert!(errs
            .iter()
            .any(|e| matches!(e, PipelineError::ZeroDimension(_))));
    }

    #[test]
    fn test_zero_crop_dims() {
        let mut pipe = ScalePipeline::new(1920, 1080);
        pipe.add_stage(PipelineStage::Crop {
            left: 0,
            top: 0,
            width: 0,
            height: 0,
        });
        let errs = pipe.validate();
        assert!(errs
            .iter()
            .any(|e| matches!(e, PipelineError::ZeroDimension(_))));
    }

    #[test]
    fn test_invalid_sharpen_strength() {
        let mut pipe = ScalePipeline::new(1920, 1080);
        pipe.add_stage(PipelineStage::HorizontalScale { target_width: 1280 });
        pipe.add_stage(PipelineStage::Sharpen { strength: 3.0 });
        let errs = pipe.validate();
        assert!(errs
            .iter()
            .any(|e| *e == PipelineError::InvalidSharpenStrength));
    }

    #[test]
    fn test_valid_sharpen_strength() {
        let mut pipe = ScalePipeline::new(1920, 1080);
        pipe.add_stage(PipelineStage::HorizontalScale { target_width: 1280 });
        pipe.add_stage(PipelineStage::Sharpen { strength: 0.5 });
        assert!(pipe.is_valid());
    }

    #[test]
    fn test_pad_stage() {
        let mut pipe = ScalePipeline::new(640, 480);
        pipe.add_stage(PipelineStage::HorizontalScale { target_width: 1440 });
        pipe.add_stage(PipelineStage::VerticalScale {
            target_height: 1080,
        });
        pipe.add_stage(PipelineStage::Pad {
            frame_width: 1920,
            frame_height: 1080,
            fill_value: 0,
        });
        assert!(pipe.is_valid());
        assert_eq!(pipe.output_dims(), (1920, 1080));
    }

    #[test]
    fn test_stage_count() {
        let pipe = ScalePipeline::simple_scale(1920, 1080, 1280, 720);
        assert_eq!(pipe.stage_count(), 2);
    }

    #[test]
    fn test_source_dims() {
        let pipe = ScalePipeline::new(3840, 2160);
        assert_eq!(pipe.source_dims(), (3840, 2160));
    }

    #[test]
    fn test_summary_string() {
        let pipe = ScalePipeline::simple_scale(1920, 1080, 1280, 720);
        let s = pipe.summary();
        assert!(s.contains("1920x1080"));
        assert!(s.contains("1280"));
        assert!(s.contains("720"));
    }

    #[test]
    fn test_pipeline_stage_display() {
        let s = PipelineStage::HorizontalScale { target_width: 1280 };
        assert_eq!(s.to_string(), "h-scale(1280)");
    }

    #[test]
    fn test_pipeline_error_display() {
        let e = PipelineError::Empty;
        assert_eq!(e.to_string(), "pipeline has no stages");
    }

    #[test]
    fn test_stages_accessor() {
        let pipe = ScalePipeline::simple_scale(1920, 1080, 1280, 720);
        assert_eq!(pipe.stages().len(), 2);
    }

    #[test]
    fn test_multiple_errors_accumulate() {
        let mut pipe = ScalePipeline::new(1920, 1080);
        pipe.add_stage(PipelineStage::HorizontalScale { target_width: 0 });
        pipe.add_stage(PipelineStage::VerticalScale { target_height: 0 });
        pipe.add_stage(PipelineStage::Sharpen { strength: -1.0 });
        let errs = pipe.validate();
        assert!(errs.len() >= 3);
    }
}
