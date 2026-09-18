//! Smoke tests verifying newly wired stabilize orphan modules compile and expose
//! at least one public item from each module.

use oximedia_stabilize::{
    lens_distortion::{CameraIntrinsics, DistortionCoeffs, DistortionModel},
    motion_mask::{MotionMaskBuilder, ThresholdMask},
    motion_vector_export::{MotionVectorExporter, MotionVectorFormat},
    perspective_warp::{PerspectiveWarpConfig, PerspectiveWarpCorrector},
    realtime_stabilize::{RealtimeConfig, RealtimeStabilizer},
    roi_stabilization::{RoiStabilizeConfig, RoiStabilizer},
    stabilize_preview::{stabilize_preview, PreviewConfig},
    stitching_stabilize::{StitchingStabilizeConfig, StitchingStabilizer},
    tripod_mode::{TripodConfig, TripodDetector},
    Frame,
};
use scirs2_core::ndarray::Array2;

fn blank_frame(w: usize, h: usize) -> Frame {
    Frame::new(w, h, 0.0, Array2::zeros((h, w)))
}

#[test]
fn test_camera_intrinsics_new_and_project() {
    let ci = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0);
    let (px, _py) = ci.project(1.0, 0.0, 1.0);
    // x = fx * (X/Z) + cx = 500 * 1.0 + 320 = 820
    assert!((px - 820.0).abs() < 1.0);
}

#[test]
fn test_distortion_model_default_variant() {
    let model = DistortionModel::default();
    assert_eq!(model, DistortionModel::BrownConrady);
}

#[test]
fn test_distortion_coeffs_default_zero() {
    let coeffs = DistortionCoeffs::default();
    assert!((coeffs.k1 - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_motion_mask_threshold_builder() {
    // ThresholdMask::build takes two frames and yields a MotionMask.
    let f1 = blank_frame(32, 32);
    let f2 = blank_frame(32, 32);
    let builder = ThresholdMask::new(15, 1);
    let mask = builder.build(&f1, &f2);
    assert_eq!(mask.width, 32);
    assert_eq!(mask.height, 32);
}

#[test]
fn test_motion_mask_builder_process() {
    // MotionMaskBuilder::process takes a single frame and maintains a background model.
    let frame = blank_frame(32, 32);
    let mut builder = MotionMaskBuilder::new();
    let mask = builder.process(&frame);
    assert_eq!(mask.width, 32);
}

#[test]
fn test_motion_vector_exporter_default_format() {
    let _exporter = MotionVectorExporter;
    // Exporter should support at least one format.
    let fmt = MotionVectorFormat::Csv;
    assert_eq!(fmt, MotionVectorFormat::Csv);
}

#[test]
fn test_perspective_warp_corrector_new() {
    let config = PerspectiveWarpConfig::default();
    let _corrector = PerspectiveWarpCorrector::new(config);
}

#[test]
fn test_realtime_stabilizer_buffered_count() {
    let config = RealtimeConfig::new();
    let stabilizer = RealtimeStabilizer::new(config);
    assert_eq!(stabilizer.buffered_count(), 0);
}

#[test]
fn test_roi_stabilizer_new() {
    let config = RoiStabilizeConfig::default();
    let _stabilizer = RoiStabilizer::new(config);
}

#[test]
fn test_stabilize_preview_empty_sequence() {
    // stabilize_preview takes (&[Frame], &PreviewConfig)
    let config = PreviewConfig::default();
    let result = stabilize_preview(&[], &config);
    // Empty input is an error, not a panic.
    assert!(result.is_err());
}

#[test]
fn test_stabilize_preview_single_frame() {
    let config = PreviewConfig::default();
    let frames = vec![blank_frame(16, 16)];
    // Single frame should succeed (identity transform).
    let result = stabilize_preview(&frames, &config);
    assert!(result.is_ok());
}

#[test]
fn test_stitching_stabilizer_new() {
    let config = StitchingStabilizeConfig::default();
    let _stabilizer = StitchingStabilizer::new(config);
}

#[test]
fn test_tripod_detector_new() {
    let config = TripodConfig::default();
    let _detector = TripodDetector::new(config);
}
